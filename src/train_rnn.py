"""GloVe word vectors + BiLSTM / TextCNN under the shared 5-fold protocol.

    python -m src.train_rnn --arch bilstm --name glove_bilstm
    python -m src.train_rnn --arch textcnn --name glove_textcnn
    python -m src.train_rnn --arch bilstm --no-pretrained --name rand_bilstm      # ablation: random init embeddings
    python -m src.train_rnn --arch bilstm --folds 0 --limit 256 --epochs 1 --no-save   # smoke test

Per fold: build the vocabulary from the training folds only (words seen in GloVe need count >= 1,
other words count >= --min-count and get a trainable random vector), encode with head+tail word
truncation to --max-len, train with Adam + class-weighted CE, evaluate the held-out fold after
every epoch (macro-F1 @0.5), early-stop with --patience, keep the best weights; the best
checkpoint's held-out probabilities are the fold's OOF probabilities and its test probabilities
are averaged over folds. The held-out probabilities at the epoch where training stopped
(best epoch + --patience, or the --epochs cap) are saved as *_last; unlike train_plm.py's
fixed-schedule *_last this stop epoch itself depends on the held-out fold, so it is NOT a
selection-free estimate - it only shows how much the best-epoch pick adds on top.
Tokenisation follows the GloVe 6B / Stanford style for lower-casing, contractions and
punctuation ("don't" -> "do" "n't", "it's" -> "it" "'s"); hyphenated words, decimals and
abbreviations are split into pieces although GloVe stores some of them whole (~0.3% of tokens).
No pack_padded_sequence (12x slower on MPS):
BiLSTM uses masked mean+max pooling, TextCNN masks convolution outputs that touch padding.
Partial runs (--folds subset / --limit) need --no-save or --allow-partial (suffixed run_id).
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")

import argparse
import functools
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import data as _data
from src.data import FOLD_SEED, LABEL_COL, N_SPLITS, TEXT_COL, load_test, load_train_folds
from src.metrics import STD_DDOF, compute_metrics
from src.train_plm import class_weights, device_mem_mb, fold_indices, free_device, pick_device, seed_everything
from src.utils.resources import Timer, machine_tag, peak_rss_mb
from src.utils.results import compute_run_stats, log_run_stats, runs_row, write_run

RESULTS_DIR = Path(os.environ.get("EE6483_RESULTS_DIR", _data.RESULTS_DIR))
NON_CONFIG_KEYS = {"name", "no_save", "allow_partial", "folds", "device", "fold_list", "partial"}
PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1
MIN_BATCH_LEN = 8  # TextCNN needs L >= largest kernel; also keeps tensor shapes few
# GloVe 6B / Stanford-style: "don't" -> "do" "n't"; "it's" -> "it" "'s"; punctuation as its own token
TOKEN_RE = re.compile(r"[a-z0-9]+(?=n't)|n't|'(?:s|re|ve|ll|d|m)(?![a-z])|[a-z0-9]+|[^\sa-z0-9]")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arch", choices=["bilstm", "textcnn"], default="bilstm")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--glove", default="glove-wiki-gigaword-300", help="gensim-downloader name of the pretrained vectors")
    p.add_argument("--no-pretrained", action="store_true", help="ablation: random-init embeddings; GloVe is still loaded so the vocabulary rule is identical")
    p.add_argument("--freeze-emb-epochs", type=int, default=1, help="keep pretrained embeddings frozen for the first N epochs")
    p.add_argument("--min-count", type=int, default=2, help="min train-fold count for words NOT in GloVe (GloVe words need 1)")
    p.add_argument("--max-len", type=int, default=160, help="word-level length budget (head+tail truncation)")
    p.add_argument("--head-frac", type=float, default=0.75)
    p.add_argument("--hidden", type=int, default=256, help="BiLSTM hidden size per direction")
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--filters", type=int, default=128, help="TextCNN filters per kernel size")
    p.add_argument("--kernels", default="3,4,5")
    p.add_argument("--dropout", type=float, default=None, help="default 0.3 for bilstm, 0.5 for textcnn")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15, help="upper bound; early stopping on val macro-F1")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="sqrt")
    p.add_argument("--device", default="auto")
    p.add_argument("--folds", default="all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--acc-slack", type=float, default=0.003)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args(argv)
    if args.dropout is None:
        args.dropout = 0.3 if args.arch == "bilstm" else 0.5
    if args.name is None:
        emb = "rand" if args.no_pretrained else "glove"
        size = f"h{args.hidden}x{args.layers}" if args.arch == "bilstm" else f"f{args.filters}k{args.kernels.replace(',', '')}"
        args.name = f"{emb}_{args.arch}_{size}_len{args.max_len}_lr{args.lr:g}_cw{args.class_weight}"
    folds = list(range(N_SPLITS)) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    if len(set(folds)) != len(folds) or not set(folds) <= set(range(N_SPLITS)):
        p.error(f"--folds must be unique ids in 0..{N_SPLITS - 1}, got {folds}")
    args.fold_list = folds
    args.partial = len(folds) < N_SPLITS or bool(args.limit)
    if args.partial and not args.no_save and not args.allow_partial:
        p.error("a --folds subset or --limit run is partial: add --no-save (smoke test) or --allow-partial (saved under a suffixed run_id)")
    return args


def run_id_for(args) -> str:
    rid = f"{args.name}_seed{args.seed}"
    if len(args.fold_list) < N_SPLITS:
        rid += "_f" + "-".join(map(str, args.fold_list))
    if args.limit:
        rid += f"_n{args.limit}"
    return rid


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def load_vectors(name: str):
    import gensim.downloader as api
    return api.load(name)


def build_vocab(train_tokens: list[list[str]], kv, min_count: int, pretrained: bool, rng: np.random.Generator):
    """Vocabulary from the training folds only. Returns (itos, stoi, embedding matrix, stats)."""
    counts = Counter(t for toks in train_tokens for t in toks)
    in_glove = (lambda w: w in kv.key_to_index) if kv is not None else (lambda w: False)
    words = [w for w, c in counts.most_common() if c >= min_count or in_glove(w)]
    itos = [PAD, UNK] + words
    stoi = {w: i for i, w in enumerate(itos)}
    dim = kv.vector_size
    emb = rng.normal(0.0, 0.1, size=(len(itos), dim)).astype(np.float32)
    emb[PAD_ID] = 0.0
    n_glove = 0
    if kv is not None and pretrained:
        for w, i in stoi.items():
            if in_glove(w):
                emb[i] = kv[w]
                n_glove += 1
    n_train_tokens = sum(counts.values())
    covered = sum(c for w, c in counts.items() if w in stoi)
    stats = {"vocab_size": len(itos), "types_in_train": len(counts), "types_in_glove": int(sum(1 for w in itos[2:] if in_glove(w))),
             "glove_init_fraction": float(n_glove / max(len(itos) - 2, 1)), "train_token_coverage": float(covered / max(n_train_tokens, 1))}
    return itos, stoi, torch.tensor(emb), stats


def encode(token_lists: list[list[str]], stoi: dict, max_len: int, head_frac: float):
    out, n_unk, n_tok, n_trunc = [], 0, 0, 0
    for toks in token_lists:
        ids = [stoi.get(t, UNK_ID) for t in toks] or [UNK_ID]
        n_unk += sum(1 for i in ids if i == UNK_ID)
        n_tok += len(ids)
        if len(ids) > max_len:
            n_trunc += 1
            head = int(max_len * head_frac)
            ids = ids[:head] + ids[len(ids) - (max_len - head):]
        out.append(ids)
    return out, {"unk_token_rate": float(n_unk / max(n_tok, 1)), "frac_truncated": float(n_trunc / max(len(out), 1))}


def make_batches(indices, encoded, labels, batch_size, shuffle, rng):
    idx = np.array(indices)
    if shuffle:
        rng.shuffle(idx)
    else:
        idx = idx[np.argsort([len(encoded[i]) for i in idx], kind="stable")]
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        L = max(MIN_BATCH_LEN, max(len(encoded[i]) for i in b))
        L = int(np.ceil(L / MIN_BATCH_LEN) * MIN_BATCH_LEN)
        ids = torch.full((len(b), L), PAD_ID, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.bool)
        for r, i in enumerate(b):
            seq = encoded[i]
            ids[r, :len(seq)] = torch.tensor(seq)
            mask[r, :len(seq)] = True
        y = torch.tensor([labels[i] for i in b], dtype=torch.long) if labels is not None else None
        yield b, ids, mask, y


def reverse_within_length(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Reverse each sequence inside its own real length; padding stays at the end.
    x: (B, L, *), mask: (B, L) bool. Applying it twice restores the original order."""
    lengths = mask.sum(1)                                                    # (B,)
    pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)             # (1, L)
    idx = torch.where(pos < lengths.unsqueeze(1), lengths.unsqueeze(1) - 1 - pos, pos)  # (B, L)
    idx = idx.view(idx.shape[0], idx.shape[1], *([1] * (x.dim() - 2))).expand_as(x)
    return x.gather(1, idx)


class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM without pack_padded_sequence (12x slower on MPS).

    The backward direction runs a separate LSTM over each sequence reversed inside its own
    length, so padding is always *after* the real tokens for both directions and the outputs at
    real positions are exactly padding-independent. With layers > 1 each direction is stacked on
    its own outputs (unlike nn.LSTM(bidirectional=True), which concatenates directions between layers).
    Pooling: masked mean + masked max over the concatenated outputs."""

    def __init__(self, emb: torch.Tensor, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=PAD_ID)
        between = dropout if layers > 1 else 0.0
        self.fwd = nn.LSTM(emb.shape[1], hidden, num_layers=layers, batch_first=True, dropout=between)
        self.bwd = nn.LSTM(emb.shape[1], hidden, num_layers=layers, batch_first=True, dropout=between)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(4 * hidden, 2)  # masked mean + masked max over the 2*hidden outputs

    def forward(self, ids, mask):
        x = self.drop(self.emb(ids))                               # (B, L, E)
        out_f, _ = self.fwd(x)
        out_b, _ = self.bwd(reverse_within_length(x, mask))
        out = torch.cat([out_f, reverse_within_length(out_b, mask)], dim=-1)   # (B, L, 2H), aligned positions
        m = mask.unsqueeze(-1).to(out.dtype)
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        mx = out.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
        return self.fc(self.drop(torch.cat([mean, mx], dim=-1)))


class TextCNNClassifier(nn.Module):
    def __init__(self, emb: torch.Tensor, filters: int, kernels: list[int], dropout: float):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(emb, freeze=False, padding_idx=PAD_ID)
        self.convs = nn.ModuleList([nn.Conv1d(emb.shape[1], filters, k) for k in kernels])
        self.kernels = kernels
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernels), 2)

    def forward(self, ids, mask):
        x = self.drop(self.emb(ids)).transpose(1, 2)            # (B, E, L)
        pooled = []
        for conv, k in zip(self.convs, self.kernels):
            h = F.relu(conv(x))                                  # (B, F, L-k+1)
            valid = mask[:, k - 1:]                              # window t..t+k-1 is inside the text iff position t+k-1 is real
            h = h.masked_fill(~valid.unsqueeze(1), 0.0)              # post-ReLU, so 0 never beats a valid window; a text shorter than k gives 0 features
            pooled.append(h.max(2).values)
        return self.fc(self.drop(torch.cat(pooled, dim=-1)))


def build_model(args, emb):
    if args.arch == "bilstm":
        return BiLSTMClassifier(emb, args.hidden, args.layers, args.dropout)
    kernels = [int(k) for k in args.kernels.split(",")]
    assert max(kernels) <= MIN_BATCH_LEN, f"largest TextCNN kernel {max(kernels)} exceeds the minimum batch length {MIN_BATCH_LEN}"
    return TextCNNClassifier(emb, args.filters, kernels, args.dropout)


@torch.no_grad()
def predict(model, indices, encoded, batch_size, device):
    was_training = model.training
    model.eval()
    probs = np.zeros(len(indices), dtype=float)
    pos = {i: r for r, i in enumerate(indices)}
    for b, ids, mask, _ in make_batches(indices, encoded, None, batch_size, shuffle=False, rng=None):
        logits = model(ids.to(device), mask.to(device))
        p = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        for i, v in zip(b, p):
            probs[pos[i]] = float(v)
    if was_training:
        model.train()
    return probs


def train_fold(k, args, df, tokens_train, tokens_test, kv, device, log):
    seed_everything(args.seed + k)
    rng = np.random.default_rng(args.seed + k)
    y_all = df[LABEL_COL].to_numpy()
    tr_idx, va_idx, te_idx = fold_indices(k, args, df, len(tokens_test))

    with Timer() as t_vocab:
        itos, stoi, emb, vocab_stats = build_vocab([tokens_train[i] for i in tr_idx], kv, args.min_count, not args.no_pretrained, rng)
        enc_train, _ = encode(tokens_train, stoi, args.max_len, args.head_frac)
    with Timer() as t_enc_test:
        enc_test, _ = encode(tokens_test, stoi, args.max_len, args.head_frac)
    _, enc_stats_tr = encode([tokens_train[i] for i in tr_idx], stoi, args.max_len, args.head_frac)
    _, enc_stats_va = encode([tokens_train[i] for i in va_idx], stoi, args.max_len, args.head_frac)
    _, enc_stats_te = encode([tokens_test[i] for i in te_idx], stoi, args.max_len, args.head_frac)
    vocab_stats.update(unk_rate_val=enc_stats_va["unk_token_rate"], unk_rate_test=enc_stats_te["unk_token_rate"], frac_truncated_train=enc_stats_tr["frac_truncated"])

    model = build_model(args, emb).to(device)
    model.train()
    weight = class_weights(y_all[tr_idx], args.class_weight)
    weight = weight.to(device) if weight is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    emb_frozen = (not args.no_pretrained) and args.freeze_emb_epochs > 0
    model.emb.weight.requires_grad_(not emb_frozen)

    best = {"macro_f1": -1.0, "epoch": 0, "state": None, "val_probs": None}
    last_val_probs, curve, bad_epochs = None, [], 0
    train_time = eval_time = 0.0
    epochs_run, early_stopped = 0, False
    for epoch in range(1, args.epochs + 1):
        if emb_frozen and epoch > args.freeze_emb_epochs:
            model.emb.weight.requires_grad_(True)
            emb_frozen = False
        run_loss, n_loss = 0.0, 0
        t0 = time.perf_counter()
        for b, ids, mask, y in make_batches(tr_idx, enc_train, y_all, args.batch_size, shuffle=True, rng=rng):
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(ids, mask).float(), y, weight=weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss in fold {k} epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            run_loss += loss.item()
            n_loss += 1
        train_time += time.perf_counter() - t0
        t1 = time.perf_counter()
        val_probs = predict(model, va_idx, enc_train, args.batch_size * 4, device)
        m = compute_metrics(y_all[va_idx], val_probs, 0.5)
        eval_time += time.perf_counter() - t1
        epochs_run = epoch
        last_val_probs = val_probs
        point = {"fold": k, "epoch": epoch, "train_loss": run_loss / max(n_loss, 1), "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"],
                 "val_f1_neg": m["f1_neg"], "val_auc": m["roc_auc"], "emb_frozen": emb_frozen, "device_mem_mb": device_mem_mb(device)}
        curve.append(point)
        improved = m["macro_f1"] > best["macro_f1"]
        if improved:
            best = {"macro_f1": m["macro_f1"], "epoch": epoch, "val_probs": val_probs, "state": {n: t.detach().to("cpu", copy=True) for n, t in model.state_dict().items()}}
            bad_epochs = 0
        else:
            bad_epochs += 1
        log(f"  fold {k} ep {epoch:2d} loss {point['train_loss']:.4f} | val acc {m['accuracy']:.4f} mF1 {m['macro_f1']:.4f} F1neg {m['f1_neg']:.4f} AUC {m['roc_auc']:.4f}{' *' if improved else ''}{' (emb frozen)' if emb_frozen else ''}")
        unfreeze_pending = emb_frozen and args.freeze_emb_epochs < args.epochs  # do not stop before the planned unfreeze
        if bad_epochs >= args.patience and not unfreeze_pending:
            early_stopped = True
            log(f"  fold {k}: early stop after epoch {epoch} (best epoch {best['epoch']})")
            break

    model.load_state_dict(best["state"])
    t2 = time.perf_counter()
    test_probs = predict(model, te_idx, enc_test, args.batch_size * 4, device)
    infer_test_s = time.perf_counter() - t2
    n_params = sum(p.numel() for p in model.parameters())
    n_params_no_emb = n_params - model.emb.weight.numel()
    emb_trained = args.no_pretrained or epochs_run > args.freeze_emb_epochs  # embeddings received gradients at some point
    n_params_trainable = n_params if emb_trained else n_params_no_emb
    mem = device_mem_mb(device)
    del model, optimizer, best["state"]
    free_device(device)
    return {"fold": k, "va_idx": va_idx, "te_idx": te_idx, "oof_probs": best["val_probs"], "oof_probs_last": last_val_probs, "test_probs": test_probs,
            "n_train": int(len(tr_idx)), "best_epoch": best["epoch"], "epochs_run": epochs_run, "early_stopped": early_stopped, "curve": curve, "vocab": vocab_stats,
            "vocab_s": t_vocab.seconds, "encode_test_s": t_enc_test.seconds, "train_s": train_time, "eval_s": eval_time, "infer_test_s": infer_test_s,
            "n_params": n_params, "n_params_no_emb": n_params_no_emb, "n_params_trainable": n_params_trainable, "emb_trained": emb_trained, "device_mem_mb": mem}


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    log = functools.partial(print, flush=True)
    run_id = run_id_for(args)
    folds = args.fold_list
    t_start = time.perf_counter()
    log(f"arch {args.arch} | embeddings {'random' if args.no_pretrained else args.glove} | device {device} | run {run_id}{' [PARTIAL]' if args.partial else ''} | seed {args.seed}")

    df = load_train_folds()
    test = load_test()
    y = df[LABEL_COL].to_numpy()
    with Timer() as t_tok_train:
        tokens_train = [tokenize(t) for t in df[TEXT_COL]]
    with Timer() as t_tok_test:
        tokens_test = [tokenize(t) for t in test[TEXT_COL]]
    with Timer() as t_glove:
        kv = load_vectors(args.glove)  # also under --no-pretrained: membership defines the vocabulary rule, vectors are then ignored
    lens = np.array([len(t) for t in tokens_train])
    log(f"tokenised in {t_tok_train.seconds + t_tok_test.seconds:.1f}s (GloVe loaded in {t_glove.seconds:.1f}s) | train word len median {np.median(lens):.0f} p95 {np.percentile(lens, 95):.0f} | > max_len {args.max_len}: {(lens > args.max_len).mean():.1%}")

    oof, oof_last = np.full(len(df), np.nan), np.full(len(df), np.nan)
    test_prob, te_seen = np.zeros(len(test)), np.zeros(len(test), dtype=bool)
    fold_results, fold_rows, fold_rows_last = [], [], []
    for k in folds:
        log(f"== fold {k} ==")
        r = train_fold(k, args, df, tokens_train, tokens_test, kv, device, log)
        fold_results.append(r)
        oof[r["va_idx"]], oof_last[r["va_idx"]] = r["oof_probs"], r["oof_probs_last"]
        test_prob[r["te_idx"]] += r["test_probs"] / len(folds)
        te_seen[r["te_idx"]] = True
        m = compute_metrics(y[r["va_idx"]], r["oof_probs"], 0.5)
        m.update(fold=k, n_train=r["n_train"], n_val=int(len(r["va_idx"])), best_epoch=r["best_epoch"], epochs_run=r["epochs_run"], early_stopped=r["early_stopped"],
                 train_s=r["train_s"], eval_s=r["eval_s"], infer_test_s=r["infer_test_s"], device_mem_mb=r["device_mem_mb"], **{f"vocab_{k2}": v for k2, v in r["vocab"].items()})
        fold_rows.append(m)
        ml = compute_metrics(y[r["va_idx"]], r["oof_probs_last"], 0.5)
        ml.update(fold=k)
        fold_rows_last.append(ml)
        v = r["vocab"]
        log(f"fold {k} best @ epoch {r['best_epoch']}/{r['epochs_run']}: acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  F1neg {m['f1_neg']:.4f}  AUC {m['roc_auc']:.4f}  (last: acc {ml['accuracy']:.4f} mF1 {ml['macro_f1']:.4f}) | vocab {v['vocab_size']} ({v['glove_init_fraction']:.0%} GloVe), unk rate val {v['unk_rate_val']:.2%} test {v['unk_rate_test']:.2%} | train {r['train_s']/60:.1f} min")

    st = compute_run_stats(y, oof, oof_last, df["fold"].to_numpy(), fold_rows, fold_rows_last, args.acc_slack)
    test_pred_pos_ratio = float((test_prob[te_seen] >= st["threshold"]).mean())
    log_run_stats(log, st, len(folds), args.partial, test_pred_pos_ratio, float(y.mean()))
    total_time = time.perf_counter() - t_start
    train_steps_s = float(sum(r["train_s"] for r in fold_results))
    eval_s = float(sum(r["eval_s"] for r in fold_results))
    vocab_s = float(sum(r["vocab_s"] for r in fold_results))
    train_s_total = t_tok_train.seconds + vocab_s + train_steps_s + eval_s          # everything needed to produce the fold models (GloVe load excluded, see glove_load_s)
    infer_test_s = t_tok_test.seconds + float(np.mean([r["encode_test_s"] + r["infer_test_s"] for r in fold_results]))  # tokenise + encode + predict 1851 rows with one model
    log(f"== time: train steps {train_steps_s/60:.1f} min + held-out evals {eval_s/60:.1f} min, wall {total_time/60:.1f} min, peak RSS {peak_rss_mb():.0f} MB")
    if args.no_save:
        log("(--no-save: nothing written)")
        return True

    config = {k2: v for k2, v in vars(args).items() if k2 not in NON_CONFIG_KEYS}
    emb_dim = int(kv.vector_size)
    resources = {
        "feature_dim": emb_dim, "n_params": int(round(np.mean([r["n_params_trainable"] for r in fold_results]))),
        "n_params_total_fold_mean": float(np.mean([r["n_params"] for r in fold_results])), "n_params_excluding_embeddings": int(fold_results[0]["n_params_no_emb"]),
        "embeddings_trained": bool(all(r["emb_trained"] for r in fold_results)),
        "tokenize_train_s": t_tok_train.seconds, "tokenize_test_s": t_tok_test.seconds, "glove_load_s": t_glove.seconds, "vocab_build_s_total": vocab_s,
        "train_steps_s_total": train_steps_s, "eval_s_total": eval_s, "train_s_total": train_s_total,
        "infer_test_s_per_model": float(np.mean([r["infer_test_s"] for r in fold_results])), "infer_test_s": infer_test_s,
        "wall_s": total_time, "peak_rss_mb": peak_rss_mb(), "device_mem_mb_max": float(max(r["device_mem_mb"] for r in fold_results)),
        "machine": machine_tag(), "device": str(device), "precision": "fp32",
        "pretrained_vectors": {"name": args.glove, "disk_mb_approx": 376, "used_for": "vocabulary membership only" if args.no_pretrained else "vocabulary membership + initialisation"},
    }
    summary = {
        "run_id": run_id, "name": args.name, "model": f"{'rand' if args.no_pretrained else 'glove'}_{args.arch}", "feature_format": "word_vectors",
        "config": {k2: v for k2, v in vars(args).items() if k2 != "fold_list"},
        "protocol": {"dedup": True, "n_train": int(len(df)), "n_splits": N_SPLITS, "folds_run": folds, "partial": args.partial, "fold_seed": FOLD_SEED, "std_ddof": STD_DDOF,
                     "threshold_rule": f"max macro-F1 s.t. acc >= best-acc - {args.acc_slack}; final threshold chosen on all OOF",
                     "model_selection": f"best val macro-F1 @0.5 after each epoch on the held-out fold, early stopping patience {args.patience} (the stop epoch itself depends on the held-out fold): optimistic; "
                                        f"*_last = checkpoint at the stop epoch (best+{args.patience} or the --epochs cap), NOT selection-free (unlike train_plm's fixed schedule)",
                     "early_stopping": {"patience": args.patience, "suppressed_until_epoch": args.freeze_emb_epochs if (not args.no_pretrained and args.freeze_emb_epochs < args.epochs) else 0,
                                        "folds_early_stopped": [r["fold"] for r in fold_results if r["early_stopped"]]},
                     "vocabulary": f"built per fold from the training folds only (words in GloVe count>=1, others count>={args.min_count}); val/test unseen words -> <unk>; "
                                   f"embeddings {'random N(0,0.1) init' if args.no_pretrained else 'GloVe init'}"},
        "tokenization": {"style": "lower-case, punctuation split, Stanford-style contractions (do n't, it 's)", "max_len_words": args.max_len, "truncation": "head_tail",
                         "train_word_len_median": float(np.median(lens)), "train_word_len_p95": float(np.percentile(lens, 95))},
        "folds": fold_rows, "folds_last": fold_rows_last, "test_pred_pos_ratio": test_pred_pos_ratio, "resources": resources,
    }
    row = runs_row(run_id, summary["model"], "word_vectors", args.seed, len(folds), args.partial, emb_dim, resources["n_params"], train_s_total, infer_test_s,
                   peak_rss_mb(), f"{device.type}/fp32", None if device.type == "cpu" else round(resources["device_mem_mb_max"]), "best_val_macro_f1", st, test_pred_pos_ratio, config)
    write_run(RESULTS_DIR, run_id, oof, oof_last, test_prob, st, [pt for r in fold_results for pt in r["curve"]], summary, row, log)
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
