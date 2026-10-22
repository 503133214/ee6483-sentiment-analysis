"""Fine-tune a pretrained encoder (distilbert / bert / roberta) under the shared 5-fold protocol.

    python -m src.train_plm --model distilbert-base-uncased --name distilbert_lr3e-5 --lr 3e-5
    python -m src.train_plm --model roberta-base --name roberta_base --lr 2e-5 --max-len 256 --batch-size 16
    python -m src.train_plm --model distilbert-base-uncased --folds 0 --limit 256 --epochs 1 --no-save   # smoke test
    python -m src.train_plm --model roberta-base --name roberta_base --resume                          # continue a crashed run

Per fold: tokenise once (head+tail truncation to --max-len), dynamic padding, AdamW with linear
warmup/decay, class-weighted cross-entropy, fp16 autocast (+GradScaler) on MPS/CUDA, evaluate the
held-out fold every --eval-every epoch (macro-F1 at 0.5) and keep the best weights in memory.
That checkpoint's held-out probabilities become the fold's OOF probabilities and its test
probabilities are averaged over folds. The decision threshold is chosen on OOF only; the test set
never influences anything.

Model selection caveat: the checkpoint is picked on the same held-out fold it is scored on
(max over a handful of evaluations), so `cv_mean_std` / `oof_at_0.5` are slightly optimistic
compared with train_tfidf.py, which selects nothing. The unselected final-checkpoint estimate is
saved alongside as `*_last` (results/oof/<run_id>_last.npy, summary `cv_mean_std_last`,
runs.csv `acc_mean_last` / `macro_f1_mean_last`); quote both in the report.

Partial runs (--folds subset or --limit) are only written with --allow-partial and get a suffixed
run_id (e.g. `_f0-1`, `_n64`) so they can never replace a full run. Each finished fold is saved to
results/summary/<run_id>_fold<k>.npz; --resume reloads them (if the config matches) after a crash.
Non-determinism: MPS/CUDA kernels are not bitwise deterministic; seeds are fixed and the report
should average several --seed values.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Cap the MPS caching allocator: the PyTorch default (1.7 x recommended working set, ~21 GB on a 16 GB Mac)
# lets it grow past physical memory and the machine thrashes in swap instead of raising OOM.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")  # ~10.8 GB on a 16 GB Mac; above this -> OOM error, never swap
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")    # release cached blocks early

import argparse
import functools
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src import data as _data
from src.data import FOLD_SEED, LABEL_COL, N_SPLITS, TEXT_COL, load_test, load_train_folds
from src.metrics import STD_DDOF, choose_threshold, compute_metrics, mean_std, nested_threshold_metrics
from src.utils.resources import Timer, machine_tag, peak_rss_mb

transformers.logging.set_verbosity_error()
# EE6483_RESULTS_DIR lets tests redirect all outputs to a sandbox
RESULTS_DIR = Path(os.environ.get("EE6483_RESULTS_DIR", _data.RESULTS_DIR))
# keys that do not change what a fold computes; excluded from the --resume config comparison
NON_CONFIG_KEYS = {"name", "no_save", "allow_partial", "resume", "folds", "device"}
PAD_MULTIPLE = 64  # pad batch length up to a multiple of 64: only a handful of tensor shapes, so the MPS allocator cannot fragment
EMPTY_CACHE_EVERY = 50  # training steps between torch.mps.empty_cache() calls (bounds cached-block growth; no effect on results)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="distilbert-base-uncased", help="HF model id: distilbert-base-uncased, bert-base-uncased, roberta-base, ...")
    p.add_argument("--name", default=None, help="run name (default: model + max_len + batch + epochs + lr + class weight)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--truncation", choices=["head_tail", "head"], default="head_tail", help="how to shorten texts longer than --max-len")
    p.add_argument("--head-frac", type=float, default=0.75, help="share of the token budget kept from the start for head_tail")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--class-weight", choices=["none", "sqrt", "balanced"], default="sqrt", help="CE class weights: none, sqrt inverse frequency (mild), inverse frequency")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--eval-every", type=float, default=0.5, help="evaluate on the held-out fold every this fraction of an epoch")
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16", help="fp16 = autocast + GradScaler on MPS/CUDA (falls back to fp32 on non-finite loss)")
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--folds", default="all", help="comma-separated fold ids to run (e.g. 0,1) or all")
    p.add_argument("--limit", type=int, default=None, help="use only the first N train rows per fold and N test rows (smoke tests)")
    p.add_argument("--acc-slack", type=float, default=0.003)
    p.add_argument("--no-save", action="store_true", help="do not write results/ artifacts (smoke tests)")
    p.add_argument("--allow-partial", action="store_true", help="save a --folds subset / --limit run under a suffixed run_id")
    p.add_argument("--resume", action="store_true", help="reuse results/summary/<run_id>_fold<k>.npz files from an interrupted run with the same config")
    args = p.parse_args(argv)
    if args.name is None:
        tag = args.model.split("/")[-1].replace("-", "_")
        args.name = f"{tag}_len{args.max_len}_bs{args.batch_size}_ep{args.epochs}_lr{args.lr:g}_cw{args.class_weight}" + ("" if args.truncation == "head_tail" else "_head")
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


def config_signature(args) -> str:
    return json.dumps({k: v for k, v in vars(args).items() if k not in NON_CONFIG_KEYS | {"fold_list", "partial"}}, sort_keys=True, default=str)


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def free_device(device: torch.device) -> None:
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def special_affixes(tokenizer) -> tuple[list[int], list[int]]:
    """Prefix/suffix special-token ids the tokenizer adds around a single sequence
    ([CLS]/[SEP] for BERT, <s>/</s> for RoBERTa) - works for fast tokenizers in transformers 5."""
    core = tokenizer("hello", add_special_tokens=False)["input_ids"]
    full = tokenizer("hello", add_special_tokens=True)["input_ids"]
    for i in range(len(full) - len(core) + 1):
        if full[i:i + len(core)] == core:
            return full[:i], full[i + len(core):]
    raise RuntimeError("could not determine special-token template")


def encode_texts(tokenizer, texts, max_len: int, truncation: str, head_frac: float):
    prefix, suffix = special_affixes(tokenizer)
    budget = max_len - len(prefix) - len(suffix)
    raw = tokenizer(list(texts), add_special_tokens=False, truncation=False)["input_ids"]
    lengths = np.array([len(x) for x in raw])
    out, n_trunc = [], 0
    for ids in raw:
        if len(ids) > budget:
            n_trunc += 1
            if truncation == "head_tail":
                head = int(budget * head_frac)
                ids = ids[:head] + ids[len(ids) - (budget - head):]
            else:
                ids = ids[:budget]
        out.append(prefix + ids + suffix)
    assert max(len(x) for x in out) <= max_len
    stats = {
        "subword_len_mean": float(lengths.mean()), "subword_len_median": float(np.median(lengths)),
        "subword_len_p95": float(np.percentile(lengths, 95)), "subword_len_max": int(lengths.max()),
        "n_truncated": n_trunc, "frac_truncated": float(n_trunc / len(raw)),
    }
    return out, stats


def make_batches(indices, encoded, labels, batch_size, pad_id, shuffle, rng):
    """Yield padded batches; shuffled for training, length-sorted for inference (faster, same results)."""
    idx = np.array(indices)
    if shuffle:
        rng.shuffle(idx)
    else:
        idx = idx[np.argsort([len(encoded[i]) for i in idx], kind="stable")]
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        L = max(len(encoded[i]) for i in b)
        L = int(math.ceil(L / PAD_MULTIPLE) * PAD_MULTIPLE)
        ids = torch.full((len(b), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(b), L), dtype=torch.long)
        for r, i in enumerate(b):
            seq = encoded[i]
            ids[r, :len(seq)] = torch.tensor(seq)
            mask[r, :len(seq)] = 1
        y = torch.tensor([labels[i] for i in b], dtype=torch.long) if labels is not None else None
        yield b, ids, mask, y


@torch.no_grad()
def predict(model, indices, encoded, batch_size, pad_id, device, use_amp):
    """Positive-class probabilities for `indices`, returned in the order of `indices`."""
    was_training = model.training
    model.eval()
    probs = np.zeros(len(indices), dtype=float)
    pos = {i: r for r, i in enumerate(indices)}
    for b, ids, mask, _ in make_batches(indices, encoded, None, batch_size, pad_id, shuffle=False, rng=None):
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(input_ids=ids, attention_mask=mask).logits
        p = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        for i, v in zip(b, p):
            probs[pos[i]] = float(v)
    if was_training:
        model.train()
    return probs


def class_weights(y_train: np.ndarray, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(y_train, minlength=2)
    assert (counts > 0).all(), f"class weights need both classes in the training subset, got counts {counts.tolist()} (increase --limit)"
    freq = counts / len(y_train)
    w = 1.0 / np.sqrt(freq) if mode == "sqrt" else 1.0 / freq
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float)


def device_mem_mb(device: torch.device) -> float:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def fold_indices(k, args, df, n_test):
    y_all = df[LABEL_COL].to_numpy()
    tr_idx = np.where(df["fold"].to_numpy() != k)[0]
    va_idx = np.where(df["fold"].to_numpy() == k)[0]
    te_idx = np.arange(n_test)
    if args.limit:
        n_va = max(args.limit // 4, 8)
        # stratified prefix so the smoke validation set always holds both classes
        va_idx = np.sort(np.concatenate([va_idx[y_all[va_idx] == c][: max(n_va // 2, 1)] for c in (0, 1)]))
        tr_idx, te_idx = tr_idx[:args.limit], te_idx[:args.limit]
    return tr_idx, va_idx, te_idx


def train_fold(k, args, df, encoded_train, encoded_test, tokenizer, device, use_amp, log):
    """Train on folds != k, evaluate on fold k. Returns OOF probs (best and last checkpoint), test probs, curve, timings."""
    seed_everything(args.seed + k)
    rng = np.random.default_rng(args.seed + k)
    y_all = df[LABEL_COL].to_numpy()
    tr_idx, va_idx, te_idx = fold_indices(k, args, df, len(encoded_test))

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    model.train()
    weight = class_weights(y_all[tr_idx], args.class_weight)
    weight = weight.to(device) if weight is not None else None
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    steps_per_epoch = math.ceil(len(tr_idx) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * total_steps), total_steps)
    scaler = torch.amp.GradScaler(device.type) if (use_amp and device.type in ("mps", "cuda")) else None
    eval_interval = max(1, int(round(steps_per_epoch * args.eval_every)))
    pad_id = tokenizer.pad_token_id

    best = {"macro_f1": -1.0, "step": 0, "epoch": 0.0, "state": None, "val_probs": None}
    last_val_probs = None
    curve, step, run_loss, n_loss = [], 0, 0.0, 0
    train_time = eval_time = 0.0
    t_fold = time.perf_counter()
    for epoch in range(args.epochs):
        for b, ids, mask, y in make_batches(tr_idx, encoded_train, y_all, args.batch_size, pad_id, shuffle=True, rng=rng):
            t0 = time.perf_counter()
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
            loss = F.cross_entropy(logits.float(), y, weight=weight, label_smoothing=args.label_smoothing)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step} (precision {'fp16' if use_amp else 'fp32'})")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            scheduler.step()
            step += 1
            if device.type == "mps" and step % EMPTY_CACHE_EVERY == 0:
                torch.mps.empty_cache()
            run_loss += loss.item()
            n_loss += 1
            train_time += time.perf_counter() - t0
            if step % eval_interval == 0 or step == total_steps:
                t1 = time.perf_counter()
                if device.type == "mps":
                    torch.mps.empty_cache()  # drop cached blocks before the eval pass; keeps allocator fragmentation in check
                val_probs = predict(model, va_idx, encoded_train, args.batch_size, pad_id, device, use_amp)
                m = compute_metrics(y_all[va_idx], val_probs, 0.5)
                eval_time += time.perf_counter() - t1
                point = {"fold": k, "step": step, "epoch": round(step / steps_per_epoch, 3), "train_loss": run_loss / n_loss,
                         "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"], "val_f1_neg": m["f1_neg"], "val_auc": m["roc_auc"],
                         "lr": scheduler.get_last_lr()[0], "device_mem_mb": device_mem_mb(device)}
                curve.append(point)
                last_val_probs = val_probs
                improved = m["macro_f1"] > best["macro_f1"]
                if improved:
                    best = {"macro_f1": m["macro_f1"], "step": step, "epoch": point["epoch"], "val_probs": val_probs,
                            "state": {n: t.detach().to("cpu", copy=True) for n, t in model.state_dict().items()}}
                log(f"  fold {k} ep {point['epoch']:.2f} step {step}/{total_steps} loss {point['train_loss']:.4f} | val acc {m['accuracy']:.4f} mF1 {m['macro_f1']:.4f} F1neg {m['f1_neg']:.4f} AUC {m['roc_auc']:.4f}{' *' if improved else ''}")
                run_loss, n_loss = 0.0, 0

    model.load_state_dict(best["state"])
    t2 = time.perf_counter()
    test_probs = predict(model, te_idx, encoded_test, args.batch_size, pad_id, device, use_amp)
    infer_test_s = time.perf_counter() - t2
    n_params = sum(p.numel() for p in model.parameters())
    hidden_size = int(getattr(model.config, "hidden_size", getattr(model.config, "dim", 0)))
    mem = device_mem_mb(device)
    del model, optimizer, scheduler, best["state"]
    free_device(device)
    return {
        "fold": k, "va_idx": va_idx, "te_idx": te_idx, "oof_probs": best["val_probs"], "oof_probs_last": last_val_probs, "test_probs": test_probs,
        "n_train": int(len(tr_idx)), "best_step": best["step"], "best_epoch": best["epoch"], "total_steps": total_steps, "curve": curve,
        "train_s": train_time, "eval_s": eval_time, "infer_test_s": infer_test_s, "fold_wall_s": time.perf_counter() - t_fold,
        "n_params": n_params, "hidden_size": hidden_size, "device_mem_mb": mem, "precision": "fp16" if use_amp else "fp32",
    }


ARRAY_KEYS = ("va_idx", "te_idx", "oof_probs", "oof_probs_last", "test_probs")


def save_fold(path: Path, r: dict, signature: str) -> None:
    meta = {k: v for k, v in r.items() if k not in ARRAY_KEYS}
    np.savez(path, **{k: np.asarray(r[k]) for k in ARRAY_KEYS}, meta=json.dumps(meta, default=float), config=signature)


def load_fold(path: Path, signature: str) -> dict | None:
    z = np.load(path, allow_pickle=False)
    if str(z["config"]) != signature:
        return None
    r = json.loads(str(z["meta"]))
    r.update({k: z[k] for k in ARRAY_KEYS})
    return r


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    use_amp = args.precision == "fp16" and device.type in ("mps", "cuda")
    t_start = time.perf_counter()
    log = functools.partial(print, flush=True)
    run_id = run_id_for(args)
    signature = config_signature(args)
    folds = args.fold_list
    save = not args.no_save
    summary_dir = RESULTS_DIR / "summary"
    if save:
        for sub in ("oof", "test_probs", "summary"):
            (RESULTS_DIR / sub).mkdir(parents=True, exist_ok=True)
        if (summary_dir / f"{run_id}.json").exists():
            log(f"note: replacing existing run {run_id}")
    log(f"model {args.model} | device {device} | precision {'fp16' if use_amp else 'fp32'} | run {run_id}{' [PARTIAL]' if args.partial else ''} | seed {args.seed}")

    df = load_train_folds()
    test = load_test()
    y = df[LABEL_COL].to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    with Timer() as t_tok_train:
        encoded_train, tok_stats = encode_texts(tokenizer, df[TEXT_COL], args.max_len, args.truncation, args.head_frac)
    with Timer() as t_tok_test:
        encoded_test, tok_stats_test = encode_texts(tokenizer, test[TEXT_COL], args.max_len, args.truncation, args.head_frac)
    log(f"tokenised in {t_tok_train.seconds + t_tok_test.seconds:.1f}s | train subword len median {tok_stats['subword_len_median']:.0f} p95 {tok_stats['subword_len_p95']:.0f} | truncated {tok_stats['frac_truncated']:.1%} (test {tok_stats_test['frac_truncated']:.1%}) at max_len {args.max_len}")

    oof = np.full(len(df), np.nan)
    oof_last = np.full(len(df), np.nan)
    test_prob = np.zeros(len(test))
    te_seen = np.zeros(len(test), dtype=bool)
    fold_results, fold_rows, fold_rows_last = [], [], []
    for k in folds:
        log(f"== fold {k} ==")
        fold_file = summary_dir / f"{run_id}_fold{k}.npz"
        r = load_fold(fold_file, signature) if (args.resume and save and fold_file.exists()) else None
        if r is not None:
            log(f"  fold {k}: loaded from {fold_file.name} (--resume)")
        else:
            try:
                r = train_fold(k, args, df, encoded_train, encoded_test, tokenizer, device, use_amp, log)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    log(f"  fold {k}: device out of memory ({str(e)[:120]}...). Re-run with a smaller --batch-size or --max-len; "
                        f"finished folds are kept in results/summary/{run_id}_fold*.npz and --resume will reuse them.")
                raise
            except FloatingPointError as e:
                if not use_amp:
                    raise
                log(f"  {e}; retrying fold {k} in fp32")
                r = None
            if r is None:  # the failed frame is released before the retry so its model/optimizer memory is freed
                free_device(device)
                r = train_fold(k, args, df, encoded_train, encoded_test, tokenizer, device, False, log)
            if save:
                save_fold(fold_file, r, signature)
        fold_results.append(r)
        oof[r["va_idx"]] = r["oof_probs"]
        oof_last[r["va_idx"]] = r["oof_probs_last"]
        test_prob[r["te_idx"]] += r["test_probs"] / len(folds)
        te_seen[r["te_idx"]] = True
        m = compute_metrics(y[r["va_idx"]], r["oof_probs"], 0.5)
        m.update(fold=k, n_train=r["n_train"], n_val=int(len(r["va_idx"])), best_step=r["best_step"], best_epoch=r["best_epoch"], total_steps=r["total_steps"],
                 train_s=r["train_s"], eval_s=r["eval_s"], infer_test_s=r["infer_test_s"], precision=r["precision"], device_mem_mb=r["device_mem_mb"])
        fold_rows.append(m)
        ml = compute_metrics(y[r["va_idx"]], r["oof_probs_last"], 0.5)
        ml.update(fold=k)
        fold_rows_last.append(ml)
        log(f"fold {k} best @ epoch {r['best_epoch']}: acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  F1neg {m['f1_neg']:.4f}  AUC {m['roc_auc']:.4f}  (last ckpt: acc {ml['accuracy']:.4f} mF1 {ml['macro_f1']:.4f}) | train {r['train_s']/60:.1f} min, eval {r['eval_s']/60:.1f} min, test infer {r['infer_test_s']:.1f}s, mem {r['device_mem_mb']:.0f} MB")

    mask = ~np.isnan(oof)
    fold_mask = df["fold"].to_numpy()[mask]
    folds_df, folds_last_df = pd.DataFrame(fold_rows), pd.DataFrame(fold_rows_last)
    keys = ["accuracy", "macro_f1", "f1_neg", "precision_neg", "recall_neg", "roc_auc"]
    summary_stats = {key: dict(zip(("mean", "std"), mean_std(folds_df[key]))) for key in keys}
    summary_stats_last = {key: dict(zip(("mean", "std"), mean_std(folds_last_df[key]))) for key in keys}
    oof_05 = compute_metrics(y[mask], oof[mask], 0.5)
    oof_05_last = compute_metrics(y[mask], oof_last[mask], 0.5)
    thr, thr_table = choose_threshold(y[mask], oof[mask], acc_slack=args.acc_slack)
    oof_thr = compute_metrics(y[mask], oof[mask], thr)
    oof_thr["note"] = "in-sample: threshold selected on these same OOF probabilities; see oof_at_threshold_nested"
    oof_nested = nested_threshold_metrics(y[mask], oof[mask], fold_mask, acc_slack=args.acc_slack)
    oof_nested["note"] = oof_nested.get("note", "") + " (nested w.r.t. the threshold only; checkpoint selection bias remains, see *_last)"
    test_pred_pos_ratio = float((test_prob[te_seen] >= thr).mean())
    precisions = sorted({r["precision"] for r in fold_results})
    run_precision = precisions[0] if len(precisions) == 1 else "mixed(" + "/".join(precisions) + ")"
    selection_note = f"checkpoint = best val macro-F1 @0.5 on the same held-out fold (evaluated every {args.eval_every} epoch, max over {math.ceil(args.epochs / args.eval_every)} evals): slightly optimistic; *_last = final checkpoint, no selection"

    log(f"\n== {len(folds)}-fold mean ± std (threshold 0.5){' [PARTIAL]' if args.partial else ''} ==")
    for key in keys:
        log(f"  {key:14s} {summary_stats[key]['mean']:.4f} ± {summary_stats[key]['std']:.4f}   (last ckpt {summary_stats_last[key]['mean']:.4f} ± {summary_stats_last[key]['std']:.4f})")
    log(f"== OOF @0.5: acc {oof_05['accuracy']:.4f}  macro-F1 {oof_05['macro_f1']:.4f}  F1neg {oof_05['f1_neg']:.4f}  AUC {oof_05['roc_auc']:.4f}   (last ckpt: acc {oof_05_last['accuracy']:.4f} macro-F1 {oof_05_last['macro_f1']:.4f})")
    log(f"== OOF @thr={thr:.2f}: acc {oof_thr['accuracy']:.4f}  macro-F1 {oof_thr['macro_f1']:.4f}  F1neg {oof_thr['f1_neg']:.4f}  (in-sample for the threshold)")
    if oof_nested["fold_thresholds"]:
        log(f"== OOF nested-threshold: acc {oof_nested['accuracy']:.4f}  macro-F1 {oof_nested['macro_f1']:.4f}  F1neg {oof_nested['f1_neg']:.4f}  fold thresholds {oof_nested['fold_thresholds']}")
    log(f"== test: predicted positive ratio @thr = {test_pred_pos_ratio:.4f} (train prior {y.mean():.4f})")
    total_time = time.perf_counter() - t_start
    train_steps_s = float(sum(r["train_s"] for r in fold_results))
    eval_s = float(sum(r["eval_s"] for r in fold_results))
    train_s_total = t_tok_train.seconds + train_steps_s + eval_s  # everything needed to produce the fold models (like tfidf vectorise + fit)
    infer_test_s = t_tok_test.seconds + float(np.mean([r["infer_test_s"] for r in fold_results]))  # tokenise + predict 1851 rows with one model
    log(f"== time: train steps {train_steps_s/60:.1f} min + held-out evals {eval_s/60:.1f} min, wall {total_time/60:.1f} min, precision {run_precision}, peak RSS {peak_rss_mb():.0f} MB")

    if not save:
        log("(--no-save: nothing written)")
        return True
    np.save(RESULTS_DIR / "oof" / f"{run_id}.npy", oof)
    np.save(RESULTS_DIR / "oof" / f"{run_id}_last.npy", oof_last)
    np.save(RESULTS_DIR / "test_probs" / f"{run_id}.npy", test_prob)
    thr_table.to_csv(summary_dir / f"{run_id}_thresholds.csv", index=False)
    pd.DataFrame([pt for r in fold_results for pt in r["curve"]]).to_csv(summary_dir / f"{run_id}_curves.csv", index=False)
    summary = {
        "run_id": run_id, "name": args.name, "model": args.model, "feature_format": "plm_finetune",
        "config": {k: v for k, v in vars(args).items() if k != "fold_list"},
        "protocol": {"dedup": True, "n_train": int(len(df)), "n_splits": N_SPLITS, "folds_run": folds, "partial": args.partial, "fold_seed": FOLD_SEED, "std_ddof": STD_DDOF,
                     "threshold_rule": f"max macro-F1 s.t. acc >= best-acc - {args.acc_slack}; final threshold chosen on all OOF",
                     "model_selection": selection_note},
        "tokenization": {"train": tok_stats, "test": tok_stats_test, "max_len": args.max_len, "truncation": args.truncation},
        "folds": fold_rows, "folds_last": fold_rows_last,
        "cv_mean_std": summary_stats, "cv_mean_std_last": summary_stats_last,
        "oof_at_0.5": oof_05, "oof_at_0.5_last": oof_05_last,
        "threshold": thr, "oof_at_threshold": oof_thr, "oof_at_threshold_nested": oof_nested, "test_pred_pos_ratio": test_pred_pos_ratio,
        "resources": {
            "feature_dim": int(fold_results[0]["hidden_size"]), "n_params": int(fold_results[0]["n_params"]),
            "tokenize_train_s": t_tok_train.seconds, "tokenize_test_s": t_tok_test.seconds,
            "train_steps_s_total": train_steps_s, "eval_s_total": eval_s, "train_s_total": train_s_total,
            "train_s_per_epoch": train_steps_s / (len(folds) * args.epochs),
            "infer_test_s_per_model": float(np.mean([r["infer_test_s"] for r in fold_results])), "infer_test_s": infer_test_s,
            "wall_s": total_time, "peak_rss_mb": peak_rss_mb(), "device_mem_mb_max": float(max(r["device_mem_mb"] for r in fold_results)),
            "machine": machine_tag(), "device": str(device), "precision": run_precision,
            "fp32_fallback_folds": [r["fold"] for r in fold_results if r["precision"] == "fp32" and use_amp],
        },
        "artifacts": {"oof": f"results/oof/{run_id}.npy", "oof_last": f"results/oof/{run_id}_last.npy", "test_probs": f"results/test_probs/{run_id}.npy",
                      "thresholds": f"results/summary/{run_id}_thresholds.csv", "curves": f"results/summary/{run_id}_curves.csv"},
    }
    (summary_dir / f"{run_id}.json").write_text(json.dumps(summary, indent=2, default=float))

    row = {
        "run_id": run_id, "model": args.model, "feature_format": "plm_finetune", "seed": args.seed, "n_splits": len(folds), "partial": args.partial,
        "feature_dim": summary["resources"]["feature_dim"], "n_params": summary["resources"]["n_params"],
        "train_s_total": round(train_s_total, 1), "infer_test_s": round(infer_test_s, 2),
        "peak_rss_mb": round(peak_rss_mb(), 1), "device": f"{device.type}/{run_precision}", "device_mem_mb": round(summary["resources"]["device_mem_mb_max"]),
        "selection": "best_val_macro_f1",
        "acc_mean": summary_stats["accuracy"]["mean"], "acc_std": summary_stats["accuracy"]["std"],
        "macro_f1_mean": summary_stats["macro_f1"]["mean"], "macro_f1_std": summary_stats["macro_f1"]["std"],
        "f1_neg_mean": summary_stats["f1_neg"]["mean"], "roc_auc_mean": summary_stats["roc_auc"]["mean"],
        "acc_mean_last": summary_stats_last["accuracy"]["mean"], "macro_f1_mean_last": summary_stats_last["macro_f1"]["mean"],
        "threshold": thr, "oof_acc_at_thr_insample": oof_thr["accuracy"], "oof_macro_f1_at_thr_insample": oof_thr["macro_f1"],
        "oof_acc_at_thr_nested": oof_nested["accuracy"], "oof_macro_f1_at_thr_nested": oof_nested["macro_f1"],
        "test_pos_ratio": test_pred_pos_ratio,
        "config": json.dumps({k: v for k, v in vars(args).items() if k not in NON_CONFIG_KEYS | {"fold_list", "partial"}}, default=str),
    }
    runs_path = RESULTS_DIR / "runs.csv"
    new = pd.DataFrame([row])
    if runs_path.exists():
        old = pd.read_csv(runs_path)
        new = pd.concat([old[old["run_id"] != run_id], new], ignore_index=True)
    new.to_csv(runs_path, index=False)
    for k in folds:  # per-fold checkpoints are only needed for --resume
        (summary_dir / f"{run_id}_fold{k}.npz").unlink(missing_ok=True)
    log(f"\nsaved: results/summary/{run_id}.json (+_curves.csv, _thresholds.csv), results/oof/{run_id}.npy (+_last), results/test_probs/{run_id}.npy; updated results/runs.csv")
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
