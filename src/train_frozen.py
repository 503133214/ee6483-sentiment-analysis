"""Frozen pretrained encoder/decoder features + logistic regression under the shared 5-fold protocol.

    python -m src.train_frozen --model roberta-base --pooling mean --name roberta_frozen_lr
    python -m src.train_frozen --model Qwen/Qwen2.5-1.5B --pooling last --load-dtype bf16 --name qwen2.5_1.5b_frozen_lr

The pretrained model is used as a fixed feature extractor (no labels involved, like a tokenizer):
one forward pass over all 7185 training reviews and the 1851 test reviews, pooled to one vector
per text. Then, per fold, a StandardScaler + LogisticRegression is fitted on the training folds
only (same folds as every other model, class_weight balanced, fixed C), scored on the held-out
fold (OOF) and applied to the test set (probabilities averaged over folds). This is the
"feature extraction vs fine-tuning" row of the resource/accuracy comparison (report section g).
"""
from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

from src import data as _data
from src.data import FOLD_SEED, LABEL_COL, N_SPLITS, TEXT_COL, load_test, load_train_folds
from src.metrics import STD_DDOF, compute_metrics
from src.train_plm import TORCH_DTYPES, encode_texts, fold_indices, make_batches, pick_device, seed_everything
from src.utils.resources import Timer, machine_tag, peak_rss_mb
from src.utils.results import compute_run_stats, log_run_stats, runs_row, write_run

transformers.logging.set_verbosity_error()
RESULTS_DIR = Path(os.environ.get("EE6483_RESULTS_DIR", _data.RESULTS_DIR))
NON_CONFIG_KEYS = {"name", "no_save", "allow_partial", "folds", "device", "fold_list", "partial", "features_cache"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--name", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pooling", choices=["mean", "last", "cls"], default="mean", help="mean over non-pad tokens; last non-pad token (decoder models); first token (CLS)")
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--truncation", choices=["head_tail", "head"], default="head_tail")
    p.add_argument("--head-frac", type=float, default=0.75)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--load-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    p.add_argument("--C", type=float, default=1.0, help="inverse L2 strength of the logistic regression (fixed, no inner selection)")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    p.add_argument("--device", default="auto")
    p.add_argument("--folds", default="all")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--acc-slack", type=float, default=0.003)
    p.add_argument("--features-cache", default=None, help="optional .npz to cache/reuse the extracted features")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--allow-partial", action="store_true")
    args = p.parse_args(argv)
    if args.name is None:
        args.name = f"{args.model.split('/')[-1].replace('-', '_')}_frozen_{args.pooling}_lr"
    folds = list(range(N_SPLITS)) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
    if len(set(folds)) != len(folds) or not set(folds) <= set(range(N_SPLITS)):
        p.error(f"--folds must be unique ids in 0..{N_SPLITS - 1}, got {folds}")
    args.fold_list = folds
    args.partial = len(folds) < N_SPLITS or bool(args.limit)
    if args.partial and not args.no_save and not args.allow_partial:
        p.error("a --folds subset or --limit run is partial: add --no-save or --allow-partial")
    return args


def run_id_for(args) -> str:
    rid = f"{args.name}_seed{args.seed}"
    if len(args.fold_list) < N_SPLITS:
        rid += "_f" + "-".join(map(str, args.fold_list))
    if args.limit:
        rid += f"_n{args.limit}"
    return rid


@torch.no_grad()
def extract(model, encoded, pad_id, batch_size, device, pooling, dtype, log):
    """One pooled vector per text, in input order."""
    model.eval()
    feats = [None] * len(encoded)
    done = 0
    for b, ids, mask, _ in make_batches(list(range(len(encoded))), encoded, None, batch_size, pad_id, shuffle=False, rng=None):
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32 and device.type in ("cuda", "mps")):
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state.float()   # (B, L, H)
        if pooling == "mean":
            m = mask.unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        elif pooling == "last":
            last = mask.sum(1) - 1
            v = h[torch.arange(h.shape[0], device=device), last]
        else:
            v = h[:, 0]
        v = v.cpu().numpy()
        for i, row in zip(b, v):
            feats[i] = row
        done += len(b)
        if done % (batch_size * 50) < batch_size:
            log(f"  extracted {done}/{len(encoded)}")
    return np.stack(feats).astype(np.float32)


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    log = functools.partial(print, flush=True)
    run_id = run_id_for(args)
    t_start = time.perf_counter()
    seed_everything(args.seed)
    log(f"frozen {args.model} | pooling {args.pooling} | device {device} | load dtype {args.load_dtype} | run {run_id}{' [PARTIAL]' if args.partial else ''}")

    df = load_train_folds()
    test = load_test()
    y = df[LABEL_COL].to_numpy()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    with Timer() as t_tok:
        enc_train, tok_stats = encode_texts(tokenizer, df[TEXT_COL], args.max_len, args.truncation, args.head_frac)
        enc_test, tok_stats_test = encode_texts(tokenizer, test[TEXT_COL], args.max_len, args.truncation, args.head_frac)
    n_tr = args.limit or len(enc_train)
    n_te = args.limit or len(enc_test)

    cache = Path(args.features_cache) if args.features_cache else None
    if cache and cache.exists():
        z = np.load(cache)
        X_train, X_test, t_feat_s, n_params, hidden = z["X_train"], z["X_test"], float(z["extract_s"]), int(z["n_params"]), int(z["hidden"])
        log(f"features loaded from {cache}")
    else:
        dtype = TORCH_DTYPES[args.load_dtype]
        try:
            model = AutoModel.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            model = AutoModel.from_pretrained(args.model, torch_dtype=dtype)
        model.config.pad_token_id = tokenizer.pad_token_id
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        hidden = int(getattr(model.config, "hidden_size", getattr(model.config, "dim", 0)))
        with Timer() as t_feat:
            X_train = extract(model, enc_train[:n_tr], tokenizer.pad_token_id, args.batch_size, device, args.pooling, dtype, log)
            X_test = extract(model, enc_test[:n_te], tokenizer.pad_token_id, args.batch_size, device, args.pooling, dtype, log)
        t_feat_s = t_feat.seconds
        del model
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, X_train=X_train, X_test=X_test, extract_s=t_feat_s, n_params=n_params, hidden=hidden)
    log(f"features: train {X_train.shape}, test {X_test.shape}, extracted in {t_feat_s/60:.1f} min, {n_params/1e6:.0f}M params")

    folds = args.fold_list
    oof, oof_last = np.full(len(df), np.nan), np.full(len(df), np.nan)
    test_prob, te_seen = np.zeros(len(test)), np.zeros(len(test), dtype=bool)
    fold_rows, fold_rows_last, t_fit_total = [], [], 0.0
    for k in folds:
        tr_idx, va_idx, te_idx = fold_indices(k, args, df, len(test))
        if args.limit:
            tr_idx, va_idx = tr_idx[tr_idx < n_tr], va_idx[va_idx < n_tr]
        with Timer() as t_fit:
            scaler = StandardScaler().fit(X_train[tr_idx])
            clf = LogisticRegression(C=args.C, class_weight=None if args.class_weight == "none" else "balanced", max_iter=5000, random_state=args.seed)
            clf.fit(scaler.transform(X_train[tr_idx]), y[tr_idx])
        t_fit_total += t_fit.seconds
        p_va = clf.predict_proba(scaler.transform(X_train[va_idx]))[:, 1]
        p_te = clf.predict_proba(scaler.transform(X_test[te_idx]))[:, 1]
        oof[va_idx], oof_last[va_idx] = p_va, p_va
        test_prob[te_idx] += p_te / len(folds)
        te_seen[te_idx] = True
        m = compute_metrics(y[va_idx], p_va, 0.5)
        m.update(fold=k, n_train=int(len(tr_idx)), n_val=int(len(va_idx)), fit_s=t_fit.seconds)
        fold_rows.append(m)
        fold_rows_last.append({**compute_metrics(y[va_idx], p_va, 0.5), "fold": k})
        log(f"fold {k}: acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  F1neg {m['f1_neg']:.4f}  AUC {m['roc_auc']:.4f}  (LR fit {t_fit.seconds:.1f}s)")

    st = compute_run_stats(y, oof, oof_last, df["fold"].to_numpy(), fold_rows, fold_rows_last, args.acc_slack)
    test_pred_pos_ratio = float((test_prob[te_seen] >= st["threshold"]).mean())
    log_run_stats(log, st, len(folds), args.partial, test_pred_pos_ratio, float(y.mean()))
    total = time.perf_counter() - t_start
    train_s_total = t_tok.seconds + t_feat_s * (n_tr / max(n_tr + n_te, 1)) + t_fit_total   # tokenise + train-side feature extraction + 5 LR fits
    infer_test_s = t_feat_s * (n_te / max(n_tr + n_te, 1)) / 1.0 + 0.05                    # feature extraction for the test rows (one pass) + LR predict
    log(f"== time: feature extraction {t_feat_s/60:.1f} min, LR fits {t_fit_total:.1f}s, wall {total/60:.1f} min")
    if args.no_save:
        log("(--no-save: nothing written)")
        return True
    config = {k2: v for k2, v in vars(args).items() if k2 not in NON_CONFIG_KEYS}
    resources = {"feature_dim": hidden, "n_params": 0, "n_params_frozen_extractor": n_params, "tokenize_s": t_tok.seconds, "extract_s_total": t_feat_s,
                 "lr_fit_s_total": t_fit_total, "train_s_total": train_s_total, "infer_test_s": infer_test_s, "wall_s": total, "peak_rss_mb": peak_rss_mb(),
                 "device_mem_mb_max": float(torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else None, "machine": machine_tag(), "device": str(device), "precision": args.load_dtype}
    summary = {"run_id": run_id, "name": args.name, "model": f"{args.model}+LR", "feature_format": "frozen_plm", "config": {k2: v for k2, v in vars(args).items() if k2 != "fold_list"},
               "protocol": {"dedup": True, "n_train": int(len(df)), "n_splits": N_SPLITS, "folds_run": folds, "partial": args.partial, "fold_seed": FOLD_SEED, "std_ddof": STD_DDOF,
                            "threshold_rule": f"max macro-F1 s.t. acc >= best-acc - {args.acc_slack}; final threshold chosen on all OOF",
                            "model_selection": "none (frozen features, fixed C); *_last identical to the main estimate",
                            "features": f"{args.pooling} pooling of the last hidden layer; extractor never sees labels; scaler + LR fitted on training folds only"},
               "tokenization": {"train": tok_stats, "test": tok_stats_test, "max_len": args.max_len, "truncation": args.truncation},
               "folds": fold_rows, "folds_last": fold_rows_last, "test_pred_pos_ratio": test_pred_pos_ratio, "resources": resources}
    row = runs_row(run_id, summary["model"], "frozen_plm", args.seed, len(folds), args.partial, hidden, 0, train_s_total, infer_test_s, peak_rss_mb(),
                   f"{device.type}/{args.load_dtype}", None if device.type != "cuda" else round(resources["device_mem_mb_max"]), "none", st, test_pred_pos_ratio, config)
    write_run(RESULTS_DIR, run_id, oof, oof_last, test_prob, st, [], summary, row, log)
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
