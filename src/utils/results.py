"""Shared result bookkeeping for the neural training scripts.

compute_run_stats(): fold mean ± std (best and last checkpoint), OOF metrics at 0.5, the OOF
threshold rule, in-sample and nested threshold metrics.
write_run(): results/oof/<run_id>.npy (+_last), results/test_probs/<run_id>.npy,
results/summary/<run_id>.json (+_thresholds.csv, _curves.csv) and the runs.csv row
(replace-by-run_id). Mirrors what train_tfidf.py / train_plm.py write so every model is comparable.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import choose_threshold, compute_metrics, mean_std, nested_threshold_metrics

STAT_KEYS = ["accuracy", "macro_f1", "f1_neg", "precision_neg", "recall_neg", "roc_auc"]


def compute_run_stats(y, oof, oof_last, fold, fold_rows, fold_rows_last, acc_slack: float) -> dict:
    mask = ~np.isnan(oof)
    folds_df, folds_last_df = pd.DataFrame(fold_rows), pd.DataFrame(fold_rows_last)
    st = {
        "cv_mean_std": {k: dict(zip(("mean", "std"), mean_std(folds_df[k]))) for k in STAT_KEYS},
        "cv_mean_std_last": {k: dict(zip(("mean", "std"), mean_std(folds_last_df[k]))) for k in STAT_KEYS},
        "oof_at_0.5": compute_metrics(y[mask], oof[mask], 0.5),
        "oof_at_0.5_last": compute_metrics(y[mask], oof_last[mask], 0.5),
    }
    thr, thr_table = choose_threshold(y[mask], oof[mask], acc_slack=acc_slack)
    oof_thr = compute_metrics(y[mask], oof[mask], thr)
    oof_thr["note"] = "in-sample: threshold selected on these same OOF probabilities; see oof_at_threshold_nested"
    nested = nested_threshold_metrics(y[mask], oof[mask], np.asarray(fold)[mask], acc_slack=acc_slack)
    nested["note"] = nested.get("note", "") + " (nested w.r.t. the threshold only; checkpoint selection bias remains, see *_last)"
    st.update(threshold=thr, thr_table=thr_table, oof_at_threshold=oof_thr, oof_at_threshold_nested=nested)
    return st


def log_run_stats(log, st: dict, n_folds: int, partial: bool, test_pred_pos_ratio: float, prior: float) -> None:
    log(f"\n== {n_folds}-fold mean ± std (threshold 0.5){' [PARTIAL]' if partial else ''} ==")
    for k in STAT_KEYS:
        a, b = st["cv_mean_std"][k], st["cv_mean_std_last"][k]
        log(f"  {k:14s} {a['mean']:.4f} ± {a['std']:.4f}   (last ckpt {b['mean']:.4f} ± {b['std']:.4f})")
    o, ol = st["oof_at_0.5"], st["oof_at_0.5_last"]
    log(f"== OOF @0.5: acc {o['accuracy']:.4f}  macro-F1 {o['macro_f1']:.4f}  F1neg {o['f1_neg']:.4f}  AUC {o['roc_auc']:.4f}   (last ckpt: acc {ol['accuracy']:.4f} macro-F1 {ol['macro_f1']:.4f})")
    t = st["oof_at_threshold"]
    log(f"== OOF @thr={st['threshold']:.2f}: acc {t['accuracy']:.4f}  macro-F1 {t['macro_f1']:.4f}  F1neg {t['f1_neg']:.4f}  (in-sample for the threshold)")
    n = st["oof_at_threshold_nested"]
    if n["fold_thresholds"]:
        log(f"== OOF nested-threshold: acc {n['accuracy']:.4f}  macro-F1 {n['macro_f1']:.4f}  F1neg {n['f1_neg']:.4f}  fold thresholds {n['fold_thresholds']}")
    log(f"== test: predicted positive ratio @thr = {test_pred_pos_ratio:.4f} (train prior {prior:.4f})")


def runs_row(run_id, model, feature_format, seed, n_folds, partial, feature_dim, n_params, train_s_total, infer_test_s,
             peak_rss_mb, device, device_mem_mb, selection, st, test_pred_pos_ratio, config) -> dict:
    cv, cvl, t, n = st["cv_mean_std"], st["cv_mean_std_last"], st["oof_at_threshold"], st["oof_at_threshold_nested"]
    return {
        "run_id": run_id, "model": model, "feature_format": feature_format, "seed": seed, "n_splits": n_folds, "partial": partial,
        "feature_dim": feature_dim, "n_params": n_params, "train_s_total": round(train_s_total, 1), "infer_test_s": round(infer_test_s, 2),
        "peak_rss_mb": round(peak_rss_mb, 1), "device": device, "device_mem_mb": device_mem_mb, "selection": selection,
        "acc_mean": cv["accuracy"]["mean"], "acc_std": cv["accuracy"]["std"], "macro_f1_mean": cv["macro_f1"]["mean"], "macro_f1_std": cv["macro_f1"]["std"],
        "f1_neg_mean": cv["f1_neg"]["mean"], "roc_auc_mean": cv["roc_auc"]["mean"],
        "acc_mean_last": cvl["accuracy"]["mean"], "macro_f1_mean_last": cvl["macro_f1"]["mean"],
        "threshold": st["threshold"], "oof_acc_at_thr_insample": t["accuracy"], "oof_macro_f1_at_thr_insample": t["macro_f1"],
        "oof_acc_at_thr_nested": n["accuracy"], "oof_macro_f1_at_thr_nested": n["macro_f1"],
        "test_pos_ratio": test_pred_pos_ratio, "config": json.dumps(config, default=str),
    }


def write_run(results_dir: Path, run_id: str, oof, oof_last, test_prob, st: dict, curves: list[dict], summary: dict, row: dict, log) -> None:
    results_dir = Path(results_dir)
    for sub in ("oof", "test_probs", "summary"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "oof" / f"{run_id}.npy", oof)
    np.save(results_dir / "oof" / f"{run_id}_last.npy", oof_last)
    np.save(results_dir / "test_probs" / f"{run_id}.npy", test_prob)
    st["thr_table"].to_csv(results_dir / "summary" / f"{run_id}_thresholds.csv", index=False)
    pd.DataFrame(curves).to_csv(results_dir / "summary" / f"{run_id}_curves.csv", index=False)
    summary = dict(summary)
    summary.update({k: st[k] for k in ("cv_mean_std", "cv_mean_std_last", "oof_at_0.5", "oof_at_0.5_last", "threshold", "oof_at_threshold", "oof_at_threshold_nested")})
    summary["artifacts"] = {"oof": f"results/oof/{run_id}.npy", "oof_last": f"results/oof/{run_id}_last.npy", "test_probs": f"results/test_probs/{run_id}.npy",
                            "thresholds": f"results/summary/{run_id}_thresholds.csv", "curves": f"results/summary/{run_id}_curves.csv"}
    (results_dir / "summary" / f"{run_id}.json").write_text(json.dumps(summary, indent=2, default=float))
    runs_path = results_dir / "runs.csv"
    new = pd.DataFrame([row])
    if runs_path.exists():
        old = pd.read_csv(runs_path)
        new = pd.concat([old[old["run_id"] != run_id], new], ignore_index=True)
    new.to_csv(runs_path, index=False)
    log(f"\nsaved: results/summary/{run_id}.json (+_curves.csv, _thresholds.csv), results/oof/{run_id}.npy (+_last), results/test_probs/{run_id}.npy; updated results/runs.csv")
