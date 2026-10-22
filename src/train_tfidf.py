"""TF-IDF + Logistic Regression baseline under the shared 5-fold protocol.

Run from the project root, e.g.
    python -m src.train_tfidf --name tfidf_lr_neg            # default: 1-2gram + NOT_ marking + C=4 + balanced
    python -m src.train_tfidf --name tfidf_lr_plain --no-negation --class-weight none --C 1

What it does
  * for each fold: fit TfidfVectorizer on the 4 training folds ONLY, train LR,
    predict the held-out fold (OOF) and the test set
  * reports per-fold and OOF metrics (accuracy, macro-F1, F1(neg), AUC, ...)
  * picks a decision threshold on the OOF probabilities (never on test)
  * saves OOF probs, test probs (5-fold average), a JSON summary, and appends one
    row with resource numbers to results/runs.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import FOLD_SEED, LABEL_COL, N_SPLITS, RESULTS_DIR, TEXT_COL, load_test, load_train_folds
from src.features.negation import mark_negation
from src.metrics import STD_DDOF, choose_threshold, compute_metrics, mean_std, nested_threshold_metrics
from src.utils.resources import Timer, machine_tag, peak_rss_mb

# keeps apostrophes inside words (don't -> don't, not don + t) and keeps NOT_ prefixes
TOKEN_PATTERN = r"(?u)\b\w[\w']*\b"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="tfidf_lr_neg", help="run name used for output files")
    p.add_argument("--seed", type=int, default=42, help="kept in run_id for a uniform schema with stochastic models; folds are fixed by data/folds.csv and lbfgs LR is deterministic, so it does not change TF-IDF+LR results")
    p.add_argument("--C", type=float, default=4.0, help="inverse L2 regularisation strength")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    p.add_argument("--ngram-max", type=int, default=2)
    p.add_argument("--min-df", type=int, default=2)
    p.add_argument("--max-df", type=float, default=0.95)
    p.add_argument("--no-sublinear-tf", action="store_true")
    p.add_argument("--no-negation", action="store_true", help="disable NOT_ negation-scope marking")
    p.add_argument("--acc-slack", type=float, default=0.003, help="threshold rule: max macro-F1 s.t. acc >= best-acc - slack")
    return p.parse_args(argv)


def build_vectorizer(args) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
        max_df=args.max_df,
        sublinear_tf=not args.no_sublinear_tf,
        token_pattern=TOKEN_PATTERN,
        lowercase=True,
        dtype=np.float32,
    )


def build_classifier(args) -> LogisticRegression:
    return LogisticRegression(
        C=args.C,
        class_weight=None if args.class_weight == "none" else "balanced",
        max_iter=5000,
        random_state=args.seed,
    )


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    t_start = time.perf_counter()

    df = load_train_folds()
    test = load_test()
    y = df[LABEL_COL].to_numpy()
    prep = (lambda s: s) if args.no_negation else mark_negation
    with Timer() as t_prep:
        texts = df[TEXT_COL].map(prep).to_numpy()
        test_texts = test[TEXT_COL].map(prep).to_numpy()

    oof = np.zeros(len(df), dtype=float)
    test_prob = np.zeros(len(test), dtype=float)
    fold_rows, feat_dims, n_params = [], [], []
    t_vec = t_fit = t_infer = 0.0
    for k in range(N_SPLITS):
        tr, va = (df["fold"] != k).to_numpy(), (df["fold"] == k).to_numpy()
        vec, clf = build_vectorizer(args), build_classifier(args)
        with Timer() as tv:
            X_tr = vec.fit_transform(texts[tr])          # fit on training folds only
            X_va = vec.transform(texts[va])
        with Timer() as tf_:
            clf.fit(X_tr, y[tr])
        with Timer() as ti:
            X_te = vec.transform(test_texts)
            p_te = clf.predict_proba(X_te)[:, 1]
        p_va = clf.predict_proba(X_va)[:, 1]
        oof[va] = p_va
        test_prob += p_te / N_SPLITS
        t_vec, t_fit, t_infer = t_vec + tv.seconds, t_fit + tf_.seconds, t_infer + ti.seconds
        feat_dims.append(X_tr.shape[1])
        n_params.append(clf.coef_.size + clf.intercept_.size)
        m = compute_metrics(y[va], p_va, 0.5)
        m.update(fold=k, n_train=int(tr.sum()), n_val=int(va.sum()), feature_dim=int(X_tr.shape[1]))
        fold_rows.append(m)
        print(f"fold {k}: acc {m['accuracy']:.4f}  macro-F1 {m['macro_f1']:.4f}  F1neg {m['f1_neg']:.4f}  AUC {m['roc_auc']:.4f}  dim {X_tr.shape[1]}")

    folds_df = pd.DataFrame(fold_rows)
    oof_05 = compute_metrics(y, oof, 0.5)
    thr, thr_table = choose_threshold(y, oof, acc_slack=args.acc_slack)
    oof_thr = compute_metrics(y, oof, thr)  # in-sample for the threshold (chosen on these same OOF probs)
    oof_thr["note"] = "in-sample: threshold selected on these same OOF probabilities; see oof_at_threshold_nested"
    oof_nested = nested_threshold_metrics(y, oof, df["fold"].to_numpy(), acc_slack=args.acc_slack)
    test_pred_pos_ratio = float((test_prob >= thr).mean())

    summary_stats = {}
    for key in ["accuracy", "macro_f1", "f1_neg", "precision_neg", "recall_neg", "roc_auc"]:
        mu, sd = mean_std(folds_df[key])
        summary_stats[key] = {"mean": mu, "std": sd}

    print("\n== 5-fold mean ± std (threshold 0.5) ==")
    for key, v in summary_stats.items():
        print(f"  {key:14s} {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"== OOF @0.5: acc {oof_05['accuracy']:.4f}  macro-F1 {oof_05['macro_f1']:.4f}  F1neg {oof_05['f1_neg']:.4f}  AUC {oof_05['roc_auc']:.4f}")
    print(f"== OOF @thr={thr:.2f}: acc {oof_thr['accuracy']:.4f}  macro-F1 {oof_thr['macro_f1']:.4f}  F1neg {oof_thr['f1_neg']:.4f}  recall_neg {oof_thr['recall_neg']:.4f}")
    print(f"   confusion @thr [[tn {oof_thr['tn']}, fp {oof_thr['fp']}], [fn {oof_thr['fn']}, tp {oof_thr['tp']}]]  (in-sample for the threshold)")
    print(f"== OOF nested-threshold (honest): acc {oof_nested['accuracy']:.4f}  macro-F1 {oof_nested['macro_f1']:.4f}  F1neg {oof_nested['f1_neg']:.4f}  fold thresholds {oof_nested['fold_thresholds']}")
    print(f"== test: predicted positive ratio @thr = {test_pred_pos_ratio:.4f} (train prior {y.mean():.4f})")

    run_id = f"{args.name}_seed{args.seed}"
    (RESULTS_DIR / "oof").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "test_probs").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "summary").mkdir(parents=True, exist_ok=True)
    np.save(RESULTS_DIR / "oof" / f"{run_id}.npy", oof)
    np.save(RESULTS_DIR / "test_probs" / f"{run_id}.npy", test_prob)
    thr_table.to_csv(RESULTS_DIR / "summary" / f"{run_id}_thresholds.csv", index=False)

    total_time = time.perf_counter() - t_start
    summary = {
        "run_id": run_id,
        "name": args.name,
        "model": "tfidf_lr",
        "feature_format": "tfidf",
        "config": vars(args),
        "protocol": {"dedup": True, "n_train": int(len(df)), "n_splits": N_SPLITS, "fold_seed": FOLD_SEED, "std_ddof": STD_DDOF, "threshold_rule": f"max macro-F1 s.t. acc >= best-acc - {args.acc_slack}; final threshold chosen on all OOF"},
        "folds": fold_rows,
        "cv_mean_std": summary_stats,
        "oof_at_0.5": oof_05,
        "threshold": thr,
        "oof_at_threshold": oof_thr,
        "oof_at_threshold_nested": oof_nested,
        "test_pred_pos_ratio": test_pred_pos_ratio,
        "resources": {
            "feature_dim_mean": float(np.mean(feat_dims)),
            "n_params": int(np.mean(n_params)),
            "preprocess_s": t_prep.seconds,
            "vectorize_s_total": t_vec,
            "train_s_total": t_fit,
            "infer_test_s_total": t_infer,
            "infer_test_s_per_model": t_infer / N_SPLITS,
            "wall_s": total_time,
            "peak_rss_mb": peak_rss_mb(),
            "machine": machine_tag(),
            "device": "cpu",
        },
        "artifacts": {
            "oof": f"results/oof/{run_id}.npy",
            "test_probs": f"results/test_probs/{run_id}.npy",
            "thresholds": f"results/summary/{run_id}_thresholds.csv",
        },
    }
    (RESULTS_DIR / "summary" / f"{run_id}.json").write_text(json.dumps(summary, indent=2))

    row = {
        "run_id": run_id, "model": "tfidf_lr", "feature_format": "tfidf", "seed": args.seed, "n_splits": N_SPLITS,
        "feature_dim": summary["resources"]["feature_dim_mean"], "n_params": summary["resources"]["n_params"],
        "train_s_total": round(t_vec + t_fit, 3), "infer_test_s": round(t_infer / N_SPLITS, 4), "peak_rss_mb": round(peak_rss_mb(), 1), "device": "cpu",
        "acc_mean": summary_stats["accuracy"]["mean"], "acc_std": summary_stats["accuracy"]["std"],
        "macro_f1_mean": summary_stats["macro_f1"]["mean"], "macro_f1_std": summary_stats["macro_f1"]["std"],
        "f1_neg_mean": summary_stats["f1_neg"]["mean"], "roc_auc_mean": summary_stats["roc_auc"]["mean"],
        "threshold": thr, "oof_acc_at_thr_insample": oof_thr["accuracy"], "oof_macro_f1_at_thr_insample": oof_thr["macro_f1"],
        "oof_acc_at_thr_nested": oof_nested["accuracy"], "oof_macro_f1_at_thr_nested": oof_nested["macro_f1"],
        "test_pos_ratio": test_pred_pos_ratio, "config": json.dumps({k: v for k, v in vars(args).items() if k != "name"}),
    }
    runs_path = RESULTS_DIR / "runs.csv"
    new = pd.DataFrame([row])
    if runs_path.exists():  # replace the row for this run_id instead of appending a duplicate
        old = pd.read_csv(runs_path)
        new = pd.concat([old[old["run_id"] != run_id], new], ignore_index=True)
    new.to_csv(runs_path, index=False)
    print(f"\nsaved: results/summary/{run_id}.json, results/oof/{run_id}.npy, results/test_probs/{run_id}.npy; updated results/runs.csv")
    return summary


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
