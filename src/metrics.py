"""Metrics and threshold selection shared by all models.

Accuracy alone is misleading on this data (85.4% positive), so every run
reports macro-F1, negative-class F1 / precision / recall, ROC-AUC and the
confusion matrix as well.

Threshold rule: pick the OOF threshold that maximises macro-F1 among thresholds
whose accuracy is within `acc_slack` of the best accuracy. The final submission
threshold is chosen on ALL OOF probabilities (never on test). Because metrics
evaluated at that threshold on the same OOF vector are in-sample for the
threshold, `nested_threshold_metrics` also reports the honest estimate: choose
the threshold on 4 folds' OOF, score the 5th, pool.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

STD_DDOF = 0  # population std over the 5 folds; stated in README/report


def metrics_from_pred(y_true, pred, prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    prob = np.asarray(prob, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "f1_neg": float(f1_score(y_true, pred, pos_label=0)),
        "precision_neg": float(precision_score(y_true, pred, pos_label=0, zero_division=0)),
        "recall_neg": float(recall_score(y_true, pred, pos_label=0, zero_division=0)),
        "f1_pos": float(f1_score(y_true, pred, pos_label=1)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc_neg": float(average_precision_score(1 - y_true, 1 - prob)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "pred_pos_ratio": float(pred.mean()),
    }


def compute_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    prob = np.asarray(prob, dtype=float)
    out = {"threshold": float(threshold)}
    out.update(metrics_from_pred(y_true, prob >= threshold, prob))
    return out


def threshold_table(y_true, prob, grid=None) -> pd.DataFrame:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    return pd.DataFrame([compute_metrics(y_true, prob, t) for t in grid])


def choose_threshold(y_true, prob, acc_slack: float = 0.003) -> tuple[float, pd.DataFrame]:
    """Max macro-F1 subject to accuracy >= best accuracy - acc_slack (hedges
    against not knowing whether the grader scores accuracy or F1). Ties -> closest to 0.5."""
    table = threshold_table(y_true, prob)
    ok = table[table["accuracy"] >= table["accuracy"].max() - acc_slack].copy()
    ok = ok[ok["macro_f1"] >= ok["macro_f1"].max() - 1e-12]
    ok["dist"] = (ok["threshold"] - 0.5).abs()
    return float(ok.sort_values("dist").iloc[0]["threshold"]), table


def nested_threshold_metrics(y_true, prob, fold, acc_slack: float = 0.003) -> dict:
    """Honest estimate of the threshold rule: for each fold k choose the threshold
    on the OTHER folds' OOF probabilities, apply it to fold k, pool the predictions."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    fold = np.asarray(fold)
    pred = np.zeros(len(y_true), dtype=int)
    fold_thresholds = []
    for k in sorted(np.unique(fold)):
        va = fold == k
        t_k, _ = choose_threshold(y_true[~va], prob[~va], acc_slack=acc_slack)
        fold_thresholds.append(t_k)
        pred[va] = (prob[va] >= t_k).astype(int)
    out = {"fold_thresholds": fold_thresholds, "note": "threshold chosen on the other folds' OOF, scored on the held-out fold, pooled"}
    out.update(metrics_from_pred(y_true, pred, prob))
    return out


def mean_std(values) -> tuple[float, float]:
    """Mean and population std (ddof=STD_DDOF) over folds."""
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=STD_DDOF))
