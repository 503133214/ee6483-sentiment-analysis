"""CPU-only checks for src/utils/results.py (shared stats + output writer).
    python -m tests.test_results
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import RESULTS_DIR
from src.metrics import compute_metrics
from src.utils.results import compute_run_stats, runs_row, write_run


def _synthetic(seed=0, n=1200):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.85).astype(int)
    oof = np.clip(y * 0.5 + rng.random(n) * 0.6, 0, 1)
    oof_last = np.clip(oof + rng.normal(0, 0.05, n), 0, 1)
    fold = np.repeat([0, 1], n // 2)
    rows, rows_last = [], []
    for k in (0, 1):
        m = fold == k
        rows.append({**compute_metrics(y[m], oof[m], 0.5), "fold": k})
        rows_last.append({**compute_metrics(y[m], oof_last[m], 0.5), "fold": k})
    return y, oof, oof_last, fold, rows, rows_last


def test_stats_and_row_schema():
    y, oof, oof_last, fold, rows, rows_last = _synthetic()
    st = compute_run_stats(y, oof, oof_last, fold, rows, rows_last, acc_slack=0.003)
    assert 0.05 <= st["threshold"] <= 0.95 and len(st["thr_table"]) == 91
    assert st["oof_at_threshold"]["note"].startswith("in-sample:") and st["oof_at_threshold_nested"]["note"].endswith("see *_last)")
    assert len(st["oof_at_threshold_nested"]["fold_thresholds"]) == 2
    assert abs(st["cv_mean_std"]["accuracy"]["mean"] - np.mean([r["accuracy"] for r in rows])) < 1e-12
    row = runs_row("x_seed1", "m", "word_vectors", 1, 2, True, 300, 10, 1.0, 0.1, 100.0, "cpu/fp32", None, "best_val_macro_f1", st, 0.85, {"lr": 1e-3})
    header = list(pd.read_csv(RESULTS_DIR / "runs.csv", nrows=0).columns)
    assert set(row) == set(header), (set(row) ^ set(header))


def test_write_run_replaces_by_run_id():
    y, oof, oof_last, fold, rows, rows_last = _synthetic()
    st = compute_run_stats(y, oof, oof_last, fold, rows, rows_last, acc_slack=0.003)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for i in range(2):
            row = runs_row("x_seed1", "m", "word_vectors", 1, 2, True, 300, 10, 1.0 + i, 0.1, 100.0, "cpu/fp32", None, "best_val_macro_f1", st, 0.85, {"i": i})
            write_run(d, "x_seed1", oof, oof_last, np.full(1851, 0.5), st, [{"fold": 0, "epoch": 1}], {"run_id": "x_seed1", "resources": {"n": np.int64(3)}}, row, lambda *a: None)
        df = pd.read_csv(d / "runs.csv")
        assert len(df) == 1 and df.loc[0, "train_s_total"] == 2.0
        summary = json.loads((d / "summary" / "x_seed1.json").read_text())
        assert summary["threshold"] == st["threshold"] and summary["artifacts"]["oof_last"].endswith("x_seed1_last.npy")
        assert np.load(d / "oof" / "x_seed1_last.npy").shape == oof_last.shape


if __name__ == "__main__":
    import inspect, sys
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1; print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
