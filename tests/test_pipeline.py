"""Quick regression checks. Run from the project root:
    python -m tests.test_pipeline        (plain python)   or   python -m pytest tests
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.check_submission import check_submission
from src.data import FOLDS_PATH, LABEL_COL, N_SPLITS, TEMPLATE_PATH, TEXT_COL, load_test, load_train_folds
from src.features.negation import mark_negation
from src.metrics import choose_threshold, compute_metrics, nested_threshold_metrics


def test_negation_marking():
    assert mark_negation("not good") == "not NOT_good"
    assert mark_negation("I do not like it, but ok") == "I do not NOT_like NOT_it , but ok"
    assert mark_negation("didnt like it") == "didnt NOT_like NOT_it"
    assert mark_negation("It doesn't fit") == "It doesn't NOT_fit"
    assert mark_negation("It doesn’t fit") == "It doesn’t NOT_fit"
    assert mark_negation("not here\nGreat price") == "not NOT_here Great price"          # newline ends scope
    assert mark_negation('not "old ladyish"') == 'not " NOT_old NOT_ladyish "'          # quotes do not end scope
    assert mark_negation("not yet--hopefully fine") == "not NOT_yet ; hopefully fine"   # -- ends scope
    assert mark_negation("not yet - hopefully fine") == "not NOT_yet ; hopefully fine"  # spaced dash ends scope
    assert mark_negation("not long-johns") == "not NOT_long NOT_- NOT_johns" or "NOT_long" in mark_negation("not long-johns")
    assert mark_negation("Not charged 8.99 for it") == "Not NOT_charged NOT_8.99 NOT_for NOT_it"
    assert mark_negation("the 8th pair") == "the 8th pair"
    assert mark_negation("café is not naïve") == "café is not NOT_naïve"
    assert mark_negation("") == ""
    assert mark_negation("!!!") == "! ! !"
    assert "NOT_want" not in mark_negation("I want it")  # 'want' is not a negator


def test_folds_protocol():
    df = load_train_folds()
    assert len(df) == 7185 and df[TEXT_COL].is_unique
    assert sorted(df["fold"].unique()) == list(range(N_SPLITS))
    sizes = df["fold"].value_counts()
    assert sizes.max() - sizes.min() <= 1
    ratios = df.groupby("fold")[LABEL_COL].mean()
    assert ratios.max() - ratios.min() < 0.01
    folds = pd.read_csv(FOLDS_PATH)
    assert len(folds) == 7185 and folds["text_hash"].is_unique


def test_metrics_and_threshold():
    rng = np.random.default_rng(0)
    y = (rng.random(2000) < 0.85).astype(int)
    prob = np.clip(y * 0.6 + rng.random(2000) * 0.5, 0, 1)
    m = compute_metrics(y, prob, 0.5)
    assert 0 <= m["macro_f1"] <= 1 and m["tn"] + m["fp"] + m["fn"] + m["tp"] == 2000
    thr, table = choose_threshold(y, prob)
    assert 0.05 <= thr <= 0.95 and len(table) == 91
    fold = rng.integers(0, 5, 2000)
    nested = nested_threshold_metrics(y, prob, fold)
    assert len(nested["fold_thresholds"]) == 5 and 0 <= nested["macro_f1"] <= 1


def _valid_submission(tmp: Path) -> Path:
    test = load_test()
    codes = pd.factorize(test[TEXT_COL])[0]
    preds = (codes % 7 != 0).astype(int)  # ~0.857 positive, identical texts get identical predictions
    out = tmp / "ok.csv"
    pd.DataFrame({TEXT_COL: test[TEXT_COL], "predictions": preds}).to_csv(out, index=False)
    return out


def test_check_submission_accepts_valid_and_rejects_broken():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        ok = _valid_submission(tmp)
        assert check_submission(ok)["ok"], check_submission(ok)["problems"]
        assert check_submission(ok, compare=ok)["agreement_with_compare"] == 1.0
        df = pd.read_csv(ok, dtype=str, keep_default_na=False)

        broken = {}
        broken["index_col"] = df.to_csv(index=True)
        broken["shuffled"] = df.sample(frac=1, random_state=1).to_csv(index=False)
        broken["float_preds"] = df.assign(predictions=df["predictions"].astype(float)).to_csv(index=False)
        broken["bool_preds"] = df.assign(predictions=df["predictions"].astype(int).astype(bool)).to_csv(index=False)
        edited = df.copy(); edited.loc[0, "reviews"] = edited.loc[0, "reviews"] + " x"
        broken["edited_review"] = edited.to_csv(index=False)
        broken["short"] = df.iloc[:-1].to_csv(index=False)
        broken["all_zero"] = df.assign(predictions="0").to_csv(index=False)
        broken["wrong_header"] = df.rename(columns={"predictions": "prediction"}).to_csv(index=False)
        broken["tab_sep"] = df.to_csv(index=False, sep="\t")
        broken["empty"] = ""
        for name, text in broken.items():
            path = tmp / f"{name}.csv"
            path.write_text(text, encoding="utf-8")
            info = check_submission(path)
            assert not info["ok"], f"{name} should fail"
        (tmp / "latin1.csv").write_bytes(ok.read_text(encoding="utf-8").encode("utf-8")[:5000] + "é".encode("latin-1"))
        assert not check_submission(tmp / "latin1.csv")["ok"]
        # bad compare file is reported, not raised
        info = check_submission(ok, compare=tmp / "float_preds.csv")
        assert not info["ok"] and "compare" in " ".join(info["problems"])
        # template itself: right shape, but all-zero predictions are flagged
        info = check_submission(TEMPLATE_PATH)
        assert info["rows"] == 1851 and not info["ok"]


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
