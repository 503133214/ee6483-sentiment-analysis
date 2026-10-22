"""Data loading, de-duplication and the shared 5-fold protocol.

Protocol (shared by EVERY model in this project):
  1. load data/raw/train.json
  2. drop exact-duplicate review texts, keep first      (7401 -> 7185 rows)
  3. StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  4. persist the fold assignment to data/folds.csv keyed by sha1(text),
     so every model uses identical folds regardless of row order.

The test set is only ever *loaded* here. Nothing is fitted on it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TRAIN_PATH = RAW_DIR / "train.json"
TEST_PATH = RAW_DIR / "test.json"
FOLDS_PATH = PROJECT_ROOT / "data" / "folds.csv"
TEMPLATE_PATH = PROJECT_ROOT / "submissions" / "submission_template.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

TEXT_COL = "reviews"
LABEL_COL = "sentiments"
N_SPLITS = 5
FOLD_SEED = 42
N_TRAIN_RAW = 7401
N_TEST = 1851


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def load_train() -> pd.DataFrame:
    """Raw training set, exactly as shipped (7401 rows, duplicates included)."""
    df = pd.read_json(TRAIN_PATH)
    assert list(df.columns) == [TEXT_COL, LABEL_COL], df.columns
    assert len(df) == N_TRAIN_RAW, len(df)
    assert df[TEXT_COL].map(lambda s: isinstance(s, str) and len(s) > 0).all()
    assert set(df[LABEL_COL].unique()) <= {0, 1}, df[LABEL_COL].unique()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    return df


def load_test() -> pd.DataFrame:
    """Unlabelled test set in its original order (1851 rows)."""
    df = pd.read_json(TEST_PATH)
    assert list(df.columns) == [TEXT_COL], df.columns
    assert len(df) == N_TEST, len(df)
    assert df[TEXT_COL].map(lambda s: isinstance(s, str) and len(s) > 0).all()
    return df


def dedup_train(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate texts (duplicates never disagree on the label)."""
    conflicts = df.groupby(TEXT_COL)[LABEL_COL].nunique()
    assert (conflicts <= 1).all(), "duplicate texts with conflicting labels"
    return df.drop_duplicates(subset=TEXT_COL, keep="first").reset_index(drop=True)


def duplicate_report(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Numbers that go into report section b)."""
    tr_counts = train[TEXT_COL].value_counts()
    te_counts = test[TEXT_COL].value_counts()
    overlap = set(tr_counts.index) & set(te_counts.index)
    ov_labels = train.drop_duplicates(TEXT_COL).set_index(TEXT_COL).loc[sorted(overlap), LABEL_COL]
    return {
        "train_rows": int(len(train)),
        "train_unique_texts": int(tr_counts.size),
        "train_redundant_rows": int(len(train) - tr_counts.size),
        "train_texts_duplicated": int((tr_counts > 1).sum()),
        "train_pos_ratio": float(train[LABEL_COL].mean()),
        "test_rows": int(len(test)),
        "test_unique_texts": int(te_counts.size),
        "test_redundant_rows": int(len(test) - te_counts.size),
        "overlap_unique_texts": int(len(overlap)),
        "overlap_test_rows": int(te_counts.loc[list(overlap)].sum()) if overlap else 0,
        "overlap_train_rows": int(tr_counts.loc[list(overlap)].sum()) if overlap else 0,
        "overlap_pos_by_text": int(ov_labels.sum()) if overlap else 0,
        "overlap_neg_by_text": int((ov_labels == 0).sum()) if overlap else 0,
    }


def make_folds(df_dedup: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = FOLD_SEED) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(df_dedup), -1, dtype=int)
    for k, (_, va_idx) in enumerate(skf.split(df_dedup[TEXT_COL], df_dedup[LABEL_COL])):
        fold[va_idx] = k
    assert (fold >= 0).all()
    return pd.DataFrame(
        {
            "text_hash": df_dedup[TEXT_COL].map(text_hash),
            LABEL_COL: df_dedup[LABEL_COL].values,
            "fold": fold,
        }
    )


def build_folds(force: bool = False) -> Path:
    """Create data/folds.csv once; later runs reuse the same file."""
    if FOLDS_PATH.exists() and not force:
        return FOLDS_PATH
    folds = make_folds(dedup_train(load_train()))
    FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_PATH, index=False)
    return FOLDS_PATH


def load_train_folds() -> pd.DataFrame:
    """De-duplicated training set with a `fold` column (the shared protocol)."""
    build_folds()
    df = dedup_train(load_train())
    df["text_hash"] = df[TEXT_COL].map(text_hash)
    folds = pd.read_csv(FOLDS_PATH, dtype={"text_hash": str, LABEL_COL: int, "fold": int})
    assert folds["text_hash"].is_unique
    merged = df.merge(folds, on="text_hash", how="left", suffixes=("", "_fold"))
    assert merged["fold"].notna().all(), "some training texts have no fold assignment; rebuild folds"
    assert (merged[LABEL_COL] == merged[f"{LABEL_COL}_fold"]).all(), "fold file labels disagree with data"
    merged = merged.drop(columns=[f"{LABEL_COL}_fold"])
    merged["fold"] = merged["fold"].astype(int)
    assert len(merged) == len(df) == len(folds)
    return merged


if __name__ == "__main__":
    train, test = load_train(), load_test()
    report = duplicate_report(train, test)
    path = build_folds()
    df = load_train_folds()
    report["dedup_rows"] = int(len(df))
    report["fold_sizes"] = df["fold"].value_counts().sort_index().tolist()
    report["fold_pos_ratio"] = df.groupby("fold")[LABEL_COL].mean().round(4).tolist()
    report["folds_path"] = str(path.relative_to(PROJECT_ROOT))
    print(json.dumps(report, indent=2))
