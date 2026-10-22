"""Validate a submission file against the course template and test.json.

    python -m src.check_submission submissions/submission.csv [--compare other.csv]

Checks: exact header, 1851 data rows (NOT `wc -l`: reviews contain newlines),
reviews identical to test.json row by row, predictions are integer 0/1,
positive ratio in a sane range, duplicated test texts get identical predictions,
and (optionally) agreement with another submission. Unreadable or unparsable
files are reported as problems rather than tracebacks.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.data import N_TEST, TEXT_COL, load_test

PRED_COL = "predictions"
EXPECTED_HEADER = f"{TEXT_COL},{PRED_COL}"
PARSE_ERRORS = (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError, FileNotFoundError)


def read_submission(path: Path) -> pd.DataFrame:
    # dtype=str + keep_default_na=False: never let pandas turn a review like "NA" into NaN
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def check_submission(path: str | Path, compare: str | Path | None = None, pos_range=(0.78, 0.92)) -> dict:
    path = Path(path)
    problems, info = [], {}

    def fail(msg):
        problems.append(msg)
        info["problems"], info["ok"] = problems, False
        return info

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            header = f.readline().rstrip("\r\n")
        df = read_submission(path)
    except PARSE_ERRORS as e:
        return fail(f"cannot read {path} as UTF-8 comma-separated CSV: {type(e).__name__}: {e}")
    if header != EXPECTED_HEADER:
        problems.append(f"header is {header!r}, expected {EXPECTED_HEADER!r} (no index column, exact names)")
    info["rows"] = len(df)
    if list(df.columns) != [TEXT_COL, PRED_COL]:
        return fail(f"columns are {list(df.columns)}, expected ['{TEXT_COL}', '{PRED_COL}']")
    if len(df) != N_TEST:
        problems.append(f"{len(df)} rows, expected {N_TEST}")
    test = load_test()
    if len(df) == N_TEST and not df[TEXT_COL].equals(test[TEXT_COL]):
        n_bad = int((df[TEXT_COL].to_numpy() != test[TEXT_COL].to_numpy()).sum())
        problems.append(f"reviews column differs from test.json in {n_bad} rows (order or text changed)")
    bad = ~df[PRED_COL].isin(["0", "1"])
    if bad.any():
        problems.append(f"{int(bad.sum())} predictions are not literal 0/1 (e.g. {df.loc[bad, PRED_COL].head(3).tolist()})")
        info["problems"], info["ok"] = problems, False
        return info
    preds = df[PRED_COL].astype(int)
    info["pos_ratio"] = float(preds.mean())
    info["n_pos"], info["n_neg"] = int(preds.sum()), int((preds == 0).sum())
    if not (pos_range[0] <= info["pos_ratio"] <= pos_range[1]):
        problems.append(f"positive ratio {info['pos_ratio']:.3f} outside {pos_range} (train prior 0.854)")
    incons = df.assign(p=preds).groupby(TEXT_COL)["p"].nunique()
    info["inconsistent_duplicate_texts"] = int((incons > 1).sum())
    if info["inconsistent_duplicate_texts"]:
        problems.append("identical test texts received different predictions")
    if compare is not None:
        try:
            other = read_submission(Path(compare))
        except PARSE_ERRORS as e:
            problems.append(f"cannot read compare file {compare}: {type(e).__name__}: {e}")
        else:
            if PRED_COL not in other or len(other) != len(df):
                problems.append("compare file has a different number of rows or no predictions column")
            elif not other[PRED_COL].isin(["0", "1"]).all():
                problems.append("compare file predictions are not literal 0/1")
            else:
                info["agreement_with_compare"] = float((other[PRED_COL].astype(int).to_numpy() == preds.to_numpy()).mean())
    info["problems"], info["ok"] = problems, not problems
    return info


def print_report(info: dict) -> None:
    for k, v in info.items():
        if k != "problems":
            print(f"{k}: {v}")
    for prob in info["problems"]:
        print("PROBLEM:", prob)
    print("RESULT:", "OK" if info["ok"] else "FAILED")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path")
    p.add_argument("--compare", default=None, help="another submission to report agreement with")
    args = p.parse_args(argv)
    info = check_submission(args.path, args.compare)
    print_report(info)
    return 0 if info["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
