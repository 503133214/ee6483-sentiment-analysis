"""Turn saved test probabilities into submission.csv and validate it.

    python -m src.make_submission --summary results/summary/tfidf_lr_neg_seed42.json
    python -m src.make_submission --summary results/summary/a.json results/summary/b.json      # ensemble
    python -m src.make_submission --probs results/test_probs/a.npy --oof results/oof/a.npy

Several probability files are averaged (soft voting). The threshold is
  * --threshold if given, else
  * the summary's own OOF-chosen threshold when there is exactly one model, else
  * re-chosen on the AVERAGED OOF probabilities of the ensemble members
    (each --summary supplies its OOF file; --probs needs a matching --oof).
Nothing is ever chosen on the test set.

The file is written to a temporary path, validated with check_submission, and
only promoted to --out when every check passes, so a bad run never overwrites a
good submission.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.check_submission import PRED_COL, check_submission, print_report, read_submission
from src.data import LABEL_COL, N_TEST, PROJECT_ROOT, TEMPLATE_PATH, TEXT_COL, load_test, load_train_folds
from src.metrics import choose_threshold, compute_metrics

DEFAULT_OUT = PROJECT_ROOT / "submissions" / "submission.csv"


def rel(p: Path) -> str:
    return str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", nargs="*", default=None, help="run summary JSON(s); each supplies test_probs, oof and threshold")
    p.add_argument("--probs", nargs="*", default=None, help=".npy test-probability files to add to the average")
    p.add_argument("--oof", nargs="*", default=None, help="OOF .npy files matching --probs (same order)")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--acc-slack", type=float, default=0.003)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--compare", default=None, help="another submission.csv to report agreement with")
    args = p.parse_args(argv)

    prob_paths, oof_paths, summary_thresholds = [], [], []
    for s in args.summary or []:
        summary = json.loads(Path(s).read_text())
        prob_paths.append(PROJECT_ROOT / summary["artifacts"]["test_probs"])
        oof_paths.append(PROJECT_ROOT / summary["artifacts"]["oof"])
        summary_thresholds.append(float(summary["threshold"]))
    prob_paths += [Path(x) for x in args.probs or []]
    oof_paths += [Path(x) for x in args.oof or []]
    if not prob_paths:
        p.error("give --summary and/or --probs")
    if args.probs and args.oof and len(args.oof) != len(args.probs):
        p.error("--oof must list exactly one file per --probs file")

    probs = np.mean([np.load(x) for x in prob_paths], axis=0)
    assert probs.shape == (N_TEST,), probs.shape

    if args.threshold is not None:
        threshold, source = args.threshold, "given on the command line"
    elif len(prob_paths) == 1 and summary_thresholds:
        threshold, source = summary_thresholds[0], "from the summary (chosen on that model's OOF)"
    else:
        if len(oof_paths) != len(prob_paths):
            p.error("an ensemble needs --threshold, or one OOF file per probability file (--summary provides it; --probs needs --oof)")
        y = load_train_folds()[LABEL_COL].to_numpy()
        oof = np.mean([np.load(x) for x in oof_paths], axis=0)
        assert oof.shape == y.shape, (oof.shape, y.shape)
        threshold, _ = choose_threshold(y, oof, acc_slack=args.acc_slack)
        m = compute_metrics(y, oof, threshold)
        source = f"re-chosen on the averaged OOF of {len(oof_paths)} members (in-sample OOF acc {m['accuracy']:.4f}, macro-F1 {m['macro_f1']:.4f})"
    preds = (probs >= threshold).astype(int)

    test = load_test()
    template = read_submission(TEMPLATE_PATH)
    assert list(template.columns) == [TEXT_COL, PRED_COL]
    assert template[TEXT_COL].equals(test[TEXT_COL]), "template reviews differ from test.json"
    sub = pd.DataFrame({TEXT_COL: test[TEXT_COL].to_numpy(), PRED_COL: preds})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    sub.to_csv(tmp, index=False)
    info = check_submission(tmp, compare=args.compare)
    print(f"members: {[rel(x) for x in prob_paths]}")
    print(f"threshold: {threshold} ({source})")
    print_report(info)
    if info["ok"]:
        os.replace(tmp, out)
        print(f"wrote {rel(out)}")
        return 0
    tmp.unlink(missing_ok=True)
    print(f"left {rel(out)} unchanged")
    return 1


if __name__ == "__main__":
    sys.exit(main())
