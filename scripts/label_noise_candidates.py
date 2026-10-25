"""Training reviews whose labels are contradicted by the models (label-noise candidates for report sections f and i).

    python -m scripts.label_noise_candidates
    python -m scripts.label_noise_candidates --llm qwen3.8_27b_zeroshot_4bit_seed42 --models tfidf_lr_neg_seed42 roberta_base_len256_seed42 qwen2.5_1.5b_lora16_len256_seed42

A review is a candidate when the zero-shot LLM (which never saw a label) assigns the opposite class
with probability >= --conf, and we record how many of the fine-tuned models (OOF probabilities, i.e.
models that did not train on that review) agree with the LLM rather than with the label.
Writes results/analysis/label_noise_candidates.csv and prints a summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import LABEL_COL, RESULTS_DIR, TEXT_COL, load_train_folds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", default="qwen3.8_27b_zeroshot_4bit_seed42")
    p.add_argument("--models", nargs="*", default=["tfidf_lr_neg_seed42", "roberta_base_len256_seed42", "qwen2.5_1.5b_lora16_len256_seed42"])
    p.add_argument("--conf", type=float, default=0.95)
    p.add_argument("--out", default=str(RESULTS_DIR / "analysis" / "label_noise_candidates.csv"))
    args = p.parse_args(argv)

    df = load_train_folds()
    y = df[LABEL_COL].to_numpy()
    llm = np.load(RESULTS_DIR / "oof" / f"{args.llm}.npy")
    probs = {m: np.load(RESULTS_DIR / "oof" / f"{m}.npy") for m in args.models}
    p_label = np.where(y == 1, llm, 1 - llm)                      # LLM probability of the given label
    cand = p_label <= 1 - args.conf
    out = df.loc[cand, [TEXT_COL, LABEL_COL, "fold"]].copy()
    out["p_pos_llm"] = llm[cand].round(4)
    agree = np.zeros(cand.sum(), dtype=int)
    for m, pr in probs.items():
        out[f"p_pos_{m}"] = pr[cand].round(4)
        agree += ((pr[cand] >= 0.5) == (llm[cand] >= 0.5)).astype(int)
    out["n_models_agree_with_llm"] = agree
    out = out.sort_values(["n_models_agree_with_llm", "p_pos_llm"], ascending=[False, True])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    n = cand.sum()
    print(f"{n} / {len(y)} training reviews ({n/len(y):.2%}) where the zero-shot LLM contradicts the label with confidence >= {args.conf}: "
          f"label 1 -> LLM negative: {(cand & (y == 1)).sum()}, label 0 -> LLM positive: {(cand & (y == 0)).sum()}")
    for m, pr in probs.items():
        print(f"  {m:40s} sides with the LLM on {((pr[cand] >= 0.5) == (llm[cand] >= 0.5)).mean():.0%} of them")
    print(f"  all {len(probs)} fine-tuned models side with the LLM on {(agree == len(probs)).sum()} reviews")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
