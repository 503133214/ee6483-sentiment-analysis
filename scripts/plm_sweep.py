"""Sequential driver: distilbert learning-rate sweep -> class-weight ablation -> roberta-base 5-fold.

    python -m scripts.plm_sweep                       # all three stages (~1 h on Apple M5, MPS fp16)
    python -m scripts.plm_sweep --stages sweep,ablation
    python -m scripts.plm_sweep --stages roberta      # reuses the best lr recorded in results/summary/plm_sweep.json

Decisions and per-stage numbers are written to results/summary/plm_sweep.json (stages run later are
merged into the existing file). Each training run is a separate process (clean MPS memory) and the
driver continues past a failed run. The two-fold sweep runs are saved with --allow-partial, so their
run_ids carry the `_f0-1` suffix and can never replace a full 5-fold run.
Rule for transferring the learning rate: roberta-base uses min(best distilbert lr, 3e-5), because
5e-5 is known to be unstable for roberta-base fine-tuning (Liu et al. 2019 use 1e-5..3e-5).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs.csv"
OUT = ROOT / "results" / "summary" / "plm_sweep.json"
PY = sys.executable
RESUME: list[str] = []

SWEEP_LRS = [2e-5, 3e-5, 5e-5]
SWEEP_FOLDS = "0,1"
SWEEP_SUFFIX = "_f0-1"
ROBERTA_LR_CAP = 3e-5
DISTIL = ("--model", "distilbert-base-uncased", "--folds", SWEEP_FOLDS, "--epochs", "3", "--max-len", "128", "--batch-size", "32", "--allow-partial")


def run(name: str, *extra: str) -> tuple[bool, float]:
    cmd = [PY, "-u", "-m", "src.train_plm", "--name", name, *extra, *RESUME]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(cmd, cwd=ROOT)
    dt = time.perf_counter() - t0
    print(f"-> {name}: {'ok' if rc == 0 else f'FAILED rc={rc}'} in {dt/60:.1f} min", flush=True)
    return rc == 0, dt


def row(run_id: str) -> dict:
    if not RUNS.exists():
        return {}
    df = pd.read_csv(RUNS)
    r = df[df["run_id"] == run_id]
    return {k: (None if pd.isna(v) else v) for k, v in r.iloc[-1].to_dict().items()} if len(r) else {}


def pick(r: dict, *keys: str) -> dict:
    return {k: r.get(k) for k in keys}


METRIC_KEYS = ("acc_mean", "acc_std", "macro_f1_mean", "macro_f1_std", "f1_neg_mean", "roc_auc_mean",
               "acc_mean_last", "macro_f1_mean_last", "oof_macro_f1_at_thr_nested", "test_pos_ratio", "train_s_total")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", default="sweep,ablation,roberta", help="comma-separated subset of sweep,ablation,roberta")
    p.add_argument("--resume", action="store_true", help="pass --resume to train_plm so finished folds of an interrupted run are reused")
    args = p.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",")]
    if args.resume:
        RESUME.append("--resume")
    log = json.loads(OUT.read_text()) if OUT.exists() else {"stages": {}}
    log.setdefault("stages", {})
    log["last_started"] = time.strftime("%Y-%m-%d %H:%M:%S")

    if "sweep" in stages:
        runs = []
        for lr in SWEEP_LRS:
            name = f"distilbert_len128_lr{lr:g}"
            ok, dt = run(name, *DISTIL, "--lr", str(lr))
            rid = f"{name}_seed42{SWEEP_SUFFIX}"
            runs.append({"lr": lr, "run_id": rid, "ok": ok, "minutes": dt / 60, **(pick(row(rid), *METRIC_KEYS) if ok else {})})
        good = [s for s in runs if s["ok"] and s.get("macro_f1_mean") is not None]
        best = max(good, key=lambda s: (round(s["macro_f1_mean"], 4), s["acc_mean"])) if good else None
        log["stages"]["sweep"] = {"folds": SWEEP_FOLDS, "runs": runs, "best_lr": best["lr"] if best else None,
                                  "selection": "max mean macro-F1 @0.5 (best checkpoint) over folds 0,1; tie: accuracy"}
        OUT.write_text(json.dumps(log, indent=2))
        if best:
            print(f"\n### best distilbert lr = {best['lr']:g} (macro-F1 {best['macro_f1_mean']:.4f})", flush=True)

    best_lr = log["stages"].get("sweep", {}).get("best_lr")
    if "ablation" in stages:
        if best_lr is None:
            print("ablation skipped: no best lr recorded (run the sweep stage first)", flush=True)
        else:
            name = f"distilbert_len128_lr{best_lr:g}_cwnone"
            ok, dt = run(name, *DISTIL, "--lr", str(best_lr), "--class-weight", "none")
            rid = f"{name}_seed42{SWEEP_SUFFIX}"
            ref = row(f"distilbert_len128_lr{best_lr:g}_seed42{SWEEP_SUFFIX}")
            log["stages"]["ablation"] = {"run_id": rid, "ok": ok, "minutes": dt / 60, "class_weight": "none",
                                         **(pick(row(rid), *METRIC_KEYS) if ok else {}), "vs_sqrt": pick(ref, *METRIC_KEYS)}
            OUT.write_text(json.dumps(log, indent=2))

    if "roberta" in stages:
        if best_lr is None:
            print("roberta stage skipped: no best lr recorded (run the sweep stage first)", flush=True)
            return 1
        roberta_lr = min(best_lr, ROBERTA_LR_CAP)
        name = "roberta_base_len256"
        ok, dt = run(name, "--model", "roberta-base", "--lr", str(roberta_lr), "--epochs", "3", "--max-len", "256", "--batch-size", "16")
        rid = f"{name}_seed42"
        log["stages"]["roberta"] = {"run_id": rid, "ok": ok, "minutes": dt / 60, "lr": roberta_lr,
                                    "lr_rule": f"min(best distilbert lr, {ROBERTA_LR_CAP:g})", **(pick(row(rid), *METRIC_KEYS) if ok else {})}
        OUT.write_text(json.dumps(log, indent=2))
        if not ok:
            return 1
    log["last_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.write_text(json.dumps(log, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
