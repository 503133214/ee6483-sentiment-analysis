"""Fetch finished runs from the Colab VM into local results/ and merge their runs.csv rows.

    python -m scripts.colab_fetch --session ee6483 --run-ids deberta_v3_base_len256_seed42 qwen2.5_0.5b_lora16_len256_seed42
    python -m scripts.colab_fetch --session ee6483 --all-new        # every run_id in the VM's runs.csv that is not in the local one

Downloads results/oof/<id>.npy (+_last), results/test_probs/<id>.npy, results/summary/<id>.json
(+_thresholds.csv, +_curves.csv when present) via `colab download`, then replaces/adds the rows
in local results/runs.csv by run_id (local rows for other run_ids are untouched).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/content/project"


def download(session: str, remote: str, local: Path) -> bool:
    local.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["colab", "download", "-s", session, remote, str(local)], capture_output=True, text=True)
    ok = r.returncode == 0 and local.exists()
    print(("  ok   " if ok else "  FAIL ") + remote + ("" if ok else f"  ({(r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else 'no output'})"))
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session", "-s", default="ee6483")
    p.add_argument("--run-ids", nargs="*", default=[])
    p.add_argument("--all-new", action="store_true")
    args = p.parse_args(argv)

    with tempfile.TemporaryDirectory() as d:
        remote_runs = Path(d) / "runs.csv"
        if not download(args.session, f"{REMOTE}/results/runs.csv", remote_runs):
            return 1
        remote = pd.read_csv(remote_runs)
    local_path = ROOT / "results" / "runs.csv"
    local = pd.read_csv(local_path) if local_path.exists() else pd.DataFrame(columns=remote.columns)
    ids = list(args.run_ids)
    if args.all_new:
        ids += [r for r in remote["run_id"] if r not in set(local["run_id"]) and r not in ids]
    if not ids:
        print("nothing to fetch"); return 0
    missing = [r for r in ids if r not in set(remote["run_id"])]
    if missing:
        print("not in the VM's runs.csv (skipped):", missing)
        ids = [r for r in ids if r not in missing]
    fetched = []
    for rid in ids:
        print(f"== {rid}")
        files = [f"oof/{rid}.npy", f"oof/{rid}_last.npy", f"test_probs/{rid}.npy", f"summary/{rid}.json", f"summary/{rid}_thresholds.csv", f"summary/{rid}_curves.csv"]
        ok = [download(args.session, f"{REMOTE}/results/{f}", ROOT / "results" / f) for f in files]
        if all(ok[:4]):  # curves/thresholds are optional (train_frozen writes no curves)
            fetched.append(rid)
        else:
            print(f"  -> {rid}: core files missing, row NOT merged")
    if fetched:
        rows = remote[remote["run_id"].isin(fetched)]
        merged = pd.concat([local[~local["run_id"].isin(fetched)], rows], ignore_index=True)
        merged.to_csv(local_path, index=False)
        print(f"merged {len(fetched)} row(s) into results/runs.csv ({len(merged)} rows now): {fetched}")
    return 0 if len(fetched) == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
