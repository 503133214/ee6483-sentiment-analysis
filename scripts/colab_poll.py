"""Executed on the Colab VM via `colab exec`: report chain status, the current job's last lines, and finished rows."""
import os, subprocess
ROOT = "/content/project"
pid = open(f"{ROOT}/logs/chain.pid").read().strip() if os.path.exists(f"{ROOT}/logs/chain.pid") else None
alive = pid and os.path.exists(f"/proc/{pid}")
print(f"chain pid {pid} alive={bool(alive)}")
chain = open(f"{ROOT}/logs/chain.log").read().splitlines() if os.path.exists(f"{ROOT}/logs/chain.log") else []
print("\n".join(l for l in chain if l.startswith(("$ ", "-> ", "CHAIN"))))
cur = [l for l in chain if l.startswith("$ ")]
if cur and alive:
    name = cur[-1].split("-m ")[1].split()[0]  # module
    jobs = [l.split()[0] for l in chain if l.startswith("-> ")]
    logs = sorted((os.path.getmtime(f"{ROOT}/logs/{f}"), f) for f in os.listdir(f"{ROOT}/logs") if f.endswith(".log") and f != "chain.log")
    if logs:
        latest = logs[-1][1]
        lines = open(f"{ROOT}/logs/{latest}", errors="replace").read().splitlines()
        keep = [l for l in lines if l.startswith(("model ", "frozen ", "tokenised", "fold ", "== ", "features", "saved", "Traceback", "RuntimeError", "Error"))]
        print(f"--- {latest}: {len(lines)} lines; key lines ---")
        print("\n".join(keep[-12:]))
        print("--- tail ---"); print("\n".join(lines[-3:]))
print(subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
if os.path.exists(f"{ROOT}/results/runs.csv"):
    import pandas as pd
    df = pd.read_csv(f"{ROOT}/results/runs.csv")
    new = df[~df.run_id.str.startswith(("tfidf", "glove", "rand", "distilbert")) & (df.run_id != "roberta_base_len256_seed42")]
    if len(new):
        print("--- new rows ---"); print(new[["run_id", "n_splits", "acc_mean", "macro_f1_mean", "f1_neg_mean", "roc_auc_mean", "train_s_total", "device_mem_mb"]].round(4).to_string(index=False))
