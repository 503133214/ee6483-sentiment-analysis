"""Executed on the Colab VM via `colab exec`: extract the uploaded project tarball and start the experiment chain in the background."""
import os, subprocess, sys, tarfile, time
ROOT = "/content/project"
os.makedirs(ROOT, exist_ok=True)
tar = "/content/project.tar.gz"
if os.path.exists(tar):
    with tarfile.open(tar) as t:
        t.extractall(ROOT)
    os.remove(tar)
    print("extracted project tarball")
os.makedirs(f"{ROOT}/logs", exist_ok=True)
log = open(f"{ROOT}/logs/chain.log", "ab")
chain = os.environ.get("CHAIN", "scripts/colab_chain.sh")  # pass another queue with: colab exec --env CHAIN=scripts/colab_chain_qwen3.sh
proc = subprocess.Popen(["bash", chain], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "PY": sys.executable, "PYTHONUNBUFFERED": "1"},
                        start_new_session=True)
open(f"{ROOT}/logs/chain.pid", "w").write(str(proc.pid))
time.sleep(3)
print("chain", chain, "started, pid", proc.pid, "alive:", proc.poll() is None)
