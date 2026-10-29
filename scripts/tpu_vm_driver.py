import glob, json, os, shutil, subprocess, sys, threading, time

hb_stop = threading.Event()
def hb():
    while not hb_stop.wait(20):
        print(json.dumps({"stage": "hb", "time": time.time()}), flush=True)
threading.Thread(target=hb, daemon=True).start()

def log(msg): print(f"[VM_DRIVER {time.strftime('%H:%M:%S')}] {msg}", flush=True)

t_inst0 = time.time()
log("Bootstrapping Python 3.12 venv and packages...")
uv = shutil.which("uv")
if not uv:
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    uv = shutil.which("uv") or "/root/.local/bin/uv"
subprocess.run([uv, "venv", "--python", "3.12", "/content/venv312"], check=True)
subprocess.run([uv, "pip", "install", "--python", "/content/venv312/bin/python",
                "vllm-tpu==0.32.0", "tpu-inference==0.32.0", "hf-transfer", "huggingface_hub"], check=True)
t_install = time.time() - t_inst0
log(f"INSTALL_SECONDS: {t_install:.2f}")

log("Downloading Qwen/Qwen3-4B to /dev/shm/qwen3...")
dl_script = '''
import os, time
from huggingface_hub import snapshot_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
t0 = time.time()
snapshot_download("Qwen/Qwen3-4B", local_dir="/dev/shm/qwen3", max_workers=12)
print(f"DL_SECONDS: {time.time()-t0:.2f}", flush=True)
'''
res_dl = subprocess.run(["/content/venv312/bin/python", "-c", dl_script], capture_output=True, text=True, check=True)
log(f"Download output: {res_dl.stdout.strip()}")

# Ensure hook is installed as sitecustomize
os.makedirs("/content/hooks", exist_ok=True)
for hp in ["/content/tpu_bench_hook.py", "/content/scripts/tpu_bench_hook.py", "scripts/tpu_bench_hook.py"]:
    if os.path.exists(hp):
        shutil.copyfile(hp, "/content/hooks/sitecustomize.py")
        break

run_single_p = "/content/tpu_run_single.py"
for sp in ["/content/tpu_run_single.py", "/content/scripts/tpu_run_single.py", "scripts/tpu_run_single.py"]:
    if os.path.exists(sp):
        run_single_p = sp
        break

results = []
# M1 runs
for mode in ["base", "opt", "warm"]:
    log(f"Starting M1 ({mode})...")
    out_f = f"/content/res_m1_{mode}.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "/content/hooks"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    p = subprocess.run(["/content/venv312/bin/python", run_single_p,
                        "--model", "/dev/shm/qwen3", "--mode", mode, "--out", out_f],
                       env=env, capture_output=True, text=True)
    log(f"M1-{mode} returncode: {p.returncode}")
    if os.path.exists(out_f):
        try:
            with open(out_f) as f: rec = json.load(f)
            rec["install_seconds"] = round(t_install, 2)
            results.append(rec)
            log(f"M1-{mode} result loaded.")
        except Exception as e:
            results.append({"model": "Qwen/Qwen3-4B", "mode": mode, "status": "failed", "error": str(e), "stdout": p.stdout, "stderr": p.stderr})
    else:
        results.append({"model": "Qwen/Qwen3-4B", "mode": mode, "status": "failed", "error": "No output json", "stdout": p.stdout, "stderr": p.stderr})

# M2 skipped analysis
m2_reason = "xbill9 patches from jev-tpu-v5e1 do not match tpu-inference 0.32.0 (hunks failed in wna16.py [5/7 failed], linear.py [1/2 failed], compressed_tensors.py [1/5 failed], fused_moe_gmm.py [1/4 failed], textonly.diff [1/1 failed]). Skipped without patching."
for mode in ["base", "opt", "warm"]:
    results.append({"model": "google/gemma-4-E2B-it-qat-w4a16-ct", "mode": mode, "status": "skipped", "reason": m2_reason, "install_seconds": round(t_install, 2)})

with open("/content/results.jsonl", "w") as f:
    for r in results: f.write(json.dumps(r) + "\n")

print("=== BENCH_RESULTS_JSONL ===", flush=True)
for r in results:
    print(json.dumps(r), flush=True)
print("=== END_BENCH_RESULTS ===", flush=True)

hb_stop.set()
