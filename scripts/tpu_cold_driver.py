import json, os, shutil, subprocess, sys, threading, time

hb_stop = threading.Event()
def hb():
    while not hb_stop.wait(20):
        print(json.dumps({"stage": "hb", "time": time.time()}), flush=True)
threading.Thread(target=hb, daemon=True).start()

def log(m): print(f"[VM2 {time.strftime('%H:%M:%S')}] {m}", flush=True)

if os.path.exists("/content/hf_token"):
    for d in ["/root/.cache/huggingface", "/root/.config/huggingface"]:
        os.makedirs(d, exist_ok=True)
        shutil.copyfile("/content/hf_token", os.path.join(d, "token"))
    os.environ["HF_TOKEN"] = open("/content/hf_token").read().strip()
    try: os.remove("/content/hf_token")
    except Exception: pass

t_cold0 = time.time()
log("Starting parallel model download & environment/cache restore...")

dl_time = [0.0]
restore_time = [0.0]

def task_dl():
    t0 = time.time()
    code = '''
import os
from huggingface_hub import snapshot_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
snapshot_download("Qwen/Qwen3-4B", local_dir="/dev/shm/qwen3", max_workers=12)
'''
    subprocess.run([sys.executable, "-c", code], check=True)
    dl_time[0] = time.time() - t0
    log(f"Model download finished in {dl_time[0]:.2f}s")

def task_restore():
    t0 = time.time()
    code = '''
import os, subprocess
from huggingface_hub import hf_hub_download
for fn in ["venv312.tar", "tpu_compile_cache.tar"]:
    try:
        p = hf_hub_download(repo_id="avgrocketman/tpu-compile-cache", filename=fn, local_dir="/content")
        subprocess.run(["tar", "-xf", p, "-C", "/content"], check=True)
        print(f"Restored {fn}", flush=True)
    except Exception as e:
        print(f"Error restoring {fn}: {e}", flush=True)
'''
    subprocess.run([sys.executable, "-c", code], check=True)
    if not os.path.exists("/content/venv312/bin/python"):
        subprocess.run(["uv", "venv", "--python", "3.12", "/content/venv312"], check=True)
        subprocess.run(["uv", "pip", "install", "--python", "/content/venv312/bin/python",
                        "vllm-tpu==0.32.0", "tpu-inference==0.32.0", "hf-transfer", "huggingface_hub"], check=True)
    restore_time[0] = time.time() - t0
    log(f"Environment & cache restore finished in {restore_time[0]:.2f}s")

th_dl = threading.Thread(target=task_dl)
th_res = threading.Thread(target=task_restore)
th_dl.start(); th_res.start()
th_dl.join(); th_res.join()

t_parallel = time.time() - t_cold0
log(f"Parallel restore & download completed in {t_parallel:.2f}s (restore={restore_time[0]:.2f}s, dl={dl_time[0]:.2f}s)")

os.makedirs("/content/hooks", exist_ok=True)
shutil.copyfile("/content/tpu_bench_hook.py", "/content/hooks/sitecustomize.py")
run_p = "/content/tpu_run_single.py"

out_f = "/content/res_M1_cold.json"
env = os.environ.copy()
env["PYTHONPATH"] = "/content/hooks"
env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
cmd = ["/content/venv312/bin/python", run_p, "--model", "/dev/shm/qwen3", "--mode", "M1-accelerated-cold",
       "--load-fix", "1", "--skip-precompile", "0", "--compilation-sizes", "[16, 512]",
       "--cache-dir", "/content/tpu_compile_cache", "--out", out_f]

log("Running M1-accelerated-cold...")
p = subprocess.run(cmd, env=env, capture_output=True, text=True)
log(f"Process exited with {p.returncode}")

rec = {}
if os.path.exists(out_f):
    with open(out_f) as f: rec = json.load(f)
    rec["install_s"] = round(restore_time[0], 2)
    rec["download_s"] = round(dl_time[0], 2)
    rec["parallel_prep_s"] = round(t_parallel, 2)
    rec["total_cold_startup_s"] = round(t_parallel + rec.get("phase_times", {}).get("total_startup_s", 0), 2)
else:
    rec = {"run_name": "M1-accelerated-cold", "status": "failed", "error": "No output json", "stderr": p.stderr}

with open("/content/cold_result.json", "w") as f: json.dump(rec, f)

print("=== COLD_BENCH_RESULTS_JSONL ===", flush=True)
print(json.dumps(rec), flush=True)
print("=== END_COLD_RESULTS ===", flush=True)

hb_stop.set()
