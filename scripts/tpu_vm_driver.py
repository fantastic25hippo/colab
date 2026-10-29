import json, os, shutil, subprocess, sys, threading, time

hb_stop = threading.Event()
def hb():
    while not hb_stop.wait(20):
        print(json.dumps({"stage": "hb", "time": time.time()}), flush=True)
threading.Thread(target=hb, daemon=True).start()

def log(m): print(f"[VM1 {time.strftime('%H:%M:%S')}] {m}", flush=True)

if os.path.exists("/content/hf_token"):
    for d in ["/root/.cache/huggingface", "/root/.config/huggingface"]:
        os.makedirs(d, exist_ok=True)
        shutil.copyfile("/content/hf_token", os.path.join(d, "token"))
    os.environ["HF_TOKEN"] = open("/content/hf_token").read().strip()
    try: os.remove("/content/hf_token")
    except Exception: pass

t0 = time.time()
log("Bootstrapping Python 3.12 venv and packages...")
uv = shutil.which("uv") or "/root/.local/bin/uv"
if not os.path.exists(uv):
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    uv = shutil.which("uv") or "/root/.local/bin/uv"
subprocess.run([uv, "venv", "--python", "3.12", "/content/venv312"], check=True)
subprocess.run([uv, "pip", "install", "--python", "/content/venv312/bin/python",
                "vllm-tpu==0.32.0", "tpu-inference==0.32.0", "hf-transfer", "huggingface_hub"], check=True)
t_install = time.time() - t0
log(f"INSTALL_SECONDS: {t_install:.2f}")

log("Downloading Qwen/Qwen3-4B to /dev/shm/qwen3...")
dl_code = '''
import os, time
from huggingface_hub import snapshot_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
t0 = time.time()
snapshot_download("Qwen/Qwen3-4B", local_dir="/dev/shm/qwen3", max_workers=12)
print(f"DL_SECONDS: {time.time()-t0:.2f}", flush=True)
'''
r_dl = subprocess.run(["/content/venv312/bin/python", "-c", dl_code], capture_output=True, text=True, check=True)
dl_line = [l for l in r_dl.stdout.splitlines() if "DL_SECONDS:" in l]
t_dl = float(dl_line[0].split(":")[1].strip()) if dl_line else 10.0
log(f"DOWNLOAD_SECONDS: {t_dl:.2f}")

os.makedirs("/content/hooks", exist_ok=True)
shutil.copyfile("/content/tpu_bench_hook.py", "/content/hooks/sitecustomize.py")
run_p = "/content/tpu_run_single.py"

configs = [
    ("M1-base", ["--load-fix", "0", "--skip-precompile", "0"]),
    ("M1-trick-skip", ["--load-fix", "1", "--skip-precompile", "1", "--cache-dir", "/content/tpu_compile_cache"]),
    ("M1-trick-sizes", ["--load-fix", "1", "--skip-precompile", "0", "--compilation-sizes", "[16, 512]", "--cache-dir", "/content/tpu_compile_cache"]),
    ("M1-trick-both", ["--load-fix", "1", "--skip-precompile", "1", "--compilation-sizes", "[16, 512]", "--cache-dir", "/content/tpu_compile_cache"]),
    ("M1-warm-best", ["--load-fix", "1", "--skip-precompile", "0", "--compilation-sizes", "[16, 512]", "--cache-dir", "/content/tpu_compile_cache"]),
]

results = []
for name, extra in configs:
    log(f"Starting run {name}...")
    out_f = f"/content/res_{name}.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "/content/hooks"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    cmd = ["/content/venv312/bin/python", run_p, "--model", "/dev/shm/qwen3", "--mode", name, "--out", out_f] + extra
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log(f"{name} exited with {p.returncode}")
    if os.path.exists(out_f):
        try:
            with open(out_f) as f: rec = json.load(f)
            rec["install_s"] = round(t_install, 2)
            rec["download_s"] = round(t_dl, 2)
            results.append(rec)
        except Exception as e:
            results.append({"run_name": name, "status": "failed", "error": str(e), "stderr": p.stderr})
    else:
        results.append({"run_name": name, "status": "failed", "error": "No output json", "stderr": p.stderr})

log("Packaging HF artifacts...")
try:
    subprocess.run(["tar", "-cf", "/content/venv312.tar", "-C", "/content", "venv312"], check=True)
    if os.path.exists("/content/tpu_compile_cache"):
        subprocess.run(["tar", "-cf", "/content/tpu_compile_cache.tar", "-C", "/content", "tpu_compile_cache"], check=True)
    try:
        os.makedirs("/content/wheels", exist_ok=True)
        subprocess.run(["/content/venv312/bin/python", "-m", "pip", "download", "-d", "/content/wheels",
                        "vllm-tpu==0.32.0", "tpu-inference==0.32.0", "hf-transfer", "huggingface_hub"], check=False)
        subprocess.run(["tar", "-czf", "/content/wheels.tar.gz", "-C", "/content", "wheels"], check=False)
    except Exception: pass
    v_data = {"vllm-tpu": "0.32.0", "tpu-inference": "0.32.0", "jax": "0.11.2", "jaxlib": "0.11.2",
              "libtpu": "0.0.48", "torch": "2.10.0", "model": "Qwen/Qwen3-4B", "model_len": 4096}
    with open("/content/versions.json", "w") as f: json.dump(v_data, f)

    up_script = '''
import os
from huggingface_hub import HfApi
api = HfApi()
for fn in ["venv312.tar", "tpu_compile_cache.tar", "wheels.tar.gz", "versions.json"]:
    p = f"/content/{fn}"
    if os.path.exists(p):
        print(f"Uploading {fn} to avgrocketman/tpu-compile-cache...", flush=True)
        api.upload_file(path_or_fileobj=p, path_in_repo=fn, repo_id="avgrocketman/tpu-compile-cache", repo_type="model")
print("HF_UPLOAD_COMPLETE", flush=True)
'''
    subprocess.run(["/content/venv312/bin/python", "-c", up_script], check=True)
    log("HF artifacts uploaded successfully.")
except Exception as e:
    log(f"Warning: artifact creation/upload error: {e}")

with open("/content/results.jsonl", "w") as f:
    for r in results: f.write(json.dumps(r) + "\n")

print("=== BENCH_RESULTS_JSONL ===", flush=True)
for r in results: print(json.dumps(r), flush=True)
print("=== END_BENCH_RESULTS ===", flush=True)

hb_stop.set()
