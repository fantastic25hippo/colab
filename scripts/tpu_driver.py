import json, os, shutil, subprocess, sys, threading, time

hb_stop = threading.Event()
def hb():
    while not hb_stop.wait(20): print(json.dumps({"stage": "hb", "t": time.time()}), flush=True)
threading.Thread(target=hb, daemon=True).start()

def log(m): print(f"[TPU_DRIVER {time.strftime('%H:%M:%S')}] {m}", flush=True)

if os.path.exists("/content/hf_token"):
    for d in ["/root/.cache/huggingface", "/root/.config/huggingface"]:
        os.makedirs(d, exist_ok=True); shutil.copyfile("/content/hf_token", os.path.join(d, "token"))
    os.environ["HF_TOKEN"] = open("/content/hf_token").read().strip()
    try: os.remove("/content/hf_token")
    except Exception: pass

t_cold0 = time.time()
dl_s, inst_s, cache_s = [0.0], [0.0], [0.0]

def task_dl():
    t0 = time.time()
    code = 'import os; from huggingface_hub import snapshot_download; os.environ["HF_HUB_ENABLE_HF_TRANSFER"]="1"; snapshot_download("Qwen/Qwen3-4B", local_dir="/dev/shm/qwen3", max_workers=12)'
    subprocess.run([sys.executable, "-c", code], check=True)
    dl_s[0] = time.time() - t0; log(f"Download done in {dl_s[0]:.2f}s")

def task_cache():
    t0 = time.time()
    code = 'from huggingface_hub import hf_hub_download; import subprocess; p=hf_hub_download(repo_id="avgrocketman/tpu-compile-cache", filename="tpu_compile_cache.tar", local_dir="/content"); subprocess.run(["tar", "-xf", p, "-C", "/content"], check=True)'
    try: subprocess.run([sys.executable, "-c", code], check=True)
    except Exception as e: log(f"Cache restore err: {e}")
    cache_s[0] = time.time() - t0; log(f"Cache restore done in {cache_s[0]:.2f}s")

def task_install():
    t0 = time.time()
    uv = shutil.which("uv") or "/root/.local/bin/uv"
    if not os.path.exists(uv): subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True); uv = shutil.which("uv") or "/root/.local/bin/uv"
    subprocess.run([uv, "venv", "--python", "3.12", "/content/venv312"], check=True)
    wh_ok = False
    try:
        code_wh = 'from huggingface_hub import hf_hub_download; import subprocess; p=hf_hub_download(repo_id="avgrocketman/tpu-compile-cache", filename="wheels.tar.gz", local_dir="/content"); subprocess.run(["tar", "-xzf", p, "-C", "/content"], check=True)'
        subprocess.run([sys.executable, "-c", code_wh], check=True)
        r = subprocess.run([uv, "pip", "install", "--python", "/content/venv312/bin/python", "--no-index", "--find-links", "/content/wheels",
                            "vllm-tpu==0.32.0", "tpu-inference==0.32.0", "hf-transfer", "huggingface_hub"], capture_output=True, text=True)
        wh_ok = (r.returncode == 0)
    except Exception: pass
    if not wh_ok:
        log("Wheel install failed or missing, falling back to online uv pip install...")
        subprocess.run([uv, "pip", "install", "--python", "/content/venv312/bin/python",
                        "vllm-tpu==0.32.0", "tpu-inference==0.32.0", "hf-transfer", "huggingface_hub"], check=True)
    inst_s[0] = time.time() - t0; log(f"Install done in {inst_s[0]:.2f}s (wheels_used={wh_ok})")

th_dl, th_c, th_i = threading.Thread(target=task_dl), threading.Thread(target=task_cache), threading.Thread(target=task_install)
th_dl.start(); th_c.start(); th_i.start()
th_dl.join(); th_c.join(); th_i.join()
t_parallel = time.time() - t_cold0

os.makedirs("/content/hooks", exist_ok=True)
shutil.copyfile("/content/tpu_bench_hook.py", "/content/hooks/sitecustomize.py")
env = os.environ.copy()
env["PYTHONPATH"] = "/content/hooks"
env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
env["BENCH_LOAD_FIX"] = "1"
env["JAX_COMPILATION_CACHE_DIR"] = "/content/tpu_compile_cache"
env["VLLM_XLA_CACHE_PATH"] = "/content/tpu_compile_cache"
env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
env["JAX_LOG_COMPILES"] = "1"
env["JAX_EXPLAIN_CACHE_MISSES"] = "1"

log("Launching tpu_engine_runner.py...")
run_cmd = ["/content/venv312/bin/python", "/content/tpu_engine_runner.py", "--model", "/dev/shm/qwen3", "--out", "/content/tpu_engine_results.json"]
p = subprocess.Popen(run_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
with open("/content/driver.log", "w") as lf:
    for line in p.stdout:
        sys.stdout.write(line); sys.stdout.flush(); lf.write(line); lf.flush()
p.wait()
log(f"Runner process exited with {p.returncode}")

eng_res = {}
if os.path.exists("/content/tpu_engine_results.json"):
    with open("/content/tpu_engine_results.json") as f: eng_res = json.load(f)

# Cache upload check
new_compiled = eng_res.get("engine_startup", {}).get("cache_compiled", 0)
if new_compiled > 0 and os.path.exists("/content/tpu_compile_cache"):
    log(f"New programs compiled ({new_compiled}). Packaging and uploading cache...")
    try:
        subprocess.run(["tar", "-cf", "/content/tpu_compile_cache.tar", "-C", "/content", "tpu_compile_cache"], check=True)
        v_data = {"vllm-tpu": "0.32.0", "tpu-inference": "0.32.0", "jax": "0.11.2", "jaxlib": "0.11.2",
                  "libtpu": "0.0.48", "torch": "2.10.0", "model": "Qwen/Qwen3-4B", "max_model_len": 17408,
                  "cache_compiled": new_compiled, "cache_hits": eng_res.get("engine_startup", {}).get("cache_hits", 0)}
        with open("/content/versions.json", "w") as f: json.dump(v_data, f)
        up_code = 'from huggingface_hub import HfApi; api=HfApi(); api.upload_file(path_or_fileobj="/content/tpu_compile_cache.tar", path_in_repo="tpu_compile_cache.tar", repo_id="avgrocketman/tpu-compile-cache"); api.upload_file(path_or_fileobj="/content/versions.json", path_in_repo="versions.json", repo_id="avgrocketman/tpu-compile-cache")'
        subprocess.run([sys.executable, "-c", up_code], check=True)
        log("Updated cache uploaded to HF.")
    except Exception as e: log(f"Cache upload err: {e}")

final_rec = {
    "run_name": "tpu_characterization_final",
    "cold_start_breakdown": {
        "download_s": round(dl_s[0], 2), "cache_restore_s": round(cache_s[0], 2),
        "install_s": round(inst_s[0], 2), "parallel_prep_s": round(t_parallel, 2),
        "weights_loaded_s": eng_res.get("engine_startup", {}).get("load_s", 0),
        "compile_s": eng_res.get("engine_startup", {}).get("compile_s", 0),
        "engine_ready_s": eng_res.get("engine_startup", {}).get("ready_s", 0),
        "first_token_from_ready_s": eng_res.get("engine_startup", {}).get("first_token_from_ready_s", 0),
        "total_fresh_vm_to_first_token_s": round(t_parallel + eng_res.get("engine_startup", {}).get("ready_s", 0) + (eng_res.get("engine_startup", {}).get("first_token_from_ready_s", 0) or 0), 2),
        "cache_compiled": eng_res.get("engine_startup", {}).get("cache_compiled", 0),
        "cache_hits": eng_res.get("engine_startup", {}).get("cache_hits", 0)
    },
    "step2_decode": eng_res.get("step2_decode", {}),
    "step3_batching": eng_res.get("step3_batching", {}),
    "step4_context": eng_res.get("step4_context", {}),
    "hbm_snapshots": eng_res.get("hbm_snapshots", {})
}

with open("/content/results.jsonl", "w") as f: f.write(json.dumps(final_rec) + "\n")
print("=== BENCH_RESULTS_JSONL ===", flush=True)
print(json.dumps(final_rec), flush=True)
print("=== END_BENCH_RESULTS ===", flush=True)
hb_stop.set()
