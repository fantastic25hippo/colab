import concurrent.futures, json, os, shutil, subprocess, sys, threading, time

hb_stop = threading.Event()
def hb():
    while not hb_stop.wait(15): print(json.dumps({"stage": "hb", "t": time.time()}), flush=True)
threading.Thread(target=hb, daemon=True).start()

def log(m):
    msg = f"[COLD_DRIVER {time.strftime('%H:%M:%S')}] {m}"
    print(msg, flush=True)
    try:
        with open("/content/build.log", "a") as f: f.write(msg + "\n")
    except Exception: pass

t_global0 = float(os.environ.get("T0_GLOBAL", time.time()))
log(f"Cold-start timer started at t0={t_global0:.2f}")

if os.path.exists("/content/hf_token"):
    for d in ["/root/.cache/huggingface", "/root/.config/huggingface"]:
        os.makedirs(d, exist_ok=True); shutil.copyfile("/content/hf_token", os.path.join(d, "token"))
    os.environ["HF_TOKEN"] = open("/content/hf_token").read().strip()
    try: os.remove("/content/hf_token")
    except Exception: pass

if not shutil.which("zstd"):
    log("Installing zstd...")
    r = subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"], capture_output=True)
    if r.returncode != 0:
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"], check=True)

log("Step 1: Remounting /dev/shm with exec...")
t_remount0 = time.time()
r_remount = subprocess.run(["mount", "-o", "remount,exec", "/dev/shm"], capture_output=True, text=True)
t_remount = time.time() - t_remount0
log(f"Remount finished in {t_remount:.2f}s (rc={r_remount.returncode})")

log("Step 2: Starting parallel download of venv and Qwen3-4B weights...")
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.makedirs("/dev/shm/qwen3", exist_ok=True)

def dl_and_unpack_venv():
    t0 = time.time()
    p = hf_hub_download(repo_id="avgrocketman/tpu-compile-cache", filename="venv312_cpu.tar.zst", local_dir="/content")
    t_dl = time.time() - t0
    t_u0 = time.time()
    subprocess.run(["tar", "-I", "zstd", "-xf", p, "-C", "/dev/shm"], check=True)
    t_unpack = time.time() - t_u0
    return {"dl_s": round(t_dl, 2), "unpack_s": round(t_unpack, 2), "total_s": round(time.time() - t0, 2)}

def dl_weights():
    t0 = time.time()
    snapshot_download("Qwen/Qwen3-4B", local_dir="/dev/shm/qwen3", max_workers=12)
    return {"dl_s": round(time.time() - t0, 2)}

t_prep0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    f_venv = pool.submit(dl_and_unpack_venv)
    f_wt = pool.submit(dl_weights)
    venv_res = f_venv.result(); wt_res = f_wt.result()
t_prep = time.time() - t_prep0
log(f"Parallel prep finished in {t_prep:.2f}s (venv: {venv_res}, weights: {wt_res})")

py_bin = "/dev/shm/vllm_env/bin/python"
log("Step 3: TPU Check A (no extra env vars)...")
t_a0 = time.time()
r_a = subprocess.run([py_bin, "-c", "import vllm, jax; print('CHECK_A_OUT:', jax.devices())"], capture_output=True, text=True)
t_check_a = time.time() - t_a0
out_a = r_a.stdout.strip()
log(f"Check A completed in {t_check_a:.2f}s: {out_a}")

log("Step 4: TPU Check B (with topology & skip MDS env vars)...")
env_b = os.environ.copy()
env_b.update({"TPU_SKIP_MDS_QUERY": "1", "TPU_TOPOLOGY": "1x1", "TPU_HOST_BOUNDS": "1,1,1",
              "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1", "TPU_WORKER_ID": "0", "TPU_WORKER_HOSTNAMES": "localhost",
              "JAX_PLATFORMS": "tpu,cpu"})
t_b0 = time.time()
r_b = subprocess.run([py_bin, "-c", "import vllm, jax; print('CHECK_B_OUT:', jax.devices())"], env=env_b, capture_output=True, text=True)
t_check_b = time.time() - t_b0
out_b = r_b.stdout.strip()
b_has_tpu = "TpuDevice" in out_b
log(f"Check B completed in {t_check_b:.2f}s (has_tpu={b_has_tpu}): {out_b}")

log("Step 5-7: Launching engine subrunner...")
cmd_eng = [py_bin, "-u", "/content/tpu_engine_subrunner.py"]
use_env = env_b if b_has_tpu else os.environ.copy()
proc = subprocess.Popen(cmd_eng, env=use_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout: sys.stdout.write(line); sys.stdout.flush()
proc.wait()
if proc.returncode != 0: raise RuntimeError(f"Engine subrunner failed with rc={proc.returncode}")

eng_data = {}
if os.path.exists("/content/engine_results.json"):
    with open("/content/engine_results.json") as f: eng_data = json.load(f)

t_first_tok_from_start = eng_data.get("first_token_ts", time.time()) - t_global0
log(f"Total Cold-Start Time to First Token: {t_first_tok_from_start:.2f}s")

log("Step 8: Packing and uploading new compile cache (tpu_compile_cache_b1_32k.tar)...")
cache_tar = "/content/tpu_compile_cache_b1_32k.tar"
subprocess.run(["tar", "-cf", cache_tar, "-C", "/dev/shm", "tpu_compile_cache"], check=True)
tar_mb = round(os.path.getsize(cache_tar) / (1024 * 1024), 2)
api = HfApi(); repo = "avgrocketman/tpu-compile-cache"
api.upload_file(path_or_fileobj=cache_tar, path_in_repo="tpu_compile_cache_b1_32k.tar", repo_id=repo)
log(f"Uploaded {cache_tar} ({tar_mb} MB)")

v_data = {}
try:
    p = hf_hub_download(repo_id=repo, filename="versions.json", local_dir="/content")
    with open(p) as f: v_data = json.load(f)
except Exception: pass
v_data["compile_cache_b1_32k"] = {
    "archive": "tpu_compile_cache_b1_32k.tar", "size_mb": tar_mb,
    "compile_count": eng_data.get("compile_count", 0), "model": "Qwen/Qwen3-4B",
    "max_model_len": 32768, "max_num_seqs": 1, "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
}
with open("/content/versions.json", "w") as f: json.dump(v_data, f, indent=2)
api.upload_file(path_or_fileobj="/content/versions.json", path_in_repo="versions.json", repo_id=repo)
log("Updated versions.json on Hugging Face.")

summary = {
    "t_remount_s": round(t_remount, 2), "venv_prep": venv_res, "weights_dl_s": wt_res["dl_s"],
    "parallel_prep_s": round(t_prep, 2), "check_a_s": round(t_check_a, 2), "check_b_s": round(t_check_b, 2),
    "check_b_used": b_has_tpu, "total_cold_start_to_first_token_s": round(t_first_tok_from_start, 2),
    "engine": eng_data, "cache_tar_mb": tar_mb
}
print("=== COLDSTART_RESULTS_JSONL ===", flush=True)
print(json.dumps(summary), flush=True)
print("=== END_COLDSTART_RESULTS ===", flush=True)
hb_stop.set()
