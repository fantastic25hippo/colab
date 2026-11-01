import base64, concurrent.futures, json, os, shutil, subprocess, sys, threading, time

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

t_vm0 = time.time()
t_global0 = float(os.environ.get("T0_GLOBAL", t_vm0))
log(f"In-VM cold-start timer started at t_vm0={t_vm0:.2f} (t_global0={t_global0:.2f})")

if os.environ.get("SUBRUNNER_B64"):
    with open("/content/tpu_engine_subrunner.py", "wb") as f:
        f.write(base64.b64decode(os.environ["SUBRUNNER_B64"]))
if os.environ.get("PATCHES_B64"):
    with open("/content/qwen_patches.py", "wb") as f:
        f.write(base64.b64decode(os.environ["PATCHES_B64"]))
if os.environ.get("CACHE_NAMING_B64"):
    with open("/content/cache_naming.py", "wb") as f:
        f.write(base64.b64decode(os.environ["CACHE_NAMING_B64"]))

sys.path.insert(0, "/content")
try:
    from cache_naming import build_cache_name
except ImportError:
    _d = os.path.dirname(os.path.abspath(__file__))
    if _d not in sys.path: sys.path.insert(0, _d)
    from cache_naming import build_cache_name

if os.environ.get("HF_TOKEN"):
    for d in ["/root/.cache/huggingface", "/root/.config/huggingface"]:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "token"), "w") as f: f.write(os.environ["HF_TOKEN"])

# Block C++ libtpu.so GCE metadata HTTP 404 retries via /etc/hosts + closed loopback port
try:
    with open("/etc/hosts", "a") as f:
        f.write("\n127.0.0.1 metadata.google.internal metadata\n")
except Exception: pass

log("Step 1: Remounting /dev/shm with exec...")
t_remount0 = time.time()
r_remount = subprocess.run(["mount", "-o", "remount,exec", "/dev/shm"], capture_output=True, text=True)
t_remount = time.time() - t_remount0
log(f"Remount finished in {t_remount:.2f}s (rc={r_remount.returncode})")

from huggingface_hub import hf_hub_download, snapshot_download
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.makedirs("/dev/shm/qwen3", exist_ok=True)
os.makedirs("/dev/shm/tpu_compile_cache", exist_ok=True)
if os.path.exists("/dev/shm/.weights_ready"):
    os.remove("/dev/shm/.weights_ready")

def dl_venv_and_cache():
    t0 = time.time()
    patch_bytes = open("/content/qwen_patches.py", "rb").read() if os.path.exists("/content/qwen_patches.py") else b""
    expected_cache = build_cache_name(
        model_slug="qwen3-4b",
        vllm_version="0.32.0",
        max_model_len=32768,
        buckets="16-32768",
        llm_args={"gpu_memory_utilization": 0.90, "max_model_len": 32768, "max_num_seqs": 1, "trust_remote_code": True},
        patch_content=patch_bytes,
    )

    def _dl_cache():
        try:
            return hf_hub_download(repo_id="avgrocketman/tpu-compile-cache",
                                   filename=expected_cache, local_dir="/dev/shm")
        except Exception:
            log(f"cache miss: {expected_cache}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p_dl:
        f_v = p_dl.submit(hf_hub_download, repo_id="avgrocketman/tpu-compile-cache",
                          filename="venv312_cpu.tar.zst", local_dir="/dev/shm")
        f_c = p_dl.submit(_dl_cache)
        f_z = p_dl.submit(hf_hub_download, repo_id="avgrocketman/tpu-compile-cache",
                          filename="zstd_bin", local_dir="/dev/shm")
        p_venv, p_cache, p_zstd = f_v.result(), f_c.result(), f_z.result()
    os.chmod(p_zstd, 0o755)
    t_dl = time.time() - t0
    c_status = f"hit ({expected_cache})" if p_cache else f"miss ({expected_cache})"
    log(f"Downloaded venv + cache {c_status} + zstd_bin to /dev/shm in {t_dl:.2f}s (zero apt-get)")
    return p_venv, p_cache, p_zstd, round(t_dl, 2)

def unpack_venv_and_cache(p_venv, p_cache, p_zstd):
    t_u0 = time.time()
    if p_cache and os.path.exists(p_cache):
        subprocess.run(["tar", "-xf", p_cache, "-C", "/dev/shm"], check=True)
        try: os.remove(p_cache)
        except Exception: pass
    else:
        shutil.rmtree("/dev/shm/tpu_compile_cache", ignore_errors=True)
        os.makedirs("/dev/shm/tpu_compile_cache", exist_ok=True)
    subprocess.run(["sh", "-c", f"{p_zstd} -d -T0 -c {p_venv} | tar -xf - -C /dev/shm"], check=True)
    try: os.remove(p_venv)
    except Exception: pass
    t_unpack = time.time() - t_u0
    log(f"Unpacked venv and cache to /dev/shm in {t_unpack:.2f}s")
    return round(t_unpack, 2)

def dl_weights():
    t0 = time.time()
    snapshot_download("Qwen/Qwen3-4B", local_dir="/dev/shm/qwen3", max_workers=12)
    with open("/dev/shm/.weights_ready", "w") as f:
        f.write("1\n")
    dur = time.time() - t0
    log(f"Weights download finished in {dur:.2f}s")
    return {"dl_s": round(dur, 2)}

log("Step 2a: Downloading venv + 32K compile cache + zstd_bin directly to /dev/shm...")
t_prep0 = time.time()
p_venv, p_cache, p_zstd, t_venv_dl = dl_venv_and_cache()

log("Step 2b: Unpacking venv + cache into /dev/shm (uncontested tmpfs I/O)...")
t_venv_unpack = unpack_venv_and_cache(p_venv, p_cache, p_zstd)
venv_res = {"dl_s": t_venv_dl, "unpack_s": t_venv_unpack, "zstd_overlapped_s": 0.0,
            "total_s": round(time.time() - t_prep0, 2)}
t_venv_ready = time.time() - t_prep0
log(f"Venv + cache ready at +{t_venv_ready:.2f}s; starting Qwen3-4B download IN PARALLEL with subrunner imports...")

pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
f_wt = pool.submit(dl_weights)

py_bin = "/dev/shm/vllm_env/bin/python"
env_b = os.environ.copy()
env_b.pop("SUBRUNNER_B64", None)
env_b.update({
    "TPU_SKIP_MDS_QUERY": "1", "TPU_TOPOLOGY": "1x1", "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1", "TPU_WORKER_ID": "0", "TPU_WORKER_HOSTNAMES": "localhost",
    "GCE_METADATA_HOST": "127.0.0.1:1", "GCE_METADATA_IP": "127.0.0.1:1",
    "GCE_METADATA_ROOT": "127.0.0.1:1", "NO_GCE_CHECK": "true",
    "JAX_PLATFORMS": "tpu,cpu",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "VLLM_LOGGING_LEVEL": "INFO"
})

proc = subprocess.Popen([py_bin, "-u", "/content/tpu_engine_subrunner.py"],
                        env=env_b, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout:
    sys.stdout.write(line); sys.stdout.flush()
proc.wait()

wt_res = f_wt.result()
pool.shutdown(wait=True)

if proc.returncode != 0:
    raise RuntimeError(f"Engine subrunner failed with rc={proc.returncode}")

eng_data = {}
if os.path.exists("/content/engine_results.json"):
    with open("/content/engine_results.json") as f: eng_data = json.load(f)

first_tok_ts = eng_data.get("first_token_ts", time.time())
t_in_vm_first_tok = first_tok_ts - t_vm0
t_global_first_tok = first_tok_ts - t_global0
log(f"In-VM Cold-Start to First Token: {t_in_vm_first_tok:.2f}s (Global from provisioning: {t_global_first_tok:.2f}s)")

summary = {
    "t_remount_s": round(t_remount, 2),
    "venv_and_cache_prep": venv_res,
    "weights_dl_s": wt_res["dl_s"],
    "in_vm_cold_start_to_first_token_s": round(t_in_vm_first_tok, 2),
    "total_cold_start_to_first_token_s": round(t_global_first_tok, 2),
    "engine": eng_data
}
print("=== COLDSTART_RESULTS_JSONL ===", flush=True)
print(json.dumps(summary), flush=True)
print("=== END_COLDSTART_RESULTS ===", flush=True)
hb_stop.set()
