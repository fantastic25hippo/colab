import base64, concurrent.futures, json, os, shutil, subprocess, sys, threading, time

hb_stop = threading.Event()
def hb():
    while not hb_stop.wait(15): print(json.dumps({"stage": "hb", "t": time.time()}), flush=True)
threading.Thread(target=hb, daemon=True).start()

def log(m):
    msg = f"[LFM25_DRIVER {time.strftime('%H:%M:%S')}] {m}"
    print(msg, flush=True)
    try:
        with open("/content/build.log", "a") as f: f.write(msg + "\n")
    except Exception: pass

t_vm0 = time.time()
t_global0 = float(os.environ.get("T0_GLOBAL", t_vm0))
log(f"In-VM 200K cold-start timer started at t_vm0={t_vm0:.2f} (t_global0={t_global0:.2f})")

# Unpack subrunner, patches, and pallas kernel into /content/
for env_k, target_f in [
    ("SUBRUNNER_B64", "/content/lfm25_engine_subrunner.py"),
    ("PATCHES_B64", "/content/tpu_v5e_patches.py"),
    ("PALLAS_B64", "/content/k8v4_pallas_kernel.py"),
]:
    if os.environ.get(env_k):
        with open(target_f, "wb") as f: f.write(base64.b64decode(os.environ[env_k]))

if os.environ.get("HF_TOKEN"):
    for d in ["/root/.cache/huggingface", "/root/.config/huggingface"]:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "token"), "w") as f: f.write(os.environ["HF_TOKEN"])

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
os.makedirs("/dev/shm/lfm25_200k", exist_ok=True)
os.makedirs("/dev/shm/tpu_compile_cache", exist_ok=True)
if os.path.exists("/dev/shm/.weights_ready"): os.remove("/dev/shm/.weights_ready")

def dl_venv_and_cache():
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as p_dl:
        f_v = p_dl.submit(hf_hub_download, repo_id="avgrocketman/tpu-compile-cache",
                          filename="venv312_cpu.tar.zst", local_dir="/dev/shm")
        f_c = p_dl.submit(hf_hub_download, repo_id="avgrocketman/tpu-compile-cache",
                          filename="tpu_compile_cache_b1_32k.tar", local_dir="/dev/shm")
        f_z = p_dl.submit(hf_hub_download, repo_id="avgrocketman/tpu-compile-cache",
                          filename="zstd_bin", local_dir="/dev/shm")
        p_venv, p_cache, p_zstd = f_v.result(), f_c.result(), f_z.result()
    os.chmod(p_zstd, 0o755)
    t_dl = time.time() - t0
    log(f"Downloaded venv + compile cache in {t_dl:.2f}s")
    return p_venv, p_cache, p_zstd, round(t_dl, 2)

def unpack_venv_and_cache(p_venv, p_cache, p_zstd):
    t_u0 = time.time()
    subprocess.run(["tar", "-xf", p_cache, "-C", "/dev/shm"], check=True)
    try: os.remove(p_cache)
    except Exception: pass
    subprocess.run(["sh", "-c", f"{p_zstd} -d -T0 -c {p_venv} | tar -xf - -C /dev/shm"], check=True)
    try: os.remove(p_venv)
    except Exception: pass
    t_unpack = time.time() - t_u0
    log(f"Unpacked venv + cache in {t_unpack:.2f}s")
    return round(t_unpack, 2)

def dl_weights():
    t0 = time.time()
    model_id = os.environ.get("LFM25_MODEL_ID", "avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU")
    snapshot_download(model_id, local_dir="/dev/shm/lfm25_200k", max_workers=12)
    cfg_p = "/dev/shm/lfm25_200k/config.json"
    if os.path.exists(cfg_p):
        with open(cfg_p) as f: c = json.load(f)
        if "layer_types" not in c or not c["layer_types"]:
            c["layer_types"] = ["full_attention" if i in (2, 6, 10, 14, 18, 22, 26, 30, 34, 38) else "conv" for i in range(40)]
            with open(cfg_p, "w") as f: json.dump(c, f, indent=2)
    with open("/dev/shm/.weights_ready", "w") as f: f.write("1\n")
    dur = time.time() - t0
    log(f"Weights ready in {dur:.2f}s")
    return {"dl_s": round(dur, 2)}

log("Step 2: Preparing venv and downloading weights in parallel...")
t_prep0 = time.time()
p_v, p_c, p_z, t_v_dl = dl_venv_and_cache()
t_v_unp = unpack_venv_and_cache(p_v, p_c, p_z)
venv_res = {"dl_s": t_v_dl, "unpack_s": t_v_unp, "total_s": round(time.time() - t_prep0, 2)}

pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
f_wt = pool.submit(dl_weights)

py_bin = "/dev/shm/vllm_env/bin/python"
env_b = os.environ.copy()
for k in ("SUBRUNNER_B64", "PATCHES_B64", "PALLAS_B64"): env_b.pop(k, None)
env_b["PYTHONPATH"] = f"/content:{env_b.get('PYTHONPATH', '')}"
env_b.update({
    "MODEL_IMPL_TYPE": "vllm", "TPU_SKIP_MDS_QUERY": "1", "TPU_TOPOLOGY": "1x1",
    "TPU_HOST_BOUNDS": "1,1,1", "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1", "TPU_WORKER_ID": "0",
    "TPU_WORKER_HOSTNAMES": "localhost", "GCE_METADATA_HOST": "127.0.0.1:1",
    "GCE_METADATA_IP": "127.0.0.1:1", "GCE_METADATA_ROOT": "127.0.0.1:1", "NO_GCE_CHECK": "true",
    "JAX_PLATFORMS": "tpu,cpu", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "VLLM_LOGGING_LEVEL": "INFO"
})

proc = subprocess.Popen([py_bin, "-u", "/content/lfm25_engine_subrunner.py"],
                        env=env_b, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout: sys.stdout.write(line); sys.stdout.flush()
proc.wait()

wt_res = f_wt.result()
pool.shutdown(wait=True)
if proc.returncode != 0: raise RuntimeError(f"LFM2.5 subrunner failed rc={proc.returncode}")

eng_data = {}
if os.path.exists("/content/engine_results.json"):
    with open("/content/engine_results.json") as f: eng_data = json.load(f)

first_tok_ts = eng_data.get("first_token_ts", time.time())
summary = {
    "t_remount_s": round(t_remount, 2), "venv_and_cache_prep": venv_res,
    "weights_dl_s": wt_res["dl_s"],
    "in_vm_cold_start_to_first_token_s": round(first_tok_ts - t_vm0, 2),
    "total_cold_start_to_first_token_s": round(first_tok_ts - t_global0, 2),
    "engine": eng_data
}
print("=== COLDSTART_RESULTS_JSONL ===", flush=True)
print(json.dumps(summary), flush=True)
print("=== END_COLDSTART_RESULTS ===", flush=True)
hb_stop.set()
