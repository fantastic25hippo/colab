import json, os, shutil, subprocess, sys, threading, time

hb_stop = threading.Event()
def hb():
    while not hb_stop.wait(20): print(json.dumps({"stage": "hb", "t": time.time()}), flush=True)
threading.Thread(target=hb, daemon=True).start()

def log(m): print(f"[WHEEL_BUILD {time.strftime('%H:%M:%S')}] {m}", flush=True)

if os.path.exists("/content/hf_token"):
    for d in ["/root/.cache/huggingface", "/root/.config/huggingface"]:
        os.makedirs(d, exist_ok=True); shutil.copyfile("/content/hf_token", os.path.join(d, "token"))
    os.environ["HF_TOKEN"] = open("/content/hf_token").read().strip()
    try: os.remove("/content/hf_token")
    except Exception: pass

uv = shutil.which("uv") or "/root/.local/bin/uv"
if not os.path.exists(uv):
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    uv = shutil.which("uv") or "/root/.local/bin/uv"

log("Creating build virtual environment...")
subprocess.run([uv, "venv", "--python", "3.12", "/content/venv_build"], check=True)

log("Installing vllm-tpu, tpu-inference, and dependencies online...")
t0 = time.time()
subprocess.run([uv, "pip", "install", "--python", "/content/venv_build/bin/python",
                "vllm-tpu==0.32.0", "tpu-inference==0.32.0", "hf-transfer", "huggingface_hub", "pip"], check=True)
t_online = time.time() - t0
log(f"Online install completed in {t_online:.2f}s")

log("Freezing full transitive closure to /content/requirements.lock...")
with open("/content/requirements.lock", "w") as f:
    subprocess.run([uv, "pip", "freeze", "--python", "/content/venv_build/bin/python"], stdout=f, check=True)

log("Downloading all wheels to /content/wheels...")
os.makedirs("/content/wheels", exist_ok=True)
t_dl0 = time.time()
subprocess.run(["/content/venv_build/bin/python", "-m", "pip", "download",
                "-r", "/content/requirements.lock", "--no-deps", "-d", "/content/wheels"], check=True)
t_dl = time.time() - t_dl0
whl_files = [f for f in os.listdir("/content/wheels") if f.endswith(".whl") or f.endswith(".tar.gz")]
log(f"Downloaded {len(whl_files)} wheels in {t_dl:.2f}s")

log("Verifying offline install into fresh /content/venv_verify...")
subprocess.run([uv, "venv", "--python", "3.12", "/content/venv_verify"], check=True)
t_off0 = time.time()
subprocess.run([uv, "pip", "install", "--python", "/content/venv_verify/bin/python",
                "--no-index", "--find-links", "/content/wheels",
                "vllm-tpu==0.32.0", "tpu-inference==0.32.0", "hf-transfer", "huggingface_hub"], check=True)
t_offline = time.time() - t_off0
log(f"Offline verify install completed in {t_offline:.2f}s")

log("Verifying imports in isolated environment...")
subprocess.run(["/content/venv_verify/bin/python", "-c",
                "import vllm, tpu_inference, jax; print('VERIFY_IMPORT_SUCCESS', vllm.__version__, tpu_inference.__version__, jax.__version__)"], check=True)

log("Packaging wheels_full.tar...")
subprocess.run(["tar", "-cf", "/content/wheels_full.tar", "-C", "/content", "wheels", "requirements.lock"], check=True)
tar_mb = os.path.getsize("/content/wheels_full.tar") / (1024 * 1024)
log(f"Archive size: {tar_mb:.1f} MB")

log("Uploading wheels_full.tar and requirements.lock to Hugging Face...")
code_up = f'''
import json, os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
repo = "avgrocketman/tpu-compile-cache"
api.upload_file(path_or_fileobj="/content/wheels_full.tar", path_in_repo="wheels_full.tar", repo_id=repo)
api.upload_file(path_or_fileobj="/content/requirements.lock", path_in_repo="requirements.lock", repo_id=repo)
v_data = {{}}
try:
    p = hf_hub_download(repo_id=repo, filename="versions.json", local_dir="/content")
    with open(p) as f: v_data = json.load(f)
except Exception: pass
v_data.update({{
    "wheel_bundle": "wheels_full.tar",
    "num_wheels": {len(whl_files)},
    "tar_size_mb": round({tar_mb}, 2),
    "online_install_s": round({t_online}, 2),
    "offline_verify_install_s": round({t_offline}, 2)
}})
with open("/content/versions.json", "w") as f: json.dump(v_data, f, indent=2)
api.upload_file(path_or_fileobj="/content/versions.json", path_in_repo="versions.json", repo_id=repo)
'''
subprocess.run(["/content/venv_build/bin/python", "-c", code_up], check=True)
log("Upload complete.")

res = {
    "stage": "wheel_builder",
    "online_install_s": round(t_online, 2),
    "wheel_download_s": round(t_dl, 2),
    "offline_verify_install_s": round(t_offline, 2),
    "num_wheels": len(whl_files),
    "tar_size_mb": round(tar_mb, 2)
}
with open("/content/wheel_results.json", "w") as f: json.dump(res, f)
print("=== WHEEL_BUILD_RESULTS_JSONL ===", flush=True)
print(json.dumps(res), flush=True)
print("=== END_WHEEL_BUILD_RESULTS ===", flush=True)
hb_stop.set()
