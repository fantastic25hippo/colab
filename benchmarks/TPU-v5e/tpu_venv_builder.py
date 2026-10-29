import json, os, shutil, subprocess, sys, threading, time

hb_stop = threading.Event()
def hb():
    while not hb_stop.wait(15): print(json.dumps({"stage": "hb", "t": time.time()}), flush=True)
threading.Thread(target=hb, daemon=True).start()

def log(m): print(f"[VENV_BUILD {time.strftime('%H:%M:%S')}] {m}", flush=True)

if os.path.exists("/content/hf_token"):
    for d in ["/root/.cache/huggingface", "/root/.config/huggingface"]:
        os.makedirs(d, exist_ok=True); shutil.copyfile("/content/hf_token", os.path.join(d, "token"))
    os.environ["HF_TOKEN"] = open("/content/hf_token").read().strip()
    try: os.remove("/content/hf_token")
    except Exception: pass

if not shutil.which("zstd"):
    log("Installing zstd...")
    subprocess.run(["apt-get", "update", "-qq"], check=True)
    subprocess.run(["apt-get", "install", "-y", "-qq", "zstd"], check=True)

uv = shutil.which("uv") or "/root/.local/bin/uv"
if not os.path.exists(uv):
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    uv = shutil.which("uv") or "/root/.local/bin/uv"

log("Creating Python 3.12 venv at /dev/shm/vllm_env...")
t_v0 = time.time()
subprocess.run([uv, "venv", "--relocatable", "--python", "3.12", "/dev/shm/vllm_env"], check=True)
t_venv = time.time() - t_v0
log(f"Venv created in {t_venv:.2f}s")

log("Installing packages with CPU-only torch...")
t_i0 = time.time()
install_cmd = [
    uv, "pip", "install", "--python", "/dev/shm/vllm_env/bin/python",
    "--extra-index-url", "https://download.pytorch.org/whl/cpu",
    "--index-strategy", "unsafe-best-match", "torch==2.10.0+cpu",
    "vllm-tpu==0.32.0", "tpu-inference==0.32.0", "hf-transfer", "huggingface_hub", "pip"
]
subprocess.run(install_cmd, check=True)
t_install = time.time() - t_i0
log(f"Install finished in {t_install:.2f}s")

log("Checking for nvidia packages in pip list...")
res_pip = subprocess.run(["/dev/shm/vllm_env/bin/python", "-m", "pip", "list"], capture_output=True, text=True, check=True)
nvidia_pkgs = [l.strip() for l in res_pip.stdout.splitlines() if "nvidia" in l.lower()]
log(f"Nvidia package count: {len(nvidia_pkgs)}")

log("Verifying TPU detection and vLLM import...")
t_imp0 = time.time()
verify_code = 'import vllm, jax; print("TPU_DEVICES:", jax.devices())'
res_test = subprocess.run(["/dev/shm/vllm_env/bin/python", "-c", verify_code], capture_output=True, text=True)
t_import = time.time() - t_imp0
if res_test.returncode != 0:
    log(f"FATAL: Import failed (code {res_test.returncode}):\n{res_test.stderr}\n{res_test.stdout}")
    sys.exit(1)
tpu_str = res_test.stdout.strip()
log(f"TPU check passed ({t_import:.2f}s): {tpu_str}")

log("Running compileall on site-packages...")
t_c0 = time.time()
site_pkgs = "/dev/shm/vllm_env/lib/python3.12/site-packages"
subprocess.run(["/dev/shm/vllm_env/bin/python", "-m", "compileall", "-q", site_pkgs], check=True)
t_compile = time.time() - t_c0
log(f"Compileall finished in {t_compile:.2f}s")

res_du = subprocess.run(["du", "-sb", "/dev/shm/vllm_env"], capture_output=True, text=True, check=True)
venv_bytes = int(res_du.stdout.split()[0]); venv_mb = venv_bytes / (1024 * 1024)
log(f"Unpacked size: {venv_mb:.1f} MB ({venv_bytes} bytes)")

log("Packing venv312_cpu.tar.zst (tar + zstd -3 -T0)...")
t_p0 = time.time()
tar_path = "/content/venv312_cpu.tar.zst"
subprocess.run(["tar", "-I", "zstd -3 -T0", "-cf", tar_path, "-C", "/dev/shm", "vllm_env"], check=True)
t_pack = time.time() - t_p0
tar_bytes = os.path.getsize(tar_path); tar_mb = tar_bytes / (1024 * 1024)
log(f"Packed size: {tar_mb:.1f} MB ({tar_bytes} bytes) in {t_pack:.2f}s")

log("Running unpack test into /dev/shm/test_unpack...")
os.makedirs("/dev/shm/test_unpack", exist_ok=True)
t_u0 = time.time()
subprocess.run(["tar", "-I", "zstd", "-xf", tar_path, "-C", "/dev/shm/test_unpack"], check=True)
t_unpack = time.time() - t_u0
log(f"Unpack test finished in {t_unpack:.2f}s")
shutil.rmtree("/dev/shm/test_unpack")

log("Uploading venv312_cpu.tar.zst to Hugging Face...")
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); repo = "avgrocketman/tpu-compile-cache"
api.upload_file(path_or_fileobj=tar_path, path_in_repo="venv312_cpu.tar.zst", repo_id=repo)
log("Uploaded venv312_cpu.tar.zst.")

res_list = subprocess.run(["/dev/shm/vllm_env/bin/python", "-m", "pip", "list", "--format=json"], capture_output=True, text=True, check=True)
pkgs = {p["name"]: p["version"] for p in json.loads(res_list.stdout)}

v_data = {}
try:
    p = hf_hub_download(repo_id=repo, filename="versions.json", local_dir="/content")
    with open(p) as f: v_data = json.load(f)
except Exception: pass

v_data["fast_start_venv"] = {
    "archive": "venv312_cpu.tar.zst", "unpacked_size_mb": round(venv_mb, 2), "packed_size_mb": round(tar_mb, 2),
    "install_time_s": round(t_install, 2), "compileall_time_s": round(t_compile, 2), "pack_time_s": round(t_pack, 2),
    "unpack_time_s": round(t_unpack, 2), "nvidia_pkgs_count": len(nvidia_pkgs), "tpu_device": tpu_str,
    "packages": {k: pkgs.get(k) for k in ["vllm", "vllm-tpu", "tpu-inference", "torch", "jax", "jaxlib", "libtpu", "transformers", "hf-transfer", "huggingface-hub"]},
    "all_packages": pkgs, "date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
}
with open("/content/versions.json", "w") as f: json.dump(v_data, f, indent=2)
api.upload_file(path_or_fileobj="/content/versions.json", path_in_repo="versions.json", repo_id=repo)
log("Updated and uploaded versions.json.")

summary = {
    "install_time_s": round(t_install, 2), "nvidia_pkg_count": len(nvidia_pkgs),
    "unpacked_mb": round(venv_mb, 2), "packed_mb": round(tar_mb, 2),
    "unpack_test_s": round(t_unpack, 2), "tpu_device": tpu_str,
    "upload_confirmed": True, "torch_version": pkgs.get("torch"), "vllm_version": pkgs.get("vllm")
}
print("=== VENV_BUILD_RESULTS_JSONL ===", flush=True)
print(json.dumps(summary), flush=True)
print("=== END_VENV_BUILD_RESULTS ===", flush=True)
hb_stop.set()
