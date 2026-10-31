import base64, io, json, os, subprocess, sys, tarfile, threading, time

hb_stop = threading.Event()
def hb():
    while not hb_stop.wait(20): print(f"[REPACK_HB {time.strftime('%H:%M:%S')}] Active processing...", flush=True)
threading.Thread(target=hb, daemon=True).start()

def log(msg): print(f"[GUEST_REPACK {time.strftime('%H:%M:%S')}] {msg}", flush=True)

t0 = time.time()
log("Step 1: Installing/verifying prerequisites (gguf, safetensors, huggingface_hub, hf_transfer)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf", "safetensors", "huggingface_hub", "hf_transfer"], check=True)

if os.environ.get("PACKAGE_B64"):
    raw_tar = base64.b64decode(os.environ["PACKAGE_B64"])
    with tarfile.open(fileobj=io.BytesIO(raw_tar)) as tf: tf.extractall("/content")
    log("Unpacked lfm25_200k modules to /content")

sys.path.insert(0, "/content")
sys.path.insert(0, "/content/lfm25_200k")

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token and os.path.exists("/root/.config/colab-cli/hf_token"):
    hf_token = open("/root/.config/colab-cli/hf_token").read().strip()

from huggingface_hub import hf_hub_download, HfApi
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
target_repo = "avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU"

log("Step 2: Downloading LFM2.5-24B-A2B-QAD-Q4_0.gguf from LiquidAI/LFM2.5-24B-A2B-GGUF...")
t_dl0 = time.time()
gguf_p = hf_hub_download(
    repo_id="LiquidAI/LFM2.5-24B-A2B-GGUF",
    filename="LFM2.5-24B-A2B-QAD-Q4_0.gguf",
    local_dir="/content",
    token=hf_token
)
t_dl = time.time() - t_dl0
file_sz_gb = os.path.getsize(gguf_p) / (1024**3)
log(f"Downloaded {file_sz_gb:.2f} GiB GGUF in {t_dl:.2f}s ({file_sz_gb * 1024 / max(1e-6, t_dl):.1f} MB/s)")

log("Step 3: Streaming repack into Safetensors shards and uploading to HF Hub...")
from lfm25_200k.repack_checkpoint import repack_lfm25_gguf

t_rp0 = time.time()
repack_lfm25_gguf(
    gguf_p=gguf_p,
    out_dir="/content/shards",
    moe_group_size=128,
    quantize_embeddings=False,
    upload_repo=target_repo,
    token=hf_token
)
t_rp = time.time() - t_rp0
log(f"Repack and streaming upload completed in {t_rp:.2f}s")

try: os.remove(gguf_p)
except Exception: pass

log("Step 4: Verifying destination repo files...")
api = HfApi(token=hf_token)
repo_files = api.list_repo_files(target_repo)
safetensors_files = sorted([f for f in repo_files if f.endswith(".safetensors")])
log(f"Verified {len(safetensors_files)} safetensors shards in {target_repo}")

total_s = time.time() - t0
summary = {
    "target_repo": target_repo,
    "shards_count": len(safetensors_files),
    "download_s": round(t_dl, 2),
    "repack_and_upload_s": round(t_rp, 2),
    "total_s": round(total_s, 2),
    "shards": safetensors_files
}
print("=== REPACK_RESULTS_JSONL ===", flush=True)
print(json.dumps(summary), flush=True)
print("=== END_REPACK_RESULTS ===", flush=True)
hb_stop.set()
