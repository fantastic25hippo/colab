import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"

try:
    import hf_transfer  # noqa: F401
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])
    import hf_transfer  # noqa: F401
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("HF_TOKEN environment variable is not set")

stage, stage_start, stop_hb = "init", time.time(), threading.Event()


def hb():
    while not stop_hb.wait(20.0):
        print(f"[heartbeat] {stage}: {time.time() - stage_start:.1f}s", flush=True)


threading.Thread(target=hb, daemon=True).start()

repo, blob_name = "avgrocketman/LFM2.5-24B-A2B-GGUF", "LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"

# 1a. Download blob
stage, stage_start = "download_blob", time.time()
print(f"Downloading {blob_name} to /tmp/blob...", flush=True)
t0 = time.time()
blob_path = hf_hub_download(repo_id=repo, filename=blob_name, local_dir="/tmp/blob", token=token)
blob_time = time.time() - t0
blob_bytes = os.path.getsize(blob_path)
blob_mbs = (blob_bytes / (1024 * 1024)) / blob_time
print(f"Blob: {blob_bytes} B in {blob_time:.2f}s = {blob_mbs:.2f} MB/s", flush=True)


# 1b. Get llama-gguf-split and split
def get_splitter():
    cmd = shutil.which("llama-gguf-split")
    if cmd:
        return cmd
    try:
        req = urllib.request.Request(
            "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest",
            headers={"User-Agent": "colab-sweep"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            rel = json.load(r)
        url = next(a["browser_download_url"] for a in rel.get("assets", []) if "bin-ubuntu-x64.zip" in a.get("name", ""))
        zip_p = "/tmp/llama-bin.zip"
        urllib.request.urlretrieve(url, zip_p)
        with zipfile.ZipFile(zip_p) as z:
            for m in z.namelist():
                if os.path.basename(m) == "llama-gguf-split":
                    z.extract(m, "/tmp/bin")
                    p = os.path.join("/tmp/bin", m)
                    os.chmod(p, 0o755)
                    return p
    except Exception as e:
        print(f"Download split binary failed ({e}), building...", flush=True)
    cmd = (
        "git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /tmp/llama.cpp "
        "&& cmake -B /tmp/llama.cpp/build -S /tmp/llama.cpp -DLLAMA_BUILD_EXAMPLES=OFF "
        "-DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF "
        "&& cmake --build /tmp/llama.cpp/build --target llama-gguf-split -j$(nproc)"
    )
    subprocess.check_call(cmd, shell=True)
    return "/tmp/llama.cpp/build/bin/llama-gguf-split"


stage, stage_start = "split", time.time()
splitter = get_splitter()
os.makedirs("/tmp/shards", exist_ok=True)
prefix = f"/tmp/shards/{os.path.splitext(blob_name)[0]}"
print(f"Splitting with {splitter}...", flush=True)
subprocess.check_call([splitter, "--split", "--split-max-size", "1G", blob_path, prefix])
shards = sorted(os.listdir("/tmp/shards"))
print(f"Shards created (N={len(shards)}):", flush=True)
for s in shards:
    sp = os.path.join("/tmp/shards", s)
    sz = os.path.getsize(sp)
    print(f"  {s}: {sz} B ({sz / (1024*1024):.2f} MB)", flush=True)

# 1c. Upload shards
stage, stage_start = "upload_shards", time.time()
print(f"Uploading /tmp/shards to {repo}/shards...", flush=True)
api = HfApi(token=token)
api.upload_folder(folder_path="/tmp/shards", path_in_repo="shards", repo_id=repo, repo_type="model")

# 1d. Wipe local files & HF caches
stage, stage_start = "cleanup", time.time()
shutil.rmtree("/tmp/shards", ignore_errors=True)
shutil.rmtree("/tmp/blob", ignore_errors=True)
shutil.rmtree(os.path.expanduser("~/.cache/huggingface"), ignore_errors=True)

# 1e. Measure snapshot_download across worker counts
results = []
for w in [1, 4, 8, 16]:
    stage, stage_start = f"download_w{w}", time.time()
    p_dir, c_dir = f"/tmp/pull-w{w}", f"/tmp/cache-w{w}"
    t_start = time.time()
    snapshot_download(
        repo_id=repo,
        allow_patterns=["shards/*"],
        local_dir=p_dir,
        max_workers=w,
        cache_dir=c_dir,
        token=token,
    )
    dur = time.time() - t_start
    b_total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p_dir) for f in fs)
    mbs = (b_total / (1024 * 1024)) / dur if dur > 0 else 0
    results.append((w, b_total, dur, mbs))
    print(f"w={w}: {b_total} B in {dur:.2f}s = {mbs:.2f} MB/s", flush=True)
    shutil.rmtree(p_dir, ignore_errors=True)
    shutil.rmtree(c_dir, ignore_errors=True)

stop_hb.set()

print("\n--- RESULTS ---", flush=True)
print(f"BLOB: {blob_bytes} B, {blob_time:.2f} s, {blob_mbs:.2f} MB/s", flush=True)
for w, b, d, m in results:
    print(f"WORKERS_{w}: {b} B, {d:.2f} s, {m:.2f} MB/s", flush=True)
print("--- END RESULTS ---", flush=True)
