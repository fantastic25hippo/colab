import os, sys, time, shutil, subprocess, threading, urllib.request, glob
from huggingface_hub import hf_hub_download, snapshot_download, HfApi

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()

tok_path = "/content/.hf_token"
if os.path.exists(tok_path):
    with open(tok_path) as f:
        tok = f.read().strip()
    if tok: os.environ["HF_TOKEN"] = tok
    os.remove(tok_path)

for k in ["HF_HUB_ENABLE_HF_TRANSFER", "HF_XET_HIGH_PERFORMANCE", "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"]:
    os.environ.setdefault(k, "1")

REPO, BLOB = "avgrocketman/LFM2.5-24B-A2B-Q4-shards", "LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"
stop_hb = False
def hb():
    while not stop_hb:
        time.sleep(20)
        if not stop_hb: print("[heartbeat] working...", flush=True)

threading.Thread(target=hb, daemon=True).start()
print(f"nproc: {sh('nproc')}\ndf:\n{sh('df -h /dev/shm /tmp')}\nnvidia-smi:\n{sh('nvidia-smi')}", flush=True)

rows = []
def time_action(name, fn):
    print(f"START {name}", flush=True)
    t0 = time.perf_counter()
    b = fn()
    dt = time.perf_counter() - t0
    mbs = (b / 1e6) / dt if dt > 0 else 0
    row_str = f"ROW {name}: {b} bytes {dt:.2f} seconds {mbs:.2f} MB/s"
    print(row_str, flush=True)
    rows.append(row_str)
    return dt, mbs

def get_sz(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs if not os.path.islink(os.path.join(r, f)))

def dl_blob(dst_file):
    cdir = f"{dst_file}_cache"
    shutil.rmtree(cdir, ignore_errors=True)
    f = hf_hub_download(REPO, BLOB, local_dir=os.path.dirname(dst_file), cache_dir=cdir)
    return os.path.getsize(f)

time_action("blob_shm", lambda: dl_blob("/dev/shm/blob/blob.gguf"))
shutil.rmtree("/dev/shm/blob", ignore_errors=True)
time_action("blob_tmp", lambda: dl_blob("/tmp/blob/LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"))

def dl_shards(d, mw, pat="shards/*"):
    cdir = f"{d}_cache"
    shutil.rmtree(cdir, ignore_errors=True)
    snapshot_download(REPO, allow_patterns=[pat], local_dir=d, cache_dir=cdir, max_workers=mw)
    return get_sz(d)

best_w, best_spd = 1, -1
for w in [1, 2, 4, 8, 12, 16]:
    d = f"/dev/shm/w{w}"
    _, spd = time_action(f"shards_shm_w{w}", lambda w=w, d=d: dl_shards(d, w))
    if spd > best_spd: best_w, best_spd = w, spd
    shutil.rmtree(d, ignore_errors=True); shutil.rmtree(f"{d}_cache", ignore_errors=True)

time_action(f"best_w{best_w}_tmp", lambda: dl_shards("/tmp/w_best", best_w))
shutil.rmtree("/tmp/w_best", ignore_errors=True); shutil.rmtree("/tmp/w_best_cache", ignore_errors=True)

flags = ["HF_HUB_ENABLE_HF_TRANSFER", "HF_XET_HIGH_PERFORMANCE", "HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"]
orig_flags = {k: os.environ.get(k) for k in flags}
for k in flags:
    os.environ[k] = "0"
    time_action(f"best_w{best_w}_shm_{k}_0", lambda: dl_shards("/dev/shm/w_flag", best_w))
    shutil.rmtree("/dev/shm/w_flag", ignore_errors=True); shutil.rmtree("/dev/shm/w_flag_cache", ignore_errors=True)
    os.environ[k] = orig_flags[k]

kept_blob = "/tmp/blob/LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"
cpu_tar = "https://github.com/ggml-org/llama.cpp/releases/download/b11954/llama-b11954-bin-ubuntu-x64.tar.gz"
cuda_tar = "https://github.com/ggml-org/llama.cpp/releases/download/b11954/llama-b11954-bin-ubuntu-cuda-12.8-x64.tar.gz"

cpu_bin_ok = False
if os.environ.get("HF_TOKEN") and os.path.exists(kept_blob):
    try:
        urllib.request.urlretrieve(cpu_tar, "/tmp/llama-cpu.tar.gz")
        sh("tar -xzf /tmp/llama-cpu.tar.gz -C /tmp")
        splitter = sh("find /tmp -name llama-gguf-split | head -n 1")
        if splitter and os.path.exists(splitter):
            os.chmod(splitter, 0o755); cpu_bin_ok = True; api = HfApi()
            for sz, pref in [("2G", "shards-8"), ("500M", "shards-32")]:
                out_dir = f"/tmp/{pref}"; os.makedirs(out_dir, exist_ok=True)
                sh(f"{splitter} --split --split-max-size {sz} {kept_blob} {out_dir}/LFM2.5-24B-A2B-UD-Q4_K_XL")
                for f in os.listdir(out_dir):
                    api.upload_file(path_or_fileobj=os.path.join(out_dir, f), path_in_repo=f"{pref}/{f}", repo_id=REPO)
                for w in [8, 16]:
                    d = f"/dev/shm/{pref}_w{w}"
                    time_action(f"{pref}_shm_w{w}", lambda d=d, w=w, p=f"{pref}/*": dl_shards(d, w, p))
                    shutil.rmtree(d, ignore_errors=True); shutil.rmtree(f"{d}_cache", ignore_errors=True)
    except Exception as e:
        print(f"Error in step d: {e}", flush=True)

winner_dir = "/dev/shm/bundle"
dl_shards(winner_dir, best_w)

load_path, srv_bin = None, None
try:
    urllib.request.urlretrieve(cuda_tar, "/tmp/llama-cuda.tar.gz")
    sh("mkdir -p /tmp/cuda && tar -xzf /tmp/llama-cuda.tar.gz -C /tmp/cuda")
    srv = sh("find /tmp/cuda -name llama-server | head -n 1")
    if srv and os.path.exists(srv):
        os.chmod(srv, 0o755); srv_bin, load_path = srv, "cuda"
except Exception as e:
    print(f"CUDA tar error: {e}", flush=True)

if not srv_bin or subprocess.run([srv_bin, "--version"], capture_output=True).returncode != 0:
    print(f"CUDA failed. nvidia-smi:\n{sh('nvidia-smi')}", flush=True)
    if not cpu_bin_ok:
        urllib.request.urlretrieve(cpu_tar, "/tmp/llama-cpu.tar.gz")
        sh("tar -xzf /tmp/llama-cpu.tar.gz -C /tmp")
    srv_bin, load_path = sh("find /tmp -name llama-server | head -n 1"), "cpu"
    if srv_bin: os.chmod(srv_bin, 0o755)

first_shard = sorted(glob.glob(f"{winner_dir}/shards/*-00001-of-*.gguf"))[0]
print(f"Starting server {srv_bin} ({load_path}) on {first_shard}...", flush=True)
p = subprocess.Popen([srv_bin, "-m", first_shard, "--no-mmap", "-ngl", "99"])
t0_srv, srv_ready = time.perf_counter(), False
while time.perf_counter() - t0_srv < 180:
    time.sleep(0.5)
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=1) as resp:
            if resp.status == 200:
                srv_ready = True; break
    except Exception: pass
dt_srv = time.perf_counter() - t0_srv
p.kill()
rows.append(f"ROW health_check: {round(dt_srv, 2)} seconds path={load_path} ready={srv_ready}")

stop_hb = True
print("--- RESULTS ---", flush=True)
for r in rows: print(r, flush=True)
print("--- END RESULTS ---", flush=True)
