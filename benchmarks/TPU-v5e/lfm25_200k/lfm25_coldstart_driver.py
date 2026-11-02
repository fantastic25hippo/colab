import base64, concurrent.futures, hashlib, json, os, re, subprocess, sys, threading, time

hf_anon = os.environ.get("LFM_HF_ANON", "1") != "0"
dl_token = False if hf_anon else os.environ.get("HF_TOKEN")

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
t_session_connect0 = float(os.environ.get("T_SESSION_CONNECT0", t_global0))
t_upload0 = float(os.environ.get("T_UPLOAD0", t_session_connect0))
t_exec0 = float(os.environ.get("T_EXEC0", t_upload0))

# Unpack subrunner, patches, pallas kernel, prompt builder, startup + cache helpers.
diag = os.environ.get("LFM_DIAG") == "1"
sub_name = "diag_short_subrunner.py" if diag else "lfm25_engine_subrunner.py"
if diag: log("LFM_DIAG=1: launching short-context diagnostic subrunner")
for env_k, target_f in [
    ("SUBRUNNER_B64", f"/content/{sub_name}"),
    ("PATCHES_B64", "/content/tpu_v5e_patches.py"),
    ("PALLAS_B64", "/content/k8v4_pallas_kernel.py"),
    ("PROMPT_B64", "/content/lfm25_prompt.py"),
    ("STARTUP_B64", "/content/lfm25_startup.py"),
    ("CACHE_B64", "/content/lfm25_cache.py"),
    ("CACHE_NAMING_B64", "/content/cache_naming.py"),
]:
    if os.environ.get(env_k):
        with open(target_f, "wb") as f: f.write(base64.b64decode(os.environ[env_k]))

sub_src = open(f"/content/{sub_name}").read() if os.path.exists(f"/content/{sub_name}") else ""
m_len = re.search(r"max_model_len=(\d+)", sub_src)
max_len = int(os.environ.get("LFM_MAX_MODEL_LEN") or (m_len.group(1) if m_len else (4096 if diag else 0)))
len_lbl = f"{max_len//1000}K" if max_len and max_len % 1000 == 0 else str(max_len)
log(f"In-VM {len_lbl} cold-start timer started at t_vm0={t_vm0:.2f} (t_global0={t_global0:.2f})")

sys.path.insert(0, "/content")
from lfm25_cache import (CACHE_DIR, cache_filename, clean_dir, dir_stats, extract_cache_key_args,
                         publish_cache, read_vllm_version, restore_cache)
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.environ["VLLM_XLA_CACHE_PATH"] = CACHE_DIR

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
if os.path.exists("/dev/shm/.weights_ready"): os.remove("/dev/shm/.weights_ready")

def dl_venv():
    t0 = time.time()
    if hf_anon:
        log("[LFM_HF_ANON] download token=none venv312_cpu.tar.zst")
        log("[LFM_HF_ANON] download token=none zstd_bin")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as p_dl:
        f_v = p_dl.submit(hf_hub_download, repo_id="avgrocketman/tpu-compile-cache",
                          filename="venv312_cpu.tar.zst", local_dir="/dev/shm", token=dl_token)
        f_z = p_dl.submit(hf_hub_download, repo_id="avgrocketman/tpu-compile-cache",
                          filename="zstd_bin", local_dir="/dev/shm", token=dl_token)
        p_venv, p_zstd = f_v.result(), f_z.result()
    os.chmod(p_zstd, 0o755)
    t_end = time.time()
    dur = round(t_end - t0, 2)
    log(f"Downloaded venv + zstd_bin in {dur}s (no Qwen XLA cache for LFM)")
    return p_venv, p_zstd, dur, t0, t_end

def unpack_venv(p_venv, p_zstd):
    t_u0 = time.time()
    subprocess.run(["sh", "-c", f"{p_zstd} -d -T0 -c {p_venv} | tar -xf - -C /dev/shm"], check=True)
    try: os.remove(p_venv)
    except Exception: pass
    t_u_end = time.time()
    dur = round(t_u_end - t_u0, 2)
    log(f"Unpacked venv in {dur}s")
    return dur, t_u0, t_u_end

def weight_manifest(root):
    files = sorted(f for f in os.listdir(root) if f.endswith(".safetensors"))
    sizes = [(f, os.path.getsize(os.path.join(root, f))) for f in files]
    total = sum(s for _, s in sizes)
    return {"shards": len(files), "total_mb": round(total / 1e6, 1),
            "per_shard_mb": {f: round(s / 1e6, 1) for f, s in sizes},
            "largest": max(sizes, key=lambda x: x[1])[0] if sizes else None}

def shm_read_probe(path, limit=1024**3):
    """Read one shard from tmpfs page cache; measures /dev/shm -> userspace copy."""
    t0, n = time.time(), 0
    with open(path, "rb") as f:
        while n < limit:
            b = f.read(8 * 1024 * 1024)
            if not b: break
            n += len(b)
    dt = max(1e-6, time.time() - t0)
    return {"read_mb": round(n / 1e6, 1), "s": round(dt, 3), "mb_s": round(n / 1e6 / dt, 1)}

def dl_weights():
    t0 = time.time()
    model_id = os.environ.get("LFM25_MODEL_ID", "avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU")
    if hf_anon: log(f"[LFM_HF_ANON] download token=none {model_id}")
    snapshot_download(model_id, local_dir="/dev/shm/lfm25_200k", max_workers=12, token=dl_token)
    cfg_p = "/dev/shm/lfm25_200k/config.json"
    if os.path.exists(cfg_p):
        with open(cfg_p) as f: c = json.load(f)
        if "layer_types" not in c or not c["layer_types"]:
            c["layer_types"] = ["full_attention" if i in (2, 6, 10, 14, 18, 22, 26, 30, 34, 38) else "conv" for i in range(40)]
            with open(cfg_p, "w") as f: json.dump(c, f, indent=2)
    t_end = time.time()
    dur = t_end - t0
    man = weight_manifest("/dev/shm/lfm25_200k")
    man["dl_s"] = round(dur, 2)
    man["t_start"], man["t_end"] = t0, t_end
    man["mb_s"] = round(man["total_mb"] / max(dur, 1e-6), 1)
    if man["largest"]: man["shm_read_probe"] = shm_read_probe(os.path.join("/dev/shm/lfm25_200k", man["largest"]))
    with open("/dev/shm/.weights_ready", "w") as f: f.write("1\n")
    log(f"Weights ready in {dur:.2f}s ({man['shards']} shards, {man['total_mb']} MB, {man['mb_s']} MB/s)")
    return man

def dl_and_unpack_cache(c_name, a_str):
    t_c0 = time.time()
    t_dl_end = t_u0 = t_u_end = None
    try:
        if hf_anon: log(f"[LFM_HF_ANON] download token=none {c_name}")
        p_tar = hf_hub_download(repo_id="avgrocketman/tpu-compile-cache",
                                filename=c_name, local_dir="/dev/shm", token=dl_token)
        t_dl_end = time.time()
        t_dl = round(t_dl_end - t_c0, 2)
        t_u0 = time.time()
        r = subprocess.run(["tar", "-xf", p_tar, "-C", "/dev/shm"], capture_output=True, text=True)
        t_u_end = time.time()
        t_unp = round(t_u_end - t_u0, 2)
        try: os.remove(p_tar)
        except Exception: pass
        if r.returncode != 0:
            clean_dir()
            res = {"hit": False, "reason": f"tar unpack failed rc={r.returncode}: {r.stderr}",
                   "cache_dl_s": t_dl, "cache_unpack_s": t_unp,
                   "t_dl_start": t_c0, "t_dl_end": t_dl_end, "t_unp_start": t_u0, "t_unp_end": t_u_end}
        else:
            res = {"hit": True, "cache_dl_s": t_dl, "cache_unpack_s": t_unp,
                   "t_dl_start": t_c0, "t_dl_end": t_dl_end, "t_unp_start": t_u0, "t_unp_end": t_u_end}
    except Exception as e:
        clean_dir()
        t_dl_end = time.time()
        res = {"hit": False, "reason": str(e), "cache_dl_s": round(t_dl_end - t_c0, 2),
               "t_dl_start": t_c0, "t_dl_end": t_dl_end}
    res.update({"filename": c_name, "vllm": "0.32.0", "key_args": a_str, "dir": dir_stats()})
    c_dl = res.get("cache_dl_s", res.get("dl_s"))
    c_unp = res.get("cache_unpack_s", res.get("unpack_s"))
    if not res.get("hit"): log(f"cache miss: {c_name}")
    log(f"Compile cache {c_name}: {res} (cache_dl_s={c_dl}, cache_unpack_s={c_unp})")
    return res

def setup_thp():
    if os.environ.get("LFM_THP", "1") == "0": return
    try:
        r_thp = subprocess.run(
            ["sh", "-c", "echo always > /sys/kernel/mm/transparent_hugepage/enabled && echo always > /sys/kernel/mm/transparent_hugepage/defrag"],
            capture_output=True, text=True)
        if r_thp.returncode == 0: log("THP OK")
        else: log(f"THP FAIL ({r_thp.stderr.strip() or r_thp.returncode})")
    except Exception as e: log(f"THP FAIL ({e})")

log("Step 2: weights, venv and LFM compile cache all in flight; unpacks after...")
t_prep0 = time.time()
par_cache = os.environ.get("LFM_CACHE_PARALLEL", "1") != "0"
early_imp = os.environ.get("LFM_EARLY_IMPORT", "1") != "0"
forced = os.environ.get("LFM_CACHE_FORCE_COLD") == "1"

args_str = extract_cache_key_args(sub_src) or f"src:{hashlib.sha256(sub_src.encode()).hexdigest()[:12]}"
patches_data = open("/content/tpu_v5e_patches.py", "rb").read() if os.path.exists("/content/tpu_v5e_patches.py") else b""
buckets_str = "16-4096" if diag else "16-8192"

pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
f_wt = pool.submit(dl_weights)
t_wt_submit = time.time()
f_cache = None
cache_res = None
cache_name = None

if par_cache:
    cache_name = cache_filename("0.32.0", args_str, patches_data, max_model_len=max_len, buckets=buckets_str)
    if diag or forced:
        clean_dir()
        cache_res = {"hit": False, "reason": "diag" if diag else "LFM_CACHE_FORCE_COLD=1",
                     "filename": cache_name, "vllm": "0.32.0", "key_args": args_str, "dir": dir_stats()}
        log(f"Compile cache {cache_name}: {cache_res}")
    else:
        f_cache = pool.submit(dl_and_unpack_cache, cache_name, args_str)

p_v, p_z, t_venv_dl, t_vdl0, t_vdl1 = dl_venv()
t_venv_unp, t_vunp0, t_vunp1 = unpack_venv(p_v, p_z)

if not par_cache:
    ver = read_vllm_version("/dev/shm/vllm_env")
    cache_name = cache_filename(ver, args_str, patches_data, max_model_len=max_len, buckets=buckets_str)
    if diag or forced:
        clean_dir()
        cache_res = {"hit": False, "reason": "diag" if diag else "LFM_CACHE_FORCE_COLD=1"}
    else:
        if hf_anon: log(f"[LFM_HF_ANON] download token=none {cache_name}")
        try:
            cache_res = restore_cache(cache_name, token=dl_token)
        except TypeError:
            _hf = os.environ.pop("HF_TOKEN", None) if hf_anon else None
            try: cache_res = restore_cache(cache_name)
            finally:
                if _hf is not None: os.environ["HF_TOKEN"] = _hf
    if not cache_res.get("hit"):
        clean_dir()
        log(f"cache miss: {cache_name}")
    cache_res.update({"filename": cache_name, "vllm": ver, "key_args": args_str, "dir": dir_stats()})
    c_dl = cache_res.get("cache_dl_s", cache_res.get("dl_s"))
    c_unp = cache_res.get("cache_unpack_s", cache_res.get("unpack_s"))
    log(f"Compile cache {cache_name}: {cache_res} (cache_dl_s={c_dl}, cache_unpack_s={c_unp})")

setup_thp()

if f_cache: cache_res = f_cache.result()
abort_on_miss = os.environ.get("LFM_ABORT_ON_CACHE_MISS", "1") != "0"
if abort_on_miss and (not cache_res or not cache_res.get("hit")):
    c_name, reason = (cache_res.get("filename", cache_name), cache_res.get("reason", "unknown")) if cache_res else (cache_name, "not_loaded")
    log(f"ABORT ON CACHE MISS: filename={c_name} reason={reason}")
    summary = {
        "ok": False,
        "error": f"compile cache miss: {c_name} (reason: {reason})",
        "cache_miss": True,
        "cache_restore": cache_res,
        "filename": c_name,
        "reason": reason,
    }
    print("=== COLDSTART_RESULTS_JSONL ===", flush=True)
    print(json.dumps(summary), flush=True)
    print("=== END_COLDSTART_RESULTS ===", flush=True)
    hb_stop.set()
    sys.exit(2)

t_spawn = time.time()
startup = {"remount_s": round(t_remount, 2), "venv_dl_s": t_venv_dl, "venv_unpack_s": t_venv_unp,
           "weights_submit_rel_s": round(t_wt_submit - t_vm0, 2),
           "prep_total_s": round(t_spawn - t_prep0, 2),
           "subrunner_spawn_rel_s": round(t_spawn - t_vm0, 2)}
if cache_res:
    startup.update({"cache_restore": cache_res,
                    "cache_dl_s": cache_res.get("cache_dl_s", cache_res.get("dl_s")),
                    "cache_unpack_s": cache_res.get("cache_unpack_s", cache_res.get("unpack_s"))})

py_bin = "/dev/shm/vllm_env/bin/python"
env_b = os.environ.copy()
for k in ("SUBRUNNER_B64", "PATCHES_B64", "PALLAS_B64", "PROMPT_B64", "STARTUP_B64", "CACHE_B64"):
    env_b.pop(k, None)
env_b["PYTHONPATH"] = f"/content:{env_b.get('PYTHONPATH', '')}"
env_b.update({
    "MODEL_IMPL_TYPE": "vllm", "TPU_SKIP_MDS_QUERY": "1", "TPU_TOPOLOGY": "1x1",
    "TPU_HOST_BOUNDS": "1,1,1", "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1", "TPU_WORKER_ID": "0",
    "TPU_WORKER_HOSTNAMES": "localhost", "GCE_METADATA_HOST": "127.0.0.1:1",
    "GCE_METADATA_IP": "127.0.0.1:1", "GCE_METADATA_ROOT": "127.0.0.1:1", "NO_GCE_CHECK": "true",
    "JAX_PLATFORMS": "tpu,cpu", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "VLLM_LOGGING_LEVEL": "INFO"
})
if diag: env_b.update({"LFM_DIAG": "1", "LFM_CONV_DEBUG": "1"})

proc = subprocess.Popen([py_bin, "-u", f"/content/{sub_name}"],
                        env=env_b, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in proc.stdout: sys.stdout.write(line); sys.stdout.flush()
proc.wait()

wt_res = f_wt.result()
if f_cache:
    cache_res = f_cache.result()
pool.shutdown(wait=True)

if cache_res:
    c_dl = cache_res.get("cache_dl_s", cache_res.get("dl_s"))
    c_unp = cache_res.get("cache_unpack_s", cache_res.get("unpack_s"))
    startup.update({"cache_restore": cache_res, "cache_dl_s": c_dl, "cache_unpack_s": c_unp})

eng_data = {}
if os.path.exists("/content/engine_results.json"):
    with open("/content/engine_results.json") as f: eng_data = json.load(f)

pub = None
if not diag and cache_res and not cache_res.get("hit") and eng_data.get("ok") and os.environ.get("LFM_NO_UPLOAD") != "1":
    try:
        pub = publish_cache(cache_name)
        log(f"Compile-cache publish: {pub}")
    except Exception as e:
        pub = {"error": f"{type(e).__name__}: {e}"}
        log(f"Compile-cache publish failed: {pub}")

startup["weights"] = wt_res
first_tok_ts = eng_data.get("first_token_ts")

# --- Host WATERFALL computation ---
# Host rows: session connect, upload, EXEC start, venv dl, venv unpack, weights dl, cache dl, cache unpack.
# Then in-VM rows from subrunner JSON (shifted by t_vm0 - t_global0). Then TTFT.
waterfall_rows = []
def _add_phase(name, t_s, t_e, origin="host"):
    if t_s is not None and t_e is not None and t_e >= t_s:
        waterfall_rows.append({"name": name, "start_rel": round(t_s - t_global0, 3), "dur_s": round(t_e - t_s, 3), "end_rel": round(t_e - t_global0, 3), "origin": origin})

# Host phases
for n, s, e in [("session connect", t_session_connect0, t_upload0), ("upload", t_upload0, t_exec0), ("EXEC start", t_exec0, t_vm0)]:
    if e > s: _add_phase(n, s, e)
_add_phase("venv dl", t_vdl0, t_vdl1); _add_phase("venv unpack", t_vunp0, t_vunp1)
if wt_res.get("t_start") and wt_res.get("t_end"): _add_phase("weights dl", wt_res["t_start"], wt_res["t_end"])
if cache_res:
    for n, k in [("cache dl", "t_dl"), ("cache unpack", "t_unp")]:
        if cache_res.get(f"{k}_start") and cache_res.get(f"{k}_end"): _add_phase(n, cache_res[f"{k}_start"], cache_res[f"{k}_end"])

# Subrunner in-VM rows
vm_off = round(float(eng_data.get("t_vm0") or t_vm0) - t_global0, 3)
for r in eng_data.get("waterfall", []):
    waterfall_rows.append({"name": r.get("name", ""), "start_rel": round(r.get("start_rel", 0.0) + vm_off, 3), "dur_s": round(r.get("dur_s", 0.0), 3), "end_rel": round(r.get("end_rel", 0.0) + vm_off, 3), "origin": "in-vm"})

for r in waterfall_rows: r["parallel"] = r.get("parallel") or (r["name"] in ("venv dl", "weights dl", "cache dl", "cache unpack"))

# TTFT row (only if not already provided by subrunner waterfall)
has_ttft = any(r.get("name", "").lower() == "ttft" for r in eng_data.get("waterfall", []))
ttft_s = eng_data.get("ttft_short_s") or eng_data.get("short", {}).get("ttft_s")
if not has_ttft and first_tok_ts and ttft_s is not None: _add_phase("ttft", first_tok_ts - ttft_s, first_tok_ts, origin="in-vm")

# Sort waterfall rows by start_rel then end_rel & calculate non-overlapping merged intervals
waterfall_rows.sort(key=lambda r: (r["start_rel"], r["end_rel"]))
merged_intervals = []
for r in waterfall_rows:
    s, e = r["start_rel"], r["end_rel"]
    if merged_intervals and s <= merged_intervals[-1][1]: merged_intervals[-1][1] = max(merged_intervals[-1][1], e)
    else: merged_intervals.append([s, e])

accounted_s = round(sum(e - s for s, e in merged_intervals), 3)
total_s = round(first_tok_ts - t_global0, 3) if first_tok_ts else (merged_intervals[-1][1] if merged_intervals else 0.0)
unaccounted_s = round(max(0.0, total_s - accounted_s), 3)

unaccounted_gaps = []
for i in range(len(merged_intervals) - 1):
    gap = round(merged_intervals[i+1][0] - merged_intervals[i][1], 3)
    if gap > 0.001:
        aft = next((r["name"] for r in waterfall_rows if abs(r["end_rel"] - merged_intervals[i][1]) < 0.005), f"{merged_intervals[i][1]:.3f}s")
        bef = next((r["name"] for r in waterfall_rows if abs(r["start_rel"] - merged_intervals[i+1][0]) < 0.005), f"{merged_intervals[i+1][0]:.3f}s")
        unaccounted_gaps.append({"after": aft, "before": bef, "gap_s": gap})

log("=== Host WATERFALL ===")
log(f"{'Phase':<24} | {'Start (s)':<10} | {'Dur (s)':<10} | {'End (s)':<10} | {'Origin':<8}")
log("-" * 72)
for r in waterfall_rows: log(f"{r['name']:<24} | {r['start_rel']:<10.3f} | {r['dur_s']:<10.3f} | {r['end_rel']:<10.3f} | {r['origin']:<8}" + (f" | parallel={str(r['parallel']).lower()}" if r.get("parallel") else ""))
log("-" * 72)
log(f"accounted_s = {accounted_s:.3f} s (sum of non-overlapping phase durations)")
log(f"unaccounted_s = {unaccounted_s:.3f} s (total = {total_s:.3f} s)")
if unaccounted_gaps: log(f"unaccounted_gaps = {unaccounted_gaps}")

summary = {
    "run_label": os.environ.get("LFM_RUN_LABEL", ""), "diag": diag,
    "startup": startup,
    "weights_dl_s": wt_res.get("dl_s"),
    "in_vm_cold_start_to_first_token_s": round(first_tok_ts - t_vm0, 2) if first_tok_ts else None,
    "total_cold_start_to_first_token_s": round(first_tok_ts - t_global0, 2) if first_tok_ts else None,
    "accounted_s": accounted_s,
    "unaccounted_s": unaccounted_s,
    "unaccounted_gaps": unaccounted_gaps,
    "waterfall": waterfall_rows,
    "engine": eng_data, "cache_publish": pub, "subrunner_rc": proc.returncode,
}
print("=== COLDSTART_RESULTS_JSONL ===", flush=True)
print(json.dumps(summary), flush=True)
print("=== END_COLDSTART_RESULTS ===", flush=True)
hb_stop.set()
if proc.returncode != 0: raise RuntimeError(f"LFM2.5 subrunner failed rc={proc.returncode}")
