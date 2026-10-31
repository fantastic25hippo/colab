import os, sys, time, json, glob, subprocess, threading, importlib.metadata

hb_stop, hb_stage = threading.Event(), ["init"]
def hb_worker():
    while not hb_stop.wait(20):
        print(json.dumps({"stage": "heartbeat", "current_stage": hb_stage[0], "timestamp": time.time()}), flush=True)
threading.Thread(target=hb_worker, daemon=True).start()

def get_pkg_ver(pkg):
    try: return importlib.metadata.version(pkg)
    except Exception: return None

def get_mem():
    try:
        import jax
        st = jax.devices()[0].memory_stats()
        return (st.get("bytes_in_use", "unavailable"), st.get("peak_bytes_in_use", "unavailable")) if st else ("unavailable", "unavailable")
    except Exception: return "unavailable", "unavailable"

# Stage a: env
try:
    import jax
    tpu_devs = [str(d) for d in jax.devices()]
except Exception as e: tpu_devs = str(e)
ram_out = subprocess.run(["free", "-g"], capture_output=True, text=True).stdout.strip()
shm_out = subprocess.run(["df", "-h", "/dev/shm"], capture_output=True, text=True).stdout.strip()
ram_tot = next((l.split()[1] for l in ram_out.splitlines() if l.startswith("Mem:")), "unavailable")
shm_sz = next((l.split()[1] for l in shm_out.splitlines() if "/dev/shm" in l), "unavailable")
env_info = {"stage": "env", "timestamp": time.time(), "python_version": sys.version, "total_ram_gb": ram_tot, "shm_size": shm_sz, "tpu_devices": tpu_devs}
print(json.dumps(env_info), flush=True)

# Stage b: install
hb_stage[0] = "install"
t_i0, inst_err = time.time(), None
p1 = subprocess.run([sys.executable, "-m", "pip", "install", "vllm-tpu"], capture_output=True, text=True)
if p1.returncode != 0:
    inst_err = f"vllm-tpu failed (code {p1.returncode}): {p1.stderr.strip() or p1.stdout.strip()}"
    print(json.dumps({"stage": "install_fallback", "error": inst_err, "timestamp": time.time()}), flush=True)
    p2 = subprocess.run([sys.executable, "-m", "pip", "install", "tpu-inference", "vllm"], capture_output=True, text=True)
    if p2.returncode != 0: inst_err += f"\ntpu-inference vllm failed (code {p2.returncode}): {p2.stderr.strip() or p2.stdout.strip()}"
t_i1 = time.time()
install_info = {"stage": "install", "timestamp": time.time(), "start": t_i0, "end": t_i1, "seconds": t_i1 - t_i0,
                "vllm_version": get_pkg_ver("vllm"), "jax_version": get_pkg_ver("jax"), "torch_xla_version": get_pkg_ver("torch_xla"), "error": inst_err}
print(json.dumps(install_info), flush=True)

# Stage c: download
hb_stage[0] = "download"
from huggingface_hub import snapshot_download
t_dl0 = time.time()
try:
    snapshot_download("google/gemma-4-E4B-it-qat-w4a16-ct", local_dir="/dev/shm/gemma", max_workers=12)
    t_dl1 = time.time()
    dl_sec = t_dl1 - t_dl0
    st_files = glob.glob("/dev/shm/gemma/**/*.safetensors", recursive=True)
    st_bytes = sum(os.path.getsize(f) for f in st_files)
    dl_mbs = (st_bytes / 1e6) / dl_sec if dl_sec > 0 else 0
    download_info = {"stage": "download", "timestamp": time.time(), "seconds": dl_sec, "safetensors_bytes": st_bytes, "mb_per_sec": dl_mbs}
    print(json.dumps(download_info), flush=True)
except Exception as e:
    import traceback
    print(json.dumps({"stage": "download", "error": traceback.format_exc(), "timestamp": time.time()}), flush=True)
    hb_stop.set(); sys.exit(0)

# Stage d: memory_before
mem_b_use, mem_b_peak = get_mem()
print(json.dumps({"stage": "memory_before", "timestamp": time.time(), "bytes_in_use": mem_b_use, "peak_bytes_in_use": mem_b_peak}), flush=True)

# Stage e: load
hb_stage[0] = "load"
t_ld0 = time.time()
try:
    import vllm
    llm = vllm.LLM(model="/dev/shm/gemma", max_model_len=8192)
    load_sec = time.time() - t_ld0
    print(json.dumps({"stage": "load", "timestamp": time.time(), "seconds": load_sec}), flush=True)
except Exception:
    import traceback
    print(json.dumps({"stage": "load", "error": traceback.format_exc(), "timestamp": time.time()}), flush=True)
    hb_stop.set(); sys.exit(0)

# Stage f: memory_after_load
hb_stage[0] = "eval"
mem_a_use, mem_a_peak = get_mem()
packed_ratio = (mem_a_use / st_bytes) if (isinstance(mem_a_use, (int, float)) and st_bytes > 0) else None
verdict = "weights unpacked to bf16" if (packed_ratio is not None and packed_ratio > 1.5) else "weights stay packed"
print(json.dumps({"stage": "memory_after_load", "timestamp": time.time(), "bytes_in_use": mem_a_use, "peak_bytes_in_use": mem_a_peak, "packed_ratio": packed_ratio, "verdict": verdict}), flush=True)

# Stage g: first_generation
from vllm import SamplingParams
sp32 = SamplingParams(max_tokens=32, temperature=0.0)
t_fg0 = time.time()
try:
    llm.generate(["Write a short story about a lighthouse."], sp32)
    fg_sec = time.time() - t_fg0
    mem_fg_use, mem_fg_peak = get_mem()
    comp_oh = (mem_fg_peak - mem_a_use) if (isinstance(mem_fg_peak, (int, float)) and isinstance(mem_a_use, (int, float))) else "unavailable"
    fg_info = {"stage": "first_generation", "timestamp": time.time(), "seconds": fg_sec, "bytes_in_use": mem_fg_use, "peak_bytes_in_use": mem_fg_peak, "compile_overhead": comp_oh}
    print(json.dumps(fg_info), flush=True)
except Exception:
    import traceback
    print(json.dumps({"stage": "first_generation", "error": traceback.format_exc(), "timestamp": time.time()}), flush=True)
    hb_stop.set(); sys.exit(0)

# Stage h: decode
sp256 = SamplingParams(max_tokens=256, temperature=0.0)
decode_runs, prompt = [], "Write a short story about a lighthouse."
for i in range(3):
    t0 = time.time()
    out = llm.generate([prompt], sp256)[0]
    tot_s = time.time() - t0
    toks = len(out.outputs[0].token_ids) if hasattr(out.outputs[0], "token_ids") else 256
    tps = toks / tot_s if tot_s > 0 else 0
    ttft = "unavailable"
    m = getattr(out, "metrics", None)
    if m:
        ft = getattr(m, "first_token_time", None)
        ar = getattr(m, "arrival_time", None) or getattr(m, "first_scheduled_time", None)
        if ft is not None and ar is not None: ttft = ft - ar
        elif getattr(m, "time_to_first_token", None) is not None: ttft = m.time_to_first_token
    decode_runs.append({"run": i + 1, "ttft": ttft, "total_seconds": tot_s, "tokens": toks, "tokens_per_second": tps})
print(json.dumps({"stage": "decode", "timestamp": time.time(), "runs": decode_runs}), flush=True)

# Stage i: done
hb_stop.set()
summary = {"stage": "done", "timestamp": time.time(), "env": env_info, "install": install_info, "download": download_info,
           "memory_before": {"bytes_in_use": mem_b_use, "peak_bytes_in_use": mem_b_peak}, "load": {"seconds": load_sec},
           "memory_after_load": {"bytes_in_use": mem_a_use, "peak_bytes_in_use": mem_a_peak, "packed_ratio": packed_ratio, "verdict": verdict},
           "first_generation": fg_info, "decode": decode_runs}
print(json.dumps(summary), flush=True)
