import builtins, json, os, subprocess, sys, time, urllib.error, urllib.request

_orig_urlopen = urllib.request.urlopen
def _fast_urlopen(url, *args, **kwargs):
    u_str = url if isinstance(url, str) else getattr(url, "full_url", str(url))
    if "metadata.google.internal" in u_str: raise urllib.error.URLError("Bypassed GCE metadata poll")
    return _orig_urlopen(url, *args, **kwargs)
urllib.request.urlopen = _fast_urlopen

try:
    from tpu_v5e_patches import install_all_patches
    install_all_patches()
except Exception: pass

os.environ.update({
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_XLA_CACHE_PATH": "/dev/shm/tpu_compile_cache",
    "JAX_COMPILATION_CACHE_DIR": "/dev/shm/tpu_compile_cache",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
})
os.makedirs("/dev/shm/tpu_compile_cache", exist_ok=True)

def log_sub(m):
    msg = f"[LFM25_SUBRUNNER {time.strftime('%H:%M:%S')}] {m}"
    print(msg, flush=True)
    try:
        with open("/content/build.log", "a") as f: f.write(msg + "\n")
    except Exception: pass

t_imp0 = time.time()
log_sub("Importing jax and vllm for LFM2.5-24B-A2B (200K context)...")
import jax
from vllm import LLM, SamplingParams
tpu_devs = str(jax.devices())
log_sub(f"TPU init complete in {time.time() - t_imp0:.2f}s: {tpu_devs}")

t_wait0 = time.time()
while not os.path.exists("/dev/shm/.weights_ready"): time.sleep(0.05)
log_sub(f"Weights barrier cleared after {time.time() - t_wait0:.2f}s wait")

def get_hbm():
    try:
        m = jax.devices()[0].memory_stats()
        return {"in_use_gib": round(m.get("bytes_in_use", 0)/(1024**3), 3),
                "peak_gib": round(m.get("peak_bytes_in_use", 0)/(1024**3), 3),
                "limit_gib": round(m.get("bytes_limit", 0)/(1024**3), 3)}
    except Exception as e: return {"error": str(e)}

t_eng0 = time.time()
model_path = "/dev/shm/lfm25_200k"
# TODO(tpu-v5e): v5e has no native FP8. vLLM 0.32.0 CacheDType has no plain
# "int8" (only "int8_per_token_head"). If LLM() rejects this string, extend
# the allowed cache dtypes; do not silently fall back to fp8/float8_e4m3fn.
llm = LLM(
    model=model_path, max_model_len=200000, max_num_seqs=1,
    max_num_batched_tokens=8192, kv_cache_dtype="int8",
    gpu_memory_utilization=0.92, enable_prefix_caching=False,
    trust_remote_code=True
)
init_dur = time.time() - t_eng0
log_sub(f"LFM2.5 Engine initialized in {init_dur:.2f}s; HBM: {get_hbm()}")

p_short = "Explain the fundamental principles of astrophysics and stellar evolution."
sp_1 = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0, ignore_eos=True)
t0 = time.perf_counter()
llm.generate([p_short], sp_1)
ttft_short = time.perf_counter() - t0
t_first_tok_wall = time.time()
log_sub(f"Short prompt TTFT: {ttft_short:.4f}s; HBM: {get_hbm()}")

sp_64 = SamplingParams(max_tokens=64, min_tokens=64, temperature=0.0, ignore_eos=True)
t0_dec = time.perf_counter()
out_64 = llm.generate([p_short], sp_64)
t_dec = time.perf_counter() - t0_dec
decode_tok_s = 64.0 / max(1e-6, t_dec)
sample_txt = out_64[0].outputs[0].text if out_64 and out_64[0].outputs else ""
log_sub(f"Decode 64 tok: {t_dec:.4f}s ({decode_tok_s:.2f} tok/s); Sample: {repr(sample_txt[:120])}")

# Progressive context sweeps: 32K and 200K
context_evals = {}
tok = llm.get_tokenizer()
all_ids = tok.encode((p_short + " ") * 25000)
for target_ctx in [32768, 200000]:
    p_len = min(len(all_ids), target_ctx - 64)
    p_input = {"prompt_token_ids": all_ids[:p_len]}
    t0_ctx = time.perf_counter()
    llm.generate(p_input, sp_1)
    ttft_ctx = time.perf_counter() - t0_ctx
    t0_cd = time.perf_counter()
    llm.generate(p_input, sp_64)
    t_cd = time.perf_counter() - t0_cd
    c_dec_tok_s = 64.0 / max(1e-6, t_cd)
    hbm_now = get_hbm()
    log_sub(f"Context {target_ctx} (p={p_len}): TTFT={ttft_ctx:.4f}s, Decode={c_dec_tok_s:.2f} tok/s, HBM={hbm_now}")
    context_evals[str(target_ctx)] = {"prompt_len": p_len, "ttft_s": round(ttft_ctx, 4), "decode_tok_s": round(c_dec_tok_s, 2), "hbm": hbm_now}

tar_cache_s = 0.0
try:
    t_tar0 = time.time()
    subprocess.run(["tar", "-cf", "/dev/shm/tpu_compile_cache_lfm25_200k.tar", "-C", "/dev/shm", "tpu_compile_cache"], check=True)
    tar_cache_s = round(time.time() - t_tar0, 2)
    log_sub(f"Packaged compile cache in {tar_cache_s}s")
except Exception as e: log_sub(f"Compile cache tar failed: {e}")

results = {
    "init_s": round(init_dur, 2), "first_token_ts": t_first_tok_wall,
    "ttft_short_s": round(ttft_short, 4), "decode_tok_s": round(decode_tok_s, 2),
    "sample_output": sample_txt[:200], "context_evals": context_evals,
    "hbm_stats": get_hbm(), "tar_cache_s": tar_cache_s
}
with open("/content/engine_results.json", "w") as f: json.dump(results, f, indent=2)
log_sub("Engine benchmarking complete; written to /content/engine_results.json")
