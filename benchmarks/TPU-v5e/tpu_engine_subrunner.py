import builtins, json, os, sys, time

orig_imp = builtins.__import__
def custom_imp(name, *args, **kwargs):
    res = orig_imp(name, *args, **kwargs)
    mod = sys.modules.get("tpu_inference.models.jax.utils.weight_utils")
    if mod and hasattr(mod, "jax"): mod.jax.clear_caches = lambda: None
    return res
builtins.__import__ = custom_imp

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_XLA_CACHE_PATH"] = "/dev/shm/tpu_compile_cache"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/dev/shm/tpu_compile_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
os.makedirs("/dev/shm/tpu_compile_cache", exist_ok=True)

import jax
from vllm import LLM, SamplingParams

def get_hbm():
    try:
        m = jax.devices()[0].memory_stats()
        return {"in_use_gib": round(m.get("bytes_in_use", 0)/(1024**3), 3),
                "peak_gib": round(m.get("peak_bytes_in_use", 0)/(1024**3), 3),
                "limit_gib": round(m.get("bytes_limit", 0)/(1024**3), 3)}
    except Exception as e: return {"error": str(e)}

t_eng0 = time.time()
llm = LLM(model="/dev/shm/qwen3", max_model_len=32768, max_num_seqs=1,
          gpu_memory_utilization=0.90, trust_remote_code=True)
t_ready = time.time()
init_dur = t_ready - t_eng0
compiles = len(os.listdir("/dev/shm/tpu_compile_cache"))
print(f"[SUBRUNNER] Ready in {init_dur:.2f}s, compile_count={compiles}", flush=True)

p_short = "The exploration of deep space has fascinated humanity for centuries. Astronomers observe distant stars and galaxies through powerful telescopes, searching for clues about cosmic microwave background and spacetime dimensions."
sp_1 = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0, ignore_eos=True)
t0 = time.perf_counter()
llm.generate([p_short], sp_1)
ttft_short = time.perf_counter() - t0
t_first_tok_wall = time.time()
print(f"[SUBRUNNER] Short prompt TTFT: {ttft_short:.4f}s", flush=True)

sp_256 = SamplingParams(max_tokens=256, min_tokens=256, temperature=0.0, ignore_eos=True)
llm.generate([p_short], sp_256) # warmup

d_runs = []
for i in range(3):
    t_start = time.perf_counter()
    out = llm.generate([p_short], sp_256)[0]
    dur = time.perf_counter() - t_start
    toks = len(out.outputs[0].token_ids)
    tps = round((toks - 1) / max(0.001, dur - ttft_short), 2)
    d_runs.append({"tokens": toks, "wall_s": round(dur, 4), "tok_s": tps})
    print(f"[SUBRUNNER] Decode {i+1}: {toks} tok in {dur:.2f}s ({tps} tok/s)", flush=True)
mean_tok_s = round(sum(r["tok_s"] for r in d_runs) / len(d_runs), 2)

tok = llm.get_tokenizer()
chunk = "Artificial intelligence and distributed computing continue to revolutionize modern science. "
c_len = len(tok.encode(chunk))
prompt_30k = chunk * int(30000 / c_len)
p30k_tokens = len(tok.encode(prompt_30k))
print(f"[SUBRUNNER] 30K prompt encoded: {p30k_tokens} tokens", flush=True)

sp_32 = SamplingParams(max_tokens=32, min_tokens=32, temperature=0.0, ignore_eos=True)
t0_30k = time.perf_counter()
llm.generate([prompt_30k], sp_32)
wall_30k = time.perf_counter() - t0_30k

t0_p1 = time.perf_counter()
llm.generate([prompt_30k], sp_1)
prefill_30k = time.perf_counter() - t0_p1
dec_dur_30k = max(0.001, wall_30k - prefill_30k)
dec_tps_30k = round(31 / dec_dur_30k, 2)
hbm = get_hbm()
print(f"[SUBRUNNER] 30K: prefill={prefill_30k:.2f}s, decode={dec_tps_30k} tok/s, wall={wall_30k:.2f}s, HBM={hbm}", flush=True)

res = {
    "engine_init_s": round(init_dur, 2), "compile_count": compiles,
    "ttft_short_s": round(ttft_short, 4), "first_token_ts": t_first_tok_wall,
    "decode_runs": d_runs, "mean_decode_tok_s": mean_tok_s,
    "ctx_30k": {"tokens": p30k_tokens, "prefill_s": round(prefill_30k, 2),
                "decode_tok_s": dec_tps_30k, "wall_s": round(wall_30k, 2)},
    "hbm": hbm
}
with open("/content/engine_results.json", "w") as f: json.dump(res, f, indent=2)
print("=== SUBRUNNER_SUCCESS ===", flush=True)
