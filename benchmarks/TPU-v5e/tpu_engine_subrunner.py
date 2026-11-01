import json, os, sys, time

sys.path.insert(0, "/content")
try:
    import qwen_patches
    qwen_patches.install_qwen_patches()
except Exception: pass

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_XLA_CACHE_PATH"] = "/dev/shm/tpu_compile_cache"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/dev/shm/tpu_compile_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
os.makedirs("/dev/shm/tpu_compile_cache", exist_ok=True)

def log_sub(m):
    msg = f"[SUBRUNNER {time.strftime('%H:%M:%S')}] {m}"
    print(msg, flush=True)
    try:
        with open("/content/build.log", "a") as f: f.write(msg + "\n")
    except Exception: pass

t_imp0 = time.time()
log_sub("Importing jax and vllm (overlapped with weight download)...")
import jax
from vllm import LLM, SamplingParams
tpu_devs = str(jax.devices())
t_import_s = time.time() - t_imp0
log_sub(f"Imports + TPU init done in {t_import_s:.2f}s: {tpu_devs}")

t_wait0 = time.time()
while not os.path.exists("/dev/shm/.weights_ready"):
    time.sleep(0.05)
t_wait_weights_s = time.time() - t_wait0
log_sub(f"Weights ready barrier passed (waited {t_wait_weights_s:.2f}s after imports)")

def get_hbm():
    try:
        m = jax.devices()[0].memory_stats()
        return {"in_use_gib": round(m.get("bytes_in_use", 0)/(1024**3), 3),
                "peak_gib": round(m.get("peak_bytes_in_use", 0)/(1024**3), 3),
                "limit_gib": round(m.get("bytes_limit", 0)/(1024**3), 3)}
    except Exception as e: return {"error": str(e)}

cache_before = len(os.listdir("/dev/shm/tpu_compile_cache"))
t_eng0 = time.time()
llm = LLM(model="/dev/shm/qwen3", max_model_len=32768, max_num_seqs=1,
          gpu_memory_utilization=0.90, trust_remote_code=True)
t_ready = time.time()
init_dur = t_ready - t_eng0
cache_after = len(os.listdir("/dev/shm/tpu_compile_cache"))
new_compiles = max(0, cache_after - cache_before)
log_sub(f"Engine ready in {init_dur:.2f}s (cache_before={cache_before}, cache_after={cache_after}, new_compiles={new_compiles})")

p_short = "Deep space exploration reveals clues about cosmic microwave background and spacetime."
sp_1 = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0, ignore_eos=True)
t0 = time.perf_counter()
llm.generate([p_short], sp_1)
ttft_short = time.perf_counter() - t0
t_first_tok_wall = time.time()
log_sub(f"Short prompt TTFT: {ttft_short:.4f}s")

sp_256 = SamplingParams(max_tokens=256, min_tokens=256, temperature=0.0, ignore_eos=True)
llm.generate([p_short], sp_256)

d_runs = []
for i in range(3):
    t_start = time.perf_counter()
    out = llm.generate([p_short], sp_256)[0]
    dur = time.perf_counter() - t_start
    toks = len(out.outputs[0].token_ids)
    tps = round((toks - 1) / max(0.001, dur - ttft_short), 2)
    d_runs.append({"tokens": toks, "wall_s": round(dur, 4), "tok_s": tps})
    log_sub(f"Decode {i+1}: {toks} tok in {dur:.2f}s ({tps} tok/s)")
mean_tok_s = round(sum(r["tok_s"] for r in d_runs) / len(d_runs), 2)

tok = llm.get_tokenizer()
chunk_a = "Artificial intelligence and distributed computing continue to revolutionize modern science. "
chunk_b = "Quantum mechanics and general relativity form the twin pillars of modern theoretical physics. "
prompt_30k_a = chunk_a * int(26000 / len(tok.encode(chunk_a)))
prompt_30k_b = chunk_b * int(26000 / len(tok.encode(chunk_b)))
p30k_tokens = len(tok.encode(prompt_30k_a))

t0_p1 = time.perf_counter()
llm.generate([prompt_30k_a], sp_1)
prefill_30k = time.perf_counter() - t0_p1

sp_32 = SamplingParams(max_tokens=32, min_tokens=32, temperature=0.0, ignore_eos=True)
t0_30k = time.perf_counter()
llm.generate([prompt_30k_b], sp_32)
wall_30k = time.perf_counter() - t0_30k
dec_tps_30k = round(31 / max(0.001, wall_30k - prefill_30k), 2)
hbm = get_hbm()
log_sub(f"Long ctx ({p30k_tokens} tok): prefill={prefill_30k:.2f}s, decode={dec_tps_30k} tok/s, wall={wall_30k:.2f}s, HBM={hbm}")

res = {
    "import_and_tpu_init_s": round(t_import_s, 2), "wait_for_weights_s": round(t_wait_weights_s, 2),
    "engine_init_s": round(init_dur, 2), "cache_entries_before": cache_before,
    "cache_entries_after": cache_after, "new_compiles": new_compiles,
    "ttft_short_s": round(ttft_short, 4), "first_token_ts": t_first_tok_wall,
    "decode_runs": d_runs, "mean_decode_tok_s": mean_tok_s,
    "ctx_long": {"tokens": p30k_tokens, "prefill_s": round(prefill_30k, 2),
                 "decode_tok_s": dec_tps_30k, "wall_s": round(wall_30k, 2)},
    "hbm": hbm
}
with open("/content/engine_results.json", "w") as f: json.dump(res, f, indent=2)
log_sub("=== SUBRUNNER_SUCCESS ===")
