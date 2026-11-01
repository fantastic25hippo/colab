import builtins, json, os, sys, time

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

phase_data = {}
def record_phase(name, t0, dur, **kw):
    phase_data[name] = {"start": round(t0, 2), "dur": round(dur, 4), **kw}
    extra = " " + " ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""
    log_sub(f"PHASE {name} start={t0:.2f} dur={dur:.2f}{extra}")

def _hook_worker():
    tw = sys.modules.get("tpu_inference.worker.tpu_worker")
    cls = getattr(tw, "TPUWorker", None) if tw else None
    if cls and not getattr(cls, "_phase_hooked", False):
        cls._phase_hooked = True
        orig_load = getattr(cls, "load_model", None)
        if callable(orig_load):
            def _w_load(self, *a, **kw):
                t0 = time.time()
                r = orig_load(self, *a, **kw)
                record_phase("weight_load_to_tpu", t0, time.time() - t0)
                return r
            cls.load_model = _w_load
        orig_warmup = getattr(cls, "compile_or_warm_up_model", None)
        if callable(orig_warmup):
            def _w_warmup(self, *a, **kw):
                t0 = time.time()
                cb = len(os.listdir("/dev/shm/tpu_compile_cache"))
                r = orig_warmup(self, *a, **kw)
                ca = len(os.listdir("/dev/shm/tpu_compile_cache"))
                nc = max(0, ca - cb)
                hits = cb if ca >= cb else 0
                record_phase("compile_warmup", t0, time.time() - t0, compiled=nc, hits=hits)
                return r
            cls.compile_or_warm_up_model = _w_warmup

_orig_imp = builtins.__import__
def _phase_imp(name, *args, **kwargs):
    res = _orig_imp(name, *args, **kwargs)
    if "tpu_inference" in name:
        _hook_worker()
    return res
builtins.__import__ = _phase_imp

t_imp0 = time.time()
log_sub("Importing jax and vllm (overlapped with weight download)...")
import jax
from vllm import LLM, SamplingParams
_hook_worker()
tpu_devs = str(jax.devices())
t_import_s = time.time() - t_imp0
record_phase("imports", t_imp0, t_import_s)
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
cache_hits = cache_before if cache_after >= cache_before else 0

if "weight_load_to_tpu" not in phase_data:
    record_phase("weight_load_to_tpu", t_eng0, 0.0)
if "compile_warmup" not in phase_data:
    wl_dur = phase_data.get("weight_load_to_tpu", {}).get("dur", 0.0)
    record_phase("compile_warmup", t_eng0 + wl_dur, max(0.0, init_dur - wl_dur),
                 compiled=new_compiles, hits=cache_hits)

log_sub(f"Engine ready in {init_dur:.2f}s (cache_before={cache_before}, cache_after={cache_after}, new_compiles={new_compiles})")

p_short = "Deep space exploration reveals clues about cosmic microwave background and spacetime."
sp_1 = SamplingParams(max_tokens=1, min_tokens=1, temperature=0.0, ignore_eos=True)
t0 = time.perf_counter()
t_ttft_wall = time.time()
llm.generate([p_short], sp_1)
ttft_short = time.perf_counter() - t0
t_first_tok_wall = time.time()
record_phase("first_token", t_ttft_wall, ttft_short)
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
    "import_and_tpu_init_s": round(t_import_s, 2), "import_start": round(t_imp0, 2),
    "wait_for_weights_s": round(t_wait_weights_s, 2),
    "engine_init_s": round(init_dur, 2), "engine_start": round(t_eng0, 2),
    "cache_entries_before": cache_before,
    "cache_entries_after": cache_after, "new_compiles": new_compiles,
    "ttft_short_s": round(ttft_short, 4), "first_token_ts": t_first_tok_wall,
    "decode_runs": d_runs, "mean_decode_tok_s": mean_tok_s,
    "ctx_long": {"tokens": p30k_tokens, "prefill_s": round(prefill_30k, 2),
                 "decode_tok_s": dec_tps_30k, "wall_s": round(wall_30k, 2)},
    "hbm": hbm,
    "phases": phase_data
}
with open("/content/engine_results.json", "w") as f: json.dump(res, f, indent=2)
log_sub("=== SUBRUNNER_SUCCESS ===")
