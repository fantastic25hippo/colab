import json, os, time, urllib.error, urllib.request
from collections.abc import Mapping

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
    msg = f"[LFM25_DIAG {time.strftime('%H:%M:%S')}] {m}"
    print(msg, flush=True)
    try:
        with open("/content/build.log", "a") as f: f.write(msg + "\n")
    except Exception: pass

t_imp0 = time.time()
log_sub("Importing jax and vllm for short-context diagnostic (max_model_len=4096)...")
import jax
from vllm import LLM, SamplingParams
log_sub(f"TPU init complete in {time.time() - t_imp0:.2f}s: {jax.devices()}")

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
llm = LLM(
    model="/dev/shm/lfm25_200k", max_model_len=4096, max_num_batched_tokens=4096,
    max_num_seqs=1, enable_prefix_caching=False, trust_remote_code=True
)
init_dur = time.time() - t_eng0
kv_dtype = "not-logged"
try:
    kv_dtype = str(llm.llm_engine.vllm_config.cache_config.cache_dtype)
except Exception as e:
    kv_dtype = f"not-logged ({e})"
log_sub(f"Engine initialized in {init_dur:.2f}s; kv dtype={kv_dtype}; HBM: {get_hbm()}")

PROMPT = "Explain the fundamental principles of astrophysics and stellar evolution."
MSGS = [{"role": "user", "content": PROMPT}]
BOS_ID = 124894
runs = {}

def _ids(x):
    if isinstance(x, Mapping): x = x.get("input_ids", [])
    if hasattr(x, "tolist"): x = x.tolist()
    return [int(i) for i in x]

def report(tag, prompt_ids, out, wall):
    ids = _ids(out[0].prompt_token_ids) if getattr(out[0], "prompt_token_ids", None) else _ids(prompt_ids)
    o = out[0].outputs[0]
    out_ids = _ids(o.token_ids)
    bos = ids.count(BOS_ID)
    log_sub(f"{tag} prompt_len={len(ids)} first40={ids[:40]} bos_count={bos}")
    log_sub(f"{tag} out_len={len(out_ids)} wall={wall:.3f}s out_ids={out_ids}")
    log_sub(f"{tag} text={o.text!r}")
    return {"prompt_len": len(ids), "prompt_ids_first40": ids[:40], "bos_count": bos,
            "out_ids": out_ids, "text": o.text, "wall_s": round(wall, 3)}

def run_chat(tag, sp):
    t0 = time.perf_counter()
    try:
        out = llm.chat(MSGS, sp, add_generation_prompt=True)
        wall = time.perf_counter() - t0
        runs[tag] = report(tag, chat_fallback, out, wall)
    except Exception as e:
        log_sub(f"{tag} FAILED: {type(e).__name__}: {e}")
        runs[tag] = {"error": f"{type(e).__name__}: {e}"}

tok = llm.get_tokenizer()
chat_fallback = []
try:
    chat_fallback = _ids(tok.apply_chat_template(MSGS, tokenize=True, add_generation_prompt=True))
except Exception as e:
    log_sub(f"chat template tokenization failed: {e}")

# Run A: card sampling params (temperature 0.1, top_k 50, rep 1.05), chat template.
run_chat("RUN_A", SamplingParams(temperature=0.1, top_k=50, repetition_penalty=1.05,
                                 max_tokens=128))
t_first_tok_wall = time.time()
# Run B: greedy chat control (temperature 0, no repetition penalty).
run_chat("RUN_B", SamplingParams(temperature=0.0, repetition_penalty=1.0,
                                 max_tokens=64))

# Run C: old raw completion path, greedy, ignore_eos (reproduces the reported bug).
try:
    t0 = time.perf_counter()
    out_c = llm.generate([PROMPT], SamplingParams(temperature=0.0, max_tokens=64,
                                                  ignore_eos=True))
    wall_c = time.perf_counter() - t0
    runs["RUN_C"] = report("RUN_C", tok.encode(PROMPT, add_special_tokens=True), out_c, wall_c)
except Exception as e:
    log_sub(f"RUN_C FAILED: {type(e).__name__}: {e}")
    runs["RUN_C"] = {"error": f"{type(e).__name__}: {e}"}

log_sub(f"RUN_A chat template fallback first40={chat_fallback[:40]} bos_count={chat_fallback.count(BOS_ID)}")

results = {"diag": True, "init_s": round(init_dur, 2), "kv_dtype": kv_dtype,
           "first_token_ts": t_first_tok_wall, "runs": runs, "hbm_stats": get_hbm()}
with open("/content/engine_results.json", "w") as f: json.dump(results, f, indent=2)
log_sub("Short-context diagnostic complete; written to /content/engine_results.json")
