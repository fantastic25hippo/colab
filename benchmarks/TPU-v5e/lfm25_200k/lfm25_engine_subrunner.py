import json, os, re, sys, time, urllib.error, urllib.request

_orig_urlopen = urllib.request.urlopen
def _fast_urlopen(url, *args, **kwargs):
    u_str = url if isinstance(url, str) else getattr(url, "full_url", str(url))
    if "metadata.google.internal" in u_str: raise urllib.error.URLError("Bypassed GCE metadata poll")
    return _orig_urlopen(url, *args, **kwargs)
urllib.request.urlopen = _fast_urlopen

import tpu_v5e_patches as patches
try:
    patches.install_all_patches()
except Exception: pass

os.environ.update({
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_XLA_CACHE_PATH": "/dev/shm/tpu_compile_cache",
    "JAX_COMPILATION_CACHE_DIR": "/dev/shm/tpu_compile_cache",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
    "SKIP_JAX_PRECOMPILE": "1",
})
os.makedirs("/dev/shm/tpu_compile_cache", exist_ok=True)

from lfm25_prompt import BOS_ID, build_long_prompt_ids, check_prompt_ids, filler_text
from lfm25_startup import PHASE, install_log_timeline, summarize_timeline
from lfm25_cache import dir_stats

# Any change here changes the XLA program set; the driver hashes this line into
# the compile-cache filename. kv=auto is vLLM's default (bf16 KV).
CACHE_KEY_ARGS = ("max_model_len=200000;max_num_seqs=1;max_num_batched_tokens=8192;"
                  "gpu_memory_utilization=0.92;prefix_caching=0;kv=auto;buckets=16|8192")

def log_sub(m):
    msg = f"[LFM25_SUBRUNNER {time.strftime('%H:%M:%S')}] {m}"
    print(msg, flush=True)
    try:
        with open("/content/build.log", "a") as f: f.write(msg + "\n")
    except Exception: pass

def get_hbm():
    try:
        m = jax.devices()[0].memory_stats()
        return {"in_use_gib": round(m.get("bytes_in_use", 0)/(1024**3), 3),
                "peak_gib": round(m.get("peak_bytes_in_use", 0)/(1024**3), 3),
                "limit_gib": round(m.get("bytes_limit", 0)/(1024**3), 3)}
    except Exception as e: return {"error": str(e)}

def write_results(res):
    with open("/content/engine_results.json", "w") as f: json.dump(res, f, indent=2)
    log_sub("Wrote /content/engine_results.json")

t_imp0 = time.time()
log_sub("Importing jax and vllm for LFM2.5-24B-A2B (200K context)...")
import jax
from vllm import LLM, SamplingParams
t_import_s = time.time() - t_imp0
log_sub(f"TPU init complete in {t_import_s:.2f}s: {jax.devices()}")

t_wait0 = time.time()
while not os.path.exists("/dev/shm/.weights_ready"):
    if time.time() - t_wait0 > 600:
        log_sub("Weights barrier timed out after 600s; continuing (LLM() will report)")
        break
    time.sleep(0.05)
t_wait_s = time.time() - t_wait0
log_sub(f"Weights barrier cleared after {t_wait_s:.2f}s wait; cache={dir_stats()}")

cache_before = dir_stats()
timeline = install_log_timeline()
t_eng0 = time.perf_counter()
err = None
try:
    # 2026-11-01: kv_cache_dtype left at the default ("auto" -> bf16). fp8 is NOT re-enabled.
    llm = LLM(model="/dev/shm/lfm25_200k", max_model_len=200000, max_num_seqs=1,
              max_num_batched_tokens=8192, gpu_memory_utilization=0.92,
              enable_prefix_caching=False, trust_remote_code=True)
except Exception as e:
    err = f"{type(e).__name__}: {e}"
init_dur = time.perf_counter() - t_eng0
cache_after = dir_stats()

spans = PHASE["spans"]
def _span(name):
    return next((s["dur_s"] for s in reversed(spans) if s["name"] == name), None)
warm = [s for s in spans if s["name"].startswith("warmup[")]
per_bucket = {}
for s in warm:
    m = re.search(r"num_tokens=(\d+)", s["name"])
    if m: per_bucket.setdefault(m.group(1), []).append(s["dur_s"])
phases = {
    "import_and_tpu_init_s": round(t_import_s, 2), "wait_for_weights_s": round(t_wait_s, 2),
    "llm_init_s": round(init_dur, 2),
    "weight_load_s": _span("weight_load"), "kv_alloc_s": _span("kv_alloc"),
    "compile_warmup_s": _span("compile_warmup"),
    "warmup_pass_total_s": None if not warm else round(sum(s["dur_s"] for s in warm), 2),
    "warmup_per_bucket_s": {k: round(sum(v), 2) for k, v in per_bucket.items()},
    "warmup_tasks": [{"name": s["name"], "dur_s": s["dur_s"]} for s in warm],
    "timeline": summarize_timeline(timeline.rows), "log_lines_seen": timeline.seen,
}
if phases["weight_load_s"] is not None and phases["compile_warmup_s"] is not None:
    phases["llm_init_unaccounted_s"] = round(init_dur - phases["weight_load_s"]
                                             - (phases["kv_alloc_s"] or 0) - phases["compile_warmup_s"], 2)

base = {"ok": err is None, "init_s": round(init_dur, 2), "phases": phases,
        "engine_args": {"max_model_len": 200000, "max_num_seqs": 1, "max_num_batched_tokens": 8192,
                        "gpu_memory_utilization": 0.92, "enable_prefix_caching": False,
                        "kv_cache_dtype": "auto (vLLM default, bf16)", "buckets": [16, 8192]},
        "cache": {"before": cache_before, "after": cache_after,
                  "new_entries": cache_after["entries"] - cache_before["entries"]},
        "hbm_after_init": get_hbm()}
if err is None:
    try: base["kv_cache_dtype_resolved"] = str(llm.llm_engine.vllm_config.cache_config.cache_dtype)
    except Exception as e: base["kv_cache_dtype_resolved"] = f"not-logged ({type(e).__name__})"
if err:
    log_sub(f"LLM() failed after {init_dur:.2f}s: {err}")
    write_results(dict(base, error=err, log_timeline=timeline.rows))
    sys.exit(0)
log_sub(f"Engine ready in {init_dur:.2f}s; weight_load={phases['weight_load_s']}s "
        f"kv_alloc={phases['kv_alloc_s']}s compile_warmup={phases['compile_warmup_s']}s; "
        f"kv_dtype={base['kv_cache_dtype_resolved']}; HBM={get_hbm()}")

p_short = "Explain the fundamental principles of astrophysics and stellar evolution."
msgs = [{"role": "user", "content": p_short}]
sp_ttft = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
sp_chat = SamplingParams(max_tokens=64, temperature=0.0)          # no ignore_eos/min_tokens/seed
sp_dec = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)

def _ids(x):
    if hasattr(x, "tolist"): x = x.tolist()
    return [int(i) for i in x]

def _out_ids(out):
    try: return _ids(out[0].outputs[0].token_ids)
    except Exception: return []

def _prompt_ids(out, fallback):
    try:
        got = getattr(out[0], "prompt_token_ids", None)
        return _ids(got) if got else list(fallback)
    except Exception:
        return list(fallback)

short, chat_txt, decode_tok_s, first_tok_ts = {}, "", None, None
context_evals = {}
try:
    t0 = time.perf_counter()
    o = llm.chat(msgs, sp_ttft, add_generation_prompt=True)
    ttft_short = time.perf_counter() - t0
    first_tok_ts = time.time()
    short["ttft_s"] = round(ttft_short, 4)
    _pids = _prompt_ids(o, [])
    short["ttft_prompt_len"] = len(_pids)
    short["ttft_bos"] = check_prompt_ids(_pids) if _pids else None
    log_sub(f"Short TTFT (chat, {short['ttft_prompt_len']} ids): {ttft_short:.4f}s")
except Exception as e:
    short["ttft_error"] = f"{type(e).__name__}: {e}"
    log_sub(f"Short TTFT FAILED: {short['ttft_error']}")

GOLDEN_B_PREFIX = [597, 4695, 20589, 34, 496, 89509, 278, 9761, 9896, 302, 5030, 37933, 1003]
golden_ok = True
try:
    t0 = time.perf_counter()
    chat_out = llm.chat(msgs, sp_chat, add_generation_prompt=True)
    chat_wall = time.perf_counter() - t0
    chat_txt = chat_out[0].outputs[0].text
    chat_ids_out = _out_ids(chat_out)
    _cids = _prompt_ids(chat_out, [])
    short["chat"] = {"wall_s": round(chat_wall, 4), "prompt_len": len(_cids),
                     "prompt_bos": check_prompt_ids(_cids) if _cids else None,
                     "out_ids": chat_ids_out, "text": chat_txt, "ignore_eos": False}
    log_sub(f"Chat 64 tok: {chat_wall:.3f}s; ids={chat_ids_out}; text={chat_txt!r}")
    if len(chat_ids_out) < len(GOLDEN_B_PREFIX):
        log_sub(f"Golden mismatch: output length {len(chat_ids_out)} < {len(GOLDEN_B_PREFIX)}")
        golden_ok = False
    else:
        for idx_g, (got_id, want_id) in enumerate(zip(chat_ids_out, GOLDEN_B_PREFIX)):
            if got_id != want_id:
                log_sub(f"Golden mismatch at idx={idx_g}: got={got_id} want={want_id}")
                golden_ok = False
                break
    if not golden_ok:
        base["ok"] = False
except Exception as e:
    short["chat"] = {"error": f"{type(e).__name__}: {e}"}
    log_sub(f"Chat 64 tok FAILED: {short['chat']['error']}")
    golden_ok = False
    base["ok"] = False

if not golden_ok:
    log_sub("Golden gate failed; skipping remaining evals")
else:
    try:
        t0 = time.perf_counter()
        dec_out = llm.chat(msgs, sp_dec, add_generation_prompt=True)
        dec_wall = time.perf_counter() - t0
        decode_tok_s = 64.0 / max(1e-6, dec_wall)
        short["decode"] = {"wall_s": round(dec_wall, 4), "tok_s": round(decode_tok_s, 2),
                           "tok_s_excl_prefill": round(63.0 / max(1e-6, dec_wall - short.get("ttft_s", 0)), 2),
                           "n_out": len(_out_ids(dec_out)), "ignore_eos": True,
                           "note": "ignore_eos=True: throughput timing only, text not judged; "
                                   "tok_s=64/wall, tok_s_excl_prefill=63/(wall-ttft)"}
        log_sub(f"Decode 64 tok (ignore_eos): {dec_wall:.4f}s ({decode_tok_s:.2f} tok/s)")
    except Exception as e:
        short["decode"] = {"error": f"{type(e).__name__}: {e}"}
        log_sub(f"Decode 64 tok FAILED: {short['decode']['error']}")

    context_evals = {}
    tok = llm.get_tokenizer()
    filler = filler_text(tok, p_short + " ", 200000)
    for target_ctx in [32768, 200000]:
        target = target_ctx - 64
        try:
            ids, meta = build_long_prompt_ids(tok, filler, target)
            bos = check_prompt_ids(ids)
            p_input = {"prompt_token_ids": ids}
            t0 = time.perf_counter()
            o_ctx = llm.generate(p_input, sp_ttft)
            ttft_ctx = time.perf_counter() - t0
            t0 = time.perf_counter()
            o_dec = llm.generate(p_input, sp_dec)
            t_dec = time.perf_counter() - t0
            hbm_now = get_hbm()
            echo = _prompt_ids(o_ctx, ids)
            log_sub(f"Context {target_ctx} (p={len(ids)}, bos={bos}, echo_bos={echo.count(BOS_ID)}, "
                    f"decode ignore_eos=True): TTFT={ttft_ctx:.4f}s, "
                    f"Decode={64.0/max(1e-6, t_dec):.2f} tok/s, HBM={hbm_now}")
            context_evals[str(target_ctx)] = {
                "prompt_len": len(ids), "ttft_s": round(ttft_ctx, 4),
                "decode_tok_s": round(64.0/max(1e-6, t_dec), 2), "decode_wall_s": round(t_dec, 4),
                "decode_tok_s_excl_prefill": round(63.0/max(1e-6, t_dec - ttft_ctx), 2),
                "prompt_bos_count": bos, "engine_prompt_bos_count": echo.count(BOS_ID),
                "prompt_ids_first40": ids[:40], "n_out_decode": len(_out_ids(o_dec)),
                "build": meta, "hbm": hbm_now}
        except Exception as e:
            context_evals[str(target_ctx)] = {"prompt_len": target,
                                              "error": f"{type(e).__name__}: {e}", "hbm": get_hbm()}
            log_sub(f"Context {target_ctx} FAILED: {context_evals[str(target_ctx)]['error']}")

results = dict(base, first_token_ts=first_tok_ts, ttft_short_s=short.get("ttft_s"),
               decode_tok_s=None if decode_tok_s is None else round(decode_tok_s, 2),
               sample_output=chat_txt[:400],
               short=short, context_evals=context_evals, hbm_stats=get_hbm(),
               log_timeline=timeline.rows, notes=[
                   "all prompts use the tokenizer chat template (add_generation_prompt=True)",
                   "short chat prompt is 22 ids -> pads to the 8192 bucket; the 2026-11-01 18:06 run used the raw 12-id prompt (bucket 16)",
                   "chat coherence run: greedy, max_tokens=64, no ignore_eos/min_tokens/seed",
                   "64-token decode runs set ignore_eos=True: throughput timing only, text not judged",
                   "kv_cache_dtype is vLLM's default (auto); fp8 is not enabled"])
write_results(results)
log_sub("Engine benchmarking complete")
