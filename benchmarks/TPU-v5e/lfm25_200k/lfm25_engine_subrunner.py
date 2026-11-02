import json, logging, os, re, sys, time, urllib.error, urllib.request

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

for k, v in {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_XLA_CACHE_PATH": "/dev/shm/tpu_compile_cache",
    "JAX_COMPILATION_CACHE_DIR": "/dev/shm/tpu_compile_cache",
    "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
    "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1",
    "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
    "SKIP_JAX_PRECOMPILE": "1",
}.items(): os.environ.setdefault(k, v)
os.makedirs(os.environ["JAX_COMPILATION_CACHE_DIR"], exist_ok=True)

from lfm25_prompt import BOS_ID, build_long_prompt_ids, check_prompt_ids, filler_text
from lfm25_startup import PHASE, install_log_timeline, summarize_timeline
from lfm25_cache import dir_stats

cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.environ.get("VLLM_XLA_CACHE_PATH") or "/dev/shm/tpu_compile_cache"
max_model_len = int(os.environ.get("LFM_MAX_MODEL_LEN") or os.environ.get("MAX_MODEL_LEN") or sys.exit("Error: LFM_MAX_MODEL_LEN or MAX_MODEL_LEN must be set"))
ctx_k = f"{max_model_len // 1000}K"

# Any change here changes the XLA program set; the driver hashes this line into
# the compile-cache filename. kv=auto is vLLM's default (bf16 KV).
CACHE_KEY_ARGS = ("max_model_len=100000;max_num_seqs=1;max_num_batched_tokens=8192;"
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

_diag_on = os.environ.get("LFM_DIAG") == "1" or os.environ.get("LFM_JOB9_DIAG") == "1"
_diag_stop, _diag_meta = None, None
if _diag_on:
    import faulthandler, shutil, subprocess, threading
    _t_g0, _sdir = float(os.environ.get("T0_GLOBAL") or time.time()), "/content/job9_stacks"; os.makedirs(_sdir, exist_ok=True)
    _ps = shutil.which("py-spy") or ("/dev/shm/vllm_env/bin/py-spy" if os.path.exists("/dev/shm/vllm_env/bin/py-spy") else None)
    _d_meth = f"py-spy ({subprocess.run([_ps, '--version'], capture_output=True, text=True).stdout.strip()})" if _ps else "faulthandler.dump_traceback"
    _diag_stop, _diag_meta = threading.Event(), {"method": _d_meth, "t_global0": _t_g0, "samples": 0}; log_sub(f"[LFM_DIAG] sampler={_d_meth} t_global0={_t_g0:.2f}")
    def _sampler():
        while not _diag_stop.is_set():
            rel = time.time() - _t_g0
            try:
                with open(f"{_sdir}/{rel:07.2f}.txt", "w") as f:
                    f.write(f"# rel_s={rel:.3f} ts={time.time():.3f} pid={os.getpid()} method={_d_meth}\n"); f.flush()
                    subprocess.run([_ps, "dump", "--pid", str(os.getpid())], stdout=f, stderr=subprocess.STDOUT, timeout=0.9) if _ps else faulthandler.dump_traceback(file=f, all_threads=True)
                _diag_meta["samples"] += 1
            except Exception: pass
            _diag_stop.wait(1.0)
    threading.Thread(target=_sampler, daemon=True).start()

t_imp0 = time.time()
log_sub(f"Importing jax and vllm for LFM2.5-24B-A2B ({ctx_k} context)...")
t0 = time.time(); import jax; log_sub(f"import jax: {time.time() - t0:.2f}s")
t0 = time.time(); from vllm import LLM, SamplingParams; log_sub(f"import vllm: {time.time() - t0:.2f}s")
_wf_meta = {}
try:
    import vllm.model_executor.models.registry as _vreg
    _orig_run_sub = _vreg._run_in_subprocess
    def _timed_run_sub(*a, **kw):
        t_sub0 = time.perf_counter()
        try: return _orig_run_sub(*a, **kw)
        finally: _wf_meta["model_inspect_s"] = round(time.perf_counter() - t_sub0, 4)
    _vreg._run_in_subprocess = _timed_run_sub
except Exception: pass
try:
    import transformers.models.auto.tokenization_auto as _ta
    _orig_from_pretrained = _ta.AutoTokenizer.from_pretrained
    def _timed_tok_pretrained(*a, **kw):
        t_tok0 = time.perf_counter()
        try: return _orig_from_pretrained(*a, **kw)
        finally: _wf_meta["tokenizer_s"] = round(time.perf_counter() - t_tok0, 4)
    _ta.AutoTokenizer.from_pretrained = _timed_tok_pretrained
except Exception: pass
t_import_s = time.time() - t_imp0
log_sub(f"TPU init complete in {t_import_s:.2f}s: {jax.devices()}")

t_wait0 = time.time()
while not os.path.exists("/dev/shm/.weights_ready"):
    if time.time() - t_wait0 > 600:
        log_sub("Weights barrier timed out after 600s; continuing (LLM() will report)")
        break
    time.sleep(0.05)
t_wait_s = time.time() - t_wait0
log_sub(f"Weights barrier cleared after {t_wait_s:.2f}s wait; cache={dir_stats(cache_dir)}")

cache_before = dir_stats(cache_dir)
timeline = install_log_timeline()
t_cfg0 = time.time()
llm_kw = dict(model="/dev/shm/lfm25_200k", max_model_len=max_model_len, max_num_seqs=1,
              max_num_batched_tokens=8192, gpu_memory_utilization=0.92,
              enable_prefix_caching=False, trust_remote_code=True)
log_sub(f"LLM() arg/config setup in {time.time() - t_cfg0:.4f}s")
t_eng0 = time.perf_counter()
err = None
config_setup_s = None
class _CfgHandler(logging.Handler):
    def emit(self, record):
        try:
            global config_setup_s
            if config_setup_s is None and "Initializing a V1 LLM engine" in record.getMessage():
                config_setup_s = round(time.perf_counter() - t_eng0, 2)
                log_sub(f"LLM() config setup completed in {config_setup_s:.2f}s")
        except Exception:
            pass
_cfgh = _CfgHandler(level=logging.INFO)
_loggers = [logging.getLogger(), logging.getLogger("vllm"), logging.getLogger("vllm.v1.engine.core")]
for _lg in _loggers: _lg.addHandler(_cfgh)
import vllm.engine.arg_utils as _au
_orig_cec = _au.EngineArgs.create_engine_config
def _wf_cec(self, *a, **kw):
    t0 = time.perf_counter()
    try: return _orig_cec(self, *a, **kw)
    finally: _wf_meta["create_engine_config_s"] = round(time.perf_counter() - t0, 4)
_au.EngineArgs.create_engine_config = _wf_cec
if _diag_on:
    import vllm.v1.engine.core as _ec
    _orig_eci = _ec.EngineCore.__init__
    def _d_eci(self, *a, **kw):
        global config_setup_s
        if config_setup_s is None: config_setup_s = round(time.perf_counter() - t_eng0, 2); log_sub(f"[LFM_DIAG] EngineCore.__init__ reached: config_setup_s={config_setup_s:.2f}s (t_rel={time.time()-_t_g0:.2f}s)")
        return _orig_eci(self, *a, **kw)
    _ec.EngineCore.__init__ = _d_eci
try:
    # 2026-11-01: kv_cache_dtype left at the default ("auto" -> bf16). fp8 is NOT re-enabled.
    if os.environ.get("LFM_PROFILE_LOAD") == "1":
        import cProfile, pstats
        pr = cProfile.Profile(); pr.enable()
        try:
            llm = LLM(**llm_kw)
        finally:
            pr.disable(); pr.dump_stats("/content/lfm_load.prof")
            s = pstats.Stats(pr); s.sort_stats("cumulative").print_stats(40); s.sort_stats("tottime").print_stats(40)
    else:
        llm = LLM(**llm_kw)
except Exception as e:
    err = f"{type(e).__name__}: {e}"
finally:
    for _lg in _loggers: _lg.removeHandler(_cfgh)
    _au.EngineArgs.create_engine_config = _orig_cec
    if _diag_on: _ec.EngineCore.__init__ = _orig_eci
    if _diag_meta is not None and "create_engine_config_s" in _wf_meta:
        _diag_meta["create_engine_config_s"] = _wf_meta["create_engine_config_s"]
init_dur = time.perf_counter() - t_eng0
cache_after = dir_stats(cache_dir)

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
    "config_setup_s": config_setup_s, "llm_init_s": round(init_dur, 2),
    "weight_load_s": _span("weight_load"), "kv_alloc_s": _span("kv_alloc"),
    "compile_warmup_s": _span("compile_warmup"),
    "warmup_pass_total_s": None if not warm else round(sum(s["dur_s"] for s in warm), 2),
    "warmup_per_bucket_s": {k: round(sum(v), 2) for k, v in per_bucket.items()},
    "warmup_tasks": [{"name": s["name"], "dur_s": s["dur_s"]} for s in warm],
    "timeline": summarize_timeline(timeline.rows), "log_lines_seen": timeline.seen,
}
if phases["weight_load_s"] is not None and phases["compile_warmup_s"] is not None:
    phases["llm_init_unaccounted_s"] = round(init_dur - phases["weight_load_s"]
                                             - (phases["kv_alloc_s"] or 0) - phases["compile_warmup_s"]
                                             - (config_setup_s or 0) - (_span("profile_run") or 0), 2)

base = {"ok": err is None, "init_s": round(init_dur, 2), "phases": phases,
        "engine_args": {"max_model_len": max_model_len, "max_num_seqs": 1, "max_num_batched_tokens": 8192,
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

def _emit_waterfall(res, first_tok_ts, ttft_s):
    t_vm0 = t_imp0
    total_vm_s = max(0.0, (first_tok_ts or time.time()) - t_vm0)
    rows = []
    def _add(name, t_start, dur):
        if dur is not None and dur > 0 and t_start is not None:
            s_rel = max(0.0, t_start - t_vm0)
            rows.append({"name": name, "start_rel": round(s_rel, 3), "dur_s": round(dur, 3), "end_rel": round(s_rel + dur, 3)})

    _add("imports_and_tpu_init", t_imp0, t_import_s)
    if t_wait_s > 0.05: _add("wait_for_weights", t_wait0, t_wait_s)
    if config_setup_s is not None: _add("config_setup", t_cfg0, config_setup_s)
    for k in ["create_engine_config_s", "model_inspect_s", "tokenizer_s"]:
        if _wf_meta.get(k): _add(k.replace("_s", ""), t_cfg0, _wf_meta[k])

    for s in spans:
        s_ts = s.get("ts") or (t_cfg0 if s.get("start_rel") is None else (t_cfg0 + s["start_rel"]))
        _add(s["name"], s_ts, s.get("dur_s"))

    if ttft_s is not None:
        ttft_start = (first_tok_ts - ttft_s) if first_tok_ts else time.time()
        _add("ttft", ttft_start, ttft_s)

    rows.sort(key=lambda r: (r["start_rel"], r["end_rel"]))
    intervals = []
    for r in rows:
        st, en = r["start_rel"], r["end_rel"]
        if not intervals or st > intervals[-1][1]:
            intervals.append([st, en])
        else:
            intervals[-1][1] = max(intervals[-1][1], en)
    accounted_s = round(sum(en - st for st, en in intervals), 3)
    unaccounted_s = round(max(0.0, total_vm_s - accounted_s), 3)

    gaps = []
    last_end = 0.0
    for st, en in intervals:
        gap = round(st - last_end, 3)
        if gap > 0.05:
            gaps.append({"after": last_end, "before": st, "gap_s": gap})
        last_end = max(last_end, en)
    if total_vm_s - last_end > 0.05:
        gaps.append({"after": last_end, "before": round(total_vm_s, 3), "gap_s": round(total_vm_s - last_end, 3)})

    log_sub(f"[WATERFALL] total_vm_s={total_vm_s:.2f}s accounted_s={accounted_s:.2f}s unaccounted_s={unaccounted_s:.2f}s")
    for r in rows:
        log_sub(f"  {r['start_rel']:7.2f}s - {r['end_rel']:7.2f}s ({r['dur_s']:6.2f}s): {r['name']}")

    res["waterfall"] = rows
    res["t_vm0"] = t_vm0
    res["accounted_s"] = accounted_s
    res["unaccounted_s"] = unaccounted_s
    res["unaccounted_gaps"] = gaps
    return res

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

if os.environ.get("LFM_SHORT_REQUEST") == "1":
    sp16 = SamplingParams(max_tokens=16, temperature=0.0)
    t0 = time.perf_counter()
    out16 = llm.chat(msgs, sp16, add_generation_prompt=True)
    lat_s = time.perf_counter() - t0
    first_tok_ts = time.time()
    m = getattr(out16[0], "metrics", None)
    ttft_s = None
    if m:
        ft = getattr(m, "first_token_time", None)
        arr = getattr(m, "arrival_time", None)
        if ft and arr: ttft_s = ft - arr
        elif getattr(m, "time_to_first_token", None): ttft_s = m.time_to_first_token
    if ttft_s is None: ttft_s = lat_s
    txt = out16[0].outputs[0].text if out16 and out16[0].outputs else ""
    tok_ids = getattr(out16[0].outputs[0], "token_ids", [])
    if _diag_stop is not None: _diag_stop.set()
    log_sub(f"[LFM_SHORT_REQUEST] TTFT: {ttft_s:.4f}s, total latency: {lat_s:.4f}s, tokens={len(tok_ids)}, text={txt!r}")
    results = dict(base, first_token_ts=first_tok_ts,
                   short_request={"ttft_s": round(ttft_s, 4), "latency_s": round(lat_s, 4), "n_tokens": len(tok_ids), "text": txt},
                   hbm_stats=get_hbm(), log_timeline=timeline.rows, **({"diag_meta": _diag_meta} if _diag_meta else {}))
    _emit_waterfall(results, first_tok_ts, ttft_s)
    write_results(results)
    log_sub("Engine short request complete")
    sys.exit(0)

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
    filler = filler_text(tok, p_short + " ", max_model_len)
    for target_ctx in [32768, max_model_len]:
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
_emit_waterfall(results, first_tok_ts, short.get("ttft_s"))
write_results(results)
log_sub("Engine benchmarking complete")
