"""LFM2.5 startup instrumentation: phase timers + vLLM/tpu-inference log timeline.

Imported by the engine subrunner. No vLLM import at module load, so it stays
importable and testable outside the TPU venv.
"""

import logging, os, re, time

PHASE = {"t0": time.perf_counter(), "spans": []}


def phase_save(name, t0, t1):
    PHASE["spans"].append({"name": name, "start_rel_s": round(t0 - PHASE["t0"], 3),
                           "dur_s": round(t1 - t0, 3)})
    print(f"[LFM25_PHASE] {name} dur_s={t1 - t0:.2f} start_rel={t0 - PHASE['t0']:.2f}", flush=True)


def install_phase_hooks():
    """Wrap TPUWorker + CompilationManager once imported (idempotent, read-only)."""
    import sys
    tw = sys.modules.get("tpu_inference.worker.tpu_worker")
    cls = getattr(tw, "TPUWorker", None) if tw else None
    if cls is not None and not getattr(cls, "_lfm_phase", False):
        cls._lfm_phase = True
        for meth, label in (("load_model", "weight_load"),
                            ("initialize_from_config", "kv_alloc"),
                            ("compile_or_warm_up_model", "compile_warmup")):
            orig = getattr(cls, meth, None)
            if not callable(orig):
                continue
            def _w(self, *a, _o=orig, _l=label, **kw):
                t0 = time.perf_counter()
                r = _o(self, *a, **kw)
                phase_save(_l, t0, time.perf_counter())
                return r
            setattr(cls, meth, _w)
    cm = sys.modules.get("tpu_inference.runner.compilation_manager")
    ccls = getattr(cm, "CompilationManager", None) if cm else None
    orig = getattr(ccls, "_run_compilation", None) if ccls else None
    if callable(orig) and not getattr(ccls, "_lfm_phase", False):
        ccls._lfm_phase = True
        def _w(self, name, fn, *a, warmup_handler=None, **kw):
            if warmup_handler is None:
                return orig(self, name, fn, *a, **kw)
            label = f"warmup[{name}:num_tokens={kw.get('num_tokens')}]"
            def _h(_fn, _args, _ckw, _wh=warmup_handler, _l=label):
                t0 = time.perf_counter()
                out = _wh(_fn, _args, _ckw)
                phase_save(_l, t0, time.perf_counter())
                return out
            return orig(self, name, fn, *a, warmup_handler=_h, **kw)
        ccls._run_compilation = _w


_KEEP = re.compile(
    r"Loading weights from|Precompile |Compilation of |Warm-up call pass|AOT lower skipped|"
    r"Finished tracing|persistent compilation cache|PERSISTENT COMPILATION CACHE|Error|Failed")


class LogTimeline(logging.Handler):
    def __init__(self, t0=None, cap=400):
        super().__init__(level=logging.INFO)
        self.t0, self.cap, self.rows, self.seen = t0 or time.time(), cap, [], 0

    def emit(self, record):
        self.seen += 1
        if len(self.rows) >= self.cap:
            return
        try:
            msg = record.getMessage()
        except Exception:
            return
        if _KEEP.search(msg):
            self.rows.append({"t": round(time.time() - self.t0, 2),
                              "log": record.name.split(".")[0], "msg": msg[:240]})


def install_log_timeline():
    h = LogTimeline()
    logging.getLogger().addHandler(h)
    return h


def summarize_timeline(rows):
    buckets, warm, shards, traces, misses = {}, None, [], 0, 0
    for r in rows:
        m = r.get("msg", "")
        mm = re.search(r"Compilation of .*?'num_tokens': (\d+).*?finished in ([\d.]+) \[secs\]", m)
        if mm:
            buckets.setdefault(mm.group(1), []).append(round(float(mm.group(2)), 2))
        wm = re.search(r"Warm-up call pass finished in ([\d.]+) \[secs\] over (\d+) tasks", m)
        if wm:
            warm = {"dur_s": float(wm.group(1)), "tasks": int(wm.group(2)), "t": r["t"]}
        if "Loading weights from" in m:
            shards.append({"t": r["t"], "file": os.path.basename(m.split("Loading weights from")[-1].strip())})
        if "Finished tracing" in m:
            traces += 1
        if "PERSISTENT COMPILATION CACHE MISS" in m:
            misses += 1
    return {"compile_per_bucket_s": {k: round(sum(v), 2) for k, v in buckets.items()},
            "compile_runs_per_bucket": {k: len(v) for k, v in buckets.items()},
            "warmup_pass": warm, "weight_shard_loads": shards,
            "tracing_lines": traces, "persistent_cache_misses": misses}
