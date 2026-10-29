import builtins, json, logging, os, sys, threading, time

stats_p = "/tmp/tpu_stats.json"
stage_req_p = "/tmp/record_hbm_stage.txt"
cache_stats = {"compiled": 0, "cached": 0}

def _save(stage, data):
    cur = {}
    if os.path.exists(stats_p):
        try:
            with open(stats_p) as f: cur = json.load(f)
        except Exception: pass
    cur[stage] = data
    with open(stats_p, "w") as f: json.dump(cur, f)

def _get_stats():
    import jax
    dev = jax.local_devices()[0]
    st = dev.memory_stats()
    return {
        "bytes_in_use": int(st.get("bytes_in_use", 0)),
        "peak_bytes_in_use": int(st.get("peak_bytes_in_use", 0)),
        "bytes_limit": int(st.get("bytes_limit", 0)),
    }

def _check_and_save_stage():
    if os.path.exists(stage_req_p):
        try:
            with open(stage_req_p) as f: stage = f.read().strip()
            if stage:
                st = _get_stats()
                _save(stage, st)
            os.remove(stage_req_p)
        except Exception: pass

class CacheLogHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "Persistent compilation cache hit" in msg:
            cache_stats["cached"] += 1
            _save("cache", cache_stats)
        elif "PERSISTENT COMPILATION CACHE MISS" in msg or ("Writing " in msg and "persistent compilation cache" in msg) or "Compiling " in msg:
            cache_stats["compiled"] += 1
            _save("cache", cache_stats)

orig_imp = builtins.__import__

def custom_imp(name, *args, **kwargs):
    res = orig_imp(name, *args, **kwargs)
    if "jax" in sys.modules:
        jmod = sys.modules["jax"]
        if not getattr(jmod, "_bench_configured", False):
            jmod._bench_configured = True
            h = CacheLogHandler()
            for l_name in ["jax._src.compiler", "jax._src.compilation_cache", "jax.interpreters.xla"]:
                l = logging.getLogger(l_name)
                l.addHandler(h); l.setLevel(logging.INFO)

    w_mod_name = "tpu_inference.models.jax.utils.weight_utils"
    w_mod = sys.modules.get(w_mod_name)
    if w_mod and hasattr(w_mod, "jax") and not getattr(w_mod, "_bench_patched", False):
        if os.environ.get("BENCH_LOAD_FIX", "0") == "1":
            try: w_mod.jax.clear_caches = lambda: None
            except Exception: pass
        w_mod._bench_patched = True

    t_mod_name = "tpu_inference.worker.tpu_worker"
    t_mod = sys.modules.get(t_mod_name)
    if t_mod and hasattr(t_mod, "TPUWorker"):
        cls = t_mod.TPUWorker
        if not getattr(cls, "_bench_hooked", False):
            cls._bench_hooked = True
            o_load = cls.load_model
            def w_load(self):
                t0 = time.perf_counter()
                res = o_load(self)
                t1 = time.perf_counter()
                st = _get_stats()
                st["dur_s"] = t1 - t0
                _save("after_load", st)
                return res
            cls.load_model = w_load

            o_init = cls.initialize_from_config
            def w_init(self, kv_cfg):
                t0 = time.perf_counter()
                res = o_init(self, kv_cfg)
                t1 = time.perf_counter()
                st = _get_stats()
                st["dur_s"] = t1 - t0
                num_blks = getattr(kv_cfg, "num_blocks", None) or getattr(kv_cfg, "num_gpu_blocks", None) or getattr(self.cache_config, "num_gpu_blocks", None)
                b_sz = getattr(kv_cfg, "block_size", None) or getattr(self.cache_config, "block_size", None) or getattr(self.model_runner, "block_size", None) or 256
                util = getattr(self.cache_config, "gpu_memory_utilization", getattr(kv_cfg, "gpu_memory_utilization", 0.90))
                st["num_gpu_blocks"] = num_blks
                st["block_size"] = b_sz
                st["gpu_memory_utilization"] = util
                _save("after_kv", st)
                return res
            cls.initialize_from_config = w_init

            o_comp = cls.compile_or_warm_up_model
            def w_comp(self):
                t0 = time.perf_counter()
                res = o_comp(self)
                t1 = time.perf_counter()
                st = _get_stats()
                st["dur_s"] = t1 - t0
                _save("after_compile", st)
                _save("cache", cache_stats)
                return res
            cls.compile_or_warm_up_model = w_comp

            o_exec = cls.execute_model
            def w_exec(self, *args, **kwargs):
                res = o_exec(self, *args, **kwargs)
                _check_and_save_stage()
                return res
            cls.execute_model = w_exec

            def _stage_loop():
                while True:
                    _check_and_save_stage()
                    time.sleep(0.1)
            threading.Thread(target=_stage_loop, daemon=True).start()

    return res

builtins.__import__ = custom_imp
