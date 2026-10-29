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
            cache_stats["cached"] += 1; _save("cache", cache_stats)
        elif "PERSISTENT COMPILATION CACHE MISS" in msg or ("Writing " in msg and "persistent compilation cache" in msg) or "Compiling " in msg:
            cache_stats["compiled"] += 1; _save("cache", cache_stats)

orig_imp = builtins.__import__

def custom_imp(name, *args, **kwargs):
    res = orig_imp(name, *args, **kwargs)
    if "jax" in sys.modules:
        jmod = sys.modules["jax"]
        if not getattr(jmod, "_bench_configured", False):
            jmod._bench_configured = True
            h = CacheLogHandler()
            for l_name in ["jax._src.compiler", "jax._src.compilation_cache", "jax.interpreters.xla"]:
                l = logging.getLogger(l_name); l.addHandler(h); l.setLevel(logging.INFO)

    w_mod = sys.modules.get("tpu_inference.models.jax.utils.weight_utils")
    if w_mod and hasattr(w_mod, "jax") and not getattr(w_mod, "_bench_patched", False):
        if os.environ.get("BENCH_LOAD_FIX", "0") == "1":
            try: w_mod.jax.clear_caches = lambda: None
            except Exception: pass
        w_mod._bench_patched = True

    t_mod = sys.modules.get("tpu_inference.worker.tpu_worker")
    if t_mod and hasattr(t_mod, "TPUWorker"):
        cls = t_mod.TPUWorker
        if not getattr(cls, "_bench_hooked", False):
            cls._bench_hooked = True
            o_load = cls.load_model
            def w_load(self):
                t0 = time.perf_counter()
                r = o_load(self)
                t1 = time.perf_counter()
                st = _get_stats()
                st["dur_s"] = t1 - t0
                _save("after_load", st)
                return r
            cls.load_model = w_load

            o_init = cls.initialize_from_config
            def w_init(self, kv_cfg):
                t0 = time.perf_counter()
                c0 = cache_stats["compiled"]
                st0 = _get_stats()
                r = o_init(self, kv_cfg)
                t1 = time.perf_counter()
                c1 = cache_stats["compiled"]
                st1 = _get_stats()
                st1["dur_s"] = t1 - t0
                st1["kv_bytes"] = st1["bytes_in_use"] - st0["bytes_in_use"]
                st1["compiles_during_kv"] = c1 - c0
                num_blks = getattr(kv_cfg, "num_gpu_blocks", getattr(self.cache_config, "num_gpu_blocks", None))
                b_sz = getattr(kv_cfg, "block_size", getattr(self.cache_config, "block_size", None)) or 256
                util = getattr(self.cache_config, "gpu_memory_utilization", getattr(kv_cfg, "gpu_memory_utilization", 0.90))
                st1["num_gpu_blocks"] = num_blks
                st1["block_size"] = b_sz
                st1["gpu_memory_utilization"] = util
                gc = getattr(self, "gpu_cache", None) or getattr(getattr(self, "model_runner", None), "gpu_cache", None)
                if gc:
                    st1["num_layers"] = len(gc)
                    if len(gc) > 0 and hasattr(gc[0], "shape"):
                        st1["layer_shape"] = list(gc[0].shape)
                        st1["layer_nbytes"] = getattr(gc[0], "nbytes", 0)
                _save("after_kv", st1)
                return r
            cls.initialize_from_config = w_init

            o_comp = cls.compile_or_warm_up_model
            def w_comp(self):
                t0 = time.perf_counter()
                r = o_comp(self)
                t1 = time.perf_counter()
                st = _get_stats()
                st["dur_s"] = t1 - t0
                _save("after_compile", st)
                _save("cache", cache_stats)
                return r
            cls.compile_or_warm_up_model = w_comp

            o_exec = cls.execute_model
            def w_exec(self, *args, **kwargs):
                r = o_exec(self, *args, **kwargs)
                _check_and_save_stage()
                return r
            cls.execute_model = w_exec

            def _stage_loop():
                while True:
                    _check_and_save_stage()
                    time.sleep(0.1)
            threading.Thread(target=_stage_loop, daemon=True).start()

    return res

builtins.__import__ = custom_imp
