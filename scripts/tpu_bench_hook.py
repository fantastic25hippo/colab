import builtins, json, os, time

stats_p = "/tmp/tpu_stats.json"

def _save(stage, data):
    cur = {}
    if os.path.exists(stats_p):
        try:
            with open(stats_p) as f: cur = json.load(f)
        except Exception: pass
    cur[stage] = data
    with open(stats_p, "w") as f: json.dump(cur, f)

def _get_stats(worker):
    import jax
    dev = worker.devices[0] if getattr(worker, "devices", None) else jax.devices()[0]
    return dev.memory_stats()

orig_imp = builtins.__import__

def custom_imp(name, *args, **kwargs):
    m = orig_imp(name, *args, **kwargs)
    if "weight_utils" in name and hasattr(m, "jax"):
        mode = os.environ.get("BENCH_MODE", "base")
        if mode in ("opt", "warm"):
            m.jax.clear_caches = lambda: None
    if "tpu_worker" in name and hasattr(m, "TPUWorker"):
        cls = m.TPUWorker
        if not getattr(cls, "_bench_hooked", False):
            cls._bench_hooked = True
            o_load = cls.load_model
            def w_load(self):
                t0 = time.time(); o_load(self); t1 = time.time()
                st = _get_stats(self)
                _save("after_load", {
                    "dur_s": t1 - t0, "t_stamp": t1,
                    "bytes_in_use": st.get("bytes_in_use"),
                    "peak_bytes_in_use": st.get("peak_bytes_in_use"),
                    "bytes_limit": st.get("bytes_limit"),
                })
            cls.load_model = w_load

            o_init = cls.initialize_from_config
            def w_init(self, cfg):
                t0 = time.time(); o_init(self, cfg); t1 = time.time()
                st = _get_stats(self)
                blocks = getattr(self.cache_config, "num_gpu_blocks", None)
                util = getattr(self.cache_config, "gpu_memory_utilization", None)
                _save("after_kv", {
                    "dur_s": t1 - t0, "t_stamp": t1,
                    "bytes_in_use": st.get("bytes_in_use"),
                    "peak_bytes_in_use": st.get("peak_bytes_in_use"),
                    "bytes_limit": st.get("bytes_limit"),
                    "num_gpu_blocks": blocks, "gpu_memory_utilization": util,
                })
            cls.initialize_from_config = w_init

            o_comp = cls.compile_or_warm_up_model
            def w_comp(self):
                mode = os.environ.get("BENCH_MODE", "base")
                if mode in ("opt", "warm") and hasattr(self, "model_runner"):
                    self.model_runner.num_tokens_paddings = [16, 512]
                t0 = time.time(); res = o_comp(self); t1 = time.time()
                st = _get_stats(self)
                _save("after_compile", {
                    "dur_s": t1 - t0, "t_stamp": t1,
                    "bytes_in_use": st.get("bytes_in_use"),
                    "peak_bytes_in_use": st.get("peak_bytes_in_use"),
                    "bytes_limit": st.get("bytes_limit"),
                })
                return res
            cls.compile_or_warm_up_model = w_comp
    return m

builtins.__import__ = custom_imp
