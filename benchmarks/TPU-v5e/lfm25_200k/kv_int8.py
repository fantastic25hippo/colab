import builtins, os, sys

_SEEN_INT8 = set()


def patch_module(name, mod):
    if not mod or os.environ.get("LFM_KV_INT8") != "1":
        return
    if name == "tpu_inference.utils" or hasattr(mod, "_DTYPE_STR_ALIAS_TO_JAX_DTYPE"):
        try:
            import jax.numpy as jnp
            d = getattr(mod, "_DTYPE_STR_ALIAS_TO_JAX_DTYPE", None)
            if isinstance(d, dict):
                d["fp8"] = d["int8"] = d["int8_per_token_head"] = jnp.int8.dtype
        except Exception:
            pass
    if "quantization" in name and hasattr(mod, "static_per_tensor_quantize_tensor") and not getattr(mod, "_lfm_quant_round_patched", False):
        try:
            import jax.numpy as jnp
            def _round_quant(dtype, tensor, scale):
                info = jnp.iinfo(dtype) if jnp.issubdtype(dtype, jnp.integer) else jnp.finfo(dtype)
                s = tensor.astype(jnp.float32) / scale
                if jnp.issubdtype(dtype, jnp.integer):
                    s = jnp.round(s)
                return jnp.clip(s, float(info.min), float(info.max)).astype(dtype)
            mod.static_per_tensor_quantize_tensor = _round_quant
            mod._lfm_quant_round_patched = True
        except Exception:
            pass
    if "kernel_hd64" in name and hasattr(mod, "_ragged_paged_attention_kernel") and not getattr(mod, "_lfm_hd64_int8_patched", False):
        try:
            import inspect, textwrap
            src = textwrap.dedent(inspect.getsource(mod._ragged_paged_attention_kernel)).replace(
                "s = jnp.matmul(q, kv.T, preferred_element_type=jnp.float32)",
                "s = jnp.matmul(q, kv.astype(q.dtype).T, preferred_element_type=jnp.float32)",
            ).replace(
                "pv = jnp.matmul(p, kv, preferred_element_type=jnp.float32)",
                "pv = jnp.matmul(p, kv.astype(q_dtype), preferred_element_type=jnp.float32)",
            )
            ns = dict(mod.__dict__)
            exec(compile(src, "<patched_kernel_hd64>", "exec"), ns)
            fn = ns["_ragged_paged_attention_kernel"]
            fn._patched_src = src
            mod._ragged_paged_attention_kernel = fn
            mod._lfm_hd64_int8_patched = True
        except Exception:
            pass
    if "flash_attn" in name and hasattr(mod, "PallasAttentionBackendImpl"):
        cls = mod.PallasAttentionBackendImpl
        if not getattr(cls, "_lfm_kv_int8_patched", False):
            orig_fwd, probed = cls.forward, set()
            def _wrap_fwd(self, layer, query, key, value, kv_cache, attn_metadata, *a, **kw):
                if os.environ.get("LFM_KV_INT8") == "1":
                    ks, vs = os.environ.get("LFM_K_SCALE"), os.environ.get("LFM_V_SCALE")
                    layer._k_scale_float = float(ks) if ks is not None else float(getattr(layer, "_k_scale_float", None) or 1.0)
                    layer._v_scale_float = float(vs) if vs is not None else float(getattr(layer, "_v_scale_float", None) or 1.0)
                    ln = getattr(layer, "layer_name", "unknown")
                    if os.environ.get("LFM_KV_SCALE_PROBE") == "1" and ln not in probed and key is not None and value is not None:
                        try:
                            import jax.numpy as jnp
                            kj = mod.jax_view(key) if hasattr(mod, "jax_view") else key
                            vj = mod.jax_view(value) if hasattr(mod, "jax_view") else value
                            probed.add(ln)
                            print(f"[LFM_KV_SCALE_PROBE] layer={ln} k_scale_est={float(jnp.max(jnp.abs(kj)))/127.0:.6f} v_scale_est={float(jnp.max(jnp.abs(vj)))/127.0:.6f}", flush=True)
                        except Exception:
                            pass
                return orig_fwd(self, layer, query, key, value, kv_cache, attn_metadata, *a, **kw)
            cls.forward = _wrap_fwd
            cls._lfm_kv_int8_patched = True


def _scan_sys_modules():
    for m_name, m in list(sys.modules.items()):
        if m is not None and m_name not in _SEEN_INT8 and m_name.startswith("tpu_inference"):
            _SEEN_INT8.add(m_name)
            patch_module(m_name, m)


def apply():
    if os.environ.get("LFM_KV_INT8") != "1":
        return
    _SEEN_INT8.clear()
    _scan_sys_modules()
    cur_imp = builtins.__import__
    if not getattr(cur_imp, "_kv_int8_wrapped", False):
        def _imp(name, *a, _orig=cur_imp, **kw):
            mod = _orig(name, *a, **kw)
            if isinstance(name, str) and name.startswith("tpu_inference"):
                _scan_sys_modules()
            return mod
        _imp._kv_int8_wrapped = True
        builtins.__import__ = _imp
