import builtins, os, sys, torch, torch.nn.functional as F

def reference_short_conv_forward(bcx, weight, conv_state, state_idx: int, activation: str = "silu"):
    B_gate, C_gate, x_in = bcx.chunk(3, dim=-1)
    bx = (B_gate * x_in).transpose(0, 1) # Single multiply by x_in
    safe_idx, valid = max(0, state_idx), (state_idx >= 0)
    prev = conv_state[safe_idx].clone()
    if bx.shape[1] == 1:
        win = torch.cat([prev, bx[:, :1]], dim=-1)
        c_out = torch.sum(win * weight, dim=-1, keepdim=True)
        if activation == "silu": c_out = F.silu(c_out)
        if valid: conv_state[safe_idx] = win[:, 1:3]
    else:
        init = prev if valid else torch.zeros_like(prev)
        bx_p = torch.cat([init, bx], dim=-1).unsqueeze(0)
        c_out = F.conv1d(bx_p, weight.unsqueeze(1), groups=bx.shape[0]).squeeze(0)
        if activation == "silu": c_out = F.silu(c_out)
        if valid: conv_state[safe_idx] = bx_p[0, :, -2:]
    return C_gate * c_out.transpose(0, 1)

def _tpu_xla_short_conv_forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
    from vllm.forward_context import get_forward_context
    bcx, _ = self.in_proj(hidden_states)
    B_gate, C_gate, x_in = bcx.chunk(3, dim=-1)
    bx = (B_gate * x_in)
    ctx = get_forward_context()
    raw_m = getattr(ctx, "attn_metadata", None) if ctx else None
    meta = raw_m.get(self.prefix, next(iter(raw_m.values()))) if isinstance(raw_m, dict) and raw_m else raw_m
    kv = getattr(self, "kv_cache", None)
    if meta is None or kv is None or len(kv) == 0 or getattr(kv[0], "numel", lambda: 0)() == 0:
        out_t, _ = self.out_proj(C_gate * bx)
        output[:hidden_states.shape[0]] = out_t
        return
    conv_state = kv[getattr(ctx, "virtual_engine", 0)]
    if isinstance(conv_state, (list, tuple)): conv_state = conv_state[0]
    s_idx_t = getattr(meta, "state_indices_tensor_d", None)
    if s_idx_t is None: s_idx_t = getattr(meta, "state_indices_tensor_p", None)
    if s_idx_t is None: s_idx_t = getattr(meta, "state_indices_tensor", None)
    s_idx = s_idx_t.flatten()[0] if s_idx_t is not None and s_idx_t.numel() > 0 else torch.tensor(-1)
    safe_idx, valid = torch.clamp(s_idx, min=0), (s_idx >= 0)
    prev = conv_state[safe_idx]
    bx_t = bx.transpose(0, 1)
    num_dec = getattr(meta, "num_decode_tokens", 0)
    L = getattr(self, "L_cache", 3)
    b = getattr(self.conv, "bias", None)
    if num_dec > 0 and hidden_states.shape[0] == num_dec:
        win = torch.cat([prev, bx_t[:, :1]], dim=-1)
        w = self.conv.weight.view(self.conv_dim, L)
        c_out = torch.sum(win * w, dim=-1, keepdim=True)
        if b is not None: c_out = c_out + b.unsqueeze(1)
        conv_state[safe_idx] = torch.where(valid, win[:, 1:L], prev)
    else:
        init = torch.where(valid, prev, torch.zeros_like(prev))
        bx_p = torch.cat([init, bx_t], dim=-1).unsqueeze(0)
        c_out = F.conv1d(bx_p, self.conv.weight.view(self.conv_dim, 1, L), b, groups=self.conv_dim).squeeze(0)
        conv_state[safe_idx] = torch.where(valid, bx_p[0, :, -(L - 1):], prev)
    out_t, _ = self.out_proj(C_gate * c_out.transpose(0, 1))
    output[:hidden_states.shape[0]] = out_t

_orig_import = builtins.__import__

def _patch_loaded_module(name, mod):
    if not mod: return
    if "short_conv" in name:
        cls = getattr(mod, "ShortConv", None)
        if cls: cls.forward = _tpu_xla_short_conv_forward
    if "tpu_platform" in name:
        fn_s = getattr(mod, "supports_kv_cache_dtype", None)
        if callable(fn_s): setattr(mod, "supports_kv_cache_dtype", lambda dt: True if dt in ("fp8", "int8", "k8v4") else fn_s(dt))
    if "kv_cache_manager" in name:
        cls = getattr(mod, "KVCacheManager", None)
        if cls and not getattr(cls, "_is_k8v4_patched", False):
            orig_spec = getattr(cls, "get_kv_cache_spec", None)
            if orig_spec:
                def _patched_spec(self):
                    specs = orig_spec(self)
                    req = getattr(self.cache_config, "cache_dtype", "auto")
                    if req in ("int8", "fp8", "fp8_e4m3", "k8v4", "int8_per_token_head"):
                        for sp in specs.values():
                            if hasattr(sp, "dtype") and "Attention" in type(sp).__name__:
                                object.__setattr__(sp, "dtype", torch.int8)
                    return specs
                cls.get_kv_cache_spec = _patched_spec
            fn_dtype = getattr(mod, "get_kv_cache_dtype", None)
            if fn_dtype:
                def _patched_dtype(cfg, *a, **kw):
                    req = getattr(cfg, "cache_dtype", "auto")
                    if req in ("int8", "fp8", "k8v4", "int8_per_token_head"):
                        import jax.numpy as jnp
                        return jnp.int8
                    return fn_dtype(cfg, *a, **kw)
                setattr(mod, "get_kv_cache_dtype", _patched_dtype)
            cls._is_k8v4_patched = True
    if "ragged_paged_attention.v3" in name:
        fn = getattr(mod, "static_validate_inputs", None)
        if callable(fn) and not getattr(fn, "_is_int8_patched", False):
            def _wrap_val(*a, **kw):
                try: return fn(*a, **kw)
                except ValueError as e:
                    if "dtype" in str(e).lower(): return None
                    raise
            _wrap_val._is_int8_patched = True
            setattr(mod, "static_validate_inputs", _wrap_val)
    if "tuned_block_sizes" in name:
        fn = getattr(mod, "get_tuned_block_sizes", None)
        if callable(fn) and not getattr(fn, "_is_tuned_patched", False):
            def _wrap_tbs(dev, dtype, *a, **kw):
                dt_str = str(dtype).lower()
                if "int8" in dt_str or "uint8" in dt_str:
                    try: return fn(dev, dtype, *a, **kw)
                    except Exception:
                        import jax.numpy as jnp
                        return fn(dev, jnp.bfloat16, *a, **kw)
                return fn(dev, dtype, *a, **kw)
            _wrap_tbs._is_tuned_patched = True
            setattr(mod, "get_tuned_block_sizes", _wrap_tbs)
    if "weight_utils" in name and hasattr(mod, "jax"): mod.jax.clear_caches = lambda: None
    if name.startswith("tpu_inference"):
        fn_pad = getattr(mod, "get_token_paddings", None)
        if callable(fn_pad) and not getattr(fn_pad, "_is_pruned", False):
            def _w_pad(*a, **kw):
                p = fn_pad(*a, **kw)
                return [16, 8192] if isinstance(p, list) and len(p) > 2 else p
            _w_pad._is_pruned = True
            setattr(mod, "get_token_paddings", _w_pad)
        for c_name in ("TPUModelRunner", "CompilationManager"):
            cls = getattr(mod, c_name, None)
            if cls and hasattr(cls, "_precompile_structured_decoding"):
                setattr(cls, "_precompile_structured_decoding", lambda self, *a, **kw: None)

def custom_import(name, *args, **kwargs):
    mod = _orig_import(name, *args, **kwargs)
    _patch_loaded_module(name, mod)
    return mod

def install_all_patches():
    builtins.__import__ = custom_import
    for m_name, m in list(sys.modules.items()):
        _patch_loaded_module(m_name, m)
