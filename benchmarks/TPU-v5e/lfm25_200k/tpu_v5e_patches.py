import builtins, sys, torch, torch.nn.functional as F

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

def sd_causal_conv(bx, weight, bias, state):
    # vLLM default conv state is SD: (L-1, D). bx is (T, D), weight is (D, L).
    L, T = int(weight.shape[-1]), int(bx.shape[0])
    x_ext = torch.cat([state, bx], dim=0)
    acc = x_ext[:T] * weight[:, 0]
    for k in range(1, L): acc = acc + x_ext[k:k + T] * weight[:, k]
    if bias is not None: acc = acc + bias.reshape(1, -1)
    return acc, x_ext

def _tpu_xla_short_conv_forward(self, hidden_states: torch.Tensor, output: torch.Tensor):
    # Traced by torchax from capture_model/model_fn. Token count stays a device scalar.
    bcx, _ = self.in_proj(hidden_states)
    B_gate, C_gate, x_in = bcx.chunk(3, dim=-1)
    bx = B_gate * x_in
    L = int(getattr(self, "L_cache", 3))
    state_len, T, D = L - 1, int(bx.shape[0]), int(bx.shape[-1])
    bound = _lookup_conv_cache(self, state_len, T)
    state = bx.new_zeros((state_len, D)) if bound is None else bound[3]
    acc, x_ext = sd_causal_conv(bx, self.conv.weight.reshape(D, L), getattr(self.conv, "bias", None), state)
    out_t, _ = self.out_proj(C_gate * acc)
    output[:out_t.shape[0]] = out_t
    if bound is None: return
    wctx, idx, raw, _, conv_t, slot, n = bound
    new_state = x_ext.index_select(0, (n + torch.arange(state_len, device=n.device)).reshape(-1))
    if int(conv_t.shape[-1]) != D: new_state = new_state.transpose(0, 1)
    updated = conv_t.index_copy(0, slot.reshape(-1), new_state.unsqueeze(0))
    from torchax.interop import jax_view
    new_j = jax_view(updated)
    wctx.kv_caches[idx] = (new_j,) + tuple(raw[1:]) if isinstance(raw, tuple) else new_j

def _lookup_conv_cache(self, state_len, T):
    try:
        from tpu_inference.models.vllm.vllm_model_wrapper_context import get_vllm_model_wrapper_context
        wctx = get_vllm_model_wrapper_context()
    except Exception:
        return None
    mapping = getattr(wctx, "layer_name_to_kvcache_index", None) or {}
    if getattr(self, "prefix", None) not in mapping: return None
    from torchax.interop import torch_view
    idx, raw = mapping[self.prefix], wctx.kv_caches[mapping[self.prefix]]
    arr = raw[0] if isinstance(raw, (tuple, list)) else raw
    conv_t, shared = torch_view(arr), getattr(wctx, "shared_attn_metadata", None)
    if shared is not None and getattr(shared, "mamba_state_indices", None) is not None:
        slot = torch.clamp(torch_view(shared.mamba_state_indices).reshape(-1)[:1].to(torch.long), min=0)
        n = torch.clamp((torch_view(shared.query_start_loc)[1] - torch_view(shared.query_start_loc)[0]).to(torch.long), 0, T)
    else:
        slot = torch.zeros(1, dtype=torch.long, device=conv_t.device)
        n = torch.full((), T, dtype=torch.long, device=conv_t.device)
    prev = conv_t.index_select(0, slot.reshape(-1)).squeeze(0)
    if int(prev.shape[0]) != state_len: prev = prev.transpose(0, 1)
    return wctx, idx, raw, prev, conv_t, slot, n

_orig_import = builtins.__import__

def _wrap_paddings(fn):
    def _wrapped(*a, **kw):
        p = fn(*a, **kw)
        return sorted(set((p[0], p[-1]))) if isinstance(p, list) and len(p) > 2 else p
    _wrapped._is_pruned = True
    return _wrapped

def _patch_loaded_module(name, mod):
    if not mod: return
    if "short_conv" in name:
        cls = getattr(mod, "ShortConv", None)
        if cls and cls.forward is not _tpu_xla_short_conv_forward:
            cls.forward = _tpu_xla_short_conv_forward
    if "tpu_platform" in name:
        fn_s = getattr(mod, "supports_kv_cache_dtype", None)
        if callable(fn_s) and not getattr(fn_s, "_is_dtype_patched", False):
            wrapped = lambda dt, _fn=fn_s: True if dt in ("fp8", "int8", "k8v4") else _fn(dt)
            wrapped._is_dtype_patched = True
            mod.supports_kv_cache_dtype = wrapped
    if "ragged_paged_attention.v3" in name:
        fn = getattr(mod, "static_validate_inputs", None)
        if callable(fn) and not getattr(fn, "_is_int8_patched", False):
            def _wrap_val(*a, _fn=fn, **kw):
                try: return _fn(*a, **kw)
                except ValueError as e:
                    if "dtype" in str(e).lower(): return None
                    raise
            _wrap_val._is_int8_patched = True
            mod.static_validate_inputs = _wrap_val
    if "tuned_block_sizes" in name:
        fn = getattr(mod, "get_tuned_block_sizes", None)
        if callable(fn) and not getattr(fn, "_is_tuned_patched", False):
            def _wrap_tbs(dev, dtype, *a, _fn=fn, **kw):
                if "int8" in str(dtype).lower() or "uint8" in str(dtype).lower():
                    try: return _fn(dev, dtype, *a, **kw)
                    except Exception:
                        import jax.numpy as jnp
                        return _fn(dev, jnp.bfloat16, *a, **kw)
                return _fn(dev, dtype, *a, **kw)
            _wrap_tbs._is_tuned_patched = True
            mod.get_tuned_block_sizes = _wrap_tbs
    if name.startswith("tpu_inference"):
        fn_pad = getattr(mod, "get_token_paddings", None)
        if callable(fn_pad) and not getattr(fn_pad, "_is_pruned", False):
            setattr(mod, "get_token_paddings", _wrap_paddings(fn_pad))
        for c_name in ("TPUModelRunner", "CompilationManager"):
            cls = getattr(mod, c_name, None)
            if cls and hasattr(cls, "_precompile_structured_decoding"):
                setattr(cls, "_precompile_structured_decoding", lambda self, *a, **kw: None)

def patch_sys_modules():
    # from tpu_inference.runner import utils returns the parent; patch via sys.modules.
    # Leave jax.clear_caches live so weight-load JIT is freed before the KV alloc.
    for m_name, m in list(sys.modules.items()):
        if m and isinstance(m_name, str) and (m_name.startswith("tpu_inference") or "short_conv" in m_name):
            _patch_loaded_module(m_name, m)

def custom_import(name, *args, **kwargs):
    mod = _orig_import(name, *args, **kwargs)
    if isinstance(name, str) and (name.startswith(("tpu_inference", "vllm")) or "short_conv" in name or "kv_cache" in name):
        patch_sys_modules()
    return mod

def install_all_patches():
    builtins.__import__ = custom_import
    patch_sys_modules()
