import builtins, logging, os, sys, torch, torch.nn.functional as F

try:
    from lfm25_startup import install_phase_hooks
except Exception:
    install_phase_hooks = None

_PAD_MODULE_NAMES = (
    "tpu_inference.runner.utils",
    "tpu_inference.runner.tpu_runner",
    "tpu_inference.runner.compilation_manager",
)
_DO_NOTHING_MSG = "Should not schedule a request that does nothing!"
_TPU_RUNNER_LOGGERS = (
    "tpu_inference.runner.tpu_runner",
    "vllm.tpu_inference.runner.tpu_runner",
)

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

def _as_torch(x):
    if x is None or isinstance(x, torch.Tensor): return x
    try:
        from torchax.interop import torch_view
        return torch_view(x)
    except Exception:
        return x

def _resolve_attn_metadata(self, wctx):
    meta = None
    try:
        from vllm.forward_context import get_forward_context
        meta = get_forward_context().attn_metadata
    except Exception:
        meta = None
    if meta is None: meta = getattr(wctx, "attn_metadata", None)
    if isinstance(meta, dict):
        pref = getattr(self, "prefix", None)
        meta = meta.get(pref) if pref is not None else None
    return meta

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
    new_state = x_ext.index_select(0, (n + torch.arange(state_len, device=conv_t.device)).reshape(-1))
    if int(conv_t.shape[-1]) != D: new_state = new_state.transpose(0, 1)
    updated = conv_t.index_copy(0, slot.reshape(-1), new_state.unsqueeze(0))
    from torchax.interop import jax_view
    new_j = jax_view(updated)
    wctx.kv_caches[idx] = (new_j,) + tuple(raw[1:]) if isinstance(raw, tuple) else new_j

_CONV_DBG = {"layer": None, "prefill": 0, "decode": 0}

def _dbg_repr(x):
    if x is None: return "none"
    try:
        if isinstance(x, torch.Tensor) and x.numel() == 1:
            return str(x.detach().cpu().item())
        if isinstance(x, (int, float, bool, str)): return str(x)
    except Exception: pass
    s = repr(x)
    return s if len(s) <= 160 else s[:157] + "..."

def _conv_debug_log(self, slot, n, keep, seq_lens, has_init, prev):
    # Debug only. Tracing may make values unreadable; nothing here may raise.
    if os.environ.get("LFM_CONV_DEBUG") != "1": return
    try:
        prefix = getattr(self, "prefix", None)
        if _CONV_DBG["layer"] is None: _CONV_DBG["layer"] = prefix
        if prefix != _CONV_DBG["layer"]: return
        n_num = None
        try:
            if isinstance(n, torch.Tensor) and n.numel() == 1:
                n_num = int(n.detach().cpu().item())
        except Exception: n_num = None
        if n_num is None: kind = "prefill" if _CONV_DBG["prefill"] == 0 else "decode"
        else: kind = "prefill" if n_num > 1 else "decode"
        if kind == "prefill":
            if _CONV_DBG["prefill"] >= 1: return
            _CONV_DBG["prefill"] += 1
        else:
            if _CONV_DBG["decode"] >= 3: return
            _CONV_DBG["decode"] += 1
        try:
            sl = seq_lens.reshape(-1)[0] if isinstance(seq_lens, torch.Tensor) and seq_lens.numel() else seq_lens
            sl_s = _dbg_repr(sl)
        except Exception as e: sl_s = f"error:{type(e).__name__}"
        try: prev_s = _dbg_repr(prev.abs().sum())
        except Exception as e: prev_s = f"error:{type(e).__name__}"
        print(f"CONVDBG layer={prefix} kind={kind} slot={_dbg_repr(slot)} n={_dbg_repr(n)} "
              f"keep={_dbg_repr(keep)} seq_lens={sl_s} "
              f"has_initial_states={_dbg_repr(has_init)} prev_abs_sum={prev_s}",
              flush=True)
    except Exception: pass

def _lookup_conv_cache(self, state_len, T):
    try:
        from tpu_inference.models.vllm.vllm_model_wrapper_context import get_vllm_model_wrapper_context
        wctx = get_vllm_model_wrapper_context()
    except Exception:
        return None
    mapping = getattr(wctx, "layer_name_to_kvcache_index", None) or {}
    if getattr(self, "prefix", None) not in mapping: return None
    idx, raw = mapping[self.prefix], wctx.kv_caches[mapping[self.prefix]]
    arr = raw[0] if isinstance(raw, (tuple, list)) else raw
    conv_t = _as_torch(arr)
    device = conv_t.device
    meta = _resolve_attn_metadata(self, wctx)
    slot_src = None
    if meta is not None:
        slot_src = getattr(meta, "state_indices_tensor", None)
        if slot_src is None: slot_src = getattr(meta, "mamba_state_indices", None)
    if slot_src is not None:
        slot = torch.clamp(_as_torch(slot_src).reshape(-1)[:1].to(dtype=torch.long, device=device), min=0)
    else:
        slot = torch.zeros(1, dtype=torch.long, device=device)
    qsl = getattr(meta, "query_start_loc", None) if meta is not None else None
    if qsl is not None:
        qsl_t = _as_torch(qsl)
        n = torch.clamp((qsl_t[1] - qsl_t[0]).to(torch.long), min=0, max=T).to(device=device)
    else:
        n = torch.full((), T, dtype=torch.long, device=device)
    prev = conv_t.index_select(0, slot.reshape(-1)).squeeze(0)
    if int(prev.shape[0]) != state_len: prev = prev.transpose(0, 1)
    zeros = torch.zeros(prev.shape, dtype=prev.dtype, device=device)
    keep = None
    has_init = None
    if meta is not None:
        for attr in ("has_initial_states", "has_initial_states_p", "has_initial_states_d"):
            has_init = getattr(meta, attr, None)
            if has_init is not None: break
    if has_init is not None:
        hi = has_init if isinstance(has_init, torch.Tensor) else _as_torch(has_init)
        if not isinstance(hi, torch.Tensor): hi = torch.tensor(hi, device=device)
        keep = hi.reshape(-1)[:1].to(dtype=torch.bool, device=device).reshape(())
    seq_t = getattr(meta, "seq_lens", None) if meta is not None else None
    if seq_t is not None:
        ctx_len = _as_torch(seq_t).reshape(-1)[0].to(dtype=torch.long, device=device)
        not_chunk0 = ctx_len > n
        keep = not_chunk0 if keep is None else (keep & not_chunk0)
    if keep is not None:
        prev = torch.where(keep, prev, zeros)
    _conv_debug_log(self, slot, n, keep, seq_t, "missing" if has_init is None else has_init, prev)
    return wctx, idx, raw, prev, conv_t, slot, n

_orig_import = builtins.__import__

def _wrap_paddings(fn):
    def _wrapped(*a, **kw):
        p = fn(*a, **kw)
        if not (isinstance(p, list) and len(p) > 2): return p
        if os.environ.get("LFM_DIAG") == "1": return [16, 4096]
        max_token_size = kw.get("max_token_size", a[1] if len(a) >= 2 else p[-1])
        return sorted({16, 4096, int(max_token_size)})
    _wrapped._is_pruned = True
    return _wrapped

def _patch_padding_modules():
    wrapped = None
    for m_name in _PAD_MODULE_NAMES:
        m = sys.modules.get(m_name)
        if not m: continue
        fn = getattr(m, "get_token_paddings", None)
        if callable(fn) and not getattr(fn, "_is_pruned", False):
            wrapped = _wrap_paddings(fn)
            setattr(m, "get_token_paddings", wrapped)
        elif getattr(fn, "_is_pruned", False):
            wrapped = fn
    if wrapped is None: return
    for m_name in _PAD_MODULE_NAMES:
        m = sys.modules.get(m_name)
        if m is not None: setattr(m, "get_token_paddings", wrapped)

class _DropEmptyScheduleFilter(logging.Filter):
    def filter(self, record):
        try: msg = record.getMessage()
        except Exception: msg = str(getattr(record, "msg", ""))
        return _DO_NOTHING_MSG not in msg

def _install_tpu_runner_warning_filter():
    filt = _DropEmptyScheduleFilter()
    for name in _TPU_RUNNER_LOGGERS:
        lg = logging.getLogger(name)
        if not any(isinstance(f, _DropEmptyScheduleFilter) for f in lg.filters):
            lg.addFilter(filt)

def _patch_vllm_model_wrapper(mod):
    cls = getattr(mod, "VllmModelWrapper", None)
    if cls is None or getattr(cls, "_modules_patched", False): return
    cls.modules = lambda self: self.model.modules() if getattr(self, "model", None) is not None else iter(())
    cls._modules_patched = True

def _patch_loaded_module(name, mod):
    if not mod: return
    # 2026-11-01: per-phase timers (weight load / KV alloc / compile+warmup).
    if install_phase_hooks and name.endswith(("worker.tpu_worker", "runner.compilation_manager")):
        try: install_phase_hooks()
        except Exception: pass
    if "short_conv" in name:
        cls = getattr(mod, "ShortConv", None)
        if cls and cls.forward is not _tpu_xla_short_conv_forward:
            cls.forward = _tpu_xla_short_conv_forward
    if name.endswith("vllm_model_wrapper") and "context" not in name:
        _patch_vllm_model_wrapper(mod)
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
        for c_name in ("TPUModelRunner", "CompilationManager"):
            cls = getattr(mod, c_name, None)
            if cls and hasattr(cls, "_precompile_structured_decoding"):
                setattr(cls, "_precompile_structured_decoding", lambda self, *a, **kw: None)

def patch_sys_modules():
    # from tpu_inference.runner import utils returns the parent; patch via sys.modules.
    # Leave jax.clear_caches live so weight-load JIT is freed before the KV alloc.
    for m_name, m in list(sys.modules.items()):
        if m and isinstance(m_name, str) and (m_name.startswith("tpu_inference") or "short_conv" in m_name or m_name.endswith("vllm_model_wrapper")):
            _patch_loaded_module(m_name, m)
    _patch_padding_modules()

def custom_import(name, *args, **kwargs):
    mod = _orig_import(name, *args, **kwargs)
    if isinstance(name, str) and (name.startswith(("tpu_inference", "vllm")) or "short_conv" in name or "kv_cache" in name):
        patch_sys_modules()
    return mod

def install_all_patches():
    builtins.__import__ = custom_import
    _install_tpu_runner_warning_filter()
    patch_sys_modules()
