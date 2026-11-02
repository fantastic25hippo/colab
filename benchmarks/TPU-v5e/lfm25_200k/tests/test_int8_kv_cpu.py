import logging, os, sys, types
import numpy as np, pytest, torch

_UV_JAX = "/home/sahil/.cache/uv/archive-v0/s66A7_ti0x9rZScx/lib/python3.13/site-packages"
_KSRC = "/home/sahil/Desktop/sahil/src-0.32.0/tpu-inference"
_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_DIR, _KSRC, _UV_JAX):
    if os.path.isdir(_p) and _p not in sys.path: sys.path.insert(0, _p)

import jax, jax.numpy as jnp

for _m, _attrs in (
    ("tpu_inference", {"__path__": [os.path.join(_KSRC, "tpu_inference")]}),
    ("tpu_inference.logger", {"init_logger": lambda name: logging.getLogger(name)}),
    ("torchax", {}), ("torchax.ops", {}),
    ("torchax.ops.mappings", {"j2t_dtype": lambda x: x, "t2j": lambda x, **kw: x, "t2j_dtype": lambda x: x}),
    ("torchax.interop", {"torch_view": lambda x: x, "jax_view": lambda x: x}),
    ("vllm", {"envs": types.SimpleNamespace(VLLM_TPU_USING_PATHWAYS=False), "utils": types.SimpleNamespace()}),
):
    if _m not in sys.modules:
        mod = types.ModuleType(_m); [setattr(mod, k, v) for k, v in _attrs.items()]; sys.modules[_m] = mod

import kv_int8, lfm25_cache, tpu_v5e_patches  # noqa: E402
from tpu_inference import utils as tpu_utils  # noqa: E402
from tpu_inference.kernels.ragged_paged_attention.v3 import kernel_hd64  # noqa: E402
from tpu_inference.layers.common import quantization as quant_mod  # noqa: E402
from tpu_inference.layers.common.quantization import quantize_kv  # noqa: E402


def test_2b_cache_key_and_apply_flag_off_vs_on(monkeypatch):
    patch_txt = open(os.path.join(_DIR, "tpu_v5e_patches.py")).read()
    args_str = lfm25_cache.extract_cache_key_args(open(os.path.join(_DIR, "lfm25_engine_subrunner.py")).read())
    monkeypatch.delenv("LFM_KV_INT8", raising=False)
    fn_off = lfm25_cache.cache_filename("0.32.0", args_str, patch_txt, max_model_len=100000, buckets="16-8192")
    assert fn_off.endswith("__42f6059b.tar") and "kvint8" not in fn_off
    stub_u, stub_k = types.ModuleType("tpu_inference.stub_u"), types.ModuleType("tpu_inference.stub_kernel_hd64")
    stub_u._DTYPE_STR_ALIAS_TO_JAX_DTYPE = {"fp8": jnp.float8_e4m3fn.dtype}
    def _dummy_k(q, kv, p, q_dtype):
        s = jnp.matmul(q, kv.T, preferred_element_type=jnp.float32)
        pv = jnp.matmul(p, kv, preferred_element_type=jnp.float32)
        return s, pv
    stub_k._ragged_paged_attention_kernel = _dummy_k
    monkeypatch.setitem(sys.modules, "tpu_inference.stub_u", stub_u)
    monkeypatch.setitem(sys.modules, "tpu_inference.stub_kernel_hd64", stub_k)
    before_u, before_k = dict(stub_u.__dict__), dict(stub_k.__dict__)
    kv_int8.apply()
    assert stub_u.__dict__ == before_u and stub_k.__dict__ == before_k
    monkeypatch.setenv("LFM_KV_INT8", "1")
    fn_on = lfm25_cache.cache_filename("0.32.0", args_str, patch_txt, max_model_len=100000, buckets="16-8192")
    assert "kvint8" in fn_on and not fn_on.endswith("__42f6059b.tar")
    kv_int8.apply()
    assert "int8" in stub_u._DTYPE_STR_ALIAS_TO_JAX_DTYPE
    assert stub_u._DTYPE_STR_ALIAS_TO_JAX_DTYPE["fp8"] == jnp.int8.dtype
    src = getattr(stub_k._ragged_paged_attention_kernel, "_patched_src", "")
    assert "kv.astype(q.dtype).T" in src and "kv.astype(q_dtype)" in src


def test_3a_int8_kv_cache_shape_and_flag(monkeypatch):
    tpu_utils._DTYPE_STR_ALIAS_TO_JAX_DTYPE["fp8"] = jnp.float8_e4m3fn.dtype
    monkeypatch.delenv("LFM_KV_INT8", raising=False)
    kv_int8.apply()
    assert tpu_utils.to_jax_dtype("fp8") == jnp.float8_e4m3fn.dtype
    monkeypatch.setenv("LFM_KV_INT8", "1")
    kv_int8.apply()
    assert tpu_utils.to_jax_dtype("fp8") == jnp.int8.dtype
    src = getattr(kernel_hd64._ragged_paged_attention_kernel, "_patched_src", "")
    assert "kv.astype(q.dtype).T" in src and "kv.astype(q_dtype)" in src

    pages = 8
    shape = kernel_hd64.get_kv_cache_shape(pages, 16, 8, 64, jnp.int8)
    assert shape == (pages, 16, 2, 4, 128)
    assert int(np.prod(shape[1:])) * jnp.dtype(jnp.int8).itemsize == 16384


def test_3b_quantize_store_dequant_round_trip(monkeypatch):
    monkeypatch.setenv("LFM_KV_INT8", "1")
    kv_int8.apply()
    k_bf16 = jax.random.normal(jax.random.PRNGKey(42), (64, 8, 64), dtype=jnp.bfloat16)
    v_bf16 = jax.random.normal(jax.random.PRNGKey(43), (64, 8, 64), dtype=jnp.bfloat16)
    k_scale = float(jnp.max(jnp.abs(k_bf16))) / 127.0
    v_scale = float(jnp.max(jnp.abs(v_bf16))) / 127.0

    k_q, v_q = quantize_kv(jnp.int8, k_bf16, v_bf16, k_scale, v_scale)
    merged = kernel_hd64.merge_kv(k_q, v_q)
    assert merged.shape == (64, 2, 4, 128) and merged.dtype == jnp.int8

    kv_cache = jnp.zeros(kernel_hd64.get_kv_cache_shape(4, 16, 8, 64, jnp.int8), dtype=jnp.int8)
    kv_cache = kv_cache.at[:4].set(merged.reshape(4, 16, 2, 4, 128))
    unmerged = kv_cache.reshape(64, 8, 128)
    k_deq = unmerged[:, :, :64].astype(jnp.float32) * k_scale
    v_deq = unmerged[:, :, 64:].astype(jnp.float32) * v_scale

    k_err = float(jnp.max(jnp.abs(k_deq - k_bf16.astype(jnp.float32))))
    v_err = float(jnp.max(jnp.abs(v_deq - v_bf16.astype(jnp.float32))))
    assert k_err <= k_scale / 2.0 + 1e-6 and v_err <= v_scale / 2.0 + 1e-6


def test_3c_pallas_cpu_and_int8_vs_bf16_attention():
    # Pallas ragged_paged_attention_hd64 requires TPU; verify CPU raises ValueError and measure ref math.
    q = jax.random.normal(jax.random.PRNGKey(1), (64, 32, 64), dtype=jnp.bfloat16)
    k = jax.random.normal(jax.random.PRNGKey(2), (64, 8, 64), dtype=jnp.bfloat16)
    v = jax.random.normal(jax.random.PRNGKey(3), (64, 8, 64), dtype=jnp.bfloat16)
    kv_cache_bf16 = jnp.zeros(kernel_hd64.get_kv_cache_shape(4, 16, 8, 64, jnp.bfloat16), dtype=jnp.bfloat16)
    kv_lens, page_indices = jnp.array([32, 32], dtype=jnp.int32), jnp.arange(4, dtype=jnp.int32)
    cu_q_lens, distribution = jnp.array([0, 32, 64], dtype=jnp.int32), jnp.array([0, 0, 2], dtype=jnp.int32)

    with pytest.raises(ValueError, match="Only interpret mode is supported on CPU backend"):
        kernel_hd64.ragged_paged_attention_hd64(
            q, k, v, kv_cache_bf16, kv_lens, page_indices, cu_q_lens, distribution,
            num_kv_pages_per_block=2, num_queries_per_block=16,
        )

    out_bf16, _ = kernel_hd64.ref_ragged_paged_attention_hd64(
        q, k, v, kv_cache_bf16, kv_lens, page_indices, cu_q_lens, distribution, sm_scale=0.125
    )
    k_scale, v_scale = float(jnp.max(jnp.abs(k))) / 127.0, float(jnp.max(jnp.abs(v))) / 127.0
    k_q, v_q = quantize_kv(jnp.int8, k, v, k_scale, v_scale)
    out_int8, _ = kernel_hd64.ref_ragged_paged_attention_hd64(
        q, k_q.astype(jnp.bfloat16), v_q.astype(jnp.bfloat16),
        kv_cache_bf16, kv_lens, page_indices, cu_q_lens, distribution,
        sm_scale=0.125, k_scale=k_scale, v_scale=v_scale,
    )
    diff = jnp.abs(out_int8.astype(jnp.float32) - out_bf16.astype(jnp.float32))
    print(f"\n[3c CPU ref; Pallas needs TPU] max_abs={float(jnp.max(diff)):.6f} mean_rel={float(jnp.mean(diff / (jnp.abs(out_bf16.astype(jnp.float32)) + 1e-6))):.6f}")


def test_3d_p2_shortconv_two_chunk_vs_numpy():
    torch.manual_seed(0)
    D, L, T = 32, 3, 64
    w_t, b_t = torch.randn(D, 1, L), torch.randn(D)
    in_w, out_w = torch.randn(3 * D, D) / np.sqrt(D), torch.randn(D, D) / np.sqrt(D)
    x_full = torch.randn(64, D)

    B_np, C_np, xin_np = np.split(x_full.numpy() @ in_w.numpy().T, 3, axis=-1)
    bx_pad = np.vstack([np.zeros((L - 1, D), dtype=np.float32), B_np * xin_np])
    w_np = w_t.numpy().reshape(D, L)
    acc_np = sum(bx_pad[k:k + 64] * w_np[:, k] for k in range(L)) + b_t.numpy().reshape(1, D)
    ref_np = (C_np * acc_np) @ out_w.numpy().T

    wctx = types.SimpleNamespace(
        layer_name_to_kvcache_index={"layer.0.conv": 0},
        kv_caches=[torch.zeros(2, L - 1, D)], attn_metadata=None,
    )
    ctx_mod = types.ModuleType("tpu_inference.models.vllm.vllm_model_wrapper_context")
    ctx_mod.get_vllm_model_wrapper_context = lambda: wctx
    sys.modules["tpu_inference.models.vllm.vllm_model_wrapper_context"] = ctx_mod

    layer = types.SimpleNamespace(
        prefix="layer.0.conv", L_cache=L, in_proj=lambda h: (h @ in_w.T, None),
        out_proj=lambda h: (h @ out_w.T, None), conv=types.SimpleNamespace(weight=w_t, bias=b_t),
    )
    outs = []
    for n_tok, ctx_len, chunk in [(48, 48, x_full[:48]), (16, 64, x_full[48:64])]:
        pad_in, pad_out = torch.zeros(T, D), torch.zeros(T, D)
        pad_in[:n_tok] = chunk
        wctx.attn_metadata = types.SimpleNamespace(
            state_indices_tensor=torch.tensor([0]), query_start_loc=torch.tensor([0, n_tok]), seq_lens=torch.tensor([ctx_len]),
        )
        tpu_v5e_patches._tpu_xla_short_conv_forward(layer, pad_in, pad_out)
        outs.append(pad_out[:n_tok].detach().numpy())

    max_diff = float(np.max(np.abs(np.vstack(outs) - ref_np)))
    assert max_diff < 1e-3


def test_3e_target_ctx_list_dedup_and_filter():
    def get_targets(max_model_len):
        return list(dict.fromkeys(t for t in (32768, max_model_len) if t <= max_model_len))

    assert get_targets(16384) == [16384]
    assert get_targets(32768) == [32768]
    assert get_targets(100000) == [32768, 100000]
    assert get_targets(200000) == [32768, 200000]
