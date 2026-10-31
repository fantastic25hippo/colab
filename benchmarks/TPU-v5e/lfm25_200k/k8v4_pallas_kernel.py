import functools, math, numpy as np, torch
try:
    import jax, jax.numpy as jnp
    from jax import lax
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

def _pallas_k8v4_kernel_body(seq_len_ref, q_int8_ref, q_sc_ref, k_tile_ref, v_pack_ref,
                             ks_tile_ref, vs_tile_ref, m_ref, l_ref, acc_ref, out_ref, *, b_kv, num_tiles):
    t_idx = pl.program_id(0)
    @pl.when(t_idx == 0)
    def _init():
        m_ref[...] = jnp.full_like(m_ref, -1e30)
        l_ref[...] = jnp.zeros_like(l_ref)
        acc_ref[...] = jnp.zeros_like(acc_ref)
    t_start, s_len = t_idx * b_kv, seq_len_ref[0]
    @pl.when(t_start < s_len)
    def _step():
        q_i8, q_sc = q_int8_ref[0], q_sc_ref[0]
        k_i8, k_sc = k_tile_ref[0], ks_tile_ref[0].astype(jnp.float32)
        logits_i32 = lax.dot_general(q_i8, k_i8, (((2,), (2,)), ((0,), (0,))), preferred_element_type=jnp.int32)
        logits = logits_i32.astype(jnp.float32) * q_sc * jnp.swapaxes(k_sc, 1, 2)
        offs = t_start + lax.broadcasted_iota(jnp.int32, (8, 1, b_kv), 2)
        logits = jnp.where(offs < s_len, logits, -1e30)
        m_prev, m_curr = m_ref[0], jnp.max(logits, axis=-1, keepdims=True)
        m_new = jnp.maximum(m_prev, m_curr)
        alpha, p = jnp.exp(m_prev - m_new), jnp.exp(logits - m_new)
        l_new = alpha * l_ref[0] + jnp.sum(p, axis=-1, keepdims=True)
        v_u8 = v_pack_ref[0]
        v_lo = ((v_u8 & 0x0F).astype(jnp.int8) - 8).astype(jnp.bfloat16)
        v_hi = (((v_u8 >> 4) & 0x0F).astype(jnp.int8) - 8).astype(jnp.bfloat16)
        v_bf16 = jnp.concatenate([v_lo, v_hi], axis=-1)
        v_sc = vs_tile_ref[0].astype(jnp.bfloat16)
        p_scaled = (p.astype(jnp.bfloat16) * jnp.swapaxes(v_sc, 1, 2))
        pv = lax.dot_general(p_scaled, v_bf16, (((2,), (1,)), ((0,), (0,))), preferred_element_type=jnp.float32)
        acc_ref[0] = alpha * acc_ref[0] + pv
        m_ref[0], l_ref[0] = m_new, l_new
    @pl.when(t_idx == num_tiles - 1)
    def _done():
        out_ref[0] = (acc_ref[0] / jnp.maximum(l_ref[0], 1e-9)).astype(jnp.bfloat16)

def pallas_k8v4_decode(q, k_cache, v_cache, ks_cache, vs_cache, seq_len):
    if not HAS_JAX: raise RuntimeError("JAX/Pallas not installed")
    b_kv, max_len = 512, k_cache.shape[1]
    num_tiles = max_len // b_kv
    sm_sc = 1.0 / math.sqrt(64.0)
    q_gqa = q.reshape(8, 4, 64)
    q_abs = jnp.max(jnp.abs(q_gqa), axis=-1, keepdims=True).astype(jnp.float32)
    q_sc = jnp.maximum(q_abs / 127.0, 1e-6)
    q_i8 = jnp.clip(jnp.round(q_gqa.astype(jnp.float32) / q_sc), -127.0, 127.0).astype(jnp.int8)
    q_pad = jnp.pad(q_i8, ((0, 0), (0, 124), (0, 0)))[None, ...]
    qs_pad = jnp.pad(q_sc * sm_sc, ((0, 0), (0, 124), (0, 0)))[None, ...]
    grid = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1, grid=(num_tiles,),
        in_specs=[
            pl.BlockSpec((1, 8, 128, 64), lambda i, _: (0, 0, 0, 0)),
            pl.BlockSpec((1, 8, 128, 1), lambda i, _: (0, 0, 0, 0)),
            pl.BlockSpec((1, 8, b_kv, 64), lambda i, _: (0, 0, i, 0)),
            pl.BlockSpec((1, 8, b_kv, 32), lambda i, _: (0, 0, i, 0)),
            pl.BlockSpec((1, 8, b_kv, 1), lambda i, _: (0, 0, i, 0)),
            pl.BlockSpec((1, 8, b_kv, 1), lambda i, _: (0, 0, i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((1, 8, 128, 1), lambda i, _: (0, 0, 0, 0)),
            pl.BlockSpec((1, 8, 128, 1), lambda i, _: (0, 0, 0, 0)),
            pl.BlockSpec((1, 8, 128, 64), lambda i, _: (0, 0, 0, 0)),
            pl.BlockSpec((1, 8, 128, 64), lambda i, _: (0, 0, 0, 0)),
        ]
    )
    res = pl.pallas_call(functools.partial(_pallas_k8v4_kernel_body, b_kv=b_kv, num_tiles=num_tiles),
                         grid_spec=grid,
                         out_shape=[jax.ShapeDtypeStruct((1, 8, 128, 1), jnp.float32),
                                    jax.ShapeDtypeStruct((1, 8, 128, 1), jnp.float32),
                                    jax.ShapeDtypeStruct((1, 8, 128, 64), jnp.float32),
                                    jax.ShapeDtypeStruct((1, 8, 128, 64), jnp.bfloat16)])(
        seq_len, q_pad, qs_pad, k_cache[None, ...], v_cache[None, ...], ks_cache[None, ...], vs_cache[None, ...]
    )[3]
    return res[:, :, :4, :].reshape(1, 32, 64)

def k8v4_decode_cpu_ref(q_th, k_i8_th, v_pack_th, ks_th, vs_th, seq_len: int):
    # Reference CPU flash-decode with INT8 QK and absorbed V scale
    B, H_q, D = q_th.shape
    H_kv, G = 8, H_q // 8
    q_gqa = q_th.view(H_kv, G, D).float()
    sm_scale = 1.0 / math.sqrt(D)
    q_sc = torch.clamp(torch.amax(torch.abs(q_gqa), dim=-1, keepdim=True) / 127.0, min=1e-6)
    q_i8 = torch.clamp(torch.round(q_gqa / q_sc), -127.0, 127.0)
    k_i8 = k_i8_th[:, :seq_len, :].float()
    ks = ks_th[:, :seq_len, :].float()
    logits_i32 = torch.bmm(q_i8, k_i8.transpose(1, 2))
    logits = logits_i32 * (q_sc * sm_scale) * ks.transpose(1, 2)
    p = torch.softmax(logits, dim=-1)
    vs = vs_th[:, :seq_len, :].float()
    p_scaled = p * vs.transpose(1, 2)
    v_u8 = v_pack_th[:, :seq_len, :].to(torch.uint8)
    v_lo = (v_u8 & 0x0F).to(torch.int8) - 8
    v_hi = ((v_u8 >> 4) & 0x0F).to(torch.int8) - 8
    v_unpacked = torch.cat([v_lo, v_hi], dim=-1).float()
    out = torch.bmm(p_scaled, v_unpacked)
    return out.view(1, H_q, D).to(q_th.dtype)
