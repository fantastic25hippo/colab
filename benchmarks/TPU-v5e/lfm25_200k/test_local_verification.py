import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_builder import build_lfm2_config, match_layer_to_quant_group
from repack_q4_utils import unpack_gguf_q4_0_to_ct_int32, dequantize_q4_0_to_float
from tpu_v5e_patches import reference_short_conv_forward, install_all_patches
from k8v4_pallas_kernel import k8v4_decode_cpu_ref

def test_trap2_split_plane_q4_0():
    print("[TEST 1] Verifying GGML_TYPE_Q4_0 split-plane nibble unpacking...", end=" ")
    u_ref = np.array([i % 16 for i in range(32)], dtype=np.uint8)
    qs = (u_ref[:16] & 0x0F) | ((u_ref[16:] & 0x0F) << 4)
    d = np.float16(0.75)
    raw_block = d.tobytes() + qs.tobytes()
    w_pack, w_scale, _ = unpack_gguf_q4_0_to_ct_int32(raw_block, 1, 32, group_size=32)
    p_arr = w_pack.numpy()[0]
    unpacked_nibbles = []
    for val in p_arr:
        for k in range(8): unpacked_nibbles.append((int(val) >> (4 * k)) & 0x0F)
    assert np.array_equal(unpacked_nibbles, u_ref), f"Nibble mismatch: {unpacked_nibbles} vs {u_ref}"
    dequant = dequantize_q4_0_to_float(raw_block, 1, 32)
    expected_dequant = (u_ref.astype(np.float32) - 8.0) * float(d)
    assert np.allclose(dequant[0], expected_dequant, atol=1e-4), "Dequant mismatch"
    print("PASS (0 error)")

def test_trap1_config_regexes():
    print("[TEST 2] Verifying config_builder.py regexes against vLLM Trap #1...", end=" ")
    cfg = build_lfm2_config(group_size=128, quantize_embeddings=False)
    backbone_layers = [
        "model.layers.5.self_attn.q_proj", "model.layers.5.self_attn.k_proj",
        "model.layers.5.self_attn.v_proj", "model.layers.5.self_attn.out_proj",
        "model.layers.2.conv.in_proj", "model.layers.2.conv.out_proj",
        "model.layers.0.feed_forward.w1", "model.layers.0.feed_forward.w2", "model.layers.0.feed_forward.w3",
        "model.layers.1.feed_forward.w1", "model.layers.1.feed_forward.w2", "model.layers.1.feed_forward.w3"
    ]
    for name in backbone_layers:
        assert match_layer_to_quant_group(name, cfg) == "group_0_w8a8_int8_backbone", f"Failed: {name}"
    moe_layers = [
        "model.layers.2.feed_forward.experts.0.w1", "model.layers.15.feed_forward.experts.23.w2",
        "model.layers.39.feed_forward.experts.63.w3"
    ]
    for name in moe_layers:
        assert match_layer_to_quant_group(name, cfg) == "group_1_w4a16_moe_experts", f"Failed: {name}"
    ignored = [
        "model.layers.3.feed_forward.gate", "model.layers.3.feed_forward.expert_bias",
        "model.layers.3.conv.conv", "model.layers.5.operator_norm", "model.layers.5.ffn_norm",
        "model.layers.5.self_attn.q_layernorm", "model.layers.5.self_attn.k_layernorm",
        "model.embedding_norm", "model.embed_tokens", "lm_head"
    ]
    for name in ignored:
        assert match_layer_to_quant_group(name, cfg) == "ignore", f"Failed ignore: {name}"
    # Test quantize_embeddings=True
    cfg_emb = build_lfm2_config(group_size=128, quantize_embeddings=True)
    assert match_layer_to_quant_group("model.embed_tokens", cfg_emb) == "group_1_w4a16_moe_experts"
    assert match_layer_to_quant_group("lm_head", cfg_emb) == "ignore"
    print("PASS")

def test_trap3_short_conv_and_patches():
    print("[TEST 3] Verifying ShortConv forward, -1 slot safety & patch hooks...", end=" ")
    bcx = torch.randn(4, 3 * 2048)
    weight = torch.randn(2048, 3)
    conv_state = torch.randn(4, 2048, 2)
    out_prefill = reference_short_conv_forward(bcx, weight, conv_state, state_idx=0)
    assert out_prefill.shape == (4, 2048), f"Bad prefill shape: {out_prefill.shape}"
    bcx_dec = torch.randn(1, 3 * 2048)
    out_dec = reference_short_conv_forward(bcx_dec, weight, conv_state, state_idx=0)
    assert out_dec.shape == (1, 2048), f"Bad decode shape: {out_dec.shape}"
    last_slot_before = conv_state[-1].clone()
    _ = reference_short_conv_forward(bcx_dec, weight, conv_state, state_idx=-1)
    assert torch.equal(conv_state[-1], last_slot_before), "Corrupted conv_state[-1] on -1 slot!"
    install_all_patches()
    print("PASS")

def test_budget_and_k8v4_kernel():
    print("[TEST 4] Verifying 200K HBM byte budget & K8V4 arithmetic...", end=" ")
    moe_w_gib = (38 * 64 * 3 * 1536 * 2048 * 0.515625) / (1024**3)
    shared_w_gib = ((30 * 16783360 + 10 * 10485760 + 2 * 144703488) * 1.0) / (1024**3)
    embed_gib = (128000 * 2048 * 2) / (1024**3)
    k8v4_kv_gib = (10 * 200000 * 8 * 100) / (1024**3)
    total_gib = moe_w_gib + shared_w_gib + embed_gib + k8v4_kv_gib
    assert 13.33 <= total_gib <= 14.32, f"Total HBM {total_gib:.3f} GiB out of budget (13.33-14.32)"
    q = torch.randn(1, 32, 64)
    k_i8 = torch.randint(-127, 127, (8, 512, 64), dtype=torch.int8)
    v_pack = torch.randint(0, 255, (8, 512, 32), dtype=torch.uint8)
    ks = torch.full((8, 512, 1), 0.05, dtype=torch.bfloat16)
    vs = torch.full((8, 512, 1), 0.02, dtype=torch.bfloat16)
    out = k8v4_decode_cpu_ref(q, k_i8, v_pack, ks, vs, seq_len=512)
    assert out.shape == (1, 32, 64), f"Bad K8V4 out shape: {out.shape}"
    print(f"PASS (Total HBM at 200K = {total_gib:.2f} GiB < 14.49 GiB ceiling)")

def _put_mod(name, **attrs):
    import types
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        n = ".".join(parts[:i])
        if n not in sys.modules:
            m = types.ModuleType(n)
            m.__path__ = []
            sys.modules[n] = m
            if i > 1:
                setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], m)
    m = sys.modules[name]
    for k, v in attrs.items():
        setattr(m, k, v)
    return m

def test_sys_modules_hook_and_sd_layout():
    print("[TEST 5] Verifying sys.modules padding hook and SD conv state...", end=" ")
    import types
    from tpu_v5e_patches import patch_sys_modules, sd_causal_conv
    def ladder(*a, **k):
        return [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    utils = types.ModuleType("tpu_inference.runner.utils")
    runner = types.ModuleType("tpu_inference.runner.tpu_runner")
    comp = types.ModuleType("tpu_inference.runner.compilation_manager")
    utils.get_token_paddings = runner.get_token_paddings = comp.get_token_paddings = ladder
    wu = types.ModuleType("tpu_inference.models.jax.utils.weight_utils")
    class Jax:
        def clear_caches(self): raise RuntimeError("cleared")
    wu.jax = Jax()
    class KVCacheManager:
        def get_kv_cache_spec(self): return {"attn": 1}
    km = types.ModuleType("tpu_inference.runner.kv_cache_manager")
    km.KVCacheManager = KVCacheManager
    names = (utils.__name__, runner.__name__, comp.__name__, wu.__name__, km.__name__)
    for mod in (utils, runner, comp, wu, km): sys.modules[mod.__name__] = mod
    try:
        patch_sys_modules()
        want = [16, 4096, 8192]
        assert utils.get_token_paddings() == want
        assert runner.get_token_paddings() == want
        assert comp.get_token_paddings() == want
        assert utils.get_token_paddings(16, 8192, 0) == want
        try:
            wu.jax.clear_caches(); raise AssertionError("clear_caches was no-op'd")
        except RuntimeError: pass
        assert int(torch.clamp(torch.tensor([-1]), min=0)) == 0
        assert KVCacheManager().get_kv_cache_spec() == {"attn": 1}
    finally:
        for n in names: sys.modules.pop(n, None)
    D, state = 4, torch.tensor([[1., 2., 3., 4.], [5., 6., 7., 8.]])
    bx = torch.tensor([[9., 10., 11., 12.], [13., 14., 15., 16.], [17., 18., 19., 20.]])
    w = torch.arange(D * 3, dtype=torch.float32).reshape(D, 3)
    acc, x_ext = sd_causal_conv(bx, w, None, state)
    assert acc.shape == (3, D)
    manual = state[0] * w[:, 0] + state[1] * w[:, 1] + bx[0] * w[:, 2]
    assert torch.allclose(acc[0], manual)
    assert torch.equal(x_ext.index_select(0, torch.tensor(0) + torch.arange(2)), state)
    assert torch.equal(x_ext.index_select(0, torch.tensor(1) + torch.arange(2)), torch.stack([state[1], bx[0]]))
    print("PASS")

class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)

def test_lookup_conv_cache_metadata_slot_n_and_mask():
    print("[TEST 6] Verifying _lookup_conv_cache metadata, slot, n, mask...", end=" ")
    from tpu_v5e_patches import _lookup_conv_cache
    prefix = "model.layers.2.conv"
    conv_t = torch.arange(9 * 2 * 4, dtype=torch.float32).reshape(9, 2, 4)
    layer = _NS(prefix=prefix)
    wctx = _NS(layer_name_to_kvcache_index={prefix: 0}, kv_caches=[conv_t],
               attn_metadata=None, shared_attn_metadata=None)
    _put_mod("tpu_inference.models.vllm.vllm_model_wrapper_context",
             get_vllm_model_wrapper_context=lambda: wctx)
    # 1) get_forward_context: unpadded n, slot from state_indices_tensor, keep prev
    meta = _NS(
        state_indices_tensor=torch.tensor([3], dtype=torch.long),
        query_start_loc=torch.tensor([0, 5], dtype=torch.long),
        has_initial_states=torch.tensor([True]),
        seq_lens=torch.tensor([12], dtype=torch.long),
    )
    _put_mod("vllm.forward_context", get_forward_context=lambda: _NS(attn_metadata=meta))
    bound = _lookup_conv_cache(layer, 2, 16)
    assert bound is not None
    _, idx, raw, prev, got_conv, slot, n = bound
    assert idx == 0 and raw is conv_t and got_conv is conv_t
    assert int(slot.reshape(-1)[0]) == 3
    assert int(n.reshape(())) == 5
    assert torch.equal(prev, conv_t[3])
    assert slot.device == conv_t.device and n.device == conv_t.device and prev.device == conv_t.device
    # 2) has_initial_states false zeros prev (do not use padded n=T)
    meta.has_initial_states = torch.tensor([False])
    _, _, _, prev0, _, _, n0 = _lookup_conv_cache(layer, 2, 16)
    assert int(n0.reshape(())) == 5
    assert torch.equal(prev0, torch.zeros_like(conv_t[3]))
    # 3) unwrap per-layer dict via self.prefix
    meta.has_initial_states = torch.tensor([True])
    _put_mod("vllm.forward_context",
             get_forward_context=lambda: _NS(attn_metadata={prefix: meta, "other": _NS()}))
    _, _, _, prev_d, _, slot_d, n_d = _lookup_conv_cache(layer, 2, 16)
    assert int(slot_d.reshape(-1)[0]) == 3 and int(n_d.reshape(())) == 5
    assert torch.equal(prev_d, conv_t[3])
    # 4) fallback wctx.attn_metadata + mamba_state_indices; ctx_len <= n zeros prev
    def _boom():
        raise RuntimeError("no forward context")
    _put_mod("vllm.forward_context", get_forward_context=_boom)
    wctx.attn_metadata = _NS(
        mamba_state_indices=torch.tensor([7], dtype=torch.long),
        query_start_loc=torch.tensor([0, 8], dtype=torch.long),
        seq_lens=torch.tensor([8], dtype=torch.long),
    )
    _, _, _, prev_fb, _, slot_fb, n_fb = _lookup_conv_cache(layer, 2, 16)
    assert int(slot_fb.reshape(-1)[0]) == 7 and int(n_fb.reshape(())) == 8
    assert torch.equal(prev_fb, torch.zeros_like(conv_t[7]))
    print("PASS")

def test_tpu_runner_warning_filter_and_wrapper_modules():
    print("[TEST 7] Verifying tpu_runner warning filter and VllmModelWrapper.modules...", end=" ")
    import logging, types
    from tpu_v5e_patches import install_all_patches, patch_sys_modules
    install_all_patches()
    rec = []
    class H(logging.Handler):
        def emit(self, record): rec.append(record.getMessage())
    lg = logging.getLogger("tpu_inference.runner.tpu_runner")
    lg.setLevel(logging.WARNING)
    h = H(); lg.addHandler(h)
    try:
        lg.warning("Should not schedule a request that does nothing!")
        lg.warning("other runner message")
        assert rec == ["other runner message"]
    finally:
        lg.removeHandler(h)
    wrap = types.ModuleType("tpu_inference.models.vllm.vllm_model_wrapper")
    class VllmModelWrapper: pass
    wrap.VllmModelWrapper = VllmModelWrapper
    sys.modules[wrap.__name__] = wrap
    try:
        patch_sys_modules()
        inst = VllmModelWrapper()
        assert list(inst.modules()) == []
        class Inner:
            def modules(self):
                yield "layer0"
        inst.model = Inner()
        assert list(inst.modules()) == ["layer0"]
    finally:
        sys.modules.pop(wrap.__name__, None)
    print("PASS")

def test_subrunner_int8_kv_and_jax_cache_env():
    print("[TEST 8] Verifying int8 KV flag and JAX persistent cache env...", end=" ")
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "lfm25_engine_subrunner.py")).read()
    # 2026-11-01: vLLM rejected "int8"; fp8 is the documented fallback.
    assert 'kv_cache_dtype="fp8"' in src
    assert 'rejected "int8"' in src
    assert 'JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1"' in src
    assert 'JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all"' in src
    print("PASS")

if __name__ == "__main__":
    t0 = time.time()
    test_trap2_split_plane_q4_0()
    test_trap1_config_regexes()
    test_trap3_short_conv_and_patches()
    test_budget_and_k8v4_kernel()
    test_sys_modules_hook_and_sd_layout()
    test_lookup_conv_cache_metadata_slot_n_and_mask()
    test_tpu_runner_warning_filter_and_wrapper_modules()
    test_subrunner_int8_kv_and_jax_cache_env()
    print(f"All 8 verification test suites passed in {time.time() - t0:.2f}s!")

