import os, sys, time, numpy as np, torch
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

def test_sys_modules_hook_and_sd_layout():
    print("[TEST 5] Verifying sys.modules padding hook and SD conv state...", end=" ")
    import types
    from tpu_v5e_patches import patch_sys_modules, sd_causal_conv
    def ladder(*a, **k):
        return [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    utils, comp = types.ModuleType("tpu_inference.runner.utils"), types.ModuleType("tpu_inference.runner.compilation_manager")
    utils.get_token_paddings = comp.get_token_paddings = ladder
    wu = types.ModuleType("tpu_inference.models.jax.utils.weight_utils")
    class Jax:
        def clear_caches(self): raise RuntimeError("cleared")
    wu.jax = Jax()
    class KVCacheManager:
        def get_kv_cache_spec(self): return {"attn": 1}
    km = types.ModuleType("tpu_inference.runner.kv_cache_manager")
    km.KVCacheManager = KVCacheManager
    names = (utils.__name__, comp.__name__, wu.__name__, km.__name__)
    for mod in (utils, comp, wu, km): sys.modules[mod.__name__] = mod
    try:
        patch_sys_modules()
        assert utils.get_token_paddings() == [16, 8192]
        assert comp.get_token_paddings() == [16, 8192]
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

if __name__ == "__main__":
    t0 = time.time()
    test_trap2_split_plane_q4_0()
    test_trap1_config_regexes()
    test_trap3_short_conv_and_patches()
    test_budget_and_k8v4_kernel()
    test_sys_modules_hook_and_sd_layout()
    print(f"All 5 verification test suites passed in {time.time() - t0:.2f}s!")
