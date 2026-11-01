import contextlib, io, os, sys, types, unittest.mock as mock, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpu_v5e_patches as P

class _NS:
    def __init__(self, **kw): self.__dict__.update(kw)

def _put_mod(name, **attrs):
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        n = ".".join(parts[:i])
        if n not in sys.modules:
            m = types.ModuleType(n); m.__path__ = []; sys.modules[n] = m
            if i > 1: setattr(sys.modules[".".join(parts[:i - 1])], parts[i - 1], m)
    m = sys.modules[name]
    for k, v in attrs.items(): setattr(m, k, v)
    return m

def _ladder(*a, **k): return [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

def test_diag_padding_buckets():
    print("[DIAG TEST 1] Verifying LFM_DIAG padding buckets...", end=" ")
    mods = [_put_mod(f"tpu_inference.runner.{n}", get_token_paddings=_ladder)
            for n in ("utils", "tpu_runner", "compilation_manager")]
    try:
        P.patch_sys_modules()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LFM_DIAG", None)
            assert mods[0].get_token_paddings() == [16, 8192]
        with mock.patch.dict(os.environ, {"LFM_DIAG": "1"}, clear=False):
            for m in mods: assert m.get_token_paddings() == [16, 4096]
            assert mods[0].get_token_paddings(16, 8192, 0) == [16, 4096]
    finally:
        for m in mods: sys.modules.pop(m.__name__, None)
    print("PASS")

def test_conv_debug_logging():
    print("[DIAG TEST 2] Verifying CONVDBG logging (one layer, 1 prefill + 3 decode)...", end=" ")
    prefix = "model.layers.2.conv"
    conv_t = torch.ones(9, 2, 4)
    layer = _NS(prefix=prefix)
    wctx = _NS(layer_name_to_kvcache_index={prefix: 0}, kv_caches=[conv_t])
    _put_mod("tpu_inference.models.vllm.vllm_model_wrapper_context",
             get_vllm_model_wrapper_context=lambda: wctx)
    try:
        def call(n_val, has_init=None, seq=12):
            meta = _NS(state_indices_tensor=torch.tensor([3]),
                       query_start_loc=torch.tensor([0, n_val]),
                       seq_lens=torch.tensor([seq]))
            if has_init is not None: meta.has_initial_states = torch.tensor([has_init])
            _put_mod("vllm.forward_context", get_forward_context=lambda: _NS(attn_metadata=meta))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf): P._lookup_conv_cache(layer, 2, 16)
            return buf.getvalue()

        def reset(): P._CONV_DBG.update({"layer": None, "prefill": 0, "decode": 0})

        reset()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LFM_CONV_DEBUG", None)
            assert "CONVDBG" not in call(5)
        reset()
        with mock.patch.dict(os.environ, {"LFM_CONV_DEBUG": "1"}, clear=False):
            pre = call(5, has_init=False)
            assert "CONVDBG" in pre and "kind=prefill" in pre, pre
            assert "slot=3" in pre and "n=5" in pre and "keep=False" in pre, pre
            assert "seq_lens=12" in pre and "has_initial_states=False" in pre, pre
            assert "prev_abs_sum=0.0" in pre, pre
            assert "CONVDBG" not in call(5), "second prefill must not log"
            dec = [call(1, has_init=True, seq=12) for _ in range(4)]
            assert all("kind=decode" in d and "keep=True" in d for d in dec[:3]), dec
            assert "prev_abs_sum=8.0" in dec[0], dec[0]
            assert "CONVDBG" not in dec[3], "fourth decode must not log"
            reset()
            miss = call(5, seq=12)
            assert "has_initial_states=missing" in miss and "keep=True" in miss, miss
            assert "prev_abs_sum=8.0" in miss, miss
            wctx.layer_name_to_kvcache_index["model.layers.6.conv"] = 1
            wctx.kv_caches.append(torch.ones(9, 2, 4))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                P._lookup_conv_cache(_NS(prefix="model.layers.6.conv"), 2, 16)
            assert "CONVDBG" not in buf.getvalue(), "only the first conv layer logs"
    finally:
        sys.modules.pop("tpu_inference.models.vllm.vllm_model_wrapper_context", None)
        sys.modules.pop("vllm.forward_context", None)
    print("PASS")

def test_diag_wiring_static():
    print("[DIAG TEST 3] Verifying diag subrunner + driver wiring (static)...", end=" ")
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "diag_short_subrunner.py")).read()
    for needle in ("max_model_len=4096", "max_num_batched_tokens=4096", "max_num_seqs=1",
                   "enable_prefix_caching=False", "trust_remote_code=True",
                   "install_all_patches()", "temperature=0.1", "top_k=50",
                   "repetition_penalty=1.05", "temperature=0.0", "ignore_eos=True",
                   "add_generation_prompt=True", "124894"):
        assert needle in src, f"missing in diag subrunner: {needle}"
    for banned in ("kv_cache_dtype", "max_model_len=200000", "32768", "200000", "seed=0"):
        assert banned not in src, f"diag subrunner still contains: {banned}"
    drv = open(os.path.join(here, "lfm25_coldstart_driver.py")).read()
    assert "diag_short_subrunner.py" in drv and "lfm25_engine_subrunner.py" in drv
    assert "LFM_CONV_DEBUG" in drv and 'env_b.update({"LFM_DIAG": "1"' in drv
    run = open(os.path.join(here, "run_lfm25_tpu.py")).read()
    assert '"--diag"' in run and "run_results_diag.json" in run and '"1800" if diag' in run
    print("PASS")

if __name__ == "__main__":
    test_diag_padding_buckets()
    test_conv_debug_logging()
    test_diag_wiring_static()
    print("All 3 diag test suites passed!")
