# LFM_BEARINGS unit tests
import contextlib, io, os, sys, types, unittest.mock as mock, pytest
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tpu_v5e_patches as P

class _NS:
    def __init__(self, **kw): self.__dict__.update(kw)

class _Arr:
    def __init__(self, shape, dtype="float32", nbytes=1024):
        self.shape, self.dtype, self.nbytes = shape, dtype, nbytes

class FullAttentionSpec:
    def __init__(self, psize=4096, msize=8192, layers=None):
        self.page_size_bytes, self._msize = psize, msize
        if layers is not None: self.layers = layers
    def max_memory_usage_bytes(self, cfg): return self._msize

class MambaSpec:
    def __init__(self, psize=1024, msize=2048, layers=None):
        self.page_size_bytes, self._msize = psize, msize
        if layers is not None: self.layers = layers
    def max_memory_usage_bytes(self, cfg): return self._msize

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

def _setup_mock_modules():
    jax_dev = _NS(memory_stats=lambda: {"bytes_in_use": 1024**3, "bytes_limit": 4 * 1024**3})
    _put_mod("jax", devices=lambda: [jax_dev])
    kv_mod = _put_mod("vllm.v1.core.kv_cache_utils",
                      max_memory_usage_bytes=lambda cfg, specs: sum(s.max_memory_usage_bytes(cfg) for s in specs))
    class MockKVCacheManager:
        def initialize_kv_cache(self, *a, **kw):
            return [_Arr((2, 16, 64)), (_Arr((1, 4)), _Arr((1, 4)))]
    mgr_mod = _put_mod("tpu_inference.runner.kv_cache_manager", KVCacheManager=MockKVCacheManager)
    class MockVllmModelWrapper:
        def load_weights(self, *a, **kw):
            m = sys.modules.get("tpu_inference.models.vllm.vllm_model_wrapper")
            v = m.vllm_get_model()
            t = m.shard_model_to_tpu(v)
            return t
    wrap_mod = _put_mod("tpu_inference.models.vllm.vllm_model_wrapper",
                        VllmModelWrapper=MockVllmModelWrapper,
                        vllm_get_model=lambda *a, **kw: {"w": 1},
                        shard_model_to_tpu=lambda *a, **kw: {"t1": 1, "t2": 2})
    P.patch_sys_modules()
    return kv_mod, MockKVCacheManager(), MockVllmModelWrapper()

@pytest.fixture(autouse=True)
def _cleanup_mock_modules():
    created = [
        "jax", "vllm.v1.core.kv_cache_utils", "vllm.v1.core", "vllm.v1", "vllm",
        "tpu_inference.runner.kv_cache_manager", "tpu_inference.runner",
        "tpu_inference.models.vllm.vllm_model_wrapper", "tpu_inference.models.vllm",
        "tpu_inference.models", "tpu_inference"
    ]
    orig = {k: sys.modules.get(k) for k in created}
    try:
        yield
    finally:
        for k, v in orig.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
        P._SEEN.clear()

def test_unset_env_silent_and_returns_orig():
    kv_mod, mgr, wrap = _setup_mock_modules()
    specs = [FullAttentionSpec(4096, 8192), MambaSpec(1024, 2048)]
    buf = io.StringIO()
    with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop("LFM_BEARINGS", None)
        os.environ.pop("LFM_BEARINGS_STOP", None)
        with contextlib.redirect_stdout(buf):
            tot = kv_mod.max_memory_usage_bytes(None, specs)
            caches = mgr.initialize_kv_cache()
            out = wrap.load_weights()
    assert tot == 10240
    assert len(caches) == 2
    assert out == {"t1": 1, "t2": 2}
    assert "[LFM_BEARINGS]" not in buf.getvalue()

def test_lfm_bearings_logs_emitted():
    kv_mod, mgr, wrap = _setup_mock_modules()
    specs = [FullAttentionSpec(4096, 8192, layers=["l1", "l2"]), MambaSpec(1024, 2048)]
    buf = io.StringIO()
    with mock.patch.dict(os.environ, {"LFM_BEARINGS": "1"}, clear=False):
        os.environ.pop("LFM_BEARINGS_STOP", None)
        with contextlib.redirect_stdout(buf):
            tot = kv_mod.max_memory_usage_bytes(None, specs)
            caches = mgr.initialize_kv_cache()
            out = wrap.load_weights()
    txt = buf.getvalue()
    assert tot == 10240 and len(caches) == 2 and out == {"t1": 1, "t2": 2}
    assert "[LFM_BEARINGS] spec class=FullAttentionSpec layers=2" in txt
    assert "[LFM_BEARINGS] total class=FullAttentionSpec" in txt
    assert "[LFM_BEARINGS] grand_total_gib=" in txt
    assert "[LFM_BEARINGS] KV array: shape=(2, 16, 64)" in txt
    assert "[LFM_BEARINGS] KV total_gib=" in txt
    assert "[LFM_BEARINGS] load CPU vllm_get_model:" in txt
    assert "[LFM_BEARINGS] load TPU shard_model_to_tpu:" in txt
    assert "tensors_moved=2" in txt
    assert "[LFM_BEARINGS] HBM after KV allocation: used=1.000 GiB" in txt
    assert "[LFM_BEARINGS] HBM after load_weights: used=1.000 GiB" in txt
    assert "HF_TOKEN" not in txt

def test_lfm_bearings_stop_raises_system_exit():
    _, mgr, _ = _setup_mock_modules()
    buf = io.StringIO()
    with mock.patch.dict(os.environ, {"LFM_BEARINGS": "1", "LFM_BEARINGS_STOP": "1"}, clear=False):
        with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as exc:
            mgr.initialize_kv_cache()
    assert exc.value.code == 0
    txt = buf.getvalue()
    assert "[LFM_BEARINGS] KV array:" in txt
    assert "[LFM_BEARINGS] KV total_gib=" in txt
    assert "[LFM_BEARINGS] HBM after KV allocation:" in txt

def test_patch_sys_modules_repeated_no_recursion():
    kv_mod, mgr, wrap = _setup_mock_modules()
    for _ in range(50):
        P.patch_sys_modules()
    buf = io.StringIO()
    with mock.patch.dict(os.environ, {"LFM_BEARINGS": "1"}, clear=False):
        os.environ.pop("LFM_BEARINGS_STOP", None)
        with contextlib.redirect_stdout(buf):
            tot = kv_mod.max_memory_usage_bytes(None, [FullAttentionSpec(4096, 8192)])
            caches = mgr.initialize_kv_cache()
            out = wrap.load_weights()
    assert tot == 8192 and len(caches) == 2 and out == {"t1": 1, "t2": 2}
    txt = buf.getvalue()
    assert txt.count("[LFM_BEARINGS] HBM after load_weights") == 1
    assert txt.count("[LFM_BEARINGS] KV total_gib=") == 1
    assert txt.count("[LFM_BEARINGS] KV array:") == 3

def test_subrunner_wiring_and_no_hf_token():
    here = os.path.dirname(os.path.abspath(__file__))
    subrunner_src = open(os.path.join(here, "lfm25_engine_subrunner.py")).read()
    assert '"SKIP_JAX_PRECOMPILE": "1"' in subrunner_src
    patches_src = open(os.path.join(here, "tpu_v5e_patches.py")).read()
    assert "HF_TOKEN" not in patches_src

def test_importing_n_dummy_modules_patches_at_most_once():
    call_counts = {}
    orig_patch = P._patch_loaded_module

    def spy_patch(name, mod):
        call_counts[name] = call_counts.get(name, 0) + 1
        return orig_patch(name, mod)

    N = 25
    dummy_names = [f"tpu_inference.dummy_submod_{i}" for i in range(N)]
    _put_mod("tpu_inference")
    with mock.patch.object(P, "_patch_loaded_module", side_effect=spy_patch):
        for name in dummy_names:
            _put_mod(name)
            P.custom_import(name)

        for _ in range(5):
            for name in dummy_names:
                P.custom_import(name)
            P.patch_sys_modules()

    for name in dummy_names:
        sys.modules.pop(name, None)
        P._SEEN.discard(name)
    sys.modules.pop("tpu_inference", None)
    P._SEEN.discard("tpu_inference")

    for name in dummy_names:
        assert call_counts.get(name, 0) <= 1
        assert call_counts.get(name, 0) == 1

def test_cache_filename_len100000():
    import lfm25_cache, re
    name = lfm25_cache.cache_filename(
        vllm_version="0.32.0", args_str="--dummy", patch_content="dummy",
        model_slug="lfm2.5-24b-a2b", max_model_len=100000, buckets="16-8192"
    )
    assert "len100000" in name
    assert re.fullmatch(r"tpu_compile_cache__lfm2\.5-24b-a2b__vllm0\.32\.0__len100000__b16-8192__[0-9a-f]{8}\.tar", name)

def test_key_args_contains_max_model_len():
    import lfm25_cache
    here = os.path.dirname(os.path.abspath(__file__))
    subrunner_src = open(os.path.join(here, "lfm25_engine_subrunner.py")).read()
    key_args = lfm25_cache.extract_cache_key_args(subrunner_src)
    assert "max_model_len=100000" in key_args

def test_in_vm_total_non_null_from_fake_timestamps():
    first_token_ts, t_vm0, t_global0 = 200.5, 100.0, 80.0
    in_vm = round(first_token_ts - t_vm0, 2) if first_token_ts else None
    total = round(first_token_ts - t_global0, 2) if first_token_ts else None
    assert in_vm is not None and in_vm == 100.5
    assert total is not None and total == 120.5
    here = os.path.dirname(os.path.abspath(__file__))
    subrunner_src = open(os.path.join(here, "lfm25_engine_subrunner.py")).read()
    idx_req = subrunner_src.find("LFM_SHORT_REQUEST")
    assert idx_req != -1
    assert "first_tok_ts = time.time()" in subrunner_src[idx_req:] or "first_token_ts = time.time()" in subrunner_src[idx_req:]



