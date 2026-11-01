"""Local (no TPU, no vLLM) tests for the 2026-11-01 startup/template changes."""

import os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lfm25_prompt import BOS_ID, build_long_prompt_ids, check_prompt_ids, filler_text
from lfm25_cache import cache_filename, extract_cache_key_args
from lfm25_startup import summarize_timeline

HERE = os.path.dirname(os.path.abspath(__file__))


class StubTok:
    """Chat template stub: HEAD + char ids + TAIL, like the LFM jinja file."""
    HEAD = [BOS_ID, 124899, 5922, 207]
    TAIL = [124900, 207, 124899, 63514, 207, 124901]

    def encode(self, s, **kw):
        return [10 + (ord(c) % 60) for c in s]

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True):
        return self.HEAD + self.encode(msgs[0]["content"]) + self.TAIL


def test_long_prompt_exact_len_and_single_bos():
    print("[TEST A] long prompt: exact ids, single BOS, template tail kept...", end=" ")
    tok = StubTok()
    filler = filler_text(tok, "Explain astrophysics. ", 2000)
    for target in (500, 2000):
        ids, meta = build_long_prompt_ids(tok, filler, target)
        assert len(ids) == target, (len(ids), target)
        assert ids[0] == BOS_ID and ids.count(BOS_ID) == 1
        assert ids[-len(StubTok.TAIL):] == StubTok.TAIL, "generation prompt tail lost"
        assert meta["tail_n"] == len(StubTok.TAIL)
    print("PASS")


def test_long_prompt_rejects_short_filler_and_bad_bos():
    print("[TEST B] guards: short filler, double BOS, wrong length...", end=" ")
    tok = StubTok()
    for bad in ([1, 2, 3], [BOS_ID, BOS_ID, 5]):
        try:
            check_prompt_ids(bad)
            raise AssertionError(f"check_prompt_ids accepted {bad}")
        except AssertionError as e:
            assert "BOS" in str(e), e
    try:
        check_prompt_ids([BOS_ID, 5], target=3)
        raise AssertionError("accepted wrong length")
    except AssertionError as e:
        assert "expected 3" in str(e), e
    try:
        build_long_prompt_ids(tok, "tiny", 100000)
        raise AssertionError("accepted filler shorter than target")
    except ValueError as e:
        assert "need >=" in str(e), e
    print("PASS")


def test_cache_filename_key():
    print("[TEST C] cache filename: version + args + patches hash...", end=" ")
    a = cache_filename("0.32.0", "max_model_len=200000", "abc123")
    assert re.fullmatch(r"tpu_compile_cache__lfm2\.5-24b-a2b__vllm0\.32\.0__len200000__b16-4096-200000__[0-9a-f]{8}\.tar", a), a
    assert a == cache_filename("0.32.0", "max_model_len=200000", "abc123")
    assert a != cache_filename("0.32.1", "max_model_len=200000", "abc123")
    assert a != cache_filename("0.32.0", "max_model_len=200000;kv=auto", "abc123")
    assert a != cache_filename("0.32.0", "max_model_len=200000", "abc124")
    assert "b1_32k" not in a
    print("PASS")


def test_extract_cache_key_args():
    print("[TEST D] CACHE_KEY_ARGS extraction from source text...", end=" ")
    src = 'CACHE_KEY_ARGS = ("max_model_len=200000;"\n                  "kv=auto;buckets=16|4096|8192")\n'
    assert extract_cache_key_args(src) == "max_model_len=200000;kv=auto;buckets=16|4096|8192"
    assert extract_cache_key_args("no constant here") == ""
    real = open(os.path.join(HERE, "lfm25_engine_subrunner.py")).read()
    got = extract_cache_key_args(real)
    assert "max_model_len=200000" in got and "kv=auto" in got and "buckets=16|4096|8192" in got, got
    print("PASS")


def test_timeline_summary():
    print("[TEST E] log timeline -> per-bucket compile / warmup / shards...", end=" ")
    rows = [
        {"t": 1.0, "msg": "Loading weights from /dev/shm/lfm25_200k/model-00001.safetensors"},
        {"t": 2.0, "msg": "Compilation of backbone --> {'num_tokens': 16, 'num_reqs': 8} finished in 9.64 [secs]."},
        {"t": 3.0, "msg": "Compilation of backbone --> {'num_tokens': 4096, 'num_reqs': 8} finished in 38.57 [secs]."},
        {"t": 4.0, "msg": "Compilation of backbone --> {'num_tokens': 8192, 'num_reqs': 8} finished in 38.45 [secs]."},
        {"t": 5.0, "msg": "Warm-up call pass finished in 12.19 [secs] over 3 tasks."},
        {"t": 6.0, "msg": "PERSISTENT COMPILATION CACHE MISS for x"},
    ]
    s = summarize_timeline(rows)
    assert s["compile_per_bucket_s"] == {"16": 9.64, "4096": 38.57, "8192": 38.45}, s
    assert s["compile_runs_per_bucket"] == {"16": 1, "4096": 1, "8192": 1}
    assert s["warmup_pass"]["dur_s"] == 12.19 and s["warmup_pass"]["tasks"] == 3
    assert s["weight_shard_loads"][0]["file"] == "model-00001.safetensors"
    assert s["persistent_cache_misses"] == 1
    print("PASS")


def test_static_wiring():
    print("[TEST F] wiring: no Qwen cache, default KV, chat template, 6 blobs...", end=" ")
    drv = open(os.path.join(HERE, "lfm25_coldstart_driver.py")).read()
    sub = open(os.path.join(HERE, "lfm25_engine_subrunner.py")).read()
    run = open(os.path.join(HERE, "run_lfm25_tpu.py")).read()
    assert "tpu_compile_cache_b1_32k.tar" not in drv
    for needle in ("restore_cache(", "publish_cache(", "cache_filename(", "LFM_CACHE_FORCE_COLD",
                   "shm_read_probe(", "weight_manifest(", "PROMPT_B64", "STARTUP_B64", "CACHE_B64"):
        assert needle in drv, f"driver missing {needle}"
    assert "kv_cache_dtype=" not in sub, "subrunner must leave kv_cache_dtype at default"
    for needle in ("add_generation_prompt=True", "CACHE_KEY_ARGS", "max_tokens=1",
                   "ignore_eos=True", "install_log_timeline(", "dir_stats()"):
        assert needle in sub, f"subrunner missing {needle}"
    assert 'f"{here}/lfm25_prompt.py"' in run and 'f"{here}/lfm25_cache.py"' in run
    assert 'f"{here}/lfm25_startup.py"' in run
    print("PASS")


if __name__ == "__main__":
    test_long_prompt_exact_len_and_single_bos()
    test_long_prompt_rejects_short_filler_and_bad_bos()
    test_cache_filename_key()
    test_extract_cache_key_args()
    test_timeline_summary()
    test_static_wiring()
    print("All 6 startup/template test suites passed!")
