"""Test canonical TPU compile cache naming."""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_naming import build_cache_name


def test_qwen_naming():
    patch_bytes = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen_patches.py"), "rb").read()
    name = build_cache_name(
        model_slug="qwen3-4b",
        vllm_version="0.32.0",
        max_model_len=32768,
        buckets="16-32768",
        llm_args={"gpu_memory_utilization": 0.90, "max_model_len": 32768, "max_num_seqs": 1, "trust_remote_code": True},
        patch_content=patch_bytes,
    )
    pattern = r"tpu_compile_cache__qwen3-4b__vllm0\.32\.0__len32768__b16-32768__[0-9a-f]{8}\.tar"
    assert re.fullmatch(pattern, name), f"Invalid Qwen cache name format: {name}"
    assert name == "tpu_compile_cache__qwen3-4b__vllm0.32.0__len32768__b16-32768__e4cc071f.tar"


def test_lfm_naming():
    lfm_patch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm25_200k", "tpu_v5e_patches.py")
    patch_bytes = open(lfm_patch, "rb").read() if os.path.exists(lfm_patch) else b""
    name = build_cache_name(
        model_slug="lfm2.5-24b-a2b",
        vllm_version="0.32.0",
        max_model_len=200000,
        buckets="16-4096-200000",
        llm_args="max_model_len=200000;max_num_seqs=1;max_num_batched_tokens=8192;gpu_memory_utilization=0.92;prefix_caching=0;kv=auto;buckets=16|4096|8192",
        patch_content=patch_bytes,
    )
    pattern = r"tpu_compile_cache__lfm2\.5-24b-a2b__vllm0\.32\.0__len200000__b16-4096-200000__[0-9a-f]{8}\.tar"
    assert re.fullmatch(pattern, name), f"Invalid LFM cache name format: {name}"


def test_determinism_and_differentiation():
    a = build_cache_name("qwen3-4b", "0.32.0", 32768, [16, 32768], {"arg": 1}, b"patch1")
    b = build_cache_name("qwen3-4b", "0.32.0", 32768, "16-32768", {"arg": 1}, b"patch1")
    assert a == b, "Sequence of ints and string bucket should match"

    c = build_cache_name("qwen3-4b", "0.32.0", 32768, "16-32768", {"arg": 2}, b"patch1")
    assert a != c, "Different args must change hash"

    d = build_cache_name("qwen3-4b", "0.32.0", 32768, "16-32768", {"arg": 1}, b"patch2")
    assert a != d, "Different patches must change hash"


if __name__ == "__main__":
    test_qwen_naming()
    test_lfm_naming()
    test_determinism_and_differentiation()
    print("All cache naming unit tests passed!")
