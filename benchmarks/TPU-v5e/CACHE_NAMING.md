# TPU Compile Cache Naming Specification

All persistent TPU XLA compilation cache archives stored in `avgrocketman/tpu-compile-cache` must be strictly bound to one model, exact vLLM engine settings, context buckets, and patch contents to prevent restoring mismatched executables across models or run configurations.

---

## 1. Canonical Archive Name Syntax

Every cache archive follows the strict format:

```text
tpu_compile_cache__<model-slug>__vllm<ver>__len<max_model_len>__b<buckets>__<8-char hash of patches+LLM args>.tar
```

### Components:
- **`tpu_compile_cache`**: Fixed prefix identifying a TPU compilation cache archive.
- **`<model-slug>`**: Lowercase alphanumeric slug identifying the exact model (e.g. `qwen3-4b`, `lfm2.5-24b-a2b`).
- **`vllm<ver>`**: Engine version string (e.g. `vllm0.32.0`).
- **`len<max_model_len>`**: Configured `max_model_len` (e.g. `len32768`, `len200000`).
- **`b<buckets>`**: Hyphen-separated list of compiled token buckets with `b` prefix (e.g. `b16-32768`, `b16-4096-200000`).
- **`<8-char hash>`**: First 8 hex characters of `SHA-256(canonical_llm_args + "\n" + patch_content)`.

Example:
```text
tpu_compile_cache__qwen3-4b__vllm0.32.0__len32768__b16-32768__4523142f.tar
```

---

## 2. Restore Policy: Exact Match or Run Cold

1. **No Fallback:** A driver may only restore a cache whose filename matches the generated name exactly.
2. **Cache Miss Logging:** If the exact archive does not exist in `avgrocketman/tpu-compile-cache`, the driver logs:
   ```text
   cache miss: <expected name>
   ```
   and immediately proceeds with a clean cold start. It must never fall back to another model's cache.

---

## 3. Helper Module

Both Qwen and LFM drivers construct the cache name using the single canonical helper in `benchmarks/TPU-v5e/cache_naming.py`:

```python
from cache_naming import build_cache_name

cache_tar = build_cache_name(
    model_slug="qwen3-4b",
    vllm_version="0.32.0",
    max_model_len=32768,
    buckets="16-32768",
    llm_args={
        "gpu_memory_utilization": 0.90,
        "max_model_len": 32768,
        "max_num_seqs": 1,
        "trust_remote_code": True,
    },
    patch_content=patch_bytes,
)
```

---

## 4. Legacy Archives (Preserved, Never Auto-Restored)

The following historical compilation caches are preserved in `avgrocketman/tpu-compile-cache` for provenance and reproducibility, but are **never auto-restored** by current drivers:

1. `tpu_compile_cache__qwen3-4b__vllm0.32.0__len17408__legacy.tar`
   - **Origin:** Built during early Qwen3-4B batching/context characterization (commit `dd1c9ac` / `dd17ba9` / `1a821ba`).
   - **Reason:** Built from multi-stage batching experiments without a single reproducible patch/argument hash.
2. `tpu_compile_cache__qwen3-4b__vllm0.32.0__len32768__b1__legacy-5c49c04.tar`
   - **Origin:** Built at commit `5c49c04` ("Run 2 cold start benchmark to first token on TPU v5e-1 with 32K context").
   - **Reason:** Built with original minimal patch (`jax.clear_caches = lambda: None`) without token padding pruning (`_wrap_get_token_paddings`) or GCE metadata bypass, differing from the current `qwen_patches.py`. Drivers will log a cache miss and run cold until a fresh cache is published.

