# Hugging Face Storage Audit & Final State

## 1. Final Repository Inventory (avgrocketman)

| Repository | Visibility | Files | Size | Role / Contents |
| :--- | :---: | :---: | :---: | :--- |
| `avgrocketman/LFM2.5-24B-A2B-Q4-shards` | **PUBLIC** | 18 | 27.71 GB | Active L4 GGUF (16 shards + single blob) |
| `avgrocketman/tpu-compile-cache` | **PUBLIC** | 6 | 1.53 GB | Active TPU v5e fast-start venv, zstd, and XLA caches |
| `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` | **PUBLIC** | 17 | 12.24 GB | Active TPU v5e weights, tokenizer, config & templates |

*Private Repositories Remaining:* **0** (confirmed via Hub API).
*Total Public Account Storage:* **41.48 GB** across 3 repos.

## 2. Deletions Executed

1. **`avgrocketman/LFM2.5-24B-A2B-GGUF`** (PRIVATE, 27.71 GB): Deleted. (100% duplicate of public shards repo).
2. **`avgrocketman/flint-GGUF`** (PRIVATE, 23.33 GB): Deleted. (Unused single blob; upstream is `unsloth/flint-GGUF`).
3. **Pruned from `avgrocketman/tpu-compile-cache`:**
   - `venv312.tar` (9.80 GB): Deleted (obsolete uncompressed venv replaced by `venv312_cpu.tar.zst`).
   - `wheels.tar.gz` (113 B): Deleted.

*Total Storage Freed:* **~60.84 GB**.

## 3. Retained Public Artifacts & Checksums

| File Path | Public Repository | Size | Verified SHA-256 |
| :--- | :--- | :---: | :--- |
| `venv312_cpu.tar.zst` | `avgrocketman/tpu-compile-cache` | 1.35 GB | `d3af735580f7ac2e49ba65a394e40ffdd8d1736f1167b00d1c0e0a0405d0df40` |
| `zstd_bin` | `avgrocketman/tpu-compile-cache` | 998.79 KB | `7c5468b370f7c47eda07281e3437fafc568f95d10420051e3aa522709f9342c5` |
| `tpu_compile_cache.tar` (Qwen3) | `avgrocketman/tpu-compile-cache` | 117.40 MB | `112ebe0c345b6714b9461affb0342da8c3889ebde1d6fa3f86e386242c78df8f` |
| `tpu_compile_cache_b1_32k.tar` (LFM2.5) | `avgrocketman/tpu-compile-cache` | 53.66 MB | `6f7d05defca5c6dd90006582a7d55c5ca8f7c57b7367660a52aad74ace7b5db3` |
| `versions.json` | `avgrocketman/tpu-compile-cache` | 8.70 KB | SHA matches git blob |
| `LFM2.5-24B-A2B-UD-Q4_K_XL.gguf` | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` | 14.88 GB | `a235d23638cc5549cb1ff61da7522c8a7e97f056f27093462a1e56f35aba91d8` |
| `shards/*-00001..00016.gguf` | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` | 14.88 GB | All 16 verified |
| `model-00001..00009.safetensors` | `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` | 12.22 GB | All 9 verified |
| `config.json` + tokenizer + templates | `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` | ~18 MB | All custom configurations intact |

## 4. Verification

- **Anonymous Access:** All 14 download paths tested via unauthenticated HTTP HEAD requests (`token=False`), returning HTTP 200.
- **Codebase Tests:** `pytest benchmarks/TPU-v5e/lfm25_200k/test_lfm25_startup.py` passed 6/6 tests.
- **Git Commit:** Committed locally on `main`; not pushed.
