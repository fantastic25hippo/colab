# Hugging Face Storage Audit & Cleanup Plan

## 1. Repository Inventory (avgrocketman)

| Repository | Visibility | Files | Total Size | Last Modified | Role / Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `avgrocketman/LFM2.5-24B-A2B-GGUF` | **PRIVATE** | 18 | 27.71 GB | 2026-10-28 11:32 UTC | 100% duplicate of public shards repo |
| `avgrocketman/flint-GGUF` | **PRIVATE** | 2 | 23.33 GB | 2026-10-28 09:57 UTC | Unused single blob (`flint-UD-Q4_K_XL.gguf`) |
| `avgrocketman/tpu-compile-cache` | **PRIVATE** | 8 | 11.33 GB | 2026-10-31 09:55 UTC | Active TPU cold-start caches + venv + zstd |
| `avgrocketman/LFM2.5-24B-A2B-Q4-shards` | **PUBLIC** | 18 | 27.71 GB | 2026-10-28 12:05 UTC | Active L4 GGUF (16 shards + single blob) |
| `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` | **PUBLIC** | 17 | 12.24 GB | 2026-10-31 14:02 UTC | Active TPU v5e weights + configs + tokenizer |

*Total Account Storage:* 102.32 GB across 5 model repos (Private: 62.37 GB, Public: 39.95 GB). No dataset or space repos exist.

## 2. File-by-File Breakdown & Overlaps

| File Path | Private Repo | Public Repo | Size | Match (SHA-256) | Action |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `LFM2.5-24B-A2B-UD-Q4_K_XL.gguf` | `.../LFM2.5-24B-A2B-GGUF` | `.../LFM2.5-24B-A2B-Q4-shards` | 14.88 GB | `a235d23638cc...` (100% match) | Delete private repo |
| `shards/*-00001..00016.gguf` (16 files) | `.../LFM2.5-24B-A2B-GGUF` | `.../LFM2.5-24B-A2B-Q4-shards` | 14.88 GB | All 16 match public exactly | Delete private repo |
| `flint-UD-Q4_K_XL.gguf` | `.../flint-GGUF` | *None* | 25.05 GB | `63c2ecc461ca...` | Delete private repo |
| `venv312_cpu.tar.zst` | `.../tpu-compile-cache` | *None* (Needed by Qwen3 & LFM2.5) | 1.35 GB | `d3af735580f7...` | Keep in public cache |
| `zstd_bin` | `.../tpu-compile-cache` | *None* (Needed for zero-apt unpack) | 998.79 KB | `7c5468b370f7...` | Keep in public cache |
| `tpu_compile_cache.tar` | `.../tpu-compile-cache` | *None* (Qwen3-4B AOT XLA cache) | 117.40 MB | `112ebe0c345b...` | Keep in public cache |
| `tpu_compile_cache_b1_32k.tar` | `.../tpu-compile-cache` | *None* (LFM 2.5 32K AOT XLA cache) | 53.66 MB | `6f7d05defca5...` | Keep in public cache |
| `versions.json` | `.../tpu-compile-cache` | *None* (Python lockfile metadata) | 8.70 KB | Match git blob | Keep in public cache |
| `wheels.tar.gz` | `.../tpu-compile-cache` | *None* | 113 B | Obsolete | Delete |
| `venv312.tar` | `.../tpu-compile-cache` | *None* (Old uncompressed venv) | 9.80 GB | `9b48f90bc1fb...` (Obsolete) | Delete |

## 3. Sensitive Data Audit

- Scanned all repositories, text metadata (`versions.json`, `config.json`, `chat_template.jinja`), gitattributes, and directory structures.
- **Result:** No tokens, API keys, passwords, credentials, `.env` files, or PII exist in any repo.

## 4. Code References & Mapping

| Referenced Hub ID / Filename | Used In Files | File Exists In | Target State |
| :--- | :--- | :--- | :--- |
| `avgrocketman/tpu-compile-cache` (`venv312_cpu.tar.zst`, `zstd_bin`, `tpu_compile_cache*.tar`) | `benchmarks/TPU-v5e/tpu_coldstart_driver.py`<br>`benchmarks/TPU-v5e/lfm25_200k/lfm25_coldstart_driver.py`<br>`benchmarks/TPU-v5e/lfm25_200k/lfm25_cache.py`<br>`scripts/tpu_xla_overhead.py` | `avgrocketman/tpu-compile-cache` (private) | Make repo **PUBLIC** (or copy to public cache) |
| `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` (weights + configs) | `benchmarks/TPU-v5e/lfm25_200k/lfm25_coldstart_driver.py` | `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` | **PUBLIC** (already public, keep) |
| `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards + blob) | `scripts/hf_bench_lfm_l4.py`<br>`scripts/hf_sweep.py`<br>`docs/WEIGHTS-LOADING.md` | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` | **PUBLIC** (already public, keep) |
| `Qwen/Qwen3-4B` | `benchmarks/TPU-v5e/tpu_coldstart_driver.py`<br>`benchmarks/TPU-v5e/RECIPE.md` | `Qwen/Qwen3-4B` | **PUBLIC** (upstream Alibaba repo, keep) |
| `avgrocketman/LFM2.5-24B-A2B-GGUF` | `docs/lfm-l4.md`<br>`gguf/README.md` | `avgrocketman/LFM2.5-24B-A2B-GGUF` (private) | Update docs to `...-Q4-shards`, delete private repo |
| `avgrocketman/flint-GGUF` | `README.md`<br>`benchmarks/README.md`<br>`docs/lfm-l4.md` | `avgrocketman/flint-GGUF` (private) | Remove references; delete private repo |

## 5. Execution Plan

### Step 2 (Pending User "go"):
1. **Preserve Public TPU Cache:**
   - Make `avgrocketman/tpu-compile-cache` **PUBLIC** via `update_repo_settings(repo_id="avgrocketman/tpu-compile-cache", private=False)`.
   - This keeps exact same paths for `venv312_cpu.tar.zst`, `zstd_bin`, `tpu_compile_cache.tar` (Qwen3), `tpu_compile_cache_b1_32k.tar` (LFM2.5), and `versions.json`.
2. **Prune Obsolete Cache Files:**
   - Delete obsolete uncompressed `venv312.tar` (9.80 GB) and empty `wheels.tar.gz` (113 B).
3. **Verify Code & Run Local Tests:**
   - Run `pytest benchmarks/TPU-v5e/lfm25_200k/test_lfm25_startup.py`.
   - List exact private repos to be deleted for Step 3 confirmation.

### Step 3 (Pending Second "go"):
1. Delete private repository `avgrocketman/LFM2.5-24B-A2B-GGUF` (27.71 GB).
2. Delete private repository `avgrocketman/flint-GGUF` (23.33 GB).
3. Confirm private side holds 0 repos (`private=True` count is 0).
4. Re-verify that every download path resolves anonymously (no `HF_TOKEN`).
