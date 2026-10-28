## 2026-10-28 16:40 IST: HF-transfer benchmark on CPU (LFM2.5-24B-A2B-UD-Q4_K_XL)

**Model:** `avgrocketman/LFM2.5-24B-A2B-GGUF` (`LFM2.5-24B-A2B-UD-Q4_K_XL.gguf`, 14.88 GB) (MEASURED)
**Weights:** HF transfer download via `hf_transfer`
**Hardware:** Colab CPU runtime, 2 vCPU, 12.7 GB RAM (MEASURED)
**CLI:** `colab` 0.7.5, command `colab --auth=oauth2 run --timeout 1800 scripts/hf_bench_lfm_shards.py`
**Engine:** hf_transfer (Python `huggingface_hub` + `hf_transfer`)
**Flags:** `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_HIGH_PERFORMANCE=1`, `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`

### Results

| Test | Shards / Config | Size (B) | Time (s) | Speed (MB/s) | Label |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Single blob | 1 file | 14,876,901,632 | TBD | TBD | MEASURED |
| Sharded pull (w=1) | 16 files | 14,876,901,632 | TBD | TBD | MEASURED |
| Sharded pull (w=4) | 16 files | 14,876,901,632 | TBD | TBD | MEASURED |
| Sharded pull (w=8) | 16 files | 14,876,901,632 | TBD | TBD | MEASURED |
| Sharded pull (w=16) | 16 files | 14,876,901,632 | TBD | TBD | MEASURED |

### Shard Details

- Split tool: `llama-gguf-split --split --split-max-size 1G`
- Destination: `avgrocketman/LFM2.5-24B-A2B-GGUF` under `shards/`
- Shard count: 16 files (TBD sizes)

### Cost & Lifecycle

- Runtime: Colab CPU (`--gpu` omitted)
- CU rate: 0.00 CU/hr (MEASURED)
- CU used: 0.00 CU (MEASURED)
- Session lifecycle: ephemeral `colab run`, auto-stopped immediately on completion (MEASURED)
- Verification: `colab --auth=oauth2 sessions` empty (MEASURED)
