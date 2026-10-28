# LFM2.5-24B-A2B-Q4 on NVIDIA L4: Hugging Face Shards & Startup Benchmark

**Date:** 2026-10-28 17:45 IST  
**Model:** `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (`LFM2.5-24B-A2B-UD-Q4_K_XL.gguf`, 14.88 GB, 16 shards) (MEASURED)  
**Hardware:** NVIDIA L4 23034 MiB (Driver 580.82.07, CUDA 13.0), CPU cores: 12 (MEASURED)  
**Storage:** `/dev/shm` 26G (0% used), `/tmp` overlay 236G (43G used, 194G avail) (MEASURED)  
**Session:** `lfm-l4` via `colab --auth=oauth2 new --gpu L4 -s lfm-l4` (Stopped) (MEASURED)  

## Measured Download Sweeps

All values below are MEASURED directly from execution of `scripts/hf_bench_lfm_l4.py` on the Colab L4 runtime.

### 1. Single Blob vs Shards Baseline

| Configuration | Target Path | Bytes | Time (s) | Throughput (MB/s) | Label |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Single Blob (`hf_hub_download`) | `/dev/shm/blob` | 14,876,901,632 | 23.61 | 630.12 | MEASURED |
| Single Blob (`hf_hub_download`) | `/tmp/blob` | 14,876,901,632 | 44.71 | 332.75 | MEASURED |

### 2. Shards Worker Sweep (`snapshot_download`, 16 shards to `/dev/shm`)

Default flags: `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_HIGH_PERFORMANCE=1`, `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`.

| Workers | Target Path | Bytes | Time (s) | Throughput (MB/s) | Label |
| ---: | :--- | ---: | ---: | ---: | :--- |
| 1 | `/dev/shm/w1` | 14,876,911,543 | 48.50 | 306.73 | MEASURED |
| 2 | `/dev/shm/w2` | 14,876,911,544 | 26.08 | 570.38 | MEASURED |
| 4 | `/dev/shm/w4` | 14,876,911,539 | 17.51 | 849.53 | MEASURED |
| 8 | `/dev/shm/w8` | 14,876,911,540 | 61.80 | 240.72 | MEASURED |
| 12 | `/dev/shm/w12` | 14,876,911,545 | 16.60 | 896.13 | MEASURED |
| 16 | `/dev/shm/w16` | 14,876,911,541 | 82.30 | 180.76 | MEASURED |

Winner worker configuration: **12 workers** reaching **896.13 MB/s** (14.88 GB downloaded in 16.60 seconds).

### 3. Destination and Environment Flag Sweep (Best Worker = 12)

| Configuration Variant | Target Path | Bytes | Time (s) | Throughput (MB/s) | Label |
| :--- | :--- | ---: | ---: | ---: | :--- |
| Disk overlay (/tmp) | `/tmp/w_best` | 14,876,911,543 | 104.82 | 141.93 | MEASURED |
| RAM disk (`HF_HUB_ENABLE_HF_TRANSFER=0`) | `/dev/shm/w_flag` | 14,876,911,545 | 76.62 | 194.18 | MEASURED |
| RAM disk (`HF_XET_HIGH_PERFORMANCE=0`) | `/dev/shm/w_flag` | 14,876,905,678 | 72.80 | 204.34 | MEASURED |
| RAM disk (`HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=0`) | `/dev/shm/w_flag` | 14,876,905,671 | 71.32 | 208.60 | MEASURED |

### 4. llama-server Health Check

| Step | Engine / Binary | Target | Time (s) | Status | Label |
| :--- | :--- | :--- | ---: | :--- | :--- |
| `llama-server --no-mmap -ngl 99` | llama.cpp b11954 (CUDA 12.8) | `*-00001-of-00016.gguf` | 180.45 | ready=False (timeout) | MEASURED |

Health check timed out after 180s on the pinned b11954 build for this experimental architecture (`LFM2.5-24B-A2B`).

### Cost & Compute Units

- GPU: NVIDIA L4
- Compute Units before: 398.04 CU (MEASURED)
- Compute Units after: 397.72 CU (MEASURED)
- Total CU consumed: 0.32 CU (MEASURED)
- Session state: Stopped and verified empty (MEASURED)
