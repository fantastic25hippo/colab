# LFM2.5-24B-A2B-Q4 on NVIDIA L4: Findings & Runbook

MEASURED facts and runbook for resuming LFM2.5-24B-A2B-Q4 work on Google Colab NVIDIA L4 runtime.

## Repositories & Artifacts

- **Public Hub Repository:** `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (private=False, usedStorage 29753805120 bytes). Allows tokenless pull.
  - Blob: `LFM2.5-24B-A2B-UD-Q4_K_XL.gguf` (14876901632 B).
  - Shards: 16 files under `shards/` (`LFM2.5-24B-A2B-UD-Q4_K_XL-00001-of-00016.gguf` through `*-00016-of-00016.gguf`).
- **Private Source Repository:** `avgrocketman/LFM2.5-24B-A2B-GGUF` (SHA256 `a235d23638cc5549cb1ff61da7522c8a7e97f056f27093462a1e56f35aba91d8`).
- **Unused Repository:** `avgrocketman/flint-GGUF`.
- **Preparation Job:** `20261028-173057-public-lfm-copy-a3cb` completed server-side Hub `copy_files` ok in ~5 min consuming 0 GPU.

## Hardware & Environment

- **Colab GPU Runtime:** NVIDIA L4 (12 vCPU, 26G shm, 23034 MiB VRAM, nvidia-smi Driver 580.82.07, CUDA 13.0, 1.54 CU/hr).
- **Compute Consumed:** Total CU used 0.32 CU (MEASURED, from INDEX).
- **CPU Benchmark Note:** CPU TBD row in `benchmarks/INDEX.md` is not MEASURED.
- **Engine Pin:** Pin llama.cpp b11954 Ubuntu tarballs (`llama-b11954-bin-ubuntu-x64.tar.gz` and `llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz`). CUDA 12.8 /health was ready=False at 180s (MEASURED); do not pin 12.8. Never cmake or git clone on Colab.

## Measured Download Sweep (Job 20261028-173833-lfm-l4-sweep-94e4)

Execution duration ~996s MEASURED across benchmarks:

### Download Sweeps & Throughput

- **Rank-1 Path / Production Configuration:**
  - Environment variables: `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_HIGH_PERFORMANCE=1`, `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`.
  - Target: 16 shards to `/dev/shm`, `snapshot_download` with `max_workers=12`.
  - Result: 16.60 s / 896.13 MB/s (MEASURED, WINNER).
- **Worker Sweep (16 shards to /dev/shm):**
  - `w=1`: 48.50 s / 306.73 MB/s (MEASURED)
  - `w=2`: 26.08 s / 570.38 MB/s (MEASURED)
  - `w=4`: 17.51 s / 849.53 MB/s (MEASURED)
  - `w=8`: 61.80 s / 240.72 MB/s (MEASURED) - collapsed UNCONFIRMED
  - `w=12`: 16.60 s / 896.13 MB/s (MEASURED, WINNER)
  - `w=16`: 82.30 s / 180.76 MB/s (MEASURED) - collapsed UNCONFIRMED
- **Single Blob vs Shards:**
  - Blob to `/dev/shm`: 23.61 s / 630.12 MB/s (MEASURED).
  - Blob to `/tmp`: 44.71 s / 332.75 MB/s (MEASURED) - `/tmp` much slower.
- **Destination & Flag Degradation (at w=12):**
  - Disk overlay `/tmp/w_best`: 104.82 s / 141.93 MB/s (MEASURED) - disk overlay `/tmp` is much slower than `/dev/shm`.
  - Flags-off hurt performance:
    - `HF_HUB_ENABLE_HF_TRANSFER=0`: 76.62 s / 194.18 MB/s (MEASURED).
    - `HF_XET_HIGH_PERFORMANCE=0`: 72.80 s / 204.34 MB/s (MEASURED).
    - `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=0`: 71.32 s / 208.60 MB/s (MEASURED).

## Production Download Recommendation

To pull weights on L4:
1. Export the three HF acceleration flags:
   `HF_HUB_ENABLE_HF_TRANSFER=1`
   `HF_XET_HIGH_PERFORMANCE=1`
   `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`
2. Download 16 shards from `avgrocketman/LFM2.5-24B-A2B-Q4-shards` using `snapshot_download(..., allow_patterns=["shards/*"], local_dir="/dev/shm/bundle", max_workers=12)`.
3. Destination: `/dev/shm` (RAM disk) instead of `/tmp`.
