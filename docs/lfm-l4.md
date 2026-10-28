# LFM2.5-24B-A2B-Q4 on NVIDIA L4

## Findings

### Repositories & Artifacts
- **Public Hub Repository:** `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (private=False, usedStorage 29753805120 bytes). Allows tokenless pull.
  - Blob: `LFM2.5-24B-A2B-UD-Q4_K_XL.gguf` (14876901632 B).
  - Shards: 16 files under `shards/` (`LFM2.5-24B-A2B-UD-Q4_K_XL-00001-of-00016.gguf` through `*-00016-of-00016.gguf`).
- **Private Source Repository:** `avgrocketman/LFM2.5-24B-A2B-GGUF` (SHA256 `a235d23638cc5549cb1ff61da7522c8a7e97f056f27093462a1e56f35aba91d8`).
- **Unused Repository:** `avgrocketman/flint-GGUF`.
- **Preparation Job:** `20261028-173057-public-lfm-copy-a3cb` completed server-side Hub `copy_files` ok in ~5 min consuming 0 GPU.

### Hardware & Environment
- **Colab GPU Runtime:** NVIDIA L4 (12 vCPU, 26G shm, 23034 MiB VRAM, nvidia-smi Driver 580.82.07, CUDA 13.0, 1.54 CU/hr).
- **Compute Consumed:** Total CU used 0.32 CU (MEASURED, from INDEX).
- **CPU Benchmark Note:** CPU TBD row in `benchmarks/INDEX.md` is not MEASURED.
- **Engine Pin:** Pin llama.cpp b11954 Ubuntu tarballs (`llama-b11954-bin-ubuntu-x64.tar.gz` and `llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz`). CUDA 12.8 /health was ready=False at 180s (MEASURED); do not pin 12.8. Never cmake or git clone on Colab.

### Measured Download Sweep (Job 20261028-173833-lfm-l4-sweep-94e4)
Execution duration ~996s MEASURED across benchmarks:
- **Rank-1 Path / Production Configuration:**
  - Flags: `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_HIGH_PERFORMANCE=1`, `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`.
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
  - Flags-off degradation:
    - `HF_HUB_ENABLE_HF_TRANSFER=0`: 76.62 s / 194.18 MB/s (MEASURED).
    - `HF_XET_HIGH_PERFORMANCE=0`: 72.80 s / 204.34 MB/s (MEASURED).
    - `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=0`: 71.32 s / 208.60 MB/s (MEASURED).

### Production Download Recommendation
To pull weights on L4:
1. Export flags: `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_HIGH_PERFORMANCE=1`, `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`.
2. Download 16 shards from `avgrocketman/LFM2.5-24B-A2B-Q4-shards` using `snapshot_download(..., allow_patterns=["shards/*"], local_dir="/dev/shm/bundle", max_workers=12)`.
3. Destination: `/dev/shm` (RAM disk) instead of `/tmp`.

## Failures

- **llama.cpp build failure on CPU Colab**: GitHub `/releases/latest` API query failed because asset `bin-ubuntu-x64.zip` was not present, causing fallback to cloning and compiling with `cmake` on Colab CPU. *Fix*: Always pin b11954 (`llama-b11954-bin-ubuntu-x64.tar.gz` and `llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz`). Never query `/releases/latest` or compile on Colab.
- **Timeout race condition**: `grok-agy --wait-secs 1800` equaled `colab --timeout 1800`; terminated mid w=1 benchmark (job `20261028-163723-lfm-q4-shard-sweep-1db5`). *Fix*: Ensure `wait_secs > timeout` strictly (e.g., `--wait-secs 2400` vs `--timeout 1800`).
- **HF_TOKEN leakage on argv**: Passing `HF_TOKEN` in `colab run` arguments leaked secrets into `events.jsonl` `CommandLine`. *Fix*: Follow no-token-on-argv strictly. Upload token via `colab upload _tasks/hf_token_upload hf_token -s SESSION`.
- **Log buffering assumptions**: `colab log` remains empty until execution terminates (flush-at-end behavior), giving false appearance of hanging. *Fix*: Benchmark scripts must emit a heartbeat every ~20s (`heartbeat ~20s`).
- **Leftover active L4 session idle wait**: Job `20261028-180811-lfm-l4-more-4cac` sat ACTIVE for ~16 min without EXEC until stopped manually with SIGTERM/KILL (exit 124). *Fix*: Do not idle-poll 20+ min without progress; terminate orphaned sessions early.
- **Unsuitable network hardware on CPU**: CPU Colab network throughput (~98 MB/s) throttles shard/worker sweeps. *Fix*: CPU Colab NIC ~98 MB/s is unsuitable; use L4 GPU instances (~250 MB/s).
- **llama-server CUDA mismatch**: Using CUDA 12.8 against driver 13.0 caused `/health` `ready=False` timeout at 180s. *Fix*: Pin b11954 with CUDA 13.4 (`llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz`).
- **Missing CLI permissions in grok-agy runner**: `grok-agy --tools code` denies execution of `colab` CLI commands. *Fix*: Always pass `--tools code --allow-cmd colab`.
- **CPU benchmark indexing**: Tentative/partial CPU runs risked confusion with validated results. *Fix*: INDEX CPU TBD row must never be treated as MEASURED.
- **CU spend governance**: Risk of unintended GPU costs. *Fix*: Spend 0 CU without explicit user go; keep sessions empty (0.00 CU/hr); run `colab stop` immediately after job completion.

## Remaining

**Do not resume without a new user "go" gate.**

### Incomplete (Blocked Until Go)
- **CUDA 13.4 llama-server /health on L4**: CUDA 12.8 failed with driver/CUDA mismatch; winner download never got successful GPU load and health verification.
- **8-shard and 32-shard Hub layouts plus timed pulls**: Only 16-shard layout is MEASURED; 8-shard and 32-shard layouts remain to be generated and measured.
- **Re-time w=8 and w=16 on L4**: Confirm timings and collapse UNCONFIRMED status.
- **Commit/push L4 RESULTS**: `benchmarks/INDEX.md` has a CPU TBD row; do not mark it MEASURED.
- **Repo stale-fact cleanup frozen**: Paused until user unblocks.
- **Prebuilt llama-bin.tar.gz missing from repo**.
- **`~/.config/colab-cli/token.json` permissions**: Still mode 664 (target mode: 600).
- **Second L4 benchmark script**: `scripts/hf_bench_lfm_l4_more.py` did not finish execution.

### Already Done (Do Not Redo)
- Public repo `avgrocketman/LFM2.5-24B-A2B-Q4-shards` is live (blob + 16 shards).
- First L4 download sweep MEASURED; winner is 16 shards into `/dev/shm` at `max_workers=12` (~16.60 s / 896 MB/s).
- Leftover L4 session stopped; CU/hr 0.00.

### Resume Recipe After Go
1. Start new session: `colab --auth=oauth2 new --gpu L4 -s NAME`
2. Upload HF token without command-line exposure: `cp token to _tasks/hf_token_upload`; `colab upload _tasks/hf_token_upload hf_token -s NAME`
3. Pin binary package: Use `llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz` (never `/releases/latest`, never CUDA 12.8, never compile).
4. Run execution with timeout coordination: `colab exec --timeout T; grok-agy --wait-secs > T; heartbeat 20s`
5. Cleanup: `colab stop` after run; never expose `HF_TOKEN` on argv.
