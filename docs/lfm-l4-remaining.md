# LFM L4 Remaining Work and Resume Runbook

Read `docs/lfm-l4-findings.md` then `docs/lfm-l4-failures.md` then this file.
**Do not resume without a new user "go" gate.**

## Incomplete (Blocked Until Go)

- **CUDA 13.4 llama-server /health on L4**:
  - CUDA 12.8 failed with driver/CUDA mismatch on L4.
  - Winner download never got a successful GPU load and health verification.
- **8-shard and 32-shard Hub layouts plus timed pulls**:
  - Only the 16-shard layout is MEASURED; 8-shard and 32-shard layouts remain to be generated and measured.
- **Re-time w=8 and w=16 on L4**:
  - Confirm timings and collapse UNCONFIRMED status.
- **Commit/push L4 RESULTS**:
  - `benchmarks/INDEX.md` has a CPU TBD row; do not mark it MEASURED.
- **Repo stale-fact cleanup frozen**:
  - Paused until user unblocks.
- **Prebuilt llama-bin.tar.gz missing from repo**.
- **`~/.config/colab-cli/token.json` permissions**:
  - Still mode 664 (target mode: 600).
- **Second L4 benchmark script**:
  - `scripts/hf_bench_lfm_l4_more.py` did not finish execution.

## Already Done (Do Not Redo)

- Public repo `avgrocketman/LFM2.5-24B-A2B-Q4-shards` is live (blob + 16 shards).
- First L4 download sweep MEASURED; winner is 16 shards into `/dev/shm` at `max_workers=12` (~16.60 s / 896 MB/s).
- Leftover L4 session stopped; CU/hr 0.00.

## Resume Recipe After Go

1. Start new session:
   `colab --auth=oauth2 new --gpu L4 -s NAME`
2. Upload HF token without command-line exposure:
   `cp token to _tasks/hf_token_upload`
   `colab upload _tasks/hf_token_upload hf_token -s NAME`
3. Pin binary package:
   Use `llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz` (never `/releases/latest`, never CUDA 12.8, never compile).
4. Run execution with timeout coordination:
   `colab exec --timeout T; grok-agy --wait-secs > T; heartbeat 20s`
5. Cleanup:
   `colab stop` after run; never expose `HF_TOKEN` on argv.
