## 2026-10-28 L4 worker-count sweep

**Command & Run:**
- Script: `scripts/hf_sweep.py`
- Args: `4,8,12,16,24 3 shm`
- Repo: `avgrocketman/LFM2.5-24B-A2B-Q4-shards`
- Destination: `/dev/shm`
- Flags: `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_HIGH_PERFORMANCE=1`, `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`
- Schedule: 3 reps with the order rotated

**Dataset & Environment Facts:**
- Files: 16
- Bytes: 14876903488
- CPUs: 12
- Errors: None (no rep had an error)

### Results

| Workers | Rep 0 (MB/s) | Rep 1 (MB/s) | Rep 2 (MB/s) | Mean (MB/s) | Label |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 4 | 995.4 | 316.5 | 773.1 | 695.0 | MEASURED |
| 8 | 1097.7 | 941.2 | 199.9 | 746.3 | MEASURED |
| 12 | 1119.7 | 666.9 | 975.9 | 920.8 | MEASURED |
| 16 | 196.4 | 191.1 | 995.1 | 460.9 | MEASURED |
| 24 | 533.3 | 194.6 | 101.3 | 276.4 | MEASURED |

**Reading (checked against the script's JSON lines, all 15 rows match):**
- No worker count collapses by itself. Every count has fast runs (about 13-15 s, 770-1120 MB/s) and slow runs (75-147 s, 100-200 MB/s). 12 workers had the best mean, but 4, 8 and 16 also hit about 1000 MB/s once, so the fast mode is not specific to 12.
- The slow runs look like random throttling on the download side, not a worker-count effect (inferred, not proven). The earlier "8 and 16 collapse" note is not confirmed.
- Pick: 8 to 12 workers (UNCONFIRMED which is best; more reps would be needed to separate them from the noise).

### Cost

- CU rate: 1.54/hr (MEASURED)
- Balance before: 397.10 CU (MEASURED)
- Balance after: 396.84 CU (MEASURED)
- CU used: 0.26 CU (MEASURED)

