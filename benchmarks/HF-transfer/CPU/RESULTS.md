## 2026-10-28 HF-transfer on CPU: NOT BENCHMARKED

The Colab CPU runtime is extremely slow for this job (2 vCPU, 12.7 GB RAM, so the 14.88 GB model cannot go into `/dev/shm` and must go to disk). The sweep (`scripts/hf_sweep.py 1,4,8,12,16 2 disk`) was started and stopped after about 15 minutes with no result. A GPU runtime is obviously faster and the goal is the fastest possible path, so no CPU download numbers were taken. No number in this entry is measured.

- Model repo for any later attempt: `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (public, 16 shards, no token needed).
- Cost: about 0.01 CU (0.08 CU/hr for the short run).
- Rerun only if CPU inference becomes a goal.
