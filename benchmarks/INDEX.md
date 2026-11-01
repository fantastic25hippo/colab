# Benchmark index

One row per entry, newest first. Format and rules: [`README.md`](README.md). Speeds are `MEASURED` tok/s (or TTFT ms where noted); `n/a` means transfer/build-only or not reached.

| Date (IST) | Model / Task | Quant | Accelerator | Engine + build | Prefill (pp512 / TTFT) | Decode (tok/s) | CU/hr | CU used | Entry |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 2026-11-01 | LiquidAI/LFM2.5-24B-A2B (200K, B=1, fp8 KV) | W8A8+W4A16 | TPU-v5e | vllm-tpu 0.32.0 (`425.1s` in-VM / `435.6s` global) | 0.161s TTFT (32.38s @200K) | 165.91 (1.94 @200K, prefill in wall) | 3.09 | 0.31 | [entry](TPU-v5e/lfm25_200k/RESULTS.md) |
| 2026-10-31 | LiquidAI/LFM2.5-24B-A2B (200K, B=1) | W8A8+W4A16 | TPU-v5e | vllm-tpu 0.32.0 (`310.9s` in-VM / `321.0s` global) | 0.153s TTFT (32.58s @200K) | 170.26 (1.93 @200K, prefill in wall) | 3.09 | 0.22 | [entry](TPU-v5e/lfm25_200k/RESULTS.md) |
| 2026-10-31 | Qwen/Qwen3-4B (32K floor cold start, Run 5) | bf16 | TPU-v5e | vllm-tpu 0.32.0 (`73.1s` in-VM / `81.2s` global, `[16,8192]` buckets, 0 apt/uploads) | 21.4 ms TTFT (2.18s @22.3K) | 86.99 (58.31 @22.3K) | 2.92 | 0.04 | [entry](TPU-v5e/tpu-final/RUN5-FLOOR-COLDSTART.md) |
| 2026-10-31 | Qwen/Qwen3-4B (32K pruned cold start, Run 4) | bf16 | TPU-v5e | vllm-tpu 0.32.0 (`90.8s` in-VM / `103.9s` global, `[16,512,8192]` buckets) | 2.19s @22.3K | 86.7 (58.35 @22.3K) | 2.92 | 0.08 | [entry](TPU-v5e/tpu-final/RUN4-PRUNED-COLDSTART.md) |
| 2026-10-31 | Qwen/Qwen3-4B (32K warm cold start, Run 3) | bf16 | TPU-v5e | vllm-tpu 0.32.0 (`131.2s` in-VM / `145.4s` global, 0 compiles) | 23.0 ms TTFT (2.18s @22.3K) | 86.99 (58.13 @22.3K) | 2.92 | 0.13 | [entry](TPU-v5e/tpu-final/RUN3-WARM-COLDSTART.md) |
| 2026-10-29 | Qwen/Qwen3-4B (32K cold start, Run 2) | bf16 | TPU-v5e | vllm-tpu 0.32.0 (`venv312_cpu` in `/dev/shm`) | 75.5 ms TTFT | 88.58 | 2.92 | 0.22 | [entry](TPU-v5e/tpu-final/RUN2-COLDSTART.md) |
| 2026-10-29 | Fast Venv Build (`venv312_cpu.tar.zst`, Run 1) | n/a | TPU-v5e | vllm-tpu 0.32.0 + `torch 2.10.0+cpu` | n/a | n/a | 2.92 | 0.13 | [entry](TPU-v5e/tpu-final/RUN1-VENV.md) |
| 2026-10-29 | Qwen/Qwen3-4B (final confirmation, B=1..32) | bf16 | TPU-v5e | vllm-tpu 0.32.0 (warm 17K cache) | 16.3 ms TTFT | 84.92 (1,653.4 B=32) | 2.92 | 0.52 | [entry](TPU-v5e/tpu-final/RESULTS.md) |
| 2026-10-29 | Qwen/Qwen3-4B (characterization, 512..16K) | bf16 | TPU-v5e | vllm-tpu 0.32.0 (cold 17K compile) | 16.5 ms TTFT | 84.92 (69.83 @16K) | 2.92 | 0.24 | [entry](TPU-v5e/tpu-characterization/RESULTS.md) |
| 2026-10-29 | Qwen/Qwen3-4B (early bucket/cache study) | bf16 | TPU-v5e | vllm-tpu 0.32.0 (`[16,512]` buckets) | n/a | 76.69 | 2.92 | 0.70 | [entry](TPU-v5e/qwen3-4b-final/RESULTS.md) |
| 2026-10-29 | Qwen/Qwen3-4B (overhead & loader patch) | bf16 | TPU-v5e | vllm-tpu 0.32.0 | n/a | 83.50 | 2.92 | 0.32 | [entry](TPU-v5e/overhead-two-models/RESULTS.md) |
| 2026-10-29 | google/gemma-4-E4B-it-qat-w4a16-ct | w4a16-ct | TPU-v5e | vllm-tpu (py3.13 install failed; dl 570 MB/s) | n/a | n/a | 2.92 | 0.18 | [entry](TPU-v5e/gemma4-e4b-qat/RESULTS.md) |
| 2026-10-28 | HF-transfer worker sweep (3 reps × 5 counts) | Q4 shards | L4 | `snapshot_download` (920.8 MB/s mean @ w=12) | n/a | n/a | 1.54 | 0.26 | [entry](HF-transfer/L4/SWEEP.md) |
| 2026-10-28 | LFM2.5-24B-A2B | UD-Q4_K_XL (16 shards) | L4 | `snapshot_download` (896 MB/s; CUDA 12.8 server failed) | n/a | n/a | 1.54 | 0.32 | [entry](LFM2.5-24B-A2B-Q4/L4/RESULTS.md) |
| 2026-10-28 | HF-transfer (CPU shard prep check) | Q4 shards | CPU | not benchmarked (NIC ~98 MB/s) | n/a | n/a | 0.08 | 0.01 | [entry](HF-transfer/CPU/RESULTS.md) |
| 2026-10-28 | HF-transfer (Qwen2.5-7B / 14B) | GGUF | L4 | `hf_transfer` (247–252 MB/s single blob) | n/a | n/a | 1.54 | 0.00 | [entry](HF-transfer/L4/RESULTS.md) |
| 2026-10-28 | HF-transfer (provisioning check) | n/a | TPU-v6e | skipped (`colab new --tpu v6e1` timed out ×2) | n/a | n/a | n/a | 0.00 | [entry](HF-transfer/TPU-v6e/RESULTS.md) |
| 2026-10-28 | HF-transfer (Qwen2.5-7B / 14B) | GGUF | TPU-v5e | `hf_transfer` (222.7 MB/s shm / 107.1 MB/s disk) | n/a | n/a | 2.92 | 0.23 | [entry](HF-transfer/TPU-v5e/RESULTS.md) |
| 2026-10-27 | HF-transfer (Qwen2.5-7B / 14B) | GGUF | G4 | `hf_transfer` (270.4 MB/s shm / 286.2 MB/s disk) | n/a | n/a | 8.90 | 0.00 | [entry](HF-transfer/G4/RESULTS.md) |
| 2026-10-27 | HF-transfer (Qwen2.5-7B / 14B) | GGUF | A100-40 | `hf_transfer` (249.1 MB/s shm / 353.0 MB/s disk) | n/a | n/a | 5.30 | 0.07 | [entry](HF-transfer/A100/RESULTS.md) |
| 2026-10-27 | HF-transfer (Qwen2.5-7B / 14B, 2 runs) | GGUF | T4 | `hf_transfer` (202.0 MB/s shm / 104.7 MB/s disk) | n/a | n/a | 1.07 | 0.15 | [entry](HF-transfer/T4/RESULTS.md) |
