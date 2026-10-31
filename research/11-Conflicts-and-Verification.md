# 11 — Conflicts, Stale Claims and Verification

The master documents (`01`–`10`) consolidate sources written between 2026-10-27 and 2026-10-29 and cross-reference them against **in-repo measured benchmarks** in [`../benchmarks/INDEX.md`](../benchmarks/INDEX.md) (`TPU-v5e/tpu-final/`, `HF-transfer/`, `LFM2.5-24B-A2B-Q4/`). Where a planning estimate disagrees with an in-repo measurement, **trust the measured run `[M]`**.

## 1. How to resolve a conflict

1. **In-repo Measured `[M]` beats Reported beats Inferred `[I]` beats Guessed `[G]`.**
2. **A versioned/reproduced claim beats an unversioned one.**
3. **Accepted ≠ effective.** A CLI flag that parses can still be ignored or silently downgraded (e.g. `--kv-cache-dtype fp8`).
4. **Watch units and definitions.** `15.75 GiB = 16.91 GB` usable HBM on v5e-1; distinguish **driver reserve (`1 - gpu_memory_utilization`)** from **net XLA runtime overhead (`peak - weights - KV`)**.

## 2. Conflict register

| # | Topic | Claim A | Claim B / In-Repo Measurement | Working resolution |
| ---: | :--- | :--- | :--- | :--- |
| 1 | HF transfer & worker count | `03`: `HF_HUB_ENABLE_HF_TRANSFER=1` is a no-op on Xet; `max_workers=4` optimal | `L4/SWEEP.md` [M]: 16 shards into `/dev/shm` hit **896–1,120 MB/s** (`16.60 s` for 14.88 GB) with `max_workers=12` (mean `920.8 MB/s`; `w=4..16` all peak ~1 GB/s when unthrottled) | When writing to `/dev/shm` (no overlayfs lock stalls), **`max_workers=8..12`** is verified fast. Slow reps (`75–147 s`) are Hub/CDN throttling noise, not worker collapse. |
| 2 | HF sequential write | Old Fast Headless: set `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1` | `03`: removed/no-op in `hf_xet` 1.3.0; sequential is mandatory | Omit on modern `hf_xet >= 1.3.0`; harmless if set. |
| 3 | Drive throughput | `06`: 40–80 MB/s typical; also claims 400–600 MB/s tuned | `02` / `RECIPE.md`: `colab drivemount` is interactive on every new VM; 50–140 MB/s | **Never use Google Drive** for headless/ephemeral runs. Use HF Hub → `/dev/shm`. |
| 4 | G4 standard RAM | Google AI Pro chat: 12.7 GB, will OOM | `HF-transfer/G4` [M]: **176 GB RAM, 87 GB `/dev/shm`, 48 vCPUs** | Trust the measured 176 GB RAM / 87 GB `/dev/shm`. |
| 5 | `gcsfuse` for weights | Google AI Pro chat recommends mounting GCS | `01`/`02`/`WEIGHTS-LOADING`: FUSE + `mmap` page-fault storm (<0.5 tok/s); no billing account | Never `mmap` from FUSE; GCS bucket plan is abandoned in this repo. |
| 6 | TPU v5e-1 XLA overhead | `10`: 4.5–6.5 GB needed beyond weights; `07`: 1.26 GiB at 0.92 | `tpu-characterization` & `overhead-two-models` [M]: **net runtime overhead (`peak - weights - KV`) is 47–67 MiB (at 4K) and 371–380 MiB (at 17.4K)** | `1.26 GiB` is just the unallocated driver reserve (`15.75 × 0.08`). True bucketed XLA runtime overhead is **0.05–0.38 GiB**. (LFM 24B still OOMs because weights alone are 12.9–14.95 GB.) |
| 7 | `--gpu-memory-utilization 0.95` on v5e-1 | `08` external report: fatal `CompileTimeHbmOom` after ~690 s | `tpu-final/RESULTS.md` [M]: **`0.95` succeeded cleanly on `Qwen3-4B`** (`54,336` KV tokens, `406 MiB` headroom, `84.91 tok/s`) | `0.95` is viable for single-stream / low-batch on 4B; use `0.90` (`1,216 MiB` headroom) as default for Batch 32 concurrency. |
| 8 | TPU prebuilt venv size & `/dev/shm` | `09` initial estimate: ~600 MB unpacked / ~250 MB archive; `RECIPE` old trap: "201 s extract" | `RUN1-VENV` & `RUN2-COLDSTART` [M]: `venv312_cpu.tar.zst` is **5.01 GB unpacked / 1.35 GB packed**, downloads in **13.50 s** and unpacks in **6.35 s** into `/dev/shm` after `mount -o remount,exec /dev/shm` | Trust `RUN1-VENV` / `RUN2-COLDSTART`. Always run `mount -o remount,exec /dev/shm` first (`0.08 s`). |
| 9 | `TPU_SKIP_MDS_QUERY=1` | `09` initial estimate: saves 20–25 s of metadata polling | `RUN2-COLDSTART` [M]: verified on `libtpu 0.0.48`; cuts `import vllm, jax` check from **15.56 s to 5.91 s (-9.65 s / -62%)** | Confirmed working on Colab v5e-1; saves **9.65 s** directly (rest of the 48 s → 5.91 s gain comes from CPU-torch + `/dev/shm` `.pyc`). |
| 10 | TPU host-sync cap (`31.8–37.2 tok/s`) | `07b`: single-step decode capped at 31.8–37.2 tok/s without `--num-scheduler-steps 8` | `tpu-final` & `RUN2` [M]: `vllm-tpu==0.32.0` achieves **84.92–88.58 tok/s** at B=1 without `--num-scheduler-steps 8` (~83% of peak HBM BW) | The 31–37 tok/s cap applies to naive TorchTPU step loops, **not** `vllm-tpu 0.32.0`. |
| 11 | TPU KV dtype | `07`: static INT8 ring KV mandatory | `08`: vLLM RPA does not support `int8`; only `fp8` is wired and may silently allocate `bf16` | Path-specific (TorchTPU custom ring vs vLLM RPA). Verify `GPU KV cache size: X tokens` in logs. |
| 12 | Accelerator CU/hour rates | Older docs: A100 ~15, T4 ~1.96, v5e ~1.20–1.75, L4 ~1.71 CU/h | `HF-transfer/*` & `tpu-final` [M]: **T4 `1.07`, L4 `1.54`, v5e-1 `2.92`, A100-40 `5.30`, G4 `8.90`, CPU `0.08` CU/h** | Always use the **measured** CU/h rates (`hardware.md`). Monthly pool read **~400 CU** (`398.80` on 10-27). |
| 13 | Weight-loading `jax.clear_caches()` | Omitted in early `09` and `FAST-COLDSTART.md` | `RECIPE.md` §3, `qwen3-4b-final`, `overhead-two-models` [M]: monkeypatching `jax.clear_caches` in `weight_utils` cuts load time from **29.4–40.5 s to 10.7–11.85 s** | Mandatory in every TPU v5e-1 vLLM launch (`sitecustomize.py` hook). |

## 3. Stale or dangerous claims (never repeat)

- **`SKIP_JAX_PRECOMPILE=1` or `compilation_sizes=[16, 512]` as a serving speedup.** Measured in `tpu-characterization`: collapses decode from **85–88 tok/s to 7.6–29.9 tok/s** due to 60–76 runtime JIT compiles.
- **Unpacking a venv into `/dev/shm` without `mount -o remount,exec /dev/shm`.** Fails `dlopen` on `libtorch_global_deps.so` (`noexec` mount default).
- **Running `vllm-tpu` on Colab's default Python 3.13.** Wheels require Python 3.12; Python 3.13 fails building from source (`gemma4-e4b-qat/RESULTS.md`).
- **Pinning `llama.cpp` CUDA 12.8 binaries on Colab L4.** Fails `/health` (`ready=False` at 180 s) against driver 580.82.07 / CUDA 13.0; pin `llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz`.
- **Passing `HF_TOKEN` on CLI `argv`.** Leaks into process tables and command logs; upload `/content/hf_token` via `colab upload`.

## 4. Open questions requiring live verification (only after Sahil's "go")

1. **Warm 32K TPU cold start:** Time a fresh-VM run restoring `tpu_compile_cache_b1_32k.tar` (53.66 MB) + `venv312_cpu.tar.zst` in `/dev/shm` with the `jax.clear_caches` hook active to confirm the projected **~70–85 s** total cold start.
2. **Sharded `venv312_cpu` archive + streaming zstd unpack:** Test whether splitting `venv312_cpu.tar.zst` (1.35 GB) into 8–12 shards cuts the 19.85 s parallel prep stage down to ~8–10 s.
3. **26K–30K TPU decode speed:** Investigate why `RUN2-COLDSTART.md` recorded 10.20 tok/s at 25.7K tokens vs 69.83 tok/s at 16K in `tpu-characterization`.
4. **L4 `LFM2.5-24B-A2B-Q4` decode speed:** Verify `llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz` loads the 16 shards from `/dev/shm` and measure `pp512` / `tg128`.
