# 09 — vLLM on TPU v5e-1: Cold Start from 235 s

Consolidated from `Accelerating vLLM Startup on TPU.pdf` and updated with **in-repo measured benchmarks** from [`../benchmarks/TPU-v5e/tpu-final/`](../benchmarks/TPU-v5e/tpu-final/RESULTS.md) (`STARTUP-BREAKDOWN.md`, `RUN1-VENV.md`, `RUN2-COLDSTART.md`) and [`../benchmarks/TPU-v5e/FAST-COLDSTART.md`](../benchmarks/TPU-v5e/FAST-COLDSTART.md). Setting: fresh Colab `v5litepod-1` (23 GB `/dev/shm`, 47 GB host RAM), `Qwen/Qwen3-4B` bf16, `vllm-tpu==0.32.0`, `tpu-inference==0.32.0`, `jax==0.11.2`, `libtpu==0.0.48`, `torch==2.10.0+cpu`, Python 3.12 venv.

## 1. Where the 235.33 s Baseline Goes [M]

Measured on commit `1a821ba` (`tpu-final/RESULTS.md`, `STARTUP-BREAKDOWN.md`, `max_model_len=17408`):

| Stage | Seconds [M] | Notes |
| :--- | ---: | :--- |
| VM create to shell ready | 7.59 | `colab new -s tpu-final --tpu v5e1` |
| Install `vllm-tpu` + `tpu-inference` (parallel bottleneck) | 91.21 | 259 wheels (~1.8+ GB, 15 NVIDIA CUDA wheels pulled by default PyPI torch). Overlaps 10.14 s weight download and 28.92 s cache restore. |
| `from vllm import LLM` (MainProcess import) | 48.00 | GCE metadata 404 polling (`15.56 s`) + uncompiled `.pyc` imports on overlayfs. |
| Model config + tokenizer | 11.00 | |
| EngineCore spawn (`VLLM_WORKER_MULTIPROC_METHOD=spawn`) | 11.00 | Child repeats Python init, imports, and metadata polling. |
| TPU device init, mesh, Gloo + abstract shape trace | 7.66 | `7.00 s` device/mesh + `0.66 s` `flax.nnx.eval_shape`. |
| Weight load (`jax.clear_caches` patch active) | 11.85 | **Without the `jax.clear_caches` patch, weight loading takes 29.4–40.5 s [M].** |
| KV cache allocation | 3.31 | 759 blocks × 64 tokens = 48,576 tokens (6.695 GiB at 0.90 util). |
| Precompile + ZMQ ready + first token | 50.96 | `48.46 s` precompile (261 persistent-cache hits, **0 real compiles**, down from 147.5 s cold) + `0.89 s` ZMQ + `1.61 s` first token. |
| **Total Fresh VM → First Token** | **235.33** | Sustains **84.92 tok/s** single-stream decode (1,653.4 tok/s at Batch 32). |

The 48.46 s warm precompile is **15.01 s Python tracing + 32.70 s jaxpr→StableHLO lowering + 3.38 s cache deserialization** across 261 programs [M]. Setting `SKIP_JAX_PRECOMPILE=1` or restricting `compilation_sizes=[16, 512]` dropped decode from **85–88 tok/s to 7.6–29.9 tok/s [M]** because 60–76 shapes JIT-compiled during generation (`tpu-characterization/RESULTS.md`).

## 2. Install & `/dev/shm` Remount: Prebuilt CPU-Torch Venv [M]

- **CPU-only torch works and removes all 15 NVIDIA CUDA packages — Measured (`RUN1-VENV.md`).** Passing `--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match "torch==2.10.0+cpu"` reduced online install time from **91.21 s to 42.08 s (-54%)** and shrank the unpacked venv from ~9.9 GB to **5,132.45 MB (5.01 GB)**.
- **Mandatory `/dev/shm` `exec` remount — Measured (`RUN1-VENV.md`, `RUN2-COLDSTART.md`).** Colab mounts `/dev/shm` with `noexec` (`rw,nosuid,nodev,noexec,...`), which causes `dlopen` of `libtorch_global_deps.so` to fail with `OSError: failed to map segment from shared object`. Running `mount -o remount,exec /dev/shm` (`0.08 s`, `rc=0`) fixes this completely.
- **Prebuilt `.pyc` + `zstd -3` archive on Hugging Face (`avgrocketman/tpu-compile-cache`):** Running `python -m compileall -q` (47.19 s) and packing with `tar -I 'zstd -3 -T0' -cf venv312_cpu.tar.zst -C /dev/shm vllm_env` produces a **1,387.55 MB (1.35 GB)** archive in 10.36 s [M].
- **Parallel download + unpack into `/dev/shm` — Measured (`RUN2-COLDSTART.md`):** Pulling `venv312_cpu.tar.zst` (`13.50 s` dl + `6.35 s` unpack = **19.85 s**) in parallel with `Qwen/Qwen3-4B` weights (`11.89 s` to `/dev/shm`) takes **19.85 s total** — saving **71.36 s (-78%)** over the 91.21 s baseline. Never use Google Drive (`colab drivemount` is interactive on every fresh VM).

## 3. Import Time and GCE Metadata Polling [M]

On Colab, `cloud_tpu_init.py` polls `http://metadata.google.internal/...` for Cloud TPU attributes that return HTTP 404 (`Unable to poll TPU GCE Metadata`). Setting static topology variables bypasses the retry loop:

```bash
export TPU_SKIP_MDS_QUERY=1
export TPU_TOPOLOGY=1x1
export TPU_HOST_BOUNDS=1,1,1
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_WORKER_ID=0
export TPU_WORKER_HOSTNAMES=localhost
export JAX_PLATFORMS=tpu,cpu
```

- **Measured impact (`RUN2-COLDSTART.md`):** `import vllm, jax; print(jax.devices())` dropped from **15.56 s (default env) to 5.91 s (-9.65 s / -62%)** with `[TpuDevice(id=0)]` verified on `libtpu 0.0.48`. Together with the `/dev/shm` CPU-torch venv + precompiled `.pyc`, total import + init fell from **48.00 s to 5.91 s (-42.09 s)** [M].

## 4. Process Model & Mandatory Weight-Loader Patch [M]

- **In-process `EngineCore` (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) — Measured (`RUN2-COLDSTART.md`):** Eliminates the `spawn` subprocess, ZMQ handshake, and duplicate import (**saves 11.00 s**).
- **Weight-loader `jax.clear_caches()` patch — Measured (`RECIPE.md` §3, `qwen3-4b-final`, `overhead-two-models`):** `JaxAutoWeightsLoader._load_module` (`weight_utils.py:1132`) calls `jax.clear_caches()` after every linear module, causing allocator thrashing. Neutralizing it via `sitecustomize.py` cuts weight loading from **29.4–40.5 s down to 10.7–11.85 s (saves ~18–29 s)** and KV cache allocation from 29.3 s down to **3.31 s**.

## 5. Precompile & Cache Behavior [M]

- **17.4K Default Buckets (`tpu-final`):** 261 cached programs deserialize in **48.46 s** (vs 147.48 s cold).
- **32K Context (`RUN2-COLDSTART.md`, `max_model_len=32768`, `max_num_seqs=1`):** Compiled **104 programs in 194.16 s** on a cold cache miss and deposited `tpu_compile_cache_b1_32k.tar` (**53.66 MB**) to `avgrocketman/tpu-compile-cache`.
- **Untested warm 32K:** At ~0.185 s tracing+lowering per cached program, 104 cached programs should take **~19–25 s** (bringing total fresh-VM-to-first-token to **~65–85 s** [I]).

## 6. Measured vs Projected Timeline

| Stage | Baseline (`17.4K` Warm) [M] | Fast Pipeline (`32K` Cold) [M] | Fast Pipeline (Warm Cache) [I] |
| :--- | ---: | ---: | ---: |
| VM provisioning | 7.59 s | 10.40 s | ~8–10 s |
| Parallel prep (`/dev/shm` venv + weights + cache) | 91.21 s | **19.85 s** | **~20 s** |
| Imports & TPU check (`TPU_SKIP_MDS_QUERY=1`) | 48.00 s | **5.91 s** | **~6 s** |
| EngineCore spawn (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) | 11.00 s | **0.00 s** | **0 s** |
| Engine init + weights (`clear_caches` patch) + KV + compile | 77.53 s (`48.5s` warm) | 246.30 s (`194.2s` cold) | **~35–45 s** (`~20–25s` warm) |
| **Fresh VM → First Token** | **235.33 s** | **282.46 s** | **~70–85 s** |

## 7. Next TPU-Specific Startup Optimizations (Zero Quality/Throughput Loss)

1. **Measure warm 32K cache restore (`tpu_compile_cache_b1_32k.tar` into `/dev/shm`):** Replaces the 194.16 s cold compile with ~20–25 s cache hits.
2. **Shard `venv312_cpu.tar.zst` (1.35 GB) into ~8–12 parts + stream-extract (`zstd -d | tar -x`):** Cuts the 19.85 s prep bottleneck (`13.50 s` single-file download + `6.35 s` unpack) to ~8–10 s.
3. **Strip unused files from `vllm_env` (tests, docs, unused `.so`):** Shrinks the 5.01 GB / 1.35 GB archive before sharding.
4. **Guard `_precompile_structured_decoding` (`tpu-inference#3360`)** when guided decoding is unused: saves ~10–15 programs (~2–3 s).
