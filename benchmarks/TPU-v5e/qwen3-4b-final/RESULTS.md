# Qwen/Qwen3-4B Early Bucket & Cache Study on Colab TPU v5e-1

> [!WARNING]
> **Superseded for decode speed & bucket guidance by [`../tpu-characterization/RESULTS.md`](../tpu-characterization/RESULTS.md) and [`../tpu-final/RESULTS.md`](../tpu-final/RESULTS.md).**
> While the `jax.clear_caches()` weight-loading fix discovered here (29.40 s $\rightarrow$ 10.70 s) and the 47–67 MiB net runtime overhead measurement remain valid, restricting `compilation_sizes=[16, 512]` or setting `SKIP_JAX_PRECOMPILE=1` caused 60–76 uncompiled shapes plus per-step `tpu_bench_hook.py` disk I/O to collapse decode to 7.6–30.4 tok/s. With default precompile buckets and out-of-band timing, true single-stream decode is **84.92–88.58 tok/s**.

Early characterization of vLLM runtime memory overhead, startup latencies, compilation caching, and accelerated cold start on Google Colab TPU v5e-1 (16 GB HBM2; 15.75 GiB usable).

## Environment & Versions

- **Host Python:** 3.12.3 venv (`/content/venv312`) via `uv`
- **Versions [measured]:** `vllm-tpu==0.32.0`, `tpu-inference==0.32.0`, `jax==0.11.2`, `jaxlib==0.11.2`, `libtpu==0.0.48`, `torch==2.10.0`
- **Hardware [measured]:** 1x Google TPU v5 lite (`tpu-v5e1`, 15.748 GiB usable driver HBM)
- **Model Geometry [measured]:** 36 layers, 8 KV heads, `head_dim: 128` (144.0 KiB / token)

## Measured Startup Phases & Memory Overhead

| Run | Load (s) | Compile (s) | Startup (s) | Weight (GiB) | KV Cache (GiB) | Peak (GiB) | Overhead (GiB) | Compiled / Cached | Decode (tok/s) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| **M1-base** | 29.40 [m] | 60.56 [m] | 177.76 [m] | 7.493 [m] | 6.683 [m] | 14.241 [m] | **0.065** [m] | 0 / 0 [m] | 76.69 [m] |
| **M1-trick-skip** | 12.16 [m] | 0.00 [m] | 43.00 [m] | 7.494 [m] | 6.645 [m] | 14.171 [m] | **0.032** [m] | 76 / 0 [m] | 7.61 [m] |
| **M1-trick-sizes** | 10.79 [m] | 20.11 [m] | 67.53 [m] | 7.494 [m] | 6.648 [m] | 14.187 [m] | **0.045** [m] | 60 / 39 [m] | 29.89 [m] |
| **M1-trick-both** | 10.70 [m] | 0.00 [m] | 36.32 [m] | 7.494 [m] | 6.645 [m] | 14.171 [m] | **0.032** [m] | 0 / 38 [m] | 18.61 [m] |
| **M1-warm-best** | 10.86 [m] | 6.99 [m] | 44.57 [m] | 7.494 [m] | 6.648 [m] | 14.187 [m] | **0.045** [m] | 0 / 69 [m] | 30.35 [m] |
| **M1-accelerated-cold** | 11.44 [m] | 6.88 [m] | 81.88 [m] | 7.494 [m] | 6.648 [m] | 14.187 [m] | **0.045** [m] | 0 / 69 [m] | 30.38 [m] |

*Legend: `[m]` = Measured directly via `jax.local_devices()[0].memory_stats()`, timer stamps, and JAX compilation cache logs.*

## Detailed Findings

1. **True Runtime Memory Overhead:**
   - Total driver limit: **15.748 GiB** (`16,909,336,064` bytes).
   - Weights in BF16: **7.493 GiB** (`8,045,061,632` bytes).
   - KV Cache allocated (at 0.90 utilization): **6.648–6.683 GiB** (189–190 blocks of 256 tokens = 48,384–48,640 tokens).
   - Peak HBM in use: **14.187–14.241 GiB**.
   - Net Runtime Overhead (`peak - weights - KV`): **0.045–0.065 GiB** (**46.9–67.0 MiB**) [measured]. Runtime overhead is exceptionally lean on TPU v5e-1.

2. **What the Startup Tricks Saved:**
   - **Weight Loading:** The load fix (removing redundant per-module `jax.clear_caches()`) dropped weight loading from **29.40 s** down to **10.70 s** (an **18.70 s / 63.6% speedup**).
   - **Compilation Buckets:** Restricting buckets via `compilation_sizes=[16, 512]` slashed startup compilation from **60.56 s** down to **20.11 s** (a **40.45 s / 66.8% reduction**).
   - **Skip Precompile:** Setting `SKIP_JAX_PRECOMPILE=1` eliminated startup compile entirely (0.00 s), reaching engine ready in **36.32 s** (down from 177.76 s, an **79.6% startup speedup**), at the expense of a 12.5 s JIT compile on the first generated token.
   - **Warm Cache Reuse:** Reusing `/content/tpu_compile_cache` dropped AOT compilation to **6.99 s** with **0 compilations and 69/69 cache hits**. The remaining 6.99 s is pure Python AST tracing and MLIR cache key hashing.

3. **Accelerated Cold Start (Fresh VM2):**
   - Parallel prep: Downloaded model weights (7.30 s) concurrently with restoring `venv312.tar` and `tpu_compile_cache.tar` from Hugging Face (`avgrocketman/tpu-compile-cache`).
   - Restored engine achieved full cache hits (**0 compiled, 69 cached**) and 6.88 s compile time on a completely fresh VM.

4. **Output Sanity Check:**
   - Generated 128 tokens; first 100 characters: `" However, the vastness of space presents challenges that require innovative solutions.  The question"` [measured].

## Hugging Face Artifacts Created

Deposited into private repository `avgrocketman/tpu-compile-cache`:
- `venv312.tar`: Prebuilt standalone virtual environment with Python 3.12, vLLM TPU 0.32.0, tpu-inference 0.32.0, libtpu 0.0.48, and JAX 0.11.2.
- `tpu_compile_cache.tar`: Serialized JAX persistent compilation cache containing all 69 AOT program hashes for Qwen3-4B on TPU v5e-1.
- `versions.json`: Manifest recording package pins, hardware architecture, and compilation sizes.

## Compute Unit (CU) Accounting

- **Initial Balance [measured]:** `393.81` CU
- **VM 1 Stop Balance [measured]:** `393.17` CU (VM 1 consumed: **0.64 CU**)
- **VM 2 Stop Balance [measured]:** `393.11` CU (VM 2 consumed: **0.06 CU**)
- **Net CU Consumed [measured]:** **0.70 CU** (well below hard budget of 1.5 CU)
- **Active Sessions at Termination [measured]:** `0` sessions (verified via `colab sessions`)

## What's Left

1. Upstream patch for `tpu-inference`: Eliminate the redundant `jax.clear_caches()` in `JaxAutoWeightsLoader._load_module` permanently to save ~18 s load time for all TPU models.
2. JAX C++ Tracing: The 6.99 s warm compile latency is dominated by Python-level NNX tracing before cache key lookup; JAX export/AOT serialization can reduce this to sub-second load.
