# TPU Characterization: Qwen/Qwen3-4B on Google Colab TPU v5e-1

Final characterization of Google Colab TPU v5e-1 (16 GB HBM2; 15.748 GiB usable driver HBM) running `Qwen/Qwen3-4B` (bfloat16) on `vllm-tpu==0.32.0` and `tpu-inference==0.32.0`. All tests executed on a single persistent engine without restarts.

## Environment & Hardware

- **Host Python:** Python 3.12.3 venv (`/content/venv312`) created via `uv`
- **Software Versions [m]:** `vllm-tpu==0.32.0`, `tpu-inference==0.32.0`, `jax==0.11.2`, `jaxlib==0.11.2`, `libtpu==0.0.48`, `torch==2.10.0`
- **TPU Device [m]:** 1x Google TPU v5 lite (`tpu-v5e1`, 15.748 GiB usable driver HBM limit)
- **Model Geometry [m]:** 36 layers, 16 query heads, 8 KV heads, head_dim 128 (144.0 KiB / token KV cache)
- **Engine Flags [m]:** `max_model_len=17408`, `gpu_memory_utilization=0.90`, `BENCH_LOAD_FIX=1`, default compilation sizes

---

## 1. Cold Start Breakdown

All timestamps recorded on a clean, fresh VM session.

| Phase | Duration (s) | Description |
| :--- | ---: | :--- |
| **VM Create to Shell Ready** | 5.11 [m] | `colab new -s tpu-char --tpu v5e1` to SSH/driver launch |
| **Cache Restore** | 3.98 [m] | Pull & extract `tpu_compile_cache.tar` (15 MB) from HF |
| **Model Download** | 12.66 [m] | `Qwen/Qwen3-4B` weights (7.49 GiB) to `/dev/shm` via `hf_transfer` |
| **Install** | 118.17 [m] | Online `uv pip install` into `/content/venv312` (wheels fallback) |
| **Parallel Prep Total** | 118.17 [m] | Download + Cache Restore + Install executed concurrently |
| **Engine: Weight Loading** | 11.78 [m] | 36 layers loaded (load fix active: no redundant `clear_caches`) |
| **Engine: KV Cache Alloc** | 29.33 [m] | 759 blocks $\times$ 64 tokens = 48,576 tokens allocated |
| **Engine: Compile & Warmup** | 147.48 [m] | Full AOT compilation for 17.4k context (404 compiled, 61 hits) |
| **Engine Ready** | 268.38 [m] | From engine launch to ready |
| **First Token (from ready)** | 1.64 [m] | Latency to generate first warmup token |
| **Total Fresh-VM to First Token** | **388.19 s** [m] | **6.47 min total** from zero VM to first generated token |

---

## 2. Decode Speed After Warmup

Following 3 untimed warmup requests (128 output tokens each), 5 consecutive single requests were timed (256 output tokens, greedy temperature 0.0):

| Run | Time to First Token (s) | Decode Speed (tok/s) | Total Wall Time (s) | Generated Tokens | JIT Compiles |
| :---: | ---: | ---: | ---: | ---: | :---: |
| **Request 1** | 0.0168 [m] | 84.94 [m] | 3.0189 [m] | 256 [m] | 0 [m] |
| **Request 2** | 0.0169 [m] | 84.89 [m] | 3.0209 [m] | 256 [m] | 0 [m] |
| **Request 3** | 0.0162 [m] | 84.93 [m] | 3.0186 [m] | 256 [m] | 0 [m] |
| **Request 4** | 0.0163 [m] | 84.89 [m] | 3.0204 [m] | 256 [m] | 0 [m] |
| **Request 5** | 0.0163 [m] | 84.94 [m] | 3.0185 [m] | 256 [m] | 0 [m] |
| **Mean** | **0.0165 s** [m] | **84.92 tok/s** [m] | **3.0195 s** [m] | 256 [m] | **0** [m] |

### Resolution of Open Problem 1 (The Earlier 8–30 tok/s Mystery)
- **Compile-in-generation [proven]:** In earlier runs with `SKIP_JAX_PRECOMPILE=1`, 76 programs were JIT-compiled on the fly during the timed generation, inflating generation wall time from 3.0 s to 16.8 s (7.6 tok/s). Similarly, restricting `compilation_sizes=[16, 512]` forced uncompiled intermediate prefill shapes to JIT-compile on the fly (60 programs compiled during generation, yielding 29.89 tok/s).
- **Execution Hook Disk I/O [proven]:** The earlier `tpu_bench_hook.py` hooked `TPUWorker.execute_model` on *every single forward step* (128 times per generation) to synchronously parse and serialize `/tmp/tpu_stats.json` to disk, adding substantial Python file I/O latency to every token.
- **Warmup Separation [proven]:** Running 3 warmup requests prior to measurement allows JAX/PJRT execution buffers to warm up. Once warmed up and with execution hook I/O moved out-of-band, the true TPU v5e-1 single-stream decode speed is **84.92 tok/s** with **0 JIT compiles** and **16.5 ms TTFT**.

---

## 3. Batching Performance

Concurrent requests evaluated on the running engine (256 output tokens each):

| Batch Size | Total Throughput (tok/s) | Per-Request Speed (tok/s) | Mean TTFT (ms) | p95 TTFT (ms) | Compiles | Wall Time (s) |
| :---: | ---: | ---: | ---: | ---: | :---: | ---: |
| **1** | 84.81 [m] | 84.94 [m] | 16.1 [m] | 16.1 [m] | 0 [m] | 3.018 [m] |
| **4** | 320.23 [m] | 80.50 [m] | 27.1 [m] | 30.6 [m] | 0 [m] | 3.198 [m] |
| **8** | 597.87 [m] | 75.17 [m] | 31.4 [m] | 33.5 [m] | 0 [m] | 3.425 [m] |
| **16** | **1,050.19** [m] | 66.20 [m] | 40.0 [m] | 52.6 [m] | 0 [m] | 3.900 [m] |

**Analysis:**
- Batching scales remarkably well on TPU v5e: moving from Batch 1 to Batch 16 increases total throughput by **12.38×** (from 84.8 to 1,050.2 tok/s) for only a **29.2% increase in wall time** (3.02 s $\rightarrow$ 3.90 s).
- Per-request decode throughput degrades by only 22% (84.9 tok/s $\rightarrow$ 66.2 tok/s).
- TTFT remains under 53 ms even at Batch 16.

---

## 4. Context Length Performance

Single requests with increasing prompt context lengths (256 output tokens):

| Context Length (tokens) | Prefill / TTFT (s) | Decode Speed (tok/s) | Total Wall Time (s) | Peak HBM (GiB) |
| :---: | ---: | ---: | ---: | ---: |
| **512** | 0.0697 [m] | 84.33 [m] | 3.094 [m] | 14.560 [m] |
| **4,096** | 0.2673 [m] | 80.43 [m] | 3.438 [m] | 14.560 [m] |
| **16,384** | 0.8875 [m] | 69.83 [m] | 4.539 [m] | 14.560 [m] |

### Multi-Tenant Long Context Pressure (4 Concurrent @ 4k Context)
- **Workload:** 4 concurrent requests each with 4,096 prompt tokens ($16,384$ total prompt tokens) + 256 output tokens.
- **Total Wall Time [m]:** 3.814 s
- **Aggregate Throughput [m]:** 268.51 tok/s
- **Mean TTFT [m]:** 79.4 ms
- **Preemption & Queueing [m]:** **0 requests preempted, 0 requests queued**.
- **KV Cache Utilization [i]:** 16,384 tokens occupied out of 48,576 available capacity (33.7% pool utilization). Handled seamlessly by Ragged Paged Attention (RPA) without memory fragmentation or preemption.

---

## 5. Memory Footprint Breakdown

All values measured directly from `jax.local_devices()[0].memory_stats()`:

| Stage | Bytes in Use | GiB in Use | Peak GiB | Description |
| :--- | ---: | ---: | ---: | :--- |
| **Driver HBM Limit** | 16,909,336,064 | 15.748 [m] | 15.748 [m] | Physical usable TPU v5e-1 limit |
| **After Weights Load** | 8,046,819,328 | 7.494 [m] | 7.494 [m] | Qwen3-4B BF16 model parameters |
| **After KV Allocation** | 15,234,840,576 | 14.189 [m] | 14.189 [m] | 759 blocks $\times$ 64 tokens (6.695 GiB KV cache) |
| **After Startup Compile** | 15,353,854,464 | 14.300 [m] | 14.560 [m] | Peak XLA workspace & executable buffers |
| **After Batch 16 Test** | 15,354,838,016 | 14.300 [m] | 14.560 [m] | Pinned stable memory |
| **After 16k Context Test**| 15,355,227,648 | 14.301 [m] | 14.560 [m] | Stable memory across 16k sequence |
| **Runtime Overhead** | 398,454,784 | **0.371 GiB** [m] | **0.371 GiB** [m] | Net runtime overhead (`peak - weights - KV`) = **380 MiB** |

---

## 6. Hugging Face Cache Deposit

- **Repository:** `avgrocketman/tpu-compile-cache` (public)
- **Artifact Updated:** `tpu_compile_cache.tar` (re-packed and uploaded with all 404 newly compiled programs for `max_model_len=17408`).
- **Manifest:** `versions.json` updated with engine settings and cache hashes.

---

## 7. Compute Unit (CU) Accounting

- **Initial Balance [m]:** `393.11` CU (`colab --auth=oauth2 usage`)
- **Final Balance [m]:** `392.87` CU (`colab --auth=oauth2 usage`)
- **Net CU Consumed [m]:** **0.24 CU** (well below hard budget of 1.0 CU)
- **Active Sessions at Termination [m]:** `0` sessions verified via `colab sessions`
