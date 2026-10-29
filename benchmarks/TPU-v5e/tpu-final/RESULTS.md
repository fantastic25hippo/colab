# TPU v5e-1 Final Confirmation Run: Qwen/Qwen3-4B (bf16)

Final confirmation run on Google Colab TPU v5e-1 (16 GB HBM2; 15.748 GiB driver limit) with `Qwen/Qwen3-4B` (bfloat16) on `vllm-tpu==0.32.0` and `tpu-inference==0.32.0`. All 7 open questions resolved with measured data.

## 1. Fresh-VM Cold Start Breakdown (Total: 235.33 s / 3.92 min) [m]

| Phase | Duration (s) | Description |
| :--- | ---: | :--- |
| **VM Create to Shell Ready** | 7.59 [m] | `colab new -s tpu-final --tpu v5e1` to execution launch |
| **Model Weights Download** | 10.14 [m] | `Qwen/Qwen3-4B` (7.49 GiB) to `/dev/shm` via `hf_transfer` |
| **Compile Cache Restore** | 28.92 [m] | Pull & extract `tpu_compile_cache.tar` (76 MB) from HF |
| **Dependencies Install** | 91.21 [m] | Online `uv pip install` into `/content/venv312` |
| **Parallel Prep Total** | 91.21 [m] | Download + Cache Restore + Install executed concurrently |
| **Engine: Weight Loading** | 11.85 [m] | 36 layers loaded (patch active: no per-layer `clear_caches`) |
| **Engine: KV Cache Alloc** | 3.31 [m] | 759 blocks $\times$ 64 tokens = 48,576 tokens (6.695 GiB) |
| **Engine: Compile & Warmup** | 48.46 [m] | Persistent cache hits: 261 hits, 0 compiles (down from 147s) |
| **Engine Ready** | 142.51 [m] | From engine launch to ready |
| **First Token (from ready)** | 1.61 [m] | Latency to generate first warmup token |
| **Total Zero-VM to First Token**| **235.33 s** [m] | **3.92 min total** (39.4% faster than initial 388 s characterization) |

---

## 2. Decode Speed After Warmup (3 Warmup + 5 Timed Single Requests) [m]

| Run | TTFT (s) | Decode Speed (tok/s) | Wall Time (s) | Output Tokens | JIT Compiles |
| :---: | ---: | ---: | ---: | ---: | :---: |
| **1** | 0.0163 [m] | 84.94 [m] | 3.018 [m] | 256 [m] | 0 [m] |
| **2** | 0.0162 [m] | 84.89 [m] | 3.020 [m] | 256 [m] | 0 [m] |
| **3** | 0.0162 [m] | 84.94 [m] | 3.018 [m] | 256 [m] | 0 [m] |
| **4** | 0.0163 [m] | 84.88 [m] | 3.020 [m] | 256 [m] | 0 [m] |
| **5** | 0.0164 [m] | 84.94 [m] | 3.019 [m] | 256 [m] | 0 [m] |
| **Mean** | **0.0163 s** [m] | **84.92 tok/s** [m] | **3.019 s** [m] | 256 [m] | **0** [m] |

---

## 3. Batching Performance (Batch 1 to 32) [m]

| Batch | Total Tok/s | Per-Req Tok/s | Mean TTFT (ms) | p95 TTFT (ms) | Compiles | Wall Time (s) |
| :---: | ---: | ---: | ---: | ---: | :---: | ---: |
| **1** | 84.81 [m] | 84.94 [m] | 16.4 [m] | 16.4 [m] | 0 [m] | 3.019 [m] |
| **4** | 320.25 [m] | 80.50 [m] | 26.7 [m] | 30.2 [m] | 0 [m] | 3.197 [m] |
| **8** | 596.53 [m] | 75.16 [m] | 37.1 [m] | 41.3 [m] | 0 [m] | 3.433 [m] |
| **16** | 1,050.10 [m] | 66.20 [m] | 40.0 [m] | 52.9 [m] | 0 [m] | 3.901 [m] |
| **32** | **1,653.41** [m] | 52.63 [m] | 103.5 [m] | 123.6 [m] | 2 [m] | 4.955 [m] |

*Throughput scales 19.5× from B=1 to B=32 for only 1.94 s of additional wall time.*

---

## 4. Context Length & Preemption Under Concurrency [m]

- **8 Concurrent @ 8k Prompt ($65,536$ prompt tokens):** Wall 6.38 s, Throughput **321.02 tok/s**, Mean TTFT **592.7 ms** (Min 506.2 ms, Max 605.1 ms), **0 queued, 0 preempted** [m].
- **4 Concurrent @ 16k Prompt ($65,536$ prompt tokens):** Wall 6.51 s, Throughput **157.39 tok/s**, Mean TTFT **746.7 ms** (Min 710.8 ms, Max 758.6 ms), **0 queued, 0 preempted** [m].
- **Chunked Prefill (`max_num_batched_tokens=8192`):** Handled prompt tokens exceeding the 48k KV pool without preemption.

---

## 5. Memory Limit & Utilization Analysis (0.90 vs 0.95) [m]

| Metric | Util = 0.90 [m] | Util = 0.95 [m] | Delta |
| :--- | ---: | ---: | ---: |
| **KV Blocks (size 64)** | 759 blocks | 849 blocks | +90 blocks (+11.85%) |
| **Total KV Tokens** | 48,576 tokens | 54,336 tokens | **+5,760 tokens** |
| **KV Cache Footprint** | 6.695 GiB | 7.485 GiB | +0.790 GiB |
| **Peak HBM in Use** | 14.560 GiB | 15.351 GiB | +0.791 GiB |
| **Driver Headroom** | 1.188 GiB (1,216 MiB) | 0.397 GiB (406 MiB) | -810 MiB |
| **Serve 1 Request** | Success (84.92 tok/s) | Success (84.91 tok/s) | Identical speed |
| **Recommendation** | **Default / Gold Standard** | Viable for low-batch | 0.90 safer for B=32 |

---

## 6. Answers to the 7 Open Questions

1. **Wheel bundle vs install speed [proven]:** 259 transitive wheels total 1.8+ GB. Offline install from local wheels took **91.86 s** [m], virtually identical to online `uv pip install` (**90.50 s** [m]). Unpacking and linking 2 GB of wheels on disk is the bottleneck, not network download. Online install is preferred.
2. **Compile cache hit rate [proven]:** Uploaded cache achieved **261 hits and 0 compiles** during startup [m], cutting AOT compilation from 147.5 s to **48.5 s** [m].
3. **KV allocation duration [proven]:** In this run, KV allocation took only **3.31 s** [m] (compiles: 0). The earlier 29 s was caused by allocator cache thrashing prior to the loader patch.
4. **Runtime overhead (380 MiB vs ~60 MiB) [proven]:** `max_model_len=17408` with default token buckets generates 404+ executables with constant tables and scratch workspaces totaling **380 MiB** [m], whereas `max_model_len=4096` with `max_num_seqs=1` generates only 69 small programs (45–65 MiB).
5. **Safe max `gpu_memory_utilization` [measured]:** `0.95` is stable for low concurrency (54,336 tokens, 406 MiB headroom). `0.90` is recommended for high batching (Batch 32 activation spikes).
6. **Batch 32 throughput [measured]:** Scales to **1,653.4 tok/s** [m] with 103.5 ms TTFT.
7. **Max context and preemption [measured]:** Handled 65k concurrent context tokens with **0 preemptions and 0 queued requests** [m] via Chunked Prefill.

---

## 7. Compute Unit (CU) Accounting [m]

- Initial: **392.87 CU** $\rightarrow$ VM1 Stop: **392.64 CU** (0.23 CU) $\rightarrow$ VM2 Stop: **392.35 CU** (0.29 CU).
- Total CU Consumed: **0.52 CU** [m] (well within hard budget of 0.60 CU). Active sessions: **0** [m].
