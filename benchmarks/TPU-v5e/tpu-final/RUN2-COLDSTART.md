# Colab TPU v5e-1 Cold Start & 32K Serving Benchmark (Run 2)

Timed cold start from bare VM to first token using `venv312_cpu.tar.zst` and Qwen3-4B bf16 at `max_model_len=32768`.

---

## 1. End-to-End Cold-Start Timeline (VM Start to First Token)

| Phase / Step | Duration | Status | Notes |
| :--- | ---: | :--- | :--- |
| **VM Provisioning & SSH Ready** | ~10.4 s | [measured] | Bare TPU v5e-1 runtime initialization |
| **Remount `/dev/shm` (exec)** | 0.08 s | [measured] | `mount -o remount,exec /dev/shm` (`rc=0`) |
| **Venv Download & Unpack** | 19.85 s | [measured] | Download: 13.50 s; Unpack: 6.35 s (into tmpfs) |
| **Weights Download (`snapshot_download`)**| 11.89 s | [measured] | Parallel with venv; 7.49 GiB to `/dev/shm` |
| **Net Parallel Prep Bottleneck** | **19.85 s** | [measured] | **-71.36 s (-78%) vs 91.21s baseline** |
| **TPU Check B (`TPU_SKIP_MDS_QUERY=1`)** | 5.91 s | [measured] | Bypasses GCE metadata 404 timeout |
| **Engine Setup & AOT 32K Compilation** | 194.16 s | [measured] | 104 programs compiled (fresh 32K cache miss) |
| **First Token (Short Prompt TTFT)** | 0.08 s | [measured] | 75.5 ms TTFT |
| **Total VM Start to First Token** | **282.46 s** | [measured] | Includes first-time 104-program compilation |

---

## 2. TPU Platform Check: A vs B

- **Check A (Default Env)**: **15.56 s** (stalls on `cloud_tpu_init.py` metadata 404 HTTP retry loop).
- **Check B (`TPU_SKIP_MDS_QUERY=1 ...`)**: **5.91 s** (**9.65 s / 62% faster**; TPU detected successfully).
- *Conclusion*: Static topology env vars safely eliminate ~10 s of metadata polling latency.

---

## 3. Serving Benchmarks & Context Scaling (`max_model_len=32768`)

- **Batch 1 Decode Speed (256 tokens, 3 runs)**:
  - Run 1: 88.55 tok/s | Run 2: 88.61 tok/s | Run 3: 88.59 tok/s
  - **Mean Decode Throughput**: **88.58 tok/s** (2.95 s per 256 tokens).
- **30K Long-Context Evaluation (25,706 tokens)**:
  - **Prefill Time**: **0.09 s** (90 ms via Chunked Prefill `max_num_batched_tokens=8192`).
  - **Decode Throughput**: **10.20 tok/s** (3.13 s total wall time).
- **HBM Memory Utilization**:
  - In Use: **14.274 GiB** (90.6% of 15.748 GiB driver limit).
  - Peak In Use: **14.376 GiB** (91.3% of limit; zero OOM, 1.37 GiB headroom).

---

## 4. Cache Deposit & Compute Unit Accounting

- **Compile Cache**: Packed 104 AOT programs as `tpu_compile_cache_b1_32k.tar` (**53.66 MB**) and deposited to `avgrocketman/tpu-compile-cache` (retaining existing cache).
- **Compute Units Used**: **0.22 CU** (392.17 $\rightarrow$ 391.95 CU; budget was 0.40 CU).
- **Session Cleanup**: 0 active sessions; recurring 5-minute timer deleted.
