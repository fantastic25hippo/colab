# Colab TPU v5e-1 Warm-Cache Fast Cold Start & 32K Benchmark (Run 3)

Measured single-user (`max_num_seqs=1`, `max_model_len=32768`) fast cold start on Google Colab TPU v5e-1 (`15.748 GiB` usable HBM2) with `Qwen/Qwen3-4B` (bf16) on `vllm-tpu==0.32.0`, restoring `venv312_cpu.tar.zst` and `tpu_compile_cache_b1_32k.tar` directly into `/dev/shm`.

---

## 1. End-to-End Cold-Start Timeline (`282.46 s` $\rightarrow$ `145.44 s` Global / `131.20 s` In-VM)

| Phase / Step | Run 2 (`32K` Cold) | Run 3 (`32K` Warm) | Delta / Notes |
| :--- | ---: | ---: | :--- |
| **VM Provisioning (`colab new --tpu v5e1`)** | ~10.40 s [m] | **4.90 s** [m] | Bare TPU v5e-1 runtime creation |
| **Script & Token Upload + Exec Handshake** | ~41.60 s [m] | **9.34 s** [m] | Single `colab exec` + pruned uploads |
| **Remount `/dev/shm` (`exec`) + THP Probe** | 0.08 s [m] | **0.06 s** [m] | `mount -o remount,exec /dev/shm` (`rc=0`; sysfs THP read-only) |
| **`apt-get install zstd`** | ~11.00 s [m] | **~11.00 s** [m] | Container package install before unpack |
| **Venv + 32K XLA Cache Download to `/dev/shm`** | 13.50 s [m] | **3.06 s** [m] | **-10.44 s (-77%)** via uncontested NIC pull to tmpfs |
| **Venv + 32K XLA Cache Unpack (`zstd -d -T0`)** | 6.35 s [m] | **7.83 s** [m] | Includes `53.66 MB` cache tar + archive deletion |
| **Weights Download (`Qwen/Qwen3-4B`, 7.49 GiB)** | 11.89 s [m] | **5.89 s** [m] | Overlapped with venv unpack (`wait_for_weights_s = 0.0 s`) |
| **Net `/dev/shm` Prep Time (Venv + Cache + Weights)** | 19.85 s [m] | **10.89 s** [m] | **-8.96 s (-45%)** vs Run 2 (`-80.32 s` vs 91.21 s baseline) |
| **Standalone TPU Pre-Checks (`Check A` + `Check B`)** | 21.47 s [m] | **0.00 s** [m] | **-21.47 s (Eliminated)** |
| **Subrunner `import jax, vllm` + PJRT TPU Init** | (in engine) | **19.30 s** [m] | First cold `.so` load + GCE metadata 404 probe |
| **Engine Init (Weights + KV + Warm 32K Cache)** | 194.16 s [m] | **88.95 s** [m] | **-105.21 s (-54%)**; `109` cache entries, **`0` new compiles** |
| **First Token (Short Prompt TTFT)** | 0.0755 s [m] | **0.0230 s** [m] | **23.0 ms TTFT** |
| **In-VM Cold Start to First Token** | 230.48 s [m] | **131.20 s** [m] | **-99.28 s (-43.1%)** |
| **Total Global Cold Start to First Token** | **282.46 s** [m] | **145.44 s** [m] | **-137.02 s (-48.5%)** vs Run 2 (`-89.89 s` vs 235.33 s Run 1) |

---

## 2. Inside the `88.95 s` Engine Initialization (`0` New Compiles)

From `/tmp/run3_coldstart.log` timestamps (`09:36:58` to `09:38:27`):
- **Config, Tokenizer & Architecture Resolution (`HF_HUB_OFFLINE=1`):** `16.0 s` (`09:36:58` $\rightarrow$ `09:37:14`).
- **Gloo Rendezvous, Mesh & Worker Init:** `2.0 s` (`09:37:14` $\rightarrow$ `09:37:16`).
- **36-Layer BF16 Weight Load (`7.49 GiB` from `/dev/shm`) + KV Cache Alloc (`48,624` tokens):** `10.0 s` (`09:37:16` $\rightarrow$ `09:37:26`).
- **AOT Subgraph Tracing, MLIR Lowering & Warmup Pass (`109` cached entries, `0` compiles):** `60.38 s` (`59.45 s` across the 10 backbone token buckets `[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]` + `0.93 s` helpers).

---

## 3. Single-Stream Decode & Long-Context Benchmarks (`max_model_len=32768`, `B=1`)

- **Short Prompt Decode (256 output tokens, 3 timed runs after warmup):**
  - Run 1: **87.01 tok/s** (`2.9539 s`) | Run 2: **86.94 tok/s** (`2.9560 s`) | Run 3: **87.03 tok/s** (`2.9530 s`)
  - **Mean Decode Speed:** **86.99 tok/s** (23.0 ms TTFT) — zero regression.
- **Long-Context Evaluation (`22,286` prompt tokens, distinct prompts to avoid prefix-cache artifact):**
  - **Prefill Time:** **2.18 s** (**10,223 tok/s** chunked prefill at `max_num_batched_tokens=8192`).
  - **Decode Throughput:** **58.13 tok/s** (`2.71 s` wall time for 32 tokens) — resolves the Run 2 `0.09 s / 10.20 tok/s` prefix-cache timing artifact.
- **HBM Utilization:**
  - In Use: **14.273 GiB** | Peak: **14.374 GiB** | Driver Limit: **15.748 GiB** (`1.374 GiB` safe headroom).

---

## 4. Compute Unit Accounting & SSH Verification

- **SSH Monitoring:** Verified `/dev/shm` at **57% utilization** (`13 GB` used / `9.7 GB` free) with unpacked venv and `Qwen3-4B` weights co-resident in RAM.
- **Compute Units Used:** **0.13 CU** (`391.95` $\rightarrow$ `391.82` CU at `2.92 CU/hr`).
- **Session Cleanup:** `0` active sessions confirmed via `colab --auth=oauth2 sessions`.
