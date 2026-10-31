# Colab TPU v5e-1 Pruned-Bucket Fast Cold Start & 32K Benchmark (Run 4)

Measured single-user (`max_num_seqs=1`, `max_model_len=32768`) cold start on Google Colab TPU v5e-1 (`15.748 GiB` usable HBM2) with `Qwen/Qwen3-4B` (bf16) on `vllm-tpu==0.32.0`, pruning token-padding buckets from 10 (`[16..8192]`) down to 3 (`[16, 512, 8192]`) and overlapping `Qwen/Qwen3-4B` weight download with `import jax, vllm`.

---

## 1. End-to-End Cold-Start Progression (`Run 1` $\rightarrow$ `Run 4` `[MEASURED]`)

| Phase / Metric | Run 1 (`17.4K` Warm) | Run 2 (`32K` Cold) | Run 3 (`32K` Warm) | **Run 4 (`32K` Pruned)** | Run 4 vs Run 2 / Run 3 |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **VM Provisioning (`colab new`)** | 7.59 s [m] | 10.40 s [m] | 4.90 s [m] | **5.06 s** [m] | Bare TPU v5e-1 boot |
| **Venv + 32K Cache Download** | 28.92 s [m] | 13.50 s [m] | 3.06 s [m] | **3.68 s** [m] | Overlapped with `zstd` install (`12.48 s`) |
| **Venv + 32K Cache Unpack** | (91.21 s pip) | 6.35 s [m] | 7.83 s [m] | **6.46 s** [m] | Uncontested tmpfs `/dev/shm` unpack |
| **Weights Download (`7.49 GiB`)** | 10.14 s [m] | 11.89 s [m] | 5.89 s [m] | **6.21 s** [m] | **100% overlapped with `import jax, vllm` (`0.0 s` wait)** |
| **`import jax, vllm` + TPU Init** | 48.00 s [m] | (in engine) | 19.30 s [m] | **19.95 s** [m] | Runs concurrently with `6.21 s` weight pull |
| **Engine Init (`LLM(...)` total)** | 142.51 s [m] | 194.16 s [m] | 88.95 s [m] | **49.84 s** [m] | **`-39.11 s` (-44.0%) vs Run 3; `-144.32 s` (-74.3%) vs Run 2** |
| **— Weight Load (`36` layers)** | 11.85 s [m] | — | 10.00 s [m] | **9.41 s** [m] | `default_loader.py` (`jax.clear_caches` patch active) |
| **— Backbone Warmup Pass** | 48.46 s [m] | ~147 s [m] | 59.45 s [m] | **18.77 s** [m] | **`-40.68 s` (-68.4%)**: `3` tasks (`[16, 512, 8192]`) vs `10` |
| **— New XLA Compiles** | 0 [m] | 104 [m] | 0 [m] | **0** [m] | `109` cache entries before & after (100% hit rate) |
| **In-VM Cold Start to 1st Token** | ~227.7 s [m] | 230.48 s [m] | 131.20 s [m] | **90.76 s** [m] | **`-40.44 s` (-30.8%) vs Run 3; `-139.72 s` (-60.6%) vs Run 2** |
| **Global Cold Start to 1st Token**| **235.33 s** [m] | **282.46 s** [m] | **145.44 s** [m] | **103.85 s** [m] | **`-41.59 s` (-28.6%) vs Run 3; `-178.61 s` (-63.2%) vs Run 2** |

---

## 2. How Token-Bucket Pruning (`10` $\rightarrow$ `3`) Saved `40.68 s`

In `tpu-inference`, `get_token_paddings` defaults to 10 powers of two (`[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]`). Because `worker0 backbone` skips AOT lowering and traces/lowers during the warmup pass (~`5.95 s` per bucket), 10 buckets cost **`59.45 s`** in Run 3 even with 100% XLA cache hits.
- Wrapping `get_token_paddings` in `tpu_engine_subrunner.py` to return **`[16, 512, 8192]`** reduced backbone warmup from 10 tasks (`59.45 s`) to **3 tasks (`18.77 s`)** (`core.py:395` `init engine` dropped from **`60.38 s` to `19.92 s`**).
- Because both the precompiler and the runtime batch padder share `get_token_paddings`, runtime batches only pad to `16` (decode), `512` (short prompts), or `8192` (chunked prefill), triggering **`0` runtime JIT compilations**.

---

## 3. Single-Stream Throughput, Long-Context & HBM Verification (`B=1`, `32K`)

- **Short Prompt Decode (256 output tokens, 3 runs):**
  - Wall time per 256-token generation: **`2.9545 s` / `2.9528 s` / `2.9544 s`** (**`86.68–86.73 tok/s`** sustained output speed, identical to Run 3's `2.954 s`; note that subtracting the `512`-bucket `0.4104 s` TTFT from `2.954 s` yields `100.26 tok/s` post-first-token rate).
- **Long-Context Benchmark (`22,286` prompt tokens):**
  - **Prefill Time:** **`2.19 s`** (**`10,176 tok/s`** via `8192` chunked prefill bucket).
  - **Decode Speed:** **`58.35 tok/s`** (`2.72 s` total wall time for 32 tokens).
- **TPU HBM Utilization (Reduced by `70 MiB` vs Run 3):**
  - In Use: **`14.204 GiB`** (vs `14.273 GiB` in Run 3) | Peak: **`14.305 GiB`** (vs `14.374 GiB` in Run 3) | Limit: **`15.748 GiB`** (`1.443 GiB` headroom).

---

## 4. Compute Unit Accounting

- **Compute Units Used:** **`0.08 CU`** (`391.82` $\rightarrow$ `391.74` CU; 38% cheaper than Run 3's `0.13 CU` and 64% cheaper than Run 2's `0.22 CU`).
- **Active Sessions:** `0` active sessions confirmed via `colab --auth=oauth2 sessions`.
