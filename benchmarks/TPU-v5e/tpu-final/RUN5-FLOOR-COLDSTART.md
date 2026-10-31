# Qwen/Qwen3-4B on TPU v5e-1 — Run 5 Floor Cold-Start (`Batch=1`, `32K` Context)

- **Date:** 2026-10-31
- **Hardware:** Google Colab TPU v5e-1 (`v5e1`, 1 chip, `15.748 GiB` HBM, `1.60 CU/hr`)
- **Engine:** `vllm-tpu==0.32.0` (`V1` engine, `VLLM_ENABLE_V1_MULTIPROCESSING=0`)
- **Model:** `Qwen/Qwen3-4B` (BF16, `max_model_len=32768`, `max_num_seqs=1`, `gpu_memory_utilization=0.90`)
- **Pre-warmed Cache Artifact:** `avgrocketman/tpu-compile-cache` (`tpu_compile_cache_b1_32k.tar`, `109` XLA entries + `zstd_bin`)
- **Raw JSON Output:** [run5_floor_coldstart_results.json](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu-final/run5_floor_coldstart_results.json)
- **Compute Units Spent (`MEASURED`):** **`0.04 CU`** (`391.74` $\rightarrow$ `391.70` CU)

---

## 1. Floor Cold-Start Optimizations Applied (`MEASURED`)

1. **Zero `apt-get` (`zstd_bin` in HuggingFace Cache Repo):** Pre-staged `/usr/bin/zstd` (`999 KB`) in `avgrocketman/tpu-compile-cache` (`zstd_bin`) and downloaded it concurrently alongside `venv312_cpu.tar.zst` and `tpu_compile_cache_b1_32k.tar` in **`2.60 s`**, eliminating `apt-get update && apt-get install zstd` (`12.48 s` in Run 4). Total venv + cache download and unpack dropped from `19.99 s` to **`9.11 s`**.
2. **Killed GCE Metadata Polling via `/etc/hosts` Loopback:** Appended `127.0.0.1 metadata.google.internal metadata` to `/etc/hosts` and set `GCE_METADATA_HOST=127.0.0.1:1`, turning remote HTTP 404 retry stalls into an immediate local `ECONNREFUSED` (`0 ms`).
3. **2-Bucket Token Padding (`[16, 8192]`) + `12`-Token Warmup Prompt:** Pruned `get_token_paddings` from `[16, 512, 8192]` (`3` tasks, `18.77 s`) to **`[16, 8192]` (`2` tasks, `12.19 s`)**, saving **`6.58 s`** of XLA executable HBM loading while keeping **`0` new XLA compilations**. Using a `12`-token prompt ensured the initial warmup/TTFT request hit the `16`-token bucket (**`21.4 ms` TTFT**).
4. **Zero Pre-Exec `colab upload` Calls:** Passed `SUBRUNNER_B64` and `HF_TOKEN` directly via `colab exec --env`, reducing `colab new` + pre-exec overhead from `13.09 s` (`7.63 s` + `5.46 s`) down to **`8.14 s` total (`4.77 s` VM provision + `3.37 s` exec attach)**.

---

## 2. Cold-Start Progression Across All 5 Runs (`MEASURED`)

| Stage / Metric | Run 1 (16K Cold) | Run 2 (32K Cold) | Run 3 (32K Warm) | Run 4 (Pruned 3-Bucket) | Run 5 (Floor 2-Bucket) | Run 2 $\rightarrow$ Run 5 |
|---|---:|---:|---:|---:|---:|---:|
| **VM Provision + Pre-Exec** | `9.29 s` | `51.98 s` | `14.24 s` | `13.09 s` | **`8.14 s`** | **`-43.84 s` (`-84.3%`)** |
| **Venv + XLA Cache Prep (`/dev/shm`)** | `16.41 s` | `20.23 s` | `10.89 s` | `19.99 s` | **`9.11 s`** | **`-11.12 s` (`-55.0%`)** |
| **Weights Download (`Qwen3-4B`)** | `9.23 s` | `10.74 s` | `9.96 s` | `6.00 s` (hidden) | **`6.03 s` (`0.00 s` net)** | **`-10.74 s` (`-100%` net)** |
| **Python Imports + TPU Init** | `29.74 s` | `29.68 s` | `30.89 s` | `20.30 s` | **`19.39 s`** | **`-10.29 s` (`-34.7%`)** |
| **Engine Init & XLA Load (`LLM(...)`)** | `148.55 s` | `169.39 s` | `88.95 s` | `49.84 s` | **`43.94 s`** | **`-125.45 s` (`-74.1%`)** |
| $\quad\hookrightarrow$ *Backbone Warmup Pass* | *10 tasks (JIT)* | *10 tasks (JIT)* | *`59.45 s` (10 tasks)* | *`18.77 s` (3 tasks)* | ***`12.19 s` (2 tasks)*** | ***`-47.26 s` vs Run 3*** |
| **New XLA Compilations** | `97` | `109` | **`0`** | **`0`** | **`0`** | **`-109` (`-100%`)** |
| **Short-Prompt TTFT (`1` tok)** | `26.5 ms` | `23.5 ms` | `23.8 ms` | `415.3 ms` | **`21.4 ms`** | **`-2.1 ms`** |
| **In-VM Cold-Start to First Token** | `204.47 s` | `230.48 s` | `131.20 s` | `90.76 s` | **`73.08 s`** | **`-157.40 s` (`3.15x` faster)** |
| **Global Cold-Start (`colab new` $\rightarrow$ 1st Tok)** | `213.76 s` | `282.46 s` | `145.44 s` | `103.85 s` | **`81.22 s`** | **`-201.24 s` (`3.48x` faster)** |
| **Compute Units Spent** | `0.19 CU` | `0.26 CU` | `0.13 CU` | `0.08 CU` | **`0.04 CU`** | **`-84.6%` CU** |

---

## 3. Inference Throughput & Regression Verification (`MEASURED`)

| Workload (`Batch=1`, `max_model_len=32768`) | Run 3 (10 Buckets) | Run 4 (3 Buckets) | Run 5 (`[16, 8192]`) | Status |
|---|---:|---:|---:|---|
| **Short-Prompt TTFT (`1` tok)** | `23.8 ms` | `415.3 ms` | **`21.4 ms`** | **Zero regression (`16`-tok bucket hit)** |
| **256-Token Decode Run 1** | `2.9508 s` (`86.99 tok/s`) | `2.9522 s` (`99.7 tok/s` adj) | **`2.9530 s` (`86.98 tok/s`)** | **Identical (`2.95 s` wall)** |
| **256-Token Decode Run 2** | `2.9531 s` (`86.92 tok/s`) | `2.9482 s` (`99.89 tok/s` adj) | **`2.9516 s` (`87.02 tok/s`)** | **Identical (`2.95 s` wall)** |
| **256-Token Decode Run 3** | `2.9500 s` (`87.01 tok/s`) | `2.9530 s` (`99.67 tok/s` adj) | **`2.9538 s` (`86.96 tok/s`)** | **Identical (`2.95 s` wall)** |
| **Long Context (`22,286` tok) Prefill** | `2.18 s` (`10,223 tok/s`) | `2.19 s` (`10,176 tok/s`) | **`2.18 s` (`10,223 tok/s`)** | **Identical (`2.18 s`)** |
| **Long Context (`22,286` tok) Decode** | `58.13 tok/s` (`2.71 s` wall) | `58.35 tok/s` (`2.72 s` wall) | **`58.31 tok/s` (`2.72 s` wall)** | **Identical (`58.3 tok/s`)** |
| **TPU HBM In-Use / Peak / Limit** | `14.192 / 14.305 / 15.748 GiB` | `14.192 / 14.305 / 15.748 GiB` | **`14.192 / 14.293 / 15.748 GiB`** | **Identical HBM footprint** |
