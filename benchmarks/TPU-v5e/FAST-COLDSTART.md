# Fast Cold Start on Colab TPU v5e-1: Universal Model Runbook

Model-independent engineering guide to achieve physical-floor cold-start latency (**`73.08 s` in-VM / `81.22 s` global** from `colab new`) on Google Colab TPU v5e-1 (`15.748 GiB` HBM limit).

---

## 1. Where Cold-Start Time Goes: Progression to the `73.08 s` Floor (`MEASURED`)

| Phase | Naive Baseline (`17.4K`) | Run 2 (`32K` Cold) | Run 3 (`32K` Warm) | Run 4 (Pruned 3-Bucket) | Run 5 (Floor 2-Bucket) | Net Gain (Run 2 $\rightarrow$ Run 5) |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **VM Provision + Pre-Exec** | ~10 s | 51.98 s | 14.24 s | 13.09 s | **8.14 s** (`4.77 s` VM + `0` uploads) | **-43.84 s (-84.3%)** |
| **Venv + XLA Cache Prep (`/dev/shm`)** | 91.21 s (`pip`) | 20.23 s | 10.89 s | 19.99 s (`apt`) | **9.11 s** (`2.60 s` dl + `6.51 s` unpack, `0` apt) | **-11.12 s (-55.0%)** |
| **Model Weights Download** | 10.14 s | 10.74 s | 9.96 s | 6.00 s (hidden) | **6.03 s (`0.00 s` net, hidden in imports)** | **-10.74 s (-100% net)** |
| **Imports & TPU Init** | 48.00 s (ext4) | 29.68 s | 30.89 s | 20.30 s | **19.39 s** (`/etc/hosts` loopback MDS kill) | **-10.29 s (-34.7%)** |
| **Engine Process Spawn** | 11.00 s | 0.00 s | 0.00 s | 0.00 s | **0.00 s** (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) | **-11.00 s (Eliminated)** |
| **Engine Init & XLA Load (`LLM(...)`)** | 63.62 s | 169.39 s (`109` compiles) | 88.95 s (`10` buckets) | 49.84 s (`3` buckets) | **43.94 s** (`2` buckets `[16, 8192]`, `0` compiles) | **-125.45 s (-74.1%)** |
| **Short-Prompt TTFT (`1` tok)** | 16.3 ms | 23.5 ms | 23.8 ms | 415.3 ms | **21.4 ms** (`12`-tok prompt hits bucket `16`) | **Zero regression** |
| **In-VM Cold-Start to 1st Token** | **~225 s** | **230.48 s** | **131.20 s** | **90.76 s** | **`73.08 s` [MEASURED]** | **`-157.40 s` (`3.15x` faster)** |
| **Global Cold-Start (`colab new` $\rightarrow$ 1st Tok)** | **235.33 s** | **282.46 s** | **145.44 s** | **103.85 s** | **`81.22 s` [MEASURED]** | **`-201.24 s` (`3.48x` faster)** |
| **Compute Units Spent** | 0.52 CU | 0.26 CU | 0.13 CU | 0.08 CU | **`0.04 CU` [MEASURED]** | **-84.6% CU** |

---

## 2. The 10 Universal Model-Independent Cold-Start Rules (`MEASURED`)

1. **Zero Pre-Exec `colab upload` Round-Trips:** Pass small helper scripts (`SUBRUNNER_B64`) and `HF_TOKEN` directly via `colab exec --env` immediately after `colab new`. Eliminates `5.46 s` of CLI file-upload round-trips [MEASURED, `RUN5-FLOOR-COLDSTART.md`].
2. **Mandatory `/dev/shm` Exec Remount:** Colab mounts `/dev/shm` with `noexec` by default. Run `mount -o remount,exec /dev/shm` first (`0.01 s`) so shared libraries (`libtorch_global_deps.so`, `libtpu.so`) can `mmap(PROT_EXEC)` from tmpfs [MEASURED, `RUN1-VENV.md:44`].
3. **Zero-`apt` Prebuilt CPU-Torch Venv + `zstd_bin`:** Standard PyPI `torch` pulls >1.4 GB of unused NVIDIA CUDA wheels. Prebuilding `vllm-tpu==0.32.0` with `torch==2.10.0+cpu`, precompiling `.pyc`, and staging `venv312_cpu.tar.zst` (`1.39 GB`), `tpu_compile_cache_b1_32k.tar` (`53.7 MB`), and `zstd_bin` (`999 KB`) in `avgrocketman/tpu-compile-cache` downloads all three in **`2.60 s`** and unpacks in **`6.51 s`** (`9.11 s` total, zero `apt-get`) [MEASURED, `RUN5-FLOOR-COLDSTART.md`].
4. **Uncontested `/dev/shm` Unpack, Then Overlapped Weight Download During Imports:** Unpack the `5.1 GB` venv into `/dev/shm` first with zero competing I/O, then start `snapshot_download(..., max_workers=12)` in a background thread at the exact moment `tpu_engine_subrunner.py` starts `import jax; from vllm import LLM`. Weight download (`6.03 s` for `7.55 GB`) finishes completely inside the `19.39 s` import window (`0.00 s` net wait) [MEASURED, `RUN5-FLOOR-COLDSTART.md`].
5. **Kill Both Python & C++ `libtpu.so` GCE Metadata Polling:** Set `TPU_SKIP_MDS_QUERY=1 TPU_TOPOLOGY=1x1 TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost`, append `127.0.0.1 metadata.google.internal metadata` to `/etc/hosts`, and monkey-patch `urllib.request.urlopen` so both Python and C++ `libtpu.so` fail fast (`ECONNREFUSED` in `0 ms`) instead of waiting on remote HTTP 404 retries [MEASURED, `RUN5-FLOOR-COLDSTART.md`].
6. **Single-Process In-Place Engine Execution:** Export `VLLM_ENABLE_V1_MULTIPROCESSING=0` to run `EngineCore` in-process, saving `11.00 s` of child `spawn` + re-import overhead [MEASURED, `STARTUP-BREAKDOWN.md:18`].
7. **Monkey-Patch `jax.clear_caches` & `_precompile_structured_decoding`:** Wrap `builtins.__import__` to no-op `weight_utils.jax.clear_caches` (prevents wiping XLA caches after weight load) and skip `_precompile_structured_decoding` (`1.74 s` saved) [MEASURED, `RUN4-PRUNED-COLDSTART.md`].
8. **Prune Single-User (`Batch=1`) Token-Padding Ladder to `[16, 8192]`:** Wrapping `get_token_paddings` across `tpu_inference` modules to return `[16, 8192]` instead of all 10 power-of-two buckets `[16, 32, ..., 8192]` cuts the `Warm-up call pass` from `59.45 s` (10 tasks) down to **`12.19 s` (2 tasks)** — saving **`47.26 s`** of XLA executable deserialization and HBM upload while requiring **`0` new XLA compilations** [MEASURED, `RUN5-FLOOR-COLDSTART.md`].
9. **Warmup with a `<=16`-Token Prompt:** Ensure the initial 1-token warmup query is `<=16` tokens so it dispatches on the `16`-token bucket (**`21.4 ms` TTFT**) rather than padding to `8192` [MEASURED, `RUN5-FLOOR-COLDSTART.md`].
10. **Strict Offline Mode Once Cached:** Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the subrunner environment to eliminate HuggingFace Hub HTTP version/config checks (`~2–3 s`) during `LLM(...)` initialization [MEASURED, `RUN4-PRUNED-COLDSTART.md`].

---

## 3. Verified Throughput & Long-Context Performance (`MEASURED`)

- **Short-Prompt TTFT (`12` tokens, bucket `16`):** **`21.4 ms`** [MEASURED, `RUN5-FLOOR-COLDSTART.md`].
- **Single-User (`Batch=1`) Decode Throughput:** **`86.99 tok/s`** mean across 3 × 256-token runs (`2.95 s` wall time per 256 tokens) [MEASURED, `RUN5-FLOOR-COLDSTART.md`].
- **Long-Context (`22,286` tokens, distinct prompts to bypass prefix cache):**
  - **Chunked Prefill (`22,286` tokens via bucket `8192`):** **`2.18 s`** (`~10,223 tok/s`) [MEASURED, `RUN5-FLOOR-COLDSTART.md`].
  - **Long-Context Decode (`@ 22,286` KV cache):** **`58.31 tok/s`** (`2.72 s` total wall time for prefill + 32 tokens) [MEASURED, `RUN5-FLOOR-COLDSTART.md`].
  - **TPU HBM Footprint:** `14.192 GiB` in-use, `14.293 GiB` peak, `15.748 GiB` limit (`1.455 GiB` free headroom) [MEASURED, `RUN5-FLOOR-COLDSTART.md`].

---

## 4. Reference Implementation (3-Script Pipeline & Subagent Workflow)

All future agents must reuse this exact 3-file architecture when starting or benchmarking any model on TPU v5e-1:

1. **Host Runner ([`run_coldstart.py`](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/run_coldstart.py)):**
   - Checks `colab --auth=oauth2 usage`, provisions the VM (`colab new -s tpu-coldstart --tpu v5e1`), and immediately launches `colab exec` with `--env T0_GLOBAL=...`, `--env SUBRUNNER_B64=...`, and `--env HF_TOKEN=...` (**zero `colab upload` calls**).
   - Always tears down the runtime in `finally:` (`colab stop -s tpu-coldstart`) and verifies `colab sessions` is empty.
2. **In-VM Cold-Start Driver ([`tpu_coldstart_driver.py`](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu_coldstart_driver.py)):**
   - Writes `/content/tpu_engine_subrunner.py` from `SUBRUNNER_B64`, writes `HF_TOKEN`, appends `127.0.0.1 metadata.google.internal metadata` to `/etc/hosts`, and runs `mount -o remount,exec /dev/shm`.
   - Downloads `venv312_cpu.tar.zst`, the model's XLA cache `.tar`, and `zstd_bin` from `avgrocketman/tpu-compile-cache` into `/dev/shm` (`max_workers=3`, zero `apt-get`), unpacks them into `/dev/shm`, and deletes the `.tar` archives immediately to free RAM.
   - Starts `snapshot_download("<model_repo>", local_dir="/dev/shm/<model>", max_workers=12)` in a background thread **at the exact moment** it spawns `/dev/shm/vllm_env/bin/python -u /content/tpu_engine_subrunner.py`.
3. **In-Process Engine Subrunner ([`tpu_engine_subrunner.py`](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu_engine_subrunner.py)):**
   - Installs the `builtins.__import__` hook before importing `jax` or `vllm` to: (a) no-op `weight_utils.jax.clear_caches`, (b) no-op `_precompile_structured_decoding`, and (c) prune `get_token_paddings` to `[16, 8192]` (`[paddings[0], paddings[-1]]`).
   - Imports `jax` and `vllm` while weights finish downloading in the background, waits on `/dev/shm/.weights_ready`, initializes `LLM(...)` in-process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `HF_HUB_OFFLINE=1`), and warms up on a `<=16`-token prompt.
4. **Flash Subagent + Live SSH Monitoring:**
   - Delegate the run to a `flash` subagent (`invoke_subagent` / `send_message`) that runs `python3 -u benchmarks/TPU-v5e/run_coldstart.py > /tmp/run_coldstart.log 2>&1` and monitors `/content/build.log` and `/dev/shm` over `colab --auth=oauth2 ssh -s tpu-coldstart --proxy-mode`.
