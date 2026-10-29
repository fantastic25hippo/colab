# Fast Cold Start on Colab TPU v5e-1: Universal Model Runbook

Model-independent engineering guide to achieve minimal cold-start latency on Google Colab TPU v5e-1 (16 GB HBM2, 15.748 GiB driver limit).

---

## 1. Where Cold-Start Time Goes: Baseline vs Fast Pipeline

| Phase | Naive Baseline (234 s) | Fast Pipeline (Run 2) | Gain / Status |
| :--- | ---: | ---: | :--- |
| **Package Install / Venv** | 91.21 s [MEASURED, STARTUP-BREAKDOWN.md:14] | 19.85 s parallel prep [MEASURED, RUN2-COLDSTART.md:13] | **-71.36 s (-78%)** |
| **Model Weights Download** | 10.14 s [MEASURED, STARTUP-BREAKDOWN.md:12] | Overlapped in 19.85 s prep [MEASURED, RUN2-COLDSTART.md:13] | **Zero added time** |
| **Imports & Platform Init**| 48.00 s on ext4 [MEASURED, STARTUP-BREAKDOWN.md:15] | 15.56 s (A) -> 5.91 s (B) [MEASURED, RUN2-COLDSTART.md:14] | -9.65 s from env vars; rest from tmpfs/CPU torch |
| **Engine Process Spawn**   | 11.00 s [MEASURED, STARTUP-BREAKDOWN.md:18] | 0.00 s (in-process engine) | **Eliminated** |
| **Weight Load & KV Alloc** | 15.16 s [MEASURED, STARTUP-BREAKDOWN.md:21-22] | Included in engine init | Streamlined |
| **JAX Compilation**        | 48.46 s warm hit [MEASURED, STARTUP-BREAKDOWN.md:23] | 194.16 s cold miss [MEASURED, RUN2-COLDSTART.md:15] | [UNTESTED warm 32K] |
| **Total to First Token**   | **233.7 s** [MEASURED, STARTUP-BREAKDOWN.md:8] | **282.46 s cold miss** [MEASURED] / **~75–85 s warm** | Estimated, not measured (no warm 32K run timed yet) |

---

## 2. Prebuilt CPU-Torch Virtual Environment

Standard PyPI `torch` pulls 15 NVIDIA CUDA packages (>1.4 GB) unused by TPU [MEASURED, STARTUP-BREAKDOWN.md:66].
- **CPU-Only Torch Pin**: `--extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match "torch==2.10.0+cpu"` completely eliminates all NVIDIA packages (**0 found**) [MEASURED, RUN1-VENV.md:12].
- **Precompile & Package**: Run `python -m compileall -q` on site-packages to eliminate runtime JIT bytecode lag (took **47.19 s** [MEASURED, RUN1-VENV.md:14]).
- **Archive with zstd**: `tar -I "zstd -3 -T0" -cf venv312_cpu.tar.zst -C /dev/shm vllm_env` packs **5,132.45 MB** into **1,387.55 MB** in **10.36 s** [MEASURED, RUN1-VENV.md:15, 17].
- **Download vs Install**: Replaces the **91.21 s** install [MEASURED, STARTUP-BREAKDOWN.md:14] with **13.50 s download** + **6.35 s unpack** [MEASURED, RUN2-COLDSTART.md:11].

---

## 3. Mandatory `/dev/shm` Exec Remount

Google Colab mounts tmpfs `/dev/shm` with `noexec` by default (`rw,nosuid,nodev,noexec,...`) [MEASURED, RUN1-VENV.md:8], causing dynamic linker `dlopen` (`libtorch_global_deps.so`) to fail with `OSError: failed to map segment from shared object` [MEASURED, RUN1-VENV.md:44].
- **Fix**: Run `mount -o remount,exec /dev/shm` before running or unpacking any venv into `/dev/shm` (took **0.08 s**, `rc=0`) [MEASURED, RUN2-COLDSTART.md:9].

---

## 4. Parallel Weights and Venv Download

Download model weights and prebuilt venv simultaneously to `/dev/shm` using Hugging Face (never use Google Drive; Drive mount is interactive and slow [MEASURED, RECIPE.md:71-73]):
- Running `hf_hub_download` (venv) and `snapshot_download` (weights) in parallel took **19.85 s total** [MEASURED, RUN2-COLDSTART.md:13] (venv dl 13.50 s, unpack 6.35 s; weights dl 11.89 s).

---

## 5. Bypass Metadata 404 Probes with Static Topology

During TPU platform init, `cloud_tpu_init.py` attempts HTTP polling on `http://metadata.google.internal/...`, hanging on repeated 404 timeouts [MEASURED, STARTUP-BREAKDOWN.md:32].
- Set static environment variables: `TPU_SKIP_MDS_QUERY=1 TPU_TOPOLOGY=1x1 TPU_HOST_BOUNDS=1,1,1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost JAX_PLATFORMS=tpu,cpu`.
- **Impact**: Reduced `import vllm, jax` verification check from **15.56 s down to 5.91 s** (**-9.65 s / -62%**) [MEASURED, RUN2-COLDSTART.md:21-23].
- **Rule**: Always run `python -c "import vllm, jax; print(jax.devices())"` after setting them to verify `[TpuDevice(id=0)]` appears.

---

## 6. Single-Process In-Place Engine Execution

Running multi-process `spawn` duplicates Python initialization, imports, and metadata probing (**11.00 s** wasted [MEASURED, STARTUP-BREAKDOWN.md:18, 34]).
- Set `VLLM_ENABLE_V1_MULTIPROCESSING=0` before initializing `LLM(...)` to run EngineCore in-process [MEASURED, RUN2-COLDSTART.md:15]. Eliminates child process launch overhead.

---

## 7. Persistent Compilation Cache Lifecycle

JAX cache keys are cryptographic hashes of lowered StableHLO MLIR modules, depending on `(max_num_seqs, max_model_len, max_num_batched_tokens, software versions)` [UNTESTED deduction from JAX lowering].
- Point `VLLM_XLA_CACHE_PATH` and `JAX_COMPILATION_CACHE_DIR` to `/dev/shm/tpu_compile_cache`.
- On cold cache miss at 32K context, JAX compiled **104 programs in 194.16 s** [MEASURED, RUN2-COLDSTART.md:15].
- Cache Hit Behavior: Even with 100% cache hits, Python tracing (**15.01 s**) and MLIR lowering (**32.70 s**) still run; cache only skips XLA binary compilation (**3.38 s** hit deserialization) [MEASURED, STARTUP-BREAKDOWN.md:46-48].
- Deposit cache as a `.tar` to private HF repo once; restore on fresh VMs.

---

## 8. Decode Speed Traps to Avoid

1. **Never use `SKIP_JAX_PRECOMPILE=1` or restrict `compilation_sizes`**: Uncompiled shapes JIT during decode, collapsing speed from **88.58 tok/s** down to **8–30 tok/s** [MEASURED, RECIPE.md:118; RUN2-COLDSTART.md:29].
2. **No per-token file I/O hooks**: Synchronous disk logging inside forward passes throttles TPU dispatch [MEASURED, RECIPE.md:120].
3. **Always run warmup before timing**: Send 1 untimed request to populate PJRT device buffers [MEASURED, RECIPE.md:119; RUN2-COLDSTART.md:28].

---

## 9. Memory Budget & Context Headroom

- Target `gpu_memory_utilization=0.90` (allocates **14.17 GiB** max [MEASURED, RECIPE.md:9]).
- Formula: $\text{Weights} + \text{KV Cache} \le 13.8\text{ GiB}$ [RULE OF THUMB / UNTESTED formula, RECIPE.md:11].
- Context scaling: At 26K context (`max_model_len=32768`), in-use HBM reached **14.274 GiB** (90.6%) and peak reached **14.376 GiB** (91.3%), maintaining **1.37 GiB safe headroom** below the 15.748 GiB limit [MEASURED, RUN2-COLDSTART.md:34-36]. Overhead grows with context length.

---

## 10. Step-by-Step Checklist for Any New Model

1. **Venv**: Download and unpack prebuilt `venv312_cpu.tar.zst` to `/dev/shm` after `mount -o remount,exec /dev/shm`.
2. **Weights**: Pull sharded weights to `/dev/shm/<model>` via `snapshot_download` in parallel with venv.
3. **Settings & TPU Check**: Export `TPU_SKIP_MDS_QUERY=1` topology vars and `VLLM_ENABLE_V1_MULTIPROCESSING=0`. Always verify with `python -c "import vllm, jax; print(jax.devices())"`.
4. **Cache Warm-up (Once)**: Run one compile session with `max_num_seqs` matching expected concurrency; pack `/dev/shm/tpu_compile_cache` to tar and upload to HF.
5. **Cold Serving**: On future fresh VMs, extract compile cache to `/dev/shm/tpu_compile_cache` before starting engine.
6. **Warmup**: Always send 1 untimed dummy request before benchmarking or serving user queries.

---

## 11. Untested Assumptions & Open Questions

1. **Warm 32K Cache Startup Time**: Exact startup duration restoring `tpu_compile_cache_b1_32k.tar` is [UNTESTED] (142 s startup expected to drop to ~48 s based on [MEASURED, RECIPE.md:100], but unmeasured for 32K).
2. **30K Prefill Latency**: Measured 0.09 s for 25.7K tokens [MEASURED, RUN2-COLDSTART.md:31] is [UNTESTED / SUSPICIOUS], likely an internal timing artifact from KV-cache re-use or asynchronous dispatch.
3. **26K Decode Speed**: Decode throughput dropping to 10.20 tok/s at 26K context [MEASURED, RUN2-COLDSTART.md:32] vs 88.58 tok/s at batch 1 short context is cause unknown [UNTESTED]. Earlier runs measured ~70 tok/s at 16K context [MEASURED, RECIPE.md:111], making 10 tok/s at 26K look especially suspicious.
4. **Colab Image Portability**: Whether `mount -o remount,exec /dev/shm` and static topology vars are universally supported across all Google Colab backend VM image variants is [UNTESTED].
