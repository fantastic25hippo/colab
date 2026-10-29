# vLLM XLA/Runtime Memory Overhead: google/gemma-4-E4B-it-qat-w4a16-ct on TPU v5e-1

Benchmark measurement of vLLM XLA and runtime memory overhead on Google Colab TPU v5e-1 (`tpu-v5e1-s-r-usw4a0-1t0wemi5c7h2h`).

## Environment & Versions

- **Host Python:** 3.12.3 venv (`/content/venv312`) via `uv`
- **Package Versions:**
  - `vllm-tpu`: `0.32.0` [measured]
  - `tpu-inference`: `0.32.0` [measured]
  - `jax`: `0.11.2` [measured]
  - `jaxlib`: `0.11.2` [measured]
  - `libtpu`: `0.0.48` [measured]
  - `torch`: `2.10.0` [measured]
- **Hardware:** 1x Google TPU v5 lite (16 GB HBM) [measured]

## Weights on Disk

- **Target:** `google/gemma-4-E4B-it-qat-w4a16-ct` downloaded to `/dev/shm/gemma` with `max_workers=12`, `HF_HUB_ENABLE_HF_TRANSFER=1`, `xet` enabled.
- **Per-file sizes [measured]:**
  - `model.safetensors`: `11,513,861,828` bytes (10.72 GiB)
  - `tokenizer.json`: `32,505,856` bytes (31.00 MiB)
  - `README.md`: `29,268` bytes
  - `config.json`: `22,528` bytes
  - `chat_template.jinja`: `19,456` bytes
  - `tokenizer_config.json`: `3,788` bytes
  - `processor_config.json`: `1,740` bytes
  - `generation_config.json`: `208` bytes
- **Total safetensors bytes on disk:** `11,513,861,828` bytes (10.72 GiB) [measured]

## Memory Measurement Table (M0–M4)

| Checkpoint | Stage | bytes_in_use | peak_bytes_in_use | bytes_limit | Status / Note |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **M0** | Baseline before engine start | 26,624 B (~0.000025 GiB) | 26,624 B (~0.000025 GiB) | 16,000,000,000 B (~14.90 GiB) | [measured] Device initialized |
| **M1** | Model weights on device | Not reached | Not reached | ~14.90 GiB | [inferred] ≥ 10.72 GiB (blocked by fork deadlock) |
| **M2** | KV cache allocation | Not reached | Not reached | ~14.90 GiB | [not reached] |
| **M3** | Engine ready (post-compile) | Not reached | Not reached | ~14.90 GiB | [not reached] |
| **M4** | Post 128-token generation | Not reached | Not reached | ~14.90 GiB | [not reached] |

## Metrics & Overhead Calculation

- **weights_on_device vs safetensors_on_disk:** `Not reached` (deadlock before weight profiling; safetensors on disk = 10.72 GiB).
- **KV cache GiB:** `Not reached`.
- **XLA / runtime overhead:** `Not reached` (see failure analysis below).
- **Compile seconds:** Stalled at >1,500 s due to accelerator fork deadlock before manual stop.
- **Verdict:** Stalled in vLLM V1 EngineCore initialization due to `fork()` deadlocking PJRT/libtpu after M0 memory probe.

## Compute Unit (CU) & Wall Time Accounting

- **CU Before:** 396.66 CU
- **CU After:** 395.12 CU
- **CU Used:** 1.54 CU [measured]
- **Wall Time:** 32m 06s [measured]

## Failure Analysis & Raw Stack Trace

### Deadlock Mechanism
1. Querying `jax.devices()[0].memory_stats()` at M0 in the main worker process (`PID 2310`) initialized libtpu and spawned background PJRT runtime threads (`pjrt-tpu-tasks`, `tpu-non-blocking-queue`).
2. vLLM V1's `launch_core_engines` executed `multiprocessing/popen_fork.py` to create `PID 3290` (`VLLM::EngineCore`).
3. POSIX `fork()` does not clone background threads into the child process. When child `PID 3290` reached `TpuRunner._init_random` (`jax.random.key`), it called into PJRT XLA compilation (`backend_compile_and_load`), where it deadlocked waiting on a mutex held by an un-cloned thread.

### Raw `py-spy` Thread Trace from Live VM (`PID 3290`):
```text
Process 3290: VLLM::EngineCore
Thread 3290 (idle): "MainThread"
    backend_compile_and_load (jax/_src/compiler.py:395)
    wrapper (jax/_src/profiler.py:420)
    _compile_and_write_cache (jax/_src/compiler.py:774)
    compile_or_get_cached (jax/_src/compiler.py:509)
    _cached_compilation (jax/_src/interpreters/pxla.py:1534)
    from_hlo (jax/_src/interpreters/pxla.py:1757)
    compile (jax/_src/interpreters/pxla.py:1257)
    _pjit_call_impl_python (jax/_src/pjit.py:1151)
    _run_python_pjit (jax/_src/pjit.py:142)
    cache_miss (jax/_src/pjit.py:266)
    _key (jax/_src/random/core.py:229)
    key (jax/_src/random/core.py:257)
    _init_random (tpu_inference/runner/tpu_runner.py:874)
    __init__ (tpu_inference/runner/tpu_runner.py:818)
    init_device (tpu_inference/worker/tpu_worker.py:471)
    _init_executor (vllm/v1/executor/uniproc_executor.py:69)
    run_engine_core (vllm/v1/engine/core.py:1482)
    _launch (multiprocessing/popen_fork.py:71)
```

### Raw Driver Log Line (`/tmp/tpu_logs/tpu_driver.*.2310`):
```text
I1029 09:27:22.134899 2310 device_util.cc:89] Found 1 TPU v5 lite chips.
I1029 09:27:22.134906 2310 tpu_version_flag.cc:77] Using auto-detected TPU version TPU v5 lite
W1029 09:27:22.135625 2310 flags_util.cc:127] Applying legacy hack to override mesh topology to 1x1.
```
