# vLLM XLA/Runtime Memory Overhead: google/gemma-4-E4B-it-qat-w4a16-ct on TPU v5e-1

Benchmark measurement of vLLM XLA and runtime memory overhead on Google Colab TPU v5e-1 (`tpu-v5e1-s-r-usw4a0-1t0wemi5c7h2h` / `run-4c5fd8`).

## Environment & Versions

- **Host Python:** 3.12.3 venv (`/content/venv312`) via `uv`
- **Package Versions [measured]:** `vllm-tpu`: `0.32.0`, `tpu-inference`: `0.32.0`, `jax`: `0.11.2`, `jaxlib`: `0.11.2`, `libtpu`: `0.0.48`, `torch`: `2.10.0`
- **Hardware:** 1x Google TPU v5 lite (16 GB HBM; driver reports 15.75 GiB usable) [measured]

## Startup Phase Times (Run 3: Full Multimodal, Spawn)

| Phase | Duration (s) | Status | Log Source / Detail |
| :--- | ---: | :--- | :--- |
| **Install** | 91.77 s | [measured] | `INSTALL_SECONDS: 91.77` (`uv pip install vllm-tpu==0.32.0`) |
| **Download** | 12.72 s | [measured] | `DOWNLOAD_INFO: 12.72s` (10.72 GiB safetensors to `/dev/shm`) |
| **Weight Load** | 151.32 s | [measured] | `Loading weights took 151.32 seconds` (tmpfs to device) |
| **Model Init / JIT** | 5.01 s | [measured] | 11:34:28 to 11:34:33: failed in `create_jit_model` |
| **KV Allocation** | Not reached | [not reached] | Blocked by Flax NNX JIT crash |
| **Precompile** | Not reached | [not reached] | 0 programs compiled (blocked before `capture_model`) |
| **Engine Ready Total** | Not reached | [not reached] | Failed at 156.33 s of engine initialization |

## Memory Measurement Table

| Metric | Value | Status / Derivation |
| :--- | ---: | :--- |
| **Total HBM Usable** | 15.75 GiB | [measured] `hbm=[(0.0, 15.75)]GiB` (driver report) |
| **Weights on Disk** | 10.72 GiB | [measured] `11,513,861,828` bytes (`model.safetensors`) |
| **Weights on Device** | ~10.15 GiB | [inferred] 9.83 GiB text + 0.315 GiB vision (header calculation) |
| **KV Cache** | Not reached | [not reached] Blocked before block allocation |
| **XLA / Runtime Overhead** | Not reached | [not reached] Blocked before M3 engine-ready |

## One-Line Verdict

**Verdict:** Full multimodal execution disproves the `limit_mm_per_prompt` hypothesis: `model.states[0][594]` (`bfloat16[256]`) remains an unmaterialized `ShapeDtypeStruct` inside Flax NNX `create_jit_model`, proving an intrinsic bug in `tpu-inference==0.32.0` for `Gemma4ForConditionalGeneration`.

## Compute Unit (CU) & Wall Time Accounting

- **Run 1 (Deadlock):** 1.54 CU (396.66 -> 395.12 CU), wall time: 32m 06s [measured]
- **Run 2 (Retry 1, text-only):** 0.29 CU (394.96 -> 394.67 CU), wall time: 6m 35s [measured]
- **Run 3 (Full Multimodal, Spawn):** 0.38 CU (394.67 -> 394.29 CU), wall time: 6m 45s [measured]
- **Total CU Consumed:** 2.21 CU [measured]; Current balance: 394.29 CU; Active sessions: 0 [verified]

## Raw vLLM Log Lines (Run 3)

```text
(EngineCore pid=16500) INFO 10-29 11:31:53 [tpu_worker.py:472] Init worker | rank=0 | is_first_rank=True | is_last_rank=True | topology_order_id=0 | is_driver_worker=True | hbm=[(0.0, 15.75)]GiB |self.devices=[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)] | total devices=[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)] | local_devices=[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
(EngineCore pid=16500) INFO 10-29 11:31:56 [weight_utils.py:1112] Filesystem type for checkpoints: TMPFS. Checkpoint size: 10.72 GiB. Available RAM: 25.94 GiB.
Loading safetensors checkpoint shards: 100% Completed | 1/1 [02:21<00:00, 141.27s/it]
(EngineCore pid=16500) INFO 10-29 11:34:28 [default_loader.py:501] Loading weights took 151.32 seconds
(EngineCore pid=16500) ERROR 10-29 11:34:33 [core.py:1510] jax._src.dtypes.InvalidInputException: Argument of type <class 'jax.ShapeDtypeStruct'> is not a valid JAX type.
(EngineCore pid=16500) ERROR 10-29 11:34:33 [core.py:1510] TypeError: Argument 'model.states[0][594]' of shape bfloat16[256] of type <class 'jax.ShapeDtypeStruct'> is not a valid JAX type.
```

## Failure Analysis Across Runs

### Run 1: Accelerator Fork Deadlock
Querying `jax.devices()[0].memory_stats()` at M0 in the main worker process initialized libtpu and background PJRT threads before vLLM forked, causing POSIX `fork()` deadlock inside `pxla.backend_compile_and_load`.

### Run 2: Text-Only with `limit_mm_per_prompt`
Ran with `VLLM_ENABLE_V1_MULTIPROCESSING=0` and `limit_mm_per_prompt={"image": 0, "audio": 0}`. Weights loaded in 159.56 s, but crashed in `create_jit_model` with `Argument 'model.states[0][594]' of shape bfloat16[256] of type <class 'jax.ShapeDtypeStruct'> is not a valid JAX type`.

### Run 3: Full Multimodal (`spawn`, no `limit_mm_per_prompt`)
Tested whether `limit_mm_per_prompt` caused unmaterialized vision states. Ran with full multimodal model, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, no limits, and `VLLM_ENABLE_V1_MULTIPROCESSING=1`. Weights loaded in 151.32 s. Multi-modal warmup completed in 28.79 s. During `_get_nnx_model` -> `create_jit_model`, the identical error occurred: `Argument 'model.states[0][594]' of shape bfloat16[256] of type <class 'jax.ShapeDtypeStruct'> is not a valid JAX type`. Per failure rules, recorded and stopped (no retry, no patch).
