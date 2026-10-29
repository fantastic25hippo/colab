# vLLM XLA/Runtime Memory Overhead: google/gemma-4-E4B-it-qat-w4a16-ct on TPU v5e-1

Benchmark measurement of vLLM XLA and runtime memory overhead on Google Colab TPU v5e-1 (`tpu-v5e1-s-r-usw4a0-1t0wemi5c7h2h` / `run-da79c7`).

## Environment & Versions

- **Host Python:** 3.12.3 venv (`/content/venv312`) via `uv`
- **Package Versions [measured]:**
  - `vllm-tpu`: `0.32.0`, `tpu-inference`: `0.32.0`, `jax`: `0.11.2`, `jaxlib`: `0.11.2`, `libtpu`: `0.0.48`, `torch`: `2.10.0`
- **Hardware:** 1x Google TPU v5 lite (16 GB HBM; driver reports 15.75 GiB usable) [measured]

## Weights on Disk

- **Target:** `google/gemma-4-E4B-it-qat-w4a16-ct` downloaded to `/dev/shm/gemma` with `max_workers=12` in 11.92 s [measured].
- **Per-file sizes [measured]:** `model.safetensors`: `11,513,861,828` bytes (10.72 GiB), `tokenizer.json`: `32,169,626` bytes (30.68 MiB).
- **Total safetensors bytes on disk:** `11,513,861,828` bytes (10.72 GiB) [measured]

## Memory Measurement Table (M0–M4)

| Checkpoint | Stage | bytes_in_use | peak_bytes_in_use | bytes_limit | Status / Raw Note |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **M0** | Baseline before engine start | 26,624 B (~0.000025 GiB) | 26,624 B (~0.000025 GiB) | 16,000,000,000 B (~14.90 GiB) | [measured] Device initialized in Run 1 |
| **M1** | Model weights on device | 10.72 GiB checkpoint | Not probed | ~14.90 GiB | [measured] `Loading weights took 159.56 seconds` |
| **M2** | KV cache allocation | Not reached | Not reached | ~14.90 GiB | [not reached] Blocked by Flax NNX JIT crash |
| **M3** | Engine ready (post-compile) | Not reached | Not reached | ~14.90 GiB | [not reached] Blocked by Flax NNX JIT crash |
| **M4** | Post 128-token generation | Not reached | Not reached | ~14.90 GiB | [not reached] |

## Metrics & Overhead Calculation

- **weights_on_device vs safetensors_on_disk:** 10.72 GiB safetensors loaded in 159.56 s; packed ratio unconfirmed (JAX query blocked).
- **KV cache GiB:** `Not reached` (crashed during Flax NNX compilation before KV block allocation).
- **XLA / runtime overhead:** `Not reached` (see failure analysis below).
- **Compile seconds:** Crashed during Flax NNX JIT compilation after 159.56 s weight load.
- **Cache Upload:** Skipped (engine did not reach ready; no complete compilation cache generated).
- **Verdict:** Blocked by upstream `tpu-inference` / `flax.nnx` bug: unmaterialized `jax.ShapeDtypeStruct` passed into JAX pjit.

## Compute Unit (CU) & Wall Time Accounting

- **Run 1 (Deadlock):** 1.54 CU (396.66 -> 395.12 CU), wall time: 32m 06s [measured]
- **Run 2 (Retry):** 0.29 CU (394.96 -> 394.67 CU), wall time: 6m 35s [measured]
- **Total CU Consumed:** 1.83 CU [measured]; Current balance: 394.67 CU; Active sessions: 0 [verified]

## Raw vLLM Log Lines

```text
(MainProcess pid=1563) INFO 10-29 10:23:18 [model.py:733] Resolved architecture: Gemma4ForConditionalGeneration
(MainProcess pid=1563) INFO 10-29 10:23:59 [tpu_worker.py:472] Init worker | rank=0 | is_first_rank=True | is_last_rank=True | topology_order_id=0 | is_driver_worker=True | hbm=[(0.0, 15.75)]GiB |self.devices=[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]
(MainProcess pid=1563) INFO 10-29 10:24:02 [weight_utils.py:1112] Filesystem type for checkpoints: TMPFS. Checkpoint size: 10.72 GiB. Available RAM: 27.77 GiB.
Loading safetensors checkpoint shards: 100% Completed | 1/1 [02:28<00:00, 148.80s/it]
(MainProcess pid=1563) INFO 10-29 10:26:41 [default_loader.py:501] Loading weights took 159.56 seconds
```

## Failure Analysis

### Run 1: Accelerator Fork Deadlock
1. Querying `jax.devices()[0].memory_stats()` at M0 in the main worker process initialized libtpu and background PJRT threads.
2. vLLM V1 executed `multiprocessing/popen_fork.py` to spawn `VLLM::EngineCore`. POSIX `fork()` did not clone background threads, causing child deadlock at `TpuRunner._init_random` inside `pxla.backend_compile_and_load`.

### Run 2 (Retry): Flax NNX ShapeDtypeStruct Bug
In Run 2 with `VLLM_ENABLE_V1_MULTIPROCESSING=0` and no M0 probe, fork deadlock was eliminated. Weights loaded fully (159.56 s). During `_get_nnx_model` -> `create_jit_model`, Flax NNX attempted to JIT-compile the model, but state tensor `model.states[0][594]` remained an abstract `jax.ShapeDtypeStruct` rather than a materialized device array:
```text
  File ".../tpu_inference/models/common/model_loader.py", line 305, in _get_nnx_model
    jit_model = create_jit_model(...)
  File ".../flax/nnx/transforms/compilation.py", line 1023, in __call__
    pure_args_out, pure_kwargs_out, pure_out = self.jitted_fn(...)
  File ".../jax/_src/interpreters/pxla.py", line 114, in shard_args
    raise dtypes.InvalidInputException(
jax._src.dtypes.InvalidInputException: Argument of type <class 'jax.ShapeDtypeStruct'> is not a valid JAX type.
TypeError: Argument 'model.states[0][594]' of shape bfloat16[256] of type <class 'jax.ShapeDtypeStruct'> is not a valid JAX type.
```
