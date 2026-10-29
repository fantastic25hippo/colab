# Startup & Memory Investigation: vLLM TPU on Colab TPU v5e-1

Local code investigation of memory allocation, startup compilation, and installation speed for `google/gemma-4-E4B-it-qat-w4a16-ct` on TPU v5e-1 (`vllm-tpu==0.32.0`, `tpu-inference==0.32.0`, `jax==0.11.2`, `torch==2.10.0`). Zero CU consumed.

---

## Q1: Do W4A16 Weights Stay 4-Bit on TPU HBM?

- **Device Dtype & Shape [proven in code]:** `u32_unpack_i4` (`tpu_inference/layers/common/quantization/__init__.py:86-98`) unpacks 8 nibbles per int32 word, centers them via `- 8`, casts `.astype(jnp.int4)`, and reshapes to `[out, in]`. `unpack_wna16_linear_weight` (`tpu_inference/layers/common/quantization/__init__.py:129-130`) slices logical features and transposes to `[in, out]` with dtype `jnp.int4`.
- **No Repack Step [proven in code]:** `process_weights_after_loading` (`tpu_inference/layers/jax/quantization/wna16.py:183-218`) puts `weight` directly into `LinearWeights`. `shard_linear_weights` (`tpu_inference/layers/common/process_weights/linear_weights.py:218`) calls `general_device_put(weight, sharding)` -> `jax.device_put(weight, sharding)` without any repacking or host dtype expansion.
- **Decode Matmul Kernel [proven in code]:** In `tpu_inference/layers/jax/quantization/wna16.py:73-82`, `_uses_kernel` evaluates `group_size < _mxu_column_size()`. For TPU v5e (128x128 MXU), `32 < 128` is True, setting `self.use_kernel = True`. Scales are reshaped to 3D `[in // group_size, 1, out]` (`wna16.py:203`), which routes `sharded_quantized_matmul` (`tpu_inference/layers/common/linear.py:226, 254`) directly to `gmm_v2` (`tokamax._src.ops.experimental.gmm_v2.gmm_v2`). As documented in `wna16.py:26-28`, `gmm_v2` streams `int4` weights from HBM and dynamically dequantizes tiles inside on-chip VMEM. Unquantized BF16 weights **never exist in HBM**.
- **On-Device Weight Arithmetic (Header Analysis) [proven from safetensors header]:**
  - Checkpoint contains 2,763 tensors; 1,352 belong to the text decoder (343 quantized linear layers, 281 unquantized weights, 42 scalars).
  - Unquantized text params: **4,161,309,738 params = 7.751 GiB** (8.323 GB). Dominated by `embed_tokens_per_layer` (`[262144, 10752]` = 5.250 GiB BF16), `lm_head` (`[262144, 2560]` = 1.250 GiB BF16), and `embed_tokens` (`[262144, 2560]` = 1.250 GiB BF16).
  - Quantized linear params: **3,972,792,320 params**. Quantization scales (group 32): 124,149,760 params = 0.231 GiB BF16.
  - Arithmetic by format:
    - **INT4 (0.5 B/param):** $3,972,792,320 \times 0.5\text{ B} = 1.850\text{ GiB} + 0.231\text{ GiB scale} + 7.751\text{ GiB unquant} = \mathbf{9.832\text{ GiB}}$ ($10.557\text{ GB}$).
    - **INT8 (1.0 B/param):** $3,972,792,320 \times 1.0\text{ B} = 3.700\text{ GiB} + 0.231\text{ GiB scale} + 7.751\text{ GiB unquant} = \mathbf{11.682\text{ GiB}}$ ($12.544\text{ GB}$).
    - **BF16 (2.0 B/param):** $3,972,792,320 \times 2.0\text{ B} = 7.400\text{ GiB} + 7.751\text{ GiB unquant} = \mathbf{15.151\text{ GiB}}$ ($16.268\text{ GB}$).
- **Verdict [inferred]:** Expected weights on device: **9.83 GiB** (~10.15 GiB if vision tower's 0.315 GiB is retained). Leaves **5.92 GiB (37.6%)** of 15.75 GiB driver HBM free (or 5.07 GiB / 34.0% of the 14.90 GiB bytes_limit). If unpacked to BF16, 15.15 GiB text weights alone would exceed the 14.90 GiB limit and immediately OOM. W4A16 stays in INT4 on HBM.

---

## Q2: What Does vLLM TPU Compile at Startup?

- **Precompile / Warmup Sequence [proven in code]:** Orchestrated by `CompilationManager.capture_model` (`tpu_inference/runner/compilation_manager.py:263`).
- **Shape Buckets Generated [proven in code]:**
  - `num_tokens_paddings`: `build_token_paddings` (`tpu_inference/runner/utils.py:195, 229`) with `min=16`, `max=512`, `gap=0` generates 6 buckets: `[16, 32, 64, 128, 256, 512]`.
  - `attn_num_reqs_paddings`: `get_attn_req_paddings` (`runner/utils.py:175`) with `max_num_seqs=1` and `ATTN_BUCKETIZED_NUM_REQS=False` generates 1 bucket: `[1]`.
  - `num_reqs_paddings`: `tpu_runner.py:1078` sets `max_num_reqs = max(1*1, MIN_NUM_SEQS=8) = 8`. `get_req_paddings` generates 1 bucket: `[8]`.
  - Multimodal encoder warmup: **0 programs**. `tpu_runner.py:1315, 1324` checks `limit_mm_per_prompt={'image': 0, 'audio': 0}`, sets `disable_mm_from_limits=True`, and forces `is_multimodal_model=False`, skipping all vision encoder/merger compiles.
- **Compiled Program Count [proven in code]:**
  1. `worker0 backbone` (`compilation_manager.py:714`): 6 tokens $\times$ 1 req = **6 programs**.
  2. `select all logits` (`compilation_manager.py:921`): 6 array_sizes $\times$ 1 indices_count = **6 programs**.
  3. `compute_logits` (`compilation_manager.py:956`): 1 req bucket (shape 8) = **1 program**.
  4. `sample` (`compilation_manager.py:996`): 1 req $\times$ 2 do_sampling $\times$ 2 logprobs = **4 programs**.
  5. `gather_logprobs` (`compilation_manager.py:1097`): 1 req bucket = **1 program**.
  6. `structured_decode` (`compilation_manager.py:1946`): 1 req bucket = **1 program**.
  - Total startup programs: **19 AOT programs** (plus 1 model initialization JIT = ~20 total). Halving `max_num_batched_tokens` to 256 drops bucket 512, saving 2 programs (17 total).
- **Skip / Shrink Options & Cache Key [proven in code]:**
  - `SKIP_JAX_PRECOMPILE=1`: Immediately returns from `capture_model` (`compilation_manager.py:264`), enabling zero-warmup startup with on-demand lazy compilation.
  - `compilation_sizes=[16, 512]`: Overrides automatic bucketing; cuts backbone compiles from 6 to 2.
  - Cache location: Configured in `compilation_manager.py:90` via `jax.config.update("jax_compilation_cache_dir", VLLM_XLA_CACHE_PATH)`.
  - Cache key: Computed by JAX/PJRT hashing (1) JAX/jaxlib/libtpu git commits, (2) HLO graph string (shapes, dtypes, sharding), (3) compiler optimization flags, and (4) target hardware architecture (TPU v5e).
- **Verdict [inferred]:** Exactly **19 AOT programs** compiled. Optimal startup settings for single-user decode: set `compilation_sizes=[16, 512]` (cuts to 11 programs) or set `SKIP_JAX_PRECOMPILE=1` (0 startup compiles, JIT compiles 16-token decode bucket on first token).

---

## Q3: Can Install Time Be Cut with a Prebuilt Environment?

- **Environment Comparison [proven]:**
  - **Laptop:** Ubuntu 24.04.1 LTS, kernel 7.0, glibc 2.39, x86_64, Python 3.12.3.
  - **Colab VM:** Ubuntu 22.04 LTS, kernel 6.6, glibc 2.35, x86_64, 24 vCPUs.
  - **Forward Incompatibility:** A venv built on the laptop using Ubuntu 24.04's system Python or toolchain will fail on Colab with `GLIBC_2.38/GLIBC_2.39 not found` because glibc is backward compatible but **never forward compatible**.
- **Relocatable Venv Procedure (if using standalone CPython or Ubuntu 22.04 container) [inferred]:**
  - Build via uv with standalone Python: `uv venv --python 3.12.3 /content/venv312` (uses Astral's glibc 2.17+ build).
  - Install dependencies into `/content/venv312`.
  - Pack: `tar --zstd -cf vllm-tpu-py312.tar.zst -C /content venv312` (expected size: ~650–800 MiB compressed, ~2.2 GiB extracted).
  - Upload to private HF repo `avgrocketman/tpu-compile-cache` under `env/vllm-tpu-0.32.0-py312/venv.tar.zst`.
  - Restore on VM: `huggingface_hub` snapshot to `/content`, extract in ~4 s. Total time: ~7 s.
- **Recommended Offline Wheel Cache Alternative [proven approach / inferred timing]:**
  - Upload an archive of downloaded pre-built wheels (`vllm-tpu`, `tpu-inference`, `jax`, `jaxlib`, `libtpu`, `torch`) to HF (~480 MB).
  - On VM: Download archive via `hf_transfer` (~2.5 s at 222 MB/s), extract to `/dev/shm/wheels` (~0.5 s), run `uv pip install --no-index --find-links /dev/shm/wheels` (~2.5 s).
  - Total install time: **~5–6 seconds** (vs 45–60 s for online PyPI pip install, an ~88% reduction), with zero glibc mismatch risk.

---

## Ranked Next TPU Tests

| Rank | Test Description | Objective | Expected Time | Expected CU (~2.9 CU/hr) |
| :--- | :--- | :--- | ---: | ---: |
| **1** | **Flax NNX Bug Patch & M1–M4 Measurement** | Patch `ShapeDtypeStruct` at `tpu_inference/models/common/model_loader.py:305` to verify engine reaches ready and confirm 9.83 GiB HBM weight footprint. | 8 min | ~0.39 CU |
| **2** | **Weight Load Latency Test (`--apply-fix`)** | Run `scripts/tpu_load_time_bench.py` on TPU v5e to measure the predicted 50% load time cut from filtering 1,411 non-text tensors and removing `jax.clear_caches`. | 5 min | ~0.24 CU |
| **3** | **`SKIP_JAX_PRECOMPILE=1` Startup vs Decode Latency** | Measure startup speedup (0 vs 19 programs) and verify first-token latency impact during 128-token generation. | 6 min | ~0.29 CU |
