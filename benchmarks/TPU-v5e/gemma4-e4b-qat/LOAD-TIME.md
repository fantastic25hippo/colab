# Investigation: vLLM TPU Weight Loading Latency (Gemma 4 E4B QAT)

Investigation of the 159.56 s load time for `google/gemma-4-E4B-it-qat-w4a16-ct` (11.5 GB in `/dev/shm`, tmpfs) on TPU v5e-1 (`vllm-tpu==0.32.0`, `tpu-inference==0.32.0`). Read-only code analysis.

## 1. Trace of the Tensor Load Path (file:line)

Each of the 2,763 checkpoint tensors traverses seven sequential stages:

1. **Loader Orchestration:** `tpu_inference/models/common/model_loader.py:285` calls `loader.load_weights(model, model_config)`. In `vllm/model_executor/model_loader/default_loader.py:501`, `DefaultModelLoader` invokes `model.load_weights(self.get_all_weights(...))`.
2. **Serial Safetensors Disk I/O:** `vllm/model_executor/model_loader/weight_utils.py:1218` opens `model.safetensors` via `safe_open(..., framework="pt")` and serially executes `param = f.get_tensor(name)` for every key one by one.
3. **Prefix Mapping & Routing:** `tpu_inference/models/jax/gemma4_mm.py:734` wraps weights in `WeightsMapper` and delegates to `JaxAutoWeightsLoader.load_weights` (`weight_utils.py:996`). Module tree recursion in `vllm/model_executor/models/utils.py:408` groups keys via `itertools.groupby`.
4. **Host JAX Param Staging (Torch -> JAX):** W4A16 linear layers register host parameters on `cpu_mesh()` (`tpu_inference/layers/jax/quantization/wna16.py:136`). For each tensor, `load_nnx_param_from_reshaped_torch` (`weight_utils.py:828`) calls `t2j` (`tpu_inference/utils.py:143-160`), converting `torch.Tensor` to CPU JAX arrays via memory copies.
5. **Dequant / Unpack & Transpose (CPU Host):** Once `weight_packed`, `weight_scale`, and `weight_shape` arrive, `WNA16LinearMethod.process_weights_after_loading` (`wna16.py:188`) calls `unpack_wna16_linear_weight` (`tpu_inference/layers/common/quantization/__init__.py:126`). `u32_unpack_i4` (`__init__.py:86-98`) runs unjitted eager bitwise shifts (`jnp.bitwise_right_shift`, `bitwise_and 0xF`, minus 8, cast to `int4`), expanding words by 8x. Weights and scales are transposed (`weight.T`, `weight_scale.T`, line 127).
6. **Sharding & Device Put:** `LinearWeights` are sharded via `shard_linear_weights` (`tpu_inference/layers/common/process_weights/linear_weights.py:221`), which calls `general_device_put` -> `jax.device_put(t, sharding)` (`tpu_inference/layers/common/utils.py:145`), initiating synchronous/asynchronous PJRT transfers to TPU HBM per tensor.
7. **Redundant Cache Clearing:** In `JaxAutoWeightsLoader._load_module` (`weight_utils.py:1132`), `jax.clear_caches()` is explicitly called after every single linear module, repeatedly flushing JAX compilation and dispatch caches.

## 2. Architectural Bottlenecks & Answers

- **Serial per-tensor processing [proven in code]:** `safetensors_weights_iterator` (`weight_utils.py:1218`) yields tensors in a single-threaded loop. Each tensor incurs Python generator, dict lookup, and JAX overhead (2,763 tensors / 159.56 s = ~57.7 ms/tensor).
- **Redundant CPU work [proven in code]:** CPU unpacks int32 -> int4 nibbles, transposes `[out, in] -> [in, out]`, and converts Torch to JAX during startup. Crucially, **1,411 out of 2,763 tensors (51.1%) are multimodal** (752 audio, 659 vision). Audio tensors are read from disk and dropped in Python; all 659 vision tensors are fully unpacked and put on TPU device even when `limit_mm_per_prompt image=0 audio=0` is set!
- **One-at-a-time device_put [proven in code]:** `general_device_put` (`utils.py:145`) puts weights and scales on TPU per linear layer rather than batching transfers across layers.
- **Existing loader options [proven in code]:**
  - `--safetensors-load-strategy eager`: Reads entire file into RAM, but tensors are still unpacked serially; no change to CPU unpack or PJRT overhead.
  - `--safetensors-load-strategy prefetch`: Prefetches disk blocks; useless on tmpfs RAM.
  - `load_format="fastsafetensors"`: Hardcoded to CUDA GPU (`device="cuda"`, fails on TPU).
  - `enable_multithread_load`: Parallelizes across multiple shard files; Gemma 4 is a single 11.5 GB file.

## 3. Candidate Fixes Ranked by Saving & Effort

| Rank | Fix Strategy | Expected Saving | Effort | Code Status & Feasibility |
| :--- | :--- | ---: | :--- | :--- |
| **1** | **Filter Multimodal Tensors (Audio + Vision)** | ~50% (~80 s) | Low | [Proven tensor count / Inferred saving] Filter `audio_tower` and `vision_tower` at the iterator level for text-only runs. Skips 1,411 tensors. |
| **2** | **Remove `jax.clear_caches()` in Module Loop** | ~15–25 s | Trivial | [Proven in code / Inferred saving] Remove `jax.clear_caches()` at `weight_utils.py:1132`. Prevents flushing JAX dispatch caches hundreds of times. |
| **3** | **JIT-Compile CPU Unpack (`u32_unpack_i4`)** | ~20–35 s | Low | [Proven in code / Inferred saving] `@jax.jit` the unpack kernel in `quantization/__init__.py:86` to avoid eager Python dispatch and 8x int32 array allocations. |
| **4** | **Pre-Convert Weights to TPU-Native Layout** | ~85% (>135 s) | Medium | [Proven format restriction / Inferred saving] Pre-unpack int4, transpose, and save to HF. **Caveat:** `compressed_tensors.py:167` hardcodes `fmt in (None, "pack-quantized")`; requires adding a direct int4 loader. |
| **5** | **Parallel Shard / Layer Unpacking** | ~60% (~95 s) | High | [Guess] Multi-thread layer unpacking with `ThreadPoolExecutor`. Requires rewriting `AutoWeightsLoader` generator semantics. |

## 4. Proposed Timing Benchmark Script

A standalone, non-destructive benchmark script is provided at [`scripts/tpu_load_time_bench.py`](file:///home/sahil/Desktop/sahil/colab/scripts/tpu_load_time_bench.py). It isolates `_get_nnx_model` weight loading from `/dev/shm/gemma` and measures baseline vs Top Fix (multimodal filtering + no `jax.clear_caches()`). Run duration: <5 min.
