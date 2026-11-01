# LFM2.5 200K TPU Bearings (Part 1)

Tagged versions:
- vllm: v0.32.0
- tpu-inference: v0.32.0

## Report

### a) Where kv_cache_dtype becomes the jnp dtype of the KV arrays
- Status: MEASURED
- Cite: `tpu-inference/tpu_inference/runner/kv_cache_manager.py:KVCacheManager.initialize_kv_cache:1002`
- Snippet:
```python
cache_dtype=t2j_dtype(layer_spec.dtype),
```
Note: In `tpu_runner.py:TPURunner.__init__:1092`, cache dtype is also converted via `kv_cache_dtype = to_jax_dtype(cache_dtype)`. In `kv_cache_manager.py:1002`, `t2j_dtype(layer_spec.dtype)` converts the layer spec dtype to the JAX dtype passed to `create_kv_caches` (allocating `jnp.zeros(..., dtype=cache_dtype)` in `kv_cache.py:_get_kv_cache_allocator:103`).

### b) Where the KV memory check sums bytes per layer spec (attention vs ShortConv)
- Status: MEASURED
- Cite: `vllm/vllm/v1/core/kv_cache_utils.py:max_memory_usage_bytes:1047`
- Snippet:
```python
return sum(spec.max_memory_usage_bytes(vllm_config) for spec in kv_cache_specs)
```
Note: Also in `vllm/vllm/v1/core/kv_cache_utils.py:_max_memory_usage_bytes_from_groups:2362` (`spec.max_memory_usage_bytes(vllm_config)`). Attention uses `FullAttentionSpec.max_memory_usage_bytes` (`vllm/v1/kv_cache_interface.py:607`: `cdiv(max_model_len, self.block_size) * self.page_size_bytes`) while ShortConv uses `MambaSpec.max_memory_usage_bytes` (`vllm/v1/kv_cache_interface.py:1076`: `self.page_size_bytes * (1 + self.num_speculative_blocks)` in non-align mode).

### c) RPA v3 get_kv_cache_shape, and whether head_dim 64 pads to 128
- Status: MEASURED
- Cite: `tpu-inference/tpu_inference/kernels/ragged_paged_attention/v3/kernel.py:get_kv_cache_shape:261`
- Snippet:
```python
def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads * 2, kv_packing) // kv_packing,
        kv_packing,
        align_to(actual_head_dim, 128),
    )
```
- Head_dim 64 pads to 128: Yes, proven by line 274: `align_to(actual_head_dim, 128)` (and `kernel_hd64.py:get_kv_cache_shape:250` which explicitly sets the dimension to `128`).

### d) VllmModelWrapper.load_weights, and whether torchax is on during model.load_weights
- Status: MEASURED
- Cite: `tpu-inference/tpu_inference/models/vllm/vllm_model_wrapper.py:VllmModelWrapper.load_weights:171`
- Snippet:
```python
with load_context, jax_context, set_current_vllm_config(
        self.vllm_config):
    model_config_for_load = vllm_config_for_load.speculative_config.draft_model_config if self.is_draft_model else vllm_config_for_load.model_config
    vllm_model = vllm_get_model(vllm_config=vllm_config_for_load,
                                model_config=model_config_for_load)
```
- Whether torchax is on during model.load_weights: No, proven by lines 241-245 (and line 199 `vllm_config_for_load.device_config.device = "cpu"`): `vllm_get_model` loads weights into PyTorch CPU tensors without `torchax.default_env()`. Model tensors are sharded to TPU later via `shard_model_to_tpu` at line 284.

### e) Flag to skip warmup/precompile
- Status: MEASURED
- Cite: `tpu-inference/tpu_inference/envs.py:<module>:264`
- Snippet:
```python
"SKIP_JAX_PRECOMPILE":
env_bool("SKIP_JAX_PRECOMPILE", default=False),
```
- Flag name: `SKIP_JAX_PRECOMPILE`
- Default: `False`
- Usage cite: `tpu-inference/tpu_inference/runner/compilation_manager.py:CompilationManager.capture_model:264`:
```python
if envs.SKIP_JAX_PRECOMPILE or self.runner.model_config.enforce_eager:
    return
```

