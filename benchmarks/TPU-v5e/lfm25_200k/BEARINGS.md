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

## Part 2: Live TPU Measurements & Claims Verification

Environment: one TPU v5e-1, `max_model_len=100000`, `kv=auto`, buckets `[16,8192]`, `LFM_BEARINGS=1 STOP=1`, `SKIP_JAX_PRECOMPILE=1`, `FORCE_COLD`, `NO_UPLOAD`.

### Phase Timings & Host Wall
- Provision: 4.61s; weights_dl: 11.50s (9 shards, 13125.7 MB @ 1141.7 MB/s); venv_dl: 4.86s; venv_unpack: 7.75s; TPU init: 16.09s
- CPU `vllm_get_model`: 133.9978s; TPU `shard_model_to_tpu`: 0.3473s (608 tensors moved); storage-to-TPU total: 134.35s
- `weight_load`: 136.87s; `llm_init`: 169.29s; host wall: 226.8s (CU 390.83 -> 390.65, cache_publish=null)
- HBM after load: used = 12.447 GiB (13,365,066,752 B), free = 3.301 GiB (3,544,269,312 B), limit = 15.75 GiB

### Claims Verification Table
| Claim | Expected | Measured | Verdict |
|---|---|---|---|
| C1: ShortConv (MambaSpec) counted per block (~8192 B x blocks per layer) | vLLM counts mamba per block across all blocks | Hybrid pad: 57,344 B/block (1x attn 32768 + 3x mamba 8192); Compact-mamba: 30 layers x 9 slots x 8192 B = 0.00 GiB (does not scale with 100K blocks) | CONFIRMED: vLLM pads mamba per block into 57,344 B; TPU compact-mamba uses fixed 9 slots x 8,192 B x 30 layers |
| C2: KV arrays bf16 last dim 128 | Runtime confirms bf16 x 128 array allocation | No `[LFM_BEARINGS] spec class=` or `KV array:` lines logged due to RecursionError | NOT CONFIRMED at runtime (no KV array log). Page size 32768 B is consistent with bf16 x last dim 128, but arrays unmeasured |
| C3: ~120s load dominated by CPU `vllm_get_model` vs TPU sharding | CPU load ~120s, TPU shard <1s | `vllm_get_model` = 133.9978s; `shard_model_to_tpu` = 0.3473s (608 tensors) | CONFIRMED: 133.9978s CPU vs 0.3473s TPU sharding |

### 200K KV Projection
- WITH overcount (prior 200K run1): 6.68 GiB needed vs 3.57 GiB free (OOM).
- WITHOUT overcount (scaled compact-mamba): attn 10 layers * (200,000 / 16) * 32,768 B = 3.81 GiB + mamba (30 * 9 * 8,192 B = 2.11 MiB) = ~3.82 GiB vs 3.301 GiB free HBM.

### Log Citations
- `[LFM_BEARINGS] load CPU vllm_get_model: dur_s=133.9978s`
- `[LFM_BEARINGS] load TPU shard_model_to_tpu: dur_s=0.3473s tensors_moved=608`
- `Total time to load model weights from storage to TPU: 134.35 seconds.`
- `[LFM_BEARINGS] HBM after load_weights: used=12.447 GiB (13365066752 B), free=3.301 GiB (3544269312 B)`
- `Hybrid KV cache: padding every layer spec to 57344 bytes (num_attn_groups=1 × attn_page=32768 + num_mamba_groups=3 × mamba_unpadded=8192).`
- `Compact-mamba KV cache: num_gpu_blocks_override=6681 (attn), _mamba_num_blocks=9 (1 blocks/request). HBM split: attn=10 layers x 6681 blocks x 32768 B = 2.04 GiB; mamba=30 layers x 9 slots x 8192 B = 0.00 GiB; total=2.04 GiB / avail=2.04 GiB.`
- `Overriding num_gpu_blocks=3821 with num_gpu_blocks_override=6681`
- `TPU KV cache size: 106,844 tokens, Maximum concurrency for 100,000 tokens per request: 1.07x`
- `LLM() failed after 169.29s: RecursionError: maximum recursion depth exceeded`


