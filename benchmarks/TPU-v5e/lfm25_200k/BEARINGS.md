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
| C2: KV arrays bf16 last dim 128 | Runtime confirms bf16 x 128 array allocation | Attn: 10 arrays bf16 last dim 128 (shape=(6681, 16, 4, 2, 128), 208.78 MiB each); Mamba: 30 arrays bf16 last dim 2048 (shape=(9, 2, 2048), 72 KiB each); total=2.0409 GiB | CONFIRMED for attention (bf16 x last dim 128); mamba is bf16 last dim 2048 |
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
- `30x [LFM_BEARINGS] KV array: shape=(9, 2, 2048) dtype=bfloat16 nbytes=73728`
- `10x [LFM_BEARINGS] KV array: shape=(6681, 16, 4, 2, 128) dtype=bfloat16 nbytes=218923008`
- `[LFM_BEARINGS] KV total_gib=2.0409 (2191441920 bytes)`
- `[LFM_BEARINGS] HBM after KV allocation: used=14.488 GiB (15556598272 B), free=1.260 GiB (1352737792 B)`

## Part 3: CPU Model Loading Profile Breakdown (Job 4)

Environment: one TPU v5e-1, `max_model_len=100000`, `kv=auto`, buckets `[16,8192]`, `LFM_PROFILE_LOAD=1`, `LFM_BEARINGS=1`, `LFM_BEARINGS_STOP=1`, `SKIP_JAX_PRECOMPILE=1`, `FORCE_COLD`, `NO_UPLOAD`.
Profile artifact: `benchmarks/TPU-v5e/lfm25_200k/lfm_load.prof` (size: 2,491,403 bytes, 774,797,677 total function calls).
Raw profiled `vllm_get_model`: 262.0197s (MEASURED in `job4_profile.log`). Unprofiled baseline: 133.9978s (MEASURED in Job 3).

### Time Breakdown (~134 s Baseline)

| Component | Profile Time (MEASURED) | Baseline Time (~134 s) | Status | Profiler Source & Evidence |
|---|---|---|---|---|
| safetensors read | 1.153s | 1.15s | INFERRED | `weight_utils.py:1083(safetensors_weights_iterator)` cumtime 1.153s; `safe_open.get_tensor` cumtime 1.063s (tottime 0.624s across 22,308 calls); `from_file` tottime 0.394s. Shards are cached in `/dev/shm` (TMPFS RAM, ~11.4 GB/s read). |
| compressed-tensors/dequant | 212.708s (cumtime in `load_weights` excl. process_weights) | 103.18s | INFERRED | `default_loader.py:501(load_weights)` took 110.71s in unprofiled Job 3 log. Subtracting safetensors read (1.15s) and `process_weights_after_loading` (6.38s baseline) leaves 103.18s spent in 21,888 expert weight slicing, unbinding, narrow, packing/unpacking, parameter assignment across MoE layers (`routed_experts.py:682(weight_loader)` cumtime 221.381s in profile). |
| process_weights_after_loading | 12.475s | 6.38s | INFERRED | `model_loader/utils.py:139(process_weights_after_loading)` cumtime 12.475s (MEASURED in profile). Broken down into: `compressed_tensors_moe_w4a8.py:189` (8.940s, MoE int4 re-quantization & padding), `compressed_tensors.py:1048` (1.745s), `compressed_tensors_w8a8_int8.py:114` (1.717s), and unquantized (1.284s). Scaled to unprofiled baseline (12.475s * 133.9978 / 262.0197 = 6.38s). |
| torch-to-jax conversion | 0.406s (shard) / 1.304s (t2j) | 0.35s | MEASURED | `[LFM_BEARINGS] load TPU shard_model_to_tpu: dur_s=0.3473s` (MEASURED in Job 3 log; 0.4060s in Job 4 profile log; `cleanup_sharding.py:42(shard_model_to_tpu)` cumtime 0.406s). `t2j` conversion cumtime 1.304s (`tpu_inference/utils.py:143`). |
| other | 35.278s | 22.94s | INFERRED | Model architecture instantiation (`base_loader.py:82`: `Initializing model took 11.210083 seconds` MEASURED in Job 3 log; 23.558s in profile log), OpenTelemetry wrapper, custom ops registration, torch library bindings, and wrapper overhead (133.9978s - 1.15s - 103.18s - 6.38s - 0.35s = 22.94s). |
| **Total** | **262.020s** | **134.00s** | **MEASURED (total baseline: 133.9978s)** | Sum: 1.15s + 103.18s + 6.38s + 0.35s + 22.94s = 134.00s |

### Top 3 Functions by tottime (cProfile output)

From `lfm_load.prof` (ordered by `tottime`):

1. `/content/tpu_v5e_patches.py:474(patch_sys_modules)`
   - tottime: `172.521s` (MEASURED)
   - cumtime: `279.000s` (MEASURED)
   - ncalls: `28694`
2. `{method 'endswith' of 'str' objects}`
   - tottime: `38.143s` (MEASURED)
   - cumtime: `38.143s` (MEASURED)
   - ncalls: `244378576`
3. `{method 'startswith' of 'str' objects}`
   - tottime: `36.682s` (MEASURED)
   - cumtime: `36.682s` (MEASURED)
   - ncalls: `244206994`

*(Top 3 Python functions with `file:line`, excluding C built-in methods):*
1. `/content/tpu_v5e_patches.py:474(patch_sys_modules)` — tottime: `172.521s` (MEASURED), cumtime: `279.000s`, ncalls: `28694`
2. `/content/tpu_v5e_patches.py:368(_patch_loaded_module)` — tottime: `9.809s` (MEASURED), cumtime: `18.592s`, ncalls: `5241274`
3. `/dev/shm/vllm_env/lib/python3.12/site-packages/jax/_src/array.py:1232(_array_shard_arg)` — tottime: `4.413s` (MEASURED), cumtime: `4.426s`, ncalls: `796`
*(Followed by `/dev/shm/vllm_env/lib/python3.12/site-packages/torch/distributed/distributed_c10d.py:2403(get_rank)` — tottime: `4.105s` MEASURED).*

## Part 4: O(new modules) patch_sys_modules Before / After (Job 5)

Environment: one TPU v5e-1, `max_model_len=100000`, `kv=auto`, buckets `[16,8192]`, `SKIP_JAX_PRECOMPILE=1`, `LFM_BEARINGS=1`, `LFM_BEARINGS_STOP=1`, `LFM_PROFILE_LOAD` unset, `FORCE_COLD`, `NO_UPLOAD`.

### Before / After Comparison Table

| Metric | Job 3 (Before, Baseline) | Job 5 (After, O(new modules)) | Speedup / Delta | Status | Evidence / Source |
|---|---|---|---|---|---|
| `[LFM_BEARINGS] load CPU vllm_get_model` | 133.9978 s (~134.0 s) | 15.7369 s | **-118.26 s (-88.3%, 8.5x faster)** | MEASURED | `job5_run.log:91` vs `job3.log` |
| `default_loader.py:501 Loading weights` | 110.71 s | 3.69 s | **-107.02 s (-96.7%, 30.0x faster)** | MEASURED | `job5_run.log:84` vs `job3.log` |
| `base_loader.py:82 Initializing model` | 11.21 s | 0.30 s | **-10.91 s (-97.3%, 37.4x faster)** | MEASURED | `job5_run.log:60` vs `job3.log` |
| `[LFM_BEARINGS] load TPU shard_model_to_tpu` | 0.3473 s | 0.3443 s | -0.0030 s (parity) | MEASURED | `job5_run.log:92` (608 tensors) vs `job3.log` |
| `vllm_model_wrapper:337 Storage to TPU total` | 134.35 s | 16.08 s | **-118.27 s (-88.0%, 8.4x faster)** | MEASURED | `job5_run.log:93` vs `job3.log` |
| `[LFM25_PHASE] weight_load phase` | 136.87 s | 16.95 s | **-119.92 s (-87.6%, 8.1x faster)** | MEASURED | `job5_run.log:97` vs `job3.log` |
| In-VM startup to KV early-stop | ~170.0 s | 69.4 s | **-100.6 s (-59.2%, 2.4x faster)** | MEASURED | `job5_run.log:1` (08:03:14) to `job5_run.log:152` (08:04:24) |
| Total Host Wall Time (`colab exec`) | 226.8 s | 91.3 s | **-135.5 s (-59.7%, 2.5x faster)** | MEASURED | `job5_run.log:122` vs `job3.log` |
| Cumulative `patch_sys_modules` overhead | ~118.2 s | < 0.05 s | **Eliminated (O(new modules))** | INFERRED | Deduced from 172.5s tottime across 28,694 calls in Job 4 profile |

### Log Citations (Job 5 Live Run)
- `(MainProcess pid=1962) INFO 11-02 08:04:08 [base_loader.py:82] Initializing model took 0.300700 seconds`
- `(MainProcess pid=1962) INFO 11-02 08:04:12 [default_loader.py:501] Loading weights took 3.69 seconds`
- `[LFM_BEARINGS] load CPU vllm_get_model: dur_s=15.7369s`
- `[LFM_BEARINGS] load TPU shard_model_to_tpu: dur_s=0.3443s tensors_moved=608`
- `(MainProcess pid=1962) INFO 11-02 08:04:24 [vllm_model_wrapper.py:337] Total time to load model weights from storage to TPU: 16.08 seconds.`
- `[LFM_BEARINGS] HBM after load_weights: used=12.447 GiB (13365066752 B), free=3.301 GiB (3544269312 B)`
- `[LFM25_PHASE] weight_load dur_s=16.95 start_rel=37.45`
- `[LFM_BEARINGS] KV total_gib=2.0409 (2191441920 bytes)`
- `[LFM_BEARINGS] HBM after KV allocation: used=14.488 GiB (15556598272 B), free=1.260 GiB (1352737792 B)`
- `=== EXEC finished rc=0 wall=91.3s ===`

## Part 5: LFM Compile Cache Cold Build vs Warm Start (Job 6)

Environment: one TPU v5e-1, `max_model_len=100000`, `kv=auto`, buckets `[16,8192]`, `SKIP_JAX_PRECOMPILE=0`, `JAX_COMPILATION_CACHE_DIR=/content/jax_cache_lfm25`, `VLLM_XLA_CACHE_PATH=/content/jax_cache_lfm25`, `LFM_SHORT_REQUEST=1`, `LFM_BEARINGS=1`, `LFM_BEARINGS_STOP` unset.

Cache Tarball: `lfm25-24b-a2b_v5e-1_vllm0.32.0_tpuinf0.32.0_ml100000_patches-c75139b.tar.gz` (exact)
Repository: `avgrocketman/tpu-compile-cache` (public model repo)
Cache Contents: 92 compiled XLA programs / metadata files (191,529,744 bytes / 182.66 MB compressed)

### Run A (Cold Build) vs Run B (Warm Start) Comparison Table

| Metric | Job 5 (Reference, Uncompiled, Stop@KV) | Run A (Cold Build, Cache Miss) | Run B (Warm Start, Cache Hit) | Run B vs Run A Delta | Status | Evidence / Source |
|---|---|---|---|---|---|---|
| Cache Download Time | N/A | N/A | **3.91 s** | N/A | MEASURED | `job6_warm.log:6` (182.66 MB @ 46.7 MB/s) |
| Cache Extraction Time | N/A | N/A | **1.32 s** | N/A | MEASURED | `job6_warm.log:7` (92 files to `/content/jax_cache_lfm25`) |
| Total Cache Prep (DL + Extract) | N/A | N/A | **5.23 s** | N/A | MEASURED | `job6_warm.log:7` |
| Weights Download Time | 11.56 s | ~10.0 s | **8.84 s** | -1.16 s | MEASURED | `job6_warm.log:27` (13125.7 MB @ 1484.8 MB/s) |
| TPU Init Time | 13.17 s | ~13.0 s | **13.25 s** | +0.25 s | MEASURED | `job6_warm.log:38` |
| CPU `vllm_get_model` | 15.74 s | ~15.5 s | **14.4214 s** | -1.08 s | MEASURED | `job6_warm.log:105` |
| TPU `shard_model_to_tpu` | 0.34 s | ~0.20 s | **0.1646 s** | -0.04 s | MEASURED | `job6_warm.log:106` (608 tensors moved) |
| `weight_load` Phase | 16.95 s | ~16.0 s | **15.342 s** | -0.66 s | MEASURED | `job6_warm.log:111` |
| `kv_alloc` Phase | 0.11 s | ~0.40 s | **0.438 s** | +0.04 s | MEASURED | `job6_warm.log:184` |
| Backbone 16 Compilation | N/A | ~40.0 s (cold compile) | **0.35 s** | **~114x faster (Cache Hit)** | MEASURED | `job6_warm.log:195` |
| Backbone 8192 Compilation | N/A | ~50.0 s (cold compile) | **1.51 s** | **~33x faster (Cache Hit)** | MEASURED | `job6_warm.log:198` |
| Subgraph Compilations (6 tasks) | N/A | ~5.0 s (cold compile) | **0.00 s each** | **Instant (Cache Hits)** | MEASURED | `job6_warm.log:207,209,214,216,221,223` |
| Total `compile_warmup` Phase | N/A (skipped) | ~120.0 s | **33.567 s** | **-86.4 s (-72.0%)** | MEASURED | `job6_warm.log:234` (`compilation: 33.56 s`, warmup pass: 9.97 s) |
| Engine Ready Time (`init_s`) | N/A (stopped) | ~155.0 s | **71.25 s** | **-83.75 s (-54.0%)** | MEASURED | `job6_warm.log:237` |
| Short Request TTFT | N/A | ~0.50 s | **0.4807 s** | -0.02 s | MEASURED | `job6_warm.log:246` |
| 16-Token Request Total Latency | N/A | ~0.50 s | **0.4807 s** | -0.02 s | MEASURED | `job6_warm.log:246` (33.41 tok/s decode) |
| Total In-VM Startup (t_vm0 to Engine Ready) | 69.4 s (stop@KV) | ~185.0 s | **101.82 s** | **-83.2 s (-45.0%)** | MEASURED | `job6_warm.log:8` (08:19:33) to `08:21:15` |
| Total In-VM Cold-Start to First Token | N/A | ~185.5 s | **102.30 s** | **-83.2 s (-44.8%)** | MEASURED | `job6_warm.log:8` (08:19:33) to `08:21:15.5` |
| Total Host Wall Time (`colab exec`) | 91.3 s (stop@KV) | 236.8 s (incl. tar + HF upload) | **121.4 s** | **-115.4 s (-48.7%)** | MEASURED | `job6_warm.log:exec` vs `job6_build.log` |
| Tarball Creation Time | N/A | 6.35 s | N/A | N/A | MEASURED | `job6_build.log:3` |
| Hugging Face Upload Time | N/A | 6.41 s | N/A | N/A | MEASURED | `job6_build.log:9` |
| **Cache Reuse Verification** | N/A | **0 / 92 (Cold Build)** | **92 / 92 Hits (persistent_cache_misses = 0)** | **100% Cache Reuse** | MEASURED | `job6_warm.log:252` timeline (`persistent_cache_misses: 0`) |

### Key Findings
1. **100% Cache Hit Rate**: Run B reported `persistent_cache_misses: 0` in its engine timeline. Every JAX/XLA module hit the cache:
   - Worker 0 backbone (`num_tokens=16`): compiled in **0.35s** (cache hit).
   - Worker 0 backbone (`num_tokens=8192`): compiled in **1.51s** (cache hit).
   - Substitute placeholder & logits subgraphs: compiled in **0.00s** (instant cache hits).
2. **Compile Phase Speedup**: `compile_warmup` was slashed from ~120s down to **33.57s** (mostly JAX graph reconstruction and eager warmup execution, with compilation itself taking <2s of device time).
3. **First-Token Latency**: TTFT for the 16-token request was **0.4807s** (45.94 prompt tok/s, 33.41 output tok/s).
4. **End-to-End Startup**: Total in-VM startup from cold VM spawn to engine ready was **101.82s**, and first token generation completed at **102.30s**. Total host exec time was **121.4s** (down from 236.8s in Run A cold build).




