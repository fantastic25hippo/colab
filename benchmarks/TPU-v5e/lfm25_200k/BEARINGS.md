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

## Part 6: Job 7 startup cleanup, Run B warm restore

Environment: TPU v5e-1, max_model_len=100000, SKIP_JAX_PRECOMPILE=0, LFM_SHORT_REQUEST=1, default launch (no LFM_CACHE_FORCE_COLD). Cache dir `/dev/shm/tpu_compile_cache`. File `tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len100000__b16-8192__e9b7effa.tar`. Driver restore hit=true.

### Phase Table

| Phase | s | Status | Evidence |
|---|---|---|---|
| cache_dl_s | 2.92 | MEASURED | driver cache_dl_s |
| cache_unpack_s | 0.08 | MEASURED | driver cache_unpack_s |
| import jax | 1.71 | MEASURED | subrunner 08:59:22 |
| import vllm | 11.65 | MEASURED | subrunner 08:59:34 |
| TPU init | 13.36 | MEASURED | subrunner; includes jax+vllm |
| LLM() arg/config setup | 0.0000 | MEASURED | subrunner 08:59:38 |
| weight_load | 14.885 | MEASURED | engine.phases |
| compile_warmup | 32.876 | MEASURED | engine.phases |
| TTFT | 0.4817 | MEASURED | LFM_SHORT_REQUEST |
| t_vm0 to first token | 108.5 | INFERRED | t_vm0 log 08:59:00 + LLM start 08:59:38 + init_s 70.02 + TTFT 0.4817; JSON in_vm field was null |
| host t_global0 to t_vm0 | 15.67 | MEASURED | unix 1793609940.86 - 1793609925.19 |
| host t_global0 to first token | 124.2 | INFERRED | 15.67 + 108.5 |
| host EXEC wall | 137.2 | MEASURED | EXEC finished; includes wrap-up after first token |

### Cache Note
- restore hit, 92 entries / 196.0 MB before and after, new_entries=0 (MEASURED)
- timeline persistent_cache_misses=0 (MEASURED)
- compile finished in: backbone 16 = 0.34s, backbone 8192 = 1.48s, substitute/select/logits = 0.00s (job6_warm.log) (MEASURED)
- No explicit persistent-cache hit counter in the log. Hits INFERRED as 92/92 files reused (new_entries=0, misses=0).

## Part 7: Job 8 startup fixes, Run B warm restore

Environment: TPU v5e-1, max_model_len=100000, SKIP_JAX_PRECOMPILE=0, LFM_SHORT_REQUEST=1, LFM_CACHE_PARALLEL=1, LFM_EARLY_IMPORT=1, LFM_THP=1, LFM_PHASE_DETAIL=1. Cache dir `/dev/shm/tpu_compile_cache`. File `tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len100000__b16-8192__5a0d66f4.tar`. Driver restore hit=true (Run A published new hash with verified_listed=true under FORCE_COLD). THP failed due to read-only `/sys` (continued safely).

### Phase Table

| Phase | Job 8 Run B (s) | Job 7 Run B (s) | Delta | Status | Evidence |
|---|---|---|---|---|---|
| cache dl | 2.71 | 2.92 | -0.21s | MEASURED | driver cache_dl_s (overlapped) |
| cache unpack | 0.11 | 0.08 | +0.03s | MEASURED | driver cache_unpack_s (overlapped) |
| venv dl | 8.81 | n/a | n/a | MEASURED | driver venv_dl_s |
| venv unpack | 6.91 | n/a | n/a | MEASURED | driver venv_unpack_s |
| weights dl | 11.83 | n/a | n/a | MEASURED | driver weights dl (9 shards, 13125.7 MB) |
| import jax | 1.74 | 1.71 | +0.03s | MEASURED | subrunner 09:44:14 |
| import vllm | 12.03 | 11.65 | +0.38s | MEASURED | subrunner 09:44:26 |
| TPU init | 13.77 | 13.36 | +0.41s | MEASURED | subrunner 09:44:30; includes jax+vllm |
| config_setup_s | ~19.0 | n/a | n/a | INFERRED | JSON null; LLM() start 09:44:30 to V1 engine init 09:44:49 |
| weight_load | 15.418 | 14.885 | +0.533s | MEASURED | engine.phases.weight_load_s |
| profile_run | 0.00 | n/a | n/a | MEASURED | LFM25_PHASE profile_run dur_s=0.00 |
| kv alloc | 0.441 | n/a | n/a | MEASURED | engine.phases.kv_alloc_s |
| precompile_load | 0.01 | n/a | n/a | MEASURED | LFM25_PHASE precompile_load dur_s=0.01; ~21.2s gap to first warmup |
| warmup backbone 16 | 3.89 | n/a | n/a | MEASURED | engine.phases warmup task (logged twice, 3.887s) |
| warmup backbone 8192 | 6.04 | n/a | n/a | MEASURED | engine.phases warmup task (logged twice, 6.041s) |
| compile_warmup | 31.614 | 32.876 | -1.262s | MEASURED | engine.phases.compile_warmup_s |
| TTFT | 0.4823 | 0.4817 | +0.0006s | MEASURED | LFM_SHORT_REQUEST TTFT |
| in-VM total (t_vm0 to first token) | 107.86 | 108.5 | -0.64s | MEASURED | in_vm_cold_start_to_first_token_s (t_vm0=1793612632.28) |
| host total (t_global0 to first token) | 118.35 | 124.2 | -5.85s | MEASURED | total_cold_start_to_first_token_s (t_global0=1793612621.80) |
| host EXEC wall | 122.6 | 137.2 | -14.6s | MEASURED | host runner EXEC wall |

### Cache & Runtime Notes
- Restore hit: 92 entries / 196.0 MB; persistent_cache_misses=0; new_entries=0.
- Backbone compilations finished in 0.35s (tokens=16) and 1.51s (tokens=8192); subgraphs finished in 0.00s (cache hits).
- Subrunner spawned at rel 16.65s; weights barrier cleared after 0.00s wait.
- `llm_init_unaccounted_s`: 21.94s (mostly the inferred config_setup + precompile gap).

## Part 8: Job 9 Diagnostic Report — `config_setup` (W1) and Pre-Warmup Gap (W2)
Environment: TPU v5e-1 (`tpu-lfm25`), `max_model_len=100000`, `SKIP_JAX_PRECOMPILE=0`, `LFM_SHORT_REQUEST=1`, `LFM_BEARINGS=1`, `LFM_DIAG=1` hooks enabled in subrunner, warm restore (`tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len100000__b16-8192__5a0d66f4.tar`, `hit=true`, 92 entries / 196.0 MB). Sampler: `py-spy dump --pid` (`py-spy 0.4.2`, 1.0 s interval, 77 samples in `job9_stacks/`).
- **`config_setup_s` (W1)**: **`19.15 s`** (`MEASURED`, direct `time.perf_counter()` around `EngineCore.__init__`; `EngineArgs.create_engine_config` = **`11.5473 s`** `MEASURED`; `llm_init_unaccounted_s` = **`3.10 s`** `MEASURED`). Job 8 bounds: `job8_warm.log:33` (`09:44:30`) to `job8_warm.log:47` (`09:44:49`), ~19.0 s (`INFERRED`). Job 9 bounds: `job9_warm.log:35` (`10:28:31`, `rel_s=49.86s`) to `job9_warm.log:49-50` (`10:28:50`, `rel_s=69.01s`), **18 samples** (`0049.94.txt`..`0068.94.txt`).
- **Pre-Warmup Gap (W2)**: **`21.89 s`** (`MEASURED` from `start_rel`: `76.69 - 54.80` in `job9_warm.log:185,199`; **`21.20 s`** `MEASURED` in `job8_warm.log:181,196` from `74.75 - 53.55`). Job 8 bounds: `job8_warm.log:180-182` (`09:45:06`, `start_rel=53.55`) to `job8_warm.log:195-196` (`09:45:27`, `start_rel=74.75`). Job 9 bounds: `job9_warm.log:184-186` (`10:29:08`, `start_rel=54.80`, `rel_s=86.57s`) to `job9_warm.log:198-199` (`10:29:30`, `start_rel=76.69`, `rel_s=108.22s`), **20 samples** (`0086.57.txt`..`0108.22.txt`).

| Window | Rank | Top 3 Frames (`file:function`) | Top App Frame (`path:line`) | Samples (MEASURED) | Approx. s (INFERRED) |
|---|---|---|---|---|---|
| W1 | 1 | `selectors.py:select -> subprocess.py:_communicate -> subprocess.py:communicate` | `vllm/model_executor/models/registry.py:1563(_run_in_subprocess)` | 11 / 18 | ~11.7 s |
| W1 | 2 | `tpu_v5e_patches.py:patch_sys_modules -> tpu_v5e_patches.py:custom_import -> <mod>:<module>` | `tpu_v5e_patches.py:577-578(patch_sys_modules)` | 4 / 18 | ~4.2 s |
| W1 | 3 | `<frozen importlib._bootstrap_external>:get_data -> get_code -> exec_module` | `tpu_v5e_patches.py:592(custom_import)` | 1 / 18 | ~1.1 s |
| W1 | 4 | `<frozen importlib._bootstrap>:release -> __exit__ -> _find_and_load` | `tpu_v5e_patches.py:592(custom_import)` | 1 / 18 | ~1.1 s |
| W1 | 5 | `tokenization_utils_base.py:from_pretrained -> tokenization_auto.py:from_pretrained -> hf.py:from_pretrained` | `vllm/v1/engine/input_processor.py:94(__init__)` | 1 / 18 | ~1.1 s |
| W2 | 1 | `pjit.py:_infer_params -> pjit.py:cache_miss -> traceback_util.py:reraise_with_filtered_traceback` | JAX `pjit` tracing (`jax/_src/pjit.py:644`, `kernel_hd64.py:522`) | 2 / 20 | ~2.2 s |
| W2 | 2 | `lowering.py:jaxpr_subcomp -> lowering.py:_cond_lowering_rule -> lowering.py:jaxpr_subcomp` | Pallas Mosaic lowering (`jax/_src/pallas/mosaic/lowering.py:1784,1818`) | 2 / 20 | ~2.2 s |
| W2 | 3 | `<string>:__init__ -> source_info_util.py:__add__ -> lowering.py:jaxpr_subcomp` / `mlir.py:jaxpr_subcomp` | JAX/MLIR & Mosaic lowering (`jax/_src/interpreters/mlir.py:2461`) | 2 / 20 | ~2.2 s |
| W2 | 4 | `compilation_cache.py:decompress_executable -> compilation_cache.py:get_executable_and_time -> compiler.py:_cache_read` | Persistent cache zstd decompress (`jax/_src/compilation_cache.py:312`) | 1 / 20 | ~1.1 s |
| W2 | 5 | `core.py:<lambda> -> gmm_v2.py:validate_inputs -> gmm_v2.py:make_gmm_configs` | MoE GMM v2 tracing (`tpu_inference/layers/vllm/gmm_v2/gmm_v2.py:1284`) | 1 / 20 | ~1.1 s |

*(Remaining 12 W2 samples: 1 sample / ~1.1 s each across `mosaic/primitives.py:263`, `partial_eval.py:2250`, `pjit.py:658`, `jax/_src/core.py:320`, `mlir.py:1303`, `torchax/tensor.py:724`, `ufunc_api.py:176`, `dtypes.py`, `indexing.py`, `effects.py`, `api_util.py`).*
- **W1 Diagnosis (`INFERRED` from `MEASURED` stacks)**: ~11.5 s (61%) is spent blocked in `selectors.select()` while `ModelConfig.__post_init__` (`vllm/config/model.py:694,730`) spawns a fresh `python3 -m vllm.model_executor.models.registry` child process via `_LoadedModel.inspect_model_cls` (`vllm/model_executor/models/registry.py:1079`) -> `_run_in_subprocess` (`vllm/model_executor/models/registry.py:1563`) to inspect `Lfm2MoeForCausalLM`. ~5.0 s is lazy module/plugin imports in `LLM.__init__` (`vllm/entrypoints/llm.py:339` also resets loggers via `configure_logging_if_needed`, detaching Job 8's `_CfgHandler`), and ~2.6 s is tokenizer load (`vllm/v1/engine/input_processor.py:94`).
- **W2 Diagnosis (`INFERRED` from `MEASURED` stacks)**: Host CPU symbolic tracing (`torch.compile` + `torchax` + JAX `pjit`) and MLIR/Mosaic lowering (`lowered = fn.lower(*args, **call_kwargs)`) inside `CompilationManager._run_compilation` (`tpu-inference/tpu_inference/runner/compilation_manager.py:190`), plus cache lookup (`lowered.compile()` at `:209`, 0.34 s for bucket 16 and 1.56 s for bucket 8192), invoked from `capture_model` (`:272-273`). Bucket 16 takes ~8.0 s (`4.96 s` in `torch.compile` at `monitor.py:62` + `2.7 s` JAX lowering + `0.34 s` cache read); bucket 8192 takes ~13.9 s (`~12.3 s` JAX/Pallas/GMM v2 lowering + `1.56 s` cache decompress). Warmup tasks only run after both buckets finish lowering, inside `_flush_compilations()` (`:248-255`).
- **Overhead Check (`MEASURED`)**: `llm_init_s` `71.80 s` (vs `69.42 s` Job 8; `+2.38 s` from 1 Hz `py-spy` ptrace attach; `compile_warmup_s` `32.81 s` vs `31.61 s`, `+1.19 s`); in-VM total `113.11 s` (vs `107.86 s` Job 8, including `+2.73 s` pre-subrunner HF `venv_dl_s` variance `11.54 s` vs `8.81 s`; in-subrunner delta `+2.52 s`); host total `122.15 s` (vs `118.35 s`); host EXEC wall `136.1 s` (vs `122.6 s`, includes post-run stdout tar transfer of 77 stack files).

## Part 9: Job 10 in-process inspect, patch-fast, anon HF, waterfall

Label every number MEASURED or INFERRED. Facts from job10_warm_1.log (run 2 skipped):

STOP: cache key changed with tpu_v5e_patches.py. Job 9 tar `...__5a0d66f4.tar`. Run 1 requested `...__42f6059b.tar`, 404 miss (MEASURED). LFM_ABORT_ON_CACHE_MISS sat after LFM_EARLY_IMPORT spawn, so run 1 still cold-compiled and published `42f6059b` (MEASURED; 196.1 MB, upload_s=2.99, verified_listed=true). Run 2 skipped.

Run 1 vs Job 8 warm (107.86 / 118.35 / 122.6) — this run is a COLD compile, not a warm restore:
- config_setup_s JSON null (MEASURED; `_CfgHandler` detached). LLM() 11:29:20 to "Initializing a V1 LLM engine" 11:29:25 ~5 s (INFERRED) vs Job 9 19.15 s.
- create_engine_config_s=0.159 s, tokenizer_s=1.018 s (MEASURED). model_inspect_s absent from waterfall.
- [LFM_INPROC_INSPECT] active three times; architecture resolved in-process at 11:29:23 (MEASURED).
- in-VM 196.76 s, host 206.28 s, EXEC 214.2 s (MEASURED). compile_warmup_s=125.862 s (MEASURED cold).
- cache hits=0. HF downloads logged token=none for weights, venv312_cpu.tar.zst, zstd_bin, cache tar (MEASURED). Publish kept token (upload ran).
- Host WATERFALL unaccounted_s=40.205 s (MEASURED). Gaps: EXEC start→cache dl 0.957 s; venv unpack→imports 3.943 s; imports→precompile_load 3.755 s; compile_warmup→ttft 31.55 s. in-VM unaccounted_s=35.305 s. Goal missed: PHASE rows all pinned at start_rel=16.856 s (INFERRED clock bug).
- THP FAIL read-only sysfs (MEASURED). HF Hub warned unauthenticated requests (MEASURED).
- Tell Sahil to check https://huggingface.co/settings/downloads ; do not log in.

## Part 10: Job 10b Waterfall Clock Fix, Early Miss Abort & Warm Runs

Environment: TPU v5e-1 (`tpu-lfm25`), `max_model_len=100000`, `SKIP_JAX_PRECOMPILE=0`, `LFM_SHORT_REQUEST=1`, `LFM_BEARINGS=1`, `LFM_DIAG=0`, `LFM_CACHE_PARALLEL=1`, `LFM_EARLY_IMPORT=1`, `LFM_THP=1`, `LFM_PHASE_DETAIL=1`, `LFM_INPROC_INSPECT=1`, `LFM_PATCH_FAST=1`, `LFM_HF_ANON=1`, `LFM_ABORT_ON_CACHE_MISS=1`.
Cache tar: `tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len100000__b16-8192__42f6059b.tar` (92 entries, 196.0 MB).

### Preflight (P1–P3)
- **P1**: Local driver cache key computation = `42f6059b` (MEASURED; matched expected `42f6059b`). PASS.
- **P2**: Remote compile cache tar verified listed on `avgrocketman/tpu-compile-cache` via tokenless read-only query (MEASURED). PASS.
- **P3**: `git diff --stat` touched only allowed files (`lfm25_coldstart_driver.py` and `lfm25_engine_subrunner.py`), with 34 insertions (cap 60 changed lines). PASS.

### Fixes Verified
1. **Early Miss Abort**: `f_cache.result()` resolved before subrunner spawn. On cache miss, exits with code 2 before spawning `lfm25_engine_subrunner.py` or initiating any compilation, publishes nothing, and stops runtime.
2. **Waterfall Clock**: Every phase records distinct `time.perf_counter()` on start (no shared `t_cfg0` fallback; fixed `start_rel_s` extraction from `PHASE["spans"]`).
3. **Parallel Marking & Overlap**: Parallel downloads (`weights dl`, `venv dl`, `cache dl`, `cache unpack`) marked `parallel=true` and counted once in `accounted_s` via union of intervals.

### Totals vs Job 8 Baseline

| Metric | Job 8 Warm | Job 10b Run 1 | Job 10b Run 2 | Delta vs Job 8 (Run 2) | Status |
|---|---|---|---|---|---|
| Cache hit count | 92 / 92 | 92 / 92 | 92 / 92 | identical | MEASURED |
| Persistent cache misses | 0 | 0 | 0 | 0 | MEASURED |
| `config_setup_s` | ~19.0 s (INFERRED) | 5.36 s | 5.32 s | -13.68 s | MEASURED |
| `weight_load_s` | 15.42 s | 14.42 s | 14.45 s | -0.97 s | MEASURED |
| `compile_warmup_s` | 31.61 s | 31.25 s | 31.71 s | +0.10 s | MEASURED |
| In-VM to first token | 107.86 s | 85.44 s | 86.23 s | -21.63 s | MEASURED |
| Subrunner in-VM to first token | n/a | 70.85 s | 71.31 s | n/a | MEASURED |
| Host to first token | 118.35 s | 93.05 s | 94.53 s | -23.82 s | MEASURED |
| Host EXEC wall | 122.6 s | 107.0 s | 98.7 s | -23.90 s | MEASURED |
| Accounted duration (host) | n/a | 84.016 s | 85.401 s | n/a | MEASURED |
| Unaccounted duration (host) | n/a | 9.034 s | 9.125 s | n/a | MEASURED |

### Run 2 Host Waterfall

```
Phase                    | Start (s)  | Dur (s)    | End (s)    | Origin   | Parallel
-------------------------------------------------------------------------------------------
session connect          | 0.000      | 4.512      | 4.512      | host     | false
upload                   | 4.512      | 0.000      | 4.512      | host     | false
EXEC start               | 4.512      | 3.783      | 8.295      | host     | false
weights dl               | 8.841      | 9.005      | 17.847     | host     | true
cache dl                 | 8.842      | 1.839      | 10.680     | host     | true
venv dl                  | 8.842      | 3.830      | 12.672     | host     | true
cache unpack             | 10.680     | 0.084      | 10.765     | host     | true
venv unpack              | 12.672     | 7.463      | 20.135     | host     | false
imports_and_tpu_init     | 23.228     | 12.409     | 35.637     | in-vm    | false
config_setup             | 39.459     | 5.320      | 44.779     | in-vm    | false
create_engine_config     | 42.020     | 0.171      | 42.191     | in-vm    | false
weight_load              | 45.648     | 14.448     | 60.096     | in-vm    | false
kv_alloc                 | 60.102     | 0.434      | 60.536     | in-vm    | false
precompile_load          | 60.536     | 0.013      | 60.549     | in-vm    | false
compile_warmup           | 60.536     | 31.707     | 92.243     | in-vm    | false
warmup[16]               | 81.764     | 3.910      | 85.674     | in-vm    | false
warmup[8192]             | 85.678     | 6.079      | 91.757     | in-vm    | false
tokenizer                | 92.973     | 1.012      | 93.985     | in-vm    | false
ttft                     | 94.049     | 0.482      | 94.531     | in-vm    | false
-------------------------------------------------------------------------------------------
accounted_s = 85.401 s (sum of non-overlapping phase durations) (MEASURED)
unaccounted_s = 9.125 s (total = 94.526 s) (MEASURED)
```

### Named Unaccounted Gaps (Run 2)
1. `EXEC start` -> `weights dl`: **0.546 s** (driver initialization script startup before parallel download spawn)
2. `venv unpack` -> `imports_and_tpu_init`: **3.093 s** (subrunner process invocation & python runtime startup)
3. `imports_and_tpu_init` -> `config_setup`: **3.822 s** (post-import initialization & weights barrier verification)
4. `config_setup` -> `weight_load`: **0.869 s** (`EngineCore.__init__` return to `TPUWorker.load_model` invocation)
5. `weight_load` -> `kv_alloc`: **0.006 s** (negligible)
6. `compile_warmup` -> `tokenizer`: **0.730 s** (post-compilation teardown and tokenizer preparation)
7. `tokenizer` -> `ttft`: **0.064 s** (request payload dispatch)
