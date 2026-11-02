# LFM2.5-24B-A2B on TPU v5e-1: System Architecture

Read this first. Every prompt reads this file before work and adds a change-log line when it changes the architecture.

## 1. Current State (What Runs Today)

### Stack and Versions
- **Model & Checkpoint**: `LiquidAI/LFM2.5-24B-A2B` via `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` (`AUDIT_A_runtime.md:3`, `MEASURED`).
- **Stack**: `vllm v0.32.0` + `tpu-inference v0.32.0` on 1× TPU v5e-1 (`15.748 GiB` / `16,909,336,064 B` HBM limit) (`AUDIT_A_runtime.md:4,74`, `MEASURED`).
- **Execution Mode**: `MODEL_IMPL_TYPE=vllm` (torchax `VllmModelWrapper`) (`AUDIT_A_runtime.md:109,121`, `MEASURED`).
- **Startup Latency**: `85.44 s` (Run 1) / `86.23 s` (Run 2) in-VM cold-start to first token (`94.53 s` host EXEC wall) (`BEARINGS.md:354,357`, `MEASURED`).
- **Compilation Cache Key**: `42f6059b` (`tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len100000__b16-8192__42f6059b.tar`, 92 entries, 196.0 MB, 0 misses, 100% cache hit rate) (`BEARINGS.md:333,348-350`, `MEASURED`).
- **CPU tests**: `./run_cpu_tests.sh` (venv `/home/sahil/Desktop/sahil/colab/.venv-cpu-tests`, pins in `requirements-cpu-tests.txt`)

### Model Architecture and Layers
- **Total Layers**: 40 layers (`Lfm2MoeForCausalLM`): 10 `full_attention` layers + 30 `conv` (`ShortConv`) layers (`AUDIT_A_runtime.md:33`, `MEASURED`).
- **Full Attention Layers (10)**: Indices `[2, 6, 10, 14, 18, 22, 26, 30, 34, 38]` hold paged KV cache (`FullAttentionSpec`) (`AUDIT_A_runtime.md:34,38`, `MEASURED`).
- **ShortConv Layers (30)**: Indices `[0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 29, 31, 32, 33, 35, 36, 37, 39]` hold recurrent state (`MambaSpec`, `conv_L_cache=3`, state shape `(2, 2048)` `bf16`) (`AUDIT_A_runtime.md:35,39`, `MEASURED`).
- **FFN Structure**: Layers 0–1 use dense SwiGLU FFN (`intermediate_size=11776`); layers 2–39 use routed MoE (64 experts, top-4 routing, `moe_intermediate_size=1536`) (`AUDIT_A_runtime.md:36`, `MEASURED`).
- **GQA & Attention Dimensions**: 32 Q heads, 8 KV heads (GQA 4:1), `head_dim=64`, sliding window `None` (`AUDIT_A_runtime.md:40-46`, `MEASURED`).

### Weights by Component
- **Safetensors Shards**: 9 shards = `13,122,818,304 B` (`12.2216 GiB`, JSON headers `2,861,856 B`) across 22,308 tensors (`AUDIT_A_runtime.md:13-14`, `MEASURED`).
- **Loaded TPU HBM**: `13,365,066,752 B` (`12.4472 GiB`, `+0.2256 GiB` net padding/RoPE delta) across 608 tensors (`AUDIT_A_runtime.md:16,27`, `MEASURED`).
- **MoE Re-Quantization on Load**: Unpacked from disk group-128 `int4` (`BF16` scales) to `F32`, then re-quantized per-channel (`block_size=2048/1536`) to `int4` with 4D `F32` scales (`Layout((0,1,2,3))` pads dim-2 `1->8` sublanes = `0.3711 GiB` HBM) (`AUDIT_A_runtime.md:20,107`, `MEASURED`/`INFERRED`).

| Component | Layers / Count | Precision & Shapes | Logical GiB | TPU HBM GiB | Status & Source |
|---|---|---|---:|---:|---|
| MoE Routed Experts | 2..39 (38 layers, 2432 experts) | `int4` `w13`/`w2` per-channel + `F32` scales `(64,1,1,3072/2048)` (padded `1->8` sublanes) | 10.7339 | 11.0586 | `INFERRED` (`AUDIT_A_runtime.md:20`) |
| Shared / Dense MLP + Router | 0..1 dense + 2..39 router/bias | Dense `w13`/`w2` fused `int8` + `F32` scales; `gate.weight` `BF16` | 0.1441 | 0.1442 | `INFERRED` (`AUDIT_A_runtime.md:21`) |
| Attention (`self_attn`) | 10 layers (100 tensors) | `int8` `qkv_proj`/`out_proj` + `0.0954 GiB` RoPE `cos_sin_cache` `(200000, 64)` `F32` | 0.0978 | 0.1932 | `INFERRED` (`AUDIT_A_runtime.md:22`) |
| ShortConv (`conv`) | 30 layers (150 tensors) | `in_proj`/`out_proj` `int8` W8A8 + depthwise `conv.conv` `BF16` `(2048, 1, 3)` | 0.4696 | 0.4700 | `INFERRED` (`AUDIT_A_runtime.md:23`) |
| Embeddings | `model.embed_tokens` (1 tensor) | `BF16` `(128000, 2048)` | 0.4883 | 0.4883 | `MEASURED` (`AUDIT_A_runtime.md:24`) |
| LM Head | `lm_head` (0 tensors) | Tied to `embed_tokens.weight` (`tie_word_embeddings: true`) | 0.0000 | 0.0000 | `MEASURED` (`AUDIT_A_runtime.md:25`) |
| Norms | 81 tensors | `BF16` `(2048,)` | 0.0003 | 0.0003 | `MEASURED` (`AUDIT_A_runtime.md:26`) |
| **Total Loaded Weights** | **608 TPU tensors** | Net +0.2256 GiB over safetensors from scale padding and RoPE cache | **11.9340** | **12.4472** | `MEASURED` (`AUDIT_A_runtime.md:27`) |

### KV Cache and ShortConv State Layout
- **Attention Kernel**: Ragged Paged Attention v3 via `kernel_hd64.py:250` (`attention_backend.py:94-95`, `MEASURED`).
- **KV Dtype & Page Layout**: `bfloat16`, block size 16 tokens/page, shape `(B, 16, 4, 2, 128)` (`32,768 B/block/layer`) (`AUDIT_A_runtime.md:43-45`, `MEASURED`).
- **Attention KV Byte Rate**: $10 \times 2 \times 8 \times 64 \times 2\text{ B} = \mathbf{2,048\text{ B/tok/layer}} = \mathbf{20,480\text{ B/tok}}$ across 10 layers (0% padding waste in `bf16`) (`AUDIT_A_runtime.md:53-54`, `MEASURED`).
- **ShortConv State Layout**: 9 slots/layer (`shape=(9, 2, 2048)` `bf16` = `73,728 B/layer`, `2,211,840 B` = `0.00206 GiB` across 30 layers, `0 B/tok` marginal) for `max_num_reqs = 8` + 1 null slot (`AUDIT_A_runtime.md:45,57`, `MEASURED`).

### Memory Budget at 100K (`job10b_warm_2.log`, `util=0.92`, cap `14.488 GiB`, limit `15.748 GiB`)

| HBM Bucket | Pre-Compile at KV Alloc | Post-Compile Engine Ready | Post-Compile Peak (16-tok) | Exact Bytes at KV Alloc | Status & Source |
|---|---:|---:|---:|---:|---|
| 1. `weights` | 12.447 GiB | 12.447 GiB | 12.447 GiB | 13,365,066,752 B | `MEASURED` (`AUDIT_A_runtime.md:76`) |
| 2. `KV` (6681 attn + 9 mamba) | 2.041 GiB | 2.041 GiB | 2.041 GiB | 2,191,441,920 B | `MEASURED` (`AUDIT_A_runtime.md:77`) |
| 3. `activations` | 0.000 GiB | 0.000 GiB | 0.051 GiB (peak delta) | 0 B | `MEASURED` (`AUDIT_A_runtime.md:78`) |
| 4. `compile/scratch` | 0.004 GiB | 0.251 GiB (persistent) | 0.251 GiB (persistent) | 4,255,744 B | `MEASURED` (`AUDIT_A_runtime.md:79`) |
| 5. `vLLM reserve` (`1 - 0.92`) | 1.260 GiB (free 1.256 GiB) | 1.009 GiB (free idle) | 0.958 GiB (free peak) | 1,352,746,886 B | `MEASURED` (`AUDIT_A_runtime.md:80`) |
| 6. `other` | 0.000 GiB | 0.000 GiB | 0.000 GiB | 0 B | `MEASURED` (`AUDIT_A_runtime.md:81`) |
| **Total HBM Used / Limit** | **14.492 / 15.748 GiB** | **14.739 / 15.748 GiB** | **14.790 / 15.748 GiB** | **16,909,336,064 B** | `MEASURED` (`AUDIT_A_runtime.md:74-82`) |

### Active Patches in `tpu_v5e_patches.py`
1. **P2 ShortConv Slicing**:
   - Replaces `ShortConv.forward` with pure-Torch/XLA `_tpu_xla_short_conv_forward` (`sd_causal_conv`).
   - Slices recurrent state at actual token boundary `n = clamp(qsl_t[1] - qsl_t[0], 0, T)` instead of padded bucket length $T \in \{16, 8192\}$ (`tpu_v5e_patches.py:71-90,154-181`, `AUDIT_A_runtime.md:112`, `MEASURED`).
2. **Token Bucket Pruning**:
   - Wraps `get_token_paddings` across `tpu_inference` to prune `[16, 32, ..., 8192]` down to `[16, 8192]`.
   - Cuts XLA backbone compilations from 10 to 2 (`tpu_v5e_patches.py:185-208`, `AUDIT_A_runtime.md:113`, `MEASURED`).
3. **MoE General Device Put**:
   - Buffers 64 experts per tensor group and calls `jax.device_put` once on the stacked tensor.
   - Eliminates per-expert host dispatch overhead during weight sharding (`tpu_v5e_patches.py:468-527`, `AUDIT_A_runtime.md:114`, `MEASURED`).
4. **Disable Structured Decoding**:
   - No-ops `_precompile_structured_decoding` on `TPUModelRunner` and `CompilationManager`.
   - Skips unused XLA precompilations (`tpu_v5e_patches.py:563-566`, `AUDIT_A_runtime.md:115`, `MEASURED`).
5. **KV Validation Bypass**:
   - Patches `supports_kv_cache_dtype`, `static_validate_inputs`, and `get_tuned_block_sizes`.
   - Permits experimental `fp8`/`int8` KV formats without engine validation crashes (`tpu_v5e_patches.py:538-562`, `AUDIT_A_runtime.md:116`, `MEASURED`).
6. **In-Process Registry Inspection**:
   - `LFM_INPROC_INSPECT=1` executes model architecture inspection in-process.
   - Skips `_run_in_subprocess`, cutting 11.5 s of subprocess spawn overhead (`tpu_v5e_patches.py:570-596`, `AUDIT_A_runtime.md:117`, `MEASURED`).
7. **$O(\text{new modules})$ Import Hook**:
   - `LFM_PATCH_FAST=1` filters seen modules in `custom_import`.
   - Prevents scanning `sys.modules` on every import, saving 118.3 s during model loading (`tpu_v5e_patches.py:599-635`, `AUDIT_A_runtime.md:118`, `MEASURED`).
8. **GCE Metadata Blackhole**:
   - Rejects loopback metadata queries via `/etc/hosts` and `urllib.request.urlopen` hooks.
   - Eliminates 10 s metadata poll timeouts on Google Colab (`lfm25_coldstart_driver.py:56-59`, `AUDIT_A_runtime.md:119`, `MEASURED`).
9. **Wrapper Delegation & Log Filter**:
   - Adds `.modules()` on `VllmModelWrapper` to fix atexit `AttributeError`.
   - Filters `"Should not schedule a request that does nothing!"` log spam (`tpu_v5e_patches.py:210-221,313-315`, `AUDIT_A_runtime.md:120`, `MEASURED`).
10. **Runtime & XLA Cache Envs**:
    - Configures `/dev/shm/tpu_compile_cache` and persistent JAX cache flags.
    - Sets strict `LFM_ABORT_ON_CACHE_MISS=1` to prevent accidental cold compile regressions (`run_job10b.py:43-47`, `AUDIT_A_runtime.md:121`, `MEASURED`).

## 2. Known Problems and Shortcomings

- **Inflated Admission Check (P1)**:
  - Root Cause: `kv_cache_manager.py:277-291` pads every layer spec to `uniform_page_size_bytes = 1 \times 32,768 + 3 \times 8,192 = 57,344\text{ B}`.
  - Formula: Across 10-layer pool, `_pool_bytes_per_block = 573,440\text{ B}`. For 200K (`12,500` attn + `3` mamba blocks), demanded memory is $12,503 \times 573,440\text{ B} = \mathbf{6.6773\text{ GiB}}$ (`6.68 GiB`) (`AUDIT_A_runtime.md:88-95`, `AUDIT_C_verdict.md:55-59`, `MEASURED`).
  - Phantom Overhead: `2.8626 GiB` (`3,073,720,320 B`) is phantom ShortConv padding multiplied by `12,500` blocks (`AUDIT_C_verdict.md:59`, `MEASURED`).
  - Available Check: Checked against `check_memory = (6681 - 1) × 573,440 B = 3.5675 GiB` (`3.57 GiB`), which is synthetically inflated; true physical available KV HBM at `util=0.92` is `2.0410 GiB` (`AUDIT_A_runtime.md:92-93,99`, `AUDIT_C_verdict.md:45,62`, `MEASURED`).
- **`bf16` KV Cannot Fit 200K**:
  - 200K `bf16` KV physically requires `3.8168 GiB` (`3.8147 GiB` attn + `0.0021 GiB` conv) (`AUDIT_A_runtime.md:62`, `MEASURED`/`INFERRED`).
  - Combined with `12.447 GiB` weights, `0.251 GiB` compile scratch, and `0.051 GiB` activations (`0.083 GiB` max), peak HBM reaches `16.566 GiB` (`16.598 GiB` max) (`AUDIT_C_verdict.md:89`, `MEASURED`/`INFERRED`).
  - Deficit: Exceeds total chip HBM (`15.748 GiB`) by `0.818 GiB`, causing OOM even at `util=1.00` (`AUDIT_A_runtime.md:100`, `AUDIT_C_verdict.md:92-93`, `MEASURED`/`INFERRED`).
- **P2 Exactness Verified on CPU (`test_int8_kv_cpu.py`)**:
  - Patched `_tpu_xla_short_conv_forward` on a 2-chunk prompt (`48 + 16 = 64` tokens, each chunk padded to `T = 64`) matches plain NumPy full-sequence causal conv with `max_abs_diff = 0.0` (`< 1e-3` threshold, `MEASURED` on CPU; TPU 200K `NOT TESTED`).
- **Compile Numbers Only Measured at 100K**:
  - Post-KV compile scratch (`0.251 GiB`) and peak activation delta (`0.051–0.083 GiB`) were measured on 100K init + 16-token request (`AUDIT_A_runtime.md:78-80`, `MEASURED`).
  - Peak memory across `25 × 8192` consecutive prefill chunks at 200K is unmeasured (`AUDIT_C_verdict.md:110`, `NOT TESTED`).
- **`int8` K/V Scales Unknown**:
  - Checkpoint lacks KV scales (`_k_scale_float = 1.0`); running with default scale `1.0` clips unscaled activations and degrades output quality (`AUDIT_C_verdict.md:79,101,111`, `NOT TESTED`).
- **Cache Key Sensitivity (`tpu_v5e_patches.py`)**:
  - The cache key hashes the raw text of `tpu_v5e_patches.py`; never edit it without planning a cold rebuild.
  - int8 flag is part of the cache key; OFF key = `42f6059b`.

## 3. Refuted Claims (Never Reuse)

- **"~0.51 GiB short"** (`BEARINGS.md:98`):
  - Refuted by `AUDIT_A_runtime.md:96-100` and `AUDIT_C_verdict.md:91`.
  - Origin: Compared 200K `bf16` KV (`3.8168 GiB`) against unreserved free HBM at `util=1.00` ($3.8168 - 3.3009 = 0.5159\text{ GiB}$), ignoring `0.92` utilization and compile/activation overhead.
  - Reality: At `util=0.92`, pre-compile shortfall is `1.7757 GiB`; with `0.302 GiB` compile/peak activations, true chip shortfall is `0.818 GiB` (`MEASURED`/`INFERRED`).
- **"head_dim pad wastes 2x in bf16"** (`AUDIT_B_research.md:42,81,207`):
  - Refuted by `AUDIT_A_runtime.md:42,54,109` and `AUDIT_C_verdict.md:43,73-75`.
  - Reality: In `tpu-inference v0.32.0`, `head_dim=64` dispatches to `kernel_hd64.py`, which packs K+V into 128 with `kv_packing=2` (`(B, 16, 4, 2, 128)`), allocating `32,768 B/block/layer` with exactly 0% padding waste in `bf16` (`MEASURED`).
- **"3.57 GiB available"** (`AUDIT_B_research.md:214,232-233`):
  - Refuted by `AUDIT_A_runtime.md:93,99` and `AUDIT_C_verdict.md:45,62`.
  - Reality: `3.5675 GiB` is synthetically inflated by multiplying blocks by `573,440 B`; physical available KV HBM at `util=0.92` is `2.0410 GiB` (`MEASURED`).
- **"Research activation budget 0.88–1.65 GiB"** (`AUDIT_B_research.md:160,287`):
  - Refuted by `AUDIT_A_runtime.md:78-80` and `AUDIT_C_verdict.md:49`.
  - Reality: Persistent compile scratch is `0.251 GiB` and peak activation delta is `0.051 GiB` (`0.083 GiB` max), totaling `0.302–0.334 GiB` (`MEASURED`).
- **"ShortConv 1 slot"** (`AUDIT_B_research.md:51,159,286`):
  - Refuted by `AUDIT_A_runtime.md:45,57` and `AUDIT_C_verdict.md:44`.
  - Reality: TPU worker allocates `max(dp*max_num_seqs, 8) + 1 = 9` slots per layer (`2,211,840 B` = `0.00206 GiB`), not 1 slot (`0.00023 GiB`) (`MEASURED`).

## 4. Target for 200K (All NOT TESTED)

### Target Specifications
- **P1 Admission Fix**: Patch `_max_memory_usage_bytes_from_groups` (`kv_cache_utils.py:2337-2366`) and `update_mamba_page_size_padded` (`kv_cache_manager.py:264-291`) so attention checks at unpadded `attn_page_size_bytes` (`16,384 B` `int8`), eliminating `2.8626 GiB` phantom demand (`AUDIT_C_verdict.md:60-62,100`, `NOT TESTED`).
- **P3 `int8` KV**: Map `kv_cache_dtype` to `jnp.int8` in `kernel_hd64.py` (`(B, 16, 2, 4, 128)`, `16,384 B/block/layer`, `1.9073 GiB` attn + `0.0021 GiB` conv = `1.9094 GiB` total KV at 200K), cast `kv.astype` before matmuls, and pass calibrated scales (`kv_int8.py:1-92`, `AUDIT_C_verdict.md:76-79,86`, patch written, CPU-tested, TPU `NOT TESTED`).
- **Utilization**: Set `gpu_memory_utilization = 0.94` (`cap = 14.803 GiB`), adding `+0.315 GiB` KV headroom while retaining `0.945 GiB` reserve (`AUDIT_C_verdict.md:90,102`, `NOT TESTED`).
- **Target Peak HBM**: `14.658 GiB` (`12.447 GiB` weights + `1.909 GiB` KV + `0.251 GiB` scratch + `0.051 GiB` activations), providing `+1.090 GiB` headroom under `15.748 GiB` limit (`AUDIT_C_verdict.md:89,92`, `NOT TESTED`).

### Top 3 Architecture Changes (Audit C §8)

| Rank | Change | GiB Saved at 200K | Accuracy Risk | Implementation Effort |
|---:|---|---|---|---|
| **1** | **P1: Fix Hybrid `MambaSpec` Admission Check** (`kv_cache_manager.py:277`, `kv_cache_utils.py:2354`) | **`2.863 GiB`** (virtual admission) | **None** (`0%`, admission math only) | **Patch** (~15 lines in `tpu_v5e_patches.py`) |
| **2** | **P3: Enable `int8` KV in `kernel_hd64.py`** (`(B, 16, 2, 4, 128)`) | **`1.907 GiB`** (physical HBM) | **Low** (`<1%` PPL with calibrated scale) | **Patch + minor kernel work** (~30 lines) |
| **3** | **Raise `gpu_memory_utilization` `0.92 -> 0.94`** | **`+0.315 GiB`** (KV capacity) | **None** | **Config** (`run_job10b.py` argument) |

### 200K Memory Options (Audit C §7)

| Bucket (GiB = $2^{30}$ B; Limit = 15.748 GiB) | (a) P1 only, bf16 KV (12,500 blk), util=0.97 | (b) P1 + P3 (int8 KV in kernel_hd64), util=0.92 | (c) B Best Config (int8 KV + bf16 scales), util=0.92 |
|---|---:|---:|---:|
| 1. `weights` (`MEASURED`, A) | 12.447 GiB | 12.447 GiB | 12.447 GiB |
| 2. `KV` @ 200K (12,500 attn + 9 mamba) | 3.817 GiB (3.8147 + 0.0021) | 1.909 GiB (1.9073 + 0.0021) | 1.969 GiB (1.9073 + 0.0596 + 0.0021) |
| 3. `compile/scratch` (persistent) (`MEASURED`, A) | 0.251 GiB | 0.251 GiB | 0.251 GiB |
| 4. `activations` (peak delta) (`MEASURED`, A) | 0.051 GiB (0.083 max) | 0.051 GiB (0.083 max) | 0.051 GiB (0.083 max) |
| **Subtotal Used at Peak (1 + 2 + 3 + 4)** | **16.566 GiB** (16.315 pre-compile) | **14.658 GiB** (14.356 pre-compile) | **14.718 GiB** (14.416 pre-compile) |
| 5. `vLLM reserve` ((1 - util) × 15.748) | 0.472 GiB (0.000 at util=1.00) | 1.260 GiB (cap 14.488 GiB) | 1.260 GiB (cap 14.488 GiB) |
| Pre-Compile KV Margin (cap - weights - KV) | -1.037 GiB (-0.516 at util=1.00) | +0.132 GiB (~270 spare blocks) | +0.072 GiB (~143 spare blocks) |
| Post-Compile Peak Headroom (15.748 - Peak) | -0.818 GiB (OOM!) | +1.090 GiB (+1.058 at 0.083 act) | +1.030 GiB (+0.998 at 0.083 act) |
| **VERDICT** | **DOESN'T FIT** (even at util=1.00) | **FITS** (14.658 / 15.748 GiB) | **FITS** (14.718 / 15.748 GiB) |

## 5. Plan, in Order: Steps 1–5

- **Step 1: Int8 KV + P2 Check at 16K**: Run `max_model_len=16384` on a `9,000`-token needle prompt (`1 × 8192` + `1 × 808` + `16` decode) with `LFM_CONV_DEBUG=1`; pass test: exact needle recall, `CONVDBG` (`n=8192 -> 808 -> 1`), and logged `k.abs().max() / 127.0`, `v.abs().max() / 127.0` across layers `2..38` (`AUDIT_C_verdict.md:105,111`, `NOT TESTED`).
- **Step 2: P1 CPU Test (0 CU)**: Patch `_max_memory_usage_bytes_from_groups` / `update_mamba_page_size_padded` in `tpu_v5e_patches.py`; pass test: local CPU unit test calling `get_kv_cache_configs` at `max_model_len=200000` asserts `needed_memory == 1.9073 GiB` (`int8`) instead of `6.6773 GiB` (`AUDIT_C_verdict.md:106`, `NOT TESTED`).
- **Step 3: 200K Alloc-Stop (`LFM_BEARINGS_STOP=1`)**: Run `max_model_len=200000`, `util=0.94`, `LFM_BEARINGS=1`, `LFM_BEARINGS_STOP=1` (`tpu_v5e_patches.py:302`); pass test: exits right after `initialize_kv_cache` (~37 s in-VM) with `num_blocks >= 12501` of `(N, 16, 2, 4, 128)` `int8` (`1.909 GiB` KV) and `used <= 14.67 GiB` (`AUDIT_C_verdict.md:107`, `NOT TESTED`).
- **Step 4: 200K XLA Warmup & Cache Tar Publish**: Compile `[16, 8192]` at `max_model_len=200000` and publish `len200000` cache tar; pass test: warm restore completes with 0 persistent cache misses (`AUDIT_C_verdict.md:108`, `NOT TESTED`).
- **Step 5: Full 200K Run**: Run `199,900`-token single-request benchmark; pass test: completes 25-chunk prefill + decode and logs `peak_bytes_in_use <= 15.748 GiB` (`AUDIT_C_verdict.md:108,110`, `NOT TESTED`).

## 6. Change Log

- `2026-11-02` **Job 9**: Profiled `config_setup` (11.5 s subprocess in `registry._run_in_subprocess`) and pre-warmup gap (21.9 s JAX/Mosaic lowering); published cache tar `5a0d66f4` (`BEARINGS.md:291-312`, `MEASURED`, hash: `c3be5a0`).
- `2026-11-02` **Job 10**: Implemented in-process registry inspect (`LFM_INPROC_INSPECT=1`) and fast module patch hook (`LFM_PATCH_FAST=1`); cold compile published cache tar `42f6059b` (`BEARINGS.md:314-329`, `MEASURED`, hash: `7733fc2`).
- `2026-11-02` **Job 10b**: Fixed waterfall clock and added early cache miss abort; verified warm restore (`85.44 s` in-VM to first token, 0 misses) on cache tar `42f6059b` (`BEARINGS.md:330-397`, `MEASURED`, hash: `5dab708`).
- `2026-11-02` **Audit A**: Audited live runtime measurements from 100K warm run (`job10b_warm_2.log`, cache `42f6059b`) and established 6-bucket HBM accounting (`AUDIT_A_runtime.md:1-121`, `MEASURED`, hash: `955ca4c`).
- `2026-11-02` **Audit B**: Audited all 6 research document groups (G1–G6), verified formulas, and cataloged calculation errors and stale assumptions (`AUDIT_B_research.md:1-289`, `INFERRED`, hash: `955ca4c`).
- `2026-11-02` **Audit C**: Adjudicated runtime vs research, proved P1 ShortConv overcount (`2.8626 GiB`) and P3 `int8` feasibility (`1.909 GiB`), and defined 200K target budget (`AUDIT_C_verdict.md:1-111`, `MEASURED`/`INFERRED`, hash: `064a438`).
- `2026-11-02 Created ARCHITECTURE.md` (hash: `4e5fd46`)
- `2026-11-02` **Job 2a**: Added `LFM_KV_INT8=1` patch (default OFF) in `tpu_v5e_patches.py` + `tests/test_int8_kv_cpu.py`; verified `int8` KV shape (`16,384 B/block`), round-trip error `<= scale/2`, CPU ref attention (`max_abs=0.031250`, `mean_rel=0.063829`; Pallas needs TPU), and P2 2-chunk ShortConv exactness (`max_abs_diff=0.0`) (hash: `bc59605`).
- `2026-11-02` **Job 2a-fix**: int8 moved to `kv_int8.py`, outside the cache hash (`tpu_v5e_patches.py` restored to `5dab708`, 100K cache key `42f6059b` restored) (hash: `635b063`).
- `2026-11-02` **Job 2b**: Wired `kv_int8.apply()` in `lfm25_engine_subrunner.py:13` and made `lfm25_cache.cache_filename` int8-aware (`LFM_KV_INT8=1` appends `__kvint8` + `kv_int8.py` hash; OFF key = `42f6059b`) (hash: `1abc812`).
- `2026-11-02` **Job 2c**: Added `./run_cpu_tests.sh` wrapper for single-command CPU test collection and execution.
- `2026-11-02` **Job 2d**: Pinned CPU test venv + runner; pins in `requirements-cpu-tests.txt`.

