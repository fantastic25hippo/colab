# SESSION A: LFM2.5-24B-A2B Runtime Audit (AS IT RUNS)

- **Model / Repo**: `LiquidAI/LFM2.5-24B-A2B` via `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` (`lfm25_coldstart_driver.py:121`, `MEASURED`).
- **Stack**: `vllm v0.32.0` + `tpu-inference v0.32.0` on 1× TPU v5e-1 (`BEARINGS.md:4-5`, `job10b_warm_2.log:37,55`, `MEASURED`).
- **Standard HBM Buckets (GiB = $2^{30}$ B, per chip, at 100K warm run `job10b_warm_2.log`)**:
  `weights = 12.447 GiB` | `KV = 2.041 GiB` | `activations/prefill workspace = 0.000 GiB (at KV check) / 0.247–0.298 GiB (post-compile & 8192-tok runtime)` | `compile/scratch = 0.247 GiB (persistent post-warmup) / 0.298 GiB (peak)` | `vLLM reserve (1 - 0.92) = 1.260 GiB` | `other = 0.000 GiB` (Total chip HBM = `15.748 GiB`, `MEASURED`).

---

## 1. Weights: HF Safetensors Headers vs TPU HBM Loaded (`MEASURED` & `INFERRED`)

- **Disk / HF Safetensors (`MEASURED`, `HfApi.repo_info` & `get_safetensors_metadata`, `token=False`)**:
  - 9 `.safetensors` shards = `13,125,680,160 B` = **`12.2242 GiB`** (`13125.7 MB` in `job10b_warm_2.log:24,299`; logged as `12.22 GiB` at `job10b_warm_2.log:77`).
  - Raw tensor payload across `22,308` tensors = `13,122,818,304 B` = **`12.2216 GiB`** (JSON headers = `2,861,856 B` = `0.0027 GiB`).
- **TPU HBM After `load_weights` (`MEASURED`, `job10b_warm_2.log:108,110,112,119`)**:
  - `608` tensors moved via `shard_model_to_tpu` (`0.1646 s`). After `jax.clear_caches()` (`tpu_runner.py:1333`), HBM used = `13,365,066,752 B` = **`12.4472 GiB`** (`12.447 GiB` in `[LFM_BEARINGS]`, `12.45 GiB` in `tpu_worker.py:558`).

| Component | Layers / Count | HF Header Dtype & Shape (`MEASURED`) | HF GiB (`MEASURED`) | Loaded TPU Dtype, Shape & Re-Quant (`MEASURED` / `INFERRED`) | TPU Logical / Physical GiB |
|---|---|---|---:|---|---:|
| **MoE Routed Experts** (`w1,w2,w3`) | `layers.2..39` (`38 × 64 = 2,432` experts, `21,888` tensors) | `weight_packed`: `I32` (8× sym `int4`, `(1536,256)`, `(2048,192)`) = `10.6875 GiB`<br>`weight_scale`: `BF16` (group-128, `(1536,16)`, `(2048,12)`) = `0.3340 GiB`<br>`weight_shape`: `I32` `(2,)` = `0.0001 GiB` | **`11.0215 GiB`** (`11,834,287,104 B`) | **RE-QUANTIZED ON LOAD** (`moe_weights.py:899-933,1111`, `job10b_warm_2.log:101-105`): unpacked to `int4`, dequantized to `F32`, re-quantized **per-channel** (`block_size=2048/1536`) to `int4` `w13_weight` `(64,2048,3072)` & `w2_weight` `(64,1536,2048)` (`10.6875 GiB`) + `F32` scales `(64,1,1,3072)` & `(64,1,1,2048)` (`0.0464 GiB` logical; `Layout((0,1,2,3))` pads dim-2 `1->8` sublanes = `0.3711 GiB` HBM). | **`10.7339 GiB`** (logical) / **`11.0586 GiB`** (HBM `INFERRED`) |
| **Shared / Dense MLP + Router** | `layers.0..1` dense (`12` tensors) + `layers.2..39` gate/bias (`76` tensors) | Dense `w1,w2,w3.weight`: `I8` (`(11776,2048)`, `(2048,11776)`) = `0.1348 GiB`; `weight_scale`: `BF16` = `0.0001 GiB`.<br>Router `gate.weight`: `BF16` `(64,2048)` = `0.0093 GiB`; `expert_bias`: `BF16` `(64,)` = `0.000005 GiB` | **`0.1441 GiB`** (`154,772,224 B`) | Dense `w1+w3` fused to `w13` `int8` (`(2048,23552)`), `w2` `int8` (`(11776,2048)`); scales cast `BF16 -> F32` (`compressed_tensors_w8a8_int8.py:85`); `gate.weight` `BF16`; `e_score_correction_bias` cast `BF16 -> F32` (`lfm2_moe.py:135`). | **`0.1442 GiB`** (`154,879,488 B` `INFERRED`) |
| **Attention (`self_attn`)** | `10` layers (`2,6,10,14,18,22,26,30,34,38`, `100` tensors) | `q,k,v,out_proj.weight`: `I8` (`(2048,2048)`, `(512,2048)`) = `0.0977 GiB`; `weight_scale`: `BF16` = `0.0001 GiB`.<br>`q/k_layernorm.weight`: `BF16` `(64,)` = `0.000002 GiB` | **`0.0978 GiB`** (`104,962,560 B`) | `q,k,v` fused to `qkv_proj` `int8` `(2048,3072)`, `out_proj` `int8` `(2048,2048)`; scales cast `BF16 -> F32`; `q/k_layernorm` `BF16`. Plus runtime RoPE `cos_sin_cache` `(200000,64)` `F32` (`0.0477 GiB` logical, padded `64->128` lanes in `{1,0}` layout = `0.0954 GiB` HBM, `vllm_model_wrapper.py:81-104`). | **`0.0978 GiB`** (+`0.0954 GiB` RoPE HBM `INFERRED`) |
| **ShortConv (`conv`)** | `30` layers (`150` tensors) | `in_proj.weight` `(6144,2048)` & `out_proj.weight` `(2048,2048)`: `I8` = `0.4688 GiB`; `weight_scale`: `BF16` = `0.0005 GiB`.<br>`conv.conv.weight`: `BF16` `(2048,1,3)` = `0.0003 GiB` | **`0.4696 GiB`** (`504,176,640 B`) | `in_proj` `(2048,6144)` & `out_proj` `(2048,2048)` `int8` (W8A8 dynamic token act quant); `weight_scale` cast `BF16 -> F32`; depthwise `conv.conv.weight` `BF16` `(2048,1,3)`. | **`0.4700 GiB`** (`504,668,160 B` `INFERRED`) |
| **Embeddings** | `model.embed_tokens` (`1` tensor) | `BF16` `(128000, 2048)` (unquantized; in `ignore` list) | **`0.4883 GiB`** (`524,288,000 B`) | `BF16` `(128000, 2048)` (`MEASURED`). | **`0.4883 GiB`** (`MEASURED`) |
| **LM Head** | `lm_head` (`0` tensors) | Tied (`tie_word_embeddings: true` in `config.json`) | **`0.0000 GiB`** (`0 B`) | Tied to `embed_tokens.weight` (`lfm2_moe.py:611`, `MEASURED`). | **`0.0000 GiB`** (`MEASURED`) |
| **Norms** | `40×operator_norm`, `40×ffn_norm`, `1×embedding_norm` (`81` tensors) | `BF16` `(2048,)` each (`4,096 B` × 81; `q/k_layernorm` `2,560 B` in Attention above) | **`0.0003 GiB`** (`331,776 B`) | `BF16` `(2048,)` (`MEASURED`). | **`0.0003 GiB`** (`MEASURED`) |
| **Total** | **`22,308` HF -> `608` TPU** | **Safetensors payload** | **`12.2216 GiB`** | **Net `+0.2256 GiB` (`+242,248,448 B`) on TPU**: `-0.2877 GiB` (group-128 -> per-channel MoE scale reduction) `+0.3247 GiB` (MoE scale 8-sublane padding) `+0.0954 GiB` (RoPE `{1,0}` cache) `+0.0932 GiB` (W8A8 scale `F32` + `(8,128)` tile alignment) (`INFERRED`). | **`12.4472 GiB`** (`13,365,066,752 B` `MEASURED`) |

---

## 2. Architecture (`MEASURED`, `config.json`, `config_builder.py:68-108`, `job10b_warm_2.log:114-185`)

- **Layers (`40` total, `Lfm2MoeForCausalLM`, `vllm/model_executor/models/lfm2_moe.py:454-478`)**:
  - **10 `full_attention` layers**: indices `[2, 6, 10, 14, 18, 22, 26, 30, 34, 38]` (every $4\text{th}$ layer starting at 2).
  - **30 `conv` (`ShortConv`) layers**: indices `[0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 23, 24, 25, 27, 28, 29, 31, 32, 33, 35, 36, 37, 39]`.
  - **FFN per layer**: Layers `0, 1` use dense SwiGLU MLP (`intermediate_size=11776`); layers `2..39` (38 layers) use routed MoE (`64` experts, top-`4`, `moe_intermediate_size=1536`).
- **Which layers hold KV / state (`job10b_warm_2.log:144-184`, `MEASURED`)**:
  - **10 `full_attention` layers** hold paged KV cache (`FullAttentionSpec`).
  - **30 `conv` (`ShortConv`) layers** hold recurrent conv state (`MambaSpec`, `conv_L_cache=3` -> state length $L - 1 = 2$, dim `2048`).
- **Attention Heads & Dimensions (`config.json`, `BEARINGS.md:29-48`, `job10b_warm_2.log:144,195`, `MEASURED`)**:
  - `num_attention_heads = 32`, **`num_key_value_heads = 8`** (GQA 4:1).
  - **Real `head_dim = 64`** vs **Padded `head_dim = 128`** on TPU: `kernel_hd64.py:250` and `kernel.py:274` (`align_to(actual_head_dim, 128)`) pad the last dimension of the combined K+V tensor from `64` to `128` (`2×` memory expansion).
- **KV Dtype, Block/Page Size, Sliding Window (`job10b_warm_2.log:114-116,144,195`, `MEASURED`)**:
  - **KV dtype**: `bfloat16` (`2 B/elem`, resolved from `kv_cache_dtype="auto"`, `kv_cache_manager.py:1002`).
  - **Block / page size**: Attention `block_size = 16` tokens/page (`32,768 B/page/layer` padded, `16,384 B/page/layer` unpadded). ShortConv state slot = `1` slot/request (`shape=(2, 2048)` `bfloat16` = `8,192 B/slot/layer`, `9` slots allocated per layer for `max_num_reqs=8` + 1 null slot).
  - **Sliding window**: **`None`** (`sliding_window` absent in `config.json`; logged as `max_model_len-131072-sw-None` at `job10b_warm_2.log:195`).

---

## 3. KV Cache Math: Bytes/Token & GiB at 100K vs 200K (`MEASURED` & `INFERRED`)

- **Bytes/Token Formula (10 `full_attention` layers, $L_{\text{attn}}=10$, $H_{kv}=8$, `bfloat16` = $2\text{ B}$)**:
  $$\text{Bytes/token (attn)} = L_{\text{attn}} \times 2_{K,V} \times H_{kv} \times D_{\text{head}} \times \text{sizeof(bf16)} = 10 \times 2 \times 8 \times D_{\text{head}} \times 2 = 320 \times D_{\text{head}}\text{ B/tok}$$
  - **Unpadded $D_{\text{head}}=64$ (`kernel_hd64.py:250` packs K+V into last dim `128` with head dim `4×2=8`, shape `(B, 16, 4, 2, 128)`)**: $16 \times 4 \times 2 \times 128 \times 2\text{ B} = 32,768\text{ B/block/layer} \implies \mathbf{2,048\text{ B/tok/layer}} = \mathbf{20,480\text{ B/tok}}$ across 10 layers (`MEASURED`, `job10b_warm_2.log:115,175-184`).
  - **Naively Padded $D_{\text{head}}=64 \to 128$ (`kernel.py:261-275`, shape `(B, 16, 8, 2, 128)` without K+V head-dim folding)**: $16 \times 8 \times 2 \times 128 \times 2\text{ B} = 65,536\text{ B/block/layer} \implies \mathbf{4,096\text{ B/tok/layer}} = \mathbf{40,960\text{ B/tok}}$ across 10 layers (`INFERRED` from `BEARINGS.md:29-46`).
  - **vLLM Hybrid Spec Padding (`kv_cache_manager.py:280`, `job10b_warm_2.log:115`)**: Pads every layer spec to `1 × 32,768 + 3 × 8,192 = 57,344 B/block` (`573,440 B/block` per 10-layer group $\div 16\text{ tok/block} = \mathbf{35,840\text{ B/tok}}$, plus `3` blocks for the 3 Mamba groups).
  - **ShortConv (`MambaSpec`) state (`job10b_warm_2.log:116,145-174`)**: Fixed $30\text{ layers} \times 9\text{ slots} \times 8,192\text{ B} = \mathbf{2,211,840\text{ B}} = \mathbf{0.002060\text{ GiB}}$ (`2.109 MiB`, `0 B/tok` marginal).

| Mode / Metric | Bytes / Token | 100K (`6,250` blocks) GiB | 100K Allocated (`6,681` blocks) GiB | 200K (`12,500` blocks) GiB | Status & Source |
|---|---:|---:|---:|---:|---|
| **1. Unpadded Attention KV (`D=64`, `kernel_hd64` `(B,16,4,2,128)`)** | `20,480 B/tok` | `1.9073 GiB` (`2,048,000,000 B`) | `2.0389 GiB` (`2,189,230,080 B`) | `3.8147 GiB` (`4,096,000,000 B`) | `MEASURED` (`job10b_warm_2.log:175-184`) |
| **2. Actual TPU Allocation (Compact-Mamba: #1 + `2.11 MiB` conv)** | `20,480 B/tok` (+`2.11 MiB`) | `1.9094 GiB` (`2,050,211,840 B`) | **`2.0409 GiB`** (`2,191,441,920 B`) | **`3.8168 GiB`** (`4,098,211,840 B`) | `MEASURED` at 100K (`job10b_warm_2.log:185`), `INFERRED` at 200K |
| **3. Padded `head_dim` $64 \to 128$ (`kernel.py` `(B,16,8,2,128)` + conv)** | `40,960 B/tok` (+`2.11 MiB`) | `3.8168 GiB` (`4,098,211,840 B`) | `4.0799 GiB` (`4,380,672,000 B`) | `7.6315 GiB` (`8,194,211,840 B`) | `INFERRED` (`BEARINGS.md:29-46`) |
| **4. vLLM Hybrid-Padded Check (`573,440 B/block` + 3 mamba blocks)** | `35,840 B/tok` (+`1.64 MiB`) | `3.3395 GiB` (`6,253` blocks) | `3.5680 GiB` (`6,681` blocks; `3.5675 GiB` usable) | **`6.6773 GiB`** (`12,503` blocks, logged **`6.68 GiB`**) | `MEASURED` (`job10b_warm_2.log:115,121`, `run_results_run1.json:100`) |

---

## 4. HBM Budget at 100K (`job10b_warm_2.log`) & Reconciliation (`MEASURED`)

- **Config (`job10b_warm_2.log:41,55,119`)**: `max_model_len=100000`, `max_num_seqs=1`, `max_num_batched_tokens=8192`, `gpu_memory_utilization=0.92`.

| Bucket | Pre-Compile / At KV Alloc (`job10b_warm_2.log:110,119,185-186`) | Post-Compile / Engine Ready & Short Req (`job10b_warm_2.log:243,299`) | Exact Bytes (At KV Alloc) | Status & Citation |
|---|---:|---:|---:|---|
| **Total HBM (`limit`)** | **`15.748 GiB`** (logged `15.75 GiB`) | **`15.748 GiB`** | `16,909,336,064 B` | `MEASURED` (`job10b_warm_2.log:71,110,119,243`: `13365066752 + 3544269312`) |
| **vLLM Cap (`0.92 × limit`)** | **`14.488 GiB`** (logged `14.49 GiB`) | **`14.488 GiB`** (`in_use` exceeds cap by `0.251 GiB` post-compile) | `15,556,589,178 B` | `MEASURED` (`tpu_worker.py:520,558`, `job10b_warm_2.log:119`) |
| **1. `weights`** | **`12.447 GiB`** (logged `12.45 GiB`) | **`12.447 GiB`** | `13,365,066,752 B` | `MEASURED` (`job10b_warm_2.log:110,112,119`) |
| **2. `KV`** (`6,681` attn + `9` mamba) | **`2.041 GiB`** (logged `2.04 GiB`) | **`2.041 GiB`** | `2,191,441,920 B` | `MEASURED` (`job10b_warm_2.log:116,119,185`) |
| **3. `activations/prefill workspace`** | **`0.000 GiB`** (`determine_available_memory` does no dummy forward pass; `tpu_worker.py:512`) | **`0.000 GiB` idle / `0.051 GiB` peak delta** (`peak 14.790 - in_use 14.739 GiB`; at `32K`/`200K` in `run_results.json:60,70`: `+0.083 GiB` peak delta) | `0 B` (at KV check) | `MEASURED` (`job10b_warm_2.log:120,243,299`: `profile_run dur_s=0.00`) |
| **4. `compile/scratch`** | **`0.004 GiB`** (`4,255,744 B` XLA metadata during KV alloc) | **`0.251 GiB` persistent** (`14.739 - 14.488 GiB`) / **`0.302 GiB` peak** (`14.790 - 14.488 GiB`) | `4,255,744 B` (at KV alloc) | `MEASURED` (`job10b_warm_2.log:186,243,299`) |
| **5. `vLLM reserve (1 - 0.92)`** | **`1.260 GiB`** (`15.748 - 14.488 GiB`; free after KV = `1.256 GiB`) | **`1.009 GiB` free idle** (`15.748 - 14.739 GiB`) / **`0.958 GiB` free at peak** (`15.748 - 14.790 GiB`) | `1,352,746,886 B` (`0.08 × limit`) | `MEASURED` (`job10b_warm_2.log:186,243,299`) |
| **6. `other`** | **`0.000 GiB`** | **`0.000 GiB`** | `0 B` | `MEASURED` |
| **Reconciliation Check** | `12.447 + 2.041 + 0.004 + 1.256 =` **`15.748 GiB`** (Δ = **`0.000 GiB`**) | `12.447 + 2.041 + 0.251 + 1.009 =` **`15.748 GiB`** (Δ = **`0.000 GiB`**) | `16,909,336,064 B` | **Reconciles to `0.000 GiB` (`< 0.05 GiB`)** |

---

## 5. 200K Failure Analysis & Verification of the `~0.51 GiB` Claim (`MEASURED` & `INFERRED`)

1. **Exact Failure at Current Settings (`run_results_run1.json:100`, `vllm/v1/core/kv_cache_utils.py:1030,2354,2620`, `MEASURED`)**:
   - **Error**: `ValueError: To serve at least one request with the model's max seq len (200000), (6.68 GiB KV cache is needed, which is larger than the available KV cache memory (3.57 GiB). Based on the available memory, the estimated maximum model length is 106816.`
   - **Why these exact numbers fail**:
     - `kv_cache_manager.py:277-291` sets `uniform_page_size_bytes = 57,344 B` on all 40 layers (`573,440 B/block` per 10-layer group).
     - `kv_cache_manager.py:460` sets `num_gpu_blocks_override = 6681` (from `2.0410 GiB` real available HBM $\div 327,680\text{ B/block}$).
     - `kv_cache_utils.py:2624-2632` sets `check_memory = (6681 - 1) × 573,440 B = 3,830,579,200 B` = **`3.5675 GiB` (`3.57 GiB`)**.
     - For `max_model_len=200000`, `_max_memory_usage_bytes_from_groups` requires `12,500` attention blocks (`200000 / 16`) + `3` mamba blocks (`1` per mamba group) = `12,503` blocks $\times 573,440\text{ B} = 7,169,720,320\text{ B}$ = **`6.6773 GiB` (`6.68 GiB`)**.
     - **Fails by**: **`3.1098 GiB`** (`3,339,141,120 B`, or **`3.11 GiB`** displayed; `5,823` blocks short; max length $(6680 - 4) \times 16 = \mathbf{106,816}$ tokens).
2. **Verify or Refute the `~0.51 GiB` Shortfall Claim (`BEARINGS.md:98`)**:
   - **Where `~0.51 GiB` (`0.5159 GiB`) comes from**: Physical compact-mamba KV at 200K is $10 \times 12,500 \times 32,768\text{ B} + 30 \times 9 \times 8,192\text{ B} = 4,098,211,840\text{ B} = \mathbf{3.8168\text{ GiB}}$. Comparing `3.8168 GiB` against **total unreserved free HBM after weights** (`3,544,269,312 B = 3.3009 GiB`, i.e. assuming `gpu_memory_utilization = 1.00` and zero compile/runtime overhead) gives $3.8168 - 3.3009 = \mathbf{0.5159\text{ GiB}}$ (`553,942,528 B`).
   - **Verdict — REFUTED for current settings (`0.92` util) and incomplete at `1.00` util**:
     - At current `gpu_memory_utilization = 0.92`, available KV HBM is **`2.0410 GiB`** (`2,191,522,426 B`), so physical `bf16` 200K KV (`3.8168 GiB`) is **`1.7757 GiB` (`1,906,689,414 B`) short** (and vLLM's unpatched capacity check is **`3.1098 GiB` short**).
     - Even if `gpu_memory_utilization` were raised to expose the full `15.748 GiB` chip limit, post-KV compiled executables and 8192-token runtime activations consume an additional **`0.251 GiB` persistent / `0.302 GiB` peak** (`job10b_warm_2.log:186,299`), making the true full-chip `bf16` shortfall $0.5159 + 0.302 = \mathbf{0.818\text{ GiB}}$ (`~0.767 GiB` persistent).

---

## 6. Every Deviation from Stock Model or Stock vLLM (`MEASURED`)

1. **Mixed W8A8-Int8 + W4A16 Checkpoint (`config_builder.py:25-66`, `config.json`)**: Quantizes `self_attn`, `conv.in/out_proj`, and `layers.0..1` MLP to symmetric per-channel `int8` (`W8A8` dynamic token activations) and `layers.2..39` MoE experts to group-128 `int4` (`W4A16`), keeping embeddings, norms, router gates, and `conv.conv` in `bfloat16` to fit 24B parameters on one 16 GB v5e-1 chip.
2. **On-Load MoE Per-Channel `int4` Re-Quantization (`tpu_inference/.../moe_weights.py:899-933,1111`, `job10b_warm_2.log:101`)**: Unpacks group-128 `int4` MoE weights (`BF16` scales) and re-quantizes them per-channel (`block_size=2048/1536`) to `int4` with 4D `float32` scales for TPU GMM execution.
3. **Config `layer_types` Injection (`lfm25_coldstart_driver.py:125-129`)**: Injects explicit 40-element `layer_types` (`full_attention` at `2,6,10,14,18,22,26,30,34,38`, else `conv`) into `/dev/shm/lfm25_200k/config.json` if missing.
4. **RPA v3 `kernel_hd64` Attention Kernel (`tpu_inference/.../attention_backend.py:94-95`, `kernel_hd64.py:250`, `job10b_warm_2.log:195-201`)**: Dispatches `head_dim == 64` to `kernel_hd64` with `bkv_p=128, bq=16/32` and KV cache shape `(B, 16, 4, 2, 128)`.
5. **Tokamax `gmm_v2` (`MoEBackend.GMM_TP`) Kernel (`tpu_inference/.../moe.py:68`, `fused_moe_gmm.py:23`, `job10b_warm_2.log:75`)**: Executes MoE routed experts and quantized linears via `gmm_v2` with `align=128` `w1/w3` padding.
6. **Hybrid KV Page Padding + Compact-Mamba Override (`tpu_inference/.../kv_cache_manager.py:280,471`, `job10b_warm_2.log:115-121`)**: Pads vLLM layer specs to `57,344 B` while overriding TPU allocation to `6,681` attention blocks + `9` recurrent ShortConv slots per layer (`BEARINGS.md:91`).
7. **Pure-Torch/XLA `ShortConv.forward` Replacement (`tpu_v5e_patches.py:71-90,528-531`)**: Replaces `ShortConv.forward` with `_tpu_xla_short_conv_forward` (`sd_causal_conv` + `kv_caches` slot indexing) so `torchax` traces ShortConv without Triton/CUDA ops or host syncs.
8. **Token Padding Bucket Pruning to `[16, 8192]` (`tpu_v5e_patches.py:185-208`)**: Wraps `get_token_paddings` across `tpu_inference` to prune `[16, 32, ..., 8192]` down to `[16, 8192]` (or `[16, 4096]` if `LFM_DIAG=1`) to cut XLA backbone compilations from 10 to 2.
9. **64-Expert Batched `general_device_put` Stacking (`tpu_v5e_patches.py:468-527`)**: Buffers 64 experts per `(layer, w1/w2/w3)` and calls `jax.device_put` once on the stacked tensor to speed up TPU sharding.
10. **Disable `_precompile_structured_decoding` (`tpu_v5e_patches.py:563-566`)**: No-ops `_precompile_structured_decoding` on `TPUModelRunner` and `CompilationManager` to skip unused XLA precompilations.
11. **KV Dtype & Block-Size Validation Bypass Patches (`tpu_v5e_patches.py:538-562`)**: Patches `supports_kv_cache_dtype`, `static_validate_inputs`, and `get_tuned_block_sizes` to permit experimental `fp8`/`int8`/`k8v4` KV dtypes (`run_results.json:14-15`; inactive at `kv=auto`).
12. **In-Process Model Registry Inspection (`tpu_v5e_patches.py:570-596`, `LFM_INPROC_INSPECT=1`)**: Replaces `vllm.model_executor.models.registry._run_in_subprocess` with direct in-process execution, saving `~11.5 s` of subprocess spawn overhead (`BEARINGS.md:298,310,351`).
13. **$O(\text{new modules})$ Import Hook (`tpu_v5e_patches.py:599-635`, `LFM_PATCH_FAST=1`)**: Tracks `_SEEN` modules in `custom_import` so `patch_sys_modules` only scans newly loaded modules, saving `~118.3 s` during weight loading (`BEARINGS.md:158-170`).
14. **GCE Metadata Fast-Fail & `/etc/hosts` Blackhole (`lfm25_engine_subrunner.py:3-8`, `lfm25_coldstart_driver.py:56-59,278-280`)**: Rejects `metadata.google.internal` in `urllib.request.urlopen` and points `GCE_METADATA_*` to `127.0.0.1:1` to eliminate GCE metadata poll timeouts on Colab.
15. **VllmModelWrapper.modules & Empty-Schedule Log Filter (`tpu_v5e_patches.py:210-221,313-315`)**: Adds `.modules()` delegation on `VllmModelWrapper` (fixing atexit `AttributeError`, `run_results.json:21,26`) and drops `"Should not schedule a request that does nothing!"` warning spam.
16. **Runtime & XLA Cache Env Overrides (`lfm25_engine_subrunner.py:15-22`, `lfm25_coldstart_driver.py:275-282`, `run_job10b.py:43-47`)**: Sets `MODEL_IMPL_TYPE=vllm`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `JAX_COMPILATION_CACHE_DIR=VLLM_XLA_CACHE_PATH=/dev/shm/tpu_compile_cache`, `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`, `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1`, `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all`, `SKIP_JAX_PRECOMPILE=0` (overrides subrunner default `1`), `TPU_TOPOLOGY=1x1`, `HF_HUB_OFFLINE=1`, `LFM_HF_ANON=1`, and `LFM_ABORT_ON_CACHE_MISS=1`.
