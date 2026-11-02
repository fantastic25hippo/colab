# SESSION C: Verdict — Runtime (Audit A) vs Research (Audit B) for LFM2.5-24B-A2B at 200K

- **Target Spec**: `LiquidAI/LFM2.5-24B-A2B` (`avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU`) alone on 1× TPU v5e-1 (`15.748 GiB` / `16,909,336,064 B` usable HBM), `max_model_len=200000`, `max_num_seqs=1`, small accuracy loss OK.
- **Rule**: Audit A (`MEASURED` via `job10b_warm_2.log`, `run_results_run1.json`) beats Audit B (`RESEARCH`). Untested B claims = **`NOT TESTED`**. Unsettled items = **`UNKNOWN`** + settling test.

## 1. Does the Research Claim 200K Fits? Under Exactly Which Config?
**Yes — conditionally.** Audit B (`AUDIT_B_research.md:274-289`) concludes 200K **fails out-of-the-box** (`6.68 GiB` demanded vs `3.57 GiB` available) and **fits (`14.42 / 15.748 GiB`, `+1.33 GiB` margin)** only under:

| Subsystem | Research Converged Spec (`AUDIT_B_research.md:278-289`) | Claimed GiB ($2^{30}$ B) | Required Mechanism / Patch Claimed by B |
|---|---|---:|---|
| **Chip & Engine** | 1× TPU v5e-1 (`15.748 GiB` limit), `vllm v0.32.0` + `tpu-inference v0.32.0`, `MODEL_IMPL_TYPE=vllm` | `15.748` ceiling | `VllmModelWrapper` (torchax) + CPU weight load before `torchax` |
| **MoE Experts (`layers.2..39`)** | `compressed-tensors` `W4A16` sym `int4`, `group_size=128` (`38×64=2,432` experts, top-4) | `11.022` (`10.688` + `0.334` scale) | Tokamax `gmm_v2` (`MoEBackend.GMM_TP`) |
| **Backbone Linear** | `compressed-tensors` `W8A8` (sym per-channel `int8` wt + dynamic per-tok `int8` act: `10` GQA, `30` Conv, `2` FFN) | `0.702` (`0.469`+`0.098`+`0.135`) | Native INT8 MXU (`VllmCompressedTensorsW8A8Int8`) |
| **Embeddings + Norms/Gates** | `BF16` tied `embed_tokens` (`0.488 GiB`) + `BF16` norms/gates (`0.010 GiB`) *(optional `emb4` saves `0.35 GiB`)* | `0.498` *(total wt = **`12.45`** w/ pad)* | `tie_word_embeddings: true` |
| **KV Cache (`10` GQA, `200K`)** | **Primary**: `INT8` (or `K8V4`) in RPA v3 with 2 KV heads folded per 128-lane VREG + in-VMEM dequant<br>**Fallback**: Unpadded `BF16` (`head_dim=64`) at `gpu_memory_utilization = 0.97` | **`1.907–1.967`** (`INT8` folded)<br>*(`1.490` `K8V4`; `3.815` `BF16`)* | **Patch P3**: `int8` KV + 2-head folding in `get_kv_cache_shape` + VMEM scale dequant |
| **ShortConv (`30` layers)** | `BF16` state `(2, 2048)` per layer (`MambaSpec`), `B = 1` (`8,192 B/layer`) | **`0.00023`** (`245.76 KB`) | **P1**: Size `MambaSpec` check by `max_num_seqs` not `12,500` blocks (`-2.861 GiB`)<br>**P2**: Slice `conv_state` at `query_start_loc[1]`, not padded `T` |
| **Batching & Reserve** | `max_num_seqs=1`, `max_model_len=200000`, `max_num_batched_tokens=8192`, buckets `[16, 8192]`, `util=0.92` (`0.96+` for `BF16`) | `~0.88` act/compile, `0.47–1.26` reserve | Prune `get_token_paddings` to `[16, 8192]` + persistent XLA cache |
| **TOTAL (200K, B=1)** | **Weights (`12.45`) + INT8 KV (`1.97`) + ShortConv (`0.0002`)** | **`14.42 / 15.748 GiB`** | **Claimed Verdict: FITS SAFELY** (with P1 + P2 + P3) |

## 2. Per-Item Test Status Against Audit A

| # | Converged Config Item (from Audit B) | Status | Audit A & Code Evidence (`file:line`) |
|---:|---|---|---|
| 1 | 1× TPU v5e-1 (`15.748 GiB` / `16,909,336,064 B` HBM) | **`TESTED`** | `AUDIT_A_runtime.md:6,74`, `job10b_warm_2.log:71,110,243` |
| 2 | `vllm v0.32.0` + `tpu-inference v0.32.0`, `MODEL_IMPL_TYPE=vllm` (torchax) | **`TESTED`** | `AUDIT_A_runtime.md:4,121`, `job10b_warm_2.log:55,72` |
| 3 | MoE `W4A16` sym `int4` (`group_size=128` on disk -> re-quantized per-channel on load, `gmm_v2`) | **`TESTED`** | `AUDIT_A_runtime.md:20,106-107,110`, `job10b_warm_2.log:75,101-105` |
| 4 | Backbone `W8A8` dynamic `int8` (`self_attn`, `conv.in/out_proj`, `layers.0..1` dense MLP) | **`TESTED`** | `AUDIT_A_runtime.md:21-23,106`, `job10b_warm_2.log:53,55` |
| 5 | Tied `BF16` `embed_tokens` (`0.4883 GiB`) + `BF16` norms/gates (`12.447 GiB` total weights) | **`TESTED`** | `AUDIT_A_runtime.md:16,24-27`, `job10b_warm_2.log:110,112` |
| 6 | Optional `emb4` (`INT4` tied embeddings saving `~0.35 GiB`) | **`NOT TESTED`** | Checkpoint `embed_tokens` is `BF16` (`AUDIT_A_runtime.md:24`) |
| 7 | Token padding buckets pruned to `[16, 8192]`, `max_num_batched_tokens=8192`, `max_num_seqs=1` | **`TESTED`** | `AUDIT_A_runtime.md:70,113`, `tpu_v5e_patches.py:185-208`, `job10b_warm_2.log:41,190,199` |
| 8 | **Patch P2**: Pure-XLA `ShortConv.forward` slicing state at `n = query_start_loc[1] - query_start_loc[0]` | **`TESTED`** *(100K)* | `AUDIT_A_runtime.md:112`, `tpu_v5e_patches.py:71-90,154-181,528-531`, `job10b_warm_2.log:252` |
| 9 | **Patch P1**: Fix vLLM `2.861 GiB` ShortConv hybrid block-multiplication check | **`NOT TESTED`** | `tpu_v5e_patches.py:247-275` only logs sizes; 200K fails `6.68 > 3.57 GiB` (`run_results_run1.json:100`) |
| 10 | **Patch P3**: `INT8` (or `K8V4`) KV cache + 2-head folding + in-VMEM dequant in RPA v3 | **`NOT TESTED`** | Running config uses `kv_cache_dtype="auto"` (`bfloat16`, `job10b_warm_2.log:55,144`, `run_results_run1.json:78`) |
| 11 | **Fallback**: `gpu_memory_utilization = 0.96–0.97` with `BF16` KV at 200K | **`NOT TESTED`** | Only `gpu_memory_utilization = 0.92` was run (`job10b_warm_2.log:119`, `run_results_run1.json:76`) |

## 3. Diff Table: Research (Audit B) vs Running (Audit A)

| Subsystem / Item | Research Claim (`AUDIT_B`) | Running Reality (`AUDIT_A`) | GiB Impact (Run $-$ Res) | Verdict | Evidence (`file:line`) |
|---|---|---|---:|---|---|
| **Total Loaded Weights** | `12.45 GiB` (`12.22` disk + `~0.23` XLA pad) | `12.4472 GiB` (`13,365,066,752 B` HBM; `12.2216 GiB` safetensors) | `0.000 GiB` | **`MATCHES`** | `AUDIT_A_runtime.md:14,16,27`, `job10b_warm_2.log:110,112` |
| **MoE Scale Granularity on TPU** | `group_size=128` `BF16` scales (`0.3340 GiB`) kept as-is | Unpacked & **re-quantized per-channel** (`block_size=2048/1536`) to `F32` `(64,1,1,N)` (`0.0464 GiB` logical, padded `1->8` sublanes = `0.3711 GiB` HBM) | `+0.037 GiB` *(in `12.447`)* | **`RESEARCH STALE`** | `AUDIT_A_runtime.md:20,107`, `job10b_warm_2.log:101-105`, `moe_weights.py:899-933,1111` |
| **RoPE Cache (`cos_sin_cache`)** | Omitted (`0 GiB`) in B's weight tables | `(200000, 64)` `F32` (`0.0477 GiB` logical, padded `64->128` lanes in `{1,0}` layout = `0.0954 GiB` HBM) | `+0.095 GiB` *(in `12.447`)* | **`RESEARCH WRONG`** *(omitted)* | `AUDIT_A_runtime.md:22,27`, `vllm_model_wrapper.py:81-104` |
| **RPA v3 `head_dim=64` Layout (`bf16`)** | G6 claims `get_kv_cache_shape` pads `64->128` via `align_to(head_dim, 128)` (`kernel.py:274`) | `head_dim==64` dispatches to `kernel_hd64.py`, which packs `K` (`64`) + `V` (`64`) into `128` and `kv_packing=2` heads: `(B, 16, 4, 2, 128)` = **`32,768 B/block/layer` (`3.8147 GiB` at 200K, 0% pad waste in `bf16`!)** | `0.000 GiB` in `bf16` *(G6 cited wrong file)* | **`RESEARCH STALE`** *(on `bf16` kernel path)* | `AUDIT_A_runtime.md:42,54,109`, `job10b_warm_2.log:144,175-184`, `attention_backend.py:94-101`, `kernel_hd64.py:236-251,981-1006` |
| **ShortConv Physical TPU Slots** | `1` slot (`0.00023 GiB`) for `max_num_seqs=1` | `9` slots (`2,211,840 B = 0.00206 GiB`) due to `max_num_reqs = max(dp*max_num_seqs, MIN_NUM_SEQS=8) = 8` + `1` null slot | `+0.0018 GiB` (`+1.87 MiB`) | **`RESEARCH WRONG`** *(minor)* | `AUDIT_A_runtime.md:45,57`, `job10b_warm_2.log:68,116,145-174`, `tpu_runner.py:1078-1079` |
| **Why `3.57 GiB` Appears as Available KV at `util=0.92`** | G6 treats `3.57 GiB` as real free HBM (`14.49 - 12.45 = 2.04 GiB`, causing math contradiction in G6) | Real free KV HBM at `0.92` is **`2.0410 GiB`** (`6,681` blocks $\times 327,680\text{ B}$). `3.57 GiB` (`3.5675 GiB`) is **synthetically inflated**: `(6681 - 1) × 573,440 B` (`uniform_page_size_bytes` padded with 3× mamba!) | **`+1.5265 GiB` illusory HBM in B** | **`RESEARCH WRONG`** *(crucial math error in G6)* | `AUDIT_A_runtime.md:89-99`, `AUDIT_B_research.md:232-233`, `job10b_warm_2.log:115-121`, `kv_cache_utils.py:2611-2633` |
| **ShortConv Admission Check (P1)** | Patch check so 30 `MambaSpec` layers aren't multiplied by `12,500` blocks (`-2.861 GiB` demand) | Unpatched: `57,344 B` $\times 10\text{ layers} = 573,440\text{ B/block} \times 12,503\text{ blocks} = \mathbf{6.6773\text{ GiB}}$ demanded (`6.68 GiB`) | `+2.8626 GiB` phantom demand | **`WE DEVIATED`** / **`NOT TESTED`** | `AUDIT_A_runtime.md:56,64,90-95`, `run_results_run1.json:100`, `kv_cache_manager.py:277-291`, `kv_cache_utils.py:2354-2366` |
| **ShortConv Slice at `query_start_loc` (P2)** | Slice at `n = query_start_loc[1]`, not padded `T` | Implemented in `tpu_v5e_patches.py:84-86,154-160` (`n = clamp(qsl_t[1] - qsl_t[0], 0, T)`) | `0.000 GiB` (correctness) | **`MATCHES`** *(tested at 100K)* | `AUDIT_A_runtime.md:112`, `tpu_v5e_patches.py:71-90,154-181,528-531` |
| **KV Cache Dtype (P3)** | `INT8` (or `K8V4`) folded KV (`1.907–1.967 GiB` at 200K) | `kv_cache_dtype="auto"` -> `bfloat16` (`3.8147 GiB` attn + `0.0021 GiB` mamba = `3.8168 GiB` at 200K) | `+1.850 to +1.910 GiB` at 200K | **`WE DEVIATED`** / **`NOT TESTED`** | `AUDIT_A_runtime.md:44,61-62`, `run_results_run1.json:78`, `tpu_v5e_patches.py:538-562` |
| **Compile / Scratch + Activations** | G4 budgets `1.65 GiB`; G6 budgets `~0.88 GiB` | **`0.251 GiB` persistent** (`14.739 - 14.488 GiB`), **`0.302 GiB` peak** (`14.790 - 14.488 GiB`, i.e. `+0.051 GiB` act delta; `+0.083 GiB` max in `run_results.json`) | `-0.578 to -1.348 GiB` *(uses much less!)* | **`RESEARCH WRONG`** *(over-budgeted)* | `AUDIT_A_runtime.md:6,78-80`, `job10b_warm_2.log:186,243,299` |
| **`gpu_memory_utilization`** | `0.92` (`INT8` KV) or `0.96–0.97` (`BF16` KV) | `0.92` (`14.488 GiB` cap, `1.260 GiB` reserve pre-compile, `0.958 GiB` free at peak) | `0.000 GiB` (at `0.92`) | **`MATCHES`** (`0.92`) / **`NOT TESTED`** (`0.96+`) | `AUDIT_A_runtime.md:70,75,80`, `job10b_warm_2.log:119,299` |

## 4. Bug Check P1: Is the `6.68 GiB` Demand Really ShortConv State Multiplied by Blocks?
- **Verified in Source (`CONFIRMED`)**:
  1. Unpadded page sizes: 10 `FullAttentionSpec` layers = **`32,768 B/block`** (`kv_cache.py:200-220`, `kernel_hd64.py:236-251`); 30 `MambaSpec` (`ShortConv`) layers = **`8,192 B/slot`** (`2 × 2048 × 2 B`, `kv_cache_interface.py:1062-1070`).
  2. In `KVCacheManager.update_mamba_page_size_padded` (`kv_cache_manager.py:264-291`), `num_attn=10, num_mamba=30 -> group_size=10, num_attn_groups=1, num_mamba_groups=3`. Lines `277-278` pad **every** layer's `page_size_padded` (`kv_cache_manager.py:135-150`, `kv_cache_interface.py:1067-1069`) to:
     $$\text{uniform\_page\_size\_bytes} = 1 \times 32,768 + 3 \times 8,192 = \mathbf{57,344\text{ B}}$$
  3. In `_max_memory_usage_bytes_from_groups` (`kv_cache_utils.py:2337-2366`), `_pool_bytes_per_block = 10 × 57,344 = 573,440 B`. For `max_model_len=200000`, the 1 attention group needs `200000 / 16 = 12,500` blocks and the 3 Mamba groups need `1` block each (`kv_cache_interface.py:1082`) = `12,503` total blocks:
     $$\text{needed\_memory} = 12,503 \times 573,440\text{ B} = \mathbf{7,169,720,320\text{ B}} = \mathbf{6.6773\text{ GiB}} \ (\text{logged as }\mathbf{6.68\text{ GiB}}\text{ at }\texttt{kv\_cache\_utils.py:1032})$$
     Of that `6.6773 GiB`, `12,500 × 327,680 B = 3.8147 GiB` is real attention KV, and **`12,500 × 245,760 B + 3 × 573,440 B = 3,073,720,320 B = 2.8626 GiB` is phantom ShortConv padding multiplied by `12,500` blocks!**
  4. **How many GiB does fixing P1 free?**
     - **Virtual Admission-Check Demand Saved**: **`2.8626 GiB`** (reducing 200K `bf16` demand from `6.6773 GiB` to `3.8147 GiB`, and 200K `int8` demand from `4.7700 GiB` to `1.9073 GiB`).
     - **Physical TPU HBM Freed**: **`0.000 GiB`**, because `_maybe_set_compact_mamba_num_blocks_override` (`kv_cache_manager.py:423-460`) and `initialize_kv_cache` (`kv_cache_manager.py:880-990`) **already** allocate only `9` Mamba slots (`0.0021 GiB`) and compute `num_gpu_blocks_override = 6,681` from the **unpadded** `327,680 B/block`! Furthermore, in `get_kv_cache_configs` (`kv_cache_utils.py:2611-2633`), `check_memory = 6,680 × 573,440 B = 3.5675 GiB` (`3.57 GiB`) was inflated by the exact same `573,440 / 327,680` factor — real free KV HBM at `util=0.92` is **`2.0410 GiB`**, not `3.57 GiB`.

## 5. Bug Check P2 (Correctness): ShortConv Padded-`T` Slice in Running Code
- **Confirmed Fixed in Running Code (`tpu_v5e_patches.py:71-90, 154-181, 528-531`)**:
  - Stock `ShortConv.forward` (`short_conv.py:205-215, 364-376`) calls `forward_cuda` (`short_conv.py:216-342`, Triton `causal_conv1d_fn`). Our patch replaces `ShortConv.forward` with `_tpu_xla_short_conv_forward` (`tpu_v5e_patches.py:71-90, 528-531`).
  - Instead of slicing at padded bucket length `T ∈ {16, 8192}`, `_lookup_conv_cache` (`tpu_v5e_patches.py:154-160`) reads `query_start_loc` from `attn_metadata` (`tpu_runner.py:3135,3148`) and computes `n = torch.clamp((qsl_t[1] - qsl_t[0]).to(torch.long), min=0, max=T)`, then slices `x_ext` at `n + torch.arange(state_len)` (`tpu_v5e_patches.py:85`) and zeroes `prev` on chunk 0 (`ctx_len <= n`, `tpu_v5e_patches.py:174-179`).
  - **Impact on 100K results (`job10b_warm_2.log`)**: **None (`B=1` is already clean)** — the fix was active during `job10b_warm_2.log` and produced coherent text (`job10b_warm_2.log:252`).

## 6. Bug Check P3: `head_dim` Pad to 128 in RPA v3, and `int8` + Head Folding Support
1. **Why Audit B (G6) Was Stale on `head_dim=64` Padding**:
   - In `RPAAttentionBackend.get_kv_cache_shape` (`attention_backend.py:94-101`), `head_dim == 64` dispatches to `kernel_hd64.py`, **not** `kernel.py:261-275`.
   - `kernel_hd64.get_kv_cache_shape` (`kernel_hd64.py:236-251`) and `merge_kv` (`kernel_hd64.py:981-1006`) **already concatenate `K` (`64`) and `V` (`64`) into the 128-lane last axis (`actual_head_dim * 2 = 128`)** and pack `kv_packing = 32 // bitwidth` (`util.py:32-34`) heads along axis 3:
     $$\text{shape} = (\text{pages},\ 16,\ \text{align\_to}(8, \text{kv\_packing}) // \text{kv\_packing},\ \text{kv\_packing},\ 128)$$
   - For `bfloat16` (`kv_packing=2`): `(B, 16, 4, 2, 128)` `bf16` = **`32,768 B/block`** (`job10b_warm_2.log:175`, **0% padding waste**).
   - For `int8` (`kv_packing=4`): `kernel_hd64.get_kv_cache_shape` **automatically** returns `(B, 16, 2, 4, 128)` `int8` = **`16,384 B/block` (`1.9073 GiB` across 10 layers at 200K)** — no head-folding layout rewrite is needed!
2. **What `int8` KV Actually Needs (~30-line patch + 2-line Pallas VPU cast)**:
   - `quantize_kv` (`flash_attn.py:259-266`, `quantization/__init__.py:303-331`) and `strided_load_bkv` (`kernel_hd64.py:734-750`) already support `jnp.int8` (`kv_packing=4` bitshift + `pltpu.bitcast`), and `tpu_v5e_patches.py:538-562` already bypasses `static_validate_inputs` (`kernel_hd64.py:1257-1258`) and `get_tuned_block_sizes` (`tuned_block_sizes_hd64.py:427`).
   - Still needed: (a) map `kv_cache_dtype="fp8"` (or `"int8_per_token_head"`) -> `jnp.int8` in `tpu_inference/utils.py:37-43`; (b) cast `kv = kv.astype(q.dtype)` in `kernel_hd64.py:397` (`flash_attention_step1_qk_softmax`) and `kv.astype(jnp.bfloat16)` in `kernel_hd64.py:458` (`flash_attention_step2_pv`) before `jnp.matmul`; (c) supply non-unit `k_scale`/`v_scale` (default `1.0` clips unscaled `bf16` K/V before `astype(int8)`).

## 7. 200K Memory Math in Standard Buckets (`GiB` = $2^{30}$ B, Using Audit A Measured Numbers)

| Bucket (`GiB` = $2^{30}$ B; Total Chip = `15.748 GiB`) | (a) **P1 only, `bf16` KV (`12,500` blk), `util=0.97`** | (b) **P1 + P3 (`int8` KV in `kernel_hd64`) at `util=0.92`** | (c) **B's Best Config (`int8` KV + `bf16` scales, `util=0.92`)** |
|---|---:|---:|---:|
| **1. `weights`** (`MEASURED`, `A`) | `12.447 GiB` | `12.447 GiB` | `12.447 GiB` |
| **2. `KV` @ 200K (`12,500` attn + `9` mamba)** | **`3.817 GiB`** (`3.8147` + `0.0021`) | **`1.909 GiB`** (`1.9073` + `0.0021`) | **`1.969 GiB`** (`1.9073` + `0.0596` scale + `0.0021`) |
| **3. `compile/scratch` (persistent)** (`MEASURED`, `A`) | `0.251 GiB` | `0.251 GiB` | `0.251 GiB` |
| **4. `activations` (peak delta)** (`MEASURED`, `A`) | `0.051 GiB` (`0.083` max) | `0.051 GiB` (`0.083` max) | `0.051 GiB` (`0.083` max) |
| **Subtotal Used at Peak (`1 + 2 + 3 + 4`)** | **`16.566 GiB`** (`16.315` pre-compile) | **`14.658 GiB`** (`14.356` pre-compile) | **`14.718 GiB`** (`14.416` pre-compile) |
| **5. `vLLM reserve` (`(1 - util) × 15.748`)** | `0.472 GiB` (`0.000` at `1.00`) | `1.260 GiB` (cap `14.488 GiB`) | `1.260 GiB` (cap `14.488 GiB`) |
| **Pre-Compile KV Margin (`cap - weights - KV`)** | **`-1.037 GiB`** (`-0.516` at `util=1.00`) | **`+0.132 GiB`** (`~270` spare blocks) | **`+0.072 GiB`** (`~143` spare blocks) |
| **Post-Compile Peak Headroom (`15.748 - Peak`)** | **`-0.818 GiB` (OOM!)** | **`+1.090 GiB`** (`+1.058` at `0.083` act) | **`+1.030 GiB`** (`+0.998` at `0.083` act) |
| **VERDICT** | **`DOESN'T FIT`** (even at `util=1.00`) | **`FITS`** (`14.658 / 15.748 GiB`) | **`FITS`** (`14.718 / 15.748 GiB`) |

## 8. Is Our Current Config Best for 1 Request at 200K on 16 GB v5e-1? Top 3 Changes
**No** — current config (`bf16` KV, unpatched `57,344 B` hybrid check, `util=0.92`) fails admission by `3.11 GiB` and physically exceeds chip HBM by `0.818 GiB` at peak.

| Rank | Change | GiB Saved at 200K | Accuracy Risk | Effort (`config` / `patch` / `kernel work`) |
|---:|---|---:|---|---|
| **1** | **P1: Fix Hybrid `MambaSpec` Admission Check (`kv_cache_manager.py:277` / `kv_cache_utils.py:2354`)**<br>Check attention blocks at unpadded `attn_page_size_bytes` (`16,384 B` `int8` / `32,768 B` `bf16`) instead of `uniform_page_size_bytes`. | **`2.863 GiB`** *(virtual admission demand)* | **None** (`0%` — admission math only) | **`patch`** (~15 lines in `tpu_v5e_patches.py`) |
| **2** | **P3: Enable `int8` KV in `kernel_hd64.py` (`(B, 16, 2, 4, 128)`)**<br>Map `kv_cache_dtype` to `jnp.int8`, cast `kv.astype(q.dtype)` in `kernel_hd64.py:397,458`, and pass calibrated `k_scale`/`v_scale`. | **`1.907 GiB`** *(physical HBM: `3.815 -> 1.907 GiB`)* | **Low** (`<1%` PPL with calibrated `k_scale`/`v_scale`; **High** if left at `1.0`) | **`patch` + minor `kernel work`** (~30 lines + XLA cache rebuild) |
| **3** | **Raise `gpu_memory_utilization` `0.92 -> 0.94` (or `emb4` INT4 Tied Embeddings)**<br>`util=0.94` adds `+0.315 GiB` KV cap while keeping `0.643 GiB` free at peak; `emb4` frees `0.362 GiB` weight HBM. | **`+0.315 GiB`** *(KV cap)* or **`0.362 GiB`** *(wt HBM)* | **None** (`util=0.94`) / **Low-Mod** (`emb4`) | **`config`** (`util=0.94`) vs **`patch` + re-pack** (`emb4`) |

## 9. Recommended Order of Work & Smallest Tests (Plus `UNKNOWN` Items)
1. **Step 1 (Correctness — P2 + `int8` KV + `k_scale`/`v_scale` at `16K` context)**: Confirm P2 (`LFM_CONV_DEBUG=1`) and test `int8` KV in `kernel_hd64.py:397,458` on a `9,000`-token needle prompt (`1 × 8192` chunk + `1 × 808` padded chunk + `16` decode steps). *Smallest test*: 1 run at `max_model_len=16384` verifying exact needle recall and `CONVDBG` (`n=8192 -> 808 -> 1`).
2. **Step 2 (Admission Check — P1 on CPU, 0 CU)**: Patch `_max_memory_usage_bytes_from_groups` / `update_mamba_page_size_padded` in `tpu_v5e_patches.py`. *Smallest test*: Local CPU unit test calling `get_kv_cache_configs` at `max_model_len=200000`, asserting `needed_memory == 1.9073 GiB` (`int8`) instead of `6.6773 GiB`.
3. **Step 3 (200K KV Allocation — P1 + P3 with `LFM_BEARINGS_STOP=1`)**: Run `max_model_len=200000`, `util=0.94`, `LFM_BEARINGS=1`, `LFM_BEARINGS_STOP=1` (`tpu_v5e_patches.py:302`). *Smallest test*: Exits right after `initialize_kv_cache` (~37 s in-VM), proving `num_blocks >= 12501` of `(N, 16, 2, 4, 128)` `int8` (`1.909 GiB` KV) and `used <= 14.67 GiB`.
4. **Step 4 (Full 200K Warmup & 1-Request Benchmark)**: Compile `[16, 8192]`, publish `len200000` cache tar, and run `199,900`-token prompt.
- **`UNKNOWN` Items & Settling Tests**:
  1. *Peak HBM across `25 × 8192` consecutive prefill chunks at 200K (`UNKNOWN`)*: Audit A only measured 100K init + 16-tok request (`+0.302 GiB` post-KV peak). Settled by logging `peak_bytes_in_use` after the 25-chunk prefill in Step 4.
  2. *Optimal `k_scale`/`v_scale` values for `int8` KV on LFM2.5's 10 GQA layers (`UNKNOWN`)*: Checkpoint lacks `kv_cache` scales (`_k_scale_float = 1.0`). Settled in Step 1 by logging `k.abs().max() / 127.0` and `v.abs().max() / 127.0` across layers `2..38` on a 16K prompt.
