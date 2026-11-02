# SESSION B: Skeptical Audit of LFM2.5-24B-A2B Research Claims on TPU v5e-1

**Target Goal Spec**: `LiquidAI/LFM2.5-24B-A2B` (`Lfm2MoeForCausalLM`) alone on a single Google Colab TPU v5e-1 (`v5litepod-1`), 16 GB HBM (`15.748 GiB` usable by XLA allocator; `16.000 GB = 14.901 GiB` if decimal), `max_model_len = 200000`, `B = 1` (one request at a time, `max_num_seqs = 1`), small accuracy loss acceptable.

---

## 0. Inventory, File Grouping & Chronology

Both target directories were inventoried (`ls -la --full-time`) and MD5-hashed. Note that on disk, the first directory has a trailing `]` in its name:
- **Dir 1**: `/home/sahil/Desktop/sahil/colab/research done about our config]` (9 files)
- **Dir 2**: `/home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/lfm25_200k/lfm25_200k/research/` (7 files)

Every file in **Dir 2** is a byte-for-byte identical copy (same MD5 and mtime) of the corresponding file in **Dir 1**. **Dir 1** contains two additional newer files (`TPU-V5e-LFM2-5-KV-Cache-Optimization.md` and its `.pdf` counterpart) not present in Dir 2. Preferring `.md` over duplicate `.pdf` yields **5 distinct document groups**, ordered below from oldest to newest by mtime:

| Group | Canonical `.md` File Read | Duplicate / Paired Files (Dir 1 & Dir 2) | MD5 (`.md` / `.pdf`) | mtime (`+05:30`) | Internal Date |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **G1** | `TPU v5e LLM Inference Analysis.md` (668,008 B) | Identical `.md` in Dir 1 & Dir 2 | `4ef76ac8c9c377f8b13ca7c8bc2bf5f9` | `2026-10-31 17:17:06` | Refs up to Oct 29, 2026 |
| **G2** | `deep-research-lfm25-24b-a2b-tpu-v5e-optimizations-20261031.md` (12,896 B) | Identical `.md` in Dir 1 & Dir 2 (condensed summary of G3 prompt/themes) | `013a645c01cdfb892148edfe112d0abb` | `2026-10-31 17:41:37` | `2026-10-31` |
| **G3** | `TPU v5e LFM2.5-24B-A2B Inference Optimization.md` (881,326 B) | Identical `.md` in Dir 1 & Dir 2 | `3ed3ad391c283d59139a0fa667f86afb` | `2026-10-31 18:03:02` | Refs up to Oct 30, 2026 |
| **G4** | `We-Are-Designing-200K-TPU-Pipeline-Transcription.md` (49,358 B) | Identical `.md` in Dir 1 & Dir 2; paired with `We are designing a production-grade 200,000-token context single-stream (B=1) inference pipeline for.pdf` (379,248 B) in Dir 1 & Dir 2 | `.md`: `1fce56e9...`<br>`.pdf`: `18c52b04...` | `.md`: `2026-10-31 18:14:21`<br>`.pdf`: `2026-10-31 18:11:19` | `31/10/2026, 18:11` |
| **G5** | `TPU v5e vLLM LFM2-MoE Investigation.md` (347,969 B) | Identical `.md` in Dir 1 & Dir 2; paired with `TPU v5e vLLM LFM2-MoE Investigation.pdf` (1,094,476 B) in Dir 1 & Dir 2 | `.md`: `f3bde901...`<br>`.pdf`: `e915df05...` | `.md`: `2026-11-01 00:53:52`<br>`.pdf`: `2026-11-01 00:56:23` | Refs late Oct 2026 |
| **G6** *(Newest)* | `TPU-V5e-LFM2-5-KV-Cache-Optimization.md` (21,876 B) | Present **only in Dir 1**; paired with `TPU v5e LFM2.5 KV Cache Optimization.pdf` (337,143 B) in Dir 1 | `.md`: `5ef70e21...`<br>`.pdf`: `f93f65bd...` | `.md`: `2026-11-02 02:25:36`<br>`.pdf`: `2026-11-02 02:23:16` | `02/11/2026, 02:23` |

*(Note: G2 and G4 share the exact same 4-question research prompt; G2 is a 157-line summary with Grok inline citation tags, whereas G4 is a 755-line Gemini transcript answering the same 4 questions in full. We audit all 6 groups below so nothing is missed.)*

---

## 1. Per-Document-Group Skeptical Audit

### Group G1: `TPU v5e LLM Inference Analysis.md`
- **Date**: mtime `2026-10-31 17:17:06 +05:30` (internal citations through `2026-10-29`).
- **1. Proposed Config**:
  - **Component Precision**: Uniform `W4A16` symmetric INT4 across `Linear` layers (`group_size=128` recommended for headroom; `group_size=32` also evaluated); tied embeddings (`128000 × 2048`) quantized to 4-bit INT4 (`emb4`) with per-row BF16 scale; RMSNorms and MoE router gates (`mlp.gate`) kept in `BF16`/`FP32`. (Also evaluates `KRAFTON/Raon-1-mini-preview` as Candidate 2).
  - **KV Dtype**: Custom `torch.int8` ring KV cache with per-token-head `bfloat16` scale (`k_scale`, `v_scale`).
  - **head_dim / Padding**: `head_dim = 64` (`32` Q heads, `8` KV heads); claims no head or MoE padding needed because `2048` and `1536` are multiples of 128/256.
  - **Layers Holding KV**: `10` GQA attention layers hold dynamic KV cache; `30` ShortConv layers hold fixed recurrent state (`conv_L_cache = 3`, shape `[30, 1, 2048, 3]` in BF16).
  - **Utilization & Batching**: `--gpu-memory-utilization 0.92`, `--max-num-batched-tokens 2048`, `--enable-chunked-prefill`, `B = 1` (`--max-num-seqs 1`, `tp = 1`).
  - **Kernels**: `vllm-tpu` RPA v3 + `gmm_v2` (`WNA16FusedMoEMethod`, PR #3653/#3660), or `TorchTPU` (`torch.compile(backend="tpu")`) with `StaticInt8RingKVCache`.
- **2. Claimed Numbers & Fit Verdict (Converted to GiB = $2^{30}$ B)**:
  - **Usable HBM**: `15.75 GiB` (`14.49 GiB` ceiling at `0.92` utilization; `1.26 GiB` left for XLA).
  - **Weights**:
    - BF16 base: `47.96 GB` = `44.67 GiB`.
    - Raw W4A16 (0.5 B/param, no scales): `11.99 GB` = `11.16 GiB`.
    - W4A16 (`group_size=128` incl. scales): claimed `12.17 GB` = `11.33 GiB`.
    - W4A16 (`group_size=32` incl. scales): claimed `11.83 GiB`.
    - Tied embeddings: `0.524 GB` (`0.488 GiB`) in BF16, or `0.131 GB` (`0.122 GiB`) in INT4 (`emb4`).
  - **KV Bytes/Token**: `20,480 B/tok` (`19.53 KiB/tok`) in BF16; `10,240 B/tok` (`10.00 KiB/tok`) in INT8.
  - **KV at 200K**:
    - BF16: `4.096 GB` = `3.815 GiB`.
    - INT8: `2.048 GB` = `1.907 GiB` (rounded to `1.91 GiB`).
  - **Other (ShortConv state)**: `368,640 B` = `0.35 MiB` = `0.00034 GiB` (also quoted as `0.00023 GiB` for 2 state slots).
  - **Compile / Scratch / vLLM Reserve**: `1.26 GiB` (the unallocated `8%` of `15.75 GiB`). Explicitly rejects earlier `4.5–6.5 GB` XLA compile overhead claims.
  - **Activations / Prefill Workspace**: Ignored (`0 GiB` budgeted).
  - **Total & Verdict**: Claims `11.33 GiB (W4A16 g128) + 1.91 GiB (INT8 KV) = 13.24 GiB` (`14.50 GiB` with `1.26 GiB` XLA reserve) -> **FITS SAFELY** (`1.25 GiB` margin below `15.75 GiB`). Claims `g32` (`11.83 + 1.91 + 1.26 = 15.00 GiB`) -> **FITS**. Claims BF16 KV (`11.16 + 3.81 + 1.26 = 16.23 GiB`) -> **FAILS (OOM)**.
- **3. Skeptical Math Verification & Flagged Errors**:
  - **ERROR 1 (Weight Scale Undercount by 2x)**: Derived in text from $23.97\text{B}$ params, where raw INT4 is $23.9716 \times 10^9 \times 0.5\text{ B} = 11.986\text{ GB} = 11.163\text{ GiB}$. With `bfloat16` (2-byte) scales at `group_size=128`, scale overhead is $23.9716 \times 10^9 / 128 \times 2\text{ B} = 0.3746\text{ GB} = \mathbf{0.349\text{ GiB}}$, giving $\mathbf{11.512\text{ GiB}}$, NOT `11.33 GiB`! The doc added only `0.174 GiB` (`1 byte` per scale instead of `2 bytes` for BF16). Same error for `group_size=32`: BF16 scales add $23.9716 \times 10^9 / 32 \times 2\text{ B} = \mathbf{1.395\text{ GiB}}$ (total $\mathbf{12.558\text{ GiB}}$), whereas the doc added `0.67 GiB` (`11.83 GiB`, i.e. 1-byte scales).
  - **ERROR 2 (Omitted KV Scale Buffers)**: The doc's `StaticInt8RingKVCache` allocates `k_scale` and `v_scale` of shape `(10, 1, 8, 200000, 1)` in `bfloat16` ($10 \times 2 \times 8 \times 200,000 \times 2\text{ B} = 64,000,000\text{ B} = \mathbf{0.0596\text{ GiB}}$), bringing true INT8 KV to `10,560 B/tok` and $\mathbf{1.967\text{ GiB}}$ at 200K, not `1.907 GiB`.
  - **ERROR 3 (410 MB Materialization Bug in Own Code)**: Section 4.3 writes `k_bf16 = self.k_int8[layer_idx].to(torch.bfloat16) * self.k_scale[layer_idx]` before `scaled_dot_product_attention`, which dequantizes all 200K tokens to BF16 (`409.6 MB` = `0.381 GiB` per K and V, `0.763 GiB` total) in HBM on every decode step.
  - **UNCLEAR**: In Section 4.2, `config.json` targets `["Linear"]` globally for W4A16, leaving it **UNCLEAR** whether attention/shortconv projections are W4A16 or BF16 in its `11.33 GiB` figure. Also cites `torch_tpu==0.2.0` ("open-sourced Oct 20, 2026") whose real upstream existence is **UNCLEAR**.
- **4. Buckets Ignored**: Prefill/decode activation workspace (at 2048 batched tokens); KV scale tensors (`0.060 GiB`); half of the BF16 weight scale bytes (`0.175–0.698 GiB`); vLLM page-table/scheduler overhead.

---

### Group G2: `deep-research-lfm25-24b-a2b-tpu-v5e-optimizations-20261031.md`
- **Date**: mtime `2026-10-31 17:41:37 +05:30` (header: `Researched: 2026-10-31`).
- **1. Proposed Config**:
  - **Component Precision**: Hybrid `compressed-tensors` `config_groups`: `group_0` = `W8A8` dynamic INT8 for `self_attn` and `conv` Linear layers; `group_1` = `W4A16` symmetric INT4 (`group_size=32` or `128`) for `38` MoE expert blocks (`mlp.experts`) from `LiquidAI/LFM2.5-24B-A2B-QAD-Q4_0.gguf`.
  - **KV Dtype**: Recommends `INT8` (or `K8V4` group-64) with in-VMEM tile dequantization inside a patched RPA v3 Pallas kernel (notes `--kv-cache-dtype fp8` silently allocates BF16 due to Issue #3126).
  - **head_dim / Padding**: `head_dim = 64`; notes `get_kv_cache_shape` aligns `head_dim` to `128`: `align_to(head_dim, 128)`.
  - **Layers Holding KV**: `10` GQA layers (`8` KV heads, `head_dim=64`) + `30` ShortConv layers (`conv_L_cache=3`).
  - **Utilization & Batching**: Not specified (`UNCLEAR`), `B = 1` single-stream.
  - **Kernels**: `tpu-inference==0.32.0` RPA v3 (`tpu_inference/kernels/ragged_paged_attention/v3/kernel.py`, requiring custom patch to remove `jnp.floating` check) + `gmm_v2` W4A16 (PR #3653/#3660) + `MODEL_IMPL_TYPE=vllm` (torchax).
- **2. Claimed Numbers & Fit Verdict (Converted to GiB = $2^{30}$ B)**:
  - **Usable HBM**: `15.748 GiB` (also written `15.75 GiB`).
  - **Weights**: Asserts 38 MoE expert blocks hold `~21.80B` params (`91%` of weights), and keeping `10` GQA + `30` short-conv layers in `W8A8 INT8` while routing experts to `W4A16` adds `+0.34 GiB`, for a total weight footprint of **`~11.67 GiB`**.
  - **KV Stream / Size at 200K**: Asserts at 200K context, INT8 KV streams `~2.05 GB/token` (`1.909 GiB`) vs `~1.23 GB/token` (`1.146 GiB`) for active W4A16 weights. Notes BF16 materialization of `[1, 8, 200000, 64]` is `410 MB` (`409.6 MB = 0.381 GiB`) per tensor.
  - **Activations / Compile / Reserve / Other**: Not quantified (`UNCLEAR` / ignored).
  - **Total & Verdict**: Asserts `~11.67 GiB` weights fits "comfortably under the 15.75 GiB usable HBM" once RPA v3 is patched for in-VMEM KV dequantization.
- **3. Skeptical Math Verification & Flagged Errors**:
  - **ERROR 1 (MoE Param Count & `11.67 GiB` Weight Math)**: These numbers (`21.80B` MoE params, `+0.34 GiB` delta, `11.67 GiB` total, `2.05 GB/tok` KV) are **just asserted** (copied directly from the user prompt of G4). Recomputing exact LFM2.5-24B-A2B MoE expert parameters: $38\text{ layers} \times 64\text{ experts} \times 3\text{ matrices} \times 2048 \times 1536 = \mathbf{22,951,231,488}$ params (`22.95B`, `95.7%` of `23.97B`), NOT `21.80B` (`21.80B` would be 36 MoE layers: $36 \times 64 \times 3 \times 2048 \times 1536 = 21.74\text{B}$). Furthermore, `11.67 GiB` ($11.33 + 0.34$) inherits G1's undercounted `11.33 GiB` W4A16 base!
  - **ERROR 2 (Ignored Effect of `align_to(head_dim, 128)` on KV Size)**: G2 quotes `get_kv_cache_shape` with `align_to(head_dim, 128)` on line 35, but fails to notice that padding `head_dim=64` to `128` **doubles** the physical KV cache size from `1.91 GiB` to **`3.81 GiB`** even in 8-bit unless 2 KV heads are folded into the 128-lane dimension!
- **4. Buckets Ignored**: Prefill/decode activations, compile/scratch workspace, vLLM reserve, KV scale overhead, and `head_dim` 128-alignment bloat.

---

### Group G3: `TPU v5e LFM2.5-24B-A2B Inference Optimization.md`
- **Date**: mtime `2026-10-31 18:03:02 +05:30` (internal citations through `2026-10-30`).
- **1. Proposed Config**:
  - **Component Precision**: Hybrid `compressed-tensors`:
    - `38` Routed MoE layers (`22.95B` params): `W4A16` symmetric INT4 (`group_size=32` to match QAD-Q4_0 grid, or `group_size=128`).
    - `30` ShortConv (`0.5035B`), `10` GQA Attention (`0.0629B`), `2` Dense SwiGLU FFNs (`0.1447B`): `W8A8` dynamic per-token INT8 (per-channel INT8 weights + dynamic per-token INT8 activations on 393 TOPS INT8 MXU).
    - Tied embeddings (`0.2621B`): `BF16` (`0.488 GiB`) or `INT4 group-32` (`emb4`, `0.138 GiB`).
    - Routers & RMSNorms (`0.0052B`): `BF16`/`FP32` (`0.010 GiB`).
  - **KV Dtype**: Evaluates **INT8 symmetric** (with BF16 group-64 scale) and **K8V4 mixed** (INT8 K + packed INT4 V + BF16 group-64 scales), dequantized per tile in 32 MB VMEM inside RPA v3.
  - **head_dim / Padding**: `head_dim = 64`. For K8V4 in RPA v3 (`[num_pages, page_size, num_kv_heads * 2, 64]` uint8/int8), slice $2h$ holds 64 INT8 K bytes; slice $2h+1$ holds 32 packed INT4 V bytes + 2B $s_K$ + 2B $s_V$ + 28 zero-pad bytes = `128 B/head/token` (`1,024 B/token/layer`), or compacted without the 28-byte pad to `800 B/token/layer`.
  - **Layers Holding KV**: `10` GQA layers hold KV cache; `30` ShortConv layers hold fixed state (`conv_L_cache = 3`, `0.00023 GiB` / `0.24 MiB` for 2 state taps, or `0.35 MiB` for 3 taps).
  - **Utilization & Batching**: `--gpu-memory-utilization 0.92` (`14.488 GiB` ceiling), `--max-num-batched-tokens 2048`, `MIN_TOKEN_BUCKET=64`, `--enable-chunked-prefill`, `--no-enable-prefix-caching`, `B = 1` (`--max-num-seqs 1`).
  - **Kernels**: Patched Pallas `ragged_paged_attention_v3` (native INT8 MXU `Q_int8 @ K_int8^T` with `preferred_element_type=jnp.int32` + VMEM V4 unpack), `gmm_v2` for W4A16 MoE, `VllmCompressedTensorsW8A8Int8LinearMethod` for backbone.
- **2. Claimed Numbers & Fit Verdict (Converted to GiB = $2^{30}$ B)**:
  - **Usable HBM**: `15.748 GiB` (`14.488 GiB` at `0.92` utilization; `1.260 GiB` reserved for XLA).
  - **Weights** (broken down by component):
    - 38 MoE experts (`22,951,231,488` params): **`12.024 GiB`** in W4A16 `group_size=32` (incl. BF16 scales); **`10.688 GiB`** in W4A16 `group_size=128` (raw, omits scales; **`11.022 GiB`** with BF16 scales).
    - 30 ShortConv Linear (`503,500,800` params, W8A8): **`0.469 GiB`**.
    - 10 GQA Attention Linear (`62,914,560` params, W8A8): **`0.059 GiB`**.
    - 2 Dense FFNs (`144,703,488` params, W8A8): **`0.135 GiB`**.
    - Tied Embeddings (`262,144,000` params): **`0.488 GiB`** (BF16) or **`0.138 GiB`** (`emb4` INT4 g32).
    - Routers & Norms (`5,190,208` params, BF16): **`0.010 GiB`**.
    - **Subtotals**:
      - `g32 MoE + BF16 emb`: **`13.185 GiB`**
      - `g32 MoE + emb4`: **`12.835 GiB`**
      - `g128 MoE (raw) + emb4`: **`11.499 GiB`** (`11.833 GiB` with g128 scales)
  - **KV Bytes/Token & Total at 200K**:
    - BF16: `20,480 B/tok` -> `4.096 GB` = **`3.815 GiB`**.
    - INT8 + BF16 group-64 scales: `10,560 B/tok` -> `2.112 GB` = **`1.967 GiB`**.
    - K8V4 + BF16 group-64 scales (unpadded): `8,000 B/tok` -> `1.600 GB` = **`1.490 GiB`**.
  - **Other (ShortConv state)**: `0.00023 GiB` (`0.24 MiB`).
  - **Compile / Scratch / vLLM Reserve**: `1.260 GiB` (the unallocated `8%` above `0.92` utilization).
  - **Activations / Prefill Workspace**: Ignored in byte sum (subsumed into the `1.260 GiB` unallocated margin).
  - **Total & Verdict**:
    - `g32 MoE + emb4 + INT8 KV`: `12.835 + 1.967 = 14.802 GiB` (exceeds `14.488 GiB` `0.92` cap by `0.314 GiB`, leaving only `0.946 GiB` for XLA -> **TIGHT / RISKS OOM AT 0.92**).
    - `g32 MoE + emb4 + K8V4 KV`: `12.835 + 1.490 = 14.325 GiB` (< `14.488 GiB` cap, `1.423 GiB` below `15.748 GiB` -> **FITS SAFELY**).
    - `g128 MoE + emb4 + INT8 KV`: `11.833 (w/ scales) + 1.967 = 13.800 GiB` (< `14.488 GiB` cap -> **FITS SAFELY**).
- **3. Skeptical Math Verification & Flagged Errors**:
  - **Verified Derivations**:
    - MoE `group_size=32` W4A16 with BF16 scales: $22,951,231,488 \times (0.5 + 2/32)\text{ B} = 12,910,067,712\text{ B} = \mathbf{12.0234\text{ GiB}}$. **Exact.**
    - INT8 KV with BF16 scales: $200,000 \times 10 \times 2 \times 8 \times (64 + 2)\text{ B} = 2,112,000,000\text{ B} = \mathbf{1.9670\text{ GiB}}$. **Exact.**
    - K8V4 KV with BF16 scales: $200,000 \times 10 \times 8 \times (64 + 32 + 4)\text{ B} = 1,600,000,000\text{ B} = \mathbf{1.4901\text{ GiB}}$. **Exact.**
  - **ERROR 1 (Omitted `0.334 GiB` Scales in `10.688 GiB` Group-128 MoE Figure)**: $22,951,231,488 \times 0.5\text{ B} = 11,475,615,744\text{ B} = 10.6875\text{ GiB}$ is pure 4-bit payload without scales! Adding BF16 `group_size=128` scales ($22,951,231,488 / 128 \times 2\text{ B} = 358,612,992\text{ B} = \mathbf{0.3340\text{ GiB}}$) raises `g128` MoE weights to **`11.022 GiB`**.
  - **ERROR 2 (RPA v3 K8V4 64-Byte Padded Slot vs `1.490 GiB` Claim)**: In Section 2's RPA v3 layout, each head's K slice is 64 B and V+scales+pad slice is padded to 64 B (`128 B/head/tok`), which physically allocates **`1.907 GiB`** in HBM (`2,048,000,000 B`), NOT `1.490 GiB` (`1.490 GiB` requires separate unpadded `v_cache_packed` `[..., 32]` and scale arrays rather than padding V+scales to 64 B).
  - **ERROR 3 (Attention Parameter Count Undercount by 2x)**: G3 claims 10 GQA layers have `62,914,560` params (`0.059 GiB`). Let's check: per GQA layer, `q_proj` is $2048 \times 2048 = 4,194,304$, `k_proj` is $2048 \times 512 = 1,048,576$, `v_proj` is $2048 \times 512 = 1,048,576$, and `o_proj` is $2048 \times 2048 = 4,194,304$. Sum per layer = $10,485,760$ params; across 10 layers = **`104,857,600` params (`0.098 GiB` in INT8)**, NOT `62,914,560`! (G3 forgot `o_proj` — $6,291,456 \times 10 = 62,914,560$ is only Q+K+V!).
  - **ERROR 4 (Dense FFN Parameter Count)**: For 2 dense layers (`moe_intermediate_size=1536` vs `intermediate_size`), G3 claims `144,703,488` params (`0.135 GiB`), which corresponds to `intermediate_size = 11776` ($2 \times 3 \times 2048 \times 11776 = 144,703,488$).
- **4. Buckets Ignored**: Explicit prefill activation buffer size (`max_num_batched_tokens=2048`); `0.334 GiB` group-128 MoE scales in the `10.688 GiB` row; `o_proj` in attention weights (`0.039 GiB`); RPA v3 28-byte per-head padding waste (`0.417 GiB`) if stored in a uniform 64-wide array.

---

### Group G4: `We-Are-Designing-200K-TPU-Pipeline-Transcription.md` (and `.pdf`)
- **Date**: `.md` mtime `2026-10-31 18:14:21 +05:30`; `.pdf` mtime `2026-10-31 18:11:19 +05:30` (internal header: `31/10/2026, 18:11`).
- **1. Proposed Config**:
  - **Component Precision**: Hybrid `compressed-tensors` (`format: "pack-quantized"`):
    - `group_0_w8a8_int8`: `W8A8` (symmetric per-channel INT8 weights + dynamic per-token INT8 activations) for `self_attn`, `conv`/`short_conv` (`in_proj`, `out_proj`), and layers `0–1` dense FFN (`w1, w2, w3`).
    - `group_1_w4a16_moe`: `W4A16` symmetric INT4 with **`group_size = 128`** (re-blocked from `QAD-Q4_0` `group_size=32` so `gmm_v2` avoids a 2.6x VPU cycle regression) for layers `2–39` `mlp.experts`.
    - `ignore` (kept in `BF16`): `lm_head`, `model.embed_tokens`, `conv.conv`, `gate`, norms.
  - **KV Dtype**: Custom Pallas **`K8V4`** ring/paged cache (`k_cache` `[8, 200704, 64] int8`, `v_cache_packed` `[8, 200704, 32] int8`, `k_scale_cache` & `v_scale_cache` `[8, 200704, 1] bfloat16`), with native INT8 MXU `Q_int8 @ K_int8^T` (`preferred_element_type=jnp.int32`) and in-VMEM V4 shift-unpacking (`b_kv = 512`).
  - **head_dim / Padding**: `head_dim = 64`; in Pallas kernel, the 4 GQA query heads per KV head are padded `4 -> 128` along M-axis inside VMEM (`[1, 8, 128, 64]`) to satisfy Pallas sublane rules; KV storage itself is unpadded along `head_dim` (`64` for K, `32` bytes packed for V), with `max_seq_len` padded to `200,704` (`392` tiles of `512`).
  - **Layers Holding KV**: `10` GQA layers hold K8V4 KV cache; `30` ShortConv layers hold `[num_cache_lines, 2048, 2]` BF16 state (`MambaSpec`, `245.76 KB` at `B=1`).
  - **Utilization & Batching**: Static buckets `(B=1, chunk=2048)`, single-stream `B = 1` (`max_num_seqs = 1`).
  - **Kernels**: Custom `pallas_k8v4_flash_decode` via `torchax.interop.call_jax`, pure-XLA `tpu_xla_short_conv_forward` monkey-patch (replaces Triton `causal_conv1d_fn`), `VllmCompressedTensorsW8A8Int8`, and `VllmCompressedTensorsMoEMethod` (`gmm_v2`).
- **2. Claimed Numbers & Fit Verdict (Converted to GiB = $2^{30}$ B)**:
  - **Usable HBM**: `15.748 GiB` (rounded to `15.75 GiB` in summary table).
  - **Weights** (Section 2 text vs Section 5 table):
    - 38 Routed MoE layers (`22.96B` params, W4A16 `group_size=128`): quoted as `11.48 GB` in Section 2, and **`10.69 GiB`** in Section 5 table (`0.670 GB/tok` active stream).
    - 30 ShortConv + 10 GQA + 2 Dense FFN (`0.73B` params, W8A8 INT8): quoted as `0.73 GB` = **`0.68 GiB`** (`0.730 GB/tok`).
    - Tied `embed_tokens` / `lm_head` (`128000 × 2048`, BF16): quoted as `0.52 GB` (`0.524 GB`) = **`0.49 GiB`**.
    - **Total Weights**: **`11.85 GiB`** (`12.73 GB`, or `10.69 + 0.68 + 0.49 = 11.86 GiB` in table).
  - **KV Bytes/Token & Total at 200K**:
    - BF16: `4.096 GB` = **`3.815 GiB`**.
    - INT8 (+ BF16 scales): `2.112 GB` = **`1.967 GiB`**.
    - K8V4 (INT8 K `1.024 GB` + INT4 V `0.512 GB` + BF16 scales `0.064 GB`): `8,000 B/tok` -> `1.600 GB` = **`1.490 GiB`**.
  - **Other (30 ShortConv State Caches `[30, 2048, 2]` BF16)**: `245.76 KB` = **`0.00023 GiB`**.
  - **Activations + Compile / Scratch (`XLA Executables + Scratchpad Buffers`, `B=1, chunk=2048`)**: **`1.65 GiB`**.
  - **vLLM Reserve**: `0.75 GiB` unallocated margin (`15.00 GiB` used out of `15.75 GiB`, equivalent to `0.952` utilization).
  - **Total & Verdict**: `10.69 + 0.68 + 0.49 + 1.49 + 0.0002 + 1.65` = **`15.00 GiB / 15.75 GiB`** -> **FITS** (`0.75 GiB` spare).
- **3. Skeptical Math Verification & Flagged Errors**:
  - **ERROR 1 (Omitted `0.334 GiB` Group-128 MoE Scales)**: Just like G3, `11.48 GB` (`10.69 GiB`) is $22.95\text{B} \times 0.5\text{ B} = 11.476\text{ GB}$ (`10.688 GiB`), which **omits the BF16 group-128 scale tensors (`weight_scale`)** that the doc's own conversion script (`convert_q4_0_to_w4a16_ct`) writes to safetensors! Adding `weight_scale` (`0.334 GiB`) increases total weights from `11.86 GiB` to **`12.19 GiB`**, and total HBM from `15.00 GiB` to **`15.33 GiB`**.
  - **ERROR 2 (Dense FFN Dimension Error in Section 2)**: Section 2 computes 2 dense FFNs as $2 \times 3 \times 2048 \times 8192 = 0.101\text{B}$ params (assuming `intermediate_size=8192`), whereas G3 used `11776` (`0.145B`).
  - **ERROR 3 (Bug in `tpu_xla_short_conv_forward` Code)**: Lines 681–682 write:
    ```python
    B_gate = (B_gate * x).transpose(0, 1)
    bx = (B_gate * x).transpose(0, 1)
    ```
    This multiplies by `x` **twice** (`B_gate * x * x`) and fails with a shape mismatch if `T != 2048` because `B_gate` was already transposed to `[2048, T]` while `x` is `[T, 2048]`!
  - **ERROR 4 (Decode-Only Pallas Kernel)**: `pallas_k8v4_flash_decode` hardcodes `q: [1, 32, 64]` (`B=1`, 1 query token). It cannot run chunked prefill (`chunk=2048`), yet the doc provides no prefill attention kernel for the custom `(k_cache, v_cache_packed, k_scale_cache, v_scale_cache)` ring buffer.
- **4. Buckets Ignored**: Group-128 MoE scale tensors (`0.334 GiB`); norms/routers (`0.010 GiB`); prefill attention kernel/workspace for K8V4.

---

### Group G5: `TPU v5e vLLM LFM2-MoE Investigation.md` (and `.pdf`)
- **Date**: `.md` mtime `2026-11-01 00:53:52 +05:30`; `.pdf` mtime `2026-11-01 00:56:23 +05:30`.
- **1. Proposed Config**:
  - **Component Precision**: Hybrid `compressed-tensors`: `W8A8` per-channel INT8 for backbone projections (`CompressedTensorsW8A8Int8LinearMethod`) + `W4A16` **`group_size = 32`** packed INT4 for the `2,432` routed MoE experts (`38 layers × 64 experts`).
  - **KV Dtype**: Tested `float8_e4m3fn` (which it claims compressed KV to 1 byte/elem = `2.04 GiB` for `213,724` tokens, but suffered a 14 s VPU software emulation bottleneck during 200K prefill); recommends switching RPA v3 KV cache dtype to **`jnp.int8`** (`sint8 -> bf16` single-instruction VPU cast or native INT8 MXU `preferred_element_type=jnp.int32`).
  - **head_dim / Padding**: `head_dim = 64` (`8` KV heads); token padding buckets pruned from 10 buckets down to **`[16, 4096, 8192]`**.
  - **Layers Holding KV**: `10` GQA layers hold `FullAttentionSpec` paged KV; `30` ShortConv layers hold `MambaSpec` conv state (`[num_slots, 2048, 2]` BF16).
  - **Utilization & Batching**: `--max-num-batched-tokens 8192` (`25` prefill chunks of `8192` for a `199,936`-token prompt), `B = 1` single-stream (`MODEL_IMPL_TYPE=vllm` over torchax).
  - **Kernels**: `tpu-inference==0.32.0` RPA v3 (`ragged_paged_attention`), `VllmModelWrapper` (torchax), patched `_lookup_conv_cache` (fixes `n = T` padding corruption and `has_initial_states`).
- **2. Claimed Numbers & Fit Verdict (Converted to GiB = $2^{30}$ B)**:
  - **Usable HBM**: `15.75 GiB`.
  - **Weights**: **`12.45 GiB`** (W8A8 backbone + W4A16 `group_size=32` MoE experts).
  - **KV Bytes/Token & Total**: At `1 B/elem`, `10,240 B/tok` (`10 KiB/tok`). For `213,724` allocated tokens (`200K` + spare blocks), claims **`2.04 GiB`** (`2,188,533,760 B = 2.0382 GiB`) across the 10 GQA layers (`1.907 GiB` for exactly `200,000` tokens).
  - **Other / Activations / Compile / Reserve**: Noticeably, `12.45 + 2.04 = 14.49 GiB`, yet line 3 states `14.74 GiB of HBM (12.45 GiB weights + 2.04 GiB hybrid KV cache)` — leaving an unexplained **`0.25 GiB`** gap in its own addition! Leaves `1.01 GiB` (`15.75 - 14.74`) unallocated for activations/XLA.
  - **Total & Verdict**: Claims **`14.74 GiB / 15.75 GiB`** -> **FITS** (claims 200K prefill ran in `32.578 s` at `6,137 tok/s`).
- **3. Skeptical Math Verification & Flagged Errors**:
  - **ERROR 1 (Addition Mismatch `12.45 + 2.04 != 14.74`)**: $12.45 + 2.04 = \mathbf{14.49\text{ GiB}}$, not `14.74 GiB` (a `0.25 GiB` phantom discrepancy).
  - **ERROR 2 (Weight Size `12.45 GiB` vs `group_size=32` vs `group_size=128`)**: In G5, `12.45 GiB` is claimed to be W8A8 backbone + W4A16 `group_size=32` experts. However, in G3 we proved that `group_size=32` MoE experts alone are `12.024 GiB`, plus `0.68 GiB` W8A8 backbone + `0.49 GiB` BF16 embeddings = **`13.19 GiB`**! And in G6 (the newest doc, prompt line 10), `12.45 GiB` is explicitly measured for **`group_size = 128`** (`W8A8 backbone + W4A16 experts, group 128, about 12.45 GiB in HBM, measured`), NOT `group_size=32`!
  - **ERROR 3 (Contradicts G2/G3/G4/G6 on `float8_e4m3fn` HBM Allocation & `head_dim` Padding)**: G5 claims `float8_e4m3fn` KV cache occupied only `2.04 GiB` (`1 byte/elem`, `head_dim=64`) in HBM. G6 (newest) proves that in `tpu-inference==0.32.0`, (a) Issue #3126 causes `fp8`/`fp8_e4m3` to allocate `bf16` or zero byte savings because RPA v3 pads `head_dim=64` to `align_to(64, 128) = 128` (`2048 B/tok/layer = 3.815 GiB`), and (b) vLLM's hybrid KV check overcounts the 30 ShortConv layers by `12,500×` (`2.861 GiB`), demanding `6.68 GiB` and refusing to start at 200K unless patched! Thus, G5's `213,724`-token `2.04 GiB` run could only have occurred with custom monkey-patches bypassing both the ShortConv block check and RPA v3 `head_dim=128` padding.
- **4. Buckets Ignored**: `8192`-token prefill activation workspace (at `max_num_batched_tokens=8192`, activations across MoE top-4 routing + `8192 × 6144` ShortConv `in_proj` require significant scratchpad memory inside the remaining `1.01 GiB`); KV scale buffers; compile scratch.

---

### Group G6 (Newest): `TPU-V5e-LFM2-5-KV-Cache-Optimization.md` (and `.pdf`)
- **Date**: `.md` mtime `2026-11-02 02:25:36 +05:30`; `.pdf` mtime `2026-11-02 02:23:16 +05:30` (internal header: `02/11/2026, 02:23`). **Newest document in the repository.**
- **1. Proposed Config**:
  - **Component Precision**: `compressed-tensors` `W8A8` backbone + `W4A16` MoE experts, **`group_size = 128`** (measured at **`12.45 GiB`** in HBM).
  - **KV Dtype**: Explains why `"auto"` (`bf16`), `"fp8"`, `"fp8_e4m3"`, and `"int8_per_token_head"` all fail to save bytes out-of-the-box on v5e RPA v3 in `v0.32.0`. Proposes two concrete fixes:
    - **Option A (BF16 KV + ShortConv capacity check fix + higher utilization)**: Keep `bf16` KV (`3.815 GiB` for 10 GQA layers at `head_dim=64`), patch `_check_enough_kv_cache_memory` / `KVCacheManager` so the 30 `MambaSpec` layers multiply by `max_num_seqs = 1` (`0.00023 GiB`) instead of `12,500` blocks (`2.861 GiB`), and raise `--gpu-memory-utilization` from `0.92` to **`0.97`** (`15.28 GiB` budget, leaving ~`2.83 GiB`–`4.00 GiB` free depending on weight overhead).
    - **Option B (FP8/INT8 KV + Head-Folding in RPA v3)**: Pass `jnp.float8_e4m3fn` (or `int8`) into `create_kv_caches` AND fold 2 KV heads (`head_dim=64`) into the 128-lane VREG axis (`num_kv_heads // 2, head_dim * 2 = 128`) in `get_kv_cache_shape` so `align_to(head_dim, 128)` does not pad `64 -> 128` with zeros, cutting 200K KV from `3.815 GiB` to **`1.907 GiB`**.
  - **head_dim / Padding**: Identifies the critical RPA v3 padding trap: `head_dim = 64` is padded 2x to `head_dim_padded = align_to(head_dim, 128) = 128` in `get_kv_cache_shape`, making 1-byte `fp8` with `head_dim_padded=128` (`2,048 B/tok/layer`) consume the exact same HBM as 2-byte `bf16` with `head_dim=64`!
  - **Layers Holding KV**: `10` GQA layers (`FullAttentionSpec`) + `30` ShortConv layers (`MambaSpec`, shape `(2, 2048)` BF16 = `8,192 B/layer`).
  - **Utilization & Batching**: Prompt tested `gpu_memory_utilization = 0.92` (which gave `3.57 GiB` free KV budget and failed); recommends `0.96`–`0.97`, compilation buckets **`[16, 8192]`**, `B = 1` (`max_num_seqs = 1`).
  - **Kernels**: `vllm-tpu==0.32.0` (`MODEL_IMPL_TYPE=vllm` torchax), RPA v3, `FusedMoE` (`gmm_v2`), plus 10-line `ShortConv` dynamic slice patch (`query_start_loc[1]`) for the `n = T` padding bug.
- **2. Claimed Numbers & Fit Verdict (Converted to GiB = $2^{30}$ B)**:
  - **Usable HBM**: **`15.748 GiB`**.
  - **Weights**: **`12.45 GiB`** in HBM (measured for W8A8 backbone + W4A16 `group_size=128` experts).
  - **Free HBM for KV at `0.92` Utilization**: **`3.57 GiB`** (measured available when engine checks KV capacity).
  - **KV Bytes/Token & Total at 200K (`12,500` blocks of `16` tokens)**:
    - Unpatched vLLM hybrid check (BF16):
      - `10` GQA layers: `32,768 B/block` (`2,048 B/tok/layer`) $\times 10 \times 12,500 = 4,096,000,000\text{ B}$ = **`3.8147 GiB`**.
      - `30` ShortConv layers (buggy block-multiplied check): `8,192 B/state` $\times 30 \times 12,500 = 3,072,000,000\text{ B}$ = **`2.8610 GiB`**.
      - Unpatched Total Demanded: $7,168,000,000\text{ B}$ = **`6.6757 GiB` (`6.68 GiB`)** -> **FAILS STARTUP** (`6.68 GiB > 3.57 GiB` free).
    - Patched ShortConv check (`max_num_seqs = 1`):
      - `30` ShortConv layers (true size): $30 \times 1 \times 8,192\text{ B} = 245,760\text{ B}$ = **`0.00023 GiB`**.
      - Patched BF16 KV Total: `3.8147 + 0.00023` = **`3.815 GiB`** (`4,096,245,760 B`). Still `0.245 GiB` above `3.57 GiB` at `0.92` utilization!
      - Patched 8-bit KV with 2-head folding (`head_dim * 2 = 128`): `1,024 B/tok/layer` = **`1.907 GiB`** (`2,048,000,000 B`) + `0.00023 GiB` ShortConv = **`1.908 GiB`** -> **FITS SAFELY** inside `3.57 GiB` free at `0.92` utilization!
  - **Total & Verdict**:
    - Unpatched `0.32.0`: **FAILS** (`6.68 GiB` needed vs `3.57 GiB` free).
    - With ShortConv check fix + 8-bit head-folded KV: `12.45 GiB (weights) + 1.91 GiB (KV) = 14.36 GiB` -> **FITS SAFELY** under `15.748 GiB`.
- **3. Skeptical Math Verification & Flagged Errors**:
  - **Verified Exact Derivation of `6.68 GiB`**:
    - Attention: $10 \times 200,000 \times 2 \times 8 \times 64 \times 2\text{ B} = 4,096,000,000\text{ B} / 2^{30} = \mathbf{3.814697\text{ GiB}}$.
    - ShortConv (overcounted by $200,000 / 16 = 12,500$ blocks): $30 \times 12,500 \times (2 \times 2048 \times 2\text{ B}) = 3,072,000,000\text{ B} / 2^{30} = \mathbf{2.861023\text{ GiB}}$.
    - Sum: $7,168,000,000\text{ B} / 2^{30} = \mathbf{6.675720\text{ GiB}} \approx \mathbf{6.68\text{ GiB}}$. **100% Mathematically Verified.**
  - **ERROR 1 (Arithmetic Inconsistency on Free Memory at `0.92` vs `0.97` Utilization)**:
    - At `gpu_memory_utilization = 0.92`, `0.92 × 15.748 GiB = 14.488 GiB`. Yet with `12.45 GiB` weights, the prompt observed **`3.57 GiB` free** for KV ($12.45 + 3.57 = 16.02\text{ GiB}$? No: $14.488 - 3.57 = 10.92\text{ GiB}$ or $15.28 - 11.28 = 4.00\text{ GiB}$). Why was `3.57 GiB` free at `0.92`? Because $14.49 - 3.57 = 10.92\text{ GiB}$ (if XLA/vLLM profiled free memory differently) OR at `0.97` utilization (`15.276 GiB`), adding `+0.05 × 15.748 = +0.787 GiB` to `3.57 GiB` gives **`4.36 GiB`** free (or `15.28 - 12.45 = 2.83 GiB`), whereas Line 201 claims `0.97` gives `4.00 GiB` free (`3.57 + 0.43` corresponds to `0.947` utilization, not `0.97`).
  - **ERROR 2 (Software Emulation of `float8_e4m3fn` on v5e)**: In Line 205, G6 suggests passing `kv_dtype=jnp.float8_e4m3fn` into RPA v3 on v5e. As G4 and G5 established, TPU v5e lacks native FP8 hardware instructions, so `float8_e4m3fn` triggers multi-instruction VPU software emulation slowdowns unless `jnp.int8` is used instead.
- **4. Buckets Ignored**: Does not separately itemize how much of the non-KV memory is XLA compile scratch vs prefill activation buffer (`buckets=[16, 8192]`), relying instead on the empirical `3.57 GiB` post-warmup free HBM report.

---

## 2. Summary of Bucket Coverage & Unit Discipline Across All 6 Doc Groups

*(All values converted to **GiB ($2^{30}$ bytes)** per TPU v5e-1 chip; usable HBM = `15.748 GiB`.)*

| Doc Group | Weights (GiB) | KV @ 200K (GiB) | ShortConv State (GiB) | Activations / Prefill Workspace | Compile / Scratch | vLLM Reserve (Unallocated HBM) | Total Claimed (GiB) | Used `GB` ($10^9$) vs `GiB` ($2^{30}$)? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **G1** (`Analysis`) | `11.33` *(err: `11.51`)* | `1.91` *(err: `1.97`)* | `0.00034` | **IGNORED** | Included in `1.26` reserve | `1.26` (`0.92` util) | `14.50` *(true: `14.74+act`)* | Both (properly converts `10^9` to `2^30`, but undercounts scales) |
| **G2** (`deep-research`) | `11.67` *(asserted)* | `1.91` (`2.05 GB/tok`) | Mentioned (`3` tok) | **IGNORED** (`0.38 GiB` bug noted) | **IGNORED** | **IGNORED** | `UNCLEAR` (`< 15.75`) | Mixes `GB` (`2.05 GB`) and `GiB` (`11.67 GiB`) |
| **G3** (`Optimization`) | `12.84` (`g32+emb4`)<br>`11.50` (`g128 raw`) | `1.97` (`INT8`)<br>`1.49` (`K8V4`) | `0.00023` | **IGNORED** | Included in `1.26` reserve | `1.26` (`0.92` util) | `14.33` (`g32+K8V4`)<br>`13.80` (`g128+INT8`) | Both (converts `GB` $\leftrightarrow$ `GiB`, omits `0.33 GiB` g128 scales) |
| **G4** (`Transcription`) | `11.86` *(err: `12.19` w/ scales)* | `1.49` (`K8V4`)<br>`1.97` (`INT8`) | `0.00023` | Combined w/ compile: `1.65 GiB` | `1.65` (incl. activations) | `0.75` (`15.00/15.75`) | `15.00` *(true: `15.33`)* | Both (`12.73 GB = 11.85 GiB`, `1.600 GB = 1.490 GiB`) |
| **G5** (`Investigation`) | `12.45` *(measured)* | `2.04` (`213.7K` tok, 8-bit) | Mentioned, unbudgeted | **IGNORED** (at `8192` chunk!) | **IGNORED** | `1.01` left (`15.75-14.74`) | `14.74` *(math err: `12.45+2.04=14.49`)* | `GiB` only |
| **G6** (`KV-Cache-Opt`, **Newest**) | `12.45` *(measured, g128)* | `6.68` (unpatched)<br>`3.81` (BF16)<br>`1.91` (8-bit folded) | `0.00023` (`2.86 GiB` before fix!) | Included in empirical warmup | Included in empirical warmup | `1.26` (`0.92` util) or `0.47` (`0.97` util) | `14.36` (8-bit folded + weights) | Both (`7,168,000,000 B = 6.68 GiB`) |

---

## 3. Key Contradictions Between Docs & Resolution by Chronology

1. **Weight Footprint of Hybrid W8A8 Backbone + W4A16 MoE (`11.67 GiB` vs `11.86 GiB` vs `12.45 GiB` vs `13.19 GiB`)**:
   - **G2** asserts `11.67 GiB`; **G3/G4** compute `11.85–11.86 GiB` for `group_size=128` (because they omitted `0.334 GiB` of BF16 `weight_scale` tensors and `0.039 GiB` of `o_proj` weights, which brings theoretical `g128` to `12.23 GiB`); **G5** claims `12.45 GiB` is `group_size=32` (contradicting G3's `13.19 GiB` for `g32`); **G6 (Newest)** clarifies that **`12.45 GiB`** is the **measured** HBM footprint of `W8A8 backbone + W4A16 experts, group 128` once loaded into TPU HBM (reflecting `12.23 GiB` theoretical tensors + `~0.22 GiB` XLA pad/alignment overhead).
2. **Why 200K KV Cache Fails Out-of-the-Box (`3.81 GiB` BF16 vs `6.68 GiB` Hybrid Overcount)**:
   - **G1–G4** assume BF16 KV at 200K requires `3.815 GiB` ($10 \times 200,000 \times 2,048\text{ B}$).
   - **G6 (Newest)** reveals that in `vllm-tpu==0.32.0`, `Lfm2MoeForCausalLM` actually requests **`6.68 GiB`** (`3.815 GiB` for 10 GQA layers + **`2.861 GiB`** for 30 ShortConv `MambaSpec` layers because `kv_cache_utils.py` multiplies `MambaSpec.page_size_bytes = 8,192 B` by `12,500` blocks instead of `max_num_seqs = 1`).
3. **8-Bit KV Cache Behavior in RPA v3 (`Issue #3126` Silent BF16 vs `head_dim=128` Padding vs `float8` Emulation)**:
   - **G2/G3/G4** say `--kv-cache-dtype fp8` silently falls back to `bfloat16` in HBM (Issue #3126).
   - **G5** claims `float8_e4m3fn` allocated `2.04 GiB` (`1 B/elem`) in HBM and caused a 14 s VPU software emulation slowdown.
   - **G6 (Newest)** reconciles the kernel layout: even if an 8-bit dtype reaches RPA v3's `get_kv_cache_shape`, RPA v3 aligns the last axis to `align_to(head_dim, 128) = 128`. Because `head_dim = 64` is padded 2x to `128`, an 8-bit KV cache **still consumes `2,048 B/tok/layer` (`3.815 GiB` at 200K)** unless two `head_dim=64` KV heads are folded into each 128-lane VREG (`num_kv_heads // 2, head_dim * 2`).
4. **MoE `group_size = 32` vs `group_size = 128`**:
   - **G2/G3/G5** claim `group_size=32` (native to `QAD-Q4_0.gguf`) works in `gmm_v2`.
   - **G4 & G6 (Newest)** conclude **`group_size = 128`** is required because `gmm_v2` tiles the reduction axis in multiples of 128 (so `group_size=32` causes a 2.6x VPU cycle regression or fails tile alignment, and costs an extra `1.00 GiB` of scale HBM).
5. **Prefill Chunk Size / Compilation Buckets (`2048` vs `8192`)**:
   - **G1/G3/G4** propose `max_num_batched_tokens = 2048`.
   - **G5 & G6 (Newest)** converge on `max_num_batched_tokens = 8192` with pruned compilation buckets **`[16, 8192]`** (or `[16, 4096, 8192]`) to keep cold XLA compile time manageable and minimize prefill iterations (25 chunks for 200K).

---

## 4. Converged Single Best Configuration Spec Table

Combining the newest empirical truths (**G6**) with the verified kernel & quantization findings (**G3/G4/G5**) yields the single best configuration the research converges on to fit `LFM2.5-24B-A2B` at `200,000` context (`B=1`) on a single TPU v5e-1 (`15.748 GiB` HBM):

| Parameter / Subsystem | Converged Target Specification | HBM Bucket (GiB = $2^{30}$ B) | Notes / Required Patch |
| :--- | :--- | :--- | :--- |
| **Chip & Usable HBM** | Single Colab TPU v5e-1 (`v5litepod-1`, `tp=1`) | **15.748 GiB** ceiling | `16 GB` physical; `15.748 GiB` XLA usable |
| **Engine & Execution Path** | `vllm-tpu==0.32.0` (`tpu-inference==0.32.0`), `MODEL_IMPL_TYPE=vllm` (torchax) | — | Uses `VllmModelWrapper`; load weights on CPU before `torchax.enable_globally()` to cut load from 120 s to <18 s |
| **MoE Experts (38 layers, 64 exp, top-4)** | `compressed-tensors` `W4A16` symmetric INT4, **`group_size = 128`** (re-blocked from `QAD-Q4_0.gguf`) | **11.022 GiB** (`10.688` raw + `0.334` BF16 scales) | Dispatched to Pallas `gmm_v2`; avoids `group_size=32`'s `+1.00 GiB` scale bloat and 2.6x VPU penalty |
| **Backbone Linear (10 GQA + 30 ShortConv + 2 Dense FFN)** | `compressed-tensors` `W8A8` (per-channel INT8 weight + dynamic per-token INT8 activation) | **0.702 GiB** (`0.469` Conv + `0.098` Attn + `0.135` FFN) | Runs on native 393 TOPS INT8 MXU (`VllmCompressedTensorsW8A8Int8`) |
| **Embeddings (`128000 × 2048` tied) + Norms/Gates** | `BF16` tied `embed_tokens` (`0.488 GiB`) + `BF16` norms/gates (`0.010 GiB`) *(optional: `emb4` saves `0.35 GiB`)* | **0.498 GiB** *(total weights = **12.45 GiB** measured in HBM)* | `12.45 GiB` includes `~0.23 GiB` XLA tensor alignment overhead |
| **KV Cache (10 GQA layers, 8 KV heads, `head_dim=64`, 200K tok)** | **Primary**: `INT8` (or `K8V4`) in RPA v3 with **2 KV heads folded per 128-lane VREG** (`[4, 128]`) + in-VMEM dequant<br>**Fallback**: Unpadded `BF16` (`head_dim=64`) at `util=0.97` | **1.907–1.967 GiB** (`INT8` folded)<br>*(or **1.490 GiB** `K8V4`; **3.815 GiB** `BF16`)* | Must fold 2 heads into 128 lanes in `get_kv_cache_shape` so `align_to(head_dim, 128)` does not pad `64 -> 128` and double HBM to `3.815 GiB` |
| **ShortConv State (30 layers, `conv_L_cache=3`)** | `BF16` state `(2, 2048)` per layer (`MambaSpec`), `B = 1` | **0.00023 GiB** (`245.76 KB`) | **Critical Patch 1**: Multiply `MambaSpec.page_size_bytes` by `max_num_seqs` (`1`), NOT `12,500` blocks (saves `2.861 GiB` phantom allocation)<br>**Critical Patch 2**: Slice `conv_state` at `query_start_loc[1]` (`n`), NOT padded bucket `T` |
| **Concurrency & Prefill Buckets** | `B = 1` (`--max-num-seqs 1`), `--max-model-len 200000`, `--max-num-batched-tokens 8192`, buckets **`[16, 8192]`** | **~0.88 GiB** (empirical warmup + activations) | Pruned buckets (`[16, 8192]`) + persistent `jax_compilation_cache_dir` |
| **vLLM / XLA Reserve (`gpu_memory_utilization`)** | `--gpu-memory-utilization 0.92` (`1.26 GiB` reserve) with 8-bit KV, or `0.96–0.97` (`0.47–0.63 GiB` reserve) with BF16 KV | **0.47–1.26 GiB** | At `0.92` util with `12.45 GiB` weights, `3.57 GiB` is free for KV — enough for `1.97 GiB` INT8 KV (`1.60 GiB` headroom), or `3.815 GiB` BF16 KV if util is raised to `0.96+` |
| **TOTAL HBM (200K, B=1)** | **Hybrid W8A8/W4A16 (g128) + Head-Folded INT8 KV + Patched ShortConv** | **14.42 GiB / 15.748 GiB** *(+ `1.33 GiB` XLA/activation margin)* | **VERDICT: FITS SAFELY** (only after applying the ShortConv block-count patch, ShortConv `n=T` slice patch, and RPA v3 head-folding / in-VMEM INT8 KV patch) |
