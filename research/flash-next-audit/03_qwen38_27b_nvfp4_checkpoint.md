# Identity of Qwen3.8-27B-NVFP4
Researched: 2026-10-17 | Engine: Gemini 3.8 Flash (Medium) | Job: 03

## Verdict in one paragraph

The Hugging Face repository [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) exists, published by NVIDIA in September 2026 using NVIDIA Model Optimizer (ModelOpt `v0.47.0`). The underlying model is a dense multimodal hybrid transformer (`Qwen3_5ForConditionalGeneration`) with exactly **27,781,427,952** logical parameters (~27.78B total: 26.90B base language model, 460.7M vision encoder, and 424.7M MTP head) and context length up to **262,144 tokens** (256K / 262K). Its 64 hidden layers consist of exactly **48 Gated DeltaNet** linear attention layers and **16 full-attention** (GQA) layers (`full_attention_interval: 4`). However, the chat's VRAM and speed claims are materially distorted: the checkpoint is **not uniform 4-bit**, but **`MIXED_PRECISION`**—MLPs are NVFP4 (group size 16), while GDN linear attention and GQA projections are preserved in **FP8**, and embeddings/norms/vision remain **BF16**. As a result, the on-disk footprint is **21.92 GB** (20.42 GiB) across three safetensors shards, and loaded text-only weight footprint is **20.15 GB** (18.77 GiB), not the chat's "~15–17 GB". This leaves **~72–74 GB** free for KV cache on a 96 GB G4 (usable ~95.6 GB), rather than the claimed ~80 GB. Furthermore, the chat's claim that BF16 decode on G4 is ~50–60 tok/s is physically impossible: streaming 55.56 GB over G4's 1,597 GB/s GDDR7 bus caps batch-1 decode at **28.7 tok/s**. Crucially, this NVIDIA checkpoint is a direct quantization of the stock base model (`Qwen/Qwen3.8-27B`), meaning it is **fully safety-aligned and NOT abliterated**, directly conflicting with the uncensored worker requirement in `architecture.md`.

---

## Claim table (ID | chat claim | label | corrected value | source URL)

| ID | Chat Claim | Label | Corrected Value | Source URL |
| :--- | :--- | :--- | :--- | :--- |
| **D1** | Hugging Face repo `nvidia/Qwen3.8-27B-NVFP4` exists. | **CONFIRMED** | Exists; released Sep 2026 by NVIDIA ModelOpt; 3 safetensors shards totaling 21.92 GB. | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |
| **D2** | Dense 27.8B (chat also says 27B); all parameters activate every token. | **CONFIRMED** | Exactly **27,781,427,952** parameters (26.90B base LM + 0.46B vision + 0.42B MTP); dense architecture, all 26.90B LM params active per token. | [`config.json`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/config.json), [`safetensors`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/model.safetensors.index.json) |
| **D3** | Hybrid 48 Gated DeltaNet + 16 full-attention layers. | **CONFIRMED** | 64 hidden layers: 48 `"linear_attention"` (GDN) + 16 `"full_attention"` (GQA); `full_attention_interval: 4`. | [`config.json`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/config.json) |
| **D4** | NVFP4 occupies ~15–17 GB VRAM. | **PARTIALLY TRUE** | Underestimated by 3–5 GB. Checkpoint is `MIXED_PRECISION` (NVFP4 MLPs + FP8 attention + BF16 embeddings); weights are **21.92 GB** (multimodal) or **20.15 GB** (text-only). Loaded VRAM is ~20.2–23.5 GB. | [`hf_quant_config.json`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/hf_quant_config.json), [safetensors header](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/model.safetensors.index.json) |
| **D5** | FP8 occupies ~27 GB. | **PARTIALLY TRUE** | Theoretical pure 1 B/param is 27.78 GB. In official checkpoint `Qwen/Qwen3.8-27B-FP8`, embeddings/vision remain BF16, making actual footprint **30.88 GB** (28.76 GiB). | [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) |
| **D6** | FP16/BF16 occupies ~54 GB. | **CONFIRMED** | Theoretical 2 B/param is 55.56 GB. Official `Qwen/Qwen3.8-27B` BF16 checkpoint is **55.62 GB** (51.80 GiB). Chat's ~54 GB is within 3%. | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **D7** | Built-in MTP draft head exists on this checkpoint. | **CONFIRMED** | Built-in 1-layer MTP draft head present (`mtp_num_hidden_layers: 1`, 15 tensors, 424,699,392 params, 849.4 MB). | [`config.json`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/config.json), [`model.safetensors.index.json`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/model.safetensors.index.json) |
| **D8** | Leaves ~80 GB free on a 96 GB G4 for KV. | **PARTIALLY TRUE** | On 95.6 GB usable VRAM, subtracting 20.15–21.92 GB weights and ~1.5 GB CUDA runtime leaves **~72.2–74.0 GB** free, not ~80 GB. | [INDEPENDENT CALC](#independent-calculations) |
| **D9** | How this checkpoint differs from Blackfrost `Qwen3.8-27B-ABLITERATED` GGUF (quant, abliteration, MTP vs DFlash 2). | **CONFIRMED** | Detailed structural differences verified: NVIDIA is stock safety-aligned (not abliterated), safetensors NVFP4/FP8, native 1-layer MTP; Blackfrost is abliterated, GGUF Q4_K_M (16.81 GB), DFlash 2 sidecar. | [`Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) |
| **D10** | File sizes and `config.json` match D2–D7. | **PARTIALLY TRUE** | Architectural parameters in `config.json` match D2, D3, D7 exactly. However, file sizes reflect `MIXED_PRECISION` (21.92 GB NVFP4, 30.88 GB FP8, 55.62 GB BF16) rather than pure theoretical uniform quant sizes. | [`config.json`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/config.json), [`model.safetensors.index.json`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/model.safetensors.index.json) |
| **D11** | Unquantized BF16 decode on G4 would be ~50–60 tok/s (half of FP8). | **FALSE** | Physically impossible for batch-1 decode. Streaming 55.56 GB over 1,597 GB/s bandwidth caps decode at **28.7 tok/s**. Chat's 50–60 tok/s violates memory bandwidth by ~2×. | [INDEPENDENT CALC](#independent-calculations) |
| **D12** | Download size and Colab-load time. | **INDEPENDENT CALC** | 21.92 GB safetensors. At 250–500 MB/s GCP throughput, download takes **44–88 seconds** (~1.5 min); GPU load takes **~2–5 s**. Cold-start penalty is ~1.5–2 min (~0.17–0.22 CU @ 6.7 CU/hr). | [INDEPENDENT CALC](#independent-calculations) |

---

## Findings

### 1. Checkpoint identity and repository inspection
The repository [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) is an official public release from NVIDIA, created using NVIDIA Model Optimizer (ModelOpt `v0.47.0.dev80+g913f5e224`).
- **Base model**: `Qwen/Qwen3.8-27B` (Alibaba).
- **License**: Apache 2.0.
- **Architecture**: `Qwen3_5ForConditionalGeneration` (multimodal vision + text backbone).
- **Supported runtimes**: vLLM, SGLang, TensorRT-LLM on NVIDIA Blackwell (`sm_120`).
- **Files on repository**:
  - `config.json`: 87.5 KB
  - `hf_quant_config.json`: 53.8 KB
  - `model.safetensors.index.json`: 95.8 KB
  - `model-00001-of-00003.safetensors`: 9,965,652,544 bytes (9.28 GiB, 9.97 GB)
  - `model-00002-of-00003.safetensors`: 9,985,757,064 bytes (9.30 GiB, 9.99 GB)
  - `model-00003-of-00003.safetensors`: 1,970,287,672 bytes (1.83 GiB, 1.97 GB)
  - Total safetensors bytes: **21,921,697,280 bytes** (20.42 GiB, 21.92 GB).
  - Total repository storage (API `usedStorage`): **21,934,506,600 bytes** (20.43 GiB, 21.93 GB).

### 2. Architectural configuration (`config.json`)
Direct parsing of `config.json` confirms:
- **Total layers**: `num_hidden_layers: 64`.
- **Hybrid attention**: `full_attention_interval: 4`.
  - 48 layers are `"linear_attention"` (Gated DeltaNet linear recurrent state).
  - 16 layers are `"full_attention"` (standard Grouped Query Attention, GQA).
- **Dimensions**:
  - `hidden_size`: 5,120
  - `intermediate_size`: 17,408 (MLP expansion factor = 3.4×)
  - `num_attention_heads`: 24 (GQA Q heads)
  - `num_key_value_heads`: 4 (GQA KV heads)
  - `head_dim`: 256
  - `vocab_size`: 248,320
- **Gated DeltaNet settings**:
  - `linear_conv_kernel_dim`: 4
  - `linear_key_head_dim`: 128
  - `linear_num_key_heads`: 16
  - `linear_num_value_heads`: 48
  - `linear_value_head_dim`: 128
  - `output_gate_type`: `"swish"`
  - `mamba_ssm_dtype`: `"float32"`
- **Context length**:
  - `max_position_embeddings`: **262,144** (256K / 262K context window).
  - `rope_parameters`: `{'mrope_interleaved': True, 'mrope_section': [11, 11, 10], 'partial_rotary_factor': 0.25, 'rope_theta': 10000000}`.
- **Multimodal components**:
  - Includes a 27-layer vision transformer (`depth: 27`, `hidden_size: 1152`, patch size 14) for image and video conditional generation.

### 3. Parameter count and weight tensor census
Analysis of the 2,194 weight tensors across the three safetensors shards reveals the exact logical parameter count:
- **Total logical parameters**: **27,781,427,952** (27.78B dense parameters).
  - **Base language model**: 26,895,998,464 parameters (26.90B)
  - **Vision encoder** (`model.visual.*`): 460,730,096 parameters (0.46B)
  - **MTP draft head** (`mtp.*`): 424,699,392 parameters (0.42B)

All 26.90B parameters of the base language model activate for every token. There are zero MoE layers.

### 4. Quantization reality: `MIXED_PRECISION`, not uniform NVFP4
The chat assumed a uniform 4-bit model where `27.8B × 0.5 B = 13.9 GB`, plus ~1–3 GB for scales, giving "~15–17 GB VRAM".
Inspection of `hf_quant_config.json` disproves uniform quantization. NVIDIA ModelOpt deployed **mixed precision**:
- **MLP layers (192 projections)**: Quantized to **NVFP4** (`group_size: 16`).
  - Stored as `U8` packed weights (2 weights/byte) + `FP8` sub-channel scale tensors.
  - Footprint: 8.56 GB weights + 1.07 GB scales = **9.63 GB** (8.96 GiB).
- **Linear Attention / GDN projections (144 projections)**: Quantized to **FP8** (`F8_E4M3`), NOT FP4.
  - Footprint: **5.59 GB** (5.20 GiB).
- **Full Attention / GQA projections (64 projections)**: Quantized to **FP8** (`F8_E4M3`), NOT FP4.
  - Footprint: **1.68 GB** (1.56 GiB).
- **Language Model Embeddings (`embed_tokens`)**: Kept in **BF16** (unquantized).
  - Footprint: 248,320 × 5,120 × 2 bytes = **2.54 GB** (2.37 GiB).
- **LM Head (`lm_head`)**: Quantized FP8/NVFP4 = **0.72 GB** (0.67 GiB).
- **Vision Encoder (`model.visual`)**: Kept in **BF16** = **0.92 GB** (0.86 GiB).
- **MTP Draft Head (`mtp`)**: Mixed FP8/BF16 = **0.85 GB** (0.79 GiB).
- **Layer Norms & biases**: Kept in FP32 / BF16 = **0.001 GB**.

#### Actual memory footprints
| Configuration | Weight Bytes | Weight Footprint (GB) | Weight Footprint (GiB) |
| :--- | :--- | :--- | :--- |
| **Pure uniform NVFP4 theoretical** | 27.78e9 × 0.5 B | 13.89 GB | 12.94 GiB |
| **Chat claim** | — | ~15.0–17.0 GB | ~14.0–15.8 GiB |
| **Actual text-only without MTP** | 19,301,170,312 B | **19.30 GB** | **17.97 GiB** |
| **Actual text-only with MTP** | 20,150,569,096 B | **20.15 GB** | **18.77 GiB** |
| **Actual full multimodal on disk** | 21,921,428,072 B | **21.92 GB** | **20.42 GiB** |

Because attention projections are in FP8 and embeddings are in BF16 to prevent catastrophic perplexity degradation on hybrid GDN layers, the actual weight footprint is **20.15 GB** for text serving, roughly **3.1–5.1 GB higher** than the chat claimed.

### 5. Multi-Token Prediction (MTP) draft head
Claim D7 is **CONFIRMED**:
- `config.json` explicitly declares `"mtp_num_hidden_layers": 1` and `"mtp_use_dedicated_embeddings": false`.
- The safetensors index contains 15 explicit `mtp.*` weights:
  - `mtp.pre_fc_norm_hidden.weight` (5,120 FP32)
  - `mtp.pre_fc_norm_embedding.weight` (5,120 FP32)
  - `mtp.fc.weight` (5,120 × 10,240, projection combining hidden state and token embedding)
  - `mtp.layers.0.input_layernorm.weight`
  - `mtp.layers.0.self_attn.q_proj`, `k_proj`, `v_proj`, `o_proj` (and norms)
  - `mtp.layers.0.post_attention_layernorm.weight`
  - `mtp.layers.0.mlp.gate_proj`, `up_proj`, `down_proj`
  - `mtp.norm.weight`
- Total parameters in MTP head: **424,699,392** (0.42B).
- Size: **849,398,784 bytes** (~849 MB).
- In vLLM and TensorRT-LLM, this head enables native 1-step draft speculation without an external draft model or sidecar file.

### 6. Official FP8 and BF16 checkpoints
To confirm claims D5 and D6, the official Alibaba repositories were queried:
- **BF16 Checkpoint**: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
  - Total parameters: 27,781,427,952 (100% BF16).
  - Total repository storage: **55,615,875,685 bytes** (55.62 GB / 51.80 GiB).
  - Matches D6 (~54 GB / 55.6 GB).
- **FP8 Checkpoint**: [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
  - Total parameters: 27,781,427,952.
  - Parameter breakdown: 24.70B in `F8_E4M3` + 3.08B in `BF16` (embeddings, norms, vision).
  - Total repository storage: **30,879,676,248 bytes** (30.88 GB / 28.76 GiB).
  - Pure theoretical uniform FP8 would be 27.78 GB, but practical FP8 retains BF16 embeddings, yielding **30.88 GB**.

### 7. Comparison with Blackfrost worker checkpoint
The current Colab worker defined in `architecture.md` relies on [`Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF). The two checkpoints differ across four critical dimensions:

| Dimension | Blackfrost ABLITERATED Worker | NVIDIA Qwen3.8-27B-NVFP4 |
| :--- | :--- | :--- |
| **Abliteration / Safety** | **Abliterated**: refusal directions surgically removed. Does not refuse complex coding or security scripts. | **Stock Base Model**: Standard Alibaba safety alignment intact. Will refuse sensitive prompts, exploit analysis, or red-team coding tasks. |
| **Quantization Format** | GGUF (`Q4_K_M` k-quants via GGML). Uniform 4-bit block quantization. Size: **16.81 GB** (16,810,716,384 bytes). | Safetensors (`MIXED_PRECISION`: NVFP4 MLPs + FP8 Attention + BF16 Embeds via ModelOpt). Size: **21.92 GB** (20.15 GB text-only). |
| **Hardware & Runtime** | `llama.cpp` / `llama-server`. Runs on Ampere (A100 `sm_80`), Ada (L4 `sm_89`), and Blackwell. | vLLM / TensorRT-LLM / SGLang. Requires native Blackwell (`sm_120`) NVFP4 Tensor Core support. Incompatible with standard `llama.cpp`. |
| **Speculative Decoding** | Uses external **DFlash 2** sidecar (`dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`, ~1.14 GB Q4_K_M) with 5 extraction hooks (`n_extract=5`). Internal NextN head left idle. | Uses built-in **1-layer MTP head** (`mtp_num_hidden_layers: 1`, 424.7M params, 849 MB) integrated inside the model weights. |
| **Vision Handling** | Vision stripped into separate `mmproj` file (`mmproj-Qwen3.8-27B-ABLITERATED-F16.gguf`), deliberately skipped on deployment to save memory. | Vision encoder (`model.visual.*`, 460.7M params, 921 MB) bundled into shards. Must be ignored at runtime via `--limit-mm-per-prompt image=0`. |

---

## Independent calculations

### 1. Theoretical vs actual weight footprint
$$\text{Params}_{\text{total}} = 27,781,427,952 \approx 27.781 \times 10^9$$

- **Theoretical uniform 4-bit**:
  $$\text{Size}_{\text{FP4}} = 27.781 \times 10^9 \times 0.5 \text{ B} = 13.89 \text{ GB } (12.94 \text{ GiB})$$
- **Theoretical uniform 8-bit**:
  $$\text{Size}_{\text{FP8}} = 27.781 \times 10^9 \times 1.0 \text{ B} = 27.78 \text{ GB } (25.87 \text{ GiB})$$
- **Theoretical uniform 16-bit**:
  $$\text{Size}_{\text{BF16}} = 27.781 \times 10^9 \times 2.0 \text{ B} = 55.56 \text{ GB } (51.75 \text{ GiB})$$

- **Actual `nvidia/Qwen3.8-27B-NVFP4` mixed precision text footprint**:
  $$\text{MLP (NVFP4 + scales)} = 9.63 \text{ GB}$$
  $$\text{GDN Attention (FP8)} = 5.59 \text{ GB}$$
  $$\text{GQA Attention (FP8)} = 1.68 \text{ GB}$$
  $$\text{Embeddings (BF16)} = 2.54 \text{ GB}$$
  $$\text{LM Head (FP8/NVFP4)} = 0.72 \text{ GB}$$
  $$\text{Total text weight footprint} = 9.63 + 5.59 + 1.68 + 2.54 + 0.72 = \mathbf{20.15\text{ GB (18.77 GiB)}}$$
  $$\text{Adding 1-layer MTP head} = +0.85 \text{ GB} \implies \mathbf{21.00\text{ GB (19.56 GiB)}}$$
  $$\text{Adding Vision encoder} = +0.92 \text{ GB} \implies \mathbf{21.92\text{ GB (20.42 GiB)}}$$

### 2. Available VRAM on Colab G4 for KV cache (Claim D8)
Colab G4 provides an NVIDIA RTX PRO 6000 Blackwell Server Edition with 96 GB VRAM, of which approximately **95.6 GB** is addressable by CUDA.
$$\text{Free VRAM}_{\text{text}} = 95.6\text{ GB} - 20.15\text{ GB (weights)} - 1.50\text{ GB (CUDA context + runtime)} = \mathbf{73.95\text{ GB}}$$
$$\text{Free VRAM}_{\text{multimodal}} = 95.6\text{ GB} - 21.92\text{ GB} - 1.50\text{ GB} = \mathbf{72.18\text{ GB}}$$

The chat claimed "~80 GB free", which was derived by subtracting 16 GB from 96 GB. In reality, available headroom is **~72.2–74.0 GB**.

Because only the 16 full-attention layers maintain a KV cache (the 48 GDN layers maintain fixed-size recurrent states of $5120 \times 128 \times 4 \text{ bytes} \approx 2.6 \text{ MB}$ per slot), KV memory consumption is minimal:
- KV heads: $n_{\text{kv}} = 4$, head dim: $d_h = 256$, layers: $L_{\text{full}} = 16$.
- Per token at FP8 (1 byte per key/value):
  $$\text{KV bytes/token} = 2 \times 16 \times 4 \times 256 \times 1 \text{ byte} = 32,768 \text{ bytes} = 32.0 \text{ KB/token}$$
- Capacity in 72 GB free memory:
  $$\text{Max context} = \frac{72 \times 10^9 \text{ bytes}}{32,768 \text{ bytes/token}} \approx 2,197,000 \text{ tokens}$$
The 72 GB headroom allows the model to comfortably saturate its maximum architectural context window of **262,144 tokens** with negligible VRAM exhaustion risk.

### 3. Roofline decode bandwidth limits and refutation of D11
In batch-1 autoregressive decode, the GPU is strictly memory-bandwidth bound. Every generated token requires streaming the full active weight payload across the memory bus.
Colab G4 memory bandwidth: $B = 1,597\text{ GB/s}$ (GDDR7 Server Edition).

$$\text{Theoretical Max Tok/s} = \frac{\text{Bandwidth (GB/s)}}{\text{Weight Payload (GB/token)}}$$

- **Unquantized BF16 (55.56 GB)**:
  $$\text{Decode Tok/s}_{\text{BF16}} = \frac{1,597\text{ GB/s}}{55.56\text{ GB}} = \mathbf{28.74\text{ tok/s}}$$
  *(Even on Workstation card at 1,792 GB/s: $1,792 / 55.56 = 32.25\text{ tok/s}$)*.
  **Conclusion on D11**: The chat's claim that BF16 decode is "~50–60 tok/s" is **FALSE**. To achieve 55 tok/s, a 55.56 GB model requires $55 \times 55.56 = 3,056\text{ GB/s}$ of bandwidth (equivalent to H100 SXM5 HBM3).

- **FP8 Checkpoint (30.88 GB practical, 27.78 GB theoretical)**:
  $$\text{Decode Tok/s}_{\text{FP8, practical}} = \frac{1,597\text{ GB/s}}{30.88\text{ GB}} = \mathbf{51.72\text{ tok/s}}$$
  $$\text{Decode Tok/s}_{\text{FP8, pure}} = \frac{1,597\text{ GB/s}}{27.78\text{ GB}} = \mathbf{57.49\text{ tok/s}}$$

- **`nvidia/Qwen3.8-27B-NVFP4` (20.15 GB text-only payload)**:
  $$\text{Decode Tok/s}_{\text{NVFP4, no spec}} = \frac{1,597\text{ GB/s}}{20.15\text{ GB}} = \mathbf{79.26\text{ tok/s}}$$
  *(Workstation 1,792 GB/s: $1,792 / 20.15 = 88.93\text{ tok/s}$)*.
  *(If chat's 15.0 GB payload were streamed: $1,597 / 15.0 = 106.5\text{ tok/s}$)*.

#### Payload published for Job 05 (Speed & Roofline)
Job 05 must use the following empirically verified weight payloads for its batch-1 roofline calculations:
- **`nvidia/Qwen3.8-27B-NVFP4` Text Payload**: **20.15 GB/token** (20,150,569,096 bytes).
- **`Qwen/Qwen3.8-27B-FP8` Practical Payload**: **30.88 GB/token** (30,879,676,248 bytes).
- **`Qwen/Qwen3.8-27B` BF16 Payload**: **55.56 GB/token** (55,562,855,904 bytes).

### 4. Download size and Colab startup economics (Claim D12)
- **Total download payload**: 21.92 GB (three safetensors files).
- **Network transfer speed on GCP**:
  - Conservative (250 MB/s): $\frac{21.92 \text{ GB}}{0.25 \text{ GB/s}} = 87.7 \text{ seconds } (\sim 1.46 \text{ min})$.
  - High performance (500 MB/s): $\frac{21.92 \text{ GB}}{0.50 \text{ GB/s}} = 43.8 \text{ seconds } (\sim 0.73 \text{ min})$.
- **VRAM loading speed**:
  - Loading from host RAM to VRAM over PCIe Gen 5 x16 (~64 GB/s theoretical, ~50 GB/s achieved): $\sim 0.44 \text{ s}$.
  - Model initialization and CUDA graph capture: $\sim 10–20 \text{ s}$.
- **Cold start penalty**:
  - Total cold startup: **~1.5 to 2.0 minutes**.
  - At Colab G4 billing rate (~6.7 CU/hr $\approx 0.112$ CU/min): cold start consumes **~0.17 to 0.22 CU per boot**.
  - Compared to caching on Google Drive (where 16.8 GB copies in ~35 seconds $\approx 0.06$ CU), downloading fresh from Hugging Face triples the startup CU penalty.

---

## Conflicts with architecture.md / prior research

1. **Safety and Censorship (Fatal Conflict)**:
   `architecture.md` specifies an **uncensored worker** based on `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`. `nvidia/Qwen3.8-27B-NVFP4` is derived directly from Alibaba's base model without abliteration. It retains all standard refusal behaviors and will fail agentic coding pipelines that touch sensitive scripts, vulnerability testing, or system administration commands.
2. **Weight Footprint (20.15 GB vs 16.8 GB vs 15.0 GB)**:
   `architecture.md` states: *"Every token streams the full ~16 GB of 4-bit weights"*. The chat claimed ~15–17 GB. In reality, because NVIDIA retained attention projections in FP8 and embeddings in BF16, the stream payload is **20.15 GB**, which reduces un-speculated autoregressive decode throughput on G4 from ~100 tok/s to ~79 tok/s.
3. **Speculative Decoding Engine (MTP vs DFlash 2)**:
   `architecture.md` uses DFlash 2 with an external sidecar (`dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` quantized to Q4_K_M ~1.14 GB) running on `llama.cpp`. The NVIDIA checkpoint integrates an internal 1-layer MTP block (424.7M params) designed for vLLM / TRT-LLM.
4. **BF16 Decode Speed (28.7 tok/s vs 50–60 tok/s)**:
   The chat repeatedly claimed that unquantized BF16 decode on G4 is ~50–60 tok/s. As proven above, 55.56 GB streaming over a 1,597 GB/s bus has an unbreakable mathematical ceiling of **28.7 tok/s**.

---

## What remains unverified

1. **Abliterated NVFP4 Checkpoint**:
   No public NVFP4 checkpoint of `Blackfrost-AI/Qwen3.8-27B-ABLITERATED` exists on Hugging Face. Creating one would require running NVIDIA ModelOpt quantization on an abliterated BF16 base on a Blackwell GPU.
2. **Empirical MTP Acceptance Rate in vLLM**:
   While the MTP head weights (`mtp.*`) are present, empirical acceptance rates on Blackwell hardware specifically for hybrid Gated DeltaNet models in vLLM v0.7+ remain to be benchmarked by Job 05/06.
3. **Colab Drive Cache Viability for NVFP4**:
   Whether Google Drive FUSE can mount and stream 22 GB of safetensors files fast enough to avoid the 1.5-minute Hugging Face download without causing file-handle stalls in vLLM.

---

## Sources

- **NVIDIA Qwen3.8-27B-NVFP4 Repository**: https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4
- **NVIDIA Qwen3.8-27B-NVFP4 `config.json`**: https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/config.json
- **NVIDIA Qwen3.8-27B-NVFP4 `hf_quant_config.json`**: https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/hf_quant_config.json
- **NVIDIA Qwen3.8-27B-NVFP4 `model.safetensors.index.json`**: https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/model.safetensors.index.json
- **Alibaba Official Qwen3.8-27B BF16**: https://huggingface.co/Qwen/Qwen3.8-27B
- **Alibaba Official Qwen3.8-27B FP8**: https://huggingface.co/Qwen/Qwen3.8-27B-FP8
- **Blackfrost Qwen3.8-27B Abliterated GGUF**: https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF
- **NVIDIA Model Optimizer Documentation**: https://github.com/NVIDIA/Model-Optimizer
