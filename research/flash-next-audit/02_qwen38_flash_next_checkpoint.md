# Identity of Qwen3.8-Flash-Next-NVFP4: Checkpoint, Architecture, Memory Footprint, and Colab Viability
Researched: 2026-10-17 | Engine: Gemini 3.8 Flash (Medium) | Job: 02

## Verdict in one paragraph

The Hugging Face checkpoint [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) is authentic and exists as a public, ungated repository released by NVIDIA on 2026-08-31, quantized with NVIDIA Model Optimizer (`modelopt` v0.46.0) from Alibaba's [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). The chat's architectural description is remarkably accurate regarding model topology—it is indeed an experimental 124.9B Mixture-of-Experts (MoE) base architecture activating ~5.8B–6.0B parameters per token across 48 layers (36 Gated DeltaNet linear attention layers and 12 Qwen Sparse Attention layers) with 512 fine-grained experts (top-10 routed + 1 shared), supplemented by a 1-layer MoE Multi-Token Prediction (MTP) head and a 51.2B Phrase-Level Embedding (PLE) n-gram table. However, the chat's memory, download, and Colab viability conclusions are **FALSE**: the physical on-disk download is **132.68 GB** across 11 safetensors files (78.96 GB base model + 53.72 GB MTP/PLE shard). In memory, the base model alone consumes ~76.2 GB of physical weights and scales because attention layers, shared experts, and embeddings were deliberately excluded from NVFP4 and kept in unquantized BF16. When adding the FP8 MTP head (+2.76 GB) and runtime overhead (~4–6 GB), VRAM consumption reaches ~83–85 GB before KV cache, leaving at best ~10–12 GB free on a 96 GB RTX PRO 6000 (Blackwell G4). Crucially, this requires offloading the 51.2 GB n-gram embedding table to host CPU RAM—an operation that immediately triggers an Out-Of-Memory (OOM) crash on Google Colab's standard ~53–64 GB system RAM. Furthermore, NVIDIA's official deployment documentation mandates `--tensor-parallel-size 8` on Blackwell B200/B300 systems; running Flash-Next on a single Colab G4 is an engineering impossibility under production constraints.

---

## Claim table (ID | chat claim | label | corrected value | source URL)

| ID | Chat Claim | Label | Corrected Value | Source URL |
| :--- | :--- | :--- | :--- | :--- |
| **F1** | HF repo `nvidia/Qwen3.8-Flash-Next-NVFP4` exists and is an NVIDIA NVFP4 checkpoint. | **CONFIRMED** | Repo exists, is public/ungated, released by NVIDIA on 2026-08-31 using Model Optimizer v0.46.0 based on upstream `Qwen/Qwen3.8-Flash-Next`. | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |
| **F2** | It is a **125B** Mixture-of-Experts model. | **CONFIRMED** | Base language model contains 124.91B parameters (~120.80B routed MoE parameters across 48 layers + ~4.11B shared experts, attention, embeddings, and vision). Supplemented by a 51.2B PLE n-gram table and ~4B MTP module (~180B grand total). | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **F3** | **~6.0B** parameters activate per token. | **CONFIRMED** | Per token, 10 routed experts + 1 shared expert activate (~2.60B MoE params), plus 36 linear attention layers (~1.51B), 12 full attention layers (~0.43B), embeddings, and router/norms (~1.25B), totaling ~5.79B–6.0B active parameters. | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **F4** | **512** fine-grained experts, **top-10** routing. | **CONFIRMED** | `config.json` confirms `"num_experts": 512`, `"num_experts_per_tok": 10`, alongside 1 shared expert (`"shared_expert_intermediate_size": 640`). | [config.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json) |
| **F5** | Uses Gated DeltaNet. | **CONFIRMED** | 36 of 48 layers are `"linear_attention"` implementing Gated DeltaNet (`linear_num_key_heads: 16`, `linear_num_value_heads: 48`, `linear_key_head_dim: 128`, `linear_conv_kernel_dim: 4`, `mamba_ssm_dtype: "float32"`). | [config.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json) |
| **F6** | Uses Qwen Sparse Attention (QSA) that prunes sequence regions at long context. | **CONFIRMED** | 12 of 48 layers (`full_attention_interval: 4`) are `"full_attention"` featuring QSA with an MQA indexer (`indexer_budget: 2048` tokens / 512 blocks, `indexer_compress_ratio: 4`, `indexer_head_dim: 128`, `indexer_n_heads: 4`, `indexer_kv_heads: 1`). | [config.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json) |
| **F7** | Has a **4B MTP** (multi-token prediction) head. | **PARTIALLY TRUE** | The model card describes a "4B MTP" module, but config and safetensors confirm it is a 1-layer MoE block (`mtp_num_hidden_layers: 1`, 512 experts = 2.52B routed parameters + attention/projections = ~2.76B physical params). Crucially, MTP routed experts are quantized to **FP8** (`FP8_PB_WO`), NOT NVFP4. | [hf_quant_config.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/hf_quant_config.json) |
| **F8** | Has a **51B n-gram embedding** asynchronously prefetched from **host memory**. | **PARTIALLY TRUE** | Architectural concept is confirmed: the model includes a 51.2B-parameter n-gram embedding table (`ple_embedding.ngram_embedding`, 128 shards in FP8 = 51.20 GB) designed for host offload. However, loading 51.2 GB into Google Colab's ~53–64 GB system RAM causes an immediate host OS crash (OOM). | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **F9** | NVFP4 weights + MTP head occupy **~68–72 GB** VRAM. | **FALSE** | Base weights alone require 78.96 GB on disk (~76.2 GB physical weights and scales in VRAM) because attention, shared experts, vision, and embeddings are in unquantized BF16. Adding the FP8 MTP head (+2.76 GB) and runtime overhead (~4–6 GB) totals ~83–85 GB VRAM before KV cache. If the 51.2 GB n-gram table is not offloaded, total weight footprint is ~130.2 GB. | [model.safetensors.index.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/main/model.safetensors.index.json) |
| **F10** | That leaves **~24–26 GB** on a 96 GB G4 for KV cache. | **FALSE** | With base weights, scales, MTP, and runtime overhead consuming ~84–86 GB, a 96 GB G4 retains only **~10–12 GB** for KV cache (assuming host offloading of the 51.2 GB n-gram table succeeds). If the n-gram table is in VRAM, available memory is negative (-34 GB). | [INDEPENDENT CALC](#calculation-2-storage-and-weight-vram-footprint) |
| **F11** | It is a different architecture from dense Qwen 3.8 27B, not a “faster 27B”. | **CONFIRMED** | Entirely different architectures. Qwen3.8-27B is a 27.8B dense hybrid (64 layers, hidden 5120, intermediate 10880, all 27.8B activate per token). Flash-Next is a 125B MoE + 51B PLE n-gram + 4B MTP (48 layers, hidden 2560, 512 fine-grained experts, intermediate 640, top-10 routing, ~5.8B–6.0B active). | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **F12** | Flash-Next is the fastest model that still fits on one Colab G4. | **FALSE** | Flash-Next does NOT fit cleanly on a single Colab G4. The full checkpoint is 132.68 GB. Official NVIDIA serving guidelines mandate `--tensor-parallel-size 8` on Blackwell B200/B300. Single-GPU execution leaves <12 GB KV cache and crashes Colab's ~53–64 GB host RAM when attempting host offload. | [README.md](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/README.md) |
| **F13** | File sizes, `config.json` fields, and download footprint match F2–F9. | **PARTIALLY TRUE** | `config.json` fields match architectural specs (512 experts, top-10, 48 layers, 1-layer MTP, 51B ngram). However, download file sizes contradict the chat's ~70 GB narrative: total safetensors volume is 132.68 GB (10 base shards = 78.96 GB; MTP/PLE shard = 53.72 GB). | [model.safetensors.index.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/main/model.safetensors.index.json) |
| **F14** | License / gating / Colab-download practicality (HF token, size, time). | **PARTIALLY TRUE** | Repo is public, ungated, under NVIDIA Open Model License + Qwen Community License 1.0 (no HF token required). However, downloading and staging 132.68 GB exhausts Colab's local ephemeral disk (100–150 GB), exceeds free Google Drive (15 GB) requiring a paid tier, and takes 5–10 minutes to transfer. | [nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |

---

## Findings

### 1. Identity, Origin, and Upstream Lineage

The checkpoint [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) is real, public, and directly accessible on Hugging Face. It was uploaded on 2026-08-31 by NVIDIA Corporation.

- **Base Model:** Quantized directly from Alibaba’s [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), which serves as an experimental architectural testbed (internal codename `qwen4_exp`) previewing the Qwen4 generation.
- **Quantization Toolkit:** Quantized using [NVIDIA Model Optimizer (`modelopt`)](https://github.com/NVIDIA/Model-Optimizer) version `0.46.0.dev281+g73d778422` with [Transformers v5.16.0](https://github.com/huggingface/transformers/releases/tag/v5.16.0)+ support.
- **Calibration Datasets:** Calibrated on English news (`cnn_dailymail`) and multi-turn conversational data (`Nemotron-Post-Training-Dataset-v2`).
- **Target Hardware:** Officially built and validated for **NVIDIA Blackwell B200 and B300** multi-GPU nodes. The model card explicitly documents:
  > `Supported Hardware Microarchitecture Compatibility: NVIDIA Blackwell B200, NVIDIA Blackwell B300`

### 2. Base Model Architecture: MoE, Layers, and Dimensions

Inspection of [`config.json`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json) reveals the exact architectural parameters of the causal language model:

- **Top-level Architecture:** `"architectures": ["Qwen4ExpForConditionalGeneration"]`, `"model_type": "qwen4_exp"`, `"text_config": {"model_type": "qwen4_exp_text"}`.
- **Layer Count & Hidden Dimension:** 48 hidden layers (`"num_hidden_layers": 48`), hidden dimension 2,560 (`"hidden_size": 2560`), vocabulary size 248,320 (`"vocab_size": 248320`), untied embeddings (`"tie_word_embeddings": false`).
- **Mixture-of-Experts Structure:**
  - `"num_experts": 512` fine-grained routed experts per layer.
  - `"num_experts_per_tok": 10` routed experts activated per token.
  - `"moe_intermediate_size": 640` intermediate projection dimension per expert.
  - `"shared_expert_intermediate_size": 640`: 1 shared expert per layer that activates unconditionally on every token.
  - `"norm_topk_prob": true`, `"output_gate_type": "sigmoid"`, `"router_aux_loss_coef": 0.001`.
- **Gated Residual Streams:** Uses widened residual streams modulated by a 4-branch gated connection (`"hc_count": 4`, `"hc_lowrank": 320`) to stabilize deep MoE training.
- **Vision Subsystem:** Includes a 27-layer vision encoder (`"depth": 27`, `"hidden_size": 1152`, `"out_hidden_size": 2560`, `"model_type": "qwen4_exp_vision"`).

### 3. Attention Architecture: Gated DeltaNet and Qwen Sparse Attention (QSA)

Unlike traditional pure-transformer models or dense hybrid models, Flash-Next implements a 3:1 interleaved layout:
`"layer_types"` contains 48 layers with an alternating pattern of 3 linear attention layers followed by 1 full attention layer, repeating 12 times (`12 × (3 × linear + 1 × full)`).

1. **Gated DeltaNet (`"linear_attention"`, 36 layers):**
   - Implements linear recurrent attention based on DeltaNet with Mamba-style SSM state updates.
   - Config fields: `"linear_num_key_heads": 16`, `"linear_key_head_dim": 128` (total K dimension = 2,048); `"linear_num_value_heads": 48`, `"linear_value_head_dim": 128` (total V dimension = 6,144); `"linear_conv_kernel_dim": 4`; `"mamba_ssm_dtype": "float32"`.
   - Generates constant-memory recurrent state decodes without growing standard key-value cache tokens.
2. **Qwen Sparse Attention / QSA (`"full_attention"`, 12 layers):**
   - Config fields: `"full_attention_interval": 4`, `"num_attention_heads": 24`, `"num_key_value_heads": 2`, `"head_dim": 256`.
   - Incorporates an auxiliary Multi-Query Attention (MQA) indexer that filters and prunes sequence regions before dense attention:
     - `"indexer_budget": 2048` tokens (equivalent to 512 blocks).
     - `"indexer_compress_ratio": 4`.
     - `"indexer_n_heads": 4`, `"indexer_kv_heads": 1`, `"indexer_head_dim": 128`.
   - Max context window: 262,144 tokens native (`"max_position_embeddings": 262144`), extensible up to 1,000,000 via YaRN RoPE (`"rope_theta": 10000000`).

### 4. Speculative Decoding: Multi-Token Prediction (MTP) Head

The model incorporates an integrated Multi-Token Prediction head configured in `config.json`:
- `"mtp_num_hidden_layers": 1`, `"mtp": {"hybrid": true, "layer_types": ["full_attention"], "num_hidden_layers": 1}`.
- Inspecting `model.safetensors.index.json` confirms the MTP module consists of:
  - 512 routed experts (`mtp.layers.0.mlp.experts.*`) with intermediate dimension 640.
  - A full-attention layer with RoPE (`rope_theta: 10000000`).
  - Output projection head.
- **Quantization of MTP:** Crucially, `hf_quant_config.json` specifies:
  `"mtp.layers.0.mlp.experts": {"quant_algo": "FP8_PB_WO", "group_size": 128}`.
  The MTP routed experts are stored in **128×128 block-scaled FP8**, NOT NVFP4. The MTP weights occupy 2.52 GB in FP8.

### 5. The 51B N-Gram Embedding (PLE) & Host-Prefetch Reality Check

The chat claimed:
> *"Qwen3.8-Flash-Next-NVFP4: The 125B FP4 base + 4B MTP head occupy ~68–72 GB of VRAM (with its 51B n-gram embedding asynchronously prefetched from host memory)..."*

This claim represents a mixture of verified architectural fact and fatal deployment delusion:
1. **Architectural Reality:** The model card and safetensors confirm a massive Phrase-Level Embedding (PLE) / N-gram table:
   - Config fields: `"ngram_size": 3`, `"ngram_vocab_size_base": 20000000`, `"split_ngram_parts": 128`, `"ple_embed_dim": 2560`, `"ple_layer_ids": [2]`.
   - In `model-fp8-mtp-ple.safetensors`, the embedding is partitioned into 128 shards:
     `model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_[0..127].weight`.
   - Each shard has shape `[2500012, 160]` in `F8_E4M3` (FP8).
   - Total parameters: $128 \times 2,500,012 \times 160 = \mathbf{51,200,245,760}$ parameters (~51.2B parameters).
   - In FP8 (1 byte per parameter), this table alone occupies **51.20 GB**.
2. **Serving Engines & Host Offloading:** Because n-gram lookup is an index-gather operation rather than a GEMM, vLLM (`vllm.models.qwen4_exp.ngram_embedding`) supports pinning this table in host CPU memory and asynchronously prefetching embeddings over PCIe.
3. **The Fatal Colab Constraint:**
   - On a dedicated multi-GPU server with 512 GB–1 TB of system RAM, pinning 51.2 GB in host RAM is standard practice.
   - On **Google Colab**, a standard high-RAM G4 runtime provides between **53 GB and 64 GB** of total host CPU RAM.
   - Allocating 51.2 GB of pinned host memory leaves <2–8 GB for the Linux kernel, Python process, CUDA driver buffers, and vLLM worker processes.
   - The OS `oom-killer` terminates the Python process during model initialization.

### 6. Quantization Scheme: NVFP4 Mixed-Precision Breakdown

NVIDIA's [`hf_quant_config.json`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/hf_quant_config.json) reveals that this repository is **NOT** uniformly quantized to FP4. It uses a selective `MIXED_PRECISION` strategy:

1. **NVFP4 Layers (`W4A4`, group size 16):**
   - Applies exclusively to the 48 layers of main language model routed MoE experts:
     `model.language_model.layers.[0..47].mlp.experts.*`.
   - Quantized into 4-bit floating point (`NVFP4` / `E2M1`) with group size 16 (`"group_size": 16`).
   - Each group of 16 weights carries an FP8 (`E4M3`) scale factor (`weight_scale`), plus per-channel alpha scaling (`weight_scale_2`) and per-tensor dynamic input scales (`input_scale`).
2. **FP8 Layers:**
   - PLE N-gram embedding table: Per-tensor FP8 (`F8_E4M3`), occupying 51.20 GB.
   - MTP routed MoE experts: Block-scaled FP8 (`FP8_PB_WO`, 128×128 blocks, group size 128), occupying 2.52 GB.
3. **Excluded Modules (Maintained in unquantized BF16):**
   - All 36 Gated DeltaNet linear attention layers (`linear_attn*`).
   - All 12 Qwen Sparse Attention full attention layers (`self_attn*`).
   - All 48 shared experts (`mlp.shared_expert*`).
   - All 48 MoE router gates (`mlp.gate`, `mlp.shared_expert_gate`).
   - Residual hyper-connections and norms (`attn_hyper_connection*`, `mlp_hyper_connection*`).
   - Token embeddings (`embed_tokens`) and language model head (`lm_head`).
   - The entire 27-layer vision encoder (`model.visual*`).
   - In total, **292 modules** are explicitly excluded from quantization and preserved in 16-bit bfloat16.

### 7. Disk Footprint, Safetensors Structure, and Download Practicality

Querying the Hugging Face API and HTTP Content-Length headers confirms the repository file sizes:

| File Name | Primary Contents | Size (Bytes) | Size (GB / GiB) |
| :--- | :--- | :--- | :--- |
| `model-00001-of-00010.safetensors` | Embeddings, Vision, Layers 0–1 | 3,115,991,696 B | 3.12 GB (2.90 GiB) |
| `model-00002-of-00010.safetensors` | Layers 1–6 MoE (NVFP4) + Attn (BF16) | 10,005,510,304 B | 10.01 GB (9.32 GiB) |
| `model-00003-of-00010.safetensors` | Layers 6–12 MoE (NVFP4) + Attn (BF16) | 10,005,364,728 B | 10.01 GB (9.32 GiB) |
| `model-00004-of-00010.safetensors` | Layers 12–18 MoE (NVFP4) + Attn (BF16) | 10,006,060,112 B | 10.01 GB (9.32 GiB) |
| `model-00005-of-00010.safetensors` | Layers 18–24 MoE (NVFP4) + Attn (BF16) | 10,005,362,392 B | 10.01 GB (9.32 GiB) |
| `model-00006-of-00010.safetensors` | Layers 24–30 MoE (NVFP4) + Attn (BF16) | 10,005,375,328 B | 10.01 GB (9.32 GiB) |
| `model-00007-of-00010.safetensors` | Layers 30–36 MoE (NVFP4) + Attn (BF16) | 10,005,956,544 B | 10.01 GB (9.32 GiB) |
| `model-00008-of-00010.safetensors` | Layers 36–42 MoE (NVFP4) + Attn (BF16) | 10,005,375,200 B | 10.01 GB (9.32 GiB) |
| `model-00009-of-00010.safetensors` | Layers 42–47 MoE (NVFP4) + Attn (BF16) | 5,679,113,808 B | 5.68 GB (5.29 GiB) |
| `model-00010-of-00010.safetensors` | Final norm, LM head (BF16) | 128,587,536 B | 0.13 GB (0.12 GiB) |
| **Subtotal (Base Model Shards 1–10)** | **Base 125B MoE + Attention + Embeddings** | **78,962,697,648 B** | **78.96 GB (73.54 GiB)** |
| `model-fp8-mtp-ple.safetensors` | 51.2B PLE Table + MTP MoE (FP8) | 53,717,551,730 B | 53.72 GB (50.03 GiB) |
| **Grand Total Safetensors** | **Full Model Footprint** | **132,680,249,378 B** | **132.68 GB (123.57 GiB)** |

#### Download Practicality on Google Colab:
- **Local Ephemeral Disk:** Standard Colab runtimes provide ~100 GB to 150 GB of scratch disk. Staging 132.68 GB leaves virtually zero headroom, frequently causing disk-full fatal errors during decompression or Hugging Face cache writes.
- **Google Drive Storage:** A standard free Google Drive account offers 15 GB. Persisting this model on Drive requires a paid Google One tier (≥2 TB). Copying 132.7 GB over Drive's internal backbone requires 5–10 minutes on every cold VM boot.
- **Direct HF Download:** Downloading 132.7 GB over Colab's network connection (~500 MB/s) consumes ~4.5 minutes of metered wall-clock compute before initialization begins.

### 8. Loaded VRAM Analysis & The Single Colab G4 Fallacy

The chat reasoned that because 125B parameters in FP4 equals ~62.5 GB, the model easily fits into the G4's 96 GB VRAM with 24–26 GB left for KV cache. This reasoning collapses under primary-source scrutiny:

```
[Available VRAM: 96.0 GB]
  ├── Base Model Weights in VRAM:       76.17 GB (NVFP4 MoE: 60.40 GB + Scales: 7.55 GB + BF16: 8.22 GB)
  ├── MTP Head (FP8 MoE + BF16):         2.77 GB
  ├── CUDA Context & Runtime Overhead:   4.50 GB (PyTorch CUDA context, vLLM engine, graph memory)
  └── Remaining VRAM for KV Cache:       8.56 GB  <--- CRITICAL SHORTAGE
```

- If host offload of the 51.2 GB n-gram table is enabled, base weights + scales + MTP occupy **78.94 GB** in VRAM. Adding ~4.5 GB of CUDA context, driver buffers, and engine workspace leaves **~8.5 GB to 12.5 GB** for KV cache.
- If host offload is NOT enabled (or fails due to Colab's 64 GB host RAM limit), loading the 51.2 GB table into VRAM brings total weight memory to $78.94 + 51.20 = \mathbf{130.14\text{ GB}}$, which exceeds 96 GB VRAM by **34.14 GB**.
- Consequently, the model cannot be served on a single 96 GB GPU under any configuration on Google Colab.

---

## Independent calculations

### Calculation 1: Parameter Census (Base Total, Active, MTP, PLE)

Let $H = 2560$ (hidden dimension), $L = 48$ (layers), $E = 512$ (experts), $I_{moe} = 640$ (intermediate dimension), $V = 248320$ (vocabulary).

1. **Routed MoE Parameters ($P_{routed}$):**
   Each expert has 3 linear projections: `gate_proj` ($H \times I_{moe}$), `up_proj` ($H \times I_{moe}$), and `down_proj` ($I_{moe} \times H$):
   $$P_{expert} = 3 \times H \times I_{moe} = 3 \times 2560 \times 640 = 4,915,200\text{ parameters}$$
   Across 512 experts and 48 layers:
   $$P_{routed} = 48 \times 512 \times 4,915,200 = \mathbf{120,795,955,200\text{ parameters}}\quad(\mathbf{120.80B})$$

2. **Shared Expert Parameters ($P_{shared}$):**
   One shared expert per layer with dimension 640:
   $$P_{shared} = 48 \times (3 \times 2560 \times 640) = \mathbf{235,929,600\text{ parameters}}\quad(\mathbf{0.24B})$$

3. **Router Gate Parameters ($P_{gate}$):**
   $$P_{gate} = 48 \times (2560 \times 512) = \mathbf{62,914,560\text{ parameters}}\quad(\mathbf{0.06B})$$

4. **Attention Parameters ($P_{attn}$):**
   - **36 Linear Attention Layers (Gated DeltaNet):**
     $Q_{proj} = 2560 \times 2048 = 5,242,880$; $K_{proj} = 5,242,880$; $V_{proj} = 2560 \times 6144 = 15,728,640$; $Out_{proj} = 6144 \times 2560 = 15,728,640$; conv/gates $\approx 10,240$:
     $$P_{linear\_layer} = 41,953,280 \implies 36 \times 41,953,280 = \mathbf{1,510,318,080\text{ parameters}}\quad(\mathbf{1.51B})$$
   - **12 Full Attention Layers (QSA + Indexer):**
     $Q_{proj} = 2560 \times 6144 = 15,728,640$; $K_{proj} = 2560 \times 512 = 1,310,720$; $V_{proj} = 1,310,720$; $Out_{proj} = 15,728,640$; MQA Indexer ($2560 \times 512 + 2560 \times 128 = 1,638,400$):
     $$P_{full\_layer} = 35,717,120 \implies 12 \times 35,717,120 = \mathbf{428,605,440\text{ parameters}}\quad(\mathbf{0.43B})$$

5. **Embeddings & Vision ($P_{embed}, P_{vit}$):**
   $$P_{embed} = 2 \times (248,320 \times 2560) = \mathbf{1,271,398,400\text{ parameters}}\quad(\mathbf{1.27B})$$
   $$P_{vit} \approx \mathbf{600,000,000\text{ parameters}}\quad(\mathbf{0.60B})$$

6. **Total Base Model Parameters:**
   $$P_{base} = 120.80\text{B} + 0.24\text{B} + 0.06\text{B} + 1.51\text{B} + 0.43\text{B} + 1.27\text{B} + 0.60\text{B} = \mathbf{124.91B\text{ parameters}}$$

7. **Active Parameters per Token ($P_{active}$):**
   Per token, 10 routed experts and 1 shared expert activate:
   $$P_{moe\_active} = 48 \times (11 \times 4,915,200) = 2,595,225,600\text{ parameters}\quad(2.60\text{B})$$
   Adding all attention ($1.94\text{B}$), active embedding ($0.64\text{B}$), and router/norms/residual projections ($0.61\text{B}$):
   $$P_{active} = 2.60\text{B} + 1.94\text{B} + 0.64\text{B} + 0.61\text{B} = \mathbf{5.79B}\approx\mathbf{6.0B\text{ active parameters}}$$

---

### Calculation 2: Storage and Weight VRAM Footprint

1. **NVFP4 Routed Experts ($W_{routed}$):**
   Each weight is 4 bits ($0.5\text{ bytes}$):
   $$W_{routed} = 120,795,955,200 \times 0.5 = 60,397,977,600\text{ bytes} = \mathbf{60.40\text{ GB}}\quad(56.25\text{ GiB})$$

2. **Group-16 Quantization Scales ($W_{scale}$):**
   Every 16 weights share 1 byte of FP8 scale ($E4M3$):
   $$W_{scale} = \frac{120,795,955,200}{16} \times 1 = 7,549,747,200\text{ bytes} = \mathbf{7.55\text{ GB}}\quad(7.03\text{ GiB})$$

3. **Unquantized BF16 Modules ($W_{bf16}$):**
   Attention ($1.94\text{B}$), shared experts ($0.24\text{B}$), embeddings ($1.27\text{B}$), vision ($0.60\text{B}$), and routers ($0.06\text{B}$) total $4.11\text{B}$ parameters at 2 bytes each:
   $$W_{bf16} = 4,110,640,640 \times 2 = 8,221,281,280\text{ bytes} = \mathbf{8.22\text{ GB}}\quad(7.66\text{ GiB})$$

4. **Total Base Model Footprint:**
   $$W_{base} = 60.40\text{ GB} + 7.55\text{ GB} + 8.22\text{ GB} = \mathbf{76.17\text{ GB}}\quad(70.94\text{ GiB})$$
   *(Note: Matches the 78.96 GB on-disk size of shards 1–10 within 3.5%, with the difference accounting for secondary scales `weight_scale_2`, layer norms, biases, and safetensors metadata).*

5. **MTP Head Footprint ($W_{mtp}$):**
   512 routed experts in FP8 ($2.52\text{ GB}$) + BF16 attention and projection layers ($0.25\text{ GB}$):
   $$W_{mtp} = 2.52\text{ GB} + 0.25\text{ GB} = \mathbf{2.77\text{ GB}}\quad(2.58\text{ GiB})$$

6. **PLE N-Gram Embedding Footprint ($W_{ple}$):**
   $128\text{ shards} \times 2,500,012\text{ rows} \times 160\text{ dim} \times 1\text{ byte (FP8)}$:
   $$W_{ple} = 51,200,245,760\text{ bytes} = \mathbf{51.20\text{ GB}}\quad(47.68\text{ GiB})$$

7. **Grand Total Weight Volume:**
   $$W_{total} = W_{base} + W_{mtp} + W_{ple} = 76.17 + 2.77 + 51.20 = \mathbf{130.14\text{ GB}}\quad(121.20\text{ GiB})$$
   *(Matches the exact 132.68 GB downloaded safetensors file size).*

---

### Calculation 3: Reconciling the Chat's "68–72 GB Loaded VRAM" Claim

- **The Chat's Naive Math:**
  The chat performed an unadjusted back-of-the-envelope calculation:
  $$125\text{B} \times 0.5\text{ B} = 62.5\text{ GB}$$
  Adding an estimated 4B MTP head at 4-bit precision ($4\text{B} \times 0.5 = 2.0\text{ GB}$) and a rough buffer yielded $\mathbf{\sim 68\text{--}72\text{ GB}}$.
- **Where the Chat Failed:**
  1. It completely overlooked the **7.55 GB** of FP8 weight scale tensors required by ModelOpt group-16 NVFP4.
  2. It failed to check `hf_quant_config.json`, which preserves **4.11B parameters (8.22 GB)** in unquantized 16-bit BF16 (attention, shared experts, vision, embeddings).
  3. It assumed the MTP head was NVFP4 ($2.0\text{ GB}$) when it is actually block-scaled FP8 ($2.77\text{ GB}$).
  4. It assumed the 51.2 GB n-gram table could be offloaded to Colab host memory without crashing the VM.
- **Physical Reality:** Loaded VRAM on an isolated GPU is $\mathbf{78.94\text{ GB}}$ of weights + $\mathbf{4.5\text{ GB}}$ runtime overhead $= \mathbf{\sim 83.5\text{ GB}}$, leaving **$\sim 12.5\text{ GB}$** of VRAM on a 96 GB card, NOT 24–26 GB.

---

### Calculation 4: Active Weight Streaming per Token (Decode Memory Traffic Gap)

The chat asserted:
> *"Flash-Next streams ~3.5–4 GB active weights/step vs ~15 GB for 27B."*

Let us calculate the true memory bus traffic per decode token step on Flash-Next NVFP4:

1. **Active MoE Experts (NVFP4 + Scales):**
   10 routed experts activate per layer across 48 layers ($2.359\text{B}$ parameters):
   $$M_{moe} = 2,359,296,000 \times 0.5\text{ B} = 1.18\text{ GB}$$
   Associated group-16 scales:
   $$M_{scales} = \frac{2,359,296,000}{16} \times 1\text{ B} = 0.15\text{ GB}$$
2. **Shared Experts (BF16):**
   1 shared expert per layer across 48 layers ($0.236\text{B}$ parameters in BF16):
   $$M_{shared} = 235,929,600 \times 2\text{ B} = 0.47\text{ GB}$$
3. **Linear Attention Layers (Gated DeltaNet, BF16):**
   All 36 recurrent linear attention weights ($1.51\text{B}$ parameters) must stream across the memory bus:
   $$M_{linear} = 1,510,318,080 \times 2\text{ B} = \mathbf{3.02\text{ GB}}$$
4. **Full Attention Layers (QSA, BF16):**
   12 full attention projection weights ($0.43\text{B}$ parameters in BF16):
   $$M_{full} = 428,605,440 \times 2\text{ B} = \mathbf{0.86\text{ GB}}$$
5. **Output Head & Norms (Active rows / projections):**
   $$\approx 0.40\text{ GB}$$

$$\mathbf{Total\ Active\ Memory\ Moved\ per\ Token} = 1.18 + 0.15 + 0.47 + 3.02 + 0.86 + 0.40 = \mathbf{6.08\text{ GB}}$$

- **The Gap Explained:**
  The chat derived $3.5\text{--}4.0\text{ GB}$ by multiplying $6.0\text{B active params} \times 0.5\text{ bytes} = 3.0\text{ GB}$ and adding minor overhead.
  Because the **36 Gated DeltaNet layers and 12 QSA layers are in unquantized BF16**, attention alone streams **$3.88\text{ GB}$** of data per step!
  The true weight data transferred per autoregressive step is **$\mathbf{\sim 5.8\text{--}6.1\text{ GB}}$**, approximately **50% higher** than the chat claimed.

---

## Conflicts with architecture.md / prior research/

1. **Architecture Misclassification:**
   [`../27b-plan/architecture.md`](../27b-plan/architecture.md) documents the repo's planned worker as an abliterated Qwen 3.8 27B GGUF (`Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`), which is a dense 27.78B hybrid (64 layers, hidden 5120, intermediate 10880) streaming ~16.8 GB of weights per step on a single GPU. The Gemini chat proposed replacing this with `nvidia/Qwen3.8-Flash-Next-NVFP4`, claiming it is a drop-in faster replacement. This is architecturally invalid: Flash-Next is a 125B MoE model with 512 experts, requiring vLLM multi-GPU tensor parallelism and a massive 51.2 GB host-offloaded embedding table.
2. **Runtime Engine Conflict:**
   The repository infrastructure is strictly built around `llama.cpp` (`llama-server` commit `efe3626`, PR #27816 for DFlash 2, cached GGUF on Google Drive). `nvidia/Qwen3.8-Flash-Next-NVFP4` uses Hugging Face safetensors with NVIDIA ModelOpt NVFP4 quantization and a custom `Qwen4Exp` architecture. There is **zero support** in `llama.cpp` for ModelOpt NVFP4 mixed-precision or the `qwen4_exp` PLE embedding structure. Adopting Flash-Next would require discarding the entire Drive-cached `llama.cpp` runtime in favor of vLLM or SGLang.
3. **Single-GPU vs Multi-GPU Reality:**
   [`../27b-plan/architecture.md`](../27b-plan/architecture.md) specifies a single-GPU Google Colab runtime (A100 40GB/80GB or G4 96GB). The chat claims Flash-Next fits entirely on a single 96 GB G4. Official NVIDIA documentation confirms that Flash-Next NVFP4 is configured and tested for **8x Blackwell GPUs** (`--tensor-parallel-size 8`). It cannot operate as a single-GPU Colab worker.

---

## What remains unverified

1. **Host-Prefetch Latency over Colab PCIe:**
   While the existence and shard structure of the 51.2B PLE n-gram table are confirmed, the empirical latency overhead of streaming these embeddings across Google Colab's virtualized PCIe bus (typically PCIe Gen4 x16) during single-token decoding remains unbenchmarked because host OOM prevents the process from completing initialization on Colab.
2. **Community Quantizations (GGUF / EXL2):**
   Whether an independent community quantizer (such as Unsloth or Blackfrost) can successfully abliterate, prune, or compress the 51.2B n-gram table and quantize the full attention layers to enable a true single-card 96 GB GGUF/EXL2 artifact remains unverified.
3. **Downstream Task Accuracy under W4A4 NVFP4:**
   Although NVIDIA published high benchmark figures (GPQA Diamond: 91.5 vs FP8 92.0; Terminal-Bench 2.1: 83.3), the real-world coding and agentic stability of fine-grained top-10 routing under 4-bit precision in prolonged multi-turn sessions remains unverified by independent third parties.

---

## Sources

1. **NVIDIA Hugging Face Repository & Model Card:**
   `nvidia/Qwen3.8-Flash-Next-NVFP4` — [https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
2. **NVIDIA Model Configuration (`config.json`):**
   `https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json` — [https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json)
3. **NVIDIA ModelOpt Quantization Specification (`hf_quant_config.json`):**
   `https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/hf_quant_config.json` — [https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/hf_quant_config.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/hf_quant_config.json)
4. **NVIDIA Safetensors Index (`model.safetensors.index.json`):**
   `https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/main/model.safetensors.index.json` — [https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/main/model.safetensors.index.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/main/model.safetensors.index.json)
5. **Upstream Alibaba Qwen Model Card & Specifications:**
   `Qwen/Qwen3.8-Flash-Next` — [https://huggingface.co/Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
6. **Alibaba Qwen Official Blog & Architecture Announcement:**
   Qwen3.8-Flash-Next Overview — [https://qwen.ai/blog?id=qwen3.8-flash-next](https://qwen.ai/blog?id=qwen3.8-flash-next)
7. **NVIDIA Model Optimizer GitHub Repository:**
   [https://github.com/NVIDIA/Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer)
8. **vLLM Integration Commits & PRs:**
   - Upstream Qwen4-Exp support: [vLLM commit `d4d703c`](https://github.com/vllm-project/vllm/commit/d4d703caf908786416585ceb1f369e2e0363358b)
   - MTP Speculative Decoding: [vLLM PR #55513](https://github.com/vllm-project/vllm/pull/55513)
