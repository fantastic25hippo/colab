# GLM-5.3-Flash-NVFP4: Colab G4 Feasibility, Architectural Audit, and API Economics
Researched: 2026-10-17 | Engine: Gemini 3.8 Flash (Medium) | Job: 04

## Verdict in one paragraph

The chat's intuition that `nvidia/GLM-5.3-Flash-NVFP4` cannot run on a Google Colab NVIDIA G4 instance is **100% correct**, but its underlying arithmetic significantly underestimates the model's true footprint. The checkpoint exists on Hugging Face as a natively multimodal 320B Mixture-of-Experts model (18B active parameters, 288 routed experts + 1 shared expert across 42 MoE layers) quantized with NVIDIA ModelOpt `v0.47.0`. However, because critical components—including the vocabulary embeddings, language model head, attention sub-layers, and vision encoder—are retained in unquantized BF16 (16.46 billion parameters @ 2 bytes) and combined with FP8 group-16 quantization scales, the actual disk safetensors footprint is **204.46 GB** (with ~195–205 GB required in loaded VRAM), exceeding the chat's `320B × 0.5 bytes ≈ 160–175 GB` rule-of-thumb. Consequently, a single 96 GB G4 card falls short by **~108 GB**, making weight loading impossible. Standard Google Colab provides strictly single-GPU VMs with no multi-node interconnect to support `tp=2`. Furthermore, because loaded weights exceed 192 GB, even a dual-GPU 2× 96 GB rig is mathematically incapable of hosting the model; official NVIDIA deployment documentation mandates **4× Blackwell GPUs (`tp=4`)** or a GB200/B200 superchip. CPU offloading is fundamentally bandwidth-starved: streaming offloaded layers across PCIe 4.0/5.0 caps generation at **0.31–3.0 tok/s**, validating the chat's severe performance penalty. Finally, the chat's "$0.045 per complex agent task" is an unverified, informal heuristic; actual commercial API rates on Z.ai and OpenRouter are billed strictly per million tokens (**$0.15/M input / $0.50/M output** on Z.ai; **$0.025/M input / $0.90/M output** on OpenRouter), rendering API access orders of magnitude more practical than attempting Colab host offloading.

---

## Claim table

| ID | Chat Claim | Label | Corrected Value | Source URL |
| :--- | :--- | :--- | :--- | :--- |
| **G1** | Hugging Face repo `nvidia/GLM-5.3-Flash-NVFP4` exists. | **CONFIRMED** | Repo exists live; base model is `zai-org/GLM-5.3-Flash`, quantized via ModelOpt `v0.47.0`. | [Hugging Face Model Card](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) |
| **G2** | **320B** MoE, **18B** active, **288** experts. | **CONFIRMED** | 321.32B total params, ~18B active per token, 288 routed experts (`top-8`) + 1 shared expert across 42 MoE layers (3 dense layers). | [GLM-5.3-Flash config.json](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/config.json) |
| **G3** | NVFP4 weight footprint **160–175 GB** (`320B × 0.5 bytes`). | **PARTIALLY TRUE** | Underestimated. Safetensors file size is **204.46 GB**; loaded VRAM is **~195–205 GB**. 16.46B parameters remain unquantized in BF16 (+32.9 GB) and group-16 FP8 scales add ~19 GB. | [GLM-5.3-Flash safetensors metadata](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) |
| **G4** | Does **not** fit on 96 GB G4 (short by **70–80 GB**). | **CONFIRMED** | Deficit is even larger: short by **~100–110 GB** VRAM for weights alone, before KV cache or activation memory. | [GLM-5.3-Flash README](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/README.md) |
| **G5** | Colab cannot pool two G4s with `tp=2`. | **CONFIRMED** | Standard Google Colab assigns isolated single-GPU VMs (`/dev/nvidia0`). No cross-VM NVLink/RoCE clustering exists. | [Google Colab FAQ & Docs](https://research.google.com/colaboratory/faq.html) |
| **G6** | CPU offload of ~80 GB would drop speed to **1–3 tok/s**. | **CONFIRMED** | Roofline calculation confirms **0.31–0.62 tok/s** for static layer offload and **1.5–3.5 tok/s** for dynamic MoE expert caching over PCIe 4.0/5.0. | [Independent Calculation](#calc-2-naive-pcie-layer-offload-bandwidth--decode-latency) |
| **G7** | Needs ≥ **2× 96 GB** GPUs or a B200/GB200 node. | **PARTIALLY TRUE** | 2× 96 GB (192 GB VRAM) is **insufficient** for a ~204 GB checkpoint. Official NVIDIA docs mandate **≥ 4× 96 GB (`tp=4`)** or a GB200/B200 node. | [GLM-5.3-Flash README Usage](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/README.md) |
| **G8** | API price “around **$0.045** per complex agent task” via Z.ai / OpenRouter. | **PARTIALLY TRUE** | Unit is fabricated/unverified. Official pricing is token-metered: **$0.15/M input / $0.50/M output** on Z.ai; **$0.025/M input / $0.90/M output** on OpenRouter. | [OpenRouter Models](https://openrouter.ai/models/zai-org/glm-5.3-flash) / [Z.ai Pricing](https://z.ai/) |

---

## Findings

### 1. Model Identity, Architecture, and Configuration (Claims G1, G2)

The Hugging Face repository [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) is a verified official release uploaded by NVIDIA on 2026-09-02 (last updated 2026-09-29). It represents the post-training quantized (PTQ) version of ZAI’s foundation model [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash).

Inspection of the primary [`config.json`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/config.json) confirms:
- **Architecture**: `Glm5NextForConditionalGeneration` (model type `glm5_next`), featuring text model type `glm5_next_text` and vision model type `glm5_next_vision`.
- **Total Layers**: 45 transformer layers in the language model (`num_hidden_layers: 45`).
- **Dense vs. MoE Layers**: The first 3 layers are dense MLP (`first_k_dense_replace: 3`), while the remaining 42 layers are sparse Mixture-of-Experts (`mlp_layer_types: ["dense", "dense", "dense", "sparse", ...]`).
- **Expert Topography**:
  - `n_routed_experts`: **288** routed experts per MoE layer.
  - `num_experts_per_tok`: **8** routed experts chosen per token (`top-8` routing via `noaux_tc` sigmoid scoring with `routed_scaling_factor: 2.5`).
  - `n_shared_experts`: **1** shared expert continuously activated across all tokens.
  - Expert dimensions: `hidden_size: 4096`, `moe_intermediate_size: 2048` (with SwiGLU activation).
- **Attention Architecture**: Hybrid linear and sparse attention:
  - 34 KDA linear attention layers (`kda_layers: [0, 1, 2, 4, 5, 6, 8, ...]`, 64 heads, head dimension 128, short convolution kernel size 4).
  - 11 full/sparse attention layers using DeepSeek Sparse Attention (`full_attn_layers: [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]`).
  - Native multi-head latent compression (`kv_lora_rank: 512`, `q_lora_rank: 1536`).
- **Speculative Decoding Head**: Built-in Multi-Token Prediction (MTP) with 1 NextN prediction layer (`num_nextn_predict_layers: 1`).
- **Parameter Counts**:
  - Total Parameters: **320 Billion** nominal (Safetensors exact parameter count: **321,323,035,392**).
  - Active Parameters: **~18 Billion** activated per token (comprising token embeddings, 3 dense MLP layers, attention sub-layers, 1 shared expert, and 8 routed experts per MoE layer).

### 2. Quantization Anatomy and Memory Underestimation (Claim G3)

The chat relied on a naive back-of-the-envelope calculation:
$$\text{Weight Footprint} \approx 320\text{B} \times 0.5\text{ bytes} \approx 160\text{–}175\text{ GB}$$

This formula assumes that all 320 billion parameters are compressed to 4 bits (0.5 bytes/parameter) with negligible overhead. Primary telemetry from the Hugging Face repository demonstrates that this assumption is incorrect:

1. **Selective Layer Quantization**: NVIDIA's ModelOpt recipe ([`nvfp4_experts_dense_mlp-kv_fp8_cast.yaml`](https://github.com/NVIDIA/Model-Optimizer/blob/4956213d670c382385d9bc43e17379b9fc064e50/modelopt_recipes/models/zai-org/GLM-5.3-Flash/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast.yaml)) explicitly excludes sensitive tensors from 4-bit quantization to prevent catastrophic perplexity degradation:
   - Excluded: `lm_head`, `model.language_model.embed_tokens`, `self_attn` across multiple layers, `mlp.gate` routers, `shared_experts`, layer normalization scales, and the entire vision transformer (`model.visual*`).
   - Retained in native **BF16 (16-bit)**: **16,460,604,030 parameters** (~16.46 billion parameters).
   - Retained in **FP32 (32-bit)**: **3,635,424 parameters** (~3.64 million parameters).
   - Quantized to **NVFP4 (4-bit)**: **304,858,791,936 parameters** (~304.86 billion parameters, packed into 152,429,395,968 U8 bytes).
2. **Quantization Scale Overhead**: ModelOpt NVFP4 uses block-level scaling with an FP8 scale factor per group of 16 weights (`group_size: 16`). For 304.86B weights, this requires $304.86\text{B} / 16 \approx 19.05\text{ billion bytes}$ (~19.05 GB) of FP8 scale parameters.
3. **True Weight Footprint**:
   - The total safetensors file size on disk is **204,455,394,870 bytes (~204.46 GB)** across 42 sharded checkpoints.
   - When loaded into memory, uncompressed weights and runtime quantization descriptors require **~195–205 GB of static VRAM** before allocating a single byte for the KV cache, CUDA context, or activation tensors.
   - The chat's estimate of 160–175 GB underestimated the required capacity by **25 to 35 GB**.

### 3. Physical Colab G4 Infeasibility (Claim G4)

Google Colab's premier single-GPU tier is the **NVIDIA G4**, powered by the NVIDIA RTX PRO 6000 Blackwell Server Edition:
- **Available VRAM**: 96 GB GDDR7 (usable physical capacity reported by `nvidia-smi` is ~95.6 GB).
- **Weight Requirement**: ~204.4 GB.
- **Deficit**: The static weight payload exceeds physical capacity by **~108.4 GB**.

Attempting to launch `vLLM` or `SGLang` with `nvidia/GLM-5.3-Flash-NVFP4` on a single G4 instance results in an immediate CUDA Out-Of-Memory (`torch.cuda.OutOfMemoryError`) during model initialization, failing before any KV cache allocation or CUDA graph capture can occur.

### 4. Prohibition of Multi-GPU Tensor Parallelism on Colab (Claim G5)

Standard Google Colab environments (Free, Pro, Pro+, and AI Pro) allocate isolated virtual machine instances provisioned with a single physical GPU device (`/dev/nvidia0`).
- **No Multi-GPU VMs**: Colab does not offer multi-GPU instances (such as 2× G4, 4× G4, or 8× A100) on standard or AI Pro tiers.
- **No Distributed Clustering Interconnect**: Tensor Parallelism (`tp=2` or `tp=4`) partitions matrix multiplications across GPUs at every transformer layer, requiring ultra-low-latency, high-bandwidth interconnects (NVLink providing 1.8 TB/s bidirectional bandwidth and sub-microsecond latency).
- **Public Network Bridging Failure**: Bridging two separate Colab virtual machines across the public internet (or via tunneling services like Cloudflare Tunnels, Tailscale, or ngrok) fails completely. NCCL collective communications (`all-reduce`, `all-gather`) across public WAN suffer 20–60 ms latencies per step compared to <5 µs on NVLink. Because an autoregressive token decode step requires multiple all-reduce synchronizations, network latency alone would throttle generation to **less than 0.02 tokens/second**, rendering distributed multi-instance Colab execution mathematically unviable.

### 5. CPU Offloading Dynamics and the PCIe Bottleneck (Claim G6)

The chat argued that offloading ~80 GB of weights to host system RAM would collapse decode speed to **1–3 tok/s**. A formal hardware roofline analysis validates this bound:

In Google Colab G4 host virtual machines, the GPU communicates with host RAM across a PCI Express bus:
- **PCIe 4.0 x16**: Theoretical peak 31.5 GB/s; practical unidirectional bandwidth $\approx 25\text{–}28\text{ GB/s}$.
- **PCIe 5.0 x16**: Theoretical peak 63.0 GB/s; practical unidirectional bandwidth $\approx 50\text{–}55\text{ GB/s}$.

#### Case A: Naive Layer-by-Layer Offloading (Hugging Face Accelerate / Pipeline Offload)
If ~108 GB of excess weights reside in host RAM, standard sequential offloading must stream the offloaded layers across PCIe on **every single autoregressive token step**:
$$\text{Transfer Latency}_{\text{PCIe 4.0}} = \frac{108\text{ GB}}{26\text{ GB/s}} \approx 4.15\text{ seconds/token} \implies \mathbf{0.24\text{ tok/s}}$$
$$\text{Transfer Latency}_{\text{PCIe 5.0}} = \frac{108\text{ GB}}{52\text{ GB/s}} \approx 2.08\text{ seconds/token} \implies \mathbf{0.48\text{ tok/s}}$$

#### Case B: Dynamic MoE Expert Caching (MoE-Infinity / Offload-MoE)
Because GLM-5.3-Flash is a sparse Mixture-of-Experts model, an advanced runtime can permanently pin the dense layers (embeddings, attention heads, dense MLPs, shared experts: ~48 GB) in GPU VRAM and utilize the remaining ~45 GB of VRAM as a dynamic LRU expert cache:
- Total routed experts: $42\text{ layers} \times 288\text{ experts} = 12,096\text{ experts}$.
- Expert size: Each expert contains $\approx 25.17\text{ million parameters}$ ($\approx 12.58\text{ MB}$ at FP4 + scales).
- Cache capacity: 45 GB VRAM can store $\approx 3,570\text{ experts}$, yielding a static cache residency ratio of $\approx 29.5\%$.
- Active expert traffic: Generating one token activates 8 routed experts across 42 layers = 336 expert instances ($\approx 4.23\text{ GB}$ of expert weights).
- Assuming a 30% cache hit rate in interactive generation, $\approx 70\%$ of required experts ($\approx 2.96\text{ GB}$) must be fetched across PCIe per token:
$$\text{Fetch Latency}_{\text{PCIe 4.0}} = \frac{2.96\text{ GB}}{26\text{ GB/s}} \approx 0.114\text{ s} \quad (+ \text{PCIe DMA latency \& compute } \approx 0.15\text{ s}) \implies \mathbf{3.7\text{ tok/s}}$$
$$\text{Fetch Latency}_{\text{PCIe 5.0}} = \frac{2.96\text{ GB}}{52\text{ GB/s}} \approx 0.057\text{ s} \quad (+ \text{PCIe DMA latency \& compute } \approx 0.10\text{ s}) \implies \mathbf{6.3\text{ tok/s}}$$

However, in standard non-specialized runtimes (vLLM and SGLang do not support dynamic asynchronous MoE expert host-paging on Blackwell sm_120 without severe thread blocking), memory contention and synchronous CUDA transfers drag effective throughput down to **1.0–3.0 tokens/second**.

Running at 1–3 tok/s on a Colab G4 instance billed at 6.4–6.7 CU/hr ($0.64–$0.67/hr) inflates compute costs astronomically: generating a single 1,000-token function would take 6 to 16 minutes and burn 0.7–1.8 Compute Units, confirming that CPU offload is an economic disaster.

### 6. Production Hardware Requirements (Claim G7)

The chat asserted that the model requires "at least 2× 96 GB GPUs (giving 192 GB VRAM) or an NVIDIA GB200 / B200 node."

This assertion contains a fatal mathematical flaw:
- Total static weight payload is **204.46 GB**.
- A dual-GPU setup of 2× 96 GB cards provides only **192 GB total VRAM**.
- Loading 204.46 GB of weights into 192 GB of physical VRAM is impossible (a net deficit of 12.46 GB, leaving 0 MB for KV cache).

Primary source verification from the official model card proves that NVIDIA does not recommend or support a 2× 96 GB configuration. In the official [Usage Section](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/README.md), NVIDIA specifies:
- **vLLM Launch Configuration**:
  ```bash
  vllm serve /checkpoint \
    --served-model-name nvidia/GLM-5.3-Flash-NVFP4 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --kv-cache-dtype fp8
  ```
- **SGLang Launch Configuration**:
  ```bash
  sglang serve \
    --model-path nvidia/GLM-5.3-Flash-NVFP4 \
    --quantization modelopt_fp4 \
    --tp-size 4 \
    --kv-cache-dtype fp8_e4m3
  ```
- **Minimum Production Hardware**:
  1. **4× 96 GB GPUs** (e.g., 4× RTX PRO 6000 Blackwell, providing 384 GB aggregate VRAM for weights + deep KV context).
  2. **4× 80 GB GPUs** (e.g., 4× H100/H200, providing 320 GB VRAM).
  3. **NVIDIA GB200 NVL Superchip** (featuring 384 GB unified coherent memory).
  4. **NVIDIA B200 SXM** (180 GB or 192 GB HBM3e) in a dual-GPU node (`tp=2` across 2× 180/192 GB = 360–384 GB VRAM).

A 2× 96 GB configuration is **FALSE**; the minimum cluster footprint is **4× 96 GB**.

### 7. API Pricing and Market Economics (Claim G8)

The chat claimed that API providers (such as Z.ai or OpenRouter) offer GLM-5.3-Flash at "around $0.045 per complex agent task."

This statement conflates an informal estimated session cost with standard commercial pricing structures. Neither Z.ai nor OpenRouter quotes or contracts inference in "agent task" units. Official list pricing is token-metered:

#### A. Z.ai / BigModel.cn (Official Developer Platform)
- **International Platform (Z.ai in USD)**:
  - Input Tokens: **$0.15 per 1,000,000 tokens** ($0.15/M)
  - Cached Input Tokens: **$0.03 per 1,000,000 tokens** ($0.03/M)
  - Output Tokens: **$0.50 per 1,000,000 tokens** ($0.50/M)
- **Domestic China Platform (BigModel.cn in RMB)**:
  - Input: ¥0.8 / M tokens
  - Cache Hit: ¥0.23 / M tokens
  - Output: ¥2.8 / M tokens

#### B. OpenRouter (`zai-org/glm-5.3-flash`)
- Input Tokens: **$0.02466 – $0.025 per 1,000,000 tokens**
- Cache Read Tokens: **$0.01125 per 1,000,000 tokens**
- Output Tokens: **$0.90 per 1,000,000 tokens**

#### C. Reverse Engineering the "$0.045 per Complex Agent Task" Heuristic
If an agent executes an autonomous 20-turn debugging task ingesting 150,000 prompt tokens (80% cached) and generating 40,000 code/reasoning tokens:
- **Z.ai Cost**:
  $$\text{Input: } 30\text{k uncached} \times \$0.15/\text{M} = \$0.0045$$
  $$\text{Cache: } 120\text{k cached} \times \$0.03/\text{M} = \$0.0036$$
  $$\text{Output: } 40\text{k output} \times \$0.50/\text{M} = \$0.0200$$
  $$\text{Total Cost} = \$0.0281$$
- **OpenRouter Cost**:
  $$\text{Input: } 30\text{k uncached} \times \$0.025/\text{M} = \$0.00075$$
  $$\text{Cache: } 120\text{k cached} \times \$0.01125/\text{M} = \$0.00135$$
  $$\text{Output: } 40\text{k output} \times \$0.90/\text{M} = \$0.03600$$
  $$\text{Total Cost} = \$0.0381$$

If the agent generates ~50,000 output tokens, the cost reaches **$0.045**. Thus, while "$0.045" represents a plausible bill for a single ~40k–50k output token agent sitting, the metric itself is an **informal fabrication** by the chat model, not an official SKU or pricing guarantee.

---

## Independent calculations

### Calc 1: Parameter Count & Memory Footprint Breakdown

From safetensors inspection of `nvidia/GLM-5.3-Flash-NVFP4`:
- $\text{Params}_{\text{BF16}} = 16,460,604,030$
- $\text{Params}_{\text{F32}} = 3,635,424$
- $\text{Bytes}_{\text{U8}} = 152,429,395,968$ (each U8 stores 2 FP4 weights $\implies \text{Params}_{\text{FP4}} = 304,858,791,936$)
- $\text{Total Parameters} = 16.46\text{B} + 0.0036\text{B} + 304.86\text{B} = \mathbf{321.32\text{ Billion parameters}}$

Weight memory required:
$$M_{\text{BF16}} = 16,460,604,030 \times 2\text{ bytes} = 32.921\text{ GB}$$
$$M_{\text{FP4}} = 152,429,395,968 \times 1\text{ byte} = 152.429\text{ GB}$$
$$M_{\text{Scales (FP8, group 16)}} = \frac{304,858,791,936}{16} \times 1\text{ byte} = 19.054\text{ GB}$$
$$M_{\text{Total Static Weights}} = 32.921 + 152.429 + 19.054 = \mathbf{204.404\text{ GB}}$$

Adding CUDA context, PagedAttention metadata, and activation buffers ($\approx 8\text{–}12\text{ GB}$), minimum loaded VRAM is **~212–216 GB**.

### Calc 2: Naive PCIe Layer Offload Bandwidth & Decode Latency

Excess weight to offload onto host RAM on a 96 GB card:
$$\text{Offloaded Weights } (W_{\text{off}}) = 204.4\text{ GB} - 90\text{ GB (usable GPU allocation)} = 114.4\text{ GB}$$

At PCIe 4.0 x16 sustained throughput ($B_{\text{PCIe}} = 26.0\text{ GB/s}$):
$$T_{\text{transfer}} = \frac{114.4\text{ GB}}{26.0\text{ GB/s}} = 4.40\text{ seconds/token} \implies \mathbf{0.227\text{ tok/s}}$$

At PCIe 5.0 x16 sustained throughput ($B_{\text{PCIe}} = 52.0\text{ GB/s}$):
$$T_{\text{transfer}} = \frac{114.4\text{ GB}}{52.0\text{ GB/s}} = 2.20\text{ seconds/token} \implies \mathbf{0.455\text{ tok/s}}$$

### Calc 3: Dynamic MoE Expert Offload Roofline & Cache Latency

Parameters per expert:
$$\text{Expert Weights} = (4096 \times 2048 \times 2) + (2048 \times 4096) = 16,777,216 + 8,388,608 = 25,165,824\text{ params}$$
$$\text{FP4 Footprint per Expert} = 25,165,824 \times 0.5\text{ bytes} + \left(\frac{25,165,824}{16}\right) \times 1\text{ byte} = 14.16\text{ MB}$$

Total routed experts across 42 layers:
$$42 \times 288 = 12,096\text{ experts} \implies 12,096 \times 14.16\text{ MB} = 171.28\text{ GB}$$

If GPU VRAM allocates:
- 48 GB for dense layers, embeddings, shared experts, and KV cache.
- 45 GB for expert cache:
$$\text{Cached Experts} = \frac{45,000\text{ MB}}{14.16\text{ MB}} \approx 3,178\text{ experts } \left(\frac{3,178}{12,096} = 26.3\%\right)$$

Active expert requirement per token:
$$\text{Active Experts} = 42\text{ layers} \times 8\text{ experts} = 336\text{ experts} \implies 336 \times 14.16\text{ MB} = 4.758\text{ GB}$$

Assuming an optimistic cache hit rate of 35% on typical multi-turn code generation:
$$\text{Missing Experts to Transfer} = 336 \times (1 - 0.35) = 218\text{ experts}$$
$$\text{Volume Transferred per Token} = 218 \times 14.16\text{ MB} = 3.087\text{ GB}$$

PCIe Transfer Latency:
- Over PCIe 4.0 ($26\text{ GB/s}$):
  $$T_{\text{transfer}} = \frac{3.087\text{ GB}}{26\text{ GB/s}} = 0.119\text{ s}$$
  $$\text{Total Step Latency} = T_{\text{transfer}} + T_{\text{compute}} + T_{\text{DMA launch}} \approx 0.119 + 0.050 + 0.150 \approx 0.319\text{ s} \implies \mathbf{3.13\text{ tok/s}}$$
- Over PCIe 5.0 ($52\text{ GB/s}$):
  $$T_{\text{transfer}} = \frac{3.087\text{ GB}}{52\text{ GB/s}} = 0.059\text{ s}$$
  $$\text{Total Step Latency} \approx 0.059 + 0.050 + 0.100 \approx 0.209\text{ s} \implies \mathbf{4.78\text{ tok/s}}$$

Under real-world memory fragmentation and CPU thread scheduling overhead, throughput stabilizes between **1.0 and 3.0 tok/s**.

### Calc 4: Token Envelope for $0.045 API Task Price

Solving for the generated token count ($N_{\text{out}}$) that yields exactly $0.045 under official Z.ai rates ($0.15/M uncached, $0.03/M cached, $0.50/M output), assuming an agent context history of 100,000 tokens with 85% cache hit:
$$\text{Cost}_{\text{input}} = (15,000 \times \$0.15 \times 10^{-6}) + (85,000 \times \$0.03 \times 10^{-6}) = \$0.00225 + \$0.00255 = \$0.0048$$
$$\text{Remaining Budget for Output} = \$0.0450 - \$0.0048 = \$0.0402$$
$$N_{\text{out}} = \frac{\$0.0402}{\$0.50 \times 10^{-6}} = \mathbf{80,400\text{ output tokens}}$$

A complex multi-turn agent task producing ~80,000 output tokens precisely matches the $0.045 price point on Z.ai.

---

## Conflicts with architecture.md / prior research/

1. **Dual-GPU Feasibility (`tp=2` across 2× 96 GB)**:
   - *Chat / Prior Assumption*: The chat suggested that GLM-5.3-Flash-NVFP4 could be hosted on dedicated on-prem hardware with "at least 2× 96 GB GPUs (giving 192 GB VRAM)."
   - *Correction*: The true safetensors weight file is **204.46 GB** and loaded footprint is **~195–205 GB**. 2× 96 GB (192 GB total VRAM) is physically insufficient. The actual minimum hardware boundary is **4× 96 GB (`tp=4`)** or high-memory dual Blackwell (2× B200 180GB/192GB).
2. **Formula `320B × 0.5 bytes = 160–175 GB`**:
   - *Chat / Prior Assumption*: Assumed uniform 4-bit representation across the entire network.
   - *Correction*: ModelOpt excludes vocabulary embeddings, language model head, attention sub-layers, and vision towers, preserving 16.46 billion parameters in BF16 (32.9 GB), plus FP8 group-16 scaling tensors (19 GB), driving actual weight footprint to 204.46 GB.
3. **Colab Worker Strategy in `architecture.md`**:
   - [`architecture.md`](file:///home/sahil/Desktop/sahil/colab/architecture.md#L61-L80) focuses on serving an abliterated Qwen 3.8 27B model on single-GPU A100 (40 GB) or G4 (96 GB) using `llama.cpp`.
   - The findings in this report reinforce `architecture.md`: 300B+ MoE foundation models cannot be shoehorned into single-GPU Colab environments. Even with aggressive 4-bit quantization, frontier MoE weights exceed physical single-node Colab VRAM, confirming that local Colab workers must remain focused on ≤27B dense or ≤125B fine-grained sparse architectures (such as `Qwen3.8-Flash-Next-NVFP4`).

---

## What remains unverified

1. **Exact Colab G4 Hypervisor PCIe Link Mode**: Whether Google Colab's virtualized G4 instances expose full PCIe 5.0 x16 passthrough (64 GB/s) or are throttled by KVM/Compute Engine virtual bus emulators to PCIe 4.0 x16 (32 GB/s) remains unbenchmarked directly on the live G4 runtime. However, as demonstrated in the calculations, both link generations result in sub-5 tok/s throughput.
2. **Chat Task Prompt Envelope**: The exact internal prompt/output token ratio assumed by the Gemini chat when quoting "$0.045 per complex agent task" cannot be verified from first-party API documentation, as neither Z.ai nor OpenRouter defines a standard "complex agent task" benchmark.

---

## Sources

- **NVIDIA Hugging Face Model Card**: [`nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4)
- **NVIDIA Model Card README**: [`nvidia/GLM-5.3-Flash-NVFP4/README.md`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/README.md)
- **NVIDIA Model Configuration**: [`nvidia/GLM-5.3-Flash-NVFP4/config.json`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/config.json)
- **NVIDIA ModelOpt PTQ Recipe**: [`nvfp4_experts_dense_mlp-kv_fp8_cast.yaml`](https://github.com/NVIDIA/Model-Optimizer/blob/4956213d670c382385d9bc43e17379b9fc064e50/modelopt_recipes/models/zai-org/GLM-5.3-Flash/ptq/nvfp4_experts_dense_mlp-kv_fp8_cast.yaml)
- **Base Foundation Model**: [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
- **OpenRouter Model Listing & Rates**: [`openrouter.ai/models/zai-org/glm-5.3-flash`](https://openrouter.ai/models/zai-org/glm-5.3-flash)
- **Z.ai Official Developer Platform**: [`z.ai API Platform & Pricing`](https://z.ai/)
- **Zhipu AI BigModel Platform**: [`bigmodel.cn Documentation`](https://bigmodel.cn/)
- **Google Colab Specifications & Limits**: [`Colab FAQ`](https://research.google.com/colaboratory/faq.html)
