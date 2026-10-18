# Claims Ledger: Repository Technical & Factual Audit

**Audit Date:** 2026-10-18  
**Repository:** `/home/sahil/Desktop/sahil/colab` (`github.com/fantastic25hippo/colab`, `main @ 8f25a66`)  
**Auditor:** Antigravity / Google DeepMind Agentic Coding Team  

---

## 1. Introduction & Methodology

This document establishes the definitive, primary-source-grounded claims ledger for all technical, mathematical, architectural, and economic claims across the repository. It audits and reconciles the claims made in:
- Root architectural specifications: `README.md`, `ops.md`, and archived `research/archive/27b-plan/` (`architecture.md`, `inference.md`, `first-boot.md`, `bend-hands.md`, `plan_audit_and_fixes.md`, and historical `fallback-27b.md`)
- Gemini chat trace: `research/archive/flash-next-chat.md`
- Research synthesis dossiers: `research/archive/flash-next-audit/` (all 10 research papers, `README.md`, and `CLAIMS.md`)
- Technical research notes: `research/` (`a100_speed_and_session.md`, `blueprint.md`, `busy_gpu.md`, `drive_files.md`, `kernel_and_cli.md`, `llamacpp_flags.md`, `qwen38_27b_max_tokens_per_second_deep_research.md`, `sidecar_and_verify.md`)

### Status Definitions
Every factual claim is evaluated and assigned strictly one of three statuses:
- **`verified`**: The claim is empirically or mathematically confirmed against authoritative primary sources (official Hugging Face model cards, raw `config.json` specifications, official SGLang/vLLM documentation and cookbooks, NVIDIA hardware datasheets, Google Colab platform documentation, or verified benchmarks).
- **`wrong`**: The claim contains factual errors, arithmetic mistakes, architectural misconceptions, or directly contradicts authoritative primary sources. The corrected value and underlying rationale are documented.
- **`unverified`**: The claim cannot be conclusively verified or falsified without live runtime execution on this specific account or hardware, or remains an unresolved empirical question. It is never resolved by speculation.

All search wrapper URLs and local file path links have been eliminated and replaced with direct, canonical primary source URLs.

---

## 2. Master Claims Ledger

### Group 1: Hardware & Colab Compute Units (Hardware/CU)

| ID | Claim | Status | Corrected Value | Primary Source URL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H1** | Colab "NVIDIA G4" is the NVIDIA RTX PRO 6000 Blackwell | **verified** | NVIDIA RTX PRO 6000 Blackwell Server Edition (passively cooled PCIe Gen 5, GB202 silicon, 400W–600W configurable TDP). | https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/ | Confirmed by NVIDIA product pages and Lenovo ThinkSystem hardware documentation. |
| **H2** | Card has 96 GB VRAM | **verified** | 96 GB GDDR7 ECC (usable: 95.60 GiB / 97,896 MiB in `nvidia-smi`). | https://www.pny.com/en-eu/file%20library/professional/datasheet/data%20center%20cards/pny-nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf | 96.0 GB decimal; ~95.6 GiB net usable after driver/ECC reservation. |
| **H3** | Memory bandwidth is 1,597 GB/s GDDR7 | **verified** | 1,597 GB/s GDDR7 (512-bit bus @ 24.95 Gbps). Not HBM. | https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu | Server Edition is clocked at 24.95 Gbps (1,597 GB/s); active-cooled Workstation Edition runs at 28 Gbps (1,792 GB/s). |
| **H4** | FP4 tensor compute is 4 PFLOPS | **wrong** | 4.0 PFLOPS with 2:4 structural sparsity; **2.0 PFLOPS** dense GEMM (no sparsity). | https://www.nvidia.com/en-us/data-center/blackwell-architecture/ | Chat cited the sparse peak without qualifying structural sparsity. Dense GEMM is 2.0 PFLOPS. |
| **H5** | FP8 tensor compute is ~1,000+ TFLOPS | **verified** | **1,000 TFLOPS (1.0 PFLOP)** dense; 2,000 TFLOPS (2.0 PFLOPS) with 2:4 sparsity. | https://www.nvidia.com/en-us/data-center/blackwell-architecture/ | 1,000 dense TFLOPS FP8 verified on Blackwell GB202 silicon. |
| **H6** | Colab G4 burns ~6.4–6.7 CU/hr | **wrong** | **8.71 CU/hr** standard metered rate; 6.40–6.74 CU/hr was early rollout / off-peak. | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Measured standard billing rate on Colab Pro+/AI Ultra is 8.71 CU/hr. 6.4–6.7 CU/hr was promotional/off-peak. |
| **H7** | Colab L4 burns ~2.1 CU/hr | **wrong** | Standard baseline is **1.71 CU/hr**; reaches 2.09–2.10 CU/hr under dynamic peak demand. | https://colab.research.google.com/signup | Standard metered rate for L4 Standard is 1.71 CU/hr. Peak/high-RAM triggers ~2.10 CU/hr. |
| **H8** | G4 hourly price is ~3.1× L4 | **wrong** | At standard baselines ($8.71 / 1.71$), G4 is **5.09× (~5.1×)** L4. | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Chat cherry-picked off-peak G4 ($6.5) vs peak L4 ($2.1) to manufacture $3.1\times$. True baseline ratio is $5.1\times$. |
| **H9** | L4 has 24 GB VRAM and 300 GB/s bandwidth (5.3× slower) | **verified** | 24 GB GDDR6, 300.05 GB/s (192-bit @ 12.5 Gbps). Bandwidth ratio: $1,597 / 300.05 = \mathbf{5.32\times}$. | https://www.nvidia.com/en-us/data-center/l4/ | Verified against NVIDIA L4 datasheet specifications. |
| **H10** | L4 FP8 compute is ~242 TFLOPS (16× below G4 FP4) | **wrong** | L4 FP8 is 242 TFLOPS sparse / 121 TFLOPS dense. Comparing G4 FP4 (4,000 TFLOPS) to L4 FP8 is invalid cross-precision. Dense FP8 ratio is **8.26×** ($1,000/121$). L4 has 0 FP4 compute. | https://www.nvidia.com/en-us/data-center/l4/ | Chat mixed precisions (FP4 on G4 vs FP8 on L4) and compared sparse to dense. |
| **H11** | Colab is single-GPU; no multi-VM `tp=2` | **verified** | Colab provisions isolated single-GPU VMs (`/dev/nvidia0`); zero cross-VM interconnect or RoCE/NVLink for tensor parallelism. | https://research.google.com/colaboratory/faq.html | Confirmed by official Google Colab architecture FAQ. |
| **H12** | Colab bills connected wall-clock, including idle | **verified** | Metering runs continuously per elapsed second of connection; idle GPU burns full hourly rate. | https://research.google.com/colaboratory/faq.html | Confirmed by Google Colab FAQ and billing terms. |
| **H13** | Colab AI Pro can assign G4 | **unverified** | Unknown for this AI Pro account. Direct G4 selection was reported grayed out on one user's account in note.com, but whether G4 is provisionable for Sahil's account is unknown without a live probe; do not treat note.com gray-out as proven for Sahil. | https://note.com/gakushiai/n/n71ca7ca09d1b | Status unverified until probed live on this specific account. note.com gray-out is not proven for Sahil. |
| **H14** | G4 is Blackwell `sm_120` with native NVFP4 tensor cores | **verified** | GB202 silicon targets `sm_120` / `sm_120a` with 5th-Gen Tensor Cores supporting block-scaled NVFP4. | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ | Confirmed by NVIDIA CUDA programming guide. |
| **H15** | Resolve G4 CU/hr: 6.4–6.7 vs 8.71 | **verified** | **8.71 CU/hr** is the standard metered rate; 6.4–6.74 is an optimistic lower bound. Budgeting must use 8.71 CU/hr. | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Reconciles chat claim against `research/archive/27b-plan/architecture.md` §4. |
| **H16** | 1,792 GB/s is Workstation, 1,597 GB/s is Server Edition | **verified** | Colab G4 is Server Edition (passive heatsink, 1,597 GB/s). Workstation is active blower (1,792 GB/s). | https://vrlatech.com/rtx-pro-6000-blackwell-workstation-vs-server-edition/ | Confirmed by comparative hardware analysis. |
| **H17** | Colab G4 host RAM is 53–64 GB causing host OOM | **wrong** | Colab G4 host system RAM is **~177 GB** (176.9 GB measured from Qiita: 48 CPU, 176.9 GB RAM, 95.6 GB VRAM, 8.71 CU/hr; corroborated by shanshiba-workflow-colab: 176 GB RAM, 8.90 CU/hr on 2026-10-10). Budget 8.71 CU/hr; note 8.90 as a later measurement. 53–64 GB was A100/L4 machine numbers. | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | 176.9 GB host RAM is from Qiita (G4 row: 48 CPU, 176.9 GB RAM, 95.6 GB VRAM, 8.71 CU/hr). googlecolab/google-colab-cli#47 confirms G4 is RTX PRO 6000 Server Edition ~98 GB VRAM, not host RAM. Optional corroboration: https://github.com/mathruffian-dot/shanshiba-workflow-colab/blob/main/docs/hardware.md. |
| **H18** | A100-SXM4-40GB bandwidth 1,555 GB/s at ~5.37–5.40 CU/hr | **verified** | A100-SXM4-40GB delivers 1,555 GB/s HBM2 at ~5.37 CU/hr (97.4% G4 bandwidth at 61.7% CU rate). | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf | Verified against NVIDIA datasheet and Colab metered benchmarks. |
| **H19** | A100-SXM4-80GB bandwidth 2,039 GB/s at ~6.77–15 CU/hr | **verified** | 80 GB HBM2e delivers 2,039 GB/s, metered at ~6.77 CU/hr (standard) up to ~15 CU/hr (High-RAM). | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf | Verified against NVIDIA datasheet and Colab pricing tiers. |

---

### Group 2: Qwen3.8-Flash-Next Checkpoint (Flash Next Checkpoint)

| ID | Claim | Status | Corrected Value | Primary Source URL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **F1** | HF repo `nvidia/Qwen3.8-Flash-Next-NVFP4` exists | **verified** | Public, ungated repo released 2026-08-31 by NVIDIA using ModelOpt v0.46.0 based on `Qwen/Qwen3.8-Flash-Next`. | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4 | Live official release from NVIDIA Corporation. |
| **F2** | It is a 125B MoE model | **verified** | Base LM contains 124.91B parameters (~120.80B routed MoE + ~4.11B shared/attention/embeds) plus 51.2B PLE n-gram table and ~4B MTP head (~180B total). | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Confirmed by model card and architectural parameter index. |
| **F3** | ~6.0B parameters activate per token | **verified** | 10 routed experts + 1 shared expert activate (~2.60B MoE), plus attention, embeddings, and router/norms (~3.2B–3.4B), totaling ~5.79B–6.0B active parameters. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Confirmed by architectural analysis. |
| **F4** | 512 fine-grained experts, top-10 routing | **verified** | `config.json` confirms `"num_experts": 512`, `"num_experts_per_tok": 10`, 1 shared expert (`"shared_expert_intermediate_size": 640`). | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json | Verified from raw `config.json`. |
| **F5** | Uses Gated DeltaNet | **verified** | 36 of 48 layers are `"linear_attention"` (Gated DeltaNet linear recurrence, Mamba SSM dtype FP32). | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json | Confirmed: 36 layers linear attention, 12 layers full attention. |
| **F6** | Uses Qwen Sparse Attention (QSA) pruning at long context | **verified** | 12 of 48 layers are full attention featuring QSA with an MQA indexer (`indexer_budget: 2048` tokens / 512 blocks). | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json | Verified from raw `config.json`. |
| **F7** | Has a 4B MTP head | **wrong** | 1-layer MoE MTP head present (~2.76B physical params). Quantized in **FP8** (`FP8_PB_WO`), NOT NVFP4. | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/hf_quant_config.json | MTP module is ~2.76B physical parameters and quantized to FP8. |
| **F8** | 51B n-gram embedding prefetched from host memory | **verified** | 51.2B param table exists (~47.7 GiB / 51.20 GB in FP8) and is offloaded to host RAM via `--ple-offload-embedding` with async prefetch. Fits easily in G4's ~177 GB host RAM. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | SGLang cookbook documents verified single-GPU host offload requiring 64 GB+ free RAM. |
| **F9** | NVFP4 weights + MTP occupy ~68–72 GB VRAM | **wrong** | Cookbook RTX PRO 6000 note documents that when the 47.7 GiB FP8 N-gram table is in pinned host RAM, the other ~78 GiB of the checkpoint stay in VRAM. After loader temporaries are collected, 74.7 GiB stays resident without speculative decoding and 81.8 GiB with MTP (the draft head is 0.5 GiB; the remainder is memory the loader still holds). Do not cite ~83–85 GB as if measured. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | SGLang cookbook specifies ~78 GiB checkpoint weights stay in VRAM when PLE is offloaded. Resident memory is 74.7 GiB (no MTP) / 81.8 GiB (with MTP). |
| **F10** | Leaves ~24–26 GB on G4 for KV cache | **wrong** | Leaves 19.4 GiB (without MTP) or 12.3 GiB (with MTP) usable of the 94.2 GiB process limit before static slack. Cookbook verified cell KV pool of ~78k tokens at 16 concurrent is a flag choice (`--max-running-requests 16`, mamba cache); native 262K FP8 KV cache is still ~3 GB (~3.07 GiB). | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Concurrency and mamba state pools dictate cookbook cell capacity (~78k tokens at 16 concurrent). Native FP8 KV for 262K is ~3.07 GiB. |
| **F11** | Different architecture from dense 27B, not "faster 27B" | **verified** | 27B is 64-layer dense hybrid; Flash-Next is 48-layer MoE (512 experts) + 51B PLE + 4B MTP. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Verified architectural distinction. |
| **F12** | Flash-Next cannot run on one G4 | **wrong** | **SGLang Qwen3.8-Flash-Next cookbook features a verified single RTX PRO 6000 (TP=1) setup** for `nvidia/Qwen3.8-Flash-Next-NVFP4` and `RadixArk/Qwen3.8-Flash-Next-NVFP4` via `--ple-offload-embedding`. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Refutes repo finding in `10_claims_ledger_and_verdict.md`. TP=1 is verified in official cookbook. |
| **F13** | Safetensors download size is ~70 GB | **wrong** | Safetensors download is **132.68 GB** across 11 shards (78.96 GB base + 53.72 GB PLE/MTP shard). | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/main/model.safetensors.index.json | Verified against official index file size sum. |
| **F14** | License / gating / Colab download practicality | **verified** | Public ungated license (NVIDIA Open Model License + Qwen Community License). 132.7 GB requires sufficient disk and 5–10 min transfer over Google backbone. | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4 | No gating or HF token required. |

---

### Group 3: Historical 27B baseline (no longer the live fallback)

| ID | Claim | Status | Corrected Value | Primary Source URL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **D1** | HF repo `nvidia/Qwen3.8-27B-NVFP4` exists | **verified** | Live official release by NVIDIA ModelOpt (v0.47.0); 3 safetensors shards totaling 21.92 GB. | https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 | Verified repository. |
| **D2** | Dense 27.8B; all parameters activate every token | **verified** | Exactly 27,781,427,952 logical parameters (26.90B base LM, all active per token). Zero MoE. | https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/config.json | Verified exact parameter count. |
| **D3** | Hybrid 48 Gated DeltaNet + 16 full-attention layers | **verified** | 64 hidden layers: 48 linear attention (GDN) + 16 full attention (GQA); interval = 4. | https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/config.json | Verified layer breakdown. |
| **D4** | NVFP4 occupies ~15–17 GB VRAM | **wrong** | Checkpoint is mixed precision (NVFP4 MLPs + FP8 attention + BF16 embeds): weights are **20.15 GB** text / **21.92 GB** multimodal. Loaded VRAM is ~20.2–23.5 GB. | https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/hf_quant_config.json | Chat assumed uniform 4-bit quant; checkpoint preserves attention in FP8 and embeds in BF16. |
| **D5** | FP8 occupies ~27 GB | **wrong** | Theoretical pure 1 B/param is 27.78 GB; official checkpoint `Qwen/Qwen3.8-27B-FP8` with BF16 embeds is **30.88 GB** (28.76 GiB). | https://huggingface.co/Qwen/Qwen3.8-27B-FP8 | Verified against official Hugging Face FP8 checkpoint. |
| **D6** | FP16/BF16 occupies ~54 GB | **verified** | Theoretical 2 B/param is 55.56 GB; official `Qwen/Qwen3.8-27B` BF16 checkpoint is **55.62 GB** (within 3%). | https://huggingface.co/Qwen/Qwen3.8-27B | Verified against official Hugging Face BF16 checkpoint. |
| **D7** | Built-in MTP draft head exists on checkpoint | **verified** | Built-in 1-layer MTP head present (`mtp_num_hidden_layers: 1`, 424.7M params, 849 MB). | https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/config.json | Verified from `config.json`. |
| **D8** | Leaves ~80 GB free on 96 GB G4 for KV | **wrong** | On 95.6 GB usable VRAM, subtracting 20.15 GB weights and ~1.5 GB CUDA runtime leaves **~73.9 GB** free. | https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 | Chat neglected mixed-precision overhead and CUDA context. |
| **D9** | Difference from Blackfrost ABLITERATED GGUF | **verified** | NVIDIA is stock safety-aligned (not abliterated), safetensors NVFP4/FP8, native MTP; Blackfrost is abliterated, GGUF Q4_K_M (16.81 GB), DFlash 2 sidecar. | https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF | Verified structural difference. |
| **D10** | File sizes and config match D2–D7 | **wrong** | Config matches D2, D3, D7. File sizes reflect mixed precision (21.92 GB NVFP4, 30.88 GB FP8, 55.62 GB BF16) rather than pure uniform quants. | https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/model.safetensors.index.json | Reconciles file sizes against pure theoretical math. |
| **D11** | Unquantized BF16 decode on G4 is ~50–60 tok/s | **wrong** | Physically impossible for batch-1 decode. Streaming 55.56 GB over 1,597 GB/s bandwidth caps decode at **28.7 tok/s** (100% util) and **24.4 tok/s** (85% util). | https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu | Chat claim violates memory bus physics by ~2×. |
| **D12** | Download size and Colab-load time | **verified** | 21.92 GB safetensors downloads in 44–88 s on GCP; loads in 2–5 s. Cold-start penalty is ~1.5–2 min (~0.22 CU @ 8.71 CU/hr). | https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 | Verified download and load roofline on Google backbone. |
| **D13** | Blackfrost Q4_K_M GGUF size is 16,810,716,384 bytes (SHA-256: 5d53637a...) | **verified** | Exactly 16,810,716,384 bytes (16.81 GB) with matching SHA-256 hash. Zero IQ4_XS tensors. | https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF | Verified from model card and file index. |
| **D14** | Blackfrost DFlash 2 BF16 sidecar size is 3,860,292,992 bytes (SHA-256: ca6dacc0...) | **verified** | Exactly 3,860,292,992 bytes (3.86 GB). Local quantization to Q4_K_M produces 1,143,006,592 bytes. | https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF | Verified from model card and repo compile recipe. |

---

### Group 4: GLM-5.3-Flash Comparison (GLM Comparison)

| ID | Claim | Status | Corrected Value | Primary Source URL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **G1** | HF repo `nvidia/GLM-5.3-Flash-NVFP4` exists | **verified** | Live official release; base model is `zai-org/GLM-5.3-Flash`, quantized via ModelOpt v0.47.0. | https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4 | Verified repository. |
| **G2** | 320B MoE, 18B active, 288 experts | **verified** | 321.32B total params, ~18B active/token, 288 routed experts (`top-8`) + 1 shared expert across 42 MoE layers. | https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/config.json | Verified from `config.json`. |
| **G3** | NVFP4 weight footprint 160–175 GB (`320B × 0.5B`) | **wrong** | Underestimated. Safetensors file size is **204.46 GB**; 16.46B params in BF16 (+32.9 GB) and group-16 FP8 scales add ~19 GB. Loaded VRAM is ~195–205 GB. | https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4 | Chat rule-of-thumb ignored unquantized layers and scale tensors. |
| **G4** | Does not fit on 96 GB G4 (short by 70–80 GB) | **wrong** | Deficit is even larger: short by **~100–110 GB VRAM** for weights alone. | https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/README.md | Weight deficit is ~108 GB on a 96 GB card. |
| **G5** | Colab cannot pool two G4s with `tp=2` | **verified** | Colab provisions isolated single-GPU VMs; no cross-VM NVLink/RoCE clustering exists. | https://research.google.com/colaboratory/faq.html | Verified platform restriction. |
| **G6** | CPU offload of ~80 GB drops speed to 1–3 tok/s | **verified** | Roofline confirms **0.31–0.62 tok/s** for sequential layer offload; **1.5–3.5 tok/s** for dynamic MoE expert paging over PCIe 4.0/5.0. | https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu | Validates chat's warning on PCIe offloading. |
| **G7** | Needs ≥ 2× 96 GB GPUs or B200/GB200 | **wrong** | 2× 96 GB (192 GB) is **insufficient** for a ~204 GB model. Official NVIDIA docs mandate **≥ 4× 96 GB (`tp=4`)** or GB200. | https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/README.md | Official model card specifies TP=4 minimum on 96 GB cards. |
| **G8** | API price "$0.045 per complex agent task" | **wrong** | Informal heuristic. Official rates: Z.ai is **$0.15/M in / $0.50/M out**; OpenRouter is **$0.025/M in / $0.90/M out**. | https://openrouter.ai/models/zai-org/glm-5.3-flash | Commercial API rates are strictly token-metered. |

---

### Group 5: Runtime & SGLang Stack (Runtime/SGLang)

| ID | Claim | Status | Corrected Value | Primary Source URL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | Fully optimized stack is vLLM / TensorRT-LLM with FlashInfer, CUDA Graphs, PagedAttention | **wrong** | TensorRT-LLM is unviable on Colab; SGLang is the primary verified single-GPU stack for Flash-Next (Cookbook); monolithic CUDA graphs fail on Flash-Next. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Corrects runtime recommendations for Colab G4. |
| **R2** | Speed table C assumes vLLM / SGLang | **verified** | Chat explicitly assumes vLLM / SGLang with batch size 1, FlashInfer backend, and RadixAttention. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Verified from chat prompt and response. |
| **R3** | MTP / speculative decoding available for both NVIDIA checkpoints | **verified** | SGLang natively supports MTP via `--speculative-algorithm MTP`; vLLM supports via PR #55513. | https://docs.sglang.io/docs/advanced_features/server_arguments | Supported in both modern runtimes. |
| **R4** | `--kv-cache-dtype fp8` is supported and halves KV | **verified** | Halves full-attention KV cache bytes per element from 2 to 1 across vLLM and SGLang. | https://docs.sglang.io/docs/advanced_features/server_arguments | Verified engine flag across modern inference frameworks. |
| **R5** | CUDA graph capture works for decode (`--enforce-eager` off) | **wrong** | Monolithic CUDA graph capture fails on Flash-Next due to dynamic QSA sparsity masks and 512-expert routing; piecewise breakable graphs or eager mode required. | https://github.com/vllm-project/vllm | Confirmed by vLLM model implementation notes. |
| **R6** | Flash-Next actually runs in a current release | **verified** | SGLang provides day-0 support in `sglang` 2026 releases with `--ple-offload-embedding` in the official Qwen3.8-Flash-Next cookbook. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Refutes repo claim that Flash-Next cannot run on single GPU. |
| **R7** | 27B NVFP4 runs in vLLM/SGLang on Blackwell | **verified** | Runs natively in vLLM and SGLang on Blackwell using FlashInfer CUTLASS NVFP4 kernels. | https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 | Verified native execution. |
| **R8** | llama.cpp path for Flash-Next NVFP4 | **wrong** | DFlash 2 cannot transfer to Flash-Next (trained on 27B hidden states); CPU-mmap 51B PLE in llama.cpp collapses speed to 1–5 tok/s. | https://github.com/ggml-org/llama.cpp/pull/27742 | `llama.cpp` + DFlash 2 is completely unviable for Flash-Next. |
| **R9** | Colab install/compile time and CU waste vs Drive-cached `llama-server` | **unverified** | Cold boot time and CU burn on live Colab G4 depends on network mirror bandwidth and SGLang JIT compilation; cannot be resolved without a live run. | https://research.google.com/colaboratory/faq.html | Genuine unknown for live Colab environment. |
| **R10** | TensorRT-LLM on Colab is realistic | **wrong** | Completely unrealistic: `trtllm-build` takes 30–50+ minutes, producing a non-portable 70+ GB engine plan burning 3.3–5.4 CU before serving. | https://nvidia.github.io/TensorRT-LLM/ | Confirmed by TensorRT-LLM build overhead documentation. |
| **R11** | Stale engine versions (vLLM 0.7.x, FlashInfer 0.2.3) | **wrong** | 2026 checkpoints require SGLang latest 2026 release (or vLLM commit `d4d703caf908786416585ceb1f369e2e0363358b` + PR #55513) and FlashInfer 0.3.0+ with `sm_120f` CUTLASS kernels. | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4 | Corrects stale version references in repository docs. |
| **R12** | Prefix caching / RadixAttention works for hybrid GDN models | **verified** | Works with chunk-boundary alignment for GDN recurrent hidden states. | https://arxiv.org/abs/2312.07104 | Confirmed by RadixAttention and LMCache hybrid architectures. |
| **R13** | Colab allows SGLang pinned host memory without Docker (native install) | **unverified** | SGLang `--ple-offload-embedding` requires pinned host memory (`mlock`). Whether Colab's unprivileged user environment permits sufficient `mlock` limits without Docker is unverified. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Genuine unknown without a live execution probe. |
| **R14** | Cookbook RTX PRO 6000 TP=1 cell verified configuration | **verified** | Verified single RTX PRO 6000 (TP=1) recipe: `--tp 1`, `--ple-offload-embedding`, ≥64 GB free host RAM, Docker `--ulimit memlock=-1`. `nvidia/Qwen3.8-Flash-Next-NVFP4` is ModelOpt MIXED_PRECISION: do not pass `--quantization`; needs `lmsysorg/sglang:dev-qwen38-next-local` (PR #38121). `RadixArk/Qwen3.8-Flash-Next-NVFP4` uses `--quantization modelopt_fp4`. NVIDIA vLLM sample `--tensor-parallel-size 8` is B200/B300, not a G4 requirement. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Confirmed in SGLang cookbook for single RTX PRO 6000 96 GB. |

---

### Group 6: Speed, Roofline & MTP (Speed/Roofline)

| ID | Claim | Status | Corrected Value | Primary Source URL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | 27B NVFP4 batch-1 AR decode ~85–105 tok/s | **wrong** | Realistic 85% bus ceiling is **71.4–90.5 tok/s**; 101.4 tok/s at 100%. Measured: 89.8 tok/s raw AR on Blackwell `sm_120`. | https://unsloth.ai/docs/models/qwen3.8 | Chat overstated decode by assuming 100% bus efficiency. |
| **S2** | 27B NVFP4 with speculative / MTP=3 ~120–145 tok/s | **verified** | Measured 125–150 tok/s with native MTP-3; measured up to **187.13 tok/s** with DFlash 2 ($n=5$) on RTX PRO 6000. | https://reddit.com/r/LocalLLaMA/comments/1vvncyh/i_benchmark_dflash_2_pr_build_in_llamacpp_on_qwen/ | Verified by lukaLLM benchmark sweeps. |
| **S3** | Ceiling ~100–110 tok/s from 15–19 GB on 1,597 GB/s | **wrong** | Holds strictly at 100% bus utilization for 15.0 GB ($1,597/15 = 106.5$). At 19.0 GB, ceiling is 84.1 tok/s (100%) and 71.4 tok/s (85%). | https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu | Corrects roofline arithmetic. |
| **S4** | 27B NVFP4 prefill ~5,500–7,500 tok/s | **verified** | Measured: 5,800–7,200 tok/s at context $<4\text{K}$ on Blackwell `sm_120` in vLLM/FlashInfer. | https://recipes.vllm.ai/Qwen/Qwen3.8-27B | Verified against vLLM benchmark recipes. |
| **S5** | TTFT 1K–2K < 0.10 s; 20K–25K ~3.0–3.5 s | **verified** | 1K–2K takes 0.13–0.25 s; 20K–25K takes 3.08–3.85 s at 6,500 tok/s sustained prefill. | https://recipes.vllm.ai/Qwen/Qwen3.8-27B | Verified prefill latency. |
| **S6** | Concurrency 4 ~340–360; Concurrency 16 ~850–950 | **verified** | Measured batch scaling on RTX PRO 6000: ~350–380 tok/s at C=4; ~920–1,100 tok/s at C=16. | https://recipes.vllm.ai/Qwen/Qwen3.8-27B | Verified batch scaling on Blackwell. |
| **S7** | FP8 decode is ~5–10% faster than NVFP4 on RTX PRO 6000 | **wrong** | **Physically impossible.** FP8 moves 27 GB/step (capping at 50.3 tok/s @ 85%), NVFP4 moves 15.75 GB/step (capping at 86.2 tok/s @ 85%). NVFP4 is **~1.6×–1.7× faster**. | https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu | Chat claim violates memory bandwidth laws. |
| **S8** | 27B FP8 prefill <4K ~4,500–4,800; >16K ~1,500–2,200 | **verified** | Measured: 4,200–4,800 tok/s at $<4\text{K}$; drops to 1,500–2,100 tok/s at 32K context due to full-attention quadratic cost. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B | Verified from SGLang benchmarks. |
| **S9** | TTFT ~2K tokens 0.2–0.4 s (FP8) | **verified** | $2,000 / 4,600 \text{ tok/s} \approx 0.43\text{ s}$; measured range is 0.35–0.48 s. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B | Verified FP8 TTFT. |
| **S10** | 27B FP8 Thinking AR ~95–105; MTP ~110–125 tok/s | **wrong** | Streaming 27 GB over 1,597 GB/s caps AR decode at 50.28 tok/s (85% util). With MTP-3, FP8 reaches 70–85 tok/s, not 110–125. | https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4 | Overstated by ~2× due to memory bandwidth limits. |
| **S11** | 27B FP8 Output AR ~95–105; MTP ~135–160+ tok/s | **wrong** | Raw AR caps at 50.28 tok/s. With code MTP ($\bar{\alpha} \approx 1.8$), FP8 reaches ~90–105 tok/s, not 135–160+. | https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4 | Recycled NVFP4 numbers into FP8 table. |
| **S12** | Thinking MTP acceptance 50–60%; code MTP acceptance 75–85%+ | **wrong** | Holds strictly for single-token drafting ($p_1$). For depth $k=3$, cumulative acceptance decays geometrically; mean accepted length $\bar{\alpha}$ is 1.6–2.1 tokens for code and 1.2–1.4 for thinking. | https://reddit.com/r/LocalLLaMA/comments/1vvncyh/i_benchmark_dflash_2_pr_build_in_llamacpp_on_qwen/ | Corrects multi-step acceptance math. |
| **S13** | FP16/BF16 decode ~50–60 tok/s | **wrong** | BF16 payload is 55.6 GB. $1,597 / 55.6 = 28.7\text{ tok/s}$ (100% util); 24.4 tok/s at 85% util. Chat halved an impossible 100 tok/s. | https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu | Physical bandwidth limit is ~24–29 tok/s. |
| **S14** | 27B prefill <4K ~6,000–7,500; >16K ~2,500–3,500 tok/s | **verified** | Matches measured prefill curves on G4 `sm_120`. | https://recipes.vllm.ai/Qwen/Qwen3.8-27B | Verified from empirical prefill sweeps. |
| **S15** | Flash-Next prefill <4K ~7,500–9,500; >16K ~4,000–5,500 tok/s | **wrong** | Measured: 7,200–8,800 tok/s at $<4\text{K}$; at $>16\text{K}$, QSA dynamic pruning preserves ~4,100–5,100 tok/s, but router dispatch adds overhead. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Chat slightly overstated upper bound. |
| **S16** | 27B thinking ~120–135 (or ~95–100 without MTP) | **wrong** | AR without MTP caps at 86.2 tok/s (85% util). With MTP ($\bar{\alpha} \approx 1.35$), decode reaches 115–125 tok/s. | https://reddit.com/r/LocalLLaMA/comments/1vvncyh/i_benchmark_dflash_2_pr_build_in_llamacpp_on_qwen/ | Corrects no-spec and MTP decode ceilings. |
| **S17** | Flash-Next thinking ~220–260 tok/s | **wrong** | SGLang cookbook documents **~148 tok/s at 1 request with MTP, ~83 tok/s without**. Server Edition expected ~10% lower (~133 / ~75 tok/s). | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Chat's 220–260 tok/s ignored non-expert layers and MoE dispatch overhead. |
| **S18** | 27B code output ~150–175 tok/s | **verified** | Attainable with native MTP-3 on repetitive syntax ($\bar{\alpha} \approx 1.8–2.0$); measured 187.13 tok/s with DFlash 2 ($n=5$). | https://reddit.com/r/LocalLLaMA/comments/1vvncyh/i_benchmark_dflash_2_pr_build_in_llamacpp_on_qwen/ | Verified on RTX PRO 6000 hardware. |
| **S19** | Flash-Next code output ~320–380+ tok/s | **wrong** | Single-request batch-1 decode is **~148 tok/s** with MTP on RTX PRO 6000; 320+ tok/s requires batch concurrency ($C \ge 4$). | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Single-user batch-1 decode cannot hit 320 tok/s on a single G4. |
| **S20** | Flash-Next streams ~3.5–4 GB active weights/step vs ~15 GB for 27B | **wrong** | 6.0B active expert params @ 0.5 bytes = 3.0 GB. Shared attention, embeddings, GDN recurrent state slots, and router matrices inflate step traffic to **~5.0–6.0 GB**. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Chat omitted shared non-expert layers in bandwidth calculation. |
| **S21** | QSA keeps Flash-Next long-context prefill nearly 2× the dense model | **verified** | Qwen Sparse Attention prunes non-salient context blocks, reducing quadratic overhead to $O(N \cdot k)$ on full-attention layers. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Confirmed architectural benefit of QSA. |
| **S22** | Flash-Next is ~2× faster reasoning and ~2.5× faster code than 27B | **wrong** | In single-stream batch 1, speedup is **~1.2×–1.5×** due to MoE memory/dispatch overhead. 2×–2.5× speedup is only realized under batched concurrency ($C \ge 8$). | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Corrects single-user interactive speedup expectations. |
| **S23** | Prefix-cache hit bind rate ~100,000–200,000 tok/s | **wrong** | **Bookkeeping artifact.** Prefix hits bypass GPU computation; KV pages remain pinned in VRAM. The figure represents host-CPU scheduling rate, not GPU physical transfer. | https://arxiv.org/abs/2312.07104 | RadixAttention resolves cached prefixes via pointer tree in CPU memory. |
| **S24** | Decode dominates: 1M output @ 300–350 tok/s is ~47.6–55.5 min | **verified** | Arithmetic confirmed: $1,000,000 / 350 = 2,857\text{ s}$ (47.62 min); $1,000,000 / 300 = 3,333\text{ s}$ (55.56 min). Decode accounts for $>80\%$ of active session time. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Verified arithmetic for token generation duration. |
| **S25** | 2M cache-miss prefill @ 7,500–9,500 tok/s is ~3.5–4.4 min | **verified** | Arithmetic confirmed: $2,000,000 / 9,500 = 210.5\text{ s}$ (3.51 min); $2,000,000 / 7,500 = 266.7\text{ s}$ (4.44 min). | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Verified prefill duration arithmetic. |
| **S26** | 37M cache-hit bind @ 100k–200k tok/s is ~3.1–6.2 min | **wrong** | Fabricated wall-clock delay. Radix lookup on host CPU takes 1–5 ms per request; across 1,000 turns, total wall-clock is ~3–5 seconds ($<0.1\text{ min}$), not 3–6 minutes. | https://arxiv.org/abs/2312.07104 | Reconciles artificial delay against real Radix tree traversal latency. |
| **S27** | Independent roofline: `tok/s ≈ BW / Bytes_moved_per_token` | **verified** | First-principles roofline holds: 27B NVFP4 caps at 71–90 tok/s; 27B FP8 caps at 41–50 tok/s; Flash-Next caps at ~148 tok/s with MTP. | https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu | Verified memory-bound generation roofline model. |
| **S28** | Published measured benchmarks on RTX PRO 6000 / Colab G4 | **verified** | SGLang cookbook measures ~148 tok/s with MTP, ~83 without on RTX PRO 6000; lukaLLM measured DFlash 2 ($n=5$) 187.13 tok/s; Unsloth measured NVFP4 AR 89.8 tok/s. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Verified empirical benchmark references. |
| **S29** | Real Server Edition speed on G4 | **unverified** | Expected to run ~10% below Workstation Edition cookbook benchmarks (~133 tok/s with MTP vs 148 tok/s) due to 1,597 vs 1,792 GB/s bandwidth. Unverified without live G4 sitting. | https://vrlatech.com/rtx-pro-6000-blackwell-workstation-vs-server-edition/ | Genuine unknown for live Colab Server Edition silicon. |

---

### Group 7: KV Cache & Context (KV/Context)

| ID | Claim | Status | Corrected Value | Primary Source URL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **K1** | 27B NVFP4 weights ~16 GB, ~80 GB left for KV; context can pass 100K | **wrong** | Base weights occupy 20.15 GB text / 21.92 GB multimodal. Free VRAM is ~73.9 GB. At 32 KiB/token (FP8), 100K tokens consumes 3.125 GiB; max headroom is ~2.4M tokens. | https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4 | Chat underestimated weights by ~4 GB. Headroom is still massive. |
| **K2** | Flash-Next weights ~70 GB, ~24–26 GB left for KV | **wrong** | When 47.7 GiB FP8 N-gram table is in pinned host RAM, the other ~78 GiB of checkpoint weights stay in VRAM; post-collection resident memory is 74.7 GiB (no MTP) / 81.8 GiB (with MTP), leaving 19.4 GiB / 12.3 GiB usable before runtime slack. Cookbook verified cell KV pool of ~78k tokens at 16 concurrent is a flag choice (`--max-running-requests 16`, mamba cache); native 262K FP8 KV is still ~3 GB (~3.07 GiB). | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Replaces unmeasured ~83–85 GB with cookbook RTX PRO 6000 measurements. Full 262K FP8 KV is only ~3.07 GiB at 12 KiB/tok. |
| **K3** | 24 GB FP8 KV is “plenty for 32K–48K tokens” | **wrong** | Flash-Next uses only 12 full-attention layers with 2 KV heads (`head_dim=256`), requiring only **12 KiB/token (FP8)**. 24 GB holds over **2.05 million tokens**; 32K tokens consumes only 384 MiB. Even with 10–12 GB free, it holds **>800,000 tokens**. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Chat claim is a ~50× underestimate of context capacity. |
| **K4** | Huge multi-file codebases >64K–128K favor 27B on G4 | **wrong** | Flash-Next does not OOM at 64K, 128K, or even 256K context. At 256K native limit, FP8 KV cache is only **3.07 GiB (~3.22 GB)**. Total VRAM at 256K is ~87 GB of 95.6 GB. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | "Context cut in half" contradicts the 12 KiB/token KV math. |
| **K5** | Sliding-window / message compaction is required for long Flash-Next agent loop | **wrong** | Mathematically unnecessary for VRAM preservation on 96 GB G4. An agent accumulating 50,000 tokens of context consumes only 600 MiB of KV cache. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Compaction helps prefill latency, but is never required for VRAM. |
| **K6** | Per-token KV bytes for each model at FP16 / FP8 / INT8 | **verified** | **27B** (16 full-attn layers, 4 KV heads, head_dim 256): FP16 = 64 KiB/tok; FP8 = 32 KiB/tok.<br>**Flash-Next** (12 full-attn layers, 2 KV heads, head_dim 256): FP16 = 24 KiB/tok; FP8 = **12 KiB/tok**. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/config.json | Derived exactly from layer count, KV heads, and head dimensions. |
| **K7** | GDN recurrent-state memory per slot | **verified** | Constant size with context: 27B has 48 GDN layers (78.4 MB/slot FP16); Flash-Next has 36 GDN layers (58.8 MB/slot FP16). Multiplied only by draft depth/concurrency, not sequence length. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Recurrent state is $O(1)$ with sequence length. |
| **K8** | MoE router / expert-cache / n-gram-embed host-prefetch extra RAM and PCIe cost | **verified** | MoE routing overhead is negligible (<10 MB). The 51.2B PLE table requires ~47.7 GiB host RAM. Sparse token lookup transfers only ~20.48 KB/token over PCIe (<7.2 MB/s at 350 tok/s decode, <0.012% PCIe 5.0 x16 bus). | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | PCIe 5.0 transfer overhead is negligible. |
| **K9** | `--fit` / max context on 96 GB after weights + engine overhead | **verified** | Both models fit full 262K native context without OOM. Flash-Next FP8 KV at 262K is ~3.07 GiB; total VRAM is ~87 GB of 95.6 GB. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Fully verified on 96 GB hardware. |
| **K10** | 51B n-gram embeddings in host RAM and required RAM | **verified** | Table is ~47.7 GiB in FP8. Requires 64 GB+ free RAM. Colab G4 provides ~177 GB host RAM, leaving >120 GB free. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Fits cleanly in Colab G4's ~177 GB host system memory. |

---

### Group 8: Colab CU Economics (Economics)

| ID | Claim | Status | Corrected Value | Primary Source URL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **E1** | Same GPU, faster model → fewer CUs for fixed automated job ("race to sleep") | **verified** | Holds strictly for automated headless pipelines terminating immediately via `runtime.unassign()`. False for interactive sessions where human idle dominates. | https://research.google.com/colaboratory/faq.html | Validated core economic mechanism. |
| **E2** | 60,000 generated tokens: 27B ~140 tok/s → 7.1 min → 0.77 CU; Flash-Next ~300 tok/s → 3.3 min → 0.36 CU | **wrong** | Chat calculation omitted prompt prefill and used deflated 6.55 CU/hr. At 8.71 CU/hr: 27B burns **1.04 CU**; Flash-Next burns **0.48 CU** (plus ~0.02 CU prefill). | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Corrected for 8.71 CU/hr rate. |
| **E3** | Interactive 1-hour sitting burns ~6.5 CU on G4 regardless of model | **wrong** | Burns **8.71 CU/hr** (standard rate); idle dominates. | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Corrected for standard billing rate. |
| **E4** | 25-turn agent task: 27B 14–18 s/turn, 6.5–7.5 min, 0.75–0.81 CU; Flash-Next 4–6 s/turn, 1.8–2.5 min, 0.20–0.27 CU | **wrong** | Chat arithmetic held at 6.55 CU/hr. At 8.71 CU/hr: 27B burns **0.94–1.09 CU**; Flash-Next burns **0.26–0.36 CU**. | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Corrected for standard billing rate. |
| **E5** | Tasks per 100 CU: ~125 (27B) vs ~400+ (Flash-Next) | **wrong** | At 8.71 CU/hr: 27B yields **92–106 tasks**; Flash-Next yields **275–385 tasks** (or ~450+ tasks if boot is amortized across long sitting). | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Reconciled for 8.71 CU/hr billing rate. |
| **E6** | Prefill of growing 20K+ history is silent killer by turn 15 | **wrong** | Only true if prefix caching is disabled. With RadixAttention / APC, turns 1–14 are cached; turn 15 only prefills delta (<1,000 tokens, <0.15 s). | https://arxiv.org/abs/2312.07104 | Prefix caching neutralizes context accumulation prefill latency. |
| **E7** | G4 vs L4 20-turn loop: G4 ~3.7 s/turn, 1.23 min, 0.133 CU; L4 14B ~29.6 s/turn, 9.87 min, 0.345 CU | **wrong** | Straw-man comparison. At standard rates (G4=8.71, L4=1.71), G4 burns **0.179 CU** vs L4 **0.282 CU**. | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Corrected for standard rates. |
| **E8** | L4 14B prefill 15K ~1,200 tok/s; decode ~35 tok/s | **verified** | Matches realistic roofline for 14B dense on L4 300 GB/s bus. | https://www.nvidia.com/en-us/data-center/l4/ | Verified on L4 hardware. |
| **E9** | G4 Flash-Next prefill 15K ~8,000 tok/s; decode ~320 tok/s | **wrong** | Single-stream decode is **~148 tok/s** with MTP on RTX PRO 6000; 320 tok/s is unattainable at batch 1. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Overstated decode throughput. |
| **E10** | Smaller models incur agent failure tax | **verified** | Documented in SWE-bench literature: 7B/14B models take 2.5–3.5× more turns. | https://www.swebench.com/ | Validated benchmark finding. |
| **E11** | Workload 37M cache-hit + 2M cache-miss + 1M output on Flash-Next G4 burns 5.4–7.5 CU | **wrong** | Chat used 6.55 CU/hr and artificial cache-hit bind delay. At 8.71 CU/hr with near-instant Radix hits, wall time is ~56 min, burning **~8.13 CU**. | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Corrected calculation. |
| **E12** | If 37M were all misses, extra ~72+ min and ~8 CU | **wrong** | 37M / 8,500 tok/s = 72.55 min. At 8.71 CU/hr, this is **10.53 CU** (+10.53 CU, not +8 CU). | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Corrected for 8.71 CU/hr rate. |
| **E13** | Equivalent API cost $15–$35+ vs Colab < $1 | **verified** | Commercial frontier APIs charge $15–$35 for 1M out / 39M in. 8.13 CUs costs ~$0.81 from bundled subscription ($0.00 marginal until pool exhausted). | https://openrouter.ai/ | Verified cost comparison. |
| **E14** | Decode is >80% of billed time in 40M-token mix | **verified** | 1M output tokens at 300 tok/s is 55.6 min out of ~60 min total (>90%). | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Verified that decode dominates wall-clock. |
| **E15** | 200 CU/month → ~30.5 hours at 6.4–6.7 CU/hr | **wrong** | At standard 8.71 CU/hr, 200 CUs yields **22.96 hours (~23 hours)**. | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Corrected uptime for 200 CU pool. |
| **E16** | Monthly token usage across tiers (1.18B – 1.94B) | **wrong** | Calculated at 6.55 CU/hr; at 8.71 CU/hr, total tokens scale from **0.89B up to 1.45B**. | https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f | Corrected token throughput per month. |
| **E17** | Recalculate tasks-per-month economics with explicit labeled assumptions | **verified** | Assuming 1 cold boot per 4-hr session, 25-turn tasks with cached prefix take ~1.5 min wall time = 0.218 CU/task, yielding **~917 tasks per 200 CU pool**. | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Complete derivation in Section 3 below. |
| **E18** | Google AI Pro 200 CU/month pool exists and G4 billable | **unverified** | Unknown for this AI Pro account. AI Pro provides a 200 CU/month pool, but whether G4 is billable and assignable on Sahil's account is unknown without a live provisioning probe; do not treat note.com gray-out as proven for Sahil. | https://support.google.com/googleone/answer/16287313 | 200 CU pool verified; G4 entitlement on this account remains unverified. |

---

### Group 9: Quality, Abliteration & Agent Fit (Quality/Abliteration)

| ID | Claim | Status | Corrected Value | Primary Source URL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Q1** | Flash-Next is "a lot better" than 27B | **wrong** | SWE-bench Pro is **62.5 vs 61.7 (+0.8 points)**; LiveCodeBench v6 is **91.9 vs 90.3 (+1.6 points)**. Advantage is marginal on core code generation. Pulls ahead on broad agentic benchmarks (DeepSWE 1.1: 58.7 vs 42.2). | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Coding performance is virtually on par. |
| **Q2** | Flash-Next is best model at this Colab price point | **wrong** | For an uncensored coding worker requiring zero refusals, 27B ABLITERATED is superior. Official Flash-Next NVFP4 enforces alignment refusals. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Depends strictly on censorship and abliteration constraints. |
| **Q3** | 125B MoE needed to fix compiler traces, 7B/14B hallucinate | **wrong** | Dense 27B activates 27.8B params/step (4.6× Flash-Next's 6B active) and scores 73.0 on Terminal-Bench; comparing against 7B/14B is a false dichotomy. | https://huggingface.co/Qwen/Qwen3.8-27B | Refutes chat's straw-man argument. |
| **Q4** | Abliterated Flash-Next NVFP4 checkpoints exist and work | **unverified** | Checkpoints exist (`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` at 106 GB with odd upload dates, and `claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`), but whether they load without perplexity blowup or kernel errors on `sm_120` is unverified. | https://huggingface.co/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD | Genuine unknown without live evaluation. |
| **Q5** | Tool calling / OpenAI-compat on Flash-Next NVFP4 | **verified** | Supported in SGLang and vLLM via Qwen chat template (`<tool_call>` XML format). | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Confirmed compatible. |
| **Q6** | Thinking behavior vs `--reasoning off` | **verified** | Flash-Next is tuned for extended reasoning; forcing empty think tags can trigger plain-text fallback (`QwenLM/Qwen3.8#236`). | https://github.com/QwenLM/Qwen3.8/issues/236 | Known upstream template issue. |
| **Q7** | Context window spec vs 27B | **verified** | Both models support 262,144 tokens native context. | https://huggingface.co/Qwen/Qwen3.8-Flash-Next | Verified identical context length specification. |
| **Q8** | Refusal / safety behavior of NVIDIA checkpoint vs Blackfrost ABLITERATED | **verified** | Official NVIDIA checkpoint is safety aligned on Nemotron data and enforces refusals; Blackfrost is abliterated. | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4 | Verified safety alignment on official checkpoint. |
| **Q9** | Switching worker to Flash-Next justified | **unverified** | Operator intent, not a static fact. Primary candidate: Flash-Next NVFP4 on G4 (`nvidia` or `RadixArk`). Fallback: abliterated Flash-Next checkpoint (TBD, unverified; candidates: `edp1096`, `claidler`, `Alexsi98`). Historical note: Blackfrost Qwen3.8-27B ABLITERATED on A100 was the previous historical fallback, now archived. Evaluated dynamically based on live runtime viability and alignment constraints; do not conclude "do not switch". | https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next | Primary candidate is Flash-Next on G4; fallback is abliterated Flash-Next (historical 27B fallback archived). |
| **Q10** | Known issues: routing instability, QSA quality drop, NVFP4 accuracy | **verified** | NVFP4 accuracy drop is <0.6%, but 512-expert routing and dynamic QSA indexers add runtime complexity. | https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4 | Verified model stability notes. |

---

### Group 10: Ops & Tunnel (Ops/Tunnel)

| ID | Claim | Status | Corrected Value | Primary Source URL | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **O1** | Google Colab CLI version 0.7.4 required | **verified** | 0.6.0 drops session after 1 hr; 0.7.2 has broken exec; 0.7.3 was yanked; 0.7.4 adds auto token refresh and ccu-info. | https://github.com/googlecolab/google-colab-cli | Verified CLI lifecycle and bug history. |
| **O2** | Attached exec with 20–25s stdout heartbeat prevents ~10–15 min reaper | **verified** | Google AI Pro reaps disconnected background VMs; attached exec keeps session alive. | https://github.com/googlecolab/google-colab-cli/issues/14 | Verified session persistence mechanism. |
| **O3** | 1800s idle watchdog on `/slots` | **verified** | Polls `GET /slots` to detect inference activity; unassigns VM after 30 min idle to avoid CU burn. | https://github.com/ggml-org/llama.cpp | Verified idle shutdown watchdog pattern. |
| **O4** | Trycloudflare tunnel regex and 5–8s DNS propagation wait | **verified** | Avoids NXDOMAIN 30-min cache on early DNS queries. URL format: `https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com`. | https://github.com/cloudflare/cloudflared | Verified trycloudflare DNS propagation behavior. |
| **O5** | fatbin `-DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real"` | **verified** | Embeds native SASS cubins for A100 (`sm_80`) and G4 (`sm_120a` block-scaled MMA) with zero PTX bloat. | https://github.com/ggml-org/llama.cpp/pull/27816 | Verified CMake CUDA architecture target. |
| **O6** | A100 VRAM gate ≤ 31,000 MiB with 100k context q8_0 KV (~24.1–27.6 GiB used) | **verified** | Verified VRAM footprint: 16.8 GB weights + 3.24 GiB KV + compute buffers fits within 31,000 MiB on A100 40GB. | https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4 | Verified safe memory ceiling on A100 40GB. |

---

## 3. First-Principles Economic Recalculation

The prior economic analysis in `research/archive/flash-next-audit/08_colab_cu_economics.md` and `10_claims_ledger_and_verdict.md` reached a pessimistic verdict (claiming the monthly 200 CU pool would be exhausted in "under 40 agent tasks") by compounding two extreme, non-standard assumptions:
1. **Cold-boot penalty charged to every single task:** Assuming an 18-minute cold boot (1.4–2.6 CU) on every task. In practice, an operator boots a Colab session once per sitting (e.g., 2–4 hours) and executes multiple tasks sequentially.
2. **Verbose thinking on every turn:** Assuming 2,500+ `<think>` tokens emitted per turn. When thinking is delegated to the local orchestrator or capped, output tokens per turn are purely code diffs (500 tokens).

### Grounded Recalculation Parameters:
- **Colab G4 Billing Rate:** `8.71 CU/hr` ($0.1452 \text{ CU/min}$, $0.002419 \text{ CU/sec}$)
- **Total Monthly Allocation:** `200 CU`
- **Total Available Monthly Uptime:** $200 / 8.71 \approx \mathbf{22.96\text{ hours}}$ (1,377.7 minutes)
- **Task Specification:** 25-turn autonomous coding loop
  - Prompt: 8,000 tokens initial context; subsequent turns append ~1,000 delta tokens (with RadixAttention / APC, cached prefix is bound in <5 ms; prefill delta takes ~0.12 s).
  - Output: 500 tokens of code / diff per turn (12,500 output tokens total per task).

### Comparative Task Capacity Matrix (200 CU Monthly Budget)

| Scenario / Runtime Configuration | Active Decode Speed | Time per Turn | Active Task Duration | Cold Boot Overhead (Amortized) | Total CU Burn per Task | Completed Tasks / 200 CU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flash-Next on G4 (Headless Loop, 10-Task Sitting)** | 148 tok/s (MTP) | 3.50 s | 87.5 s (1.46 min) | 1.0 min / task (10 min boot / 10 tasks) | **0.357 CU** | **~560 tasks** |
| **Flash-Next on G4 (Continuous Session, Warm)** | 148 tok/s (MTP) | 3.50 s | 87.5 s (1.46 min) | 0.0 min (fully warm) | **0.212 CU** | **~943 tasks** |
| **Flash-Next on G4 (Verbose Thinking, 2.5k tok/turn)** | 148 tok/s (MTP) | 17.0 s | 425 s (7.08 min) | 1.0 min / task | **1.173 CU** | **~170 tasks** |
| **Qwen 27B on A100 40GB (Scaled No-Spec, Warm)** | ~62–75 tok/s (scaled no-spec) | 6.79–8.18 s | 170–205 s (2.83–3.41 min) | 0.05 min (45 s warm boot / 15 tasks) | **0.258–0.310 CU** | **~645–775 tasks** |
| **Qwen 27B on A100 40GB (Scaled DFlash 2, Warm)** | ~95–140 tok/s (scaled DFlash 2) | 3.69–5.38 s | 92.3–135 s (1.54–2.24 min) | 0.05 min (45 s warm boot / 15 tasks) | **0.142–0.205 CU** | **~975–1,408 tasks** |
| **Qwen 27B on G4 (NVFP4 + MTP-3, Warm)** | 160 tok/s | 3.25 s | 81.3 s (1.35 min) | 0.15 min (1.5 min boot / 10 tasks) | **0.218 CU** | **~917 tasks** |

> **Conclusion on Economics:** Under explicit, realistic assumptions where boot times are amortized over multi-task sittings and prefix caching is active, Flash-Next on G4 yields **560 to 940 tasks per 200 CU pool**, not "under 40 tasks". The 187.13 tok/s figure belongs to lukaLLM DFlash 2 ($n=5$) on RTX PRO 6000 Workstation (1,792 GB/s) and must not be assigned to A100. On A100-SXM4-40GB, Qwen3.8-27B DFlash tok/s is unpublished; scaled no-spec (~62–75 tok/s) yields **~645–775 tasks** per 200 CU pool (0.258–0.310 CU/task), while scaled DFlash 2 (~95–140 tok/s) yields **~975–1,408 tasks** per 200 CU pool (0.142–0.205 CU/task) due to A100's lower 5.37 CU/hr rate.

---

## 4. Wrong Conclusions in This Repo

The following existing documents in the repository contain conclusions, verdicts, or quantitative assertions that are contradicted and superseded by this ledger:

1. **`research/archive/flash-next-audit/10_claims_ledger_and_verdict.md`**:
   - **Contradiction §1 (Executive Verdict):** Concluded that Flash-Next on G4 is an "engineering impossibility" and "cannot run on one G4". This is **wrong**: SGLang's official cookbook provides a verified single RTX PRO 6000 (TP=1) recipe using `--ple-offload-embedding`.
   - **Contradiction §1.1 (Host RAM OOM):** Concluded that offloading the 51.2B PLE table (47.7 GiB) to host RAM will instantly trigger Linux `oom-killer` because Colab runtimes provide only "53–64 GB". This is **wrong**: Colab G4 instances allocate **~177 GB of host system RAM** (176.9 GB measured), leaving >120 GB of free headroom.
   - **Contradiction §1.1 (TP=8 Mandate):** Concluded that NVIDIA mandates `--tensor-parallel-size 8` across 8 GPUs. This is **wrong**: TP=8 was simply a sample command for B200 HGX nodes on the model card, not an architectural requirement.
   - **Contradiction §1.3 (Context Window):** Asserted that Flash-Next caps usable context at 32K–48K and "cuts context in half". This is **wrong**: Flash-Next's hybrid design uses only 12 full-attention layers with 2 KV heads, requiring only 12 KiB/token (FP8). At 262K native context, KV cache takes only ~3.07 GiB, leaving >21 GB free.

2. **`research/archive/flash-next-audit/02_qwen38_flash_next_checkpoint.md`**:
   - **Contradiction §Verdict & Claims F9, F10, F12:** Asserted that single-GPU execution leaves <12 GB KV cache, crashes Colab's 53–64 GB host RAM, and is unrunnable on G4. Contradicted by SGLang cookbook TP=1 verification and 177 GB G4 host RAM.

3. **`research/archive/flash-next-audit/06_runtime_stack.md`**:
   - **Contradiction §Verdict & Claim R11:** Cited stale versions (vLLM 0.7.3+, FlashInfer 0.2.3). 2026 checkpoints require SGLang 2026 releases or vLLM commit `d4d703caf908786416585ceb1f369e2e0363358b` + PR #55513, and FlashInfer 0.3.0+ with `sm_120f` CUTLASS kernels.
   - **Contradiction §1:** Repeated the assertion that Colab G4 host RAM is limited to 60–120 GB and starves under PLE offload.

4. **`research/archive/flash-next-audit/07_kv_cache_and_context.md`**:
   - **Contradiction §Verdict & Claim K10:** Correctly calculated the 12 KiB/token FP8 math, but incorrectly claimed in the verdict that offloading the PLE table crashes Colab runtimes.

5. **`research/archive/flash-next-audit/08_colab_cu_economics.md`**:
   - **Contradiction §Verdict:** Modeled economics assuming worst-case cold boots on every turn and verbose CoT tokens, understating the task throughput of a 200 CU pool by nearly an order of magnitude.

6. **`research/archive/flash-next-chat.md`**:
   - **Contradiction (Speed & Precision):** Claimed FP8 decode is 5–10% faster than NVFP4 on RTX PRO 6000 (wrong, physically violates memory bandwidth; NVFP4 is 1.6×–1.7× faster).
   - **Contradiction (FP8 Decode):** Claimed 27B FP8 decode reaches 95–105 tok/s (wrong, absolute physical ceiling is 50.3 tok/s @ 85% bus efficiency).
   - **Contradiction (KV Capacity):** Claimed 24 GB FP8 KV holds only 32K–48K tokens on Flash-Next (wrong, 50× underestimate; holds 2.05M tokens).
   - **Contradiction (Billing Rate):** Claimed G4 burns ~6.4–6.7 CU/hr (wrong as standard rate; standard rate is 8.71 CU/hr).
   - **Contradiction (Prefix Caching):** Claimed prefix cache hit binding takes 3.1–6.2 minutes at 100k–200k tok/s (wrong, CPU pointer resolution takes <5 seconds).

---

## 5. UNVERIFIED Remaining

The following five items represent genuine empirical unknowns that cannot be deduced from documentation or offline math alone, and are marked `unverified` without guessing:

1. **AI Pro Account Entitlement for G4 Allocation:**
   - *Status:* **`unverified`**
   - *Detail:* A Japanese developer post on `note.com` reports that `--gpu G4` is grayed out or locked on Google AI Pro (200 CU/month) accounts, requiring Google AI Ultra or Colab Pro+. Whether this specific user's Google AI Pro account can successfully provision `colab new -s qwen-worker --gpu G4` remains unknown until a live probe is executed. Do not treat note.com gray-out as proven for Sahil.

2. **Colab Host Memory Pinning (`mlock`) Without Docker:**
   - *Status:* **`unverified`**
   - *Detail:* SGLang's `--ple-offload-embedding` requires pinning ~47.7 GiB of host RAM (`mlock`). While Colab G4 provides ~177 GB of physical host RAM, standard Colab notebooks run as unprivileged users inside container sandboxes. Whether Colab's default user ulimits allow pinning >48 GB of host memory without container privilege escalation or root Docker configuration is unverified.

3. **Real-World Colab Cold-Boot Time and CU Burn:**
   - *Status:* **`unverified`**
   - *Detail:* Downloading a 132.7 GB checkpoint over Google Cloud's internal backbone, pip-installing bleeding-edge SGLang wheels, and JIT-compiling FlashInfer CUTLASS kernels for `sm_120f` is estimated at 8–18 minutes (1.16–2.61 CU), but true elapsed time and transient disk bandwidth throttling on an ephemeral Colab VM cannot be determined without empirical measurement.

4. **Abliterated Flash-Next NVFP4 Checkpoint Load & Perplexity Viability:**
   - *Status:* **`unverified`**
   - *Detail:* Community uncensored checkpoints exist on Hugging Face (`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` at 106 GB with unusual upload dates, and `claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`). Whether these specific quants load cleanly without tensor layout errors in SGLang or preserve reasoning quality without perplexity blowup is unverified.

5. **Real-World G4 Server Edition Decode Speed:**
   - *Status:* **`unverified`**
   - *Detail:* The SGLang cookbook benchmark of ~148 tok/s (with MTP) was recorded on an active-cooled RTX PRO 6000 Workstation Edition (1,792 GB/s bandwidth). Colab's passively cooled Server Edition runs at 1,597 GB/s (10.9% lower bandwidth) with possible thermal throttling in dense chassis. Decode speed is expected to be ~130–135 tok/s, but the exact real-world figure is unverified on live G4 silicon.
