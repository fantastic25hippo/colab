# Master Claims Ledger, Architectural Synthesis, and Final Verdict

Researched: 2026-10-17 | Engine: Gemini 3.8 Flash (Medium) | Job: 10 (Synthesizer)

---

## 1. Executive Verdict (One Page)

### Can the operator switch the Colab worker to Qwen 3.8 Flash-Next NVFP4 on G4?

**Verdict:** **NO.**

The operator must **not** switch the Colab worker to `nvidia/Qwen3.8-Flash-Next-NVFP4` on Colab G4. The proposal in `Running Qwen flash next on Colab.md` is an unviable paper architecture constructed on invalid hardware assumptions, arithmetic hallucinations, broken runtime compatibility, and severe operational regressions. The project must remain locked to the planned worker: **Blackfrost Qwen 3.8 27B ABLITERATED (`Q4_K_M`) on A100-SXM4-40GB via `llama.cpp` + DFlash 2**.

The three strongest reasons are:

### 1. Entitlement Impossibility, Checkpoint Bloat, and Host Memory Collapse
* **Account Entitlement Lockout:** Direct provisioning of Colab `--gpu G4` (NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB GDDR7) is tier-gated by Google. On the operator's **Google AI Pro (200 CU/month)** account, G4 is grayed out under Google's Premium GPU Access policy (accessible only on Google AI Ultra or Colab Pro+). Attempting `colab new --gpu G4` fails or returns service unavailable.
* **132.7 GB Checkpoint Footprint:** The chat claimed Flash-Next is a ~70 GB download that fits easily on a 96 GB card. Primary source inspection of `nvidia/Qwen3.8-Flash-Next-NVFP4` reveals the download is **132.68 GB** across 11 safetensors files (78.96 GB base model + 53.72 GB PLE/MTP shard). It exhausts Colab's ephemeral scratch disk (~100–150 GB) and exceeds standard Google Drive allocations.
* **The Fatal 51.2 GB Host RAM Crash:** The model's 51.2B-parameter Phrase-Level Embedding (PLE) n-gram table (51.20 GB in FP8) is designed to be offloaded to host system RAM. While datacenter clusters have 512+ GB host RAM, standard Google Colab runtimes provide only **53–64 GB of host RAM**. Pinning 51.2 GB in host memory instantly triggers the Linux kernel Out-Of-Memory killer (`oom-killer`), destroying the VM process before inference begins. NVIDIA's official deployment documentation mandates **`--tensor-parallel-size 8` across 8 Blackwell GPUs**; running it on a single Colab VM is an engineering impossibility.

### 2. Destruction of the Zero-Overhead Toolchain and Operational Stack
* **`llama.cpp` and DFlash 2 Incompatibility:** The locked production stack relies on a pre-compiled, Drive-cached `llama-server` binary (`efe3626` / `b11552`) with a dedicated Blackfrost DFlash 2 sidecar trained on 27B hidden representations. Flash-Next is an experimental MoE architecture (`qwen4_exp`, 512 experts, top-10 routing) with no mathematical compatibility with DFlash 2.
* **Cold-Start CU Burn:** Migrating Flash-Next forces the adoption of vLLM or SGLang. On an ephemeral Colab instance, pip-installing dependencies, fetching 132.7 GB of weights, and JIT-compiling FlashInfer CUTLASS kernels for Blackwell workstation silicon (`sm_120f` lacking TMEM) takes **13 to 24 minutes** on every single boot. This burns **1.4 to 2.6 CUs ($0.14–$0.26)** before serving a single prompt, compared to **<45 seconds (0.07 CU)** for the Drive-cached `llama-server`.
* **Broken Tool-Calling & Plain-Text Fallback:** Flash-Next inherits the upstream `QwenLM/Qwen3.8#236` Jinja template bug: when thinking is disabled (`--reasoning off`), it injects `<think>\n\n</think>\n\n`, causing the model to treat reasoning as completed and emit plain conversational text rather than structured `<tool_call>` blocks.

### 3. Censorship Re-Introduction and the Thinking-Token Economic Paradox
* **Re-introduction of Safety Refusals:** The project explicitly requires an uncensored coding worker. The official `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint is stock, aligned, and calibrated on Nemotron safety datasets. It will systematically refuse cybersecurity, penetration testing, exploit analysis, and low-level systems administration prompts that the current Blackfrost ABLITERATED checkpoint handles without hesitation.
* **The "Reasoning" Economic Trap:** Flash-Next achieves its benchmark gains only when running in extended thinking mode (`reasoning_effort=xhigh`), generating 1,500 to 3,500 `<think>` tokens per turn. Even at 240 tok/s decode, generating 2,500 thinking tokens + 500 code tokens takes **10.7 to 12.5 seconds** per turn. In contrast, the 27B worker with thinking delegated to the local orchestrator emits only 500 pure code tokens at ~160 tok/s in **3.1 to 4.3 seconds**. In a 25-turn agent loop, Flash-Next burns **0.48 to 0.56 CU**, whereas the non-thinking 27B worker burns **0.15 to 0.16 CU**. Flash-Next burns **3× to 4× more CUs per deliverable** while taking 2.5× longer in wall-clock time.
* **True Billing Reality:** Colab G4 bills at **8.71 CU/hr** (standard metered rate), not the chat's deflated 6.4–6.7 CU/hr. At 8.71 CU/hr, a 200 CU monthly pool yields only **22.9 hours** of uptime. Sinking those units into cold boots and verbose thinking tokens exhausts the monthly quota in under 40 agent tasks.

---

## 2. Master Claims Ledger

Labels adhere strictly to the findings in research papers 01 through 09. Where worker papers identified discrepancies or where primary file inspections superseded theoretical estimates (e.g., Job 02's empirical safetensors analysis overriding Job 07's theoretical weight estimate), the corrected reality is documented with explicit justification.

| ID | Chat claim (short) | Worker Job | Label | Corrected value | Best source |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **H1** | Colab G4 is NVIDIA RTX PRO 6000 Blackwell | Job 01 | **CONFIRMED** | RTX PRO 6000 Blackwell Server Edition (passively cooled PCIe Gen 5, GB202 silicon). | [NVIDIA Server Edition Datasheet](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) |
| **H2** | Card has 96 GB VRAM | Job 01 | **CONFIRMED** | 96 GB GDDR7 ECC (usable: 95.60 GiB / 97,896 MiB in `nvidia-smi`). | [PNY Server Edition Datasheet](https://www.pny.com/en-eu/file%20library/professional/datasheet/data%20center%20cards/pny-nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf) |
| **H3** | Memory bandwidth is 1,597 GB/s GDDR7 | Job 01 | **CONFIRMED** | 1,597 GB/s GDDR7 (512-bit bus @ 24.95 Gbps). Not HBM. Workstation card is 1,792 GB/s. | [Lenovo ThinkSystem LP2263](https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu) |
| **H4** | FP4 tensor compute is 4 PFLOPS | Job 01 | **PARTIALLY TRUE** | 4.0 PFLOPS with 2:4 structural sparsity; **2.0 PFLOPS** dense GEMM (no sparsity). | [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/) |
| **H5** | FP8 tensor compute is ~1,000+ TFLOPS | Job 01 | **CONFIRMED** | **1,000 TFLOPS (1.0 PFLOP)** dense; 2,000 TFLOPS with 2:4 structural sparsity. | [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/) |
| **H6** | Colab G4 burns ~6.4–6.7 CU/hr | Job 01 | **PARTIALLY TRUE** | **8.71 CU/hr** standard metered rate; 6.40–6.74 CU/hr was early rollout / off-peak fallback. | [McCormick Colab Pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/), [Qiita Benchmark](https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f) |
| **H7** | Colab L4 burns ~2.1 CU/hr | Job 01 | **PARTIALLY TRUE** | Standard baseline is **1.71 CU/hr**; reaches 2.09–2.10 CU/hr under dynamic peak demand. | [McCormick Colab Pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/) |
| **H8** | G4 hourly price is ~3.1× L4 | Job 01 | **PARTIALLY TRUE** | True only for off-peak G4 vs peak L4 ($6.5/2.1 = 3.1\times$). At standard rates ($8.71/1.71$), G4 is **5.09× (~5.1×)** L4. | [Job 01 Hardware & CU Audit](01_colab_g4_hardware_and_cu.md) |
| **H9** | L4 has 24 GB VRAM and 300 GB/s bandwidth (5.3× slower) | Job 01 | **CONFIRMED** | 24 GB GDDR6, 300.05 GB/s (192-bit @ 12.5 Gbps). Bandwidth ratio: $1,597 / 300.05 = \mathbf{5.32\times}$. | [NVIDIA L4 Datasheet](https://www.nvidia.com/en-us/data-center/l4/) |
| **H10** | L4 FP8 compute is ~242 TFLOPS (16× below G4 FP4) | Job 01 | **PARTIALLY TRUE** | L4 is 242 TFLOPS sparse / 121 TFLOPS dense FP8. Cross-precision comparison. Dense FP8 ratio is **8.26×** ($1,000/121$). | [NVIDIA L4 Datasheet](https://www.nvidia.com/en-us/data-center/l4/) |
| **H11** | Colab is single-GPU; no multi-VM `tp=2` | Job 01 | **CONFIRMED** | Colab runtimes allocate exactly one GPU per VM; zero cross-VM interconnect for tensor parallel. | [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) |
| **H12** | Colab bills connected wall-clock, including idle | Job 01 | **CONFIRMED** | Metering runs continuously per second of connection; idle GPU burns full hourly rate. | [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) |
| **H13** | Colab AI Pro can actually assign a G4 | Job 01 | **FALSE** | Direct G4 selection is grayed out / tier-gated on AI Pro (200 CU); requires AI Ultra or Pro+. | [note.com AI Pro Test](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en), [Google Developers Blog](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/) |
| **H14** | G4 is Blackwell `sm_120` with native NVFP4 tensor cores | Job 01 | **CONFIRMED** | GB202 silicon targets `sm_120` (and `sm_120a` MMA targets); 5th-Gen Tensor Cores. | [CUDA C++ Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) |
| **H15** | Resolve G4 CU/hr: 6.4–6.7 vs architecture.md 8.71 | Job 01 | **INDEPENDENT CALC** | **8.71 CU/hr is standard metered rate**. 6.4–6.7 was early rollout. Plan must budget at 8.71. | [Job 01 Hardware & CU Audit](01_colab_g4_hardware_and_cu.md) |
| **H16** | 1,792 GB/s is Workstation, 1,597 GB/s is Server Edition | Job 01 | **CONFIRMED** | Colab G4 is Server Edition (passive heatsink, 1,597 GB/s). Workstation is active blower (1,792 GB/s). | [VRLA Tech Workstation vs Server](https://vrlatech.com/rtx-pro-6000-blackwell-workstation-vs-server-edition/) |
| **F1** | HF repo `nvidia/Qwen3.8-Flash-Next-NVFP4` exists | Job 02 | **CONFIRMED** | Public, ungated repo released 2026-08-31 by NVIDIA using ModelOpt v0.46.0. | [HF nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |
| **F2** | It is a 125B MoE model | Job 02 | **CONFIRMED** | Base LM has 124.91B parameters, plus 51.2B PLE n-gram table and ~4B MTP head (~180B total). | [HF Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **F3** | ~6.0B parameters activate per token | Job 02 | **CONFIRMED** | 10 routed experts + 1 shared expert + attention, embeddings, and norms activate ~5.79B–6.0B params. | [HF Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **F4** | 512 fine-grained experts, top-10 routing | Job 02 | **CONFIRMED** | `config.json` confirms `"num_experts": 512`, `"num_experts_per_tok": 10`, 1 shared expert. | [Flash-Next config.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json) |
| **F5** | Uses Gated DeltaNet | Job 02 | **CONFIRMED** | 36 of 48 layers are `"linear_attention"` (Gated DeltaNet linear recurrence, Mamba SSM dtype FP32). | [Flash-Next config.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json) |
| **F6** | Uses Qwen Sparse Attention (QSA) pruning at long context | Job 02 | **CONFIRMED** | 12 of 48 layers are full attention with QSA MQA indexer (`indexer_budget: 2048` tokens). | [Flash-Next config.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/config.json) |
| **F7** | Has a 4B MTP head | Job 02 | **PARTIALLY TRUE** | 1-layer MoE MTP head present (~2.76B physical params). Quantized in **FP8**, NOT NVFP4. | [Flash-Next hf_quant_config.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/hf_quant_config.json) |
| **F8** | 51B n-gram embedding prefetched from host memory | Job 02 | **PARTIALLY TRUE** | 51.2B param table exists (51.20 GB in FP8). Offloading to Colab's ~53–64 GB host RAM triggers **host OS OOM**. | [Job 02 Checkpoint Audit](02_qwen38_flash_next_checkpoint.md) |
| **F9** | NVFP4 weights + MTP occupy ~68–72 GB VRAM | Job 02 | **FALSE** | Base weights alone take **78.96 GB** on disk (~76.2 GB in VRAM) due to 292 unquantized BF16 modules. With MTP and overhead, footprint is **~83–85 GB** before KV cache. | [Job 02 Checkpoint Audit](02_qwen38_flash_next_checkpoint.md) |
| **F10** | Leaves ~24–26 GB on G4 for KV cache | Job 02 | **FALSE** | Leaves at best **~10–12 GB** free (if host offload works). If 51.2 GB PLE stays in VRAM, deficit is -34 GB. | [Job 02 Checkpoint Audit](02_qwen38_flash_next_checkpoint.md) |
| **F11** | Different architecture from dense 27B, not "faster 27B" | Job 02 | **CONFIRMED** | 27B is 64-layer dense hybrid; Flash-Next is 48-layer MoE (512 experts) + 51B PLE + 4B MTP. | [HF Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **F12** | Flash-Next is fastest model that fits on one Colab G4 | Job 02 | **FALSE** | Does not fit cleanly. Official NVIDIA docs mandate **`--tensor-parallel-size 8`** on Blackwell B200/B300. | [Flash-Next README.md](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/raw/main/README.md) |
| **F13** | File sizes, config fields, download match F2–F9 | Job 02 | **PARTIALLY TRUE** | Config fields match, but safetensors download is **132.68 GB**, not ~70 GB. | [model.safetensors.index.json](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/resolve/main/model.safetensors.index.json) |
| **F14** | License / gating / Colab-download practicality | Job 02 | **PARTIALLY TRUE** | Public ungated license, but staging 132.68 GB exhausts Colab local disk and takes 5–10 min. | [HF nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |
| **D1** | HF repo `nvidia/Qwen3.8-27B-NVFP4` exists | Job 03 | **CONFIRMED** | Released Sep 2026 by NVIDIA ModelOpt; 3 safetensors shards totaling 21.92 GB. | [HF nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |
| **D2** | Dense 27.8B; all parameters activate every token | Job 03 | **CONFIRMED** | Exactly 27,781,427,952 logical parameters (26.90B base LM, all active per token). Zero MoE. | [Qwen3.8-27B config.json](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/config.json) |
| **D3** | Hybrid 48 Gated DeltaNet + 16 full-attention layers | Job 03 | **CONFIRMED** | 64 hidden layers: 48 linear attention (GDN) + 16 full attention (GQA); interval = 4. | [Qwen3.8-27B config.json](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/config.json) |
| **D4** | NVFP4 occupies ~15–17 GB VRAM | Job 03 | **PARTIALLY TRUE** | Underestimated. Mixed precision (NVFP4 MLPs + FP8 attention + BF16 embeds) requires **20.15 GB** text weights (21.92 GB multimodal). | [Qwen3.8-27B hf_quant_config.json](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/hf_quant_config.json) |
| **D5** | FP8 occupies ~27 GB | Job 03 | **PARTIALLY TRUE** | Theoretical uniform 1 B/param is 27.78 GB; official checkpoint with BF16 embeds is **30.88 GB**. | [HF Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) |
| **D6** | FP16/BF16 occupies ~54 GB | Job 03 | **CONFIRMED** | Theoretical is 55.56 GB; official `Qwen/Qwen3.8-27B` BF16 checkpoint is **55.62 GB**. Within 3%. | [HF Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **D7** | Built-in MTP draft head exists on checkpoint | Job 03 | **CONFIRMED** | 1-layer MTP head present (`mtp_num_hidden_layers: 1`, 424.7M params, 849 MB). | [Qwen3.8-27B model.safetensors.index.json](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4/raw/main/model.safetensors.index.json) |
| **D8** | Leaves ~80 GB free on 96 GB G4 for KV | Job 03 | **PARTIALLY TRUE** | On 95.6 GB usable VRAM, subtracting 20.15 GB weights and 1.5 GB runtime leaves **~73.9 GB** free. | [Job 03 Checkpoint Audit](03_qwen38_27b_nvfp4_checkpoint.md) |
| **D9** | Difference from Blackfrost ABLITERATED GGUF | Job 03 | **CONFIRMED** | NVIDIA is stock/aligned (not abliterated), safetensors NVFP4/FP8, native MTP; Blackfrost is abliterated, GGUF Q4_K_M (16.81 GB), DFlash 2 sidecar. | [HF Blackfrost ABLITERATED GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) |
| **D10** | File sizes and config match D2–D7 | Job 03 | **PARTIALLY TRUE** | Config matches D2, D3, D7. File sizes reflect mixed precision (21.92 GB NVFP4, 30.88 GB FP8, 55.62 GB BF16). | [Job 03 Checkpoint Audit](03_qwen38_27b_nvfp4_checkpoint.md) |
| **D11** | Unquantized BF16 decode on G4 is ~50–60 tok/s | Job 03 | **FALSE** | **Physically impossible.** Streaming 55.56 GB over 1,597 GB/s caps decode at **28.7 tok/s** (100% util) and **24.4 tok/s** (85% util). | [Job 03 Independent Calculations](03_qwen38_27b_nvfp4_checkpoint.md#calc-3-roofline-decode-bandwidth-limits-and-refutation-of-d11) |
| **D12** | Download size and Colab-load time | Job 03 | **INDEPENDENT CALC** | 21.92 GB safetensors downloads in 44–88 s on GCP; loads in 2–5 s. Cold-start penalty ~1.5–2.0 min (~0.22 CU). | [Job 03 Independent Calculations](03_qwen38_27b_nvfp4_checkpoint.md#calc-4-download-size-and-colab-startup-economics-claim-d12) |
| **G1** | HF repo `nvidia/GLM-5.3-Flash-NVFP4` exists | Job 04 | **CONFIRMED** | Live official release; base model is `zai-org/GLM-5.3-Flash`, quantized via ModelOpt v0.47.0. | [HF nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) |
| **G2** | 320B MoE, 18B active, 288 experts | Job 04 | **CONFIRMED** | 321.32B total params, ~18B active/token, 288 routed experts (`top-8`) + 1 shared expert across 42 MoE layers. | [GLM-5.3-Flash config.json](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/config.json) |
| **G3** | NVFP4 weight footprint 160–175 GB (`320B × 0.5B`) | Job 04 | **PARTIALLY TRUE** | Underestimated. Safetensors file size is **204.46 GB**; 16.46B params in BF16 (+32.9 GB) and group-16 FP8 scales add ~19 GB. Loaded VRAM is ~195–205 GB. | [GLM-5.3-Flash metadata](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) |
| **G4** | Does not fit on 96 GB G4 (short by 70–80 GB) | Job 04 | **CONFIRMED** | Deficit is even larger: short by **~100–110 GB VRAM** for weights alone. | [GLM-5.3-Flash README](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/README.md) |
| **G5** | Colab cannot pool two G4s with `tp=2` | Job 04 | **CONFIRMED** | Colab provisions isolated single-GPU VMs; no cross-VM NVLink/RoCE clustering exists. | [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) |
| **G6** | CPU offload of ~80 GB drops speed to 1–3 tok/s | Job 04 | **CONFIRMED** | Roofline confirms **0.23–0.46 tok/s** for sequential layer offload; **1.0–3.0 tok/s** for dynamic MoE expert paging. | [Job 04 Findings §5](04_glm53_flash_nvfp4.md) |
| **G7** | Needs ≥ 2× 96 GB GPUs or B200/GB200 | Job 04 | **PARTIALLY TRUE** | 2× 96 GB (192 GB) is **insufficient** for a ~204 GB model. Official NVIDIA docs mandate **≥ 4× 96 GB (`tp=4`)** or GB200. | [GLM-5.3-Flash README](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4/raw/main/README.md) |
| **G8** | API price "$0.045 per complex agent task" | Job 04 | **PARTIALLY TRUE** | Informal heuristic. Official rates: Z.ai is **$0.15/M in / $0.50/M out**; OpenRouter is **$0.025/M in / $0.90/M out**. | [OpenRouter GLM-5.3-Flash](https://openrouter.ai/models/zai-org/glm-5.3-flash) |
| **S1** | 27B NVFP4 batch-1 AR decode ~85–105 tok/s | Job 05 | **PARTIALLY TRUE** | 71.4–90.5 tok/s realistic ceiling (85% bus util for 15.0–15.75 GB); 101.4 tok/s @ 100%. Measured: 89.8 tok/s raw AR. | [Unsloth Qwen 3.8 Docs](https://unsloth.ai/docs/models/qwen3.8) |
| **S2** | 27B NVFP4 with speculative / MTP=3 ~120–145 tok/s | Job 05 | **CONFIRMED** | 125–150 tok/s with native MTP-3; measured up to **187.13 tok/s** with DFlash 2 ($n=5$) on RTX PRO 6000. | [lukaLLM Benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1vvncyh/i_benchmark_dflash_2_pr_build_in_llamacpp_on_qwen/) |
| **S3** | Ceiling ~100–110 tok/s from 15–19 GB on 1,597 GB/s | Job 05 | **PARTIALLY TRUE** | Holds strictly at 100% bus utilization for 15.0 GB ($1,597/15 = 106.5$). At 19.0 GB, ceiling is 84.1 tok/s (100%) and 71.4 tok/s (85%). | [NVIDIA RTX PRO 6000 Specs](https://www.techpowerup.com/gpu-specs/rtx-pro-6000-blackwell.c4240) |
| **S4** | 27B NVFP4 prefill ~5,500–7,500 tok/s | Job 05 | **CONFIRMED** | Measured: 5,800–7,200 tok/s at context $<4\text{K}$ on Blackwell `sm_120` in vLLM/FlashInfer (~10–12% MFU). | [vLLM Qwen3.8 Recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) |
| **S5** | TTFT 1K–2K < 0.10 s; 20K–25K ~3.0–3.5 s | Job 05 | **CONFIRMED** | 1K–2K takes 0.13–0.25 s; 20K–25K takes 3.08–3.85 s at 6,500 tok/s sustained prefill. | [RunPod Qwen3.8-27B Study](https://www.runpod.io/blog/qwen3-8-27b-on-runpod-the-theory-behind-frontier-class-agentic-coding-that-fits-on-a-single-24gb-worker) |
| **S6** | Concurrency 4 ~340–360; Concurrency 16 ~850–950 | Job 05 | **CONFIRMED** | Measured batch scaling: ~350–380 tok/s at C=4; ~920–1,100 tok/s at C=16; saturates at 3,200+ at C=64. | [RunPod RTX PRO 6000 Study](https://www.runpod.io/blog/qwen3-8-27b-on-runpod-the-theory-behind-frontier-class-agentic-coding-that-fits-on-a-single-24gb-worker) |
| **S7** | FP8 decode is ~5–10% faster than NVFP4 on RTX PRO 6000 | Job 05 | **FALSE** | **KILLED.** Physically impossible. FP8 moves 27 GB/step (capping at 50.3 tok/s @ 85%), NVFP4 moves 15.75 GB/step (capping at 86.2 tok/s @ 85%). NVFP4 is **~1.6×–1.7× faster**. | [ZeroEntropy NVFP4 vs FP8 Study](https://zeroentropy.dev/articles/fp4-vs-fp8-blackwell-benchmarks) |
| **S8** | 27B FP8 prefill <4K ~4,500–4,800; >16K ~1,500–2,200 | Job 05 | **CONFIRMED** | Measured: 4,200–4,800 tok/s at $<4\text{K}$; drops to 1,500–2,100 tok/s at 32K context due to attention quadratic cost. | [SGLang Qwen3.8 Cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B) |
| **S9** | TTFT ~2K tokens 0.2–0.4 s (FP8) | Job 05 | **CONFIRMED** | $2,000 / 4,600 \text{ tok/s} \approx 0.43\text{ s}$; measured range is 0.35–0.48 s. | [SGLang Qwen3.8 Cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B) |
| **S10** | 27B FP8 Thinking AR ~95–105; MTP ~110–125 tok/s | Job 05 | **FALSE** | Streaming 27 GB over 1,597 GB/s caps AR decode at 50.28 tok/s (85% util). With MTP-3, FP8 reaches 70–85 tok/s, not 110–125. | [hanxiao L4/Blackwell Scaling Model](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) |
| **S11** | 27B FP8 Output AR ~95–105; MTP ~135–160+ tok/s | Job 05 | **FALSE** | Raw AR caps at 50.28 tok/s. With code MTP ($\bar{\alpha} \approx 1.8$), FP8 reaches ~90–105 tok/s, not 135–160+. | [hanxiao L4/Blackwell Scaling Model](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) |
| **S12** | Thinking MTP acceptance 50–60%; code 75–85%+ | Job 05 | **PARTIALLY TRUE** | Holds strictly for single-token drafting ($p_1$). For depth $k=3$, cumulative acceptance decays geometrically; mean accepted length $\bar{\alpha}$ is 1.6–2.1 for code, 1.2–1.4 for thinking. | [Inco AI DFlash 2 Blog](https://inco.ai/blog/dflash2/) |
| **S13** | FP16/BF16 decode ~50–60 tok/s | Job 05 | **FALSE** | Streaming 54–55.6 GB over 1,597 GB/s caps decode at **25.14 tok/s** (85% util) and **29.57 tok/s** (100% util). | [Unsloth Discussion #66](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/66) |
| **S14** | 27B prefill <4K ~6,000–7,500; >16K ~2,500–3,500 | Job 05 | **CONFIRMED** | Matches measured prefill curves on G4 `sm_120`. Context decay driven by 16 full-attention layers. | [vLLM Qwen3.8 Recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) |
| **S15** | Flash-Next prefill <4K ~7,500–9,500; >16K ~4,000–5,500 | Job 05 | **PARTIALLY TRUE** | Measured: 7,200–8,800 tok/s at $<4\text{K}$; at $>16\text{K}$, QSA dynamic pruning preserves ~4,100–5,100 tok/s. | [Reddit 256K Benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1vvncyh/) |
| **S16** | 27B thinking ~120–135 (or ~95–100 without MTP) | Job 05 | **PARTIALLY TRUE** | AR without MTP caps at 86.2 tok/s (85% util). With MTP ($\bar{\alpha} \approx 1.35$), decode reaches 115–125 tok/s. 135 tok/s requires DFlash 2 drafter. | [lukaLLM DFlash 2 Benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1vvncyh/i_benchmark_dflash_2_pr_build_in_llamacpp_on_qwen/) |
| **S17** | Flash-Next thinking ~220–260 tok/s | Job 05 | **UNVERIFIED** | Theoretical ceiling assuming 3.5 GB stream. Measured single-stream Flash-Next decode on single GPU is **44–74.4 tok/s raw**, reaching ~125 tok/s with MTP. | [HF nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |
| **S18** | 27B code output ~150–175 tok/s | Job 05 | **CONFIRMED** | Attainable with native MTP-3 on syntax ($\bar{\alpha} \approx 1.8–2.0$); measured **187.13 tok/s** with DFlash 2 ($n=5$). | [lukaLLM DFlash 2 Benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1vvncyh/i_benchmark_dflash_2_pr_build_in_llamacpp_on_qwen/) |
| **S19** | Flash-Next code output ~320–380+ tok/s | Job 05 | **UNVERIFIED** | Theoretical ceiling assuming 3.5 GB streaming and $\bar{\alpha} \ge 3.0$. Single-stream benchmarks cap at ~125–150 tok/s; 320+ requires batching ($C \ge 4$). | [HF Flash-Next Discussion](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/discussions/3) |
| **S20** | Flash-Next streams ~3.5–4 GB active weights vs ~15 GB for 27B | Job 05 | **PARTIALLY TRUE** | 6.0B active expert params @ 0.5B = 3.0 GB. Shared attention, embeddings (248K vocab), GDN recurrent state slots, and router matrices inflate step traffic to **~5.8–6.1 GB**. | [Job 02 Independent Calculations](02_qwen38_flash_next_checkpoint.md#calculation-4-active-weight-streaming-per-token-decode-memory-traffic-gap) |
| **S21** | QSA keeps Flash-Next long-context prefill nearly 2× dense | Job 05 | **CONFIRMED** | QSA prunes non-salient context blocks, reducing quadratic attention overhead from $O(N^2)$ to $O(N \cdot k)$ on full-attention layers. | [Qwen 3.8 Architecture Whitepaper](https://qwen.ai/blog/qwen3.8) |
| **S22** | Flash-Next is ~2× faster reasoning and ~2.5× faster code | Job 05 | **FALSE** | In single-stream batch 1, speedup is ~1.2×–1.5× due to MoE memory/dispatch overhead. 2×–2.5× speedup is realized only under concurrency ($C \ge 8$). | [NVIDIA ModelOpt Benchmarks](https://github.com/NVIDIA/TensorRT-Model-Optimizer) |
| **S23** | Prefix-cache hit bind rate ~100k–200k tok/s | Job 05 | **PARTIALLY TRUE** | **Bookkeeping artifact.** Cached KV pages remain pinned in VRAM. Figure represents host-CPU scheduling rate, not GPU physical transfer. | [SGLang RadixAttention Paper](https://arxiv.org/abs/2312.07104) |
| **S24** | Decode dominates: 1M output @ 300–350 tok/s is ~47.6–55.5 min | Job 05 | **INDEPENDENT CALC** | Arithmetic confirmed: $1,000,000 / 350 = 2,857\text{ s}$ (47.6 min); $1,000,000 / 300 = 3,333\text{ s}$ (55.6 min). Decode is $>80\%$ of active session time. | [Job 05 Findings §5](05_speed_roofline_and_mtp.md) |
| **S25** | 2M cache-miss prefill @ 7,500–9,500 tok/s is ~3.5–4.4 min | Job 05 | **INDEPENDENT CALC** | Arithmetic confirmed: $2,000,000 / 9,500 = 210.5\text{ s}$ (3.51 min); $2,000,000 / 7,500 = 266.7\text{ s}$ (4.44 min). | [Job 05 Findings §5](05_speed_roofline_and_mtp.md) |
| **S26** | 37M cache-hit bind @ 100k–200k tok/s is ~3.1–6.2 min | Job 05 | **FALSE** | Fabricated wall-clock delay. Radix lookup on host CPU takes 1–5 ms/turn; across 1,000 turns, total wall-clock is **~3–5 seconds ($<0.1\text{ min}$)**, not 3–6 minutes. | [SGLang RadixAttention Paper](https://arxiv.org/abs/2312.07104) |
| **S27** | Independent roofline: `tok/s ≈ BW / bytes_per_tok` batch-1 | Job 05 | **INDEPENDENT CALC** | Confirms 27B NVFP4 caps at 71–90 tok/s; 27B FP8 caps at 41–50 tok/s; Flash-Next caps at 224–271 tok/s (5.8 GB payload). | [Job 05 Independent Calculations](05_speed_roofline_and_mtp.md#2-independent-roofline-model-for-colab-g4--rtx-pro-6000-blackwell) |
| **S28** | Published measured benchmarks on RTX PRO 6000 / Colab G4 | Job 05 | **CONFIRMED** | lukaLLM measured on RTX PRO 6000: raw AR 67.97 tok/s; DFlash 2 ($n=5$) **187.13 tok/s**; DFlash 2 + `k4v` **304.92–343.52 tok/s**. Unsloth measured NVFP4 AR 89.8 tok/s. | [lukaLLM Benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1vvncyh/i_benchmark_dflash_2_pr_build_in_llamacpp_on_qwen/) |
| **R1** | Optimized stack is vLLM / TRT-LLM with FlashInfer | Job 06 | **PARTIALLY TRUE** | Stack is vLLM / SGLang with FlashInfer CUTLASS kernels; TRT-LLM is unviable on Colab; monolithic CUDA graphs fail on Flash-Next. | [vLLM qwen4_exp Docs](https://docs.vllm.ai/en/latest/models/supported_models.html) |
| **R2** | Speed table C assumes vLLM / SGLang | Job 06 | **CONFIRMED** | Table C explicitly specifies vLLM / SGLang with batch size 1, FlashInfer, and RadixAttention. | [Running Qwen flash next on Colab.md](../flash-next-chat.md#L81-L90) |
| **R3** | MTP / speculative decoding available in those engines | Job 06 | **PARTIALLY TRUE** | Both models embed MTP heads; supported in vLLM/SGLang, but encounters graph-break penalties on GDN recurrent states. | [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) |
| **R4** | `--kv-cache-dtype fp8` supported and halves KV | Job 06 | **CONFIRMED** | Halves full-attention KV cache by 50% (from 2 bytes to 1 byte per element) across vLLM and SGLang. | [vLLM KV Cache Config](https://docs.vllm.ai/en/latest/serving/engine_args.html) |
| **R5** | CUDA graph capture works for decode (`--enforce-eager` off) | Job 06 | **FALSE** | Monolithic CUDA graph capture fails on Flash-Next due to dynamic QSA sparsity masks, 512-expert routing, and MTP branching. Requires piecewise breakable graphs. | [vLLM Config vllm.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py) |
| **R6** | Flash-Next actually runs in current vLLM/SGLang | Job 06 | **PARTIALLY TRUE** | Supported upstream as `qwen4_exp`, but single-GPU Colab G4 serving is hindered by host RAM limits for the 51B PLE table (~51 GB) and memory profiler bugs (Issue #60489). | [vLLM Issue #53908](https://github.com/vllm-project/vllm/issues/53908) |
| **R7** | 27B NVFP4 runs in vLLM/SGLang on Blackwell | Job 06 | **CONFIRMED** | Runs natively in vLLM v0.7.0+ and SGLang v0.5.19+ on Blackwell using FlashInfer CUTLASS NVFP4 kernels; fits comfortably in ~21.9 GB VRAM. | [HF nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |
| **R8** | llama.cpp path for Flash-Next; DFlash 2 applicability | Job 06 | **FALSE** | DFlash 2 CANNOT transfer (trained on 27B states). CPU-mmap 51B PLE table in `llama.cpp` collapses decode speed to **1–5 tok/s**. | [Job 06 Findings §4](06_runtime_stack.md#4-llamacpp-reality-ggufnvfp4-support-and-dflash-2-non-transferability) |
| **R9** | Colab install/compile time and CU waste vs Drive-cached `llama-server` | Job 06 | **INDEPENDENT CALC** | Cold boot vLLM takes 13–24 min (1.41–2.60 CU @ 6.5 CU/hr); Drive-cached static `llama-server` boots in <1.5 min (0.16 CU). Switching wastes ~1.25–2.44 CU on every boot. | [Job 06 Independent Calculations](06_runtime_stack.md#1-colab-session-boot-latency-and-cu-cost-model) |
| **R10** | TensorRT-LLM on Colab: realistic or not | Job 06 | **FALSE** | Completely unrealistic. `trtllm-build` takes 30–50+ minutes for Flash-Next, producing a non-portable 70+ GB engine plan that burns 3.3–5.4 CU before running. | [TensorRT-LLM Build Guide](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html) |
| **R11** | Required engine versions, CUDA, extra wheels for NVFP4 on `sm_120` | Job 06 | **CONFIRMED** | Requires CUDA 13.0, FlashInfer >= v0.2.3 (`FLASHINFER_CUDA_ARCH_LIST="12.0f"`), ModelOpt >= 0.23.0, vLLM >= v0.7.3. `sm_120` lacks TMEM and requires CUTLASS kernels. | [FlashInfer sm_120 Docs](https://docs.flashinfer.ai/installation.html) |
| **R12** | Prefix caching / RadixAttention works for hybrid GDN | Job 06 | **PARTIALLY TRUE** | Works, but requires chunk-boundary alignment (`--mamba-cache-mode align` in vLLM; Unified Radix Cache in SGLang) because GDN recurrent states cannot be sliced at arbitrary tokens. | [LMCache Hybrid Caching](https://lmcache.ai/blog/hybrid-architecture-caching) |
| **K1** | 27B NVFP4 weights ~16 GB, ~80 GB left for KV; context passes 100K | Job 07 | **CONFIRMED** | Base weights + 65th MTP head occupy ~16.8 GB. Free VRAM is ~76.3 GB. At FP8 KV (32 KiB/token), 100K tokens consumes 3.125 GiB. Max context >2M tokens. | [Job 07 KV Cache Audit](07_kv_cache_and_context.md) |
| **K2** | Flash-Next weights ~70 GB, ~24–26 GB left for KV | Job 07 | **FALSE** *(Overridden by Job 02)* | Job 07 confirmed the chat's theoretical math, but Job 02's primary safetensors audit proved real weights are **78.96 GB base** (due to 292 BF16 modules) + 2.76 GB MTP = **~83.5 GB loaded VRAM**. True free VRAM is **~10–12 GB** (if host offload works), not 24–26 GB. | [Job 02 Checkpoint Audit](02_qwen38_flash_next_checkpoint.md#calculation-3-reconciling-the-chats-6872-gb-loaded-vram-claim) |
| **K3** | 24 GB FP8 KV is "plenty for 32K–48K tokens" | Job 07 | **PARTIALLY TRUE** | 24 GB holds 32K–48K, but claiming this as the headroom is a ~50× underestimate of KV capacity. Flash-Next uses only 12 full-attn layers with 2 KV heads (12 KiB/tok at FP8). 24 GB holds over **2.05M tokens**; 32K tokens consumes only 384 MiB. | [Job 07 KV Cache Audit](07_kv_cache_and_context.md) |
| **K4** | Multi-file codebases >64K–128K favor 27B on G4 | Job 07 | **FALSE** | Flash-Next does not OOM at 64K, 128K, or 256K context. At 256K native limit, its FP8 KV cache consumes a mere **3.07 GB**. Memory capacity does not favor 27B over Flash-Next. | [Job 07 KV Cache Audit](07_kv_cache_and_context.md) |
| **K5** | Sliding-window / compaction required for long Flash-Next loop | Job 07 | **FALSE** | Compaction is mathematically unnecessary for VRAM preservation on G4. 50K context consumes only 600 MiB of KV cache. | [Job 07 KV Cache Audit](07_kv_cache_and_context.md) |
| **K6** | Per-token KV bytes for each model at FP16 / FP8 / INT8 | Job 07 | **INDEPENDENT CALC** | **27B** (16 layers, 4 heads, dim 256): FP16 = 64 KiB/tok; FP8 = 32 KiB/tok; INT4 = 16 KiB/tok.<br>**Flash-Next** (12 layers, 2 heads, dim 256): FP16 = 24 KiB/tok; FP8 = 12 KiB/tok; INT4 = 6 KiB/tok. | [Job 07 Findings §3](07_kv_cache_and_context.md#3-kv-cache-footprint-per-token-k6) |
| **K7** | GDN recurrent-state memory per slot, speculation multiplier | Job 07 | **INDEPENDENT CALC** | **27B**: 78.4 MB/slot (FP16). MTP=3 requires 4 slots = 313.6 MB.<br>**Flash-Next**: 58.8 MB/slot (FP16). MTP=3 requires 4 slots = 235.2 MB. | [Job 07 Findings §4](07_kv_cache_and_context.md#4-gated-deltanet-gdn-recurrent-state-memory-k7) |
| **K8** | MoE router / expert-cache / n-gram-embed host-prefetch cost | Job 07 | **CONFIRMED** | Routing overhead negligible (<10 MB); MoE CUTLASS scratch is ~1.2 GB. Sparse token lookup transfers only ~20.48 KB/token over PCIe (7.17 MB/s @ 350 tok/s decode, <0.012% PCIe 5.0 bandwidth). | [Job 07 Findings §5](07_kv_cache_and_context.md#5-the-512b-n-gram--engram-embedding-host-ram--pcie-analysis-k8-k10) |
| **K9** | `--fit` / max context on 96 GB after weights + CUDA context | Job 07 | **INDEPENDENT CALC** | Both models fit their full 262K native context window with tens of gigabytes of free VRAM headroom. | [Job 07 Independent Calculations](07_kv_cache_and_context.md#2-context-headroom--token-capacity-analysis-k1-k2-k3-k4-k9) |
| **K10** | 51B n-gram embeddings in host RAM and Colab host RAM reality | Job 07 | **CONFIRMED** | 51.2B table offloaded to host RAM via `--engram-config '{"cpu_offload": true}'`. Requires 51.2 GB (FP8) or 102.4 GB (BF16) host RAM. Completely crashes standard Colab runtimes (51–53 GB host RAM). | [Job 07 Findings §5](07_kv_cache_and_context.md#5-the-512b-n-gram--engram-embedding-host-ram--pcie-analysis-k8-k10) |
| **E1** | Same GPU, faster model → fewer CUs for automated job ("race to sleep") | Job 08 | **CONFIRMED** | True strictly for automated headless runs with immediate `runtime.unassign()`; false for interactive sessions where human idle dominates. | [Google Colab FAQ](https://colab.research.google.com/faq) |
| **E2** | 60,000 tokens: 27B ~140 tok/s → 7.1 min → 0.77 CU; Flash-Next ~300 tok/s → 3.3 min → 0.36 CU | Job 08 | **PARTIALLY TRUE** | Arithmetic matches at 6.55 CU/hr, but omits prefill. At standard 8.71 CU/hr: 27B burns **1.04 CU**; Flash-Next burns **0.48 CU**. | [architecture.md](../27b-plan/architecture.md#L68) |
| **E3** | Interactive 1-hour sitting burns ~6.5 CU on G4 regardless of model | Job 08 | **CONFIRMED** | Connected wall-clock billing meters idle time at full rate: burns 6.55 CU/hr (chat rate) or 8.71 CU/hr (standard rate). | [Google Colab FAQ](https://colab.research.google.com/faq) |
| **E4** | 25-turn task: 27B 14–18 s/turn, 6.5–7.5 min, 0.75–0.81 CU; Flash-Next 4–6 s/turn, 1.8–2.5 min, 0.20–0.27 CU | Job 08 | **PARTIALLY TRUE** | Chat arithmetic holds at 6.55 CU/hr assuming uncached prefill. At 8.71 CU/hr: 27B burns **0.94–1.09 CU**; Flash-Next burns **0.26–0.36 CU**. | [architecture.md](../27b-plan/architecture.md#L68) |
| **E5** | Tasks per 100 CU: ~125 (27B) vs ~400+ (Flash-Next) | Job 08 | **PARTIALLY TRUE** | At standard 8.71 CU/hr: 92–106 tasks (27B) vs 275–385 tasks (Flash-Next). Under realistic 150 tok/s decode: ~180 tasks for Flash-Next. | [Colab Pay-As-You-Go](https://colab.research.google.com/signup) |
| **E6** | Prefill of growing 20K+ history is silent killer by turn 15 | Job 08 | **PARTIALLY TRUE** | True only when prompt history is uncached. With prefix caching (RadixAttention/vLLM), prefill is restricted to delta tokens (<1k tok/turn, <0.15 s). | [vLLM Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html) |
| **E7** | G4 vs L4 20-turn loop: G4 3.7 s/turn, 1.23 min, 0.133 CU; L4 14B 29.6 s/turn, 9.87 min, 0.345 CU | Job 08 | **PARTIALLY TRUE** | Arithmetic matches at G4=6.55 and L4=2.1. At repo rates (G4=8.71, L4=1.71): G4 burns **0.179 CU**; L4 burns **0.282 CU**. Apples-to-oranges architectural comparison. | [architecture.md](../27b-plan/architecture.md#L65-L71) |
| **E8** | L4 14B prefill 15K ~1,200 tok/s; decode ~35 tok/s | Job 08 | **CONFIRMED** | Matches realistic roofline for 14B dense FP8/INT4 streaming across L4's 300 GB/s bandwidth. | [NVIDIA L4 Datasheet](https://www.nvidia.com/en-us/data-center/l4/) |
| **E9** | G4 Flash-Next prefill 15K ~8,000 tok/s; decode ~320 tok/s | Job 08 | **PARTIALLY TRUE** | Assumes peak MTP draft acceptance (>75%). If MTP is unoptimized or code entropy is high, decode drops to 150–220 tok/s. | [Job 05 Findings §4](05_speed_roofline_and_mtp.md) |
| **E10** | Smaller models incur an "agent failure tax" (more turns) | Job 08 | **CONFIRMED** | Well-documented in benchmark literature; 7B/14B models average 2.5–3.5× more turns on SWE-bench due to syntax errors and hallucinated parameters. | [SWE-bench](https://www.swebench.com/) |
| **E11** | Workload 37M hit + 2M miss + 1M out burns 5.4–7.5 CU (avg ~6.8), 50–65 min | Job 08 | **INDEPENDENT CALC** | Chat reproduces at 6.55 CU/hr, but relies on artificial 100k–200k tok/s "cache bind". With instant prefix skips, wall time is 56.0 min (6.11 CU @ 6.55 CU/hr; **8.13 CU @ 8.71 CU/hr**). | [Job 08 Findings §5](08_colab_cu_economics.md) |
| **E12** | If 37M were all misses, extra ~72+ min and ~8 CU | Job 08 | **PARTIALLY TRUE** | 37M / 8,500 tok/s = 72.55 min. At 6.55 CU/hr this is 7.92 CU; at standard 8.71 CU/hr this is **10.53 CU**. Net delta is +67.9 min (+9.86 CU). | [architecture.md](../27b-plan/architecture.md#L68) |
| **E13** | Equivalent API cost $15–$35+ vs Colab < $1 for ~7 CU | Job 08 | **PARTIALLY TRUE** | $15–$35 reflects Tier-1 frontier APIs (Claude 3.5 Sonnet / GPT-4o). High-efficiency APIs (Gemini 2.0 Flash / Haiku) cost **$1.50–$4.60**. 7 CUs costs $0.70 at Colab retail overage. | [Job 08 Findings §7](08_colab_cu_economics.md) |
| **E14** | Decode is >80% of billed time in that 40M-token mix | Job 08 | **CONFIRMED** | 1M output tokens takes 47.6–55.5 min out of 54–66 min (83.8%–88.1%). With instant prefix skips, decode is **93.1%** of active runtime. | [Job 08 Findings §5](08_colab_cu_economics.md) |
| **E15** | 200 CU/month → ~30.5 hours at 6.4–6.7 CU/hr | Job 08 | **CONFIRMED** | $200 / 6.55 = \mathbf{30.53\text{ hours}}$. At standard 8.71 CU/hr ([architecture.md](../27b-plan/architecture.md#L68)), 200 CUs yields only **22.96 hours**. | [Google One AI Premium](https://one.google.com/about/ai-premium) |
| **E16** | 200 CU yields 1.18B / 1.42B / 1.69B / 1.94B tokens at 320/400/500/600 tok/s | Job 08 | **INDEPENDENT CALC** | Chat arithmetic verified at 6.55 CU/hr with 10.1 min prompt overhead. At standard 8.71 CU/hr, monthly tokens drop to **0.89B / 1.06B / 1.27B / 1.45B**. | [Job 08 Findings §6](08_colab_cu_economics.md) |
| **E17** | Recalculate E2, E4, E7, E11, E15, E16 from first principles | Job 08 | **INDEPENDENT CALC** | Complete recalculation demonstrated across all rates, prompt latencies, and decode tiers. | [Job 08 Independent Calculations](08_colab_cu_economics.md#independent-calculations) |
| **E18** | Google AI Pro 200 CU/month pool exists and G4 is billable | Job 08 | **PARTIALLY TRUE** | Google AI Pro ($19.99/mo) bundles 200 CUs. However, G4 is gated/grayed out and cannot be billed on AI Pro. | [Google One AI Premium FAQ](https://support.google.com/googleone/answer/16287313) |
| **Q1** | Flash-Next is "a lot better" than 27B | Job 09 | **PARTIALLY TRUE** | On SWE-bench Pro, Flash-Next is **62.5 vs 61.7 (+0.8)**; on LiveCodeBench v6, it is **91.9 vs 90.3 (+1.6)**. Marginal on coding; substantially better on general agentic benchmarks (DeepSWE 1.1: 58.7 vs 42.2). | [HF Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **Q2** | Flash-Next is best model at this Colab price point | Job 09 | **PARTIALLY TRUE** | Largest parameter model, but unrunnable on standard Colab runtimes due to host RAM crashes and G4 tier lockout. For reliable serving, 27B is strictly superior. | [Job 09 Quality Audit](09_quality_abliteration_agent_fit.md) |
| **Q3** | 125B MoE fixes compiler traces in 1–2 turns; 7B/14B hallucinate | Job 09 | **PARTIALLY TRUE** | True vs 7B/14B, but false as a justification against 27B. Dense 27B activates 27.8B parameters every step (4.6× Flash-Next's 6B active) and scores 73.0 on Terminal-Bench; it does not hallucinate flags. | [HF Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Q4** | Abliterated / uncensored Flash-Next checkpoint exists | Job 09 | **PARTIALLY TRUE** | Community quants exist, but official `nvidia/Qwen3.8-Flash-Next-NVFP4` is stock/aligned. None support `llama.cpp` + DFlash 2. | [Job 09 Quality Audit](09_quality_abliteration_agent_fit.md) |
| **Q5** | Tool calling / Hermes / OpenAI-compat on Flash-Next NVFP4 | Job 09 | **CONFIRMED** | Supported in vLLM via `--tool-call-parser qwen3_coder` using XML `<tool_call>` syntax. Requires vLLM/SGLang; cannot run in `llama.cpp`. | [vLLM Parser Docs](https://docs.vllm.ai/en/latest/models/extensions.html) |
| **Q6** | Thinking behavior vs `--reasoning off` on 27B worker | Job 09 | **CONFIRMED** | Flash-Next generates 1,500–3,500 `<think>` tokens per turn. 27B with `--reasoning off` emits 500 pure code tokens in 3.3 s, burning **60% fewer CUs**. | [Job 09 Findings §5](09_quality_abliteration_agent_fit.md#5-thinking-tokens-vs---reasoning-off-the-cu-burn-paradox) |
| **Q7** | Context window spec vs 27B | Job 09 | **CONFIRMED** | Both models claim 262K native context. 27B leaves ~74 GB VRAM free on G4; Flash-Next leaves only ~10–12 GB free after true weights and runtime buffers. | [Job 09 Findings §7](09_quality_abliteration_agent_fit.md) |
| **Q8** | Refusal / safety of NVIDIA checkpoint vs Blackfrost | Job 09 | **CONFIRMED** | Official NVIDIA checkpoint enforces strict safety alignments calibrated on Nemotron. Re-introduces refusals on cybersecurity and exploit code. | [Flash-Next README.md](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |
| **Q9** | Switching worker to Flash-Next on G4 is justified vs A100 27B | Job 09 | **FALSE** | **Unjustified.** Cold boots jump from <45 s to 15–20 min, refusals return, and thinking tokens inflate CU burn. | [Job 09 Quality Audit](09_quality_abliteration_agent_fit.md) |
| **Q10** | Known issues: routing instability, QSA drop, NVFP4 accuracy | Job 09 | **CONFIRMED** | NVFP4 accuracy drop is minor (<1.5%), but 512-expert routing jitter and QSA context truncation introduce runtime instability not found in dense 27B. | [Flash-Next Model Card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) |

---

## 3. Internal Contradictions in the Gemini Chat

A forensic cross-examination of `Running Qwen flash next on Colab.md` reveals severe internal mathematical, physical, and architectural contradictions where the chat model repeatedly contradicted its own claims across successive prompt turns:

### A. The Three Incompatible 27B Speed Tables
Across turns 1, 2, and 3, the chat presented three mutually contradictory performance tables for Qwen 3.8 27B:

```
Table A ("Optimized Stack", turn 1):
  - 27B NVFP4 standard AR decode:  ~85 – 105 tok/s (streaming ~15–19 GB payload over 1,597 GB/s)
  - 27B NVFP4 with MTP=3:          ~120 – 145 tok/s
  - Prefill throughput:            ~5,500 – 7,500 tok/s
  - Assertion:                     "FP8 is actually ~5–10% faster on decode than NVFP4 on the RTX PRO 6000"

Table B ("Native FP8 Precision", turn 2):
  - 27B FP8 standard AR decode:    ~95 – 105 tok/s (streaming ~27 GB payload over 1,597 GB/s)
  - 27B FP8 with MTP:              ~110 – 125 tok/s (thinking) / ~135 – 160+ tok/s (code)
  - Prefill throughput:            ~4,500 – 4,800 tok/s (<4K) / ~1,500 – 2,200 tok/s (>16K)
  - Assertion:                     "Streaming 27 GB across 1,597 GB/s gives a hardware ceiling of ~100 tok/s"

Table C ("NVFP4 vs Flash-Next", turn 3):
  - 27B NVFP4 thinking decode:     ~120 – 135 tok/s (or ~95–100 tok/s without MTP)
  - 27B NVFP4 code decode:         ~150 – 175 tok/s
  - Prefill throughput:            ~6,000 – 7,500 tok/s (<4K) / ~2,500 – 3,500 tok/s (>16K)
```

#### Why these tables represent impossible physics:
1. **The 27 GB FP8 100 tok/s Hallucination:** In Table B, the chat asserts that streaming a 27 GB payload across a 1,597 GB/s memory bus yields a hardware ceiling of "~100 tokens/sec."
   $$\text{Physical Ceiling} = \frac{1,597\text{ GB/s}}{27.0\text{ GB}} = \mathbf{59.15\text{ tok/s}}\quad\text{(at 100% bus utilization)}$$
   $$\text{Realistic Ceiling (85% bus efficiency)} = 59.15 \times 0.85 = \mathbf{50.28\text{ tok/s}}$$
   Achieving 100 tok/s while reading 27 GB per step requires $27\text{ GB} \times 100\text{ tok/s} = \mathbf{2,700\text{ GB/s}}$ of memory bandwidth—exceeding the physical capacity of the G4 by **69%**.
2. **The Inverted Precision Hierarchy:** In Table A, the chat claims that "FP8 is ~5–10% faster on decode than NVFP4 on the RTX PRO 6000." Because batch-1 autoregressive decode is strictly memory-bandwidth bound, moving 27 GB (FP8) cannot be faster than moving 15.75 GB (NVFP4). On Blackwell `sm_120`, which features native hardware NVFP4 tensor cores, NVFP4 is **1.6× to 1.7× faster** on decode than FP8 (86.2 tok/s vs 50.3 tok/s @ 85% bus efficiency).
3. **The BF16 Cascade Error:** In Table B, the chat states that unquantized BF16 decode is ~50–60 tok/s by halving the FP8 figure. Because the FP8 baseline was already mathematically impossible, halving it produced a number (50–60 tok/s) that requires 2,700–3,240 GB/s for a 54 GB payload. Physical BF16 decode on G4 caps at **25.14 tok/s** at 85% bus efficiency.

### B. Colab G4 Burn Rate: 6.4–6.7 CU/hr vs. `architecture.md` 8.71 CU/hr
* Throughout turns 4, 5, 7, and 8, the chat consistently models G4 consumption at **~6.4–6.7 CU/hr** (using ~6.5 or 6.55 in its calculations).
* In contrast, the locked workspace architecture ([`architecture.md`](../27b-plan/architecture.md#L68)) records G4 at **~8.71 CU/hr**.
* **Resolution:** Independent empirical billing audits (Job 01 H6/H15 and Job 08 E15) confirm that **8.71 CU/hr is the standard metered rate** on Google Colab for dedicated G4 provisioning. The 6.40–6.74 CU/hr figure was an early rollout promotional rate or an off-peak capacity fallback.
* **Impact:** Modeling economics at 6.55 CU/hr underestimates the operator's actual Colab credit spend by **25.4%** ($8.71 / 6.55 = 1.33\times$).

### C. Memory Bandwidth: 1,597 GB/s vs. 1,792 GB/s
* The chat states G4 bandwidth is **1,597 GB/s**. Prior literature and early community benchmarks cited **1,792 GB/s**.
* **Resolution:** Job 01 (§2 and H16) confirmed both numbers exist in official NVIDIA datasheets, representing two distinct hardware SKUs:
  1. **RTX PRO 6000 Blackwell Workstation Edition:** 600W TDP, active dual-fan cooling, 28 Gbps GDDR7 on a 512-bit bus $= \mathbf{1,792\text{ GB/s}}$.
  2. **RTX PRO 6000 Blackwell Server Edition (Colab G4 SKU):** Passively cooled datacenter chassis SKU, clocked down to 24.95 Gbps $= \mathbf{1,597\text{ GB/s}}$.
* The chat correctly named 1,597 GB/s, but failed to apply it to its own decode ceiling arithmetic.

### D. Prefix-Cache Hit: 100k–200k tok/s and 3.1–6.2 Minutes "Bind Time"
* In turn 7, the chat modeled a 40M token workload with 37M cache hits, claiming:
  > *"Cache Hit (KV Lookup & Binding): 37,000,000 tokens @ ~100,000 – 200,000 tok/s = ~3.1 – 6.2 minutes (~280s)."*
* **The Accounting Fallacy:** Job 05 (§5) and Job 08 (§5) proved that in modern serving engines (vLLM Automatic Prefix Caching, SGLang RadixAttention), cached KV blocks are **already resident in GPU VRAM**.
* A prefix hit performs zero GPU FLOPs, zero weight reads, and zero memory transfers across the GDDR7 or PCIe bus. It is a CPU radix-tree lookup that takes **1 to 5 milliseconds** per request.
* Across a 1,000-turn agent session, matching 37 million cached tokens takes a cumulative total of **~3 to 5 seconds (<0.1 minute)**, not 3.1 to 6.2 minutes. The chat padded the session duration by 3 to 6 minutes of non-existent GPU work.

---

## 4. Recalculated Economics the Operator Can Trust

The table below presents recalculated economics from first principles using **verified hardware metrics, corrected CU/hr burn rates, and empirical decode speeds**, contrasting the chat's fantasy figures against trusted ground reality.

### Benchmark Workload Assumptions:
1. **Automated 25-Turn Agent Task:** 8,000 tokens prompt context + 500 output tokens per turn. Total: 200k prompt tokens, 12.5k output tokens.
2. **Standard 1M Output Batch:** 1,000,000 generated tokens (assuming 85% cache hit on prompt history).
3. **Monthly Pool:** Exactly **200 Compute Units (CUs)** from Google AI Pro ($19.99/mo).

| Operational Metric | Chat Claim (Flash-Next Fantasy) | Plan A: Locked Worker (27B ABLITERATED on A100-40GB) | Plan B: 27B NVFP4 on G4 (G4 Server Edition) | Flash-Next NVFP4 on G4 (Realistic Reasoning On) | Flash-Next NVFP4 on G4 (Reasoning Off / Fragile) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GPU Accelerator** | G4 (RTX PRO 6000) | **A100-SXM4-40GB** | G4 Server Edition | G4 Server Edition | G4 Server Edition |
| **CU Burn Rate** | ~6.55 CU/hr | **5.38 CU/hr** | **8.71 CU/hr** | **8.71 CU/hr** | **8.71 CU/hr** |
| **Active Weight Payload** | ~3.5 GB (claimed) | **16.8 GB (Q4_K_M)** | **20.15 GB (NVFP4)** | **78.96 GB base** | **78.96 GB base** |
| **Sustained Decode Speed** | 300–350 tok/s | **160 tok/s** (DFlash 2) | **150 tok/s** (MTP-3) | **240 tok/s** (Think) / **320** (Code) | **200 tok/s** (No Spec / Breakable) |
| **Cold Boot Latency / CU** | ~0 min (ignored) | **<45 s (0.07 CU)** | **1.5 min (0.22 CU)** | **18.0 min (2.61 CU)** | **18.0 min (2.61 CU)** |
| **Turn Latency (25-Turn Task)** | 4.0 – 6.0 s | **4.27 s** | **4.47 s** | **10.70 s** | **3.65 s** |
| **Task Wall Time (25 Turns)** | 1.8 – 2.5 min | **1.78 min** (106.8 s) | **1.86 min** (111.8 s) | **4.46 min** (267.5 s) | **1.52 min** (91.3 s) |
| **CU per 25-Turn Agent Task** | **0.20 – 0.27 CU** | **0.160 CU** | **0.270 CU** | **0.647 CU** | **0.221 CU** |
| **Tasks per 200 CU Pool** | ~800+ tasks | **1,250 tasks** | **740 tasks** | **309 tasks** | **904 tasks** |
| **CU per 1M Output Tokens** | **3.46 – 4.14 CU** | **9.34 CU** | **16.13 CU** | **28.45 CU** *(with think)* | **12.10 CU** |
| **Max Monthly Uptime (200 CU)** | ~30.5 hours | **37.17 hours** | **22.96 hours** | **22.96 hours** | **22.96 hours** |
| **Monthly Tokens / 200 CU** | 1.18B – 1.94B | **0.78 Billion** | **0.57 Billion** | **0.32 Billion** | **0.76 Billion** |
| **Refusal / Censorship Behavior** | Not evaluated | **Zero refusals (Abliterated)** | Standard refusals | Strict Nemotron refusals | Strict Nemotron refusals |

### Critical Economic Findings for the Operator:
1. **The Locked A100 Plan Wins on Task Throughput:** The planned worker (**Qwen 3.8 27B ABLITERATED on A100 40GB**) completes **1,250 agent tasks** per 200 CU pool. Flash-Next on G4 with thinking enabled completes only **309 tasks**—a **75% reduction in usable agent work** per month.
2. **A100 SXM4 Bandwidth Efficiency:** A100-SXM4-40GB delivers 1,555 GB/s of HBM2 bandwidth at 5.38 CU/hr (**289.0 GB/s per CU/hr**). G4 Server Edition delivers 1,597 GB/s of GDDR7 bandwidth at 8.71 CU/hr (**183.4 GB/s per CU/hr**). A100 delivers **57.6% more memory streaming per compute unit**.
3. **The Cold-Start Disaster:** If the operator launches ephemeral sessions for single tasks, Flash-Next burns **2.61 CUs** on environment setup and weight fetching before the task starts, expanding per-task cost from 0.65 CU to **3.26 CUs** (slashing monthly capacity to just **61 tasks**).

---

## 5. Recommended Next Sitting

The operator should proceed with the first sitting according to the existing roadmap in [`first-boot.md`](../27b-plan/first-boot.md) and [`inference.md`](../27b-plan/inference.md). Do not implement code, edit configurations, or attempt to run Flash-Next.

Execute the following sequence in order:

### Step 1: Provision Standard A100-40GB Worker
* Command: `colab new -s qwen-worker --gpu A100` (strictly avoid `--high-mem`).
* Immediate SKU Guard: Check `nvidia-smi`. Confirm `A100-SXM4-40GB` (~39.6 GiB usable VRAM). If the VM is assigned an A100 80GB (`HIGH_RAM`, ~80 GB VRAM, host RAM >110 GB), stop immediately via `colab stop -s qwen-worker` to avoid the 7.52–15.0 CU/hr billing trap.

### Step 2: Mount Google Drive and Verify Pre-Cached Artifacts
* Mount Drive via `colab drivemount` into `/content/drive`.
* Verify the presence of the pre-staged files in `/content/drive/MyDrive/colab_llm/`:
  1. `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (16.8 GB, SHA-256: `5d53637a...`)
  2. `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (3.86 GB, SHA-256: `ca6dacc0...`)
  3. Precompiled `bin/llama-server` and `bin/llama-quantize` (commit `efe3626` / tag `b11552`).
  4. Pre-cached `bin/cloudflared-linux-amd64`.

### Step 3: First-Boot Sidecar Quantization (One-Time Only)
* If `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` is missing from Drive:
  ```bash
  /content/drive/MyDrive/colab_llm/bin/llama-quantize \
    /content/drive/MyDrive/colab_llm/dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf \
    /content/drive/MyDrive/colab_llm/dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf \
    Q4_K_M
  ```
  Quantization takes ~25–35 seconds. Once written to Drive, this step is never repeated.

### Step 4: Launch `llama-server` Playbook A
* Start the server with DFlash 2 speculation, Q8_0 KV cache, and FlashAttention:
  ```bash
  /content/drive/MyDrive/colab_llm/bin/llama-server \
    -m /content/drive/MyDrive/colab_llm/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf \
    -md /content/drive/MyDrive/colab_llm/dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf \
    --spec-type draft-dflash,ngram-map-k4v \
    -c 100000 \
    -ctk q8_0 -ctv q8_0 \
    -fa on \
    --parallel 1 \
    --port 8080 \
    --host 0.0.0.0 \
    --api-key "$LLAMA_API_KEY"
  ```
* Verify that post-initialization VRAM consumption is **≤ 31,000 MiB** (leaving ~9 GB VRAM headroom on the 40GB card).

### Step 5: Smoke Test Speculative Decode Speed and Tool Calling
* Tunnel the endpoint via Cloudflare:
  ```bash
  /content/drive/MyDrive/colab_llm/bin/cloudflared tunnel --url http://localhost:8080
  ```
* Send a 500-token coding prompt with `--reasoning off` (temperature 0, greedy).
* Grep `llama-server` stdout logs:
  - Confirm DFlash 2 draft acceptance rate: $\bar{\alpha} \ge 2.8$ tokens/step.
  - Confirm sustained decode throughput: **140 to 180 tokens/sec**.
  - Confirm zero refusals on exploit/low-level script queries.

### Step 6: (Optional Probe Only) Verify G4 Entitlement Status
* Run a one-shot CLI allocation probe from your local machine:
  ```bash
  colab new -s g4-entitlement-probe --gpu G4
  ```
* If the command succeeds, record the exact allocated host specs via `nvidia-smi` and check current hourly CU deduction, then terminate immediately:
  ```bash
  colab stop -s g4-entitlement-probe
  ```
* If the command fails with entitlement or availability errors, treat G4 as permanently inaccessible for this billing cycle. Do not modify the locked A100 architecture.

---

## 6. Synthesis Sources

1. **Hardware & Billing:**
   - [NVIDIA RTX PRO 6000 Blackwell Server Edition Datasheet](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)
   - [PNY RTX PRO 6000 Server Edition Datasheet](https://www.pny.com/en-eu/file%20library/professional/datasheet/data%20center%20cards/pny-nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf)
   - [Lenovo ThinkSystem LP2263 Information Brief](https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu)
   - [NVIDIA L4 Tensor Core GPU Datasheet](https://www.nvidia.com/en-us/data-center/l4/)
   - [Google Colab FAQ & Resource Allocation Policies](https://research.google.com/colaboratory/faq.html)
   - [McCormick ML Colab GPU Pricing & Feature Matrix (2024–2026)](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/)
2. **Model Checkpoints & Configurations:**
   - [Hugging Face `nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (`config.json`, `hf_quant_config.json`, `model.safetensors.index.json`)
   - [Hugging Face `Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Official Model Card & Architecture)
   - [Hugging Face `nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) (`config.json`, `hf_quant_config.json`)
   - [Hugging Face `Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (Base BF16 Model Card)
   - [Hugging Face `Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) (Base FP8 Checkpoint)
   - [Hugging Face `nvidia/GLM-5.3-Flash-NVFP4`](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) (Official Model Card & `tp=4` Specification)
   - [Hugging Face `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) (Target GGUF, DFlash 2 Sidecar, and Metadata)
3. **Inference Runtimes & Kernels:**
   - [vLLM Project GitHub Repository](https://github.com/vllm-project/vllm) (`qwen4_exp.py`, Issue #53908, Issue #60489, Issue #54710)
   - [FlashInfer Documentation & Blackwell sm_120 Installation](https://docs.flashinfer.ai/installation.html)
   - [SGLang Project Documentation & RadixAttention Specification](https://lmsysorg.mintlify.app/)
   - [TensorRT-LLM Command Reference (`trtllm-build`)](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html)
   - [llama.cpp PR #27816 (DFlash 2 Integration on Master)](https://github.com/ggml-org/llama.cpp/pull/27816)
   - [llama.cpp PR #27742 (qwen4exp & GGML_TYPE_NVFP4)](https://github.com/ggml-org/llama.cpp/pull/27742)
4. **Empirical Benchmarks & Speculative Studies:**
   - [lukaLLM DFlash 2 Benchmark Suite on RTX PRO 6000](https://www.reddit.com/r/LocalLLaMA/comments/1vvncyh/i_benchmark_dflash_2_pr_build_in_llamacpp_on_qwen/)
   - [Inco AI DFlash 2 Diffusion-Based Speculative Drafting Analysis](https://inco.ai/blog/dflash2/)
   - [ZeroEntropy NVFP4 vs FP8 Blackwell Performance Audit](https://zeroentropy.dev/articles/fp4-vs-fp8-blackwell-benchmarks)
   - [hanxiao Qwen3.8-27B L4 & Blackwell Scaling Model](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)
   - [RunPod RTX PRO 6000 & Qwen 3.8 Concurrency Benchmark](https://www.runpod.io/blog/qwen3-8-27b-on-runpod-the-theory-behind-frontier-class-agentic-coding-that-fits-on-a-single-24gb-worker)
