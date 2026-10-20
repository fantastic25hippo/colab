# Decision: Qwen Inference Architecture & Worker Deployment on Google Colab

**Date:** 2026-10-18  
**Repository:** `/home/sahil/Desktop/sahil/colab` ([`github.com/fantastic25hippo/colab`](https://github.com/fantastic25hippo/colab))  
**Source of Truth:** [`claims-ledger.md`](claims-ledger.md)

---

## 1. Current Decision & Architectural Rationale

### Strategic Overview: Dual-Worker Intent
The target architecture is a **dual-model deployment strategy** tailored to operator intent:
1. **Primary / Leading Candidate:** [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (or [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)) served via **SGLang (TP=1)** with `--ple-offload-embedding` on **Google Colab G4** ([NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB GDDR7](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)).
2. **Fallback Candidate:** An abliterated Flash-Next checkpoint (TBD, unverified) if refusal mitigation is required. Candidates: [`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD`](https://huggingface.co/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD) (106 GB), [`claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`](https://huggingface.co/claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4) (105.48 GB), and [`Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`](https://huggingface.co/Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) (112 GB, GGUF, different runtime).

The system does **not** conclude "cannot run on one G4" or "do not switch". Single-GPU execution of Flash-Next on G4 is mathematically sound and officially verified by SGLang. The dual-model structure exists because stock NVIDIA checkpoints enforce alignment refusals on sensitive coding tasks, community abliterated checkpoints remain unverified for perplexity, and live Colab G4 provisioning must be probed.

---

### Why Flash-Next on Colab G4 is the Primary Target

- **Architectural Efficiency:** Flash-Next is a 124.91B base MoE model (512 fine-grained routed experts, top-10 active routing, 1 shared expert) where only ~6.0B parameters activate per token (~2.6B MoE + ~3.4B shared/attention/embeddings). In addition, it incorporates a 51.2B FP8 N-gram Prompt Lookup Embedding (PLE) table (~47.7 GiB) and a 1-layer FP8 Multi-Token Prediction (MTP) head (~2.76B physical parameters).
- **Verified TP=1 Recipe:** The [SGLang Qwen3.8-Flash-Next Cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next) documents and verifies single RTX PRO 6000 (TP=1) execution using `--ple-offload-embedding`. The sample `--tensor-parallel-size 8` on NVIDIA's model card was an HGX B200 reference command, not an architectural mandate for single-card serving.
- **Host RAM Clearance:** Colab G4 VMs allocate **~177 GB of physical host RAM** (176.9 GB measured; 48 vCPUs), completely invalidating previous assumptions that Colab host RAM is limited to 53–64 GB. The 47.7 GiB FP8 PLE table offloads into pinned host memory with >120 GB of free headroom remaining. Asynchronous token lookups transfer only ~20.48 KB/token over PCIe 5.0 (<7.2 MB/s at 350 tok/s decode, consuming <0.012% of PCIe 5.0 x16 bandwidth).
- **VRAM Budget & Footprint:** With the PLE table in host RAM, ~78 GiB of checkpoint weights remain in VRAM. After loader temporaries are collected, post-collection resident memory is 74.7 GiB without speculative decoding and 81.8 GiB with MTP. On G4's 95.60 GiB usable VRAM (97,896 MiB in `nvidia-smi`), this leaves 19.4 GiB (without MTP) or 12.3 GiB (with MTP) usable before static runtime slack.
- **Context Capacity:** Flash-Next uses a hybrid layout where 36 of 48 layers are Gated DeltaNet (GDN) linear recurrent attention (fixed $O(1)$ memory per slot, 58.8 MB/slot FP16) and only 12 layers are full attention (2 KV heads, `head_dim=256`). Consequently, its full-attention KV cache requires only **12 KiB per token in FP8** (half the 24 KiB/token of FP16). The entire 262,144-token native context consumes only **~3.07 GiB** of KV cache. Total VRAM footprint at 262K context is ~87 GB of 95.6 GB, fitting cleanly on a single G4 without out-of-memory (OOM) errors.
- **Inference Speed:** The SGLang cookbook benchmark measures **~148 tok/s** single-request batch-1 decode with MTP (~83 tok/s without MTP) on an RTX PRO 6000 Workstation (1,792 GB/s). Colab's Server Edition (1,597 GB/s) is projected to deliver ~130–135 tok/s with MTP.

---

### Why the Fallback Exists & Still-True Constraints

1. **Alignment Refusals in Stock Checkpoints:** Official checkpoints (`nvidia/Qwen3.8-Flash-Next-NVFP4`) were aligned using Nemotron safety data and reject sensitive cybersecurity, systems programming, and penetration testing prompts. An abliterated worker is mandatory for an unrestricted coding assistant.
2. **Abliterated Checkpoints are Unverified:** Community abliterated quants exist on Hugging Face (`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD`, `claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`, and `Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`), but whether they load cleanly in SGLang (or llama.cpp for GGUF) or preserve reasoning quality without perplexity blowup on Blackwell `sm_120` remains empirically unverified.
3. **Open Colab Infrastructure Probes:** G4 provisioning entitlement on Google AI Pro accounts and Colab's unprivileged user environment memory pinning (`mlock`) limits without Docker remain unverified until probed live.
4. **`llama.cpp` + DFlash 2 is Unviable for Flash-Next:** DFlash 2 was trained exclusively on Qwen 27B dense hidden states and cannot transfer to Flash-Next. Furthermore, mmapping the 51.2B PLE table across CPU memory in `llama.cpp` collapses token generation to 1–5 tok/s.
5. **TensorRT-LLM is Definitively Out:** `trtllm-build` takes 30–50+ minutes, generates a non-portable 70+ GB engine plan, and burns 3.3–5.4 CU before a single token can be generated. SGLang is the sole viable single-GPU production engine for Flash-Next.

---

## 2. Candidate Evaluation Matrix

| Candidate | Status | Architecture & Precision | Download / Resident Size | Primary URL / Source | Strategic Rationale & Notes |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **`nvidia/Qwen3.8-Flash-Next-NVFP4`** | **chosen** | 124.91B MoE (10/512 active, ~6.0B active/tok) + 51.2B FP8 PLE + ~2.76B FP8 MTP | 132.68 GB (11 shards); ~78 GiB VRAM + ~47.7 GiB Host RAM | [Hugging Face](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | **Primary leading candidate.** Official ModelOpt v0.46.0 release. Verified in SGLang TP=1 cookbook. Mixed precision: cookbook passes no `--quantization` flag. Subject to stock safety refusals. |
| **`RadixArk/Qwen3.8-Flash-Next-NVFP4`** | **chosen** | Same 125B MoE + 51B PLE architecture; ModelOpt FP4 quant | Similar footprint; no `--quantization` flag passed | [Hugging Face](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | **Alternative chosen candidate.** Verified alongside NVIDIA checkpoint in SGLang cookbook. Stock alignment. Cookbook passes no `--quantization` flag. |
| **`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD`** | **fallback candidate, unverified** | Abliterated Flash-Next quant; 106 GB total checkpoint | 106 GB download | [Hugging Face](https://huggingface.co/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD) | Uncensored community candidate. Note: created 2026-09-21 predates listed base `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated` (2026-09-29). Loading in SGLang is UNVERIFIED. |
| **`claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`** | **fallback candidate, unverified** | Abliterated Flash-Next quant with NVFP4 LM head | 105.48 GB download | [Hugging Face](https://huggingface.co/claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4) | Community uncensored checkpoint. Requires live verification for kernel errors and reasoning degradation on Blackwell in SGLang (UNVERIFIED). |
| **`Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`** | **fallback candidate, unverified** | Abliterated Flash-Next GGUF quant | 112 GB download | [Hugging Face](https://huggingface.co/Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) | Community uncensored checkpoint (GGUF, different runtime). Needs llama.cpp support for this architecture, also UNVERIFIED. |
| **`nvidia/GLM-5.3-Flash-NVFP4`** | **rejected** | 321.32B MoE (18B active, 288 routed experts + 1 shared) | 204.46 GB safetensors; loaded VRAM ~195–205 GB | [Hugging Face](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | **Physically unviable on single G4.** Weight footprint exceeds 96 GB by ~100–110 GB. Colab cannot cluster multi-VM TP=2. Offloading over PCIe collapses decode to 0.3–3.5 tok/s. NVIDIA mandates $\ge 4\times 96\text{ GB}$ (TP=4). |

---

## 3. GPU Platform Analysis: Colab G4 (RTX PRO 6000 Server Edition)

For complete platform hardware specifications across all Colab accelerator shapes (T4, L4, A100 40GB/80GB, G4, H100, and TPUs), see [`hardware.md`](hardware.md).

| Metric / Dimension | Colab G4 Specification | Architectural & Operational Impact |
| :--- | :---: | :--- |
| **Silicon Architecture** | GB202 (`sm_120` / `sm_120a`), 5th-Gen Tensor Cores | Native NVFP4 block-scaled MMA tensor acceleration. |
| **Usable VRAM** | **95.60 GiB** (97,896 MiB in `nvidia-smi`) | Holds Flash-Next weights (~78 GiB) with room for 262K KV cache (~3.07 GiB). |
| **Memory Bandwidth** | **1,597 GB/s** GDDR7 (512-bit bus @ 24.95 Gbps) | Colab's passively cooled Server Edition runs at 1,597 GB/s (1,792 GB/s is the active Workstation card). |
| **FP4 Tensor Compute** | 2.0 PFLOPS dense / 4.0 PFLOPS (2:4 sparse) | Native NVFP4 matrix math enables fast MoE computation and MTP drafting. |
| **FP8 Tensor Compute** | 1,000 TFLOPS (1.0 PFLOP) dense / 2.0 PFLOPS sparse | Accelerated FP8 attention and MTP draft execution on Blackwell. |
| **Standard Billing Rate** | **8.71 CU/hr** ($0.1452 \text{ CU/min}$) | Standard metered rate for G4 on Google AI Pro (8.90 CU/hr noted in later samples). |
| **200 CU Pool Monthly Uptime** | **~22.96 hours (~23 hours)** | Dedicated operational hours per month from the Google AI Pro 200 CU pool. |
| **Host System RAM** | **~177 GB** (176.9 GB measured; 48 vCPUs) | Host RAM comfortably fits the 47.7 GiB FP8 PLE table with >120 GB free headroom. |
| **Cluster Interconnect** | None (single-GPU VM only, `/dev/nvidia0`) | Strictly single-GPU. Multi-VM `tp=2` is impossible on Colab; TP=1 with PLE offload is required. |

---

## 4. First-Principles Economic Recalculation

For the full, canonical mathematical derivation and 6-row capacity matrix across Flash-Next and historical 27B configurations, see [`claims-ledger.md §3`](claims-ledger.md#3-first-principles-economic-recalculation).

### Operational Recap
- **Billing Rates & Budget:** Colab G4 standard metered rate is `8.71 CU/hr` ($0.1452/\text{min}$, $0.002419/\text{sec}$), providing **~22.96 hours** (1,377.7 minutes) from the 200 Compute Units (CU) monthly pool.
- **Task Delivery:** Under practical multi-task sittings with prefix caching active and boot overhead amortized, Flash-Next on G4 delivers **560 to 943 tasks per 200 CU pool** (0.212–0.357 CU/task), resolving the prior "under 40 tasks" fallacy.

---

## 5. Open Empirical Probes (UNVERIFIED Status)

The 5 open empirical unknowns (AI Pro G4 entitlement, native `mlock` without Docker, real cold-boot latency and CU burn, abliterated checkpoint viability, and real Server Edition decode speed) are tracked canonically in [`claims-ledger.md §5`](claims-ledger.md#5-unverified-remaining). Operational verification procedures are executed via [`runbooks/g4-probe.md`](runbooks/g4-probe.md).
