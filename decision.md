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

### Reconciling the "Under 40 Tasks" Fallacy
Prior repository drafts (`research/archive/flash-next-audit/08_colab_cu_economics.md` and `10_claims_ledger_and_verdict.md`) asserted that a 200 CU monthly pool would be depleted in "under 40 agent tasks". That conclusion resulted from compounding two unrealistic, worst-case assumptions:
1. **Cold-boot overhead charged to every single task:** Assuming an 8–18 minute cold boot (1.16–2.61 CU) on every task. In practice, an operator boots a session once per sitting (2–4 hours) and executes multiple tasks sequentially.
2. **Unconstrained verbose reasoning tokens:** Assuming 2,500+ `<think>` tokens generated per turn. When extended thinking is delegated to the local orchestrator or capped, task generation consists of concise code diffs (500 tokens/turn).

### Explicit Labeled Assumptions (Ledger §3)
- **Billing Rates:** G4 standard metered rate = `8.71 CU/hr` ($0.1452/\text{min}$, $0.002419/\text{sec}$).
- **Monthly Allocation:** 200 Compute Units (CU) from Google AI Pro pool.
- **Total Dedicated Uptime:** 22.96 hours on G4.
- **Task Specification:** 25-turn autonomous coding loop.
  - *Prefill:* 8,000 initial context tokens; subsequent turns append ~1,000 delta tokens. With SGLang RadixAttention / APC prefix caching, cached history binds in <5 ms on host CPU; delta prefill takes ~0.12 s.
  - *Generation:* 500 tokens of code / diff per turn (12,500 output tokens total per task).

### Comparative Task Capacity Matrix (200 CU Monthly Budget)

| Scenario / Configuration | Decode Speed | Time / Turn | Active Task Time | Amortized Boot Overhead | CU Burn / Task | Tasks per 200 CU Pool |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flash-Next on G4 (Headless Loop, 10-Task Sitting)** | 148 tok/s (MTP) | 3.50 s | 87.5 s (1.46 min) | 1.0 min / task (10 min boot / 10 tasks) | **0.357 CU** | **~560 tasks** |
| **Flash-Next on G4 (Continuous Session, Warm)** | 148 tok/s (MTP) | 3.50 s | 87.5 s (1.46 min) | 0.0 min (fully warm session) | **0.212 CU** | **~943 tasks** |
| **Flash-Next on G4 (Verbose Thinking, 2.5k tok/turn)** | 148 tok/s (MTP) | 17.0 s | 425 s (7.08 min) | 1.0 min / task (10 min boot / 10 tasks) | **1.173 CU** | **~170 tasks** |

> [!IMPORTANT]
> **Economics Summary:** Flash-Next on G4 delivers **560 to 943 tasks per 200 CU pool** across typical coding session profiles, demonstrating viable economics under practical operating conditions without requiring multi-GPU nodes.

---

## 5. Open Empirical Probes (UNVERIFIED Status)

The following five items represent empirical unknowns that cannot be proven by documentation or offline mathematics alone. They remain explicitly classified as **`unverified`** and will be resolved via live runtime testing:

1. **AI Pro Account Entitlement for G4 Allocation:**
   - *Status:* **`unverified`**
   - *Unknown:* An engineering post on [`note.com`](https://note.com/gakushiai/n/n71ca7ca09d1b) reported that direct `--gpu G4` selection was grayed out on a Google AI Pro account (requiring Google AI Ultra or Colab Pro+). Whether Sahil's specific Google AI Pro account can provision `colab new -s qwen-worker --gpu G4` is unknown without a live probe. The note.com report is not assumed to apply to Sahil.
2. **Colab Host Memory Pinning (`mlock`) Without Docker:**
   - *Status:* **`unverified`**
   - *Unknown:* SGLang's `--ple-offload-embedding` requires pinning ~47.7 GiB of host RAM. While Colab G4 provides ~177 GB of physical host RAM, standard Colab notebooks run as unprivileged users inside container sandboxes. Whether Colab's default user limits allow pinning >48 GB without root Docker container execution (`--ulimit memlock=-1`) is unverified.
3. **Real-World Colab Cold-Boot Time and CU Burn:**
   - *Status:* **`unverified`**
   - *Unknown:* Downloading the 132.68 GB checkpoint over Google Cloud's internal backbone, installing SGLang wheels, and JIT-compiling FlashInfer CUTLASS kernels for `sm_120f` is estimated at 8–18 minutes (1.16–2.61 CU). True elapsed time and transient disk bandwidth throttling on ephemeral Colab instances require live empirical measurement.
4. **Abliterated Flash-Next Checkpoint Load & Perplexity Viability:**
   - *Status:* **`unverified`**
   - *Unknown:* Community uncensored checkpoints exist on Hugging Face ([`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD`](https://huggingface.co/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD) at 106 GB, [`claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`](https://huggingface.co/claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4) at 105.48 GB, and [`Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`](https://huggingface.co/Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) at 112 GB). Whether these specific quants load cleanly without tensor layout errors in SGLang (or llama.cpp for GGUF) or preserve reasoning quality without perplexity blowup on Blackwell `sm_120` is unverified.
5. **Real-World G4 Server Edition Decode Speed:**
   - *Status:* **`unverified`**
   - *Unknown:* SGLang's ~148 tok/s benchmark was recorded on an active-cooled RTX PRO 6000 Workstation (1,792 GB/s). Colab's passively cooled Server Edition runs at 1,597 GB/s (10.9% lower memory bandwidth) with potential thermal constraints in high-density multi-tenant racks. Real-world generation speed is expected to be ~130–135 tok/s with MTP, but must be verified on live G4 hardware.
