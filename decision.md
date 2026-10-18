# Decision: Qwen Inference Architecture & Worker Deployment on Google Colab

**Date:** 2026-10-18  
**Repository:** `/home/sahil/Desktop/sahil/colab` ([`github.com/fantastic25hippo/colab`](https://github.com/fantastic25hippo/colab))  
**Source of Truth:** [`claims-ledger.md`](claims-ledger.md)

---

## 1. Current Decision & Architectural Rationale

### Strategic Overview: Dual-Worker Intent
The target architecture is a **dual-model deployment strategy** tailored to operator intent:
1. **Primary / Leading Candidate:** [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (or [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)) served via **SGLang (TP=1)** with `--ple-offload-embedding` on **Google Colab G4** ([NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB GDDR7](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/)).
2. **Fallback Worker:** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) (`Q4_K_M`) served via **`llama.cpp`** (commit `efe3626` / tag `b11552`) with a locally quantized DFlash 2 `Q4_K_M` sidecar on **Google Colab A100-SXM4-40GB**.

The system does **not** conclude "cannot run on one G4" or "do not switch". Single-GPU execution of Flash-Next on G4 is mathematically sound and officially verified by SGLang. The dual-model structure exists because stock NVIDIA checkpoints enforce alignment refusals on sensitive coding tasks, community abliterated NVFP4 checkpoints remain unverified for perplexity, and live Colab G4 provisioning must be probed before deprecating the proven A100 fallback.

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
2. **Abliterated NVFP4 Checkpoints are Unverified:** Community abliterated quants exist on Hugging Face (`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` and `claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`), but whether they load cleanly in SGLang or preserve reasoning quality without perplexity blowup on Blackwell `sm_120` remains empirically unverified.
3. **Open Colab Infrastructure Probes:** G4 provisioning entitlement on Google AI Pro accounts and Colab's unprivileged user environment memory pinning (`mlock`) limits without Docker remain unverified until probed live.
4. **`llama.cpp` + DFlash 2 is Unviable for Flash-Next:** DFlash 2 was trained exclusively on Qwen 27B dense hidden states and cannot transfer to Flash-Next. Furthermore, mmapping the 51.2B PLE table across CPU memory in `llama.cpp` collapses token generation to 1–5 tok/s.
5. **TensorRT-LLM is Definitively Out:** `trtllm-build` takes 30–50+ minutes, generates a non-portable 70+ GB engine plan, and burns 3.3–5.4 CU before a single token can be generated. SGLang is the sole viable single-GPU production engine for Flash-Next.
6. **The Blackfrost 27B Fallback is Rock-Solid:** Blackfrost Qwen3.8-27B ABLITERATED Q4_K_M GGUF (16.81 GB) with DFlash 2 Q4_K_M (1.14 GB) on A100-SXM4-40GB provides zero refusals, guaranteed VRAM stability (≤31,000 MiB with 100K `q8_0` KV cache), and a <45-second warm boot from Google Drive at a lower billing rate (~5.37 CU/hr).

---

## 2. Candidate Evaluation Matrix

| Candidate | Status | Architecture & Precision | Download / Resident Size | Primary URL / Source | Strategic Rationale & Notes |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **`nvidia/Qwen3.8-Flash-Next-NVFP4`** | **chosen** | 124.91B MoE (10/512 active, ~6.0B active/tok) + 51.2B FP8 PLE + ~2.76B FP8 MTP | 132.68 GB (11 shards); ~78 GiB VRAM + ~47.7 GiB Host RAM | [Hugging Face](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) | **Primary leading candidate.** Official ModelOpt v0.46.0 release. Verified in SGLang TP=1 cookbook. Mixed precision: do not pass `--quantization`. Subject to stock safety refusals. |
| **`RadixArk/Qwen3.8-Flash-Next-NVFP4`** | **chosen** | Same 125B MoE + 51B PLE architecture; ModelOpt FP4 quant | Similar footprint; uses `--quantization modelopt_fp4` | [Hugging Face](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | **Alternative chosen candidate.** Verified alongside NVIDIA checkpoint in SGLang cookbook. Stock alignment. |
| **`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD`** | **unverified** | Abliterated Flash-Next quant; 106 GB total checkpoint | 106 GB download | [Hugging Face](https://huggingface.co/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD) | Uncensored community candidate. Load stability, tensor structure, and perplexity on `sm_120` remain unverified without a live test. |
| **`claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`** | **unverified** | Abliterated Flash-Next quant with NVFP4 LM head | ~80–110 GB download | [Hugging Face](https://huggingface.co/claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4) | Community uncensored checkpoint. Requires live verification for kernel errors and reasoning degradation on Blackwell. |
| **`Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`** | **fallback** | Dense 27.8B (48 GDN + 16 full-attn); `Q4_K_M` GGUF | 16.81 GB base (SHA: `5d53637a...`) + 1.14 GB sidecar | [Hugging Face](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) | **Primary operational fallback.** Fully abliterated (zero refusals). Runs rock-solid on Colab A100-SXM4-40GB via `llama.cpp` + DFlash 2. VRAM fits 100K context ≤31,000 MiB. |
| **`nvidia/Qwen3.8-27B-NVFP4`** | **rejected** | Dense 27.8B; mixed NVFP4 MLPs + FP8 attention + BF16 embeds | 21.92 GB (3 shards); loaded VRAM ~20.2–23.5 GB | [Hugging Face](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) | Rejected as primary worker. Stock alignment (refuses requests). If deploying on G4, Flash-Next is significantly faster (~148 vs ~71–90 tok/s AR); if an uncensored model is required, Blackfrost 27B on A100 is superior. |
| **`nvidia/GLM-5.3-Flash-NVFP4`** | **rejected** | 321.32B MoE (18B active, 288 routed experts + 1 shared) | 204.46 GB safetensors; loaded VRAM ~195–205 GB | [Hugging Face](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) | **Physically unviable on single G4.** Weight footprint exceeds 96 GB by ~100–110 GB. Colab cannot cluster multi-VM TP=2. Offloading over PCIe collapses decode to 0.3–3.5 tok/s. NVIDIA mandates $\ge 4\times 96\text{ GB}$ (TP=4). |

---

## 3. GPU Platform Analysis: Colab G4 vs Colab A100

| Metric / Dimension | Colab G4 (NVIDIA RTX PRO 6000 Blackwell Server Edition) | Colab A100 (NVIDIA A100-SXM4-40GB) | Architectural & Economic Impact |
| :--- | :---: | :---: | :--- |
| **Silicon Architecture** | GB202 (`sm_120` / `sm_120a`), 5th-Gen Tensor Cores | GA100 (`sm_80`), 3rd-Gen Tensor Cores | G4 provides native NVFP4 block-scaled MMA tensor acceleration; A100 has 0 FP4 compute. |
| **Usable VRAM** | **95.60 GiB** (97,896 MiB in `nvidia-smi`) | **~39.60 GiB** (~40,536 MiB in `nvidia-smi`) | G4 holds Flash-Next weights (~78 GiB) with room for 262K KV cache (~3.07 GiB); A100 holds 27B weights (16.8 GB) + 100K KV. |
| **Memory Bandwidth** | **1,597 GB/s** GDDR7 (512-bit bus @ 24.95 Gbps) | **1,555 GB/s** HBM2 (5,120-bit bus @ 2.4 Gbps) | **A100 delivers 97.4% of G4 memory bandwidth.** (Note: 1,792 GB/s is the active Workstation card, not Colab's Server Edition). |
| **FP4 Tensor Compute** | 2.0 PFLOPS dense / 4.0 PFLOPS (2:4 sparse) | None (0 TFLOPS) | Native NVFP4 matrix math enables fast MoE computation and MTP drafting. |
| **FP8 Tensor Compute** | 1,000 TFLOPS (1.0 PFLOP) dense / 2.0 PFLOPS sparse | None native (simulated via software/INT8) | Accelerated FP8 attention and MTP draft execution on Blackwell. |
| **Standard Billing Rate** | **8.71 CU/hr** ($0.1452 \text{ CU/min}$) | **~5.37–5.40 CU/hr** ($0.0895 \text{ CU/min}$) | **A100 burns 61.7% of the CU rate of G4.** (6.4–6.7 CU/hr on G4 was promotional off-peak; 8.90 CU/hr noted in later samples). |
| **200 CU Pool Monthly Uptime** | **~22.96 hours (~23 hours)** | **~37.24 hours (~37 hours)** | A100 provides ~14 additional operational hours per month from the same Google AI Pro budget. |
| **Host System RAM** | **~177 GB** (176.9 GB measured; 48 vCPUs) | **53–64 GB** (standard Colab instance RAM) | G4 host RAM comfortably fits the 47.7 GiB FP8 PLE table with >120 GB free; A100 cannot host-offload 51B tables. |
| **Cluster Interconnect** | None (single-GPU VM only, `/dev/nvidia0`) | None (single-GPU VM only, `/dev/nvidia0`) | Both instances are strictly single-GPU. Multi-VM `tp=2` is impossible on Colab. |

> [!NOTE]
> **A100-80GB / High-RAM Warning:** Google Colab A100-80GB or High-RAM instances burn **~6.77 to 15.0 CU/hr**. High-RAM shapes are barred; sessions must be immediately terminated if the provisioned VM reports >110 GB host RAM or ~80 GB VRAM.

---

## 4. First-Principles Economic Recalculation

### Reconciling the "Under 40 Tasks" Fallacy
Prior repository drafts (`research/archive/flash-next-audit/08_colab_cu_economics.md` and `10_claims_ledger_and_verdict.md`) asserted that a 200 CU monthly pool would be depleted in "under 40 agent tasks". That conclusion resulted from compounding two unrealistic, worst-case assumptions:
1. **Cold-boot overhead charged to every single task:** Assuming an 8–18 minute cold boot (1.16–2.61 CU) on every task. In practice, an operator boots a session once per sitting (2–4 hours) and executes multiple tasks sequentially.
2. **Unconstrained verbose reasoning tokens:** Assuming 2,500+ `<think>` tokens generated per turn. When extended thinking is delegated to the local orchestrator or capped, task generation consists of concise code diffs (500 tokens/turn).

### Explicit Labeled Assumptions (Ledger §3)
- **Billing Rates:** G4 standard metered rate = `8.71 CU/hr` ($0.1452/\text{min}$, $0.002419/\text{sec}$). A100 standard metered rate = `5.37 CU/hr` ($0.0895/\text{min}$, $0.001492/\text{sec}$).
- **Monthly Allocation:** 200 Compute Units (CU) from Google AI Pro pool.
- **Total Dedicated Uptime:** 22.96 hours on G4; 37.24 hours on A100.
- **Task Specification:** 25-turn autonomous coding loop.
  - *Prefill:* 8,000 initial context tokens; subsequent turns append ~1,000 delta tokens. With SGLang RadixAttention / APC prefix caching, cached history binds in <5 ms on host CPU; delta prefill takes ~0.12 s.
  - *Generation:* 500 tokens of code / diff per turn (12,500 output tokens total per task).

### Comparative Task Capacity Matrix (200 CU Monthly Budget)

| Scenario / Configuration | Decode Speed | Time / Turn | Active Task Time | Amortized Boot Overhead | CU Burn / Task | Tasks per 200 CU Pool |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Flash-Next on G4 (Headless Loop, 10-Task Sitting)** | 148 tok/s (MTP) | 3.50 s | 87.5 s (1.46 min) | 1.0 min / task (10 min boot / 10 tasks) | **0.357 CU** | **~560 tasks** |
| **Flash-Next on G4 (Continuous Session, Warm)** | 148 tok/s (MTP) | 3.50 s | 87.5 s (1.46 min) | 0.0 min (fully warm session) | **0.212 CU** | **~943 tasks** |
| **Flash-Next on G4 (Verbose Thinking, 2.5k tok/turn)** | 148 tok/s (MTP) | 17.0 s | 425 s (7.08 min) | 1.0 min / task (10 min boot / 10 tasks) | **1.173 CU** | **~170 tasks** |
| **Qwen 27B on A100 40GB (Scaled No-Spec, Warm)** | ~62–75 tok/s | 6.79–8.18 s | 170–205 s (2.83–3.41 min) | 0.05 min / task (45 s boot / 15 tasks) | **0.258–0.310 CU** | **~645–775 tasks** |
| **Qwen 27B on A100 40GB (Scaled DFlash 2, Warm)** | ~95–140 tok/s | 3.69–5.38 s | 92.3–135 s (1.54–2.24 min) | 0.05 min / task (45 s boot / 15 tasks) | **0.142–0.205 CU** | **~975–1,408 tasks** |
| **Qwen 27B on G4 (NVFP4 + MTP-3, Warm)** | 160 tok/s | 3.25 s | 81.3 s (1.35 min) | 0.15 min / task (1.5 min boot / 10 tasks) | **0.218 CU** | **~917 tasks** |

> [!IMPORTANT]
> **Speed Benchmark Attribution:** The 187.13 tok/s benchmark belongs strictly to lukaLLM running DFlash 2 ($n=5$) on an active-cooled RTX PRO 6000 Workstation (1,792 GB/s) and must **not** be attributed to A100. On A100-SXM4-40GB, Qwen 27B DFlash 2 throughput is unpublished; scaled estimates yield 95–140 tok/s, which delivers **975 to 1,408 tasks per 200 CU pool** due to A100's lower 5.37 CU/hr billing rate. Flash-Next on G4 delivers **560 to 943 tasks**, demonstrating viable economics under all practical operating conditions.

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
4. **Abliterated Flash-Next NVFP4 Checkpoint Load & Perplexity Viability:**
   - *Status:* **`unverified`**
   - *Unknown:* Community uncensored checkpoints exist on Hugging Face ([`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD`](https://huggingface.co/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD) at 106 GB, and [`claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`](https://huggingface.co/claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4)). Whether these specific quants load cleanly without tensor layout errors in SGLang or preserve reasoning quality without perplexity blowup on Blackwell `sm_120` is unverified.
5. **Real-World G4 Server Edition Decode Speed:**
   - *Status:* **`unverified`**
   - *Unknown:* SGLang's ~148 tok/s benchmark was recorded on an active-cooled RTX PRO 6000 Workstation (1,792 GB/s). Colab's passively cooled Server Edition runs at 1,597 GB/s (10.9% lower memory bandwidth) with potential thermal constraints in high-density multi-tenant racks. Real-world generation speed is expected to be ~130–135 tok/s with MTP, but must be verified on live G4 hardware.
