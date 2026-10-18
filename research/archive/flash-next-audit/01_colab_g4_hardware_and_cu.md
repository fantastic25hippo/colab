# Colab G4 Hardware Architecture, Tensor Compute Rooflines, and Compute Unit Economics

Researched: 2026-10-17 | Engine: Gemini 3.8 Flash (Medium) | Job: 01

## Verdict in one paragraph

The Google Colab `--gpu G4` accelerator is conclusively identified as the **NVIDIA RTX PRO 6000 Blackwell Server Edition**, featuring **96 GB GDDR7 ECC** VRAM, a 512-bit bus running at ~25 Gbps for **1,597 GB/s** peak memory bandwidth (passively cooled, 400W–600W configurable), and compute capability **`sm_120` / `sm_120a`** with native block-scaled NVFP4 Tensor Cores. Peak tensor throughput is **4.0 PFLOPS FP4** and **2.0 PFLOPS FP8** with 2:4 structural sparsity (**2.0 PFLOPS FP4** and **1.0 PFLOP / 1,000 TFLOPS FP8** dense). The conflict between the Gemini chat's **~6.4–6.7 CU/hr** burn rate and `architecture.md`'s **8.71 CU/hr** reflects Colab's dynamic tiering: 8.71 CU/hr is the standard metered rate for explicit G4 provisioning in 2026, while 6.40–6.74 CU/hr was captured during early rollouts, off-peak periods, and backend stockout fallbacks. Crucially, **Google AI Pro (200 CU/month) subscribers cannot reliably select G4**: direct allocation is grayed out and gated to Google AI Ultra and Colab Pro+, making G4 accessible only through rare capacity fallbacks. Standard Colab is strictly single-GPU with zero multi-VM tensor parallelism (`tp=2`), and billing runs continuously on connected wall-clock regardless of whether the GPU is actively computing or idling.

---

## Claim table

| ID | Chat claim | Label | Corrected value | Source URL |
| :--- | :--- | :--- | :--- | :--- |
| **H1** | Colab "NVIDIA G4" is the NVIDIA RTX PRO 6000 Blackwell. | **CONFIRMED** | NVIDIA RTX PRO 6000 Blackwell Server Edition (passively cooled PCIe Gen 5). | [NVIDIA Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [Lenovo ThinkSystem LP2263](https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu) |
| **H2** | That card has **96 GB** VRAM. | **CONFIRMED** | 96 GB GDDR7 with ECC (95.6 GiB / 97,896 MiB usable reported by `nvidia-smi`). | [PNY Server Edition Datasheet](https://www.pny.com/en-eu/file%20library/professional/datasheet/data%20center%20cards/pny-nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf) |
| **H3** | Memory bandwidth is **1,597 GB/s** GDDR7. | **CONFIRMED** | 1,597 GB/s GDDR7 (512-bit bus @ 24.95 Gbps). Not HBM. | [NVIDIA Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/), [PNY Datasheet](https://www.pny.com/en-eu/file%20library/professional/datasheet/data%20center%20cards/pny-nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf) |
| **H4** | FP4 tensor compute is **4 PFLOPS**. | **PARTIALLY TRUE** | 4.0 PFLOPS with 2:4 structural sparsity; **2.0 PFLOPS** dense GEMM (no sparsity). | [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/), [JarvisLabs RTX PRO 6000](https://jarvislabs.ai/) |
| **H5** | FP8 tensor compute is **~1,000+ TFLOPS**. | **CONFIRMED** | **1,000 TFLOPS (1.0 PFLOP)** dense; 2,000 TFLOPS (2.0 PFLOPS) with 2:4 sparsity. | [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/), [E2E Networks Blackwell](https://www.e2enetworks.com/) |
| **H6** | Colab G4 burns **~6.4–6.7 CU/hr** (chat also uses **~6.5**). | **PARTIALLY TRUE** | **8.71 CU/hr** standard metered rate; 6.40–6.74 CU/hr observed only in early rollouts or off-peak fallbacks. | [McCormick Colab Pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/), [Qiita Colab GPU Benchmark](https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f) |
| **H7** | Colab L4 burns **~2.1 CU/hr**. | **PARTIALLY TRUE** | Standard baseline is **1.71 CU/hr**; reaches **2.09–2.10 CU/hr** under dynamic peak demand or High-RAM. | [McCormick Colab Pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/), [GPUPerHour Colab Analysis](https://gpuperhour.com/blog/google-colab-alternatives) |
| **H8** | G4 hourly price is **~3.1×** L4. | **PARTIALLY TRUE** | $6.5 / 2.1 = 3.1\times$ is cherry-picked. Against standard baselines ($8.71 / 1.71$), G4 is **5.09× (~5.1×)** L4; against peak L4 ($8.71 / 2.1$), it is **4.15×**. | [McCormick Colab Pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/), [Qiita Colab Benchmark](https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f) |
| **H9** | L4 has **24 GB** VRAM and **300 GB/s** bandwidth (**5.3×** slower decode streaming than G4). | **CONFIRMED** | 24 GB GDDR6, 300.05 GB/s (192-bit bus @ 12.5 Gbps). Bandwidth ratio: $1,597 / 300.05 = \mathbf{5.32\times}$. | [NVIDIA L4 Datasheet](https://www.nvidia.com/en-us/data-center/l4/), [hanxiao L4 Study](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) |
| **H10** | L4 FP8 tensor compute is **~242 TFLOPS** (**16×** below G4 FP4). | **PARTIALLY TRUE** | L4 FP8 is 242 TFLOPS sparse / 121 TFLOPS dense. Comparing G4 FP4 (4,000 TFLOPS) to L4 FP8 (242 TFLOPS) is an apples-to-oranges cross-precision comparison. FP8-to-FP8 ratio is **8.26×**, not 16×. L4 has zero native FP4 capability. | [NVIDIA L4 Datasheet](https://www.nvidia.com/en-us/data-center/l4/), [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/) |
| **H11** | Standard Colab is **single-GPU**; you cannot tensor-parallel two G4/A100 VMs. | **CONFIRMED** | Runtimes assign exactly one virtual accelerator per VM; no multi-node NVLink or cross-VM interconnect for `tp=2`. | [Google Colab FAQ](https://research.google.com/colaboratory/faq.html), [Colab CLI Session Docs](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md) |
| **H12** | Colab bills **connected wall-clock**, including idle GPU time, not tokens. | **CONFIRMED** | Billing is strictly metered per elapsed second connected to the VM runtime. Idle time burns full CU/hr. | [Google Colab FAQ](https://research.google.com/colaboratory/faq.html), [Google Developers Blog](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/) |
| **H13** | Colab AI Pro / this account can actually assign a G4. | **FALSE** | Direct G4 selection is grayed out / tier-gated on Google AI Pro (200 CU); accessible only via rare backend stockout fallback. | [note.com Google AI Pro Test](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en), [Google Developers Blog](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/) |
| **H14** | G4 is Blackwell `sm_120` with native NVFP4 tensor cores. | **CONFIRMED** | GB202 silicon targets `sm_120` (and `sm_120a` for block-scaled MMA instructions) with 5th-Gen Tensor Cores. | [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/), [FlashInfer Blackwell Docs](https://flashinfer.ai/) |
| **H15** | Existing repo figure **~8.71 CU/hr** for G4 is wrong, or the chat's **6.4–6.7** is wrong. Resolve conflict with `architecture.md`. | **INDEPENDENT CALC** | **8.71 CU/hr is the current standard rate**; 6.4–6.74 is an optimistic lower bound from early rollouts/fallback. The plan must budget at 8.71 CU/hr. | [McCormick Colab Pricing](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/), [Reddit Colab G4 Thread](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/) |
| **H16** | Existing repo figure **1,792 GB/s** is the Workstation card, **1,597 GB/s** is Colab Server Edition. Confirm which SKU Colab G4 is. | **CONFIRMED** | Colab G4 is the Server Edition (passive heatsink, 1,597 GB/s). Workstation Edition is 1,792 GB/s (active blower, 28 Gbps). | [VRLA Tech Workstation vs Server](https://vrlatech.com/rtx-pro-6000-blackwell-workstation-vs-server-edition/), [Lenovo ThinkSystem LP2263](https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu) |

---

## Findings

### 1. Hardware SKU and Identity of Colab `--gpu G4`

When an operator provisions a `--gpu G4` instance in Google Colab (or through `google-colab-cli` via `colab new -s <session> --gpu G4`), the virtual machine is backed by Google Cloud's G4 accelerator tier.

* **Marketing Name:** NVIDIA RTX PRO 6000 Blackwell Server Edition (sometimes labeled in driver strings as `NVIDIA RTX PRO 6000 Server Edition`).
* **Underlying Architecture:** NVIDIA Blackwell (GB202 die derivative), manufactured on the custom TSMC 4NP process.
* **Core Count:** 24,064 CUDA cores, 752 5th-Generation Tensor Cores, and 188 4th-Generation RT cores.
* **Compute Capability:** `sm_120` (general Blackwell client/workstation instruction set) and `sm_120a` (architecture-accelerated target supporting block-scaled Tensor Core MMA instructions). Enterprise data center Blackwell dies (B100/B200) use `sm_100`.
* **VRAM Capacity:** 96 GB GDDR7 with hardware ECC. In guest Linux kernels, `nvidia-smi` reports total VRAM as **97,896 MiB (~95.60 GiB)** due to driver and page table reservations.

### 2. Memory Bandwidth: 1,597 GB/s vs. 1,792 GB/s

A persistent conflict in recent LLM serving literature is whether the RTX PRO 6000 delivers 1,597 GB/s or 1,792 GB/s. Official datasheets from NVIDIA, PNY, and Lenovo clarify that NVIDIA manufactures two distinct hardware SKUs under the RTX PRO 6000 name:

1. **RTX PRO 6000 Blackwell Workstation Edition:** Built for desktop workstations. Features active dual-fan cooling, a 600W max TDP, and 28 Gbps GDDR7 memory over a 512-bit bus. Peak bandwidth is calculated as:
   $$\frac{512 \text{ bits}}{8 \text{ bits/byte}} \times 28.0 \text{ Gbps} = 64 \text{ bytes} \times 28.0 \text{ GHz} = \mathbf{1,792 \text{ GB/s}}$$
   This card was used by `lukaLLM` in local benchmark sweeps.
2. **RTX PRO 6000 Blackwell Server Edition (Colab G4 SKU):** Designed for high-density 1U/2U/4U server chassis. It features a **passive heatsink** relying on chassis airflow, a configurable 400W–600W power envelope, and memory clocked at ~24.95 Gbps over the 512-bit bus:
   $$\frac{512 \text{ bits}}{8 \text{ bits/byte}} \times 24.953 \text{ Gbps} \approx \mathbf{1,597 \text{ GB/s}}$$

**Memory Type Distinction:** Both SKUs use **GDDR7 memory**, not High Bandwidth Memory (HBM2/HBM2e/HBM3e). HBM is reserved for NVIDIA's A100, H100, H200, and B200 accelerators. While GDDR7 provides high peak bandwidth, its efficiency under uncoalesced random memory reads (such as sparse MoE routing) is lower than HBM's multi-channel crossbar.

### 3. Tensor Core Compute Rooflines (NVFP4, FP8, FP16)

The chat claims 4 PFLOPS FP4 and ~1,000+ TFLOPS FP8. The official Blackwell Tensor Core specification reveals how these numbers relate to structural sparsity:

* **FP4 Tensor Compute (NVFP4 / MXFP4):**
  * **Dense (Unstructured):** **2.0 PFLOPS (2,000 TFLOPS)**
  * **Sparse (2:4 Structural Sparsity):** **4.0 PFLOPS (4,000 TFLOPS)**
  * *Verdict:* Standard dense GEMMs in vLLM or TensorRT-LLM without 2:4 pruning run against the **2.0 PFLOPS** ceiling. The chat's 4 PFLOPS figure assumes 2:4 structural sparsity.
* **FP8 Tensor Compute (FP8 E4M3 / E5M2):**
  * **Dense (Unstructured):** **1.0 PFLOP (1,000 TFLOPS)**
  * **Sparse (2:4 Structural Sparsity):** **2.0 PFLOPS (2,000 TFLOPS)**
  * *Verdict:* The chat's "~1,000+ TFLOPS" accurately reflects the **dense FP8** hardware capability.
* **FP16 / BF16 Tensor Compute:**
  * **Dense:** **500 TFLOPS (0.5 PFLOP)**
  * **Sparse:** **1,000 TFLOPS (1.0 PFLOP)**
* **TF32 / FP32 Vector Compute:**
  * TF32 Tensor Core: 234 TFLOPS (sparse) / 117 TFLOPS (dense)
  * FP32 Vector (CUDA Cores): 120 TFLOPS

### 4. Colab Compute Unit Burn Rates (Resolution of H15)

The conflict between the chat's **6.4–6.7 CU/hr** and `architecture.md`'s **8.71 CU/hr** is the most significant economic discrepancy in the project. Historical and recent community benchmarks (Qiita, McCormickML, Reddit `r/GoogleColab`, note.com) explain this variation:

| Hardware Accelerator | Documented Baseline CU/hr | Observed Peak / Demand CU/hr | Approximate USD Cost/hr ($0.10/CU) | Notes & Machine Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Tesla T4** | 1.19 | 1.35 | ~$0.12–$0.14 | 16 GB GDDR6, standard compute |
| **NVIDIA L4** | **1.71** | **2.09–2.10** | ~$0.17–$0.21 | 24 GB GDDR6, standard shape; 2.1 under peak demand |
| **A100-SXM4-40GB** | **5.37–5.40** | 5.40 | ~$0.54 | 40 GB HBM2, standard shape (`--gpu A100`) |
| **A100-SXM4-80GB** | 6.77 | **7.52** | ~$0.68–$0.75 | 80 GB HBM2e, high-mem shape (`--gpu A100 --high-mem`) |
| **G4 (RTX PRO 6000)** | **6.40–6.74** | **8.71** | ~$0.67–$0.87 | 96 GB GDDR7; **8.71 is the standard rate** |
| **NVIDIA H100 80GB** | 18.02 | 18.05 | ~$1.80 | 80 GB HBM3, Ultra/Pro+ tier only |

#### Why both numbers exist:
1. **8.71 CU/hr:** Measured consistently by multiple independent 2026 benchmarks (Qiita 2026-02, McCormickML 2026-03, ThunderCompute) when directly selecting and running G4 under standard cluster allocations on Colab Pro/Pro+.
2. **6.40–6.74 CU/hr:** Observed during Google's initial G4 rollout phase and documented in community threads when Colab's backend allocator dynamically substituted a G4 during regional L4/A100 stockouts, or during low-demand off-peak windows.
3. **Resolution:** **8.71 CU/hr is the standard rate.** Using 6.4–6.7 CU/hr in financial models underestimates the operational cost of running a G4 by **23% to 26%**.

### 5. Google AI Pro Entitlement: Can This Plan Assign G4?

**No.** Following Google's September 22, 2026 integration of Colab benefits into Google AI plans:
* **Google AI Pro (200 CU/month):** Granted priority access to standard GPUs (**T4, L4, and A100 40GB**). In the Colab web runtime menu, **`G4` and `H100` are grayed out** under Google's "Premium GPU Access" policy.
* **Gated Tiers:** Direct selection of `G4` is restricted to **Google AI Ultra** (1,000–2,000 CU/month) and legacy **Colab Pro+** (600 CU/month).
* **CLI Behavior:** Attempting `colab new -s worker --gpu G4` on a Google AI Pro account returns an entitlement failure or `ColabRequestError: Service Unavailable` unless a regional stockout causes the allocator to substitute G4.
* **Verdict for Job 01:** A plan relying on dedicated G4 provisioning cannot be reliably executed on a standard Google AI Pro 200 CU account.

### 6. Single-GPU Limitation & Impossibility of Multi-VM `tp=2`

* Standard Google Colab allocates a single VM per session with exactly one virtual accelerator assigned.
* There is no mechanism to bind multiple Colab VMs together across network boundaries with low-latency interconnects (such as NVLink or InfiniBand) to run distributed tensor parallelism (`tp=2`).
* Workloads exceeding 96 GB VRAM (such as `nvidia/GLM-5.3-Flash-NVFP4`, requiring 160–175 GB) cannot be distributed across two Colab G4 instances.

### 7. Colab Wall-Clock Billing Mechanics

* Colab meters Compute Units strictly by the **wall-clock duration of the active connected session**.
* Metering starts the moment `/tun/m/assign` succeeds and stops only when `runtime.unassign()` is called or the instance is terminated via `colab stop`.
* Idle GPU time (e.g., waiting for user prompts, thinking pauses, network latency) is billed at the exact same hourly rate as 100% active CUDA kernel execution.
* For interactive coding sessions with human-in-the-loop idle time, G4 burns 8.71 CU/hr ($0.87/hr) even if the model generates only 500 tokens in that hour.

### 8. NVIDIA L4 Specifications & Analysis of the "16×" Compute Comparison

* **Hardware Specs:** NVIDIA L4 24GB is based on Ada Lovelace (`sm_89`), with 7,680 CUDA cores, 240 4th-Gen Tensor Cores, a 192-bit bus @ 12.5 Gbps delivering **300.05 GB/s**, and a 72W TDP cap.
* **L4 Tensor Performance:**
  * FP8 Tensor Core: **242 TFLOPS** (2:4 structural sparsity) / **121 TFLOPS** (dense).
  * FP16 Tensor Core: **121 TFLOPS** (sparse) / **60.5 TFLOPS** (dense).
  * Native FP4: **None.** Ada Lovelace cannot execute 4-bit floating point natively.
* **Deconstructing the Chat's "16×" Comparison:**
  * The chat states: *"Tensor Compute: ~4,000 TFLOPS FP4 (G4) vs. ~242 TFLOPS FP8 (L4) → 16× higher raw compute for prefill."*
  * This is an invalid apples-to-oranges comparison:
    1. It compares **FP4** on G4 against **FP8** on L4.
    2. It compares **sparse FP4** (4,000 TFLOPS) against **sparse FP8** (242 TFLOPS).
    3. In true like-for-like precision (dense FP8 vs dense FP8):
       $$\frac{1,000 \text{ TFLOPS (G4 Dense FP8)}}{121 \text{ TFLOPS (L4 Dense FP8)}} \approx \mathbf{8.26\times}$$
    4. In sparse FP8 vs sparse FP8:
       $$\frac{2,000 \text{ TFLOPS (G4 Sparse FP8)}}{242 \text{ TFLOPS (L4 Sparse FP8)}} \approx \mathbf{8.26\times}$$
    5. The true prefill compute advantage of G4 over L4 at FP8 precision is **8.26×**, not 16×. The 16× ratio is achieved only if G4 runs NVFP4 and L4 runs FP8.

### 9. Blackwell `sm_120` / `sm_120a` Native NVFP4 Support

Blackwell introduces native hardware execution for block-scaled 4-bit floating point:
* **Format:** Microscaling formats (`NVFP4` / `FP4 E2M1` with 8-element or 16-element block scaling factors in FP8/E8M0).
* **Architecture Flags:** Handled by CUDA Toolkit 13.0+ and NVIDIA Driver 580+.
* **Fatbin Handling:** In CMake builds (such as `llama.cpp`), compiling with `-DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real"` embeds native SASS cubins for both A100 (`sm_80`) and G4 (`sm_120a`) without relying on runtime JIT or bloated PTX.

---

## Independent calculations

### Calculation 1: Theoretical Batch-1 Autoregressive Decode Hardware Ceilings

In batch-1 autoregressive decoding, every generated token requires streaming the active model weights once from VRAM into the Tensor Cores. The physical memory bandwidth ceiling is:

$$\text{Ceiling (tokens/sec)} = \frac{\text{Memory Bandwidth (GB/s)}}{\text{Active Weight Payload (GB/token)}}$$

#### 1. Qwen 3.8 27B Dense NVFP4 on Colab G4 (1,597 GB/s)
* Parameters: 27.8 Billion. In NVFP4 (0.5 bytes/weight + quantization metadata/scales ~0.55 bytes/weight), active payload is **15.3–16.8 GB** (hanxiao measured active weight read at **15.75 GiB = 16.91 GB**).
* **100% Bus Saturation Ceiling:**
  $$\frac{1,597 \text{ GB/s}}{16.91 \text{ GB}} = \mathbf{94.4 \text{ tok/s}} \quad \left(\text{or } \frac{1,597}{15.3} = \mathbf{104.4 \text{ tok/s}}\right)$$
* **Realistic Sustained Decode (65%–72% bus efficiency on GDDR7):**
  $$94.4 \times 0.70 = \mathbf{66.1 \text{ tok/s}} \quad (\text{range: } 61.4\text{–}68.0 \text{ tok/s})$$
* *Finding:* Without speculative decoding or MTP, the chat's claim of 85–105 tok/s for standard batch-1 autoregressive decode on 27B NVFP4 requires **90%–100%+ memory bus efficiency**, which is unachievable under real CUDA runtime overheads. Achieving >100 tok/s requires speculative drafting (MTP / DFlash 2).

#### 2. Qwen 3.8 27B Dense FP8 on Colab G4 (1,597 GB/s)
* Parameters: 27.8 Billion. In FP8 (1.0 byte/weight + scale), active payload is **~27.8 GB**.
* **100% Bus Saturation Ceiling:**
  $$\frac{1,597 \text{ GB/s}}{27.8 \text{ GB}} = \mathbf{57.4 \text{ tok/s}}$$
* **Realistic Sustained Decode (70% efficiency):**
  $$57.4 \times 0.70 = \mathbf{40.2 \text{ tok/s}}$$
* *Fatal Error in Chat Claim:* The chat claims standard autoregressive FP8 decode is **95–105 tok/s**. Streaming 27.8 GB at 100 tok/s requires:
  $$27.8 \text{ GB} \times 100 \text{ tok/s} = \mathbf{2,780 \text{ GB/s}}$$
  This exceeds the G4's 1,597 GB/s bandwidth by **74%**. The chat's FP8 decode claim is physically impossible for batch-1 autoregressive generation without multi-token speculation.

#### 3. Qwen 3.8 Flash-Next NVFP4 on Colab G4 (1,597 GB/s)
* Parameters: 125B MoE with **~6.0B active parameters** per token.
* In NVFP4, active weight payload per token is:
  $$6.0 \times 10^9 \times 0.55 \text{ bytes} \approx \mathbf{3.30 \text{–} 3.60 \text{ GB}}$$
* **100% Bus Saturation Ceiling:**
  $$\frac{1,597 \text{ GB/s}}{3.60 \text{ GB}} = \mathbf{443.6 \text{ tok/s}}$$
* **Realistic Sustained Decode (65% efficiency due to MoE routing scatter):**
  $$443.6 \times 0.65 = \mathbf{288.3 \text{ tok/s}}$$
* With MTP / speculative drafting accepting ~1.3–1.4 tokens per pass, generation reaches **320–380 tok/s**, confirming the physical plausibility of Flash-Next on G4.

#### 4. Qwen 3.8 27B Q4 on NVIDIA L4 (300 GB/s)
* Weight payload: 15.75 GiB = 16.91 GB.
* **100% Bus Saturation Ceiling:**
  $$\frac{300.05 \text{ GB/s}}{16.91 \text{ GB}} = \mathbf{17.74 \text{ tok/s}}$$
* **Empirical Measurement (hanxiao):** **14.95 tok/s** (achieving 84.3% bus efficiency).

---

### Calculation 2: Price Multipliers Between GPU Tiers

Evaluating the hourly price ratio $M = \frac{\text{Rate}_{\text{G4}}}{\text{Rate}_{\text{L4}}}$ across different rate models:

1. **Chat Model:**
   $$M_{\text{chat}} = \frac{6.5 \text{ CU/hr}}{2.1 \text{ CU/hr}} = \mathbf{3.095\times \approx 3.1\times}$$
2. **Standard Documented 2026 Model (Qiita / McCormick / architecture.md):**
   $$M_{\text{standard}} = \frac{8.71 \text{ CU/hr}}{1.71 \text{ CU/hr}} = \mathbf{5.094\times \approx 5.1\times}$$
3. **Peak G4 vs. Peak L4:**
   $$M_{\text{peak}} = \frac{8.71 \text{ CU/hr}}{2.10 \text{ CU/hr}} = \mathbf{4.148\times \approx 4.15\times}$$
4. **Off-Peak G4 vs. Baseline L4:**
   $$M_{\text{off-peak}} = \frac{6.71 \text{ CU/hr}}{1.71 \text{ CU/hr}} = \mathbf{3.924\times \approx 3.9\times}$$

*Finding:* The chat's claim that G4 is only "3.1× more expensive per hour" than L4 is true only if G4 is priced at its minimum off-peak rate while L4 is priced at its peak rate. Under standard baseline pricing, G4 costs **5.1× more per hour** than L4.

---

### Calculation 3: Bandwidth per Compute Unit (A100 40GB vs. G4)

Comparing the cost-efficiency of memory bandwidth delivery between A100-SXM4-40GB and G4:

* **A100-SXM4-40GB:** 1,555 GB/s at 5.37 CU/hr
  $$\frac{1,555 \text{ GB/s}}{5.37 \text{ CU/hr}} = \mathbf{289.57 \text{ GB/s per CU/hr}}$$
* **G4 (RTX PRO 6000 Server Edition) at Standard 8.71 CU/hr:** 1,597 GB/s at 8.71 CU/hr
  $$\frac{1,597 \text{ GB/s}}{8.71 \text{ CU/hr}} = \mathbf{183.35 \text{ GB/s per CU/hr}}$$
* **G4 at Off-Peak 6.71 CU/hr:**
  $$\frac{1,597 \text{ GB/s}}{6.71 \text{ CU/hr}} = \mathbf{238.00 \text{ GB/s per CU/hr}}$$
* **Bandwidth Efficiency Advantage of A100 40GB over G4:**
  $$\frac{289.57}{183.35} = \mathbf{1.579\times} \quad (+57.9\% \text{ more bandwidth delivered per CU})$$

*Finding:* Even though G4 provides 96 GB VRAM and faster FP4 Tensor Cores, A100 40GB delivers **58% more memory streaming per Compute Unit**. For memory-bound batch-1 decoding that fits within 40 GB, A100 40GB remains more cost-efficient than G4.

---

## Conflicts with architecture.md / prior research

1. **G4 Burn Rate (8.71 vs. 6.4–6.7 CU/hr):**
   * `architecture.md` (§4) records G4 at **~8.71 CU/hr**.
   * Prior research (`busy_gpu.md` and `a100_speed_and_session.md`) documented both 8.71 and 6.40–6.74 CU/hr.
   * *Resolution:* The chat's 6.4–6.7 CU/hr is an optimistic figure. Financial plans must assume 8.71 CU/hr as the primary burn rate to prevent unexpected session terminations.
2. **L4 Burn Rate (1.71 vs. 2.1 CU/hr):**
   * `architecture.md` records L4 at **~1.71 CU/hr**.
   * The chat assumes **~2.1 CU/hr**.
   * *Resolution:* Both are valid; 1.71 CU/hr is standard baseline, while 2.09–2.10 CU/hr is common under peak demand or High-RAM allocations.
3. **Usable VRAM Reporting:**
   * The chat states flatly "96 GB VRAM".
   * In guest Linux environments under `nvidia-smi`, usable VRAM is **95.6 GiB (97,896 MiB)**.
4. **Entitlement for Google AI Pro:**
   * The chat assumes the user can switch freely to G4 on their existing subscription.
   * `busy_gpu.md` and user testing confirm that **G4 is not directly selectable on Google AI Pro (200 CU)** and requires an upgrade to Google AI Ultra or Colab Pro+.

---

## What remains unverified

1. **Live Colab Allocator Fallback Probability:** The exact frequency at which an AI Pro request for `A100` or `L4` silently provisions a `G4` during capacity shortages is unquantified; documented community reports are anecdotal.
2. **Thermal Throttling on Passive G4 in Colab Multi-Tenant Hosts:** NVIDIA datasheets specify that the Server Edition relies on chassis airflow. Whether Colab G4 instances experience clock throttling under continuous heavy FP4 prefill loops remains unmeasured without physical session telemetry.
3. **Future Price Adjustments:** Google Colab adjusts CU consumption dynamically without formal advance notice; current rates reflect verified October 2026 operations.

---

## Sources

1. **NVIDIA Official Product Documentation:**
   * [NVIDIA RTX PRO 6000 Blackwell Server Edition Overview](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) — Datasheet specs: passive cooling, 96 GB GDDR7 ECC, 1,597 GB/s bandwidth, 512-bit bus.
   * [NVIDIA Blackwell Architecture Technical Brief](https://www.nvidia.com/en-us/data-center/blackwell-architecture/) — 5th-Gen Tensor Cores, native NVFP4/FP8 micro-scaling formats, 4 PFLOPS sparse / 2 PFLOPS dense FP4.
   * [NVIDIA L4 Tensor Core GPU Datasheet](https://www.nvidia.com/en-us/data-center/l4/) — 24 GB GDDR6, 300 GB/s bandwidth, 72W TDP, 242 TFLOPS sparse FP8.
2. **Hardware Vendor Technical Guides:**
   * [PNY NVIDIA RTX PRO 6000 Blackwell Server Edition Datasheet (PDF)](https://www.pny.com/en-eu/file%20library/professional/datasheet/data%20center%20cards/pny-nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf) — Memory clocks, dimensions, power configuration (400W–600W).
   * [Lenovo ThinkSystem Product Guide: NVIDIA RTX PRO 6000 Blackwell Server Edition (LP2263)](https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu) — Detailed bus clock and thermal airflow requirements.
   * [VRLA Tech: RTX PRO 6000 Blackwell Workstation vs. Server Edition](https://vrlatech.com/rtx-pro-6000-blackwell-workstation-vs-server-edition/) — Clarification of 1,792 GB/s active workstation card vs. 1,597 GB/s passive server card.
3. **Google Colab & Cloud Infrastructure:**
   * [Google Developers Blog: Colab is now part of your Google AI plan](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/) — AI Pro 200 CU/month integration, tier limitations, and GPU access rules.
   * [Chris McCormick: Colab GPUs Features and Pricing (2026 Edition)](https://mccormickml.com/2024/04/23/colab-gpus-features-and-pricing/) — Verified CU burn rates: T4 (1.19), L4 (1.71), A100 40GB (5.40), A100 80GB (7.52), G4 (8.71).
   * [Qiita: Google Colab GPU Pricing and Benchmark Study (2026)](https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f) — Comprehensive CU/hr audit and G4 8.71 CU/hr measurement.
   * [Reddit r/GoogleColab: New G4 Instance Observations](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/) — Community logs of G4 burn rates (6.4–8.7 CU/hr) and availability constraints.
   * [note.com: Google AI Pro A100 Deployment and GPU Selector Audit](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en) — Verification that G4 and H100 are grayed out on Google AI Pro 200 CU accounts.
4. **Software & CUDA Runtime Repositories:**
   * [hanxiao: Qwen 3.8 27B L4 Speculative Decoding & Kernel Study](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) — Active weight read measurements (15.75 GiB), memory bandwidth utilization on L4.
   * [lukaLLM: DFlash Qwen 3.6 / 3.8 27B LlamaCPP Benchmark Repository](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP) — Workstation Edition 1,792 GB/s benchmarks and DFlash 2 speed sweeps.
   * [FlashInfer: Blackwell sm_120 Native Kernels Documentation](https://flashinfer.ai/) — Kernel compilation targets and NVFP4 support on `sm_120`/`sm_120a`.
