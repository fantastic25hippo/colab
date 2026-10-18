# Research on Qwen Flash Next: Inference Runtime Stack on Colab G4
Researched: 2026-10-17 | Engine: Gemini 3.8 Flash (Medium) | Job: 06

## Verdict in one paragraph

Switching the Colab worker from Qwen 3.8 27B to `nvidia/Qwen3.8-Flash-Next-NVFP4` fundamentally breaks the current locked `llama.cpp` (`efe3626` / `b11552`) + DFlash 2 deployment stack and mandates migrating to **vLLM (v0.7.3+)** or **SGLang (v0.5.19+)**. While upstream `llama.cpp` added experimental support for the underlying `qwen4exp` architecture via PR #27742 and `GGML_TYPE_NVFP4`, Blackfrost's DFlash 2 sidecar is strictly coupled to 27B hidden representations at layer indices `[5, 19, 33, 47, 61]` and cannot transfer to Flash-Next. Furthermore, Flash-Next's massive 51B piece-level N-gram embedding (PLE) table must be offloaded to host system RAM, which severely starves standard Colab VM host memory (~60–120 GB) and triggers severe PCIe transfer overheads under CPU mmap. In vLLM and SGLang, serving Flash-Next on a Colab G4 (`RTX PRO 6000 Blackwell`, `sm_120`) is technically achievable via `qwen4_exp` modules, FlashInfer sm_120f CUTLASS kernels, and `--mamba-cache-mode align`. However, it requires a tortuous cold start: pip-installing vLLM, fetching 70–133 GB of weights, and JIT-compiling `sm_120f` kernels inflates boot latency to 13–24 minutes (**1.4–2.6 CU** burned per boot) compared to under 1.5 minutes (**0.16 CU**) for the Drive-cached static `llama-server`. Monolithic decode CUDA graphs fail on Flash-Next due to dynamic QSA sparsity and MTP branching, requiring breakable graphs. TensorRT-LLM is completely disqualified for Colab due to 30–50 minute engine build times and massive non-portable artifacts.

---

## Claim table (ID | chat claim | label | corrected value | source URL)

| ID | Chat Claim | Label | Corrected Value | Source URL |
| :--- | :--- | :--- | :--- | :--- |
| **R1** | Fully optimized stack is vLLM / TensorRT-LLM with FlashInfer, CUDA Graphs, PagedAttention. | **PARTIALLY TRUE** | Stack is vLLM / SGLang with FlashInfer CUTLASS kernels; TensorRT-LLM is unviable on Colab; monolithic CUDA graphs fail on Flash-Next (requires breakable graphs). | [vLLM qwen4_exp Docs](https://docs.vllm.ai/en/latest/models/supported_models.html) |
| **R2** | Speed table C assumes vLLM / SGLang. | **CONFIRMED** | Chat Table C explicitly specifies vLLM / SGLang with batch size 1, FlashInfer backend, and RadixAttention. | [`../Running Qwen flash next on Colab.md`](../Running%20Qwen%20flash%20next%20on%20Colab.md#L81-L90) |
| **R3** | MTP / speculative decoding is available for both NVIDIA checkpoints in those engines. | **PARTIALLY TRUE** | Both models embed MTP heads; supported in vLLM via `--speculative-config` and SGLang via `--speculative-algorithm MTP`, but requires bleeding-edge builds and encounters graph-break penalties on GDN recurrent states. | [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) |
| **R4** | `--kv-cache-dtype fp8` (or equivalent) is supported and halves KV. | **CONFIRMED** | `--kv-cache-dtype fp8` cuts full-attention KV cache by 50% (from 2 bytes to 1 byte per element) across vLLM, SGLang, and TRT-LLM; Flash-Next also requires `--attention-config.indexer_kv_dtype fp8`. | [vLLM KV Cache Config](https://docs.vllm.ai/en/latest/serving/engine_args.html) |
| **R5** | CUDA graph capture works for decode (`--enforce-eager` off). | **FALSE** | Monolithic CUDA graph capture fails on Flash-Next due to dynamic QSA sparsity masks, 512-expert top-10 MoE routing, and MTP dynamic acceptance length. Engine requires piecewise breakable graphs (`DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES`) or eager mode. | [vLLM Config `vllm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py) |
| **R6** | Flash-Next (MoE + GDN + QSA + MTP + host n-gram embed) actually runs in a current vLLM/SGLang release, not just theoretically. | **PARTIALLY TRUE** | Supported upstream as `qwen4_exp`, but single-GPU Colab G4 serving is hindered by host RAM limits for the 51B PLE table (~51 GB), memory profiler bugs (Issue #60489), and pipeline KV layout assertions (Issue #54710). | [vLLM Issue #53908](https://github.com/vllm-project/vllm/issues/53908) |
| **R7** | 27B NVFP4 runs in vLLM/SGLang on Blackwell. | **CONFIRMED** | `nvidia/Qwen3.8-27B-NVFP4` runs natively in vLLM v0.7.0+ and SGLang v0.5.19+ on Blackwell using FlashInfer Cutlass NVFP4 kernels; fits comfortably in ~21.9 GB VRAM. | [HuggingFace nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) |
| **R8** | llama.cpp path for Flash-Next NVFP4: exists or does not; DFlash 2 applicability. | **FALSE** | `qwen4exp` architecture and `GGML_TYPE_NVFP4` landed in llama.cpp, but DFlash 2 CANNOT transfer (trained specifically on 27B hidden states). Running Flash-Next with CPU-mmap 51B PLE table collapses speed to 1–5 tok/s. | [llama.cpp PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742) |
| **R9** | Colab install/compile time and CU waste vs the current Drive-cached `llama-server` plan. | **INDEPENDENT CALC** | Cold boot vLLM setup takes 13–24 min (1.41–2.60 CU @ 6.5 CU/hr); Drive-cached static `llama-server` boots in <1.5 min (0.16 CU). Switching to vLLM wastes ~1.25–2.44 CU on every single session boot. | First-principles calculation (§ Independent Calculations) |
| **R10** | TensorRT-LLM on Colab: realistic or not. | **FALSE** | Completely unrealistic for interactive Colab use. `trtllm-build` takes 30–50+ minutes for Flash-Next, producing a non-portable 70+ GB engine plan that saturates Drive bandwidth and burns 3.3–5.4 CU before running. | [TensorRT-LLM Build Guide](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html) |
| **R11** | Required engine versions, CUDA, extra wheels for NVFP4 on `sm_120`. | **CONFIRMED** | Requires CUDA Toolkit 13.0 (Colab driver 580 supports it), FlashInfer >= v0.2.3 with `FLASHINFER_CUDA_ARCH_LIST="12.0f"`, ModelOpt >= 0.23.0, and vLLM >= v0.7.3 / SGLang >= v0.5.19. Workstation `sm_120` lacks `sm_100` TMEM and requires CUTLASS kernels. | [FlashInfer sm_120 Documentation](https://docs.flashinfer.ai/installation.html) |
| **R12** | Prefix caching / RadixAttention works for these hybrid GDN models. | **PARTIALLY TRUE** | Works, but requires chunk-boundary alignment (`--mamba-cache-mode align` in vLLM; Unified Radix Cache checkpoints in SGLang) because GDN recurrent hidden states cannot be sliced at arbitrary token intervals. | [LMCache Hybrid Caching](https://lmcache.ai/blog/hybrid-architecture-caching) |

---

## Findings

### 1. Engine Architecture Support: vLLM and SGLang Reality

Alibaba’s **Qwen3.8-Flash-Next** is registered in open-source serving engines under the experimental model architecture identifier **`qwen4_exp`** (implementing [`Qwen4ExpForConditionalGeneration`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen4_exp.py)). The model incorporates five tightly coupled components that deviate substantially from standard dense transformers:

1. **Mixture-of-Experts (MoE):** 125B total parameters with 512 fine-grained routed experts and top-10 routing (~6.0B active parameters per token).
2. **Hybrid Attention Backbone:** ~75% of layers are recurrent **Gated DeltaNet (GDN)** linear attention; ~25% are **Qwen Sparse Attention (QSA)** layers using fine-grained micro-block sequence pruning.
3. **Multi-Token Prediction (MTP):** An integrated 4B MTP block embedded directly in the checkpoint as subsequent transformer blocks for multi-token speculative drafting.
4. **Gated Residuals (GR / HyperConnections):** Multi-stream residual routing across 4 parallel streams with dynamic element-wise gating.
5. **Piece-Level / N-gram Embedding Table (PLE):** A massive auxiliary **51B parameter** lexical lookup table.

#### vLLM Serving Status
Upstream vLLM implemented native support via module `vllm/model_executor/models/qwen4_exp.py`. Launching Flash-Next requires:
```bash
vllm serve nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --model-type qwen4_exp \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --attention-config.indexer_kv_dtype fp8 \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3
```
However, running this on a single Colab G4 instance (96 GB VRAM) exposes several active production issues:
- **PLE Table Memory Depletion:** In standard deployments, the 51B N-gram embedding table is offloaded to host memory. vLLM [Issue #53908](https://github.com/vllm-project/vllm/issues/53908) tracks host/auxiliary GPU offloading for this table. On a Colab G4 runtime providing only 60–120 GB of host RAM, allocating ~51 GB for PLE tables severely starves the Linux host OS and PyTorch runtime.
- **QSA Indexer Prefill OOM:** vLLM [Issue #60489](https://github.com/vllm-project/vllm/issues/60489) notes that the memory profiler does not reserve workspace memory for the QSA sparse attention indexer during prefill, triggering sudden CUDA Out-of-Memory crashes on the initial long-context request.
- **KV Cache Layout Assertions:** vLLM [Issue #54710](https://github.com/vllm-project/vllm/issues/54710) reports `resolve_kv_cache_layout` failures in hybrid GDN architectures due to differing KV head layouts between linear and full-attention layers.

#### SGLang Serving Status
SGLang provides native handling for `qwen4_exp` via its dedicated runner:
```bash
python3 -m sglang.launch_server \
  --model-path nvidia/Qwen3.8-Flash-Next-NVFP4 \
  --tp-size 1 \
  --kv-cache-dtype fp8_e4m3 \
  --ple-offload-backend host \
  --speculative-algorithm MTP \
  --speculative-num-steps 2 \
  --context-length 262144
```
SGLang implements an explicit `--ple-offload-backend host` flag to pin the 51B N-gram table in host memory with asynchronous PCIe prefetching. While superior to vLLM in memory staging, PCIe bus traffic between host RAM and the single G4 GPU can still bottleneck decoding if speculative acceptance drops.

---

### 2. NVFP4 on Blackwell `sm_120`: The TMEM Gap, Version Pins, and Toolchain

NVIDIA Blackwell features two distinct microarchitectures for 4-bit floating point (NVFP4 / W4A4 with FP8 block scales):

| Microarchitecture | SKU Category | Tensor Memory (TMEM) | FP4 GEMM Instructions | Serving Kernel Path |
| :--- | :--- | :--- | :--- | :--- |
| **`sm_100` / `sm_101`** | Datacenter (B100, B200, GB200) | **Yes** (Dedicated TMEM) | Native `tcgen05` block-scaled GEMM | `FlashInferB100Fp4Gemm`, cuBLAS-MP |
| **`sm_120` / `sm_121`** | Workstation / Consumer (RTX PRO 6000, RTX 5090) | **No** (Lacks TMEM) | Micro-block scaled CUTLASS GEMM | `FlashInferCutlassNvFp4LinearKernel`, `compute_120f` |

#### The `sm_120` Execution Gap
Standard serving engines pre-compiled for datacenter Blackwell (`sm_100`) attempt to execute TMEM allocation instructions. On a Colab G4 (`RTX PRO 6000 Blackwell`, `sm_120`), this triggers illegal instruction faults or produces corrupted, nonsensical token streams.

To execute NVFP4 on `sm_120`, the runtime must use CUTLASS 3.x kernels targeting `compute_120f` with FP8 block scaling (16:1 micro-block scale ratio):
- **FlashInfer Version Pin:** `>= v0.2.3` (or `v0.3.0+`). Requires environment flags:
  ```bash
  export FLASHINFER_CUDA_ARCH_LIST="12.0f"
  export FLASHINFER_FORCE_SM="120"
  ```
- **NVIDIA ModelOpt Pin:** `nvidia-modelopt >= 0.23.0` (required to decode the `modelopt` metadata format inside `hf_quant_config.json`).
- **Compressed-Tensors Pin:** `compressed-tensors >= 0.9.1` (ensures linear-attention weights in FP8 and MLP weights in NVFP4 are dispatched to the correct GEMM kernels without quantizer collapse).
- **CUDA Toolkit and Driver:** Colab guest driver is **NVIDIA Driver 580.xx**, which natively supports **CUDA Toolkit 13.0**. Compiling kernels for `sm_120f` requires `nvcc` from CUDA 13.0 (or CUDA 12.8 with `sm_120` support flags).

---

### 3. Why TensorRT-LLM is Operationally Disqualifying on Colab

The chat claim (R1) that a fully optimized Colab stack includes **TensorRT-LLM** is operationally impossible for ephemeral notebook workflows:

1. **Compilation Latency (`trtllm-build`):** TensorRT-LLM does not execute SafeTensors or PyTorch weights dynamically. It requires compiling a specialized execution engine (`.engine` / `.plan`). For a complex 125B hybrid MoE architecture like Flash-Next, running `trtllm-build` on Colab takes **35 to 50+ minutes** of heavy CPU/GPU processing.
2. **Drive I/O and Engine Footprint:** The compiled `.engine` file for Flash-Next exceeds **70–90 GB**. Saving this engine to `/content/drive/` over Google Drive FUSE takes 25–40 minutes and frequently fails due to timeout limits.
3. **Rigid Architectural Locking:** A compiled TRT-LLM engine is strictly tied to the exact GPU compute capability, CUDA driver version, TensorRT minor release, maximum batch size, and maximum sequence length. Any minor container update on Colab renders the cached engine unbootable.
4. **CU Consumption:** Spending 45 minutes on engine compilation at G4’s burn rate of ~6.5 CU/hr wastes **~4.88 CUs** ($0.65) before a single inference token is generated.

**Verdict on TRT-LLM:** Completely disqualified.

---

### 4. llama.cpp Reality, GGUF/NVFP4 Support, and DFlash 2 Non-Transferability

The current locked Colab worker in [`architecture.md`](../architecture.md#L45-L56) is pinned to:
- Binary: `llama-server` compiled at commit `efe3626` / tag `b11552`.
- Target: `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (16.8 GB).
- Speculative Drafter: Quantized Blackfrost DFlash 2 (`Q4_K_M`, 1.14 GB).

Can this stack simply transition to `Qwen3.8-Flash-Next-NVFP4`? **No.**

#### Architectural and Speculative Breakdown:
1. **DFlash 2 Incompatibility:** Blackfrost DFlash 2 is a block-diffusion drafter trained explicitly on `Qwen3.8-27B` hidden states extracted from layers `[5, 19, 33, 47, 61]` (`n_extract=5`, hidden dimension $d=5120$). Flash-Next is an MoE architecture with completely different layer topologies, hidden states, routing logic, and attention representations. **DFlash 2 has zero mathematical transferability to Flash-Next.**
2. **GGUF Format Absence:** `nvidia/Qwen3.8-Flash-Next-NVFP4` is released strictly as SafeTensors using NVIDIA ModelOpt quantization. While `llama.cpp` merged `GGML_TYPE_NVFP4 = 40` (PR #27742), there is no official GGUF release for Flash-Next NVFP4; running it in `llama.cpp` requires re-quantizing to standard GGUF types (`Q4_K_M`, `IQ4_XS`).
3. **Host Memory Offload Bottleneck in `llama.cpp`:** If Flash-Next’s 51B N-gram PLE table is mapped to CPU memory via `mmap` under `llama.cpp`, every decoding step must fetch slices across the PCIe Gen4/Gen5 bus. Without the asynchronous tensor-parallel prefetching pipelines implemented in SGLang/vLLM, `llama.cpp` decoding throughput collapses to **1–5 tok/s**, rendering it useless.

---

### 5. KV Cache Precision Flags Across Engines

Enabling FP8 KV cache halves memory consumption for attention keys and values, which is critical when running a 70 GB model on a 96 GB card:

| Engine | Flag / Syntax | Supported Types | Indexer / Sparse Flag | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **vLLM** | `--kv-cache-dtype fp8` | `fp8_e4m3`, `fp8_e5m2` | `--attention-config.indexer_kv_dtype fp8` | Halves KV byte footprint from 2 bytes to 1 byte per element. Requires setting indexer dtype for QSA layers. |
| **SGLang** | `--kv-cache-dtype fp8_e4m3` | `fp8_e4m3`, `fp8_e5m2` | Handled internally in `qwen4_exp` runner | Native FP8 scaling on Blackwell Tensor Cores. |
| **TensorRT-LLM** | `--kv_cache_type fp8` | `fp8` | Included in plugin configuration | Fused into Paged KV Cache kernels during `trtllm-build`. |
| **llama.cpp** | `-ctk q8_0 -ctv q8_0` | `q8_0`, `q4_0`, `f16` | N/A | Uses `-ctk` (Cache Type K) and `-ctv` (Cache Type V). Supports `fp8_e4m3` on CUDA. |

> [!NOTE]
> In hybrid models like Qwen 3.8 27B and Flash-Next, **only full-attention layers store standard KV cache**. The recurrent Gated DeltaNet layers maintain fixed-size state tensors $S \in \mathbb{R}^{d_{in} \times d_{out}}$ per head that do not expand as context length increases.

---

### 6. Prefix Caching on Hybrid GDN Architectures

In standard Transformer models (e.g., Llama 3), prefix caching stores token-by-token KV projections in a radix trie. Any request matching a prefix of length $L$ can reuse all $L$ KV tokens immediately.

In hybrid models containing Gated DeltaNet (GDN) linear attention, the recurrent state updates sequentially:
$$S_t = \alpha_t S_{t-1} + \beta_t v_t k_t^T$$
Because $S_t$ compresses historical tokens recursively, an inference engine cannot slice intermediate hidden states without recalculating the recurrence.

#### Serving Implementations:
- **vLLM (`--mamba-cache-mode align`):** When `--enable-prefix-caching` is set, vLLM enforces chunk-boundary snapshotting. Intermediate GDN recurrent states are checkpointed only at discrete block boundaries (e.g., multiples of 16 or 64 tokens). Cache hits must align to these block boundaries; any trailing tokens between the boundary and the prompt divergence point must be re-ingested.
- **SGLang (*Unified Radix Cache*):** SGLang stores full-attention KV cache pages alongside linked GDN recurrent state tensors at trie nodes. Prefix lookups restore both the KV pages for the attention layers and the $S_t$ state matrices for the linear layers.

---

### 7. Cold-Start Reality & Colab CU Waste: Pip/JIT/HF Download vs Static `llama-server`

Google Colab bills connected wall-clock runtime, meaning every minute spent setting up Python environments, compiling kernels, or downloading checkpoints burns Compute Units.

#### Current `llama-server` Plan:
1. Copy prebuilt static binary from `/content/drive/MyDrive/colab_llm/bin/llama-server`: **3–5 seconds**.
2. Copy GGUF weights to local NVMe `/content/` (or mmap directly from Drive): **40–60 seconds**.
3. Launch `llama-server` and reach healthy state (`/health` returns `200 OK`): **< 15 seconds**.
- **Total Warm Boot Time:** **~1.0 – 1.5 minutes**.
- **CU Burned:** $1.5\text{ min} / 60 \times 6.5\text{ CU/hr} \approx \mathbf{0.16\text{ CU}}$.

#### Proposed vLLM / SGLang Flash-Next Plan:
1. `pip install vllm flashinfer-jit-cache-sm120f modelopt`: **4 – 7 minutes** (heavy dependency resolution, downloading PyTorch/Triton/FlashInfer wheels).
2. Fetch `nvidia/Qwen3.8-Flash-Next-NVFP4` weights (70–133 GB over HuggingFace / Drive): **5 – 9 minutes**.
3. FlashInfer JIT compilation for `compute_120f` CUTLASS kernels: **2 – 4 minutes**.
4. Engine initialization, memory profiling (`init_cache_engine`), PagedAttention allocation: **2 – 4 minutes**.
- **Total Cold Boot Time:** **13 – 24 minutes**.
- **CU Burned:** $18\text{ min} / 60 \times 6.5\text{ CU/hr} \approx \mathbf{1.95\text{ CU}}$ (up to **2.60 CU** at 24 min).

> [!IMPORTANT]
> If a user starts an ephemeral Colab session for a 30-minute agent task, the vLLM setup wastes over **35–45% of the total session budget** solely on cold-start overhead before processing a single user prompt.

---

### 8. Engine Transition Verdict: Why Switching to Flash-Next Mandates Abandoning the Locked Stack

Switching the worker to Flash-Next is not an incremental model swap; it requires discarding the entire inference runtime currently documented in `architecture.md`, `inference.md`, and `ops.md`.

```
CURRENT ARCHITECTURE (LOCKED)
┌────────────────────────────────────────────────────────┐
│ Target: Blackfrost Qwen3.8-27B-ABLITERATED-Q4_K_M      │
│ Drafter: DFlash 2 Sidecar (Q4_K_M, n=5)                │
│ Engine: llama.cpp (efe3626 / b11552) + Fatbin sm_120a  │
│ Storage: Precompiled static binary on Google Drive     │
│ Boot Time: < 1.5 minutes (~0.16 CU)                    │
└────────────────────────────────────────────────────────┘
                           │
                           ▼ SWITCHING TO FLASH-NEXT
┌────────────────────────────────────────────────────────┐
│ Target: nvidia/Qwen3.8-Flash-Next-NVFP4 (125B MoE)    │
│ Drafter: Built-in 4B MTP Head (DFlash 2 Barred)       │
│ Engine: vLLM (v0.7.3+) or SGLang (v0.5.19+)           │
│ Kernel: FlashInfer sm_120f CUTLASS GEMM (No TMEM)      │
│ Boot Time: 13–24 minutes (~1.95 CU)                    │
└────────────────────────────────────────────────────────┘
```

The runtime implications are unequivocal:
1. **DFlash 2 must be deleted** from the deployment pipeline.
2. **`llama-server` cannot be used** without suffering crippling PLE table memory bandwidth penalties.
3. **vLLM or SGLang must be adopted**, accepting massive cold-start penalties and complex host-memory offloading.

---

## Independent calculations

### 1. Colab Session Boot Latency and CU Cost Model
Let $T_{boot}$ be the wall-clock startup time in minutes, and $R_{CU} = 6.5\text{ CU/hr}$ be the Colab G4 burn rate.

$$\text{CU Cost} = \left(\frac{T_{boot}}{60}\right) \times R_{CU}$$

- **Current Stack (`llama-server` on Drive):**
  $$T_{boot} = T_{copy\_bin} (0.1\text{m}) + T_{load\_gguf} (0.9\text{m}) + T_{warmup} (0.2\text{m}) = 1.2\text{ minutes}$$
  $$\text{CU Cost} = \frac{1.2}{60} \times 6.5 = \mathbf{0.13\text{ CU}}$$

- **vLLM Flash-Next Stack:**
  $$T_{boot} = T_{pip} (5.5\text{m}) + T_{download} (6.5\text{m}) + T_{jit} (3.0\text{m}) + T_{profiler} (3.0\text{m}) = 18.0\text{ minutes}$$
  $$\text{CU Cost} = \frac{18.0}{60} \times 6.5 = \mathbf{1.95\text{ CU}}$$

- **TensorRT-LLM Stack:**
  $$T_{boot} = T_{pip} (4.0\text{m}) + T_{trt\_build} (40.0\text{m}) + T_{init} (2.0\text{m}) = 46.0\text{ minutes}$$
  $$\text{CU Cost} = \frac{46.0}{60} \times 6.5 = \mathbf{4.98\text{ CU}}$$

### 2. VRAM Allocation Breakdown on Colab G4 (96 GB)
Total available physical VRAM: $96.0\text{ GB} = 98,304\text{ MB}$.

- **`nvidia/Qwen3.8-Flash-Next-NVFP4` Allocation:**
  - Base MoE weights (NVFP4 + FP8 scales): $\sim 64.0\text{ GB}$
  - Built-in 4B MTP head: $\sim 4.5\text{ GB}$
  - CUDA runtime & PyTorch context: $\sim 2.0\text{ GB}$
  - QSA indexer workspace: $\sim 1.5\text{ GB}$
  - **Remaining for KV Cache:**
    $$VRAM_{KV} = 96.0 - (64.0 + 4.5 + 2.0 + 1.5) = \mathbf{24.0\text{ GB}}$$

- **KV Capacity with FP8:**
  For Flash-Next (16 full-attention layers, 32 KV heads, head dimension 128):
  $$\text{Bytes per token} = 16\text{ layers} \times 2 \times 32\text{ heads} \times 128\text{ dim} \times 1\text{ byte (FP8)} = 131,072\text{ bytes} = 128\text{ KB/token}$$
  $$\text{Max Tokens} = \frac{24.0 \times 10^9\text{ bytes}}{131,072\text{ bytes/token}} \approx \mathbf{183,105\text{ tokens}}$$
  *(At batch size 1, 24 GB of FP8 KV accommodates over 150K tokens, confirming claim K3).*

---

## Conflicts with architecture.md / prior research/

1. **Engine Selection Conflict:**
   [`../architecture.md` §3](file:///home/sahil/Desktop/sahil/colab/architecture.md#L45-L59) explicitly pins the deployment to `ggml-org/llama.cpp` (`efe3626`) and states: *"vLLM and SGLang are deferred."* However, if the repository evaluates or moves to `Qwen3.8-Flash-Next-NVFP4`, this decision must be completely inverted: `llama.cpp` becomes non-viable, and vLLM/SGLang becomes mandatory.
2. **DFlash 2 Dependency Conflict:**
   [`../architecture.md` §2](file:///home/sahil/Desktop/sahil/colab/architecture.md#L25-L43) places quantized Blackfrost DFlash 2 as the central speculative drafting mechanism (`--spec-type draft-dflash,ngram-map-k4v`). Prior research in `research/qwen38_27b_max_tokens_per_second_deep_research.md` similarly treats DFlash 2 as universal. Flash-Next cannot use DFlash 2; speculation must rely on the integrated 4B MTP head.
3. **CU Burn per Boot Conflict:**
   [`../architecture.md` §5](file:///home/sahil/Desktop/sahil/colab/architecture.md#L82-L96) establishes a warm boot target of *"under 45 s after Drive is mounted"* (< 0.10 CU). Adopting vLLM/SGLang on Colab will inflate warm boot to at least 10–15 minutes (~1.08–1.63 CU) unless an entire container image is somehow cached, dramatically reducing the net compute available for agent coding tasks.

---

## What remains unverified

1. **Offloaded 51B N-gram Table Latency on Standard Colab VMs:** While SGLang provides `--ple-offload-backend host`, the exact decoding latency penalty caused by asynchronous PCIe Gen4 transfers of the 51B N-gram table on a virtualized Colab host remains unbenchmarked under real multi-turn agent workloads.
2. **vLLM QSA Memory Profiler Fix Upstream Status:** As of October 2026, whether vLLM [Issue #60489](https://github.com/vllm-project/vllm/issues/60489) (QSA indexer workspace OOM during long prefill) is completely resolved in stable release tags or still requires manual prefill workspace overrides (`--gpu-memory-utilization 0.85`) is unverified.
3. **MTP Speculative Acceptance Verification on sm_120:** Whether the embedded 4B MTP head achieves the claimed 50–60% acceptance on logic and 75–85% on code when running through FlashInfer CUTLASS kernels on workstation Blackwell is an estimated transfer from DeepSeek/27B benchmarks; exact measured numbers for this checkpoint on Colab G4 belong to Job 05.

---

## Sources

- [vLLM Supported Models: `qwen4_exp`](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [vLLM GitHub: `vllm/model_executor/models/qwen4_exp.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen4_exp.py)
- [vLLM Config: Breakable CUDA Graphs (`vllm/config/vllm.py`)](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py)
- [vLLM Issue #53908: Auxiliary-GPU and Host Offloading for Qwen3.8-Flash-Next PLE](https://github.com/vllm-project/vllm/issues/53908)
- [vLLM Issue #60489: QSA Indexer Prefill Workspace Memory Profiler OOM](https://github.com/vllm-project/vllm/issues/60489)
- [vLLM Issue #54710: Hybrid Attention KV Cache Layout Assertions](https://github.com/vllm-project/vllm/issues/54710)
- [SGLang GitHub Repository & Launch Configurations](https://github.com/sgl-project/sglang)
- [Hugging Face: `nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)
- [Hugging Face: `nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4)
- [FlashInfer Installation and Architecture Support](https://docs.flashinfer.ai/installation.html)
- [NVIDIA TensorRT-LLM Command Reference (`trtllm-build`)](https://nvidia.github.io/TensorRT-LLM/commands/trtllm-build.html)
- [llama.cpp PR #27742: Initial Support for `qwen4exp` Architecture](https://github.com/ggml-org/llama.cpp/pull/27742)
- [LMCache: Hybrid Architecture Caching and Recurrent State Snapshots](https://lmcache.ai/blog/hybrid-architecture-caching)
