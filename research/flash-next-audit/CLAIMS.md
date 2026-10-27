# Numbered claims from `Running Qwen flash next on Colab.md`

Extracted 2026-10-17. Source is a Gemini chat, not a measured sitting.
Job 01–09 each own a slice. Job 10 grades the whole ledger.

The chat mixes three different 27B speed tables (NVFP4 “optimized stack”,
FP8, then NVFP4-vs-Flash-Next). Treat those as **separate claim groups**,
not one consistent model.

---

## H — Hardware and Colab billing (Job 01)

| ID | Claim |
| :--- | :--- |
| H1 | Colab “NVIDIA G4” is the NVIDIA RTX PRO 6000 Blackwell. |
| H2 | That card has **96 GB** VRAM. |
| H3 | Memory bandwidth is **1,597 GB/s** GDDR7. |
| H4 | FP4 tensor compute is **4 PFLOPS**. |
| H5 | FP8 tensor compute is **~1,000+ TFLOPS**. |
| H6 | Colab G4 burns **~6.4–6.7 CU/hr** (chat also uses **~6.5**). |
| H7 | Colab L4 burns **~2.1 CU/hr**. |
| H8 | G4 hourly price is **~3.1×** L4. |
| H9 | L4 has **24 GB** VRAM and **300 GB/s** bandwidth (**5.3×** slower decode streaming than G4). |
| H10 | L4 FP8 tensor compute is **~242 TFLOPS** (**16×** below G4 FP4). |
| H11 | Standard Colab is **single-GPU**; you cannot tensor-parallel two G4/A100 VMs. |
| H12 | Colab bills **connected wall-clock**, including idle GPU time, not tokens. |
| H13 | Colab AI Pro / this account can actually assign a G4. |
| H14 | G4 is Blackwell `sm_120` with native NVFP4 tensor cores. |
| H15 | Existing repo figure **~8.71 CU/hr** for G4 is wrong, or the chat’s **6.4–6.7** is wrong. Resolve the conflict with [`../27b-plan/architecture.md`](../27b-plan/architecture.md). |
| H16 | Existing repo figure **1,792 GB/s** is the Workstation card, **1,597 GB/s** is Colab Server Edition. Confirm which SKU Colab G4 is. |

---

## F — Qwen3.8-Flash-Next-NVFP4 (Job 02)

| ID | Claim |
| :--- | :--- |
| F1 | Hugging Face repo `nvidia/Qwen3.8-Flash-Next-NVFP4` exists and is an NVIDIA NVFP4 checkpoint. |
| F2 | It is a **125B** Mixture-of-Experts model. |
| F3 | **~6.0B** parameters activate per token. |
| F4 | **512** fine-grained experts, **top-10** routing. |
| F5 | Uses Gated DeltaNet. |
| F6 | Uses Qwen Sparse Attention (QSA) that prunes sequence regions at long context. |
| F7 | Has a **4B MTP** (multi-token prediction) head. |
| F8 | Has a **51B n-gram embedding** asynchronously prefetched from **host memory**. |
| F9 | NVFP4 weights + MTP head occupy **~68–72 GB** VRAM. |
| F10 | That leaves **~24–26 GB** on a 96 GB G4 for KV cache. |
| F11 | It is a different architecture from dense Qwen 3.8 27B, not a “faster 27B”. |
| F12 | Flash-Next is the fastest model that still fits on one Colab G4. |
| F13 | File sizes, `config.json` fields, and download footprint match F2–F9. |
| F14 | License / gating / Colab-download practicality (HF token, size, time). |

---

## D — Qwen3.8-27B-NVFP4 (Job 03)

| ID | Claim |
| :--- | :--- |
| D1 | Hugging Face repo `nvidia/Qwen3.8-27B-NVFP4` exists. |
| D2 | Dense **27.8B** (chat also says 27B); **all** parameters activate every token. |
| D3 | Hybrid **48 Gated DeltaNet + 16 full-attention** layers. |
| D4 | NVFP4 occupies **~15–17 GB** VRAM. |
| D5 | FP8 occupies **~27 GB**. |
| D6 | FP16/BF16 occupies **~54 GB**. |
| D7 | Built-in MTP draft head exists on this checkpoint. |
| D8 | Leaves **~80 GB** free on a 96 GB G4 for KV. |
| D9 | How this NVIDIA NVFP4 checkpoint differs from Blackfrost `Qwen3.8-27B-ABLITERATED` GGUF (quant, abliteration, MTP vs DFlash 2). |
| D10 | File sizes and `config.json` match D2–D7. |
| D11 | Unquantized BF16 decode on G4 would be **~50–60 tok/s** (half of FP8). |
| D12 | Download size and Colab-load time. |

---

## G — GLM-5.3-Flash-NVFP4 (Job 04)

| ID | Claim |
| :--- | :--- |
| G1 | Hugging Face repo `nvidia/GLM-5.3-Flash-NVFP4` exists. |
| G2 | **320B** MoE, **18B** active, **288** experts. |
| G3 | NVFP4 weight footprint **160–175 GB** (`320B × 0.5 bytes`). |
| G4 | Does **not** fit on 96 GB G4 (short by **70–80 GB**). |
| G5 | Colab cannot pool two G4s with `tp=2`. |
| G6 | CPU offload of ~80 GB would drop speed to **1–3 tok/s**. |
| G7 | Needs ≥ **2× 96 GB** GPUs or a B200/GB200 node. |
| G8 | API price “around **$0.045** per complex agent task” via Z.ai / OpenRouter. |

---

## S — Speed, roofline, MTP, cache (Job 05)

Chat table A — “fully optimized NVFP4 27B stack” (vLLM/TRT-LLM, FlashInfer, CUDA Graphs, PagedAttention):

| ID | Claim |
| :--- | :--- |
| S1 | 27B NVFP4 batch-1 AR decode **~85–105 tok/s**. |
| S2 | 27B NVFP4 with speculative / MTP=3 **~120–145 tok/s**. |
| S3 | Hardware ceiling ~**100–110 tok/s** from streaming **15–19 GB** over 1,597 GB/s. |
| S4 | Prefill **~5,500–7,500 tok/s**. |
| S5 | TTFT 1K–2K **< 0.10 s**; 20K–25K **~3.0–3.5 s**. |
| S6 | Concurrency 4 **~340–360** aggregate tok/s; concurrency 16 **~850–950**. |
| S7 | On RTX PRO 6000, **FP8 decode is ~5–10% faster than NVFP4**; NVFP4 is level or slightly ahead on heavy prefill. |

Chat table B — 27B **FP8**:

| ID | Claim |
| :--- | :--- |
| S8 | Prefill <4K **~4,500–4,800 tok/s**; 16K–32K **~1,500–2,200 tok/s**. |
| S9 | TTFT ~2K tokens **0.2–0.4 s**. |
| S10 | Thinking AR **~95–105**; MTP **~110–125** tok/s. |
| S11 | Output AR **~95–105**; MTP **~135–160+** tok/s. |
| S12 | Thinking MTP acceptance **50–60%**; code MTP acceptance **75–85%+**. |
| S13 | FP16/BF16 decode **~50–60 tok/s**. |

Chat table C — NVFP4 27B vs Flash-Next, vLLM/SGLang, batch 1:

| ID | Claim |
| :--- | :--- |
| S14 | 27B prefill <4K **~6,000–7,500**; >16K **~2,500–3,500** tok/s. |
| S15 | Flash-Next prefill <4K **~7,500–9,500**; >16K **~4,000–5,500** tok/s. |
| S16 | 27B thinking **~120–135** (or **~95–100** without MTP). |
| S17 | Flash-Next thinking **~220–260** tok/s. |
| S18 | 27B code output **~150–175** tok/s. |
| S19 | Flash-Next code output **~320–380+** tok/s. |
| S20 | Flash-Next streams **~3.5–4 GB** active weights/step vs **~15 GB** for 27B. |
| S21 | QSA keeps Flash-Next long-context prefill nearly **2×** the dense model. |
| S22 | Flash-Next is **~2×** faster reasoning and **~2.5×** faster code than 27B. |

Cache and later economics numbers:

| ID | Claim |
| :--- | :--- |
| S23 | Prefix-cache **hit** bind rate **~100,000–200,000 tok/s**. |
| S24 | Decode dominates: 1M output @ 300–350 tok/s is **~47.6–55.5 min**. |
| S25 | 2M cache-miss prefill @ 7,500–9,500 tok/s is **~3.5–4.4 min**. |
| S26 | 37M cache-hit bind @ 100k–200k tok/s is **~3.1–6.2 min**. |
| S27 | Independent roofline: `tok/s ≈ bandwidth / bytes_moved_per_token` for batch-1 decode. Recalculate for 27B NVFP4, 27B FP8, Flash-Next 6B-active NVFP4. Include overhead (KV, MoE routing, embeddings, MTP). |
| S28 | Published measured benchmarks on RTX PRO 6000 / Colab G4 for these exact checkpoints. Prefer measured over estimated. |

Tables A, B, C **disagree** with each other on 27B prefill and decode. Call that out.

---

## R — Runtime stack (Job 06)

| ID | Claim |
| :--- | :--- |
| R1 | Fully optimized stack is **vLLM / TensorRT-LLM** with FlashInfer, CUDA Graphs, PagedAttention. |
| R2 | Speed table C assumes **vLLM / SGLang**. |
| R3 | MTP / speculative decoding is available for both NVIDIA checkpoints in those engines. |
| R4 | `--kv-cache-dtype fp8` (or equivalent) is supported and halves KV. |
| R5 | CUDA graph capture works for decode (`--enforce-eager` off). |
| R6 | Flash-Next (MoE + GDN + QSA + MTP + host n-gram embed) actually **runs** in a current vLLM/SGLang release, not just theoretically. |
| R7 | 27B NVFP4 runs in vLLM/SGLang on Blackwell. |
| R8 | llama.cpp path for Flash-Next NVFP4: exists or does not; DFlash 2 applicability. |
| R9 | Colab install/compile time and CU waste vs the current Drive-cached `llama-server` plan. |
| R10 | TensorRT-LLM on Colab: realistic or not. |
| R11 | Required engine versions, CUDA, extra wheels for NVFP4 on `sm_120`. |
| R12 | Prefix caching / RadixAttention works for these hybrid GDN models. |

---

## K — KV cache and context (Job 07)

| ID | Claim |
| :--- | :--- |
| K1 | 27B NVFP4 weights **~16 GB**, **~80 GB** left for KV; context can pass **100K**. |
| K2 | Flash-Next weights **~70 GB**, **~24–26 GB** left for KV. |
| K3 | 24 GB FP8 KV is “plenty for **32K–48K** tokens”. |
| K4 | Huge multi-file codebases **>64K–128K** favor 27B on G4. |
| K5 | Sliding-window / message compaction is required for a long Flash-Next agent loop. |
| K6 | Per-token KV bytes for each model at FP16 / FP8 / INT8, accounting for hybrid attention (only full-attn layers grow KV). |
| K7 | GDN recurrent-state memory per slot, and how speculation multiplies it. |
| K8 | MoE router / expert-cache / n-gram-embed host-prefetch extra RAM and PCIe cost. |
| K9 | `--fit` / max context on 96 GB after weights + CUDA context + engine overhead. |
| K10 | Whether 51B n-gram embeddings really live in **host RAM** and what Colab host RAM that implies. |

---

## E — Colab CU economics (Job 08)

| ID | Claim |
| :--- | :--- |
| E1 | Same GPU, faster model → fewer CUs for a fixed automated token job (“race to sleep”). |
| E2 | 60,000 generated tokens: 27B **~140 tok/s → 7.1 min → 0.77 CU**; Flash-Next **~300 tok/s → 3.3 min → 0.36 CU**. |
| E3 | Interactive 1-hour sitting burns **~6.5 CU** on G4 regardless of model (idle dominates). |
| E4 | 25-turn agent task: 27B **14–18 s/turn, 6.5–7.5 min, 0.75–0.81 CU**; Flash-Next **4–6 s/turn, 1.8–2.5 min, 0.20–0.27 CU**. |
| E5 | Tasks per 100 CU: **~125** (27B) vs **~400+** (Flash-Next). |
| E6 | Prefill of growing 20K+ history becomes the silent killer by turn 15. |
| E7 | G4 vs L4 20-turn loop: G4 **~3.7 s/turn, 1.23 min, 0.133 CU**; L4 14B **~29.6 s/turn, 9.87 min, 0.345 CU**. |
| E8 | L4 14B prefill 15K **~1,200 tok/s**; decode **~35 tok/s**. |
| E9 | G4 Flash-Next prefill 15K **~8,000 tok/s**; decode **~320 tok/s**. |
| E10 | Smaller models incur an “agent failure tax” (more turns). Qualitative, but flag if overclaimed. |
| E11 | Workload 37M cache-hit + 2M cache-miss + 1M output on Flash-Next G4 burns **5.4–7.5 CU** (avg **~6.8**), **50–65 min**. |
| E12 | If 37M were all misses, extra **~72+ min** and **~8 CU**. |
| E13 | Equivalent API cost **$15–$35+** vs Colab **< $1** for ~7 CU. |
| E14 | Decode is **>80%** of billed time in that 40M-token mix. |
| E15 | 200 CU/month → **~30.5 hours** at 6.4–6.7 CU/hr. |
| E16 | Same 92.5/5.0/2.5 split, 200 CU yields **1.18B / 1.42B / 1.69B / 1.94B** total tokens at 320/400/500/600 decode tok/s. |
| E17 | Recalculate E2, E4, E7, E11, E15, E16 from first principles. Show CU = hours × rate. |
| E18 | Google AI Pro 200 CU/month pool still exists and G4 is billable from it. |

---

## Q — Quality, abliteration, agent fit (Job 09)

| ID | Claim |
| :--- | :--- |
| Q1 | Flash-Next is “a lot better” than 27B (chat’s qualitative claim). |
| Q2 | Flash-Next is the best model at this Colab price point. |
| Q3 | 125B MoE has the reasoning capacity to fix compiler traces in 1–2 turns; 7B/14B hallucinate flags and regexes. |
| Q4 | An abliterated / uncensored Flash-Next checkpoint exists (needed for the current uncensored worker plan). |
| Q5 | Tool calling / Hermes / OpenAI-compat on Flash-Next NVFP4. |
| Q6 | Thinking / `<think>` behavior vs `--reasoning off` on the 27B worker. |
| Q7 | Context window spec vs 27B. |
| Q8 | Refusal / safety behavior of the NVIDIA checkpoint vs Blackfrost ABLITERATED. |
| Q9 | Whether switching the worker to Flash-Next on G4 is justified vs staying on A100 27B llama.cpp. |
| Q10 | Known issues: routing instability, QSA quality drop, NVFP4 accuracy vs FP8/BF16. |
