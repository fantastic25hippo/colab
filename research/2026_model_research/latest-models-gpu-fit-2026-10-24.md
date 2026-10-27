# Deep Research: Maximum Fit of Modern General-Purpose & Coding LLMs (May 2026 – Present) Across Google Colab Accelerators (GPUs & TPUs)

**Date:** October 24, 2026  
**Investigator:** Antigravity AI (Autonomous Deep Research Engine)  
**Strict Release Horizon:** May 1, 2026 – October 24, 2026 (Present Day)  
**Scope Filter:** **General-Purpose Language Models, Reasoning Models, and Dedicated Coding/Agentic Models ONLY.**  
*(Specialized genomic/DNA models like Carbon-X, audio/speech models like Ming-3.0-audio, pure video/diffusion denoisers like Nemotron TwoTower, and embedding/reranking models have been strictly excluded).*  
**Primary Ground-Truth Registries:** [NLP.COM.AI Open Weight Models Dataset](https://nlp.com.ai/models/) (`data.jsonl`), [Startrise AI Models Release Tracker](https://www.startrise.io/models/).

---

## 1. Executive Summary

Filtering the 128 open-weight foundation models released between **May 1, 2026 and October 24, 2026** down to general-purpose text, reasoning, and code/agent LLMs yields two fundamental architectural realities:

1. **The Dense Model Ceiling (~33B):** Between May and October 2026, open-weight foundation labs completely ceased releasing monolithic general-purpose dense models larger than 35B. Training and inference economics have driven the entire industry to Mixture of Experts (MoE) for scales beyond 35B. The absolute largest general dense models released in this window max out at **31B–33B** (e.g., *Agnes-3.0-Flash* [33.1B], *Equinox-31B* [31.0B], *Granite 4.2 30B* [29.3B], and *Qwen3.8-27B* [27.0B]).
2. **MoE Spans Every Scale:** Sparse MoE architectures now cover every tier of deployment:
   * **16B–24B total parameter MoEs** (e.g., *LFM2.5-24B-A2B*, *Instella-MoE-16B-A3B*) designed specifically for 16 GB accelerators like the Tesla T4 and TPU v5e.
   * **30B–40B total parameter MoEs** (e.g., *KAT-Coder-V2.5-Dev*, *Flint*, *Nemotron 3.5 Lightning 30B-A3B*) that maximize 24 GB / 32 GB accelerators like the L4 and TPU v6e.
   * **60B–80B MoEs** (e.g., *Comet 60B*, *Kolibri-1*, *Nemotron 3.5 Bolt 74B-A9B*) for 40 GB / 80 GB A100s.
   * **120B–125B MoEs** (e.g., *Qwen3.8-Flash-Next*, *Ornith-1.5-122B*, *Nemotron 3.5 Super VL*) engineered specifically to saturate single 80 GB / 96 GB datacenter GPUs (A100 80GB, Blackwell G4) without multi-node clustering.

---

## 2. Master Sizing Matrix Across All Seven Colab Accelerators (GPUs & TPUs)

All sizing calculations assume the optimal high-throughput precision formats established in your research:
* **NVIDIA GPUs (T4, L4, A100):** 4-bit INT4 (W4A16 via Marlin / ExLlamaV2) $\approx$ **0.55 GB per billion parameters**.
* **NVIDIA G4 (Blackwell RTX PRO 6000):** Native NVFP4 with microscaling $\approx$ **0.54 GB per billion parameters**.
* **Google Cloud TPU v5e:** Native INT8 (W8A8 systolic array via AQT/MaxText) $\approx$ **1.05 GB per billion parameters** (or W4A16 via VPU unpacking $\approx$ 0.55 GB/B).
* **Google Cloud TPU v6e (Trillium):** Native FP8 (E4M3 systolic array via AQT/MaxText) $\approx$ **1.05 GB per billion parameters** (or W4A16 $\approx$ 0.55 GB/B).

| Colab Accelerator | Hardware Class & Memory | Usable Weight VRAM | Max **Dense** Model (General / Code, May–Oct 2026) | Max **Mixture of Experts** Model (General / Code, May–Oct 2026) |
| :--- | :---: | :---: | :--- | :--- |
| **NVIDIA Tesla T4** | GPU (16 GB GDDR6) | ~13.5 GB | **Krake V2 Legacy (21.0B)**<br>• Released: *Sept 24, 2026* by NovelAI<br>• 4-bit Weight: **~11.6 GB** (leaves ~1.9 GB for context)<br>*(Safe alternative: Nemotron 3.5 Bolt E4B 8.0B @ ~4.4 GB with ~9.1 GB context headroom)* | **LFM2.5-24B-A2B (24.0B Total / 2.45B Active)**<br>• Released: *Oct 13, 2026* by Liquid AI<br>• 4-bit Weight: **~13.2 GB** (fits tight 2k context)<br>*(Safe alternative: Instella-MoE-16B-A3B [AMD, July 24, 2026] @ ~8.8 GB with ~4.7 GB context headroom)* |
| **NVIDIA L4** | GPU (24 GB GDDR6) | ~21.0 GB | **Agnes-3.0-Flash (33.1B)**<br>• Released: *Sept 11, 2026* by Agnes AI<br>• 4-bit Weight: **~18.2 GB**<br>• Leaves ~2.8 GB for 4k context<br>*(Alternative: Equinox-31B @ ~17.0 GB; Granite 4.2 30B @ ~16.1 GB)* | **KAT-Coder-V2.5-Dev (35.0B Total / 3.0B Active)**<br>• Released: *July 23, 2026* by Kuaishou KwaiKAT<br>• 4-bit Weight: **~19.2 GB** (leaves ~1.8 GB for context)<br>*(Alternative: Nemotron 3.5 Lightning 30B-A3B @ ~16.5 GB with ~4.5 GB context headroom)* |
| **NVIDIA A100 (40 GB)** | GPU (40 GB HBM2) | ~36.0 GB | **Agnes-3.0-Flash (33.1B)**<br>• Released: *Sept 11, 2026*<br>• 4-bit Weight: **~18.2 GB**<br>• Leaves **~17.8 GB free VRAM** for 64k+ context and concurrent batching | **Comet 60B (61.1B Total / 12.3B Active)**<br>• Released: *Oct 15, 2026* by Multiverse Computing<br>• 4-bit Weight: **~33.6 GB** (leaves ~2.4 GB headroom for context)<br>*(Alternative: Flint [Oct 21, 2026, 39.8B total] @ ~21.9 GB with ~14.1 GB headroom)* |
| **NVIDIA A100 (80 GB)** | GPU (80 GB HBM2e) | ~74.0 GB | **Agnes-3.0-Flash (33.1B)** *(Largest general dense model in 2026 dataset)*<br>• 4-bit Weight: ~18.2 GB<br>• Leaves **~55.8 GB free** for massive 128k context and large batch serving | **Qwen3.8-Flash-Next (125.0B Total / 6.0B Active)**<br>• Released: *Aug 26, 2026* by Alibaba Qwen Team<br>• 4-bit Weight: **~68.8 GB** (leaves ~5.2 GB headroom)<br>*(Alternative: Ornith-1.5-122B [Oct 20, 2026, MIT license] @ ~67.1 GB with ~6.9 GB context headroom)* |
| **NVIDIA G4 (RTX PRO 6000)** | GPU (96 GB GDDR7) | ~89.0 GB | **Agnes-3.0-Flash (33.1B)** *(or custom dense models up to ~140B)*<br>• In NVFP4, 33B consumes only **~17.9 GB**, leaving **~71.1 GB free** for extreme 128k+ sequences | **Qwen3.8-Flash-Next (125.0B Total / 6.0B Active)**<br>• Weight in NVFP4: **~67.5 GB**, leaving **~21.5 GB dedicated GDDR7 headroom** for 64k+ context<br>*(Alternative: Ornith-1.5-122B [122B total] @ ~65.8 GB with ~23.2 GB dedicated VRAM)* |
| **Google Cloud TPU v5e** | TPU (16 GB HBM2e) | ~13.5 GB | **NVIDIA Nemotron 3.5 Bolt E4B (8.0B)**<br>• Released: *Oct 23, 2026* (NVIDIA)<br>• Native INT8 Weight: **~8.4 GB** (leaves ~5.1 GB for KV cache)<br>*(In W4A16: Can fit up to Krake V2 Legacy 21.0B @ ~11.6 GB)* | **Mellum2-12B-A2.5B (12.0B Total / 2.5B Active)**<br>• Released: *May 26, 2026* by JetBrains<br>• Native INT8 Weight: **~12.6 GB** (leaves ~0.9 GB headroom)<br>*(In W4A16 via VPU: Fits Instella-MoE-16B-A3B [16B total] @ ~8.8 GB with ~4.7 GB context headroom)* |
| **Google Cloud TPU v6e (Trillium)** | TPU (32 GB HBM) | ~28.0 GB | **Krake V2 Legacy (21.0B)**<br>• Released: *Sept 24, 2026* by NovelAI<br>• Native FP8 Weight: **~22.1 GB** (leaves ~5.9 GB for KV cache)<br>*(In W4A16 via VPU: Fits Agnes-3.0-Flash 33.1B @ ~18.2 GB with ~9.8 GB context headroom)* | **LFM2.5-24B-A2B (24.0B Total / 2.45B Active)**<br>• Released: *Oct 13, 2026* by Liquid AI<br>• Native FP8 Weight: **~25.2 GB** (leaves ~2.8 GB headroom in systolic array)<br>*(In W4A16 via VPU: Fits KAT-Coder-V2.5-Dev 35.0B @ ~19.2 GB with ~8.8 GB context headroom)* |

---

## 3. Verified Chronological Catalog: General-Purpose & Coding Dense LLMs (May 1 – Oct 24, 2026)

All entries are strictly filtered from `nlp.com.ai` and `startrise.io`, restricted to general language, reasoning, and coding models:

| Release Date | Model Name | Developer / Lab | Parameter Size | License | Architecture & Serving Target |
| :--- | :--- | :--- | :---: | :--- | :--- |
| **2026-10-23** | **[NVIDIA Nemotron 3.5 Bolt E4B](https://nlp.com.ai/models/nvidia-nemotron-35-bolt-e4b/)** | NVIDIA Corporation (USA) | **8.0B** (4B effective) | OpenMDW 1.1 | General reasoning, conversational tool use, and agentic workflows. Engineered specifically for a ~4.0 GB runtime footprint in NVFP4 / Q4_K_M GGUF. Supports companion DFlash speculative draft models. |
| **2026-10-20** | **[Spark-X2.6-4B](https://nlp.com.ai/models/spark-x26-4b/)** | SparkLLM / XHToken (Singapore) | **4.11B** | Apache 2.0 | Compact dense reasoning model for edge, mobile, and low-latency agentic loops. |
| **2026-10-16** | **[Nemotron Labs Dory 1.0 4B](https://nlp.com.ai/models/nemotron-labs-dory-10-4b/)** | NVIDIA Corporation (USA) | **3.98B** | OpenMDW 1.1 | Experimental looped dense architecture re-applying shared transformer blocks to trade inference compute for higher accuracy. |
| **2026-09-24** | **[Krake V2 Legacy](https://nlp.com.ai/models/krake-v2-legacy/)** | NovelAI / Anlatan (USA) | **21.0B** | Proprietary Open Weights | Dense GPT-NeoX-20B fine-tune optimized for long-horizon creative and open-ended text generation. Weight in INT4: **~11.6 GB** (Fits T4 16GB). |
| **2026-09-20** | **[Hemmingway-1](https://nlp.com.ai/models/hemmingway-1/)** | Altworld (Switzerland / South Africa) | **27.0B** | Apache 2.0 | General foundation language model (26.9B serialized BF16 parameters). Fits L4 24GB in 4-bit (~14.8 GB). |
| **2026-09-15** | **[Zero (MC001)](https://nlp.com.ai/models/zero-mc001/)** | Moving Castles | **8.0B** | MIT | Dense general conversational foundation model. |
| **2026-09-11** | **[Agnes-3.0-Flash](https://nlp.com.ai/models/agnes-30-flash/)** | Agnes AI (Singapore) | **33.09B** | Apache 2.0 | **Largest general dense model released in May–Oct 2026.** Weight in 4-bit: **~18.2 GB**. Fits L4 24GB with ~2.8 GB headroom and A100 40GB with ~17.8 GB headroom. |
| **2026-08-25** | **[Granite 4.2 30B](https://nlp.com.ai/models/granite-42/)** | IBM (USA) | **29.28B** | Apache 2.0 | Enterprise general text and coding model. Weight in 4-bit: ~16.1 GB. |
| **2026-08-14** | **[Qwen3.8-27B](https://nlp.com.ai/models/qwen38-27b/)** | Alibaba Qwen Team (China) | **27.0B** | Apache 2.0 | Dense multimodal foundation model with native thinking budget controller. |
| **2026-07-27** | **[Neutrino-1 8B](https://nlp.com.ai/models/neutrino-1/)** | Fermion Research | **8.19B** | Apache 2.0 | General reasoning and problem-solving LLM. |
| **2026-07-20** | **[BTL-3](https://nlp.com.ai/models/btl-3/)** | Bad Theory Labs (Nigeria) | **27.0B** | Apache 2.0 | Multilingual dense model optimized for low-resource reasoning and African language tasks. |
| **2026-05-18** | **[Equinox-31B](https://nlp.com.ai/models/equinox-31b/)** | LatitudeGames | **31.0B** | MIT | Dense general narrative and interactive text generation model. Weight in 4-bit: ~17.0 GB. |

---

## 4. Verified Chronological Catalog: General-Purpose & Coding MoE Models (May 1 – Oct 24, 2026)

All entries are strictly filtered from `nlp.com.ai` and `startrise.io`, restricted to general language, reasoning, and coding models:

| Release Date | Model Name | Developer / Lab | Total Params | Active Params | Key Architectural Specs & Hardware Fit |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **2026-10-21** | **[Flint](https://nlp.com.ai/models/flint/)** | Recursive Superintelligence (USA) | **39.8B** | 2.5B | Apache 2.0 MoE coding LLM, 262K context, designed to run on single devices. Weight in 4-bit: **~21.9 GB**. Fits tight on L4, comfortably on A100 40GB. |
| **2026-10-21** | **[Raon-1 mini preview](https://nlp.com.ai/models/raon-1-mini-preview/)** | KRAFTON AI (South Korea) | **28.8B** | 4.3B | General dialogue and agentic MoE. Weight in 4-bit: **~15.8 GB**. Fits L4 24GB with ~5.2 GB headroom. |
| **2026-10-20** | **[Ornith-1.5-122B](https://nlp.com.ai/models/ornith-15-122b/)** | Ornith Team | **122.0B** | 8.0B | **MIT license.** General reasoning, coding, and agent model (84.5 SWE-bench Verified). 48 layers, 256 routed experts / 8 active, YaRN 1M context. Official NVFP4 & FP8. **Weight ~67.1 GB; fits single 80GB A100 & 96GB G4.** |
| **2026-10-15** | **[Comet 60B](https://nlp.com.ai/models/comet-60b/)** | Multiverse Computing (Spain) | **61.1B** | 12.3B | General enterprise compressed MoE. Weight in 4-bit: **~33.6 GB**. **Fits A100 40GB with ~2.4 GB headroom.** |
| **2026-10-15** | **[NVIDIA Nemotron 3.5 Bolt 74B-A9B](https://nlp.com.ai/models/nvidia-nemotron-35-bolt-74b-a9b/)** | NVIDIA Corporation (USA) | **74.0B** | 9.0B | LatentMoE general agent model. Weight in NVFP4 / 4-bit: **~40.7 GB**. Fits A100 80GB and G4 96GB with massive headroom. |
| **2026-10-15** | **[NVIDIA Nemotron 3.5 Super VL 120B-A12B](https://nlp.com.ai/models/nvidia-nemotron-35-super-vl-120b-a12b/)** | NVIDIA Corporation (USA) | **120.0B** | 12.0B | General multimodal MoE with 1M context. Weight in NVFP4: **~66.0 GB**. Fits A100 80GB and G4 96GB. |
| **2026-10-13** | **[LFM2.5-24B-A2B](https://nlp.com.ai/models/lfm25-24b-a2b/)** | Liquid AI (USA) | **24.0B** | 2.45B | General hybrid MoE with double-gated linear convolution layers. Weight in 4-bit: **~13.2 GB**. **Fits Tesla T4 16GB with tight 2k context; fits TPU v6e in native FP8 (~25.2 GB).** |
| **2026-10-03** | **[Kolibri-1](https://nlp.com.ai/models/kolibri-1/)** | Aleph Alpha (Germany) | **78.1B** | 3.46B | General English-German foundation MoE, Apache 2.0, 1M context. Weight in 4-bit: **~43.0 GB**. Fits A100 80GB & G4 96GB. |
| **2026-09-18** | **[AliceAI-Foundation-80B-A3B-Base](https://nlp.com.ai/models/aliceai-foundation-80b-a3b-base/)** | Yandex (Russia) | **80.0B** | 3.0B | General language foundation MoE. Weight in 4-bit: **~44.0 GB**. Fits A100 80GB & G4 96GB. |
| **2026-08-26** | **[Qwen3.8-Flash-Next](https://nlp.com.ai/models/qwen38-flash-next/)** | Alibaba Qwen Team (China) | **125.0B** | 6.0B | **Flagship general reasoning & coding MoE.** 125B MoE backbone + 51B N-gram embeddings. Weight in NVFP4: **~67.5 GB**. Fits G4 96GB and A100 80GB. |
| **2026-08-11** | **[NVIDIA Nemotron 3.5 Lightning 30B-A3B](https://nlp.com.ai/models/nvidia-nemotron-35-lightning-30b-a3b/)** | NVIDIA Corporation (USA) | **30.0B** | 3.0B | High-throughput low-latency general MoE. Weight in 4-bit: **~16.5 GB**. Fits L4 24GB. |
| **2026-08-04** | **[Ling-3.0-flash](https://nlp.com.ai/models/ling-30-flash/)** | inclusionAI (China) | **124.0B** | 5.1B | General reasoning MoE. Weight in 4-bit: **~68.2 GB**. Fits A100 80GB & G4 96GB. |
| **2026-07-24** | **[Instella-MoE-16B-A3B](https://nlp.com.ai/models/instella-moe-16b-a3b/)** | AMD (USA) | **16.0B** | 2.8B | General reasoning & code MoE. Weight in 4-bit: **~8.8 GB**. **Comfortably fits Tesla T4 16GB with ~4.7 GB context headroom; fits TPU v5e in W4A16.** |
| **2026-07-23** | **[KAT-Coder-V2.5-Dev](https://nlp.com.ai/models/kat-coder-v25-dev/)** | Kuaishou KwaiKAT (China) | **35.0B** | 3.0B | **Dedicated coding foundation MoE.** Weight in 4-bit: **~19.2 GB**. Fits L4 24GB with ~1.8 GB headroom; fits TPU v6e in W4A16. |
| **2026-07-21** | **[Laguna S 2.1](https://nlp.com.ai/models/laguna-s-21/)** | Poolside (USA) | **118.0B** | 8.0B | Software engineering & coding MoE. Weight in 4-bit: **~64.9 GB**. Fits A100 80GB & G4 96GB. |
| **2026-07-06** | **[Nemotron-Labs-3-Puzzle-75B-A9B](https://nlp.com.ai/models/nemotron-labs-3-puzzle-75b-a9b/)** | NVIDIA (USA) | **75.3B** | 9.3B | Reasoning and logic MoE with official NVFP4. Weight: **~41.4 GB**. Fits A100 80GB & G4 96GB. |
| **2026-07-02** | **[Laguna XS 2.1](https://nlp.com.ai/models/laguna-xs-21/)** | Poolside (USA) | **33.0B** | 3.0B | Fast coding & reasoning MoE. Weight in 4-bit: **~18.1 GB**. Fits L4 24GB. |
| **2026-06-09** | **[North Mini Code](https://nlp.com.ai/models/north-mini-code/)** | Cohere / Cohere Labs (Canada) | **30.0B** | 3.0B | Dedicated coding MoE. Weight in 4-bit: **~16.5 GB**. Fits L4 24GB. |
| **2026-05-26** | **[Mellum2-12B-A2.5B](https://nlp.com.ai/models/mellum2-12b-a25b/)** | JetBrains | **12.0B** | 2.5B | IDE & code completion MoE. Weight in 4-bit: **~6.6 GB**. Fits Tesla T4 16GB with ~6.9 GB context headroom; **fits TPU v5e in native INT8 (~12.6 GB).** |
| **2026-05-06** | **[ZAYA1-8B](https://nlp.com.ai/models/zaya1-8b/)** | Zyphra (USA) | **8.4B** | 760M | Ultra-sparse general reasoning MoE. Weight in 4-bit: **~4.6 GB**. Fits Tesla T4 16GB and TPU v5e. |

---

## 5. Architectural Deep-Dive on TPU Deployments in Colab

### A. Google Cloud TPU v5e (16 GB HBM2e)
* **Hardware Matrix:** The v5e Matrix Multiply Units (MXUs) natively accelerate **bfloat16** (197 TFLOPs) and **INT8** (393 TOPs). FP8 is **not** supported in hardware.
* **Native Mode Deployment:** Running in native **INT8 (W8A8)** via AQT/MaxText provides double the compute throughput of BF16 without decompression stalls.
  * In native INT8 (~1.05 GB/B), the usable 13.5 GB budget limits dense models to **8B** (e.g., *NVIDIA Nemotron 3.5 Bolt E4B* @ ~8.4 GB) and MoE models to **12B** (e.g., *Mellum2-12B-A2.5B* @ ~12.6 GB).
* **Quantized W4A16 Mode:** If using W4A16 (unpacking 4-bit weights into BF16 registers via the Vector Processing Unit), memory pressure drops to ~0.55 GB/B, allowing the **24B MoE** (*LFM2.5-24B-A2B*) or **16B MoE** (*Instella-MoE-16B-A3B*) to fit.

### B. Google Cloud TPU v6e / Trillium (32 GB HBM)
* **Hardware Matrix:** The v6e features native hardware datapaths for **FP8 (E4M3)** in the MXU (~1,836 TFLOPs, exactly 2$\times$ its 918 TFLOPs BF16 peak) and 32 GB of High-Bandwidth Memory (~1,638 GB/s).
* **Native Mode Deployment:** Serving in native **FP8 (W8A8 with FP8 KV cache)** via MaxText or JAX/AQT achieves maximum systolic throughput with zero dequantization overhead.
  * In native FP8 (~1.05 GB/B), the usable 28.0 GB budget comfortably accommodates **21B dense models** (e.g., *Krake V2 Legacy* @ ~22.1 GB) and **24B MoE models** (e.g., *LFM2.5-24B-A2B* @ ~25.2 GB).
* **Quantized W4A16 Mode:** Deploying W4A16 allows serving **33B–35B models** (e.g., *Agnes-3.0-Flash 33.1B* dense @ ~18.2 GB, or *KAT-Coder-V2.5-Dev 35B* MoE @ ~19.2 GB) with ~9–10 GB of HBM headroom for deep context windows.
