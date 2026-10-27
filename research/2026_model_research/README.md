# 2026 Colab Model Speed, Capacity & Sizing Research

**Research Session Date:** October 24, 2026  
**Scope:** Modern open-weight foundation models (released May 1, 2026 – October 24, 2026) evaluated across all Google Colab accelerators (Tesla T4, L4, A100-40GB, A100-80GB, Blackwell G4, TPU v5e, and TPU v6e). Strictly limited to general-purpose language, reasoning, and coding LLMs (excluding domain-specific biology/audio/video models).

---

## Dossier Index

| Document | Core Objective | Key Deliverable |
| :--- | :--- | :--- |
| **[`coding-task-speed-single-stream-2026-10-24.md`](coding-task-speed-single-stream-2026-10-24.md)** | Single-stream coding task speed evaluation (Batch Size = 1) across all 7 accelerators. | Hardware-grouped speed matrix with **4 models per accelerator (2 Dense + 2 MoE)**, single-stream decode tok/s, reasoning token overhead, prefill speed, and standardized 1k-in/350-code E2E completion times. |
| **[`latest-models-gpu-fit-2026-10-24.md`](latest-models-gpu-fit-2026-10-24.md)** | Maximum parameter sizing and capacity fit for modern models across Colab hardware. | Identifies the modern ~33B dense ceiling (*Agnes-3.0-Flash*) and catalogs MoE architectures from 16B up to 125B (*Qwen3.8-Flash-Next*, *Ornith-1.5-122B*) for 16GB, 24GB, 40GB, 80GB, and 96GB memory budgets. |
| **[`throughput-tokens-per-second-estimates-2026-10-24.md`](throughput-tokens-per-second-estimates-2026-10-24.md)** | Hardware roofline derivations and optimal precision configurations. | Hardware memory bandwidth calculations, MBU utilization baselines, and precision formats (INT4 Marlin, NVFP4, Native INT8, Native FP8). |

---

## Hardware Coverage
1. **NVIDIA Tesla T4 (16 GB GDDR6)**
2. **NVIDIA L4 (24 GB GDDR6)**
3. **NVIDIA A100 (40 GB HBM2)**
4. **NVIDIA A100 (80 GB HBM2e)**
5. **NVIDIA G4 (Blackwell RTX PRO 6000 96 GB GDDR7)**
6. **Google Cloud TPU v5e (16 GB HBM2e)**
7. **Google Cloud TPU v6e / Trillium (32 GB HBM)**
