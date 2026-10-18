# Colab Qwen Worker

Sahil's source of truth for running Qwen on Google Colab (git repository: [`fantastic25hippo/colab`](https://github.com/fantastic25hippo/colab)), billed against a **Google AI Pro 200 CU/month** pool.

---

## Current Decision (Short)

The repository implements a **dual-model deployment strategy** tailored to operator intent:
- **Primary / Leading Candidate:** [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (or [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)) served via **SGLang (TP=1)** with `--ple-offload-embedding` on **Colab G4** (RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM, ~177 GB host RAM, 8.71 CU/hr). Single-card serving is verified in the official SGLang cookbook (~148 tok/s with MTP).
- **Fallback Worker:** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) (`Q4_K_M`) served via **`llama.cpp`** (commit `efe3626` / tag `b11552`) with DFlash 2 `Q4_K_M` sidecar on **Colab A100-SXM4-40GB** (~5.37 CU/hr). Provides guaranteed zero refusals, fast Drive warm-boot (<45 s), and rock-solid memory stability (≤31,000 MiB with 100K context).

For full technical proofs, rooflines, and the candidate evaluation matrix, see [`decision.md`](decision.md). Factual audit and calculations are maintained in [`claims-ledger.md`](claims-ledger.md).

---

## Quick Start

1. **G4 Probe:** Check if your AI Pro account can provision G4 and verify unprivileged host memory pinning (`mlock`):  
   → Follow [`runbooks/g4-probe.md`](runbooks/g4-probe.md)
2. **Flash Next SGLang:** Launch the primary Flash-Next NVFP4 worker with SGLang TP=1 on G4:  
   → Follow [`runbooks/flash-next-sglang.md`](runbooks/flash-next-sglang.md)
3. **27B Fallback:** Launch the uncensored Blackfrost 27B worker with `llama.cpp` + DFlash 2 on A100:  
   → Follow [`runbooks/fallback-27b.md`](runbooks/fallback-27b.md)

Account authentication, session persistence, heartbeat, and shutdown procedures are documented in [`ops.md`](ops.md).

---

## Document Map

| Document | Role & Description | Status |
| :--- | :--- | :--- |
| [`claims-ledger.md`](claims-ledger.md) | **Source of truth.** Factual and numerical audit of all hardware, checkpoint, runtime, and economic claims across the repo. | Canonical |
| [`decision.md`](decision.md) | Current architecture decision, candidate evaluation matrix, G4 vs A100 comparison, and grounded economics. | Canonical |
| [`ops.md`](ops.md) | Colab CLI 0.7.4 workflow, Drive mount, trycloudflare tunnel, heartbeat loops, and watchdog shutdown. | Shipping |
| [`runbooks/g4-probe.md`](runbooks/g4-probe.md) | Operational probe for G4 entitlement, hardware specs, host mlock, and CU burn. | Shipping |
| [`runbooks/flash-next-sglang.md`](runbooks/flash-next-sglang.md) | Operational step-by-step launch procedure for Flash-Next NVFP4 via SGLang on G4. | Shipping |
| [`runbooks/fallback-27b.md`](runbooks/fallback-27b.md) | Operational step-by-step launch procedure for uncensored 27B worker via `llama.cpp` on A100. | Shipping |
| [`research/`](research/README.md) | Evidence archive (historical research notes, 27B plan archive, Flash-Next audit dossier, chat traces). | **Archive only** |

> [!IMPORTANT]
> **Precedence Rule:** `research/` (and historical research dossiers) is strictly an archive. Shipping specifications (`claims-ledger.md`, `decision.md`, `README.md`, `ops.md`, and `runbooks/`) take precedence and win on any conflict.
