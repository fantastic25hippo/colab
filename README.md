# Colab Qwen Worker

Sahil's source of truth for running Qwen on Google Colab (git repository: [`fantastic25hippo/colab`](https://github.com/fantastic25hippo/colab)), billed against a **Google AI Pro 200 CU/month** pool.

---

## Current Decision (Short)

The repository implements a **dual-model deployment strategy** tailored to operator intent:
- **Primary / Leading Candidate:** [`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4) (or [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)) served via **SGLang (TP=1)** with `--ple-offload-embedding` on **Colab G4** (RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM, ~177 GB host RAM, 8.71 CU/hr). Single-card serving is verified in the official SGLang cookbook (~148 tok/s with MTP).
- **Fallback: abliterated Flash-Next (TBD, unverified):** An abliterated checkpoint if refusal mitigation is required. Candidates: [`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD`](https://huggingface.co/edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD) (106 GB), [`claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`](https://huggingface.co/claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4) (105.48 GB), and [`Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`](https://huggingface.co/Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) (112 GB, GGUF, different runtime). See [`runbooks/fallback-abliterated.md`](runbooks/fallback-abliterated.md).

For full technical proofs, rooflines, and the candidate evaluation matrix, see [`decision.md`](decision.md). Factual audit and calculations are maintained in [`claims-ledger.md`](claims-ledger.md).

---

## Quick Start

1. **G4 Probe:** Check if your AI Pro account can provision G4 and verify unprivileged host memory pinning (`mlock`):  
   → Follow [`runbooks/g4-probe.md`](runbooks/g4-probe.md)
2. **Flash Next SGLang:** Launch the primary Flash-Next NVFP4 worker with SGLang TP=1 on G4:  
   → Follow [`runbooks/flash-next-sglang.md`](runbooks/flash-next-sglang.md)
3. **Fallback: abliterated Flash-Next (TBD, unverified):** Evaluation and candidate selection across `edp1096`, `claidler`, and `Alexsi98` (GGUF):  
   → Follow [`runbooks/fallback-abliterated.md`](runbooks/fallback-abliterated.md)

Account authentication, session persistence, heartbeat, and shutdown procedures are documented in [`ops.md`](ops.md).

---

## Document Map

| Document | Role & Description | Status |
| :--- | :--- | :--- |
| [`claims-ledger.md`](claims-ledger.md) | **Source of truth.** Factual and numerical audit of all hardware, checkpoint, runtime, and economic claims across the repo. | Canonical |
| [`decision.md`](decision.md) | Current architecture decision, candidate evaluation matrix, G4 platform analysis, and grounded economics. | Canonical |
| [`ops.md`](ops.md) | Colab CLI 0.7.4 workflow, Drive mount, trycloudflare tunnel, heartbeat loops, and watchdog shutdown. | Shipping |
| [`runbooks/g4-probe.md`](runbooks/g4-probe.md) | Operational probe for G4 entitlement, hardware specs, host mlock, and CU burn. | Shipping |
| [`runbooks/flash-next-sglang.md`](runbooks/flash-next-sglang.md) | Operational step-by-step launch procedure for Flash-Next NVFP4 via SGLang on G4. | Shipping |
| [`runbooks/fallback-abliterated.md`](runbooks/fallback-abliterated.md) | Fallback: abliterated Flash-Next (TBD, unverified) candidate evaluation and testing runbook (`edp1096`, `claidler`, `Alexsi98`). | Shipping |
| [`research/`](research/README.md) | Evidence archive (historical research notes, 27B plan archive, Flash-Next audit dossier, chat traces). | **Archive only** |

> [!IMPORTANT]
> **Precedence Rule:** `research/` (and historical research dossiers) is strictly an archive. Shipping specifications (`claims-ledger.md`, `decision.md`, `README.md`, `ops.md`, and `runbooks/`) take precedence and win on any conflict.
