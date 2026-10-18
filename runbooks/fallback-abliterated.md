# Runbook: Fallback Abliterated Flash-Next

Procedures for evaluating, verifying, and deploying an abliterated Qwen 3.8 Flash-Next checkpoint as an operational fallback when uncensored execution (zero refusals on sensitive systems programming, security, and penetration testing tasks) is required.

Primary references:
- SGLang Qwen 3.8 Flash-Next Cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
- Primary SGLang Launch Runbook: [`flash-next-sglang.md`](flash-next-sglang.md)
- Hardware & Entitlement Probe: [`g4-probe.md`](g4-probe.md)

---

## 1. Candidate Overview

Three community abliterated checkpoints exist on Hugging Face as candidate fallbacks. Loading stability, quantization integrity, and reasoning retention remain **UNVERIFIED**:

| Candidate | Precision & Runtime | HF Size | Date (Created) | Notes & Status |
| :--- | :--- | :---: | :---: | :--- |
| **`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD`** | NVFP4 (SGLang) | 106 GB | 2026-09-21 | Note: Created date (2026-09-21) predates its listed base checkpoint `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated` (2026-09-29). Loading in SGLang is **UNVERIFIED**. |
| **`claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`** | NVFP4 (SGLang) | 105.48 GB | 2026-10-09 | NVFP4 quantized weights with NVFP4 LM head. Loading in SGLang is **UNVERIFIED**. |
| **`Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`** | GGUF (`llama.cpp`) | 112 GB | 2026-09-17 | Different runtime (`llama.cpp`). Requires `llama.cpp` support for Flash-Next GDN + QSA hybrid architecture on Blackwell, which is **UNVERIFIED**. |

---

## 2. Selection Test Protocol

To qualify a candidate checkpoint for operational fallback use, it must pass three consecutive verification gates:

### Gate 1: Checkpoint Loading Verification
- **NVFP4 Candidates (`edp1096`, `claidler`)**:
  Deploy via SGLang using the standard G4 flags from [`flash-next-sglang.md`](flash-next-sglang.md) (TP=1, `--ple-offload-embedding`, no `--quantization` flag).
  - *Pass Criteria*: SGLang initializes without missing tensor assertions, shape mismatches, or CUDA illegal memory access on Blackwell `sm_120`.
- **GGUF Candidate (`Alexsi98`)**:
  Test with latest `llama.cpp` build supporting hybrid recurrent GDN layers.
  - *Pass Criteria*: Model weights load without panic, and offload to GPU memory succeeds without CPU mmap thrashing.

### Gate 2: GSM8K Reasoning Benchmark
Run an evaluation subset of GSM8K (minimum 50 problems) against the local `/v1` endpoint:
- *Pass Criteria*: Accuracy remains within 2.0% of stock Qwen 3.8 Flash-Next performance without reasoning degradation or perplexity explosion.

### Gate 3: Refusal Check
Submit standard sensitive security and systems engineering prompts (e.g., buffer overflow reproduction, exploit payload analysis):
- *Pass Criteria*: Zero refusal responses; prompt generates compliant technical output.

---

## 3. Operational Deployment

Once a candidate passes Gates 1–3:
1. Cache the qualified checkpoint to Google Drive under `/content/drive/MyDrive/colab_llm/checkpoints/<candidate_name>/` (see [`ops.md`](../ops.md)).
2. For NVFP4 candidates, execute via SGLang following [`flash-next-sglang.md`](flash-next-sglang.md), pointing `--model-path` to the cached checkpoint directory.
