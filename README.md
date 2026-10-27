# Colab Inference Worker

Sahil's notes for running open-source models on Google Colab GPUs/TPUs (git repository: [`fantastic25hippo/colab`](https://github.com/fantastic25hippo/colab)), billed against a **Google AI Pro 200 CU/month** pool.

---

## Current Focus

- **Speed.** The only goal right now is the fastest single-stream decode for coding work.
- **Model by model.** Models are evaluated one at a time (fit, precision, speed ceiling, then a real measurement), not by picking one GPU plan up front.
- **Starting with Flint.** Per the newest research (2026-10-24), **Flint** (Recursive; 39.8B MoE, ~2.5B active, Apache 2.0, released 2026-10-21) is the speed pick:
  - [`coding-task-speed-single-stream-2026-10-24.md`](research/2026_model_research/coding-task-speed-single-stream-2026-10-24.md): on A100-40GB, Flint (W4A16 Marlin) is the "single-stream speed champion" at ~810 tok/s roofline, ~0.90 s for a 1k-in / 350-code completion.
  - [`throughput-tokens-per-second-estimates-2026-10-24.md`](research/2026_model_research/throughput-tokens-per-second-estimates-2026-10-24.md): the same row notes the raw bandwidth math (~815 tok/s) versus a calibrated ~165 tok/s raw decode (~260 tok/s with speculation), and recommends an FP16/BF16 KV cache for Flint.
  - [`colab_inference_planning_conversation.md`](colab_inference_planning_conversation.md): Flint on TPU v6e using the released INT4 checkpoint (`RecursiveSI/flint-int4`) with bf16 activations and an int8 KV cache, and the open items still to check (INT4 scheme, KV cache size, max context).
- **Nothing has been run on Colab yet.** All speeds above are research estimates, not measurements.

The earlier plan (Qwen3.8 Flash-Next NVFP4 on G4 with SGLang, its abliterated fallback, runbooks, guards and cost model) was removed on 2026-10-27 because the newer research supersedes it. It is still in git history; the last commit that has it is `a65bded`.

---

## What Is Here (newest first)

| Path | What it is | Date |
| :--- | :--- | :--- |
| [`colab_inference_planning_conversation.md`](colab_inference_planning_conversation.md) | Planning record: performance model, precision guidance, Flint facts, model-first evaluation plan, cost-per-task metric, parent/worker design. | 2026-10-27 |
| [`GPU and TPU Speed Optimization.md`](GPU%20and%20TPU%20Speed%20Optimization.md) (also `.docx`, `.pdf`, `.zip`) | Audit and corrections of the per-accelerator precision documents, with a best-precision table for each accelerator. | 2026-10-24 |
| [`research/2026_model_research/`](research/2026_model_research/README.md) | **Current source of truth for model speed and fit.** May to Oct 2026 models across all 7 Colab accelerators. | 2026-10-24 |
| [`research/precision_format/`](research/precision_format/) | One precision-format note per accelerator (L4, A100, G4, T4, TPU v5e, TPU v6e), `.md` and `.pdf`. Corrected by the Speed Optimization document above. | 2026-10-24 to 10-27 |
| [`NVFP4 Versus FP4 Differences Explained.md`](NVFP4%20Versus%20FP4%20Differences%20Explained.md) | Chat notes on NVFP4 vs FP4. The first part is general; the later parts discuss Qwen3.8 Flash-Next checkpoints and are no longer the focus. | 2026-10-27 (added) |
| [`ops.md`](ops.md) | Model-independent Colab operations: account, Colab CLI, Drive mount, tunnel and API key, heartbeat, quota. | updated 2026-10-27 |
| [`hardware.md`](hardware.md) | Google Colab AI Pro runtime hardware specifications and comparison (T4, L4, A100, G4, H100, TPU v5e/v6e). | 2026-10-20 |

> [!NOTE]
> Where research notes disagree, prefer the newer date. The 2026-10-24 dossier in `research/2026_model_research/` is the newest.
