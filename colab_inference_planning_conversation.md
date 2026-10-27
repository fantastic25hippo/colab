# Colab Inference Planning: Conversation Record

This document records the planning discussion covering TPU and GPU inference on Google Colab, model selection, precision formats, cost modeling, and the parent-worker agent design. It is a summary of the reasoning in the conversation, not a verbatim transcript.

---

## 1. Hardware available on Colab

Specifications as reported in the conversation:

| Accelerator | Memory | Notes |
|---|---|---|
| TPU v6e | 32 GB HBM, ~1640 GB/s, 918 TFLOPS BF16 | Host RAM 32 GB, 8 vCPUs |
| TPU v5e | 16 GB HBM, ~820 GB/s (approximate, verify) | |
| T4 | 16 GB, ~320 GB/s (approximate, verify) | Turing, no native bf16 |
| L4 | 24 GB, ~300 GB/s (approximate, verify) | |
| A100 40 GB / 80 GB | ~1.5 to 2 TB/s (approximate, verify) | Mature CUDA kernel ecosystem |
| RTX PRO 6000 (Colab G4) | 96 GB, ~1.8 TB/s (approximate, verify) | Check which runtimes expose it |

Bandwidth and memory figures for GPUs other than the TPU v6e were given from memory and should be verified against spec sheets before use.

---

## 2. Core performance model

Single-stream decoding (batch size 1) is memory-bandwidth-bound, not compute-bound.

- Ceiling: tokens/s ≈ memory bandwidth ÷ bytes of active weights read per token
- Real throughput typically reaches 50 to 70 percent of the ceiling, lower with long contexts, KV cache reads, and kernel overhead.
- Ridge point on TPU v6e: roughly 560 FLOPs per byte. Batch 1 decode sits near 1 to 2 FLOPs per byte, so the chip is far below compute limits.
- The main levers are fewer bytes per token (quantization, MoE sparsity) and more tokens sharing each weight read (batching, which is less relevant for a single user).

---

## 3. Precision guidance

General principles discussed:

- Use int8 or int4 weights with bf16 activations on TPU. Decode speed depends on weight bytes, not activation precision, at batch 1.
- Dequantization must fuse into the matmul. If XLA materializes bf16 weights in HBM before the dot, the quantization benefit is lost. Verify in the compiled HLO or profile.
- An int8 KV cache halves KV memory and helps with long contexts.
- FP8 does not help on TPU v6e (no native FP8 matmul path). On A100, there are no FP8 tensor cores, so FP8 KV is not a good choice; use FP16 or INT8 KV there. FP8 is a Hopper (H100) feature.
- The W4A16 + FP8 KV suggestion from an agent was assessed as sound in direction on A100, but Marlin is a CUDA-specific kernel that does not transfer to TPU, and the FP8 KV part does not apply to A100.

Recommended choices for Flint on TPU v6e:

- Weights: the released INT4 checkpoint (Hugging Face: RecursiveSI/flint-int4), matching its exact quantization scheme.
- Activations: bf16.
- KV cache: int8.
- Requantizing to int8 is not viable (roughly 40 GB), and bf16 (roughly 80 GB) does not fit.

---

## 4. The model: Flint

Facts verified via web search during the conversation:

- Name: Flint, by Recursive (RecursiveSI).
- Total parameters: 39.8B. Active parameters per token: approximately 2.5B.
- Architecture: Mixture-of-Experts, reasoning-capable, multi-turn tool use.
- Context window: 262K tokens.
- Released: 2026-10-21. Knowledge cutoff listed as 2026-06.
- Weights: open, on Hugging Face (full and INT4 variants).
- Ollama default build (flint:40b): about 23 GB on disk.
- Publisher-reported results: 54.9% on Terminal-Bench 2.1, 89.7% on SWE-bench Pro v2, 74.2% on SWE-bench Verified. These are self-reported and depend on evaluation setup.

Open items to verify from the model card and config.json:

- Exact INT4 quantization scheme (group size, symmetry, packing, which layers are quantized, whether embeddings and LM head are quantized).
- Attention configuration: number of layers, KV heads, head dimension, any sliding-window or compressed attention. These determine KV cache size per token.
- Expert count and top-k routing, and whether there are shared experts.

Memory budget on TPU v6e (32 GB):

- Weights at INT4: about 23 GB.
- Runtime overhead (XLA, buffers): roughly 2 to 3 GB.
- Remaining for KV cache and activations: roughly 6 to 8 GB, which caps context length. KV bytes per token = 2 × layers × KV heads × head dim × bytes per element. The full 262K context will not fit.

Rough decode ceiling: about 1.25 GB of active weights per token at 4 bits, plus the LM head if unquantized. Theoretical ceiling is in the high hundreds of tokens per second on v6e; realistic throughput will be well below that.

---

## 5. Building the inference stack

- No existing NVIDIA-oriented stack (vLLM CUDA backends, llama.cpp GGUF paths, Marlin, Ollama) runs on TPU. The model definition must be ported to JAX or Flax, including the router, expert indexing, and attention variant.
- Components to write or adapt:
  - Weight loading and quantization, streaming shard by shard (host RAM is only 32 GB).
  - Forward pass in JAX, with top-k expert indexing so only active experts are read.
  - Quantized dequant-matmul kernel in Pallas, fused into the matmul.
  - Attention kernels, adaptable from JAX/Pallas ops examples (flash and paged attention).
  - Jitted decode loop with `lax.while_loop` or `lax.scan`, KV cache donation, fixed shape buckets to avoid recompilation.
- Single-user serving removes the need for continuous batching, paged KV scheduling, and request routing.
- Speculative decoding (prompt-lookup or a small draft model) is a useful add-on at batch 1.
- Profile with jax.profiler and xprof. Compare achieved bandwidth against the ceiling. Avoid host sync (`.item()`, Python conditionals on device values) in the loop.
- Suggested path: correct but slow version first, then replace hot spots one at a time (quantized matmul, then attention).

---

## 6. Model-first evaluation plan

Agreed approach: evaluate per model, then select hardware, rather than per GPU.

Per model, the agent should extract from the paper, config.json, and model card:

- Total and active parameters, layers, expert count and top-k, KV heads and head dimension.
- Native and released precision, quantization scheme.

For each model and each GPU:

1. Fit: weights + KV cache at target context + 2 to 3 GB overhead + headroom (about 10 to 15 percent). Record the maximum context that fits.
2. Precision choice: lowest-bit weight format with a working kernel on that hardware. Kernel availability is a hard constraint.
3. Host RAM: shard size plus conversion overhead must fit host RAM during loading. Check the RAM each Colab runtime reports.
4. Speed ceiling: active bytes per token ÷ bandwidth, multiplied by an efficiency factor reflecting kernel maturity (for example 0.5 for early custom kernels, 0.7 when tuned). Measure at least one real point per model/GPU pair.
5. Cost: based on runtime billing, including load time and warmup.

A script was offered (not yet written) to take a config.json and GPU spec table and output fit, maximum context, ceiling, and cost per task for each model/GPU/precision combination.

---

## 7. Cost modeling

Early discussion used cost per token. This was revised.

Reasoning: Colab bills by the hour of runtime, regardless of model size. A larger active model that is faster can finish work sooner and cost less.

Revised metric:

cost per task = hourly rate × (expected tokens per task ÷ tokens per second) ÷ 3600 + fixed overhead

- Expected tokens per task includes retries. Retry rate is often the largest term.
- Tokens per second should be measured on realistic long-context workloads, not short prompts.
- Fixed overhead (loading, compilation, warmup) is billed each session. Batch tasks within a session where possible.
- Rank by expected cost per verified successful task, not by tokens per second or cost per token alone.

Worked comparison discussed: a 500 tok/s model completing 175,000 tokens (100,000 plus 75,000 for rewrites) takes about 350 seconds. A 100 tok/s model completing 100,000 tokens in one shot takes about 1,000 seconds. On the same hourly rate, the faster model is about three times cheaper per task, provided the faster model's failures are caught by verification and the slower model's one-shot success holds.

Caveats:

- The slower model must actually succeed in one shot. Retry rates should be measured on the same tasks for both models.
- Verification must be cheap and reliable. Silent errors (wrong logic that passes weak tests) make a fast model look good while producing bad work.
- End-to-end task time includes prefill, reading files, and tool outputs, not only decode speed.

Suggested measurement: run both models on 20 to 30 representative tasks and record total wall-clock time including verification, number of verified successes, and number of attempts.

---

## 8. Parent-worker agent architecture

Proposed design:

- Parent: a stronger proprietary model (options mentioned included Claude Opus, Claude Sonnet, Gemini, and Grok). Responsible for planning, decomposing tasks, writing briefs, acceptance tests, and reviewing results.
- Worker: the cheap model running on Colab. Responsible for all code generation and bulk writing.
- Principle: the worker should be as fast as possible while clearing the verifier's bar; quality is enforced through tests and specs written by the parent.

Guidelines agreed:

- Parent output should stay small: task briefs, specs, acceptance tests, short verdicts. Use a fixed structured format to prevent the parent from writing implementations.
- Output tokens are the expensive part. Input tokens (including cached input) are cheap, so passing test output back to the parent for a short verdict is efficient.
- Verification should rely on automated checks (tests, linters, type checks). The parent reads only failures and returns a pass/fail verdict with a one-line fix instruction.
- Keep the parent's context small: trim tool outputs to relevant failure lines and store full worker output on disk.
- Use prompt caching where supported.
- Watch for the parent quietly writing full code in briefs. Specify in the brief that the worker writes the implementation.
- Tasks that need design reasoning may be better done by the parent directly; mechanical tasks suit the fast worker.

Metrics to track:

- Parent output tokens per verified task.
- Colab hours per verified task.
- Verified completions per Colab hour.
- Parent tokens consumed per task (input and output).

Key risk: silent failures that tests do not catch. Invest in tests and specifications before investing in a faster model.

---

## 9. Open items and next steps

- Read the Flint model card and config.json for the INT4 scheme and attention parameters; compute the KV cache size and maximum context on TPU v6e.
- Verify GPU and TPU specifications before relying on them, especially the v5e, T4, L4, A100, and RTX PRO 6000 figures.
- Have the agent extract architecture fields from each candidate model's config.json rather than summarizing from papers, since papers may describe training configurations that differ from released checkpoints.
- Build the fit-and-cost script: inputs are config.json and a GPU spec table (bandwidth, memory, kernel support flags, hourly rate entered by the user). Outputs one row per model, GPU, and precision with fit, maximum context, speed ceiling, and cost per verified task.
- Run the 20 to 30 task benchmark on at least two models to compare verified completions per Colab hour.
- Confirm which Colab runtimes actually expose each accelerator and its reported memory before planning around them.
