# 07 — TPU v5e-1 Runtime, Memory and XLA Practices

Consolidated from `TPU-V5e-Best-Practices.md`, itself distilled from `TPU_v5e_LLM_Inference_Analysis.pdf` and `MoE_Models_on_TPU_v5e-1.pdf` (the two source PDFs are no longer in the folder). Scope: single-chip `v5litepod-1`, 16 GB HBM, memory budget, XLA overhead, quant format, KV cache and acceleration. The vLLM/TorchTPU launch details are in `07b-TPU-v5e-vLLM-and-TorchTPU.md`.

## 1. Hardware envelope

- **Chip:** single-chip `v5litepod-1`, 1 TensorCore, 4 Matrix Multiply Units (MXUs).
- **HBM:** 16 GB nominal HBM2e, **15.75 GiB (16.91 GB decimal) visible to XLA/PjRt**.
- **Bandwidth:** 819 GB/s effective (859 GB/s or 800 GiB/s theoretical peaks quoted).
- **Compute:** **197 TFLOPS BF16, 393 TOPS INT8**.
- **On-chip:** 32 MB VMEM/transposed SRAM (compiler-managed), Vector Processing Unit (VPU) for dequant + activations.
- **Roofline:** batch-1 autoregressive decode is **1–2 FLOPs/byte** versus a ridge of ~240 FLOPs/byte → strictly memory-bandwidth bound. Per-token latency = active params + KV cache streamed over HBM. All params must stay in HBM to avoid catastrophic host-offload stalls.

## 2. Memory budget truth

External production reports (`vllm/vllm-tpu` and `TorchTPU 0.2.0`) plus **in-repo measurements on `v5litepod-1` (`../benchmarks/TPU-v5e/tpu-final/RESULTS.md`, `tpu-characterization/RESULTS.md`)**:

- **In-repo measured (`Qwen/Qwen3-4B` bf16, `vllm-tpu==0.32.0`, `libtpu==0.0.48`, `jax==0.11.2`, `transformers==4.57.1`):**
  - Usable driver HBM limit: **15.748 GiB (`16,909,336,064` B)** [M].
  - At `gpu_memory_utilization=0.90`: allocates **14.189 GiB** (`7.494 GiB` weights + `6.695 GiB` KV for 48,576 tokens across 759 blocks of 64), peak HBM **14.560 GiB**, leaving **1.188 GiB (1,216 MiB)** driver headroom. **Measured net XLA runtime overhead (`peak - weights - KV`) is only 0.045–0.371 GiB (47–380 MiB)** [M].
  - At `gpu_memory_utilization=0.95`: allocates **54,336 KV tokens (`7.485 GiB`)**, peak HBM **15.351 GiB** (**406 MiB** headroom), and serves single-stream requests stably at **84.91 tok/s** [M]. Use `0.90` for Batch 32 concurrency.
- **External report (`--gpu-memory-utilization 0.92`):** reserves **92% = 14.49 GiB** for weights + KV, leaving **1.26 GiB** (`15.75 × 0.08`) unallocated at the bottom of HBM for XLA HLO program images + scratchpads (expanding to 3.15 GiB at `0.80` when un-pruned multimodal towers or huge prefill buckets are compiled).

## 3. The “4.5–6.5 GB XLA overhead” claim

A prior analysis claimed XLA needs 4.5–6.5 GB beyond model weights. Both source docs verify this is an **artifact**, overstating steady-state overhead by >3× when static shape bucketing + pinned KV allocation are applied. It does accurately reflect worst-case unpinned/dynamic-shape graph allocation.

**When 4.5–6.5 GB actually materializes (three misconfigurations):**
1. Unbucketed PyTorch/XLA or naive `torch.compile` — every seq-len change triggers a new full-model HLO graph cached simultaneously in HBM.
2. Large startup profiler pass — engine's startup memory profiler runs dummy forwards with `--max-num-batched-tokens 8192+` before KV pool allocation.
3. Fallback unpack path — loading HF `compressed-tensors` checkpoints through pathways that allocate temporary BF16 unpacking buffers in HBM during init.

Failure signature:
```
RuntimeProgramAllocationFailure: Attempting to reserve 384.11M at the bottom of memory... There are 347.33M free
```

**How to contain to 1.26–1.80 GiB:**
- Pin KV cache pool in bytes to bypass the profiler: `--kv-cache-memory` / `kv_cache_memory_bytes`, e.g. `5923602432` (5.52 GiB measured) or `6442450944` (6.0 GiB).
- Restrict `--max-num-batched-tokens 2048` — trims the power-of-two prefill compilation ladder to `[16,32,64,…,2048]`.
- Use a text-only override to prune vision towers.
- Restrict compile buckets; size `gmm_v2` tiles for VMEM headroom.
- Use static shapes, Bounded Dynamism, and a Static Ring KV Cache.

## 4. Requirements to run on Colab v5e-1

- **Runtime:** select TPU v5e-1 (`v5litepod-1`) fresh runtime.
- **vLLM path:** `jax[tpu]>=0.5.0`, `vllm`, `tpu-inference` from `vllm-project/tpu-inference`, `llm-compressor`, `compressed-tensors>=0.19.0`, `transformers>=5.20.2`.
- **TorchTPU path:** `torch_tpu 0.2.0`, `libtpu 0.0.50`, `jax 0.11.2`.
- **Persistent compile cache (mandatory for fast restarts; stage in `/dev/shm`):**
  ```bash
  export VLLM_XLA_CACHE_PATH="/dev/shm/tpu_compile_cache"
  export JAX_COMPILATION_CACHE_DIR="/dev/shm/tpu_compile_cache"
  mkdir -p /dev/shm/tpu_compile_cache
  ```
  In-repo measured (`tpu-final/RESULTS.md`): restoring the persistent cache cuts AOT compilation **67% (147.5 s → 48.5 s, 261 cache hits / 0 compiles)** and warm-cache cold start **39% (388 s → 235 s)** [M] (external report cited 857 s → 497 s).
- **Quant format:** NVFP4/MXFP4/FP8 (`mxfp8`, `nvfp4`, `fp8`) are **GPU-only, unsupported on v5e silicon** — no native FP8/FP4 execution units. Convert to **W4A16 compressed-tensors `pack_quantized` symmetric int4** (8× int4 packed into int32 + BF16 group scales), `group_size=128` preferred (saves ~0.5 GiB vs 32). Keep MoE routers/gates in high precision.
- **Embeddings:** quantize `embed_tokens` + `lm_head` to a 4-bit packed grid — saves **0.7–1.86 GiB HBM** and active bandwidth. Needs the `jev-tpu-v5e1` 4-bit embedding patch for vLLM.

## 5. KV cache best practices

- **Static INT8 Ring KV is mandatory at 200K** — prevents OOM (1.91 GiB vs 3.81 GiB BF16) and boosts 200K decode **1.63×** (250 vs 153 tok/s example).
- Format: symmetric integer INT8 with per-token-head BF16 scales inside a pre-allocated ring buffer.
- For SWA models: Ring KV bounds 21× SWA-1024 layers to 1024 tokens, 18× SWA-512 layers to 512 tokens, etc. Without it KV balloons 4×.
- For MLA models: cache the compressed latent (e.g. `kv_lora_rank 512` + rope tail 64 = 576 dims) → **0.75 GiB at 200K** vs 1.5 GiB BF16.

## 6. Quant conversion best practices

- Conversion does **not** require a GPU server.
- Streaming shard-by-shard on a cheap 2-vCPU/15 GB RAM machine or directly inside the Colab host CPU: **~388 s (6.5 min)**.
- Packing logic: symmetric int4, clamp to `[-8,7]` → unsigned `[0..15]`, pack 8 per int32 word.
- `llm-compressor` recipe: `QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["re:.*gate$"])` + embedding targets.
- Repack QAD-Q4_0 GGUF (13.58 GB) → preserves the trained 4-bit grid without PTQ rounding loss.

## 7. Acceleration techniques that work on v5e-1

1. **INT8 Ring KV** — zero extra compute, mandatory.
2. **N-Gram Prompt Lookup** — `--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'` — zero HBM, works natively; the MXU tile is the same cost for `[1,6]` as `[1,1]` → **1.5–2.4×** practical speedup on RAG/summarization/code-edit.
3. **Draft-model speculative** — only if HBM headroom exists (quantize a 0.3B draft to INT8 0.30 GB or W4A16 0.15 GB); verifies N tokens in a single MXU pass at the same weight BW cost.

## 8. Quick checklist for Colab

- [ ] Verify `jax.devices()` shows v5e-1, 15.75 GiB.
- [ ] Install `torch_tpu 0.2.0` + `libtpu 0.0.50` **or** vllm-jax tpu-inference with PRs #3653/#3660.
- [ ] Set compile-cache env vars.
- [ ] Convert weights to W4A16 CT + 4-bit embeddings offline, streaming, <12 GB RAM.
- [ ] Pin KV cache memory in bytes; `max-num-batched-tokens 2048`; `gpu-memory-utilization 0.92`.
- [ ] Use `device_map="tpu"` or packed shards direct to TPU; never unpack to BF16 in HBM.
- [ ] Use Static INT8 Ring KV; pass `cache_pos` as a 0-D TPU tensor.
- [ ] Compile with `torch.compile(backend="tpu")` — only two shapes.
- [ ] Buffer generated token IDs on-chip; sync every 64–128 tokens (or 8 via scheduler steps).
- [ ] Use INT8 KV dtype, not the FP8 flag (see conflict in 08/11).
- [ ] Enable the SWA Ring KV allocator if the model has sliding windows.

## 9. Caveats

- The “1.26 GiB XLA overhead” measurement assumes bucketed static shapes and pinned KV. An unbucketed model can still hit 4.5–6.5 GB and OOM; this is why LFM2.5-24B-A2B fails on v5e-1 (see 10).
- `--kv-cache-dtype int8` vs `fp8` is contested; verify actual KV allocation in `kv_cache_utils.py` logs. See `07b` for the vLLM/TorchTPU launch details.
- TorchTPU 0.2.0 + libtpu 0.0.50 is a fast-moving stack; pin versions.
