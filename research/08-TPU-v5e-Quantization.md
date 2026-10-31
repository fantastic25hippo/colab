# 08 — TPU v5e-1 Quantization: W4A16, W8A8 and KV Cache

Consolidated from `TPU v5e LLM Quantization Guide.pdf` (5 pages) and `TPU-V5e-LLM-Quantization-Guide.md`. Setting: Colab TPU v5e-1, 15.748 GiB usable HBM, Python 3.12, vllm-tpu 0.32.0, tpu-inference 0.32.0, jax/jaxlib 0.11.2, libtpu 0.0.48, torch 2.10.0 CPU, bf16 Qwen3-4B at 85–88 tok/s, batch 1, 32K+ context.

## 1. Verdict

**Partly.** Compressed-tensors W4A16 (wNa16) and AWQ W4A16 can run and keep weights packed in 4-bit in HBM on TPU v5e-1 using tpu-inference 0.32.0, saving ~75% of linear weight memory vs bf16. But support is split across two divergent paths (`MODEL_IMPL_TYPE=vllm` vs JAX-native), requires recently merged PRs (#3487 for vLLM/torchax; #3653 and #3660 for JAX-native), and **v5e has no native INT4 MXU compute**. 4-bit weights are unpacked and widened to bf16 inside on-chip VMEM before matrix multiplication. W4A16 saves HBM capacity — allowing 7B–12B models or huge KV caches on one 15.75 GiB chip — but **W8A8 INT8 is the only low-precision format on v5e with native 2× MXU compute** (393 TOPS vs 197 TFLOPS bf16) and consistently delivers the highest throughput and lowest prefill latency for long contexts.

## 2. Quantization format support matrix (tpu-inference 0.32.0 and main)

tpu-inference maintains two separate registries:
1. **vLLM/Torchax path** (`MODEL_IMPL_TYPE=vllm`): `tpu_inference/models/vllm/experimental/__init__.py` (`_TPU_VLLM_MODELS`) + `tpu_inference/layers/vllm/quantization/*`; overrides upstream configs via `get_tpu_quantization_config`.
2. **JAX-native path** (`MODEL_IMPL_TYPE=flax_nnx` or model default): `tpu_inference/models/common/model_loader.py` (`_MODEL_REGISTRY`) + `tpu_inference/layers/jax/quantization/*`.

Top-level methods in `tpu_inference/layers/common/quant_methods.py`: COMPRESSED_TENSORS, AWQ, FP8, MXFP4, NVFP4, DSV4_FP8, UNQUANTIZED.

| Format | vLLM/Torchax path | JAX-native path | Source / PRs |
| :--- | :--- | :--- | :--- |
| **compressed-tensors W4A16 (wNa16)** | Supported (Linear & MoE via PR #3487) | Supported for linears (PR #3653 via `gmm_v2`); MoE requires PR #3660 | `layers/vllm/quantization/compressed_tensors`, `layers/jax/quantization/compressed_tensors` |
| **compressed-tensors W8A8 (INT8)** | Supported (dynamic per-token act + INT8 weight) | Not supported | `layers/vllm/quantization/compressed_tensors` |
| **compressed-tensors W8A8 (FP8)** | Supported | Supported (Linear & `Fp8FusedMoEMethod`) | same |
| **compressed-tensors W8A16 (INT8 weight-only)** | Not supported (fails in `get_scheme`; Issue #3539) | Not supported (`NotImplementedError`) | same |
| **compressed-tensors W4A8 (FP8) / W4A4 (NVFP4)** | Supported (`w4a8_fp8`, `w4a4_nvfp4`) | Not supported | same |
| **AWQ (W4A16 INT4)** | Supported (added pre-v0.28.0) | Not supported (not in `layers/jax/quantization/`) | `layers/vllm/quantization/awq.py` (`VllmAWQConfig`) |
| **FP8 (`--quantization fp8`)** | Supported | Supported (`Fp8Tensorwise…`, `Fp8Blockwise…`) | `layers/vllm/quantization/fp8.py`; `layers/jax/quantization/fp8.py` |
| **MXFP4 / NVFP4** | Supported (`mxfp4`, `nvfp4`) | Supported (`Mxfp4Config` only) | `layers/jax/quantization/mxfp4.py` |
| **GPTQ (`--quantization gptq`)** | Not supported | Not supported | omitted from `quant_methods.py` |

**INT8 weight-only (W8A16) note:** Pallas tutorials demonstrate fused INT8 weight-only kernels, but inside the compressed-tensors dispatcher (`compressed_tensors.py:101–151`) the selection chain is `_is_fp8_w4a8 → _is_nvfp4_format → _is_fp8_w8a8 → _is_wNa16_group_channel` (added by #3487) → `_is_dynamic_token_w8a8`. Any weight-only config (`input_activations: null`) outside the INT4 wNa16 predicate, such as W8A16, falls through to `_is_dynamic_token_w8a8(weight_quant, None)` and crashes with `AttributeError: 'NoneType' object has no attribute 'num_bits'` (Issue #3539, filed 2026-09-08).

## 3. Issue #3540, sibling bugs and PRs

- **Issue #3540** (2026-09-08): “JAX-native path has no wNa16 scheme, so compressed-tensors W4A16 is vLLM-path-only.” `tpu_inference/layers/jax/quantization/compressed_tensors.py:154` contained `# TODO: w4a8 / wNa16 schemes need their own JAX methods (not yet ported)` and raised `NotImplementedError` for linear layers. PR #3487 added W4A16 only to the vLLM/torchax path.
- **Issue #3539** (same day): `NoneType` AttributeError in `get_scheme` on the vLLM path for weight-only configs outside #3487's predicate.
- **Issue #3542** (same day): on the JAX path (`compressed_tensors.py:101–114`), an MoE layer with compressed-tensors W4A16 weights checked `_is_fp8_w8a8`, saw False, and silently returned `UnquantizedFusedMoEMethod(layer)` — feeding packed uint32/int32 words into the dense MoE kernel as if unquantized, producing garbage without raising.
- **PR #3487** (merged pre-0.32.0): compressed-tensors W4A16 INT4 linear + MoE support on the vLLM/torchax path.
- **PR #3653**: JAX-path W4A16 linear method on `gmm_v2`; linked to #3540.
- **PR #3660**: `WNA16FusedMoEMethod`, stacked on #3653, adds JAX-path W4A16 MoE expert support via `gmm_v2` (group size 32; standard fused MoE kernel requires block sizes that are multiples of 256).

**Packed or dequantized?** Both merged paths keep weights packed as 4-bit in HBM:
- Empirical: a **12B** model (bf16 22.18 GiB JAX / 24.56 GiB PyTorch) loads in W4A16 at **9.46 GiB HBM** with PR #3653. A **26B-A4B** model (bf16 48.07 GiB) loads in W4A16 at **17.43 GiB HBM**.
- Execution: `gmm_v2` streams packed 4-bit weights + bf16 group scales from HBM into VMEM, dequantizes each tile to bf16 in VMEM, multiplies against bf16 activations on the MXU.

## 4. Hardware capabilities and batch-1 speed math

- **Usable HBM:** 15.75 GiB total (~14.49 GiB for weights + KV at `--gpu-memory-utilization 0.92`).
- **HBM bandwidth:** 800 GiB/s (859 GB/s). **MXU:** 197 TFLOPS bf16, 393 TOPS int8 (exact 2×).
- **No native INT4 or FP8 MXU on v5e or v6e.** INT8 (W8×A8) is the only low-precision format with native MXU acceleration. Neither v5e nor v6e has native fp8 compute (native fp8 arrived in TPU7x/Ironwood); int4 has no native MXU instruction. Both int4 and fp8 weights are widened to bf16 in VMEM/VPU before the MXU.
- **Critical arithmetic intensity:** ≈ **240 FLOPs/byte** (197 TFLOPS ÷ 859 GB/s). Batch-1 decode matmul is ≈ **1 FLOP/byte** → strictly memory-bandwidth bound.

**Why W4A16 does not give 4× decode over bf16 (and often trails W8A8):**
1. **Unquantized embeddings/lm_head.** Standard compressed-tensors W4A16 leaves `embed_tokens` and `lm_head` in 16-bit. For a 2B–4B model with a large vocab (Qwen3 ~152K, Gemma 262K), the 16-bit `lm_head` is read completely on every token and is a massive fraction of per-token HBM traffic.
2. **On-chip VPU dequant overhead.** Unpacking int4 nibbles, shifts/masks, and multiplying by bf16 group scales inside VMEM adds pipeline latency vs native int8 or bf16 loads.
3. **Long-context KV reads (32K+).** Decode attention has arithmetic intensity ≈ 8 FLOPs/byte and reads the entire active KV cache every step. For Qwen3-4B (36 layers, 8 KV heads, head_dim 128), a 32K bf16 KV cache is `32768 × 36 × 8 × 128 × 2 × 2 ≈ 4.5 GiB`. At batch 1/32K, reading 4.5 GiB per token takes ≈ **5.2 ms** of HBM bandwidth — more than reading the whole 4B model's quantized weights (~2.2 GiB INT4 ≈ 2.6 ms).

**Realistic W4A16 speedup for a 4B model at batch 1:**
- **Short context (<2K):** bf16 ~86 tok/s → W4A16 with 16-bit `lm_head` ~**1.15–1.35×** (~100–115 tok/s); up to **1.5–1.7×** only if `lm_head` is also 4-bit.
- **Long context (32K+):** bf16 KV reads dominate; end-to-end advantage shrinks to **<1.2×**, and 32K TTFT is slower than bf16 because prefill is compute-bound (W4A16 still computes bf16 at 197 TFLOPS plus dequant overhead).

## 5. Best-supported format: W8A8 INT8

**W8A8 INT8 (dynamic activation + INT8 weight, compressed-tensors, `compressed_tensors_w8a8_int8`) on the vLLM path** is the best-performing and best-supported quantized format on v5e.

Why it wins:
1. Halves weight HBM traffic vs bf16 (1 byte/param vs 2).
2. Unlocks the native **393 TOPS INT8 MXU** (2× bf16), accelerating compute-bound 32K+ prefill.

**Published TPU v5e-1 benchmarks (Oct 2026, xbill/Google Developer Experts):**
- **E2B, ~2.5B active, batch 1:** bf16 **144 tok/s** decode; W8A8 INT8 + 4-bit embeddings **243 tok/s** (**1.69×**), benchmark accuracy **0.677 vs 0.683** bf16. At ~3,600 prompt tokens, W8A8 maintains **1.38×** throughput and significantly lower TTFT.
- **12B on one v5e-1:** bf16 (22.4 GiB) **OOM**; W8A8 INT8 + 4-bit embeddings fits and serves **57 tok/s at batch 1, 216 tok/s at batch 4, 713 tok/s at batch 16**, GSM8K **96.44%**.
- **12B, 16 streams, same file size:** INT8 **624 tok/s** vs FP8 **518 tok/s**, because v5e has no native FP8 MXU and must convert FP8 weights to bf16 before each multiply.

## 6. KV cache on TPU: the fp8/int8 trap

- **`--kv-cache-dtype int8` is not supported by the TPU Ragged Paged Attention (RPA) kernel.** tpu-inference only wires `fp8`, `fp8_e4m3`, `fp8_e5m2` in its configuration surface.
- **Critical warning (Issue #3126):** `--kv-cache-dtype fp8` is accepted by the CLI and logs `INFO [cache.py:279] Using fp8 data type to store kv cache`, but on v5e/v6e in many configurations (especially torchax path, `tp_size >= num_kv_heads`, or depending on RPA kernel dispatch) it **silently allocates a bfloat16 KV cache** — zero memory savings, zero throughput change.
- **How to verify:** compare the init log `INFO [kv_cache_utils.py:...] GPU KV cache size: X tokens` for `--kv-cache-dtype auto` vs `fp8`. If X does not roughly double, you got bf16.
- **Conflict with 07:** the Best Practices doc claims `--kv-cache-dtype int8` gives true 8-bit compression (likely the TorchTPU custom ring path). The quantization guide says vLLM RPA does not support int8. Treat this as path-specific; verify in logs.

## 7. Exact flags and environment variables

- **Path selection:**
  - W8A8 INT8 (compressed-tensors), AWQ INT4, or vLLM-path W4A16: `export MODEL_IMPL_TYPE=vllm`.
  - JAX-native W4A16 (requires PR #3653): `export MODEL_IMPL_TYPE=flax_nnx` (or auto for models in `_MODEL_REGISTRY`).
- **Memory:** external benchmarks report `--gpu-memory-utilization 0.95` can trigger `CompileTimeHbmOom` on large/unbucketed models and recommend **0.90 or 0.92** (~14.17–14.49 GiB usable). **In-repo measured (`../benchmarks/TPU-v5e/tpu-final/RESULTS.md`):** on `Qwen3-4B` bf16 (`max_model_len=17408`), `0.95` ran stably at single-stream / low batch (**54,336 KV tokens, 406 MiB headroom, 84.91 tok/s**), while `0.90` (**48,576 KV tokens, 1,216 MiB headroom**) is recommended for Batch 32 concurrency.
- **Bucketing for batch-1 long context:**
  - `--max-num-seqs 1` — eliminates high-batch padding buckets, cutting startup compile time and scratchpad HBM (or `--max-num-seqs 16..32` for multi-agent council batching).
  - `--max-model-len 32768` (or `17408` / `65536`) — tpu-inference precompiles a power-of-2 token-padding ladder (`16, 32, …, 8192`); non-power-of-2 `max_model_len` (such as `17408` in `tpu-final`) is supported directly.
  - `--max-num-batched-tokens 4096` (or 8192) — chunked prefill so 32K+ prompts don't exhaust activation HBM.
  - `--quantization compressed-tensors` (or `awq`) — auto-detected from `config.json`, or pass explicitly.

## 8. Minimal smoke-test checkpoints (1B–4B, non-Gemma)

- **W8A8 INT8 (recommended for v5e):** `RedHatAI/Qwen3-4B-w8a8` or `RedHatAI/Qwen2.5-3B-Instruct-w8a8` [exact Qwen3-4B slug unverified]; `RedHatAI/Llama-3.2-3B-Instruct-w8a8` and `NeuralMagic/Llama-3.2-3B-Instruct-quantized.w8a8` follow the standard schema.
- **W4A16 INT4 (compressed-tensors):** `RedHatAI/Qwen2.5-3B-Instruct-quantized.w4a16`, `NeuralMagic/Llama-3.2-3B-Instruct-quantized.w4a16` — symmetric group-128 format tested against `_is_wNa16_group_channel`.
- **AWQ W4A16 INT4 (`--quantization awq`, supported pre-0.28.0 on `MODEL_IMPL_TYPE=vllm`):** `Qwen/Qwen3-4B-AWQ`, `Qwen/Qwen2.5-3B-Instruct-AWQ`.

## 9. Version/commit guidance and risks

- **tpu-inference 0.32.0 (released 2026-10-22):** PR #3487 (vLLM-path compressed-tensors W4A16 linear + MoE) and AWQ W4A16 are merged and included. You do **not** need main for W4A16/W8A8 on the vLLM/torchax path. PR #3653 (JAX-path W4A16 on `gmm_v2`) appears in release highlights, but an Oct 26 benchmark report says stock Docker images required cherry-picking #3299, #3653, #3660 for JAX-only MoE/Gemma-4 W4A16. #3299 is Gemma-4 `k_norm`-specific and not relevant unless using Gemma 4; #3660 is needed for JAX-path W4A16 MoE.
- **Risks of main/nightly:**
  1. On 2026-10-02 the `vllm/vllm-tpu:nightly` build broke two working 4-bit builds simultaneously.
  2. **Issue #3859** (2026-10-20): on current main, `quantize_tensor` computes scales/reciprocals in the input dtype; with float16, small scales underflow and `nan_to_num` clamps reciprocals to 65504, destroying accuracy. Always keep `--dtype bfloat16`.
  3. **Issue #3542:** without PR #3660, a non-FP8 quantized MoE on the JAX path silently serves packed int32 words as dense bf16.
  4. **LKG/vLLM version skew:** tpu-inference main frequently moves its Last Known Good upstream vLLM commit (e.g. PR #3637 AuxOutput refactor), breaking compatibility with a pre-installed vllm-tpu 0.32.0 wheel unless both are rebuilt. Stay on 0.32.0 release wheels.

## 10. Ranked recommendation for single-chip v5e-1 (batch 1, 32K+)

### Rank 1 — W8A8 INT8 (compressed-tensors) on the vLLM path (`MODEL_IMPL_TYPE=vllm`), best for 3B–8B
A 4B–8B model in INT8 takes ~4.2–8.5 GiB HBM, leaving 6.0–10.3 GiB for KV (enough for 40K–75K tokens of 16-bit KV on a 4B–8B GQA model). Native 393 TOPS INT8 cuts 32K+ prefill latency nearly in half and boosts batch-1 decode **1.4–1.7×** over bf16.
```bash
export MODEL_IMPL_TYPE=vllm
vllm serve <W8A8-compressed-tensors-repo> \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096
```

### Rank 2 — W4A16 (compressed-tensors or AWQ) on the vLLM path, best only for fitting 8B–14B
If you need a larger model (Qwen3-8B or 12B–14B) plus a 32K–64K KV cache on one 15.75 GiB chip, W8A8 leaves too little room. W4A16 shrinks an 8B model to ~4.8 GiB, leaving ~9.6 GiB for KV. Expect lower 32K prefill efficiency than W8A8 because weights widen to bf16 before every MXU multiply.

### Rank 3 — unquantized bf16, best baseline for ≤4B if zero degradation is required
`Qwen3-4B` bf16 (**7.494 GiB** measured) + 32K bf16 KV (**4.5 GiB** at 144 KiB/token) = **12.0 GiB**, well inside 14.17 GiB (at 0.90 util), measured at **84.92–88.58 tok/s** short-context (**80.4 tok/s at 4K, 69.8 tok/s at 16K** in `tpu-characterization/RESULTS.md`). At 64K+ context (~9.0 GiB KV), 7.49 + 9.0 > 14.6 GiB → OOM; W8A8 becomes mandatory.

## 11. Key sources

tpu-inference issues #3540, #3539, #3542, #3126, #3859; PRs #3487, #3653, #3660, #3299, #3637; vLLM JAX docs; xbill Medium posts on Gemma3/Gemma 4 QAT/v5e; llm-d Google Cloud TPU docs; Hugging Face fused INT8 weight-only Pallas; tpu-px measurement series.

**Caveat:** these are Oct 2026 findings for 0.32.0. The W4A16 JAX path and KV-cache dtype behavior are moving targets; verify the installed wheel and inspect logs before trusting a flag.
