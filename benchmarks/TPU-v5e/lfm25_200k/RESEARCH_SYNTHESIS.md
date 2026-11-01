# Research Synthesis: LFM2.5-24B-A2B on TPU v5e-1 (200K Context)

## 1. Research Files Inventory
Research lives at lfm25_200k/lfm25_200k/research/; PDFs are exports of matching mds (mtime ~3 min later):
- `deep-research-lfm25-24b-a2b-tpu-v5e-optimizations-20261031.md` | 2026-10-31 | Title: "Deep Research: Production-Grade 200K Context Single-Stream Inference Pipeline for LiquidAI/LFM2.5-24B-A2B on TPU v5e-1" | Question: How to fit 200K single-stream inference within 14.49 GiB usable memory (#3126 410MB HBM materialization, W8A8+W4A16 quantization, GGUF repacking, short-conv execution).
- `TPU v5e LLM Inference Analysis.md` | 2026-10-31 17:17 | Title: "High-Throughput Single-Stream Inference of Hybrid MoE Language Models on Google Colab TPU v5e-1" | Question: Comparative feasibility of Raon-1-mini vs LFM2.5-24B-A2B on TPU v5e-1 under 14.49 GiB ceiling, theoretical vs measured decode speeds, and XLA workspace limits.
- `TPU v5e LFM2.5-24B-A2B Inference Optimization.md` | 2026-10-31 18:03 | Title: "Architecting a 200K-Token Single-Stream Inference Pipeline for LiquidAI LFM2.5-24B-A2B on TPU v5e-1" | Question: HBM roofline modeling, in-VMEM K8V4 Pallas flash-decode kernel architecture, multi-group compressed-tensors config, and streaming GGUF-to-safetensors repacking.
- `We are designing a production-grade 200,000-token context single-stream (B=1) inference pipeline for.pdf` | 2026-10-31 18:11 | Title: "We are designing a production-grade 200,000-token context single-stream (B=1) inference pipeline" | Question: Upstream bottlenecks: preventing 410 MB HBM materialization via custom Pallas flash-decode, native INT8 MXU dot products on v5e, hybrid config, and streaming GGUF repacking.
- `We-Are-Designing-200K-TPU-Pipeline-Transcription.md` | 2026-10-31 18:14 | Title: "We are designing a production-grade 200,000-token context single-stream (B=1) inference pipeline (Transcription)" | Question: Markdown transcript of 18:11 PDF documenting upstream bug #3126 resolution, Pallas K8V4 flash-decode kernel implementation, and multi-group compressed-tensors audit.
- `TPU v5e vLLM LFM2-MoE Investigation.md` | 2026-11-01 00:53 | Title: "Technical Root-Cause Investigation and XLA Kernel Analysis for LFM2.5-24B-A2B at 200K Context on TPU v5e-1" | Question: Code-level root causes and fixes for short-conv state padding corruption (n=T in VllmModelWrapper), empty scheduling warnings, float8 vs int8 KV prefill on v5e, token bucket pruning [16, 4096, 8192], persistent XLA compile cache, and shutdown AttributeError.
- `TPU v5e vLLM LFM2-MoE Investigation.pdf` | 2026-11-01 00:56 | Title: "Technical Root-Cause Investigation and XLA Kernel Analysis for LFM2.5-24B-A2B at 200K Context on TPU v5e-1 (PDF)" | Question: PDF export of the Nov 1 00:53 investigation covering short-conv padding corruption, int8 KV prefill VPU optimization, token padding bucket pruning, and persistent XLA compile cache.

## 2. Chosen Design & Resolution of Disagreements
Where documents disagree, the newer Nov 1 00:53 investigation takes precedence over Oct 31 documents.

- **KV Cache Dtype, Layout & Upstream Bug #3126**:
  - *Chosen Design*: INT8 KV cache in Pallas ring buffer layout (2.04 GiB at 200K context, 4 attention layers * 2 * 4 heads * head_dim 128 * 200,000). If vLLM 0.32 CacheDType validation rejects int8, use `float8_e4m3fn` as interim baseline (`TPU v5e vLLM LFM2-MoE Investigation.md`, Section "2. Prefill Latency & Throughput Optimization (float8_e4m3fn vs int8)").
  - *Disagreement Resolution*: Older Oct 31 docs proposed K8V4 (1.49 GiB) or FP8 `float8_e4m3fn`. Newer Nov 1 benchmarks showed TPU v5e lacks native FP8 arithmetic hardware and emulates it on VPU, achieving only 6,137 tok/s prefill. Native INT8 on TPU v5e MXU delivers 393 TOPS and achieves 14,210 tok/s.
  - *Bypassing 410 MB HBM Materialization (#3126)*: Upstream vLLM TPU worker dequantizes KV into global HBM as bfloat16 prior to attention (`kv_cache.astype(jnp.bfloat16)`), allocating 410 MB ephemeral HBM which exceeds the 14.49 GiB ceiling. In-VMEM Pallas flash-decode streams quantized tiles directly to 16 MB VMEM registers and dequantizes on the fly, avoiding global HBM materialization (`We-Are-Designing-200K-TPU-Pipeline-Transcription.md`, Section "1. Upstream Bug #3126 & Upstream vLLM TPU PR #3487 Deep-Dive").

- **Weights Quantization & Checkpoint**:
  - *Chosen Design*: Hybrid `compressed-tensors` precision: Backbone (short-conv, dense attn, router, layernorms, embed/unembed) uses W8A8 dynamic per-token INT8 (~0.73 GiB). 38 MoE blocks (64 routed experts, top-8) use W4A16 (~11.48 GiB), totaling ~12.21 GiB resident weights (`We-Are-Designing-200K-TPU-Pipeline-Transcription.md`, Section "3. Complete Model Configuration & Weight Sizing").
  - *Group Size Resolution*: Older Oct 31 doc recommended group 128 to match 128-element VPU vector registers. Newer Nov 1 doc selected group 32 because source checkpoint `LiquidAI/LFM2.5-24B-A2B-GGUF` (`LFM2.5-24B-A2B-QAD-Q4_0.gguf`) uses block size 32; repacking directly to W4A16 group 32 avoids requantization distortion (`TPU v5e vLLM LFM2-MoE Investigation.md`, Section "1. Memory Budget and Weights Configuration").

- **Short-Conv State Padding Bug (`n=T`)**:
  - *Root Cause*: In `VllmModelWrapper._lookup_conv_cache`, missing `attn_metadata` caused fallback to bucket size `n = T` (e.g. 8192) instead of valid sequence length `prompt_len`. Zero-padding tokens rolled into `conv_state`, corrupting history and causing severe subword repetition and gibberish (`TPU v5e vLLM LFM2-MoE Investigation.md`, Section "1C. Short-Convolution State Corruption in VllmModelWrapper").
  - *Resolution*: Monkeypatch `_lookup_conv_cache` to extract `query_start_loc`, slice unpadded `x[:prompt_len]`, and check `has_initial_states` (`TPU v5e vLLM LFM2-MoE Investigation.md`, Section "3. Patch Implementation and Verification").

- **Memory Ceiling, Token Padding Buckets & Max Model Len**:
  - *Ceiling & Utilization*: 14.49 GiB usable HBM budget (16.00 GB physical minus 1.26 GiB XLA workspace). Configured with `gpu_memory_utilization = 0.92` (12.21 GiB weights + 2.04 GiB KV + 0.24 GiB conv states, leaving 0.24 GiB safety margin) (`TPU v5e LLM Inference Analysis.md`, Section "Memory Budget & Usable Capacity"; `TPU v5e vLLM LFM2-MoE Investigation.md`, Section "1. Memory Budget and Weights Configuration").
  - *Token Buckets Resolution*: Older Oct 31 docs used 10 power-of-2 buckets [16..8192] consuming 1.8 GiB XLA buffer memory and 12+ min compile time. Newer Nov 1 doc pruned buckets to 3: `[16, 4096, 8192]`, cutting compilation time by 70% and freeing 1.1 GiB HBM (`TPU v5e vLLM LFM2-MoE Investigation.md`, Section "4. Token Padding Buckets Optimization").
  - *Max Model Len*: 200,000 tokens for single-stream 200K context (`deep-research-lfm25-24b-a2b-tpu-v5e-optimizations-20261031.md`, Section "Executive Summary").

- **Startup, Expert Stacking & Persistent Cache**:
  - *Expert Stacking & GMM v2*: Stack 64 experts along dim 0 during weight load into `[64, hidden_dim, ffn_dim]` to eliminate 7,296 individual tensor allocations and dispatch to Pallas `gmm_v2` via `WNA16FusedMoEMethod` (`We-Are-Designing-200K-TPU-Pipeline-Transcription.md`, Section "2. MoE Expert Stacking and GMM v2 Kernel Integration").
  - *Persistent Cache*: Persistent XLA compile cache (`JAX_COMPILATION_CACHE_DIR=/tmp/xla_vllm_cache`, `TF_PERSISTENT_CACHE_DIRECTORY=/tmp/xla_vllm_cache`) drops engine init from ~503s to ~15.2s (`TPU v5e vLLM LFM2-MoE Investigation.md`, Section "5. Engine Startup & Compilation Optimization").

## 3. Gap Analysis Table

| Setting | Research Says | Code Does | Match/Gap | File:Line | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| KV dtype/layout/GiB | INT8 (or FP8 interim) in Pallas ring buffer; 2.04 GiB at 200K | Defaults to "auto (vLLM default, bf16)"; needs 6.68 GiB; patches allow fp8/int8/k8v4 | GAP | `lfm25_engine_subrunner.py:73-76`, `tpu_v5e_patches.py:313-337` | MEASURED |
| #3126 410MB bf16 materialization | In-VMEM Pallas flash-decode unpacks KV in VMEM registers, avoiding HBM alloc | `k8v4_pallas_kernel.py` exists in repo but is never imported or called on execution path | GAP | `k8v4_pallas_kernel.py:11-45`, `AUDIT.md:20,26` | CODE / MEASURED |
| Weights scheme/group/GiB/repo | W8A8 dynamic backbone (~0.73 GiB) + W4A16 group-32 MoE (~11.48 GiB); ~12.21 GiB total from `LiquidAI/LFM2.5-24B-A2B-GGUF` QAD-Q4_0 | W8A8 dynamic backbone + W4A16 group-128 MoE; loads 9 safetensors shards (13.13 GB) from Hub repo `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU`. Scheme matches; group 128 vs chosen 32 is the gap | PARTIAL GAP | `config_builder.py:25-66`, `lfm25_coldstart_driver.py:104` | CODE / MEASURED |
| Short-conv n=T patch | Monkeypatch `_lookup_conv_cache` to slice unpadded `x[:prompt_len]` via `query_start_loc` | `_tpu_xla_short_conv_forward` and `_lookup_conv_cache` patch valid length and conv reset | MATCH | `tpu_v5e_patches.py:41-91,136-181,307-310` | CODE |
| gpu_memory_utilization | Set to 0.92 (allocates 14.49 GiB usable HBM budget) | Sets `gpu_memory_utilization=0.92` in `LLM()` args | MATCH | `lfm25_engine_subrunner.py:75`, `run_results_run1.json:76` | MEASURED |
| 14.49 GiB ceiling | Hard resident ceiling: 12.21 GiB weights + 2.04 GiB KV + 0.24 GiB states within 15.75 GiB | Total limit is 15.748 GiB; peak reached 12.447 GiB, leaving 3.57 GiB free for KV cache | MATCH | `run_results_run1.json:95-100` | MEASURED |
| Buckets | Pruned to 3 buckets: `[16, 4096, 8192]` (saves 1.1 GiB XLA buffer memory) | Pruned to 2 buckets: `[16, 8192]` via `_wrap_paddings` (`[16, 4096]` if `LFM_DIAG=1`) | PARTIAL GAP | `tpu_v5e_patches.py:185-192`, `run_results_run1.json:79-82` | MEASURED |
| max_model_len | Configured to 200,000 tokens for 200K single-stream context | Sets `max_model_len=200000` in `LLM()`; config `max_position_embeddings=200000` | MATCH | `lfm25_engine_subrunner.py:74`, `config_builder.py:100` | MEASURED |
| Expert stacking | Stack 64 experts along dim 0 to eliminate 7,296 individual tensor allocations | Stacking patch `_stacked_device_put` buffers 64 expert tensors and performs stacked put | MATCH | `tpu_v5e_patches.py:247-306` | CODE |
| gmm_v2 | Dispatches stacked W4A16 experts to Pallas `gmm_v2` kernel via `WNA16FusedMoEMethod` | Not referenced or implemented in active runner/patches; uses standard linear methods | GAP | `tpu_v5e_patches.py:266-283` | INFERRED |
| Compile cache | Persistent XLA cache via `JAX_COMPILATION_CACHE_DIR=/tmp/xla_vllm_cache` | Code uses `/dev/shm/tpu_compile_cache` and VLLM_XLA_CACHE_PATH; MIN_COMPILE_TIME_SECS is 0 not 1 | PARTIAL | `lfm25_engine_subrunner.py:15-21` | MEASURED |
| Run 1 6.68 vs 3.57 GiB | 200K KV cache requires 2.04 GiB at FP8/INT8, fitting inside 3.57 GiB free window | Subrunner defaulted to bf16 KV (needs 6.68 GiB), crashing with ValueError (3.57 GiB free) | GAP | `run_results_run1.json:78,100`, `lfm25_engine_subrunner.py:73-76` | MEASURED |

## 4. Ranked Smallest Changes to START at 200K then RUN
Ranked strictly by unblock-start, then correctness, then speed. Do not implement.

1. **Unblock-Start (P0: Resolve Run 1 OOM Crash on Startup - Single LLM() Change)**
   - *Change*: Re-enable `kv_cache_dtype="fp8"` on `LLM()` in `lfm25_engine_subrunner.py:73-76` (subrunner.py:73 comment: fp8 is NOT re-enabled; auto->bf16). vLLM 0.32 CacheDType may reject plain int8, so try fp8 first, int8 second. Update `CACHE_KEY_ARGS` `kv=fp8` in `lfm25_engine_subrunner.py:30-31`.
   - *Impact*: Reduces required KV cache memory at 200,000 tokens from 6.68 GiB (bf16) to 2.04 GiB, fitting within the 3.57 GiB available HBM. This is the only change required to START (2.04 GiB vs 6.68).
   - *Files*: `lfm25_engine_subrunner.py`.

2. **Correctness for RUN (P1: Prevent Upstream #3126 410MB HBM Materialization)**
   - *Change*: Hook in-VMEM unpack (`k8v4_pallas_kernel.py` / custom Pallas flash-decode) into active attention dispatch only if fp8/int8 still materializes 410MB bf16 (#3126).
   - *Note*: Do not treat group-32 repack or restoring bucket 4096 as start blockers: weights already loaded in Run 1; code buckets [16, 8192] are newer than the Nov 1 [16, 4096, 8192] research.
   - *Impact*: Prevents catastrophic 410 MB HBM materialization during attention execution on 200K sequences.
   - *Files*: `tpu_v5e_patches.py`, `k8v4_pallas_kernel.py`.

3. **Speed & Latency (P2: Published Compile Cache & Kernel Acceleration)**
   - *Note*: `LFM_CACHE_FORCE_COLD` is a run flag, not a source edit.
   - *Change*: Use published compile cache in `/dev/shm/tpu_compile_cache` with warmed XLA executables matching `[16, 8192]` to accelerate cold start.
   - *Change*: Integrate Pallas `gmm_v2` / `WNA16FusedMoEMethod` for stacked MoE expert dispatch for speed (not start).
   - *Files*: `lfm25_engine_subrunner.py`, `tpu_v5e_patches.py`.
