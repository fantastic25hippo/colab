Read `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/_jobs/SHARED_PROTOCOL.md` and follow it exactly.

You are Job 05. Speed roofline, MTP, prefix-cache. This is the core numbers audit.

Write ONLY this file:
`/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/05_speed_roofline_and_mtp.md`

Own claims **S1–S28** in CLAIMS.md.

## What you must settle

The source chat publishes **three inconsistent 27B speed tables**. List them side by side and say they cannot all be true.

Then build an independent roofline for a **Colab G4 / RTX PRO 6000 Blackwell Server Edition**. Use bandwidth **1,597 GB/s** unless Job 01’s sources (you may fetch the datasheet yourself) say otherwise. Do not wait for Job 01.

### Roofline method (required)

Batch-1 decode is memory-bound:

```
tok/s_ceiling ≈ bandwidth_GBs / (bytes_moved_per_token)
bytes_moved_per_token ≈ loaded_weight_bytes_for_activated_params
                       + KV_read_for_this_step
                       + MoE routing / extra
```

Compute ceilings for:

- 27B dense NVFP4 (~15–19 GB payload — sensitivity table)
- 27B FP8 (~27 GB)
- 27B BF16 (~54 GB)
- Flash-Next 6B-active NVFP4 (~3.5–4 GB as claimed, AND a second row that streams more because MoE still has routing/all-expert overhead)

State the **utilization** you assume (70%, 85%, 100%). Real kernels are not 100% of datasheet bandwidth.

Prefill is compute-bound. Use advertised FP4 / FP8 TFLOPS and a utilization (e.g. 40–60% for real GEMM+attention). Show why 5.5k–9.5k tok/s might or might not hold at 4K vs 16K+.

### MTP / speculative

- Native MTP on Qwen 3.8 vs DFlash 2 vs ngram-map-k4v. Do not mix them.
- Chat’s 50–60% thinking vs 75–85% code acceptance: find measured acceptance, or label UNVERIFIED.
- Mean accepted length × (1 / target_forward_ms) = speculative tok/s.

### Cache hits

- 100k–200k tok/s “cache hit bind”: this is usually **cached-token accounting**, not a real memory-bind of 37M tokens in 3 minutes. Explain what vLLM/SGLang prefix-cache / RadixAttention actually charges in wall-clock for a hit. Recalculate E/S cache-hit minutes if the 100k–200k figure is a bookkeeping artifact.

### Measured numbers

Hunt for **measured** tok/s on RTX PRO 6000 / Blackwell 96 GB for:

- nvidia/Qwen3.8-27B-NVFP4
- nvidia/Qwen3.8-Flash-Next-NVFP4
- Unsloth NVFP4 variants
- lukaLLM / RunPod / Unsloth / SGLang cookbook numbers already cited in `research/qwen38_27b_max_tokens_per_second_deep_research.md`

Mark estimated vs measured.

Concurrency 4 and 16 aggregate tok/s: find batch-throughput curves or label UNVERIFIED.

FP8 decode 5–10% faster than NVFP4 on this GPU: find a Blackwell benchmark or kill it.

## Do not

Do not compute monthly CU tables (Job 08). You may publish per-phase seconds for 2M miss / 37M hit / 1M decode so Job 08 can reuse them.
