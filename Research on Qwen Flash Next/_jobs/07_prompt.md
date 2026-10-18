Read `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/_jobs/SHARED_PROTOCOL.md` and follow it exactly.

You are Job 07. VRAM, KV cache, context headroom on 96 GB G4.

Write ONLY this file:
`/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/07_kv_cache_and_context.md`

Own claims **K1–K10** in CLAIMS.md.

## What you must settle

Build a VRAM bill of materials for each model on one 96 GB G4:

```
usable_vram ≈ 95.6 GB
- CUDA context / engine workspace
- weights (NVFP4)
- MTP head
- GDN recurrent state × (1 + draft_depth) × max_batch
- KV cache = ctx_tokens × layers_with_kv × bytes_per_token
- MoE extra (router, expert scratch)
- n-gram / host embeddings if real
```

For **27B NVFP4** and **Flash-Next NVFP4**:

- bytes/token of KV at FP16, FP8, INT8, using **only the full-attention layers** (Qwen 3.8 27B is 16/64). Flash-Next may differ — use its config if you can fetch it.
- max context at batch=1 after weights, for FP8 KV and FP16 KV.
- Chat: 27B → ~80 GB free, 100K+; Flash-Next → 24–26 GB free, 32K–48K at FP8. Recalculate.
- Chat: huge repos >64K–128K favor 27B. True only if Flash-Next actually OOMs there.

The 51B n-gram embedding “asynchronously prefetched from host memory” claim: if Job 02 would cover identity, you still must answer the **RAM/PCIe** question. Colab host RAM is not infinite. If the mechanism is real, estimate host RAM and PCIe traffic per token. If it is not real, say FALSE here too.

GDN state: prior research (`architecture.md`, `research/qwen38_27b_max_tokens_per_second_deep_research.md`) says ~78–154 MB per slot for 27B. Reuse that method for Flash-Next if config is available.

Recommend `--kv-cache-dtype` and a safe `-c` / `max_model_len` for an agent loop.

Do not do CU economics or engine install.
