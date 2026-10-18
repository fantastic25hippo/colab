Read `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/_jobs/SHARED_PROTOCOL.md` and follow it exactly.

You are Job 03. Identity of Qwen3.8-27B-NVFP4.

Write ONLY this file:
`/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/03_qwen38_27b_nvfp4_checkpoint.md`

Own claims **D1–D12** in CLAIMS.md.

## What you must settle

Fetch:

- https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4

If missing, find the real NVIDIA / Unsloth NVFP4 27B repo. Report the actual id.

Extract from card + `config.json` + weight index:

- parameter count (chat: 27.8B dense, all activated)
- 48 GDN + 16 full attention (`full_attention_interval`)
- MTP / NextN head present?
- NVFP4 on-disk and loaded size (chat 15–17 GB)
- FP8 size (~27 GB) and BF16 (~54 GB) — compute and find official checkpoints
- context length
- how it differs from the current worker:
  - Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF Q4_K_M (~16.8 GB)
  - DFlash 2 sidecar
  - abliteration: NVIDIA checkpoint is almost certainly NOT abliterated

## Required calculations

- 27.8e9 × 0.5 B = 13.9 GB raw NVFP4; 15–17 GB loaded is plausible if embeddings/MTP/scales add 1–3 GB. Confirm from files.
- 27.8e9 × 1 B = 27.8 GB FP8; 27.8e9 × 2 B = 55.6 GB BF16.
- Bandwidth-bound decode bytes/token ≈ loaded weight footprint (plus a little KV). Leave tok/s to Job 05, but publish the byte payload they should use.

Do not audit Flash-Next, GLM, or CU tables.
