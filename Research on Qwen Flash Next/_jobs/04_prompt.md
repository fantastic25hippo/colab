Read `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/_jobs/SHARED_PROTOCOL.md` and follow it exactly.

You are Job 04. GLM-5.3-Flash-NVFP4 — can it run on Colab G4?

Write ONLY this file:
`/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/04_glm53_flash_nvfp4.md`

Own claims **G1–G8** in CLAIMS.md.

## What you must settle

Fetch:

- https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4

If 404, find the real GLM-5 / GLM-4.5 / GLM-5.3 Flash NVIDIA NVFP4 (or Zhipu) card. Naming in the chat may be wrong. Report the actual repo and the actual parameter counts.

Confirm:

- total params, active params, expert count
- NVFP4 disk and loaded VRAM
- chat formula `320B × 0.5 bytes ≈ 160–175 GB`
- 96 GB G4 fit: no
- tensor parallel 2×96 GB: yes on a real box, no on standard Colab
- CPU offload tok/s: do not accept 1–3 tok/s without a measured or well-argued PCIe bound
- API pricing on OpenRouter / Z.ai for the closest GLM Flash SKU. “$0.045 per complex agent task” is not a standard price unit — convert to $/M tokens if you find a rate, and say the task-price is UNVERIFIED if they made it up.

Stay inside GLM + Colab-fit. Do not re-do Qwen architecture.
