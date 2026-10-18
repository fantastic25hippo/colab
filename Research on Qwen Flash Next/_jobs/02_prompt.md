Read `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/_jobs/SHARED_PROTOCOL.md` and follow it exactly.

You are Job 02. Identity of Qwen3.8-Flash-Next-NVFP4.

Write ONLY this file:
`/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/02_qwen38_flash_next_checkpoint.md`

Own claims **F1–F14** in CLAIMS.md.

## What you must settle

Fetch the Hugging Face repo the chat names:

- https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4

If that exact repo is missing, search for the real NVIDIA / Qwen / Unsloth NVFP4 Flash-Next (or Qwen3.5/3.8 Next/Flash) checkpoint. Report the **actual repo id**. Do not pretend a 404 exists.

From `config.json`, `README.md`, safetensors index, and model card, extract:

- total parameters vs active parameters
- MoE: number of experts, experts-per-token (top-k), shared experts
- hidden size, layers, vocab
- attention mix: Gated DeltaNet, full attention interval, Qwen Sparse Attention (QSA) — quote the config keys
- MTP / nextn / speculative head size (chat says 4B)
- any “n-gram embedding” / 51B host-prefetch story — this is a high-bullshit-risk claim; confirm or kill it
- quantization: NVFP4 (FP4), group size, scale tensors
- on-disk size and estimated loaded VRAM
- license, gating, required transformers/vLLM version
- max context

Also look at the **upstream unquantized** Qwen Flash-Next / Qwen3.Next card if NVIDIA only publishes the quant.

## Required calculations

- NVFP4 bytes ≈ active-and-total params × 0.5 B, plus scales, embeddings, MTP, unused experts (MoE stores ALL experts, not just active).
- Chat says 68–72 GB in VRAM for 125B NVFP4. Independent calc: 125e9 × 0.5 B = 62.5 GB weights before overhead. Does 68–72 GB loaded make sense?
- Active 6B × 0.5 B = 3 GB — chat says 3.5–4 GB moved per token. Note the gap.

Do not do GPU datasheets, CU math, or GLM. Do not write speed tables except quoting the card if it publishes them.
