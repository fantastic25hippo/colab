Read `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/_jobs/SHARED_PROTOCOL.md` and follow it exactly.

You are Job 06. Inference runtime stack on Colab G4.

Write ONLY this file:
`/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/06_runtime_stack.md`

Own claims **R1–R12** in CLAIMS.md.

## What you must settle

For **Qwen3.8-Flash-Next NVFP4** and **Qwen3.8-27B NVFP4**, as of Oct 2026:

| Engine | Runs this arch? | NVFP4 kernels on sm_120? | MTP / spec? | FP8 KV? | Prefix cache / GDN state? | CUDA graphs? | Colab install reality |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |

Engines: vLLM, SGLang, TensorRT-LLM, llama.cpp, NInfer if relevant.

Specific questions:

1. Does a current vLLM or SGLang release actually serve Flash-Next (MoE + GDN + QSA + MTP)? Quote docs, PRs, or cookbooks. A marketing sentence is not enough.
2. NVFP4 on Blackwell: compressed-tensors / modelopt / FlashInfer version pins.
3. TensorRT-LLM on a Colab notebook: almost certainly painful (compile, drive cache). Say so with evidence.
4. llama.cpp: GGUF path for Flash-Next? NVFP4 in ggml? DFlash 2 is a 27B-only sidecar — does it transfer?
5. `--kv-cache-dtype fp8` flag names per engine.
6. Prefix caching for hybrid GDN (`mamba-cache-mode align` etc).
7. Cold-start CU cost on Colab: pip install vLLM vs the current plan of a Drive-cached static `llama-server`. Estimate minutes.
8. Required CUDA (12.8 / 13.x) vs Colab guest driver 580.

The current locked worker is llama.cpp `efe3626` / `b11552` + DFlash 2. Switching to Flash-Next almost certainly **changes engine**. Make that explicit.

Do not invent tok/s. Point at Job 05 for numbers. Do not do CU monthly math.
