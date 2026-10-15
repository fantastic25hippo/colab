# Pinning Hugging Face Artifacts for Blackfrost Qwen 3.8 27B Abliterated (`llama.cpp` + DFlash 2 Sidecar)

**Date Researched:** October 14, 2026 | **Research Depth:** Deep (Multi-Hop Repository & GGUF Header Audit) | **Target Runtime:** `llama.cpp` (`llama-server` / `llama-quantize` with DFlash 2 PR #27342 support) on 24–96 GB NVIDIA GPUs (L4, A100 40GB/80GB, RTX PRO 6000 Blackwell / G4 96GB)

**Executive Summary:** The primary `llama.cpp` target model (`Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf`, **16.8 GB** / **16,810,716,384 bytes**, SHA-256 `5d53637a...5509`) and the metadata-matched BF16 DFlash 2 draft sidecar (`dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`, **3.86 GB** / **3.60 GiB**, SHA-256 `ca6dacc0...4fb3`) are both live at [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain) (HEAD commit `994bb4e`). A pre-quantized `Q4_K_M` DFlash 2 sidecar is **`ABSENT`** from the Blackfrost repository; first boot must either run `llama-quantize` on the downloaded 3.86 GB BF16 sidecar to produce a universal **\~1.14 GB** (`1,143,006,592 bytes`) `Q4_K_M` sidecar that fits every GPU SKU (L4 24GB through A100/G4 96GB), or retain the 3.86 GB BF16 file only on 80–96 GB GPUs. Every Blackfrost main GGUF quant also natively embeds Qwen 3.8's 65th-block Multi-Token Prediction (MTP) head (`qwen35.block_count = 65`, `qwen35.nextn_predict_layers = 1`), providing a zero-sidecar fallback via `--spec-type draft-mtp` ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)).

## 6. Google Drive Packing List (Primary Shipping Manifest)

**Human Operator Storage Gate — Check Google Drive Free Space First:** Before initiating downloads or first-boot quantization, the human operator **must verify available Google Drive quota**. A standard free 15 GB Google Drive account cannot hold even the single `Q4_K_M` target file (`16.81 GB`). You require a Google One or Google Workspace Drive volume with at least **22.0 GB of free space** (plus ensure your Google Drive Trash is emptied, as trashed LFS artifacts still count against storage quota) to stage the target model, the BF16 DFlash 2 source sidecar, and the locally quantized `Q4_K_M` DFlash 2 output sidecar simultaneously.

| Local Filename on Drive | Source Repository & Revision / Commit | Direct Source URL | Exact Size (Decimal GB / Bytes / GiB) | SHA-256 Checksum | One-Time Action | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` | [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain) (`main` @ `994bb4e`; file commit [4f6732c](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fblame%2Fmain%2FQwen3.8-27B-ABLITERATED-Q4_K_M.gguf), 2026-08-16) | [Resolve URL](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fresolve%2Fmain%2FQwen3.8-27B-ABLITERATED-Q4_K_M.gguf) | **16.8 GB** (`16,810,716,384` B / `15.66 GiB`) | `5d53637a59cfcd3a4d8354e254ffd44943e5a693da2405a3e228c62962355509` | **Download** directly to Drive; verify SHA-256 against `SHA256SUMS.txt` | 
| `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` | [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain) (`main` @ `994bb4e`; file commit [b669c83](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fcommits%2Fmain%2FREADME.md), 2026-08-22) | [Resolve URL](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fresolve%2Fmain%2Fdflash-Qwen3.8-27B-ABLITERATED-BF16.gguf) | **3.86 GB** (`~3,860,292,992` B / `3.60 GiB`) | `ca6dacc0bba87332b4eae6f103fa4c0c800ca871657e83d2aed658351b2c4fb3` | **Download** to Drive; feed into `llama-quantize` on first boot (or serve directly on 80–96 GB GPUs) | 
| `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` | `ABSENT` on Hugging Face (generated locally from `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`) | Local build from [Blackfrost BF16 Sidecar](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fresolve%2Fmain%2Fdflash-Qwen3.8-27B-ABLITERATED-BF16.gguf) | **\~1.14 GB** (`~1,143,006,592` B / `1.06 GiB`) | Generated on first boot (record local `sha256sum`) | **Quantize** once via `llama-quantize ... Q4_K_M` and save to Drive as the universal sidecar | 
| `SHA256SUMS.txt` | [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain) (commit [b669c83](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fcommits%2Fmain%2FREADME.md), 2026-08-22) | [Resolve URL](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fresolve%2Fmain%2FSHA256SUMS.txt) | **1.02 kB** (`1,020` B) | Verified via Git commit `b669c83` | **Download** and run `sha256sum -c --ignore-missing SHA256SUMS.txt` | 
| `bin/llama-server` & `bin/llama-quantize` | [ggml-org/llama.cpp PR #27342](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fpull%2F27342) (Blackfrost pinned commit `1deefcca395743049c3820ab8f9b15043f3e9446` or post-2026-08-27 `master`) | [Blackfrost Repo Instructions](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF) | **\~0.12 GB** (`~120,000,000` B) | Build-specific (CUDA architecture dependent) | **Compile** once with `-DGGML_CUDA=ON` and cache binaries on Drive | 

### Total Drive Footprint Breakdown

* **Peak Staging Footprint (Target `Q4_K_M` + Source `BF16` DFlash 2 + Quantized `Q4_K_M` DFlash 2 + Checksums + Binaries):** **`~21,934,016,988 bytes`** (**`~21.93 GB` decimal** / **`~20.43 GiB` binary**), comprising `16,810,716,384` bytes (`Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf`) + `~3,860,292,992` bytes (`dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`) + `~1,143,006,592` bytes (`dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf`) + `1,020` bytes (`SHA256SUMS.txt`) + `~120,000,000` bytes (compiled `llama.cpp` binaries) ([Blackfrost File Tree](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain), [Akicou DFlash2 GGUF Byte Table](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FAkicou%2FQwen3.8-27B-DFlash2-GGUF)).

* **Steady-State Universal SKU Footprint (Target `Q4_K_M` + Quantized `Q4_K_M` DFlash 2 Sidecar + Binaries, after deleting `BF16` source sidecar):** **`~18,073,723,996 bytes`** (**`~18.07 GB` decimal** / **`~16.83 GiB` binary**).

* **Steady-State High-VRAM Footprint (Target `Q4_K_M` + Unquantized `BF16` DFlash 2 Sidecar + Binaries, if skipping quantization on 80–96 GB GPUs):** **`~20,791,010,396 bytes`** (**`~20.79 GB` decimal** / **`~19.36 GiB` binary**).

## 1. Blackfrost Target GGUF (`Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`)

### Repository Confirmation & Lineage

The canonical, active repository is confirmed as [https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF) (`main` branch HEAD commit `994bb4e`, 23 commits total, total repository size **165 GB** across all ladder files) ([Blackfrost File Tree](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain)). It is a standard K-quant GGUF ladder converted directly from the weight-level abliterated parent checkpoint [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-BF16) (18 BF16 safetensors shards, `55,586,124,697` repository bytes), which in turn derives from [Qwen/Qwen3.8-27B](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-BF16) under the Apache-2.0 license. No IQ/IK or importance-matrix (`imatrix`) quants are published in this repository ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)).

### Complete Repository File List & Target 4-Bit Selection

Opening the actual file tree at [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/tree/main](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain), [SHA256SUMS.txt](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain), and the Git LFS pointer history reveals **12 GGUF files** (9 main text models, 1 BF16 DFlash 2 sidecar, and 2 multimodal vision projectors) alongside 4 repository metadata/deployment entries:

| Exact Filename | File Role | Revision / Commit Hash | Commit Date | Listed Size (Exact Bytes where published) | Published SHA-256 (`SHA256SUMS.txt` / LFS Pointer) | Selection Status | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| `Qwen3.8-27B-ABLITERATED-Q2_K.gguf` | Main 2-bit K-quant (+MTP) | `4f6732c` | 2026-08-16 | **10.9 GB** | `52788bf01bf59457c3aec96e1183f0328b8062d4f6f12679f4bf48d16f8c39c0` | Skip (severe quality trade-off) | 
| `Qwen3.8-27B-ABLITERATED-Q3_K_S.gguf` | Main 3-bit small K-quant (+MTP) | [4f6732c](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fblob%2Fmain%2FQwen3.8-27B-ABLITERATED-Q3_K_S.gguf) | 2026-08-16 | **12.3 GB** | `a6269b9c4ff5b214d01a95f1c20b5f25136ba89a8c8cfbb832bb6a49614e06b8` | Skip (tight-VRAM only) | 
| `Qwen3.8-27B-ABLITERATED-Q3_K_M.gguf` | Main 3-bit medium K-quant (+MTP) | `4f6732c` (initial [e44855a](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fcommit%2Fe44855a243311868b4aca940d7b132190b7c35af)) | 2026-08-16 | **13.5 GB** (pre-MTP `13,301,444,800` B) | `18580f97cc179c2042ccca8ec52fa32921255ade624b5e28b69733c36ce4de2f` | Skip | 
| `Qwen3.8-27B-ABLITERATED-Q4_K_S.gguf` | Main 4-bit small K-quant (+MTP) | `4f6732c` | 2026-08-16 | **15.8 GB** (pre-MTP `15,586,316,480` B) | `af5cab19cfdf0ee54e7bb23600fee1c753e4f8ff2efbbeb7f8b54d1acdeac36a` | Skip (lower-memory 4-bit) | 
| **`Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf`** | **Primary 4-bit medium K-quant (+MTP)** | [4f6732c](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fblame%2Fmain%2FQwen3.8-27B-ABLITERATED-Q4_K_M.gguf) (initial `c8a221a`) | **2026-08-16** | **16.8 GB** (**`16,810,716,384` B** / `15.66 GiB`) | **`5d53637a59cfcd3a4d8354e254ffd44943e5a693da2405a3e228c62962355509`** | **DOWNLOAD (Locked Primary Target)** | 
| `Qwen3.8-27B-ABLITERATED-Q5_K_S.gguf` | Main 5-bit small K-quant (+MTP) | `4f6732c` | 2026-08-16 | **19 GB** (`19.0 GB`) | `ed58ad7c506f20eeda9d599446f858c37630e79cdec360bb5ada36fd7c92950c` | Skip | 
| `Qwen3.8-27B-ABLITERATED-Q5_K_M.gguf` | Main 5-bit medium K-quant (+MTP) | `4f6732c` | 2026-08-16 | **19.5 GB** | `37e02edc312a35e73eca50ae26d72458f341e17427eeb1724d9927723256c49c` | Skip | 
| `Qwen3.8-27B-ABLITERATED-Q6_K.gguf` | Main 6-bit K-quant (+MTP) | [4f6732c](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fblob%2Fmain%2FQwen3.8-27B-ABLITERATED-Q6_K.gguf) | 2026-08-16 | **22.4 GB** | `34264197c0188b1b09da05692fcdd4625b7d2231d40b07a4b29836bdb7b4d2a1` | Skip | 
| `Qwen3.8-27B-ABLITERATED-Q8_0.gguf` | Main 8-bit quant (+MTP) | `4f6732c` (initial [eb696a3](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fcommit%2Feb696a3b71978eeaeb6451c190f6323223d14fd0)) | 2026-08-16 | **29 GB** (`29.0 GB`; pre-MTP `28,595,765,440` B) | `c01b61cf17a2ca61ed1476d0910f652388279fa91f0c223721a391457787cf39` | Skip | 
| **`dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`** | **DFlash 2 5-layer BF16 sidecar** | [b669c83](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fcommits%2Fmain%2FREADME.md) | **2026-08-22** | **3.86 GB** (`3.60 GiB`) | **`ca6dacc0bba87332b4eae6f103fa4c0c800ca871657e83d2aed658351b2c4fb3`** | **DOWNLOAD (Source for `Q4_K_M` Sidecar)** | 
| `mmproj-Qwen3.8-27B-ABLITERATED-F16.gguf` | Full-fidelity F16 vision projector | [811fa2f](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fcommits%2F811fa2f6376c2c3e852577e959577c22b87e68ac) | 2026-08-14 | **928 MB** (`0.93 GB`) | Not listed in `SHA256SUMS.txt` | Skip (text-only / DFlash 2 incompatible) | 
| `mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf` | Compact Q8_0/F16 vision projector | [bfcedff](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fblob%2Fmain%2Fmmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf) | 2026-08-14 | **629 MB** (`0.63 GB`) | `6b8b9c95513e0798f4d48f467bced39491da62f418249bfb7ef12a857d15e5be` | Skip (text-only / DFlash 2 incompatible) | 
| `SHA256SUMS.txt` | Checksum manifest for 9 quants + DFlash2 | [b669c83](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fcommits%2Fmain%2FREADME.md) | 2026-08-22 | **1.02 kB** | N/A (text manifest) | Download for verification | 
| `README.md` | Model card documentation | [b669c83](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fblame%2Fmain%2FREADME.md) | 2026-08-22 | **13.7 kB** (`13,712` B) | N/A | Reference only | 
| `GGUF_conversion_agent_instructions.md` | Self-conversion guide | `994bb4e` | 2026-08-22 | **8.75 kB** | N/A | Skip | 
| `deploy/` (`serve.sh`, `DEPLOYMENT.md`) | One-command launch scripts | [b669c83](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fcommits%2Fmain%2Fdeploy%2Fserve.sh) | 2026-08-22 | Directory | N/A | Optional reference | 

### Embedded 65th-Block MTP Head (`--spec-type draft-mtp` Fallback)

**Yes — confirmed in both the model card and the live GGUF file header.** On August 16, 2026 (commit [4f6732c](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fcommits%2Fmain%2FREADME.md), *"Embed native MTP head in every GGUF quant"*), Blackfrost rebuilt all nine main GGUF files to embed Qwen 3.8's native 65th NextN/MTP draft block directly inside each main model file, eliminating the separate `mtp-` sidecar files pushed on August 15 (`648b4e1`) ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)).

> "Every main GGUF contains Qwen3.8's native 65th NextN/MTP block. Load only the selected model quant and enable MTP speculation directly: ... `--spec-type draft-mtp --spec-draft-n-max 3` ... Do not pass `--spec-draft-model`: the MTP head is already inside the main file. The rebuilt Q4_K_M canary was verified with llama.cpp as a 65-block model (`n_layer=64`, `n_layer_all=65`) and produced measurable native drafting: 14 of 21 drafted tokens accepted (66.7%) in the release smoke test." — [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF README](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fblame%2Fmain%2FREADME.md)

Inspecting the GGUF header metadata of `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` directly confirms `qwen35.block_count = 65` and `qwen35.nextn_predict_layers = 1` across `866` total tensors ([Blackfrost Q4_K_M File Page](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain)). If DFlash 2 ever fails to load on an updated `llama.cpp` binary, `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` can immediately fall back to single-file native MTP drafting with `--spec-type draft-mtp` and no `--spec-draft-model` flag.

### Architecture Metadata (`64` Layers, `full_attention_interval = 4`, `48` GDN + `16` Full)

The Blackfrost model card prose quotes the high-level architecture as:

> "Architecture: Qwen3.8 dense hybrid VLM · 64 text layers · Gated DeltaNet + full attention · 27-layer vision tower" — [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF Specifications](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)

Inspecting the actual GGUF key-value header table embedded inside `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (`46` KV pairs, `866` tensors) confirms the exact architectural metadata ([Blackfrost Q4_K_M GGUF Metadata](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain)):

* `general.architecture`: `qwen35`

* `general.file_type`: `Q4_K_M`

* `qwen35.block_count`: `65` (`n_layer = 64` text decoder blocks + `1` embedded NextN/MTP block)

* `qwen35.nextn_predict_layers`: `1`

* `qwen35.full_attention_interval`: `4`

* `qwen35.context_length`: `262144`

* `qwen35.embedding_length`: `5120` | `qwen35.feed_forward_length`: `17408`

* `qwen35.attention.head_count`: `24` | `qwen35.attention.head_count_kv`: `4` | `key_length` / `value_length`: `256`

* `qwen35.rope.freq_base`: `10000000` | `qwen35.rope.dimension_count`: `64` | `qwen35.rope.dimension_sections`: `[11, 11, 10, 0]`

* `qwen35.ssm.conv_kernel`: `4` | `qwen35.ssm.state_size`: `128` | `qwen35.ssm.group_count`: `16` | `qwen35.ssm.time_step_rank`: `48` | `qwen35.ssm.inner_size`: `6144`

Because `qwen35.full_attention_interval = 4` over 64 text blocks (`n_layer = 64`), every fourth layer is a grouped-query full-attention block ($64 / 4 = 16$ full-attention layers that maintain a KV cache) and the remaining $64 - 16 = 48$ layers are Gated DeltaNet (GDN) linear-attention recurrence blocks with constant state size, matching upstream [Qwen/Qwen3.8-27B](https://www.google.com/search?q=https%3A%2F%2Frecipes.vllm.ai%2FQwen%2FQwen3.8-27B) and [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4). Note that while `qwen35.full_attention_interval = 4` and `qwen35.block_count = 65` are explicitly verified in the GGUF header table, the Blackfrost card prose does not print the literal strings `"48"` and `"16"` — first boot should confirm the 48 GDN + 16 full-attention layer allocation in `llama-server` initialization logs.

### Vision Projector (`mmproj`) Files: Required or Skip?

**Skip — do not download either `mmproj` file.**

1. **Text-Only Guidance:** The Blackfrost model card explicitly instructs: *"Text only: omit `--mmproj` and do not download a projector."* ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)).

2. **DFlash 2 + Vision Incompatibility:** Community benchmarks on `llama.cpp` PR #27342 ([r/LocalLLaMA PR #27342 Thread](https://www.google.com/search?q=https%3A%2F%2Fwww.reddit.com%2Fr%2FLocalLLaMA%2Fcomments%2F1w00ngm%2Fsupport_for_dflash2_in_llamacpp_has_been_merged%2F), [r/LocalLLaMA 3-Day DFlash 2 Benchmark](https://www.google.com/search?q=https%3A%2F%2Fwww.reddit.com%2Fr%2FLocalLLaMA%2Fcomments%2F1vvncyh%2Fi_benchmark_dflash_2_pr_build_in_llamacpp_on_qwen%2F)) and [HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FHermiHg%2FQwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF) document that loading `--mmproj` alongside `--spec-type draft-dflash` fails during multimodal processing unless the drafter carries the `dflash.rope.dimension_sections` M-RoPE metadata fix and a patched `llama.cpp` vision branch. Because Blackfrost's sidecar was built on August 22, 2026 (`b669c83`) prior to the M-RoPE metadata fix, `mmproj-Qwen3.8-27B-ABLITERATED-F16.gguf` (928 MB) and `mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf` (629 MB) must be omitted.

## 2. Blackfrost DFlash 2 Sidecar

### Exact Filename, Size, Repository & GGUF Header Metadata

* **Repository:** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain) (`main` branch)

* **Exact Filename:** `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`

* **Commit & Date:** Commit `b669c83bd32ae324b32064144d8cf2be40e4e57b` (short: [b669c83](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fcommits%2Fmain%2FREADME.md), *"Add BF16 DFlash2 GGUF speculative sidecar"*), committed **August 22, 2026**.

* **Exact Size:** Listed as **`3.86 GB`** in the Hugging Face file tree and **`3.86 GB / 3.60 GiB`** in the model card specifications table (`~3,860,292,992 bytes` for an 81-tensor BF16 DFlash 2 GGUF) ([Blackfrost File Tree](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain), [Akicou DFlash2 GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FAkicou%2FQwen3.8-27B-DFlash2-GGUF)).

* **SHA-256 Checksum:** `ca6dacc0bba87332b4eae6f103fa4c0c800ca871657e83d2aed658351b2c4fb3` ([SHA256SUMS.txt](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain)).

### Pre-Quantized `Q4_K_M` Sidecar Availability

* **Status in Blackfrost Repo:** **`ABSENT`**.

* Inspection of [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/tree/main](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain) confirms that `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` is the **only** DFlash 2 file published by Blackfrost. No `Q4_K_M`, `Q4_K_S`, or `Q8_0` DFlash 2 sidecar exists in the repository.

* **Required First-Boot Action:** To deploy a `Q4_K_M` metadata-matched sidecar, **first boot must run `llama-quantize dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf Q4_K_M`**.

### Metadata Matching & Target Layer Fusion (`5, 19, 33, 47, 61` vs. GGUF `[6, 20, 34, 48, 62]`)

The Blackfrost model card explicitly documents that `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` is converted from [incoai/Qwen3.8-27B-DFlash2](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2) with the Blackfrost target model's metadata:

> **DFlash2 source:** `incoai/Qwen3.8-27B-DFlash2 · Apache-2.0 · converted to GGUF with this target's metadata` "It uses the selected main quant as the target and a dedicated 5-layer BF16 GGUF conversion of `incoai/Qwen3.8-27B-DFlash2` as the draft model. The sidecar adds 3.86 GB (3.60 GiB) of weights and retains the target model's tokenizer, chat template, and native 262,144-token context metadata." — [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)

Regarding the hidden-state fusion layers (`5, 19, 33, 47, 61`):

* **Upstream `config.json` (0-indexed):** In [z-lab/Qwen3.8-27B-DFlash2/blob/main/config.json](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fz-lab%2FQwen3.8-27B-DFlash2%2Fblob%2Fmain%2Fconfig.json) (commit `ac04198`, mirror of `incoai/Qwen3.8-27B-DFlash2`), `dflash_config.target_layer_ids` is explicitly defined as **`[5, 19, 33, 47, 61]`** (0-indexed target decoder blocks whose hidden states are concatenated into the $5 \times 5,120 = 25,600$-dimensional `fc` feature projection, as also documented in [syvai/Qwen3.8-27B-DFlash2-W4A16](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fsyvai%2FQwen3.8-27B-DFlash2-W4A16), [Akicou/Qwen3.8-27B-DFlash2-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FAkicou%2FQwen3.8-27B-DFlash2-GGUF), and [SGLang SpecForge](https://www.google.com/search?q=https%3A%2F%2Fsgl-project.github.io%2FSpecForge%2Frecipes%2Fqwen3.8-27b-dflash2-disaggregated.html)).

* **Inspected Blackfrost GGUF Header (1-indexed):** Opening the live GGUF metadata table on `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (`36` KV pairs, `81` tensors) shows how `convert_hf_to_gguf.py` serializes those exact 5 layers into GGUF 1-based indexing ([Blackfrost File Tree](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain)):

  * `general.architecture`: `dflash`

  * `general.name`: `Qwen3.8-27B-ABLITERATED-DFlash2-BF16`

  * `general.file_type`: `BF16` | `general.size_label`: `1.9B`

  * `dflash.block_count`: `5` | `dflash.context_length`: `262144`

  * `dflash.embedding_length`: `5120` | `dflash.feed_forward_length`: `17408`

  * `dflash.attention.head_count`: `32` | `dflash.attention.head_count_kv`: `8` | `dflash.attention.causal`: `false`

  * `dflash.attention.sliding_window`: `2048` | `dflash.attention.sliding_window_pattern`: `[true, true, true, true, true]`

  * `dflash.block_size`: `8` | `dflash.conv_kernel_size`: `2` | `dflash.conv_group_size`: `16`

  * `dflash.selector_rank`: `256` | `dflash.selector_top_k`: `16`

  * `dflash.target_layers`: **`[6, 20, 34, 48, 62]`** (1-indexed representation of 0-indexed layers `[5, 19, 33, 47, 61]`)

### Measured DFlash 2 Acceptance on the Abliterated Pair

The **only** published DFlash 2 acceptance and speed measurement on the Blackfrost abliterated checkpoint is Blackfrost's own **109.31 tok/s median (`n=3` prompts, 256 tokens each, `n_max=3`) server-edition Blackwell (G4 / NVIDIA RTX PRO 6000 Blackwell Server Edition 96 GB) validation run** pairing `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` with the **BF16** sidecar (`dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`) at full `262,144` context (`-ctk q8_0 -ctv q8_0`, `--spec-draft-type-k f16 --spec-draft-type-v f16`) ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)):

| Validation Prompt (`n=3`, 256 tokens, deterministic) | Decode Speed (`tok/s`) | Accepted / Drafted Tokens | Acceptance Rate | 
| ----- | ----- | ----- | ----- | 
| Speculative decoding explanation | 104.47 | 156 / 295 | 52.9% | 
| Production event-logging design | 109.31 | 157 / 292 | 53.8% | 
| TLS 1.3 handshake explanation | 118.60 | 165 / 269 | 61.3% | 
| **Median / Weighted Total (Blackfrost `n=3` G4/Blackwell Run)** | **109.31** | **478 / 856** | **55.8%** | 

**Scope Constraint:** This 55.8% acceptance (109.31 tok/s median) is strictly labeled as Blackfrost's `n=3` (3-prompt, `n_max=3`) 96 GB Blackwell Server Edition run using the **BF16** sidecar and must **not** be generalized to other hardware, longer generations, higher `n_max` values (e.g., `n_max=5` or `7`), non-greedy sampling, or the quantized `Q4_K_M` sidecar ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)). No published acceptance measurements exist yet for a quantized `Q4_K_M` derivative of `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`.

### Deployment Strategy Across 40–96 GB GPUs: Keep BF16 vs. Quantize to `Q4_K_M` for a Single Drive Artifact

Because only `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (`3.86 GB` / `3.60 GiB`) is published on Hugging Face, operators deploying across 40–96 GB GPUs (A100 40GB, A100 80GB, G4/RTX PRO 6000 96GB) and L4 (24 GB) face a choice between keeping the BF16 sidecar or quantizing once to `Q4_K_M` (`~1.14 GB` / `1.06 GiB`):

* **Recommendation — Quantize Once to `Q4_K_M` for a Single Universal Drive Artifact:** You should **still quantize `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` to `Q4_K_M` (`~1.14 GB`) on first boot and standardize on the `Q4_K_M` sidecar as your single Drive deployment artifact across every SKU (24 GB L4, 40 GB A100, 80 GB A100, and 96 GB G4/Blackwell)**.

  1. **Lower Memory-Bandwidth Tax Per Verification Cycle:** Drafting is memory-bandwidth bound (`hanxiao`). Every block-drafting step streams the entire 5-layer draft transformer from GPU memory. Quantizing the drafter from BF16 (`3.60 GiB`) to `Q4_K_M` (`1.06 GiB`) eliminates **2.54 GiB of HBM/GDDR6 reads per verification step** without reducing acceptance ([incoai DFlash2 GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF), [hanxiao L4 Analysis](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)).

  2. **Critical Context & Graph Headroom on A100 40GB and L4 24GB:** Combined weights for Target `Q4_K_M` (`16.8 GB`) + BF16 DFlash 2 (`3.86 GB`) occupy **20.66 GB** before KV caches, compute buffers, and per-shape CUDA graph allocations. On L4 (24 GB), `20.66 GB` leaves insufficient headroom for long context and triggers `cudaGraphInstantiate` OOMs ([hanxiao L4 Analysis](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)). On A100 40GB, shrinking the drafter from `3.86 GB` to `1.14 GB` reclaims **2.72 GB of VRAM**, which buys **\~44,500 additional tokens of F16 KV cache** (at `64 KiB/token` across the 16 full-attention layers) or **\~89,000 tokens of Q8_0 KV cache** ([Umesh Malik KV Math](https://www.google.com/search?q=https%3A%2F%2Fumesh-malik.com%2Fblog%2Fqwen3-8-27b-vram-kv-cache-math)).

  3. **When Keeping BF16 is Acceptable:** On **A100 80GB** or **G4 / RTX PRO 6000 96GB**, the `20.66 GB` combined footprint easily fits alongside a full `262,144`-token KV cache (`16 GiB` at F16 or `8 GiB` at Q8_0). Serving `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` directly on 80–96 GB GPUs is completely acceptable if you want zero first-boot quantization overhead and exact parity with Blackfrost's `109.31 tok/s` validation setup ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)).

## 3. Quantizing the BF16 Sidecar → `Q4_K_M`

### Acceptance Preservation: Stock (`incoai` / `z-lab`) vs. Abliterated (`Blackfrost-AI`)

Speculative decoding with DFlash 2 is mathematically **lossless**: in greedy decoding (`temp=0`), the target model verifies every drafted token so emitted text matches the target model's greedy choice; under stochastic sampling, rejection sampling preserves the target model's exact output distribution ([Inco AI DFlash 2 Blog](https://www.google.com/search?q=https%3A%2F%2Finco.ai%2Fblog%2Fdflash2%2F), [hanxiao L4 Quality Verification](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)). Quantizing the DFlash 2 sidecar from `BF16` to `Q4_K_M` only affects draft acceptance length, never output quality.

#### 1. Stock `incoai` / `z-lab` DFlash 2 Quantization Results (Measured)

* **Official `incoai/Qwen3.8-27B-DFlash2-GGUF` (and mirror `z-lab/Qwen3.8-27B-DFlash2-GGUF`) Evaluation:** Tested against stock `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` at Qwen 3.8's recommended sampling (`temperature=1.0, top_p=0.95, top_k=20`, `xhigh` reasoning effort, `max_tokens=2048`, `n_max=7`) on the first 8 GSM8K test problems ([incoai/Qwen3.8-27B-DFlash2-GGUF Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF)):

| Stock Draft GGUF Quant (`incoai` / `z-lab`) | File Size (Decimal GB / GiB) | Mean Acceptance Length (Completion Tokens / Verification Steps) | 
| ----- | ----- | ----- | 
| `Qwen3.8-27B-DFlash2-BF16.gguf` | 3.86 GB (3.60 GiB) | **5.28** | 
| `Qwen3.8-27B-DFlash2-Q8_0.gguf` | 2.06 GB (1.92 GiB) | **5.13** | 
| **`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`** | **1.14 GB (1.06 GiB)** | **5.39** | 

* **Independent Stock Greedy & Sampling Benchmarks:**

  * **L4 24GB Greedy (`temp=0`) Benchmark ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)):** Pairing `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (`1.14 GiB`) with `Qwen3.8-27B-UD-Q4_K_XL.gguf` at `n_max=7` achieved **5.573** mean accepted length on GSM8K-shaped prompts (**3.63** across the 7-workload suite) and outperformed `Q8_0` in end-to-end decode throughput (**35.99 tok/s** for `Q4_K_M` vs. **35.39 tok/s** for `Q8_0`), with **40/40** arithmetic accuracy identical to `--spec-type none`.

  * **RTX 5090 Benchmark ([GenerationAI Medium Study](https://www.google.com/search?q=https%3A%2F%2Fmedium.com%2F%40GenerationAI%2Fone-draft-model-two-verdicts-dflash-2-takes-qwen3-8-27b-f16ab29004d0)):** `Q4_K_M` (`1.14 GB`) and `Q8_0` (`2.06 GB`) DFlash 2 drafters accepted at the exact same rate (**73–74%**), with `Q4_K_M` running slightly faster (**215 tok/s** median decode).

  * **24GB GPU Quantization Sweep ([HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FHermiHg%2FQwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF)):** At `temp=1.0` (medium reasoning), reference `Q4_K_M` achieved **0.539** acceptance (`2.62` draft len, `95.2 tok/s`) at `n_max=3`, **0.466** (`2.86` draft len, `99.2 tok/s`) at `n_max=4`, and **0.403** (`3.01` draft len, `104.2 tok/s`) at `n_max=5`.

  * **Why DFlash 2 Survives `Q4_K_M` Unlike DFlash 1 on Qwen 3.6:** On older Qwen 3.6 DFlash 1 (`spiritbuun/Qwen3.6-27B-DFlash-GGUF`), sliding-window attention layers (`[S,S,S,S,F]`) suffered a 17-point acceptance drop at `Q4_K_M` (43% → 28%) ([spiritbuun Qwen3.6 DFlash Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fspiritbuun%2FQwen3.6-27B-DFlash-GGUF)). In DFlash 2 for Qwen 3.8, the addition of two-tap dynamic depthwise convolutions before/after every attention and FFN sublayer plus the rank-256 bilinear candidate selector ([Inco AI DFlash 2 Blog](https://www.google.com/search?q=https%3A%2F%2Finco.ai%2Fblog%2Fdflash2%2F)) completely stabilizes 4-bit quantization (`5.39` at `Q4_K_M` vs. `5.28` at `BF16`).

#### 2. Abliterated Pair Quantization Results (`Blackfrost-AI`)

* **Published Status:** For the Blackfrost abliterated pair, **only the BF16 sidecar has published acceptance numbers** (**55.8%** weighted acceptance, **478 / 856** tokens across 3 deterministic 256-token prompts at `n_max=3`, **109.31 tok/s** median on RTX PRO 6000 Blackwell) ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)).

* **Quantized `Q4_K_M` Abliterated Acceptance:** **`UNKNOWN` (unmeasured in published literature — first boot must confirm).** However, because `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` contains the exact same 81 weight tensors as `incoai/Qwen3.8-27B-DFlash2` (differing only in GGUF header metadata), quantizing its weights to `Q4_K_M` produces the exact same quantized weight matrices as `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M`.

### Exact `llama-quantize` Invocation & Expected Output Size

Using a `llama-quantize` binary compiled from `llama.cpp` PR #27342 (commit `1deefcca395743049c3820ab8f9b15043f3e9446` or upstream `master` after August 27, 2026), run the exact invocation documented in [Akicou/Qwen3.8-27B-DFlash2-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FAkicou%2FQwen3.8-27B-DFlash2-GGUF):

```
./build/bin/llama-quantize \
  ./dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf \
  ./dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf \
  Q4_K_M

```

* **Expected Tensor Payload & File Size:**

  * `llama-quantize` internal tensor size report: `3671.02 MiB` (BF16) → **`1079.61 MiB`** (`Q4_K_M` across all 81 tensors, `1,924,404,480` parameters) ([Akicou DFlash2 GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FAkicou%2FQwen3.8-27B-DFlash2-GGUF)).

  * Final GGUF file size on disk (including the 248,320-token vocabulary and GGUF header metadata): **`1,143,006,592 bytes`** (**`1.14 GB` decimal** / **`1.06 GiB` binary**, **4.71 BPW**) ([Akicou DFlash2 GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FAkicou%2FQwen3.8-27B-DFlash2-GGUF), [incoai Q4_K_M Blob Page](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF%2Fblob%2Fmain%2FQwen3.8-27B-DFlash2-Q4_K_M.gguf)).

### Known Quantization & Conversion Traps

1. **Stock `llama-quantize` Binary Rejects `general.architecture = dflash`:** Pre-PR #27342 release binaries of `llama.cpp` do not recognize the `dflash` architecture key or its 81 tensors (including the dynamic convolution bases and selector codebooks) and abort immediately ([Akicou DFlash2 GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FAkicou%2FQwen3.8-27B-DFlash2-GGUF)). You must compile `llama-quantize` from the same DFlash 2 PR #27342 branch used for `llama-server`.

2. **Do Not Use `--output-tensor-type` Overrides or Graft an Output Head:** DFlash 2 GGUF files contain **no `output.weight` or `token_embd.weight` tensors** (`src/models/dflash.cpp` leaves `output` null and shares the target model's embedding and LM head at runtime). As proven in [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4), grafting a cheap `q2_K` output head onto DFlash 2 degrades acceptance length (`3.40` vs. `3.52`) and throughput (`35.34` vs. `35.88 tok/s`) because DFlash 2's bilinear selector traces candidate paths over the top-16 logits per position.

3. **No `imatrix` Required for `Q4_K_M` (and `llama-imatrix` Fails on Standalone DFlash 2):** Official `incoai`/`z-lab` `Q4_K_M` files use standard uncalibrated `llama-quantize ... Q4_K_M` and reach `5.39` acceptance length ([incoai DFlash2 GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF)). Moreover, passing a DFlash 2 GGUF directly to `llama-imatrix` fails with `dflash requires ctx_other to be set` because a DFlash 2 drafter cannot run a forward pass without target-model hidden states from layers `[5, 19, 33, 47, 61]` ([Akicou DFlash2 GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FAkicou%2FQwen3.8-27B-DFlash2-GGUF), [HermiHg Q2_K_S-MIX Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FHermiHg%2FQwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF)).

4. **Vocab / Chat Template Trap When Converting Raw Safetensors:** If you ever convert `incoai/Qwen3.8-27B-DFlash2` from safetensors via `convert_hf_to_gguf.py` instead of quantizing Blackfrost's `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`, running without `--target-model-dir` crashes on missing tokenizer files ([spiritbuun DFlash Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fspiritbuun%2FQwen3.5-27B-DFlash-GGUF), [r/LocalLLM Thread](https://www.google.com/search?q=https%3A%2F%2Fwww.reddit.com%2Fr%2FLocalLLM%2Fcomments%2F1w3slho%2Fqwen3827b_q4_k_m_with_quantized_dflash2_via%2F)). Quantizing `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` directly with `llama-quantize` avoids this completely and preserves Blackfrost's embedded chat template and tokenizer metadata.

5. **Missing `dflash.rope.dimension_sections` Key in Blackfrost's Aug 22 Sidecar:** Our GGUF header inspection revealed that `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (commit `b669c83`, 2026-08-22) does **not** contain `dflash.rope.dimension_sections = [11, 11, 10, 0]`, which `z-lab` and `incoai` added days later (commits `2d9571f` on 2026-08-24 and `5196282` on 2026-08-28, *"Add dflash.rope.dimension_sections to GGUFs (mrope fix)"*) ([incoai File Tree](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF%2Ftree%2Fmain), [z-lab File Tree](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fz-lab%2FQwen3.8-27B-DFlash2-GGUF%2Ftree%2Fmain)). For text-only inference, M-RoPE `[11, 11, 10, 0]` is mathematically identical to standard 1D RoPE ([hanxiao L4 Analysis](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)), and Blackfrost's pinned commit `1deefcca395743049c3820ab8f9b15043f3e9446` loads it cleanly; however, first boot on newer `llama.cpp` commits must confirm whether `llama-server` requires `dflash.rope.dimension_sections`.

## 4. Stock DFlash 2 vs. Abliterated Target (Do Not Mix Unless Proven)

### File Audit of Stock DFlash 2 Repositories

Inspecting the actual Hugging Face file trees for the four stock repositories ([incoai/Qwen3.8-27B-DFlash2](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2%2Ftree%2Fmain), [z-lab/Qwen3.8-27B-DFlash2](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fz-lab%2FQwen3.8-27B-DFlash2%2Ftree%2Fmain), [incoai/Qwen3.8-27B-DFlash2-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF%2Ftree%2Fmain), and [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fz-lab%2FQwen3.8-27B-DFlash2-GGUF%2Ftree%2Fmain)) yields the following exact inventory:

| Repository ID | Revision / Commit | Exact Filename | Listed Size (Exact Bytes where published) | Commit Date | Notes / Checksum | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| [incoai/Qwen3.8-27B-DFlash2](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2%2Ftree%2Fmain) | `main` (2 commits; mirrored at `z-lab`) | `model.safetensors` | **3.85 GB** (BF16 safetensors) | 2026-08-18 (updated 2026-08-28) | Base safetensors repo; **GGUF files: `ABSENT`** | 
| [incoai/Qwen3.8-27B-DFlash2](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2%2Ftree%2Fmain) | `main` | `config.json`, `README.md`, `.gitattributes` | `1.24 kB`, `5.4 kB`, `1.58 kB` | 2026-08-18 / 2026-08-28 | No `tokenizer.json` shipped in repo | 
| [z-lab/Qwen3.8-27B-DFlash2](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fz-lab%2FQwen3.8-27B-DFlash2%2Ftree%2Fmain) | `main` @ [50307d4](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fz-lab%2FQwen3.8-27B-DFlash2%2Ftree%2Fmain) (`config.json` @ [ac04198](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fz-lab%2FQwen3.8-27B-DFlash2%2Fblob%2Fmain%2Fconfig.json)) | `model.safetensors` | **3.85 GB** (BF16 safetensors) | 2026-08-18 (updated 2026-09-12) | Mirror of `incoai`; **GGUF files: `ABSENT`** | 
| [z-lab/Qwen3.8-27B-DFlash2](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fz-lab%2FQwen3.8-27B-DFlash2%2Ftree%2Fmain) | `main` @ `50307d4` | `config.json`, `README.md`, `.gitattributes` | `1.24 kB`, `5.4 kB`, `1.58 kB` | 2026-08-18 / 2026-09-12 | Defines `target_layer_ids: [5, 19, 33, 47, 61]` | 
| [incoai/Qwen3.8-27B-DFlash2-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF%2Ftree%2Fmain) | `main` @ [5196282](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF%2Ftree%2Fmain) (initial [6cb5872](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF%2Fcommit%2F6cb5872e2cee6b4e780a8414922350be8e42d65c)) | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | **1.14 GB** (`1,143,006,592` B / `1.1 GB` in card table) | 2026-08-29 (mrope fix) | SHA-256: `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` ([Blob](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF%2Fblob%2Fmain%2FQwen3.8-27B-DFlash2-Q4_K_M.gguf)) | 
| [incoai/Qwen3.8-27B-DFlash2-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF%2Ftree%2Fmain) | `main` @ `5196282` | `Qwen3.8-27B-DFlash2-Q8_0.gguf` | **2.06 GB** (`2.0 GB` in card table) | 2026-08-29 (mrope fix) | Stock Qwen 3.8 metadata + `dflash.rope.dimension_sections` | 
| [incoai/Qwen3.8-27B-DFlash2-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2-GGUF%2Ftree%2Fmain) | `main` @ `5196282` | `Qwen3.8-27B-DFlash2-BF16.gguf` | **3.86 GB** (`3,860,292,992` B / `3.8 GB` in card table) | 2026-08-29 (mrope fix) | Stock Qwen 3.8 metadata + `dflash.rope.dimension_sections` | 
| [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fz-lab%2FQwen3.8-27B-DFlash2-GGUF%2Ftree%2Fmain) | `main` @ [2d9571f](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fz-lab%2FQwen3.8-27B-DFlash2-GGUF%2Ftree%2Fmain) (4 commits) | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, `Qwen3.8-27B-DFlash2-Q8_0.gguf`, `Qwen3.8-27B-DFlash2-BF16.gguf` | **1.14 GB**, **2.06 GB**, **3.86 GB** | 2026-08-24 (mrope fix) | Exact mirror of `incoai/Qwen3.8-27B-DFlash2-GGUF` | 

### Can Stock DFlash 2 Be Paired with the Blackfrost Abliterated Target?

**Verdict: `UNTESTED` for direct GGUF file mixing (`incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M` + `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`) — assume the Blackfrost metadata-matched sidecar only.**

Two evidence points must be separated clearly:

1. **Weight-Level Compatibility (Proven by Blackfrost & Corroborated on MLX):** Blackfrost's own model card states that `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` is a GGUF conversion of the stock [incoai/Qwen3.8-27B-DFlash2](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fincoai%2FQwen3.8-27B-DFlash2) checkpoint using the Blackfrost target's metadata ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)). Because weight-level abliteration orthogonalizes only the refusal direction without fine-tuning or pruning the 64 layers, the stock DFlash 2 weights achieved **55.8% acceptance** (`478 / 856` tokens) against the Blackfrost abliterated target once packaged with Blackfrost's metadata. (Similarly, on Apple Silicon MLX, [jfan/Qwen3.8-27B-heretic-dflash2](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fjfan%2FQwen3.8-27B-heretic-dflash2) tested base `z-lab/Qwen3.8-27B-DFlash2` against an abliterated/Heretic `Qwen3.8-27B-heretic-ara-mxfp4` target and measured $k = 2.87\text{–}3.58$ accepted tokens per step).

2. **GGUF File-Level Mixing (`incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M` with Blackfrost Target — `UNTESTED`):** No published source or benchmark has tested loading the stock `incoai/Qwen3.8-27B-DFlash2-GGUF` (`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`) file alongside `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF:Q4_K_M` in `llama-server`. While both share the `248,320`-token Qwen 3.8 vocabulary, the stock `incoai` GGUF carries stock `Qwen/Qwen3.8-27B` header metadata and template strings rather than Blackfrost's embedded short execution prompt and `general.finetune = ABLITERATED` metadata ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)). Per the locked rule, **do not mix stock `incoai`/`z-lab` GGUF files with the Blackfrost target**; always quantize Blackfrost's metadata-matched `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`.

## 5. `medismera` Fallback (`medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic`)

[medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fmedismera%2FQwen3.8-27B-OBLITERATED-Mythos-Class-Agentic) is a Safetensors-only SGLang/vLLM fallback repository derived from `OBLITERATUS/Qwen3.8-27B-OBLITERATED` (`Qwen/Qwen3.8-27B` base) that repairs broken upstream tool-calling and reasoning loops via a canonical 9.41 kB Jinja2 chat template ("Mythos-Class Adversarial Self-Review" and "Hierarchical Task Tree Decomposition") and a `reasoning_effort="medium"` default ([medismera Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fmedismera%2FQwen3.8-27B-OBLITERATED-Mythos-Class-Agentic), [Commit 3cdc6bc](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fmedismera%2FQwen3.8-27B-OBLITERATED-Mythos-Class-Agentic%2Fcommit%2F3cdc6bc03221faddceefc1d3780dc18b2791d14a)). Although commit [f41aee7](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fmedismera%2FQwen3.8-27B-OBLITERATED-Mythos-Class-Agentic%2Fcommit%2Ff41aee7c04e1edb2506cdfe329228c024a65846a) (September 7, 2026) briefly mirrored GGUF files (`Q2_K` through `Q8_0`, `IQ4_XS`, `mmproj`), commit `8a74ea0` (*"chore: purge all GGUF files to strictly standardize on native SGLang Safetensors"*) deleted all GGUF files from the repository — making GGUFs **`ABSENT`** in `medismera` (static third-party GGUFs exist separately at [mradermacher/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fmradermacher%2FQwen3.8-27B-OBLITERATED-Mythos-Class-Agentic-GGUF)). Inspecting the live repository branch dropdown confirms that revisions **`awq`** (commit `d5f585d`) and **`fp8`** (commit `40e3561`) **both exist** alongside **`main`** (HEAD commit `b10479b`, October 14, 2026; weights committed in [349d65c](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fmedismera%2FQwen3.8-27B-OBLITERATED-Mythos-Class-Agentic%2Fcommit%2F349d65cff6b19b9117fd38bb410c5ff9b197e569) on September 7, 2026), though the actual file sizes in `awq` (`18.7 GB` + `849 MB` MTP = `19.55 GB`) are larger than the model card's `"~16 GB"` prose estimate. Finally, **no published pairing of `medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic` with DFlash 2 exists** (**`UNTESTED`** — neither the model card nor community benchmarks document running `incoai/Qwen3.8-27B-DFlash2` against `medismera`).

### `medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic` File Lists by Revision

| Branch / Revision | Commit Hash & Date | Exact Filenames in Branch File Tree | Listed File Sizes (Exact Bytes where published) | Branch Total Weight Size | 
| ----- | ----- | ----- | ----- | ----- | 
| **`main`** ([Tree](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fmedismera%2FQwen3.8-27B-OBLITERATED-Mythos-Class-Agentic%2Ftree%2Fmain)) | HEAD `b10479b` (2026-10-14); weights @ [349d65c](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2Fmedismera%2FQwen3.8-27B-OBLITERATED-Mythos-Class-Agentic%2Fcommit%2F349d65cff6b19b9117fd38bb410c5ff9b197e569) (2026-09-07) | **29 Safetensors shards:** `model-00001-of-00028.safetensors` (`2.54 GB` / `2,542,796,928` B), `model-00002` through `model-00026` (`1.86 GB`–`1.95 GB` each; `1,858,073,840`–`1,952,630,800` B), `model-00027-of-00028.safetensors` (`2.54 GB` / `2,542,796,896` B), `model-00028-of-00028.safetensors` (`1.1 GB` / `1,101,059,064` B), `model-extra-00001-of-00001.safetensors` (`1.77 GB` / `1,770,897,616` B). **Config/Tokenizer (14 files):** `.gitattributes` (`2.34 kB`), `LICENSE` (`11.5 kB`), `README.md` (`10.2 kB`), `abliteration_metadata.json` (`2.91 kB`), `chat_template.jinja` (`9.41 kB`), `config.json` (`3.69 kB`), `generation_config.json` (`250 B`), `hard_negative_residue.json` (`3.89 kB`), `merges.txt` (`3.35 MB`), `model.safetensors.index.json` (`114 kB`), `preprocessor_config.json` (`390 B`), `tokenizer.json` (`12.8 MB`), `tokenizer_config.json` (`16.8 kB`), `video_preprocessor_config.json` (`385 B`), `vocab.json` (`6.72 MB`). | **55.6 GB** (`55,586,124,697` B across 29 shards + metadata) | **55.6 GB** (BF16) | 
| **`awq`** (`/tree/awq`) | HEAD `d5f585d` (Sep 2026) | **2 Safetensors files:** `model.safetensors` (**`18.7 GB`**), `model-mtp.safetensors` (**`849 MB`**). **Config/Tokenizer (15 files):** `.gitattributes` (`2.34 kB`), `LICENSE` (`11.5 kB`), `README.md` (`10.6 kB`), `abliteration_metadata.json` (`2.91 kB`), `chat_template.jinja` (`9.41 kB`), `config.json` (`20.7 kB`), `generation_config.json` (`250 B`), `hard_negative_residue.json` (`3.89 kB`), `merges.txt` (`3.35 MB`), `model.safetensors.index.json` (`201 kB`), `preprocessor_config.json` (`390 B`), `tokenizer.json` (`12.8 MB`), `tokenizer_config.json` (`16.8 kB`), `video_preprocessor_config.json` (`385 B`), `vocab.json` (`6.72 MB`). | `model.safetensors`: **18.7 GB**; `model-mtp.safetensors`: **849 MB** | **19.55 GB** (4-bit AWQ + MTP) | 
| **`fp8`** (`/tree/fp8`) | HEAD `40e3561` (Sep 2026) | **2 Safetensors shards:** `model-00001-of-00002.safetensors` (**`20 GB`**), `model-00002-of-00002.safetensors` (**`9.95 GB`**) (Note: `model-mtp.safetensors` is **`ABSENT`** in `fp8`). **Config/Tokenizer (15 files):** `.gitattributes` (`2.34 kB`), `LICENSE` (`11.5 kB`), `README.md` (`10.6 kB`), `abliteration_metadata.json` (`2.91 kB`), `chat_template.jinja` (`9.41 kB`), `config.json` (`10.2 kB`), `generation_config.json` (`250 B`), `hard_negative_residue.json` (`3.89 kB`), `merges.txt` (`3.35 MB`), `model.safetensors.index.json` (`173 kB`), `preprocessor_config.json` (`390 B`), `tokenizer.json` (`12.8 MB`), `tokenizer_config.json` (`16.8 kB`), `video_preprocessor_config.json` (`385 B`), `vocab.json` (`6.72 MB`). | `model-00001-of-00002.safetensors`: **20 GB**; `model-00002-of-00002.safetensors`: **9.95 GB** | **29.95 GB** (8-bit FP8) | 

## “Do Not Download” List (Exclusions & Anti-Traps)

* **Do NOT download `mmproj-Qwen3.8-27B-ABLITERATED-F16.gguf` (928 MB) or `mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf` (629 MB):** Unused for text-only worker deployments and incompatible with unpatched DFlash 2 speculative decoding in `llama.cpp` ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF), [r/LocalLLaMA PR #27342 Thread](https://www.google.com/search?q=https%3A%2F%2Fwww.reddit.com%2Fr%2FLocalLLaMA%2Fcomments%2F1w00ngm%2Fsupport_for_dflash2_in_llamacpp_has_been_merged%2F)).

* **Do NOT download non-`Q4_K_M` quants from `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF`:** Skip `Q2_K` (10.9 GB), `Q3_K_S` (12.3 GB), `Q3_K_M` (13.5 GB), and `Q4_K_S` (15.8 GB) due to unnecessary quality degradation, and skip `Q5_K_S` (19.0 GB), `Q5_K_M` (19.5 GB), `Q6_K` (22.4 GB), and `Q8_0` (29.0 GB) as they violate the locked \~16.8 GB `Q4_K_M` target class ([Blackfrost File Tree](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Ftree%2Fmain)).

* **Do NOT download pre-August 16, 2026 (`c8a221a`) Blackfrost `Q4_K_M` files or deprecated `mtp-*` sidecar files (`648b4e1`):** Always verify that `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` matches commit `4f6732c` (`16,810,716,384` bytes, SHA-256 `5d53637a59cfcd3a4d8354e254ffd44943e5a693da2405a3e228c62962355509`), which natively embeds the 65th MTP block ([Blackfrost Q4_K_M Blame](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF%2Fblame%2Fmain%2FQwen3.8-27B-ABLITERATED-Q4_K_M.gguf)).

* **Do NOT download stock `incoai/Qwen3.8-27B-DFlash2-GGUF` or `z-lab/Qwen3.8-27B-DFlash2-GGUF` to pair with the Blackfrost target:** Mixing stock-metadata DFlash 2 GGUFs with the Blackfrost abliterated target is `UNTESTED` and lacks Blackfrost's embedded chat template metadata.

* **Do NOT download older DFlash 1 or DSpark drafters:** Skip `mrchuy/Qwen3.8-27B-DFlash-drafter-bootstrap-GGUF` (transplanted Qwen 3.6 weights, 2.02 accepted length), `kstoyanov99/Qwen3.8-27B-Dflash` (1.81 accepted length), `DimInfer/Qwen3.8-27B-Dspark-v1`, and `RadixArk/Qwen3.8-27B-DSpark`, all of which underperform native MTP and DFlash 2 in `llama.cpp` ([hanxiao L4 Rejected Configurations](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)).

* **Do NOT download stock Unsloth (`unsloth/Qwen3.8-27B-GGUF`), `ggml-org/Qwen3.8-27B-GGUF`, Huihui, Gemma, `gpt-oss`, or Ternary Bonsai checkpoints:** Excluded by locked deployment specification.

## Remaining Unknowns & First-Boot `llama-server` Log Verification Gate

Because no live GPU execution takes place during file-pinning research, the first boot of `llama-server` must explicitly verify the following four unknowns in the startup and slot telemetry logs:

1. **Quantized `Q4_K_M` DFlash 2 Acceptance Rate on the Blackfrost Abliterated Pair (`UNKNOWN`):**

   * *Why it is unknown:* Blackfrost only benchmarked the `BF16` sidecar (`55.8%` acceptance across 3 prompts at `n_max=3`), while `incoai` and `hanxiao` only benchmarked the `Q4_K_M` sidecar on stock `Qwen/Qwen3.8-27B` (`5.39` and `5.573` mean accepted length on GSM8K at `n_max=7`).

   * *First-boot check:* Inspect `llama-server` decode logs (or `/metrics` / slot timings) on a deterministic test prompt to confirm that `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` achieves $\ge 50\%$ token acceptance (or mean accepted length $\ge 2.5$ at `n_max=3` and $\ge 3.5$ at `n_max=5..7`) without template-induced distribution shift.

2. **48 GDN + 16 Full-Attention Layer Allocation in `llama-server` Startup Logs (`UNKNOWN` in Blackfrost prose, verified via `qwen35.full_attention_interval = 4` in GGUF header):**

   * *Why it must be confirmed:* While the GGUF header explicitly sets `qwen35.block_count = 65`, `qwen35.nextn_predict_layers = 1`, and `qwen35.full_attention_interval = 4`, the Blackfrost model card prose only states `"64 text layers · Gated DeltaNet + full attention"` without printing `"48"` and `"16"`.

   * *First-boot check:* Verify in `llama-server` model-loading output that `n_layer = 64`, `n_layer_all = 65`, that the KV cache allocates across exactly **16 full-attention layers** (`64 KiB/token` at F16 KV or `32 KiB/token` at Q8_0 KV), and that the remaining **48 layers** allocate fixed recurrent Gated DeltaNet (`ssm_states`) buffers.

3. **`dflash.rope.dimension_sections` Absence on Newer `llama.cpp` Commits (`UNKNOWN` beyond commit `1deefcca`):**

   * *Why it is unknown:* Blackfrost's `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (2026-08-22, `b669c83`) lacks the `dflash.rope.dimension_sections = [11, 11, 10, 0]` GGUF KV entry added to `z-lab`/`incoai` on August 24–28, 2026.

   * *First-boot check:* Confirm that `llama-server` loads `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` without throwing a missing-key error on `dflash.rope.dimension_sections`. If using a post-August 27, 2026 upstream `llama.cpp` commit that requires this key, inject `dflash.rope.dimension_sections = [11, 11, 10, 0]` via `gguf-py` (`llama.cpp/gguf-py/scripts/gguf_set_metadata.py`) or pin Blackfrost's tested commit `1deefcca395743049c3820ab8f9b15043f3e9446`.

4. **Target-to-Draft Layer Hook Verification (`[5, 19, 33, 47, 61]` / GGUF `[6, 20, 34, 48, 62]`):**

   * *First-boot check:* Confirm in the `llama-server` speculative decoding initialization banner that `draft-dflash` binds `dflash.target_layers = [6, 20, 34, 48, 62]` (0-indexed blocks `5, 19, 33, 47, 61`) and clamps `n_draft_max` to `7` (`dflash.block_size - 1`).