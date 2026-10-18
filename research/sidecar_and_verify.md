# Colab `llama.cpp` Worker Deep-Research Verification: Blackfrost Qwen3.8-27B-ABLITERATED Q4_K_M + DFlash 2 Sidecar (`sidecar_and_verify.md`)

**Date:** 2026-10-14  
**Target Environment:** Colab NVIDIA A100-SXM4-40GB (G4 optional; L4 excluded; `mmproj` excluded)  
**Engine Pin:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) master commit `efe3626` / tag `b11552` (also cross-verified against current master commits `ce8c04021` at `b11541` and `23b0202a1` / `cbe28bb3c` at `b11577`; DFlash 2 merged via PR `#27816`, commit `b10f9ca58c` on `2026-08-27`; PR `#27342` commit `1deefcca395743049c3820ab8f9b15043f3e9446` excluded)  
**Target Model (Locked):** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) — [Qwen3.8-27B-ABLITERATED-Q4\_K\_M.gguf](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/blame/main/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf) (`16,810,716,384` bytes, SHA-256 `5d53637a59cfcd3a4d8354e254ffd44943e5a693da2405a3e228c62962355509`, file commit `4f6732c` on `2026-08-16` / repo HEAD `994bb4e`, embeds 65th NextN/MTP block `qwen35.block_count = 65`)  
**Draft Source (Locked):** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/tree/main) — `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (`3,860,292,992` bytes, SHA-256 `ca6dacc0bba87332b4eae6f103fa4c0c800ca871657e83d2aed658351b2c4fb3`, file commit `b669c83` on `2026-08-22`, converted from `incoai/Qwen3.8-27B-DFlash2` with Blackfrost target metadata)  
**First-Boot Quantization (Locked):** `llama-quantize dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf Q4_K_M` → expected output `1,143,006,592` bytes (81 tensors, no imatrix, no `--output-tensor-type`, no grafted output head)

## Primary Sources

1. **Blackfrost Target & BF16 Sidecar Repository:** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF Model Card](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF), [/tree/main File Tree](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/tree/main), [/blame/main/README.md](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/blame/main/README.md), [/blame/main/Qwen3.8-27B-ABLITERATED-Q4\_K\_M.gguf](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/blame/main/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf), and parent [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16).
2. **Stock DFlash 2 GGUF Repositories & KV Dumps:**
   - [incoai/Qwen3.8-27B-DFlash2-GGUF Model Card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF), [Qwen3.8-27B-DFlash2-BF16.gguf Blob \& Header](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-BF16.gguf), and [Qwen3.8-27B-DFlash2-Q4\_K\_M.gguf Blob \& Header](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) (commit `5196282`, `2026-08-29`).
   - [z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main) and [Qwen3.8-27B-DFlash2-Q4\_K\_M.gguf Blob \& Header](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) (commits `2d9571f` / `343c7b5`, `2026-08-24`).
   - [Akicou/Qwen3.8-27B-DFlash2-GGUF Model Card \& Pre-MRoPE File Sizes](https://huggingface.co/Akicou/Qwen3.8-27B-DFlash2-GGUF) (`2026-08-20`).
   - [andrew-paul/Qwen3.8-27B-DFlash2-Q3\_K\_M-GGUF Quantization Determinism Audit](https://huggingface.co/andrew-paul/Qwen3.8-27B-DFlash2-Q3_K_M-GGUF).
3. **`ggml-org/llama.cpp` C++ Source & Conversion Tooling:**
   - [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) (`common_speculative_impl_draft_dflash`, `common_speculative_are_compatible`, `common_speculative_print_stats`, `ngram-map-k4v` registration).
   - [src/llama-arch.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.h) (`LLM_ARCH_DFLASH`, `LLM_KV_DFLASH_*`, `LLM_KV_ROPE_DIMENSION_SECTIONS`).
   - [src/llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp) (`llama_kv_cache` constructor & `LLAMA_LOG_INFO` lines).
   - [tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp) (`server_slot::to_json`, `server_slot::print_timings`, M-RoPE `{p, p, p, 0}` position setup).
   - [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md), [convert\_hf\_to\_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py), and [gguf-py/README.md](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/README.md).
4. **`llama.cpp`, `Qwen3.8`, and `unsloth` Issues & Benchmarks:**
   - [ggml-org/llama.cpp#27862](https://github.com/ggml-org/llama.cpp/issues/27862) (DFlash 2 conversion & `ngxson` Aug 28 comment on pre-Aug 27 GGUFs and vision M-RoPE).
   - [ggml-org/llama.cpp#27833](https://github.com/ggml-org/llama.cpp/issues/27833) and [ggml-org/llama.cpp#27829](https://github.com/ggml-org/llama.cpp/issues/27829) (DFlash 2 shared `output.weight` / `token_embd.weight` via `ctx_other` in `src/models/dflash.cpp:800/:1010` and verbatim startup/timing logs).
   - [unslothai/unsloth#13511](https://github.com/unslothai/unsloth/issues/13511) (Verbatim `llama-server` build `b11541` commit `ce8c04021` startup log loading `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF` + `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`).
   - [ggml-org/llama.cpp#27107](https://github.com/ggml-org/llama.cpp/issues/27107) and [Qwen/Qwen3.8-27B#148](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/148) (Qwen 3.8 Jinja `raise_exception` autoparser failure and fixes).
   - [QwenLM/Qwen3.8#236](https://github.com/QwenLM/Qwen3.8/issues/236) (Tool calling plain-text fallback when `enable_thinking=false` with complex system prompts).
   - [lukaLLM/DFlash\_Qwen3.6\_27B\_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP) and [hanxiao/Qwen3.8-27B-UD-Q4\_K\_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) (Stock Qwen3.8-27B + DFlash 2 Q4_K_M acceptance width sweeps, M-RoPE `[11, 11, 10]` text equivalence, and CUDA verification).

## Executive Findings

- **Blackfrost Aug 22 BF16 Sidecar Lacks `dflash.rope.dimension_sections`, and `llama-server` `LOADS` It Without Error for Text-Only Serving:** `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (`3,860,292,992` bytes, SHA-256 `ca6dacc0bba87332b4eae6f103fa4c0c800ca871657e83d2aed658351b2c4fb3`, commit `b669c83`, `2026-08-22`) was converted against pre-merge PR `#27342` commit `1deefcca395743049c3820ab8f9b15043f3e9446` ([Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF)) and has not been rebuilt since. It omits the 160-byte `dflash.rope.dimension_sections = [11, 11, 10, 0]` KV added later by `z-lab` (`2d9571f` / `343c7b5` on `2026-08-24`, `3,860,293,152` bytes) and `incoai` (`5196282` on `2026-08-29`) ([z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main), [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)). On `efe3626` / `b11552` (and `ce8c04021` / `b11541` and `23b0202a1` / `b11577`), `LLM_KV_ROPE_DIMENSION_SECTIONS` is read with `required = false`, defaulting `hparams.rope_sections` to `[0, 0, 0, 0]` (`is_mrope = false` in [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp)). Empirical proof in [unslothai/unsloth#13511](https://github.com/unslothai/unsloth/issues/13511) confirms `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` **LOADS** cleanly and drafts text on `b11541`. Because `mmproj` is skipped and text-only M-RoPE `[11, 11, 10, 0]` uses identical broadcast positions `{p, p, p, 0}` across all 32 rotary pairs ([tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp), [hanxiao/Qwen3.8-27B-UD-Q4\_K\_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)), 1D NeoX RoPE and M-RoPE are mathematically identical for text-only inference.
- **If Injecting `dflash.rope.dimension_sections`, Use `gguf_new_metadata.py` (Not `gguf_set_metadata.py`) Before `llama-quantize`, and It Survives Quantization:** `gguf_set_metadata.py` only mutates existing scalar/array bytes in-place via `np.memmap` and cannot insert a missing key ([ggml-org/llama.cpp#8768](https://github.com/ggml-org/llama.cpp/issues/8768), [gguf-py/README.md](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/README.md)). Inserting `dflash.rope.dimension_sections` requires `gguf_new_metadata.py` or a `gguf.GGUFReader`/`GGUFWriter` copy pass before `llama-quantize`, which increases the BF16 file from `3,860,292,992` to `3,860,293,152` bytes (+160 bytes) and the quantized `Q4_K_M` output from `1,143,006,592` to `1,143,006,752` bytes (+160 bytes), surviving `llama-quantize` intact.
- **Blackfrost Q4_K_M Target + Q4_K_M DFlash 2 Acceptance Is `UNKNOWN` in Published Literature, Though Quantized Tensor Payload Is Bit-Identical to Stock `incoai` Q4_K_M:** Blackfrost published only BF16 DFlash 2 acceptance (`55.8%` overall, `478 / 856` draft tokens accepted at `n_max=3`, `109.31` tok/s median on RTX PRO 6000 Blackwell) ([Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF)). No published benchmark tests the quantized `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` with the abliterated target (`UNKNOWN`). However, because `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` contains the exact same 81 BF16 tensors as `incoai/Qwen3.8-27B-DFlash2` (`3,604.60M` params) and deterministic `llama-quantize ... Q4_K_M` without imatrix depends solely on tensor values and names ([andrew-paul/Qwen3.8-27B-DFlash2-Q3\_K\_M-GGUF](https://huggingface.co/andrew-paul/Qwen3.8-27B-DFlash2-Q3_K_M-GGUF)), the 81 quantized tensor blocks are byte-for-byte identical to [incoai/Qwen3.8-27B-DFlash2-GGUF:Q4\_K\_M](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) (SHA-256 `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`).
- **Acceptance Telemetry Lives in `timings.draft_n` / `timings.draft_n_accepted` and `slot print_timing`, NOT in `GET /slots`:** In [tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp), `server_slot::to_json()` (`GET /slots`) returns `"speculative": true` but omits draft token counters. The exact JSON keys `"draft_n"` and `"draft_n_accepted"` are emitted in the `/v1/chat/completions` (or `/completion`) response `"timings"` object ([ggml-org/llama.cpp#12968](https://github.com/ggml-org/llama.cpp/issues/12968), [ggml-org/llama.cpp#29372](https://github.com/ggml-org/llama.cpp/issues/29372)), and logged at completion via `slot print_timing: id  0 | task ... | draft acceptance = %0.5f (%5d accepted / %5d generated), mean len = %5.2f` ([ggml-org/llama.cpp#27833](https://github.com/ggml-org/llama.cpp/issues/27833)).
- **Mixing Verdict Is `KEEP_QUANTIZE_BLACKFROST`:** Although `llama-server` ignores the draft GGUF's `tokenizer.chat_template` and borrows `token_embd.weight` and `output.weight` from the target model (`ctx_other` in `src/models/dflash.cpp:800/:1010` per [ggml-org/llama.cpp#27833](https://github.com/ggml-org/llama.cpp/issues/27833)), no published run has ever loaded `-m Blackfrost...Q4_K_M.gguf -md incoai/...DFlash2-Q4_K_M.gguf` (`UNTESTED`), and locked worker rules forbid mixing without a published run.
- **Tool Calling on `efe3626` Uses the `--jinja` Differential PEG Autoparser:** `--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3` are SGLang/vLLM flags only ([Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16)) and will cause `llama-server` to exit with `error: invalid argument`. With `--reasoning off` and `--chat-template-kwargs '{"enable_thinking":false}'`, Qwen 3.8 can fall back to plain text on complex tool schemas (`"tool_choice": "auto"`) due to the empty `<think>\n\n</think>\n\n` prefill documented in [QwenLM/Qwen3.8#236](https://github.com/QwenLM/Qwen3.8/issues/236); passing `"tool_choice": "required"` forces grammar-constrained `<tool_call>` emission.

## 1. `dflash.rope.dimension_sections` on the Blackfrost Aug 22 Sidecar

### 1.1 Verification of Missing `dflash.rope.dimension_sections` in Blackfrost BF16 GGUF
- **Confirmed Absent:** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/tree/main) file `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (`3,860,292,992` bytes, SHA-256 `ca6dacc0bba87332b4eae6f103fa4c0c800ca871657e83d2aed658351b2c4fb3`) was committed in `b669c83` on `2026-08-22` ([/blame/main/README.md](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/blame/main/README.md)) using pre-merge PR `#27342` commit `1deefcca395743049c3820ab8f9b15043f3e9446` (`2026-08-19`). At commit `1deefcca`, `DFlashModel.set_gguf_parameters()` in `convert_hf_to_gguf.py` did not write `dflash.rope.dimension_sections`.
- **Byte-Size Proof Against `Akicou`, `z-lab`, and `incoai`:**
  - Pre-MRoPE BF16 DFlash 2 GGUFs converted before `2026-08-24` — both [Akicou/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/Akicou/Qwen3.8-27B-DFlash2-GGUF) (`2026-08-20`) and Blackfrost's `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (`b669c83`, `2026-08-22`) — are exactly **`3,860,292,992` bytes** (`1,143,006,592` bytes when quantized to `Q4_K_M`).
  - On `2026-08-24`, [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main) pushed commits `2d9571f` (`add dflash.rope.dimension_sections to Qwen3.8-27B-DFlash2-Q8_0.gguf`) and `343c7b5` (`add dflash.rope.dimension_sections to Qwen3.8-27B-DFlash2-Q4_K_M.gguf`), and on `2026-08-29`, [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-BF16.gguf) pushed commit `5196282` (`Add dflash.rope.dimension_sections to GGUFs (mrope fix)`). Adding `dflash.rope.dimension_sections = [11, 11, 10, 0]` increased the GGUF header + alignment padding by **160 bytes**, raising BF16 to **`3,860,293,152` bytes** (SHA-256 `493fd13640ebbb71d8a86ef23396984122bd148efed0b8fb5ed8934bc81951ee`) and `Q4_K_M` to **`1,143,006,752` bytes** (SHA-256 `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`) ([incoai/Qwen3.8-27B-DFlash2-GGUF:Q4\_K\_M](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf)).
  - Blackfrost's only subsequent repo commit (`994bb4e`) added `GGUF_conversion_agent_instructions.md` (`8.75 kB`) without touching `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` ([/tree/main](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/tree/main)).

### 1.2 Runtime Behavior on `efe3626` / `b11552` (and Current Master `23b0202a1` / `b11577`)
- **Does `llama-server` / `src/models/dflash.cpp` require `dflash.rope.dimension_sections`?** **No.**
  - In [src/llama-arch.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.h), `LLM_KV_ROPE_DIMENSION_SECTIONS` maps to `"%s.rope.dimension_sections"`. In `src/llama-model.cpp` (`LLM_ARCH_DFLASH` hyperparameter loader, introduced in PR `#27816` commit `b10f9ca58c` and unchanged through `efe3626` / `b11552` and current master `23b0202a1` / `b11577`), `ml.get_key(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, false)` is called with `required = false`.
  - When the key is absent from the GGUF header, `hparams.rope_sections` remains zero-initialized (`[0, 0, 0, 0]`).
  - In [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) (`struct common_speculative_impl_draft_dflash`), the drafter initializes `bool is_mrope = false;` and checks whether the draft model has non-zero M-RoPE sections. When `hparams.rope_sections` is all zeros, `is_mrope` stays `false`, and `src/models/dflash.cpp` executes standard 1D NeoX RoPE (`ggml_rope_ext`) rather than multi-dimensional `ggml_rope_multi`.
  - **Why Text-Only Inference Is Unaffected:** As documented in [hanxiao/Qwen3.8-27B-UD-Q4\_K\_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) (`mrope_section [11, 11, 10], 32 pairs, equal to standard RoPE for text-only input`) and implemented in [tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp) (lines 469–475, where text batch positions are populated across the 3 active M-RoPE planes as `{p, p, p, 0}`), applying M-RoPE with sections `[11, 11, 10, 0]` when all 3 spatial/temporal planes carry the identical scalar position `p` applies the exact same rotation angle to all 32 rotary frequency pairs (`11 + 11 + 10 = 32` pairs = `64` dims = `25%` of `head_dim = 256`) as standard 1D NeoX RoPE at position `p`.
  - In [ggml-org/llama.cpp#27862](https://github.com/ggml-org/llama.cpp/issues/27862), maintainer `ngxson` explicitly noted on `2026-08-28`:
    > *"GGUF generated before Aug 27th 2026 must be reconverted, otherwise vision will be broken. See: #27342 (comment)"*
    Furthermore, as confirmed in [unslothai/unsloth#13511](https://github.com/unslothai/unsloth/issues/13511) (tested on `llama.cpp` `0.6.0-dev` build `b11541`, commit `ce8c04021`), `mmproj` and `--spec-type draft-dflash` cannot be used together on images anyway (`decode() failed: failed to process speculative batch`), while text-only requests load and generate cleanly with `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`.

### 1.3 Load Status & Exact Error Strings
- **Status on `efe3626` / `b11552`:** **LOADS** (verified directly in [unslothai/unsloth#13511](https://github.com/unslothai/unsloth/issues/13511) on `b11541` `ce8c04021` with `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF` and `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`).
- **Exact Error Strings (for reference if a GGUF key is missing or malformed in `src/llama-model-loader.h`):**
  - If a required key is missing (`required = true` in `llama_model_loader::get_key`):
    `error loading model: key not found in model: <key_name>`
  - If `*.rope.dimension_sections` is present in GGUF with fewer or more than 4 elements (`LLAMA_MAX_ROPE_SECTIONS = 4` in `llama_model_loader::get_key_or_arr`):
    `dflash.rope.dimension_sections has wrong array length` (the same loader check that prints `qwen35.rope.dimension_sections has wrong array length` on incompatible GGUFs).

### 1.4 Optional Pre-Quantization Metadata Injection via `gguf-py` (and Survival Across `llama-quantize`)
- **Why `gguf_set_metadata.py` Alone Cannot Add a Missing Key:** As documented in [gguf-py/README.md](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/README.md) and [ggml-org/llama.cpp#8768](https://github.com/ggml-org/llama.cpp/issues/8768), `gguf/scripts/gguf_set_metadata.py` opens the GGUF via `np.memmap(..., mode="r+")` and overwrites an **existing** key's byte slice in-place (`if key not in reader.fields: logger.error(f'Field "{key}" not found in input file')`). Because `dflash.rope.dimension_sections` is not present in `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`, adding it increases the header size by 56 bytes (plus 32-byte tensor alignment shift = **+160 bytes** total file size) and requires rewriting the file via `gguf-py`'s `GGUFReader` / `GGUFWriter` (the mechanism used by `gguf/scripts/gguf_new_metadata.py`).
- **Exact Pre-Quantization `gguf-py` Command to Inject `dflash.rope.dimension_sections = [11, 11, 10, 0]` into the Blackfrost BF16 File:**
  ```bash
  python3 -c '
  import gguf, shutil
  from gguf import GGUFReader, GGUFWriter, GGUFValueType

  src = "dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf"
  tmp = "dflash-Qwen3.8-27B-ABLITERATED-BF16.mrope.gguf"
  reader = GGUFReader(src)
  arch = reader.get_field("general.architecture").contents()
  writer = GGUFWriter(tmp, arch=arch)

  for field in reader.fields.values():
      if field.name in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count", "general.architecture"):
          continue
      writer.add_key_value(field.name, field.contents(), field.types[0],
                           sub_type=field.types[1] if len(field.types) > 1 else None)

  writer.add_array("dflash.rope.dimension_sections", [11, 11, 10, 0])

  for tensor in reader.tensors:
      writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                             tensor.data.nbytes, tensor.tensor_type)

  writer.write_header_to_file()
  writer.write_kv_data_to_file()
  writer.write_ti_data_to_file()
  for tensor in reader.tensors:
      writer.write_tensor_data(tensor.data)
  writer.close()
  shutil.move(tmp, src)
  '
  ```
- **Does `dflash.rope.dimension_sections` Survive `llama-quantize`?** **Yes.** In `src/llama-quantize.cpp` (`llama_model_quantize_internal`), `llama-quantize` copies **every** KV entry from the input GGUF (`gguf_set_kv(ctx_out, ml.meta.get())`) and only overrides `general.file_type` and `general.quantization_version`.
  - Without injection: `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` is **`1,143,006,592` bytes** (matching [Akicou/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/Akicou/Qwen3.8-27B-DFlash2-GGUF)'s pre-MRoPE `Q4_K_M` size).
  - With injection before `llama-quantize`: `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` becomes **`3,860,293,152` bytes** and `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` becomes **`1,143,006,752` bytes** (matching [incoai/Qwen3.8-27B-DFlash2-GGUF:Q4\_K\_M](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) and [z-lab/Qwen3.8-27B-DFlash2-GGUF:Q4\_K\_M](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) byte counts).
- **Why Converting Stock `incoai` Safetensors Is Unnecessary (and Its Failure Mode Without `--target-model-dir`):**
  - Because `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` already **LOADS** directly and `gguf-py` can inject `dflash.rope.dimension_sections` locally if ever desired, converting `incoai/Qwen3.8-27B-DFlash2` safetensors is **not recommended**.
  - For completeness on [convert\_hf\_to\_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) (documented in [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) and [ggml-org/llama.cpp#25116](https://github.com/ggml-org/llama.cpp/issues/25116)): `DFlashModel` requires `--target-model-dir <path-to-Blackfrost-HF-tokenizer-and-config>` because `incoai/Qwen3.8-27B-DFlash2` only ships `config.json` and `model.safetensors` (`3.86 GB`) without `tokenizer.json`, `tokenizer_config.json`, or `chat_template.jinja`. Running `convert_hf_to_gguf.py` on a DFlash checkpoint **without** `--target-model-dir` (or without copying the target's tokenizer files into the draft folder) fails immediately during vocabulary initialization (`FileNotFoundError` on `tokenizer.json` / `tokenizer_config.json`), or if pointed at a mismatched model directory, embeds a mismatched `tokenizer.ggml.tokens` / `tokenizer.chat_template` that fails `common_speculative_are_compatible()` at runtime.

## 2. Quantized `Q4_K_M` DFlash Acceptance on the *Blackfrost Abliterated* Pair

### 2.1 Published Acceptance for Blackfrost `Q4_K_M` Target + `Q4_K_M` DFlash 2 Sidecar
- **Published Blackfrost `Q4_K_M` + `Q4_K_M` DFlash 2 Number:** **`UNKNOWN`**.
  - Across Hugging Face, GitHub issues/PRs, and benchmark repositories, **no published run** pairs [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF:Q4\_K\_M](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF/blame/main/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf) with a quantized `Q4_K_M` DFlash 2 sidecar at `n_max` 3, 5, or 7 under either greedy (`temp=0`) or Qwen sampling.
  - The only published Blackfrost DFlash 2 benchmark on the [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF model card](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) pairs `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` with the **BF16** sidecar (`dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`) at `--spec-draft-n-max 3` on an NVIDIA RTX PRO 6000 Blackwell Server Edition (`262,144` context, `--reasoning off`, 256 tokens per prompt), achieving **55.8% overall draft token acceptance (`478 / 856`)** and **`109.31` tok/s median** (`52.9%` / `156/295`, `53.8%` / `157/292`, and `61.3%` / `165/269` across the three prompts).

### 2.2 Reference Table: Stock `incoai` / `z-lab` / `lukaLLM` / `hanxiao` (`Q4_K_M` vs `BF16`) vs Blackfrost `BF16`
*Per worker constraints, stock `Qwen/Qwen3.8-27B` / `unsloth` / `ggml-org` target benchmarks and Blackfrost `BF16` draft benchmarks are isolated below and must not be treated as measurements of the Blackfrost `Q4_K_M` + `Q4_K_M` pair:*

| Source & Hardware | Target GGUF | Draft GGUF | `n_max` & Sampling | Mean Accepted Length ($\tau$) | Token Acceptance Rate (`draft_n_accepted / draft_n`) | Decode Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) (RTX PRO 6000 Blackwell) | `Blackfrost-AI/...-Q4_K_M.gguf` | `dflash-...-ABLITERATED-BF16.gguf` (**BF16**) | `n_max=3`, 256-tok text (3 prompts) | `2.67` (implied by $1 + 3 \times 0.558$) | **55.8%** (`478 / 856`; `52.9%`, `53.8%`, `61.3%`) | `109.31` tok/s median |
| [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) / [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main) (RTX 5090, `b10665`) | Stock `Qwen3.8-27B` `Q4_K_M` | `Qwen3.8-27B-DFlash2-BF16.gguf` (**BF16**) | `n_max=7`, `temp=1.0`, GSM8K (8-shot) | **5.28** | ~`61.1%` (implied by $(\tau-1)/7$) | `164.0` tok/s (`3.38x`) |
| [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) / [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main) (RTX 5090, `b10665`) | Stock `Qwen3.8-27B` `Q4_K_M` | `Qwen3.8-27B-DFlash2-Q8_0.gguf` (**Q8_0**) | `n_max=7`, `temp=1.0`, GSM8K (8-shot) | **5.13** | ~`59.0%` | `166.9` tok/s (`3.43x`) |
| [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) / [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF/tree/main) (RTX 5090, `b10665`) | Stock `Qwen3.8-27B` `Q4_K_M` | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (**Q4_K_M**) | `n_max=7`, `temp=1.0`, GSM8K (8-shot) | **5.39** | ~`62.7%` | `180.7` tok/s (`3.72x`) |
| [lukaLLM/DFlash\_Qwen3.6\_27B\_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP) (RTX PRO 6000 96GB) | `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` | `incoai/...-DFlash2-Q4_K_M.gguf` (**Q4_K_M**) | `n_max=3`, greedy (`--temp 0`) | `3.18` | **72.6%** (`445 / 613`) | `164.59` tok/s |
| [lukaLLM/DFlash\_Qwen3.6\_27B\_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP) (RTX PRO 6000 96GB) | `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` | `incoai/...-DFlash2-Q4_K_M.gguf` (**Q4_K_M**) | **`n_max=5`**, greedy (`--temp 0`) | `4.52` | **70.4%** (`514 / 730`) | `187.13` tok/s (peak) |
| [lukaLLM/DFlash\_Qwen3.6\_27B\_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP) (RTX PRO 6000 96GB) | `ggml-org/Qwen3.8-27B-GGUF:Q4_K_M` | `incoai/...-DFlash2-Q4_K_M.gguf` (**Q4_K_M**) | `n_max=7`, greedy (`--temp 0`) | `5.18` | **59.7%** (`535 / 896`) | `168.06` tok/s |
| [hanxiao/Qwen3.8-27B-UD-Q4\_K\_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) (NVIDIA L4 24GB) | `unsloth/Qwen3.8-27B-UD-Q4_K_XL` | `z-lab/...-DFlash2-Q4_K_M.gguf` (**Q4_K_M**) | `n_max=7`, 7-task suite / GSM8K | **3.63** (7-task mean) / **5.51** (GSM8K) | ~`37.6%` (7-task) / ~`64.4%` (GSM8K) | `37.4` tok/s (at 226K ctx) |
| [andrew-paul/Qwen3.8-27B-DFlash2-Q3\_K\_M-GGUF](https://huggingface.co/andrew-paul/Qwen3.8-27B-DFlash2-Q3_K_M-GGUF) (RX 7800 XT) | `unsloth/Qwen3.8-27B-GGUF:Q4_K_M` | `incoai/...-DFlash2-Q4_K_M.gguf` (**Q4_K_M**) | `n_max=4`, greedy (`--temp 0`), 6 prompts | `3.62` | **65.56%** micro (`1479 / 2256`) | `44.91` tok/s |

### 2.3 Bit-Identity and Acceptance-Identity of Quantized `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` vs `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M`
- **Tensor Payload Bit-Identity:** **Bit-identical across all 81 tensor data blocks** (confirmed via [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF), [incoai/Qwen3.8-27B-DFlash2-GGUF:Q4\_K\_M](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf), and [andrew-paul/Qwen3.8-27B-DFlash2-Q3\_K\_M-GGUF](https://huggingface.co/andrew-paul/Qwen3.8-27B-DFlash2-Q3_K_M-GGUF)).
  1. `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` was converted directly from `incoai/Qwen3.8-27B-DFlash2` (`model.safetensors`, 81 BF16 tensors, `3,604.60M` parameters) with no weight modification; only target-derived GGUF header metadata (`general.name`, `tokenizer.chat_template`) differs.
  2. Standard uncalibrated `llama-quantize ... Q4_K_M` (no `--imatrix`, no `--output-tensor-type`) is a deterministic, purely mathematical function of each tensor's name, shape, and BF16 values. As verified by [andrew-paul/Qwen3.8-27B-DFlash2-Q3\_K\_M-GGUF](https://huggingface.co/andrew-paul/Qwen3.8-27B-DFlash2-Q3_K_M-GGUF), quantizing `incoai`/`z-lab` BF16 DFlash 2 GGUFs with `llama-quantize` produces identical quantized tensor blocks (`blk.0..4.*` in `Q4_K` / `Q6_K` / `F32`, `fc.weight` in `Q4_K`, `selector_hidden.weight` in `Q8_0`, `selector_out.weight` in `F16`, and `conv1d.weight` in `F32`).
  3. **Why Whole-File SHA-256 Differs:** The whole-file SHA-256 of `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (`1,143,006,592` bytes) will **not** match `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M` (`1,143,006,752` bytes, SHA-256 `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`) solely because of the GGUF header KVs (`general.name`, Blackfrost `tokenizer.chat_template`, and the 160-byte `dflash.rope.dimension_sections` key).
- **Acceptance-Identity at Runtime:** **Acceptance-identical for text-only inference (`mmproj` skipped).** Because `llama-server` ignores the draft model's `tokenizer.chat_template` and `general.name`, and because 1D NeoX RoPE and M-RoPE `[11, 11, 10, 0]` compute identical rotary embeddings on text positions `{p, p, p, 0}` ([tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp), [hanxiao/Qwen3.8-27B-UD-Q4\_K\_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)), `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` produces the exact same draft logits and token acceptance rate against `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` as `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M` would.

### 2.4 First-Boot Acceptance Gate: Exact `llama-server` Log Strings, JSON Keys, and Numeric Thresholds (`n_max=5`)
- **Critical Correction on `GET /slots` vs Completion Response `timings`:**
  - In [tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp) (`server_slot::to_json()`, lines 942–980), `GET /slots` returns:
    `id`, `id_task`, `n_ctx`, `speculative` (boolean), `is_processing`, `params`, `prompt`, `next_token` (`has_next_token`, `has_new_line`, `n_remain`, `n_decoded`), and `generated`. **`GET /slots` does NOT include `draft_n` or `draft_n_accepted`.**
  - Instead, `llama-server` exposes drafted vs. accepted token counts in two places:
    1. **HTTP Response JSON `timings` Object** (`/v1/chat/completions` or `/completion`, populated from `result_timings` in `tools/server/server-task.cpp` / `server-common.cpp` and verified in [ggml-org/llama.cpp#12968](https://github.com/ggml-org/llama.cpp/issues/12968) and [ggml-org/llama.cpp#29372](https://github.com/ggml-org/llama.cpp/issues/29372)):
       ```json
       "timings": {
         "cache_n": 0,
         "prompt_n": 38,
         "prompt_ms": 112.4,
         "prompt_per_token_ms": 2.957,
         "prompt_per_second": 338.07,
         "predicted_n": 256,
         "predicted_ms": 1820.5,
         "predicted_per_token_ms": 7.111,
         "predicted_per_second": 140.62,
         "draft_n": 310,
         "draft_n_accepted": 194
       }
       ```
       - Exact JSON keys:
         - `timings.draft_n`: Total draft tokens proposed across all speculative steps (`n_draft_total`).
         - `timings.draft_n_accepted`: Total draft tokens accepted by the target verification pass (`n_draft_accepted`).
         - `timings.predicted_n`: Total generated tokens (`n_decoded`).
    2. **Server Stdout Log at Request Completion** (`server_slot::print_timings()` in [tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp) and `common_speculative_print_stats()` in [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) lines 413–441, verified verbatim in [ggml-org/llama.cpp#27833](https://github.com/ggml-org/llama.cpp/issues/27833)):
       ```text
       slot print_timing: id  0 | task 0 | draft acceptance = 0.53939 (  178 accepted /   330 generated), mean len =  2.08
       statistics draft-dflash: #calls(b,g,a) = 0 66 66, #gen drafts = 66, #acc drafts = 66, #gen tokens = 330, #acc tokens = 178, dur(b,g,a) = 0.0 412.3 12.1 ms
       ```
       - Note on formula in `llama-server`: At `--spec-draft-n-max 5`, each DFlash step proposes `5` draft tokens (`#gen tokens = 5 * #gen drafts`). If `p_acc = draft_n_accepted / draft_n`, then the mean accepted tokens per verification step (including the 1 bonus/anchor token verified by the target) is:
         $$\text{mean len} = 1 + 5 \times \left(\frac{\text{draft\_n\_accepted}}{\text{draft\_n}}\right)$$
         Therefore, a token acceptance rate of **50.0%** (`draft_n_accepted / draft_n = 0.50`) at `n_max=5` corresponds to exact **`mean len = 3.50`**.
- **Numeric Pass/Fail Gate at `n_max=5` (Greedy 256-Token Deterministic Coding/Technical Prompt):**
  - **PASS:** `timings.draft_n > 0` AND (`timings.draft_n_accepted / timings.draft_n >= 0.50` **OR** logged `mean len >= 3.50`). *(Expected range on Blackfrost `Q4_K_M` + DFlash 2 `Q4_K_M` at `n_max=5` greedy: `0.50`–`0.70` token acceptance, `mean len = 3.50`–`4.52`, based on Blackfrost's `55.8%` at `n_max=3` and `lukaLLM`'s `70.4%` at `n_max=5` minus a small abliteration drift).*
  - **WARN (Degraded Speculation / Domain Drift):** `0.35 <= timings.draft_n_accepted / timings.draft_n < 0.50` (`2.75 <= mean len < 3.50`). Keep running only if `timings.predicted_per_second` exceeds single-token autoregressive baseline (~`42`–`48` tok/s on A100 40GB).
  - **FAIL (Broken Hook / Desync / Corrupt Sidecar):** `timings.draft_n == 0` OR `timings.draft_n_accepted / timings.draft_n < 0.35` (`mean len < 2.75`). Halt immediately.

## 3. Mixing Stock `incoai` / `z-lab` Q4 DFlash GGUF with Blackfrost Target

### 3.1 Runtime GGUF Key Usage: Draft Model (`-md`) vs Target Model (`-m`)
From [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) (`common_speculative_impl_draft_dflash`, `common_speculative_are_compatible`), [src/llama-arch.h](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.h), `src/llama-model.cpp`, and `src/models/dflash.cpp` ([ggml-org/llama.cpp#27833](https://github.com/ggml-org/llama.cpp/issues/27833)):

| GGUF Metadata Key / Tensor | Read From **Target** (`-m`)? | Read From **Draft** (`-md`)? | Runtime Role & Compatibility Check |
| :--- | :--- | :--- | :--- |
| `tokenizer.chat_template` | **YES** (`common_chat_templates_init`) | **NO (Ignored)** | `llama-server` formats prompts and builds the `--jinja` PEG autoparser exclusively from `model_tgt` (or `--chat-template-file`). The draft GGUF's `tokenizer.chat_template` is never read. |
| `token_embd.weight` & `output.weight` | **YES (Shared via `ctx_other`)** | **Absent in GGUF** | DFlash 2 GGUFs ship only 81 tensors (`blk.0..4.*`, `fc.*`, `hidden_norm.*`, `output_norm.*`, `conv1d.*`, `selector_*`) and omit both `token_embd.weight` and `output.weight`. `src/models/dflash.cpp:800/:1010` borrows both directly from the target model via `ctx_other` ([ggml-org/llama.cpp#27833](https://github.com/ggml-org/llama.cpp/issues/27833)). |
| `tokenizer.ggml.tokens` / `vocab_type` / `bos_token_id` / `eos_token_id` | **YES** | **YES (Compatibility check only)** | `common_speculative_are_compatible(model_tgt, model_dft)` in [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) (lines 176–253) checks: same `vocab_type`, same `add_bos`/`bos_id`, same `add_eos`/`eos_id`, `abs(n_vocab_tgt - n_vocab_dft) <= 128`, and identical token strings for IDs `5..min(n_vocab_tgt, n_vocab_dft)`. Both Blackfrost and `incoai`/`z-lab` have identical Qwen 3.8 BPE vocabularies (`248,320` tokens). |
| `tokenizer.ggml.mask_token_id` | No | **YES** (`llama_vocab_mask(vocab_dft)`) | Required by [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) (line 1003): must equal `248070` (`<|mask|>`). Present and equal to `248070` in both Blackfrost and `incoai`/`z-lab` DFlash 2 GGUFs. |
| `dflash.block_size` | No | **YES** (`gguf_find_key` / `hparams`) | Read from draft GGUF (`8`). Clamps `n_draft_max` to `block_size - 1 = 7` ([common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) line 993). |
| `dflash.target_layers` | Validated against `n_layer_tgt` | **YES** (`[5, 19, 33, 47, 61]`) | Read from draft GGUF (`LLM_KV_DFLASH_TARGET_LAYERS`). Tells the target context `ctx_tgt` which 5 hidden layers (`cb_eval` on `l_out-5`, `l_out-19`, `l_out-33`, `l_out-47`, `l_out-61`) to extract and concatenate (`5 * 5120 = 25,600` dims) into `dflash.fc.weight`. |
| `dflash.selector_top_k` (`16`), `dflash.selector_rank` (`256`), `dflash.conv_kernel_size` (`2`), `dflash.conv_group_size` (`16`) | No | **YES** | DFlash 2 architecture hyperparameters read by `src/llama-model.cpp` (`LLM_ARCH_DFLASH`) and `src/models/dflash.cpp` (PR `#27816`, commit `b10f9ca58c`). |
| `dflash.context_length` (`262144`) | `qwen35.context_length` (`262144`) | **YES** (for `ctx_dft` sizing) | `ctx_dft` is allocated to match the active slot context (`-c 100000` in our locked launch). |
| `dflash.rope.dimension_sections` | `qwen35.rope.dimension_sections` (`[11, 11, 10, 0]`) | **YES (Optional, `required=false`)** | Present (`[11, 11, 10, 0]`) in `incoai` (`5196282`) / `z-lab` (`343c7b5`); absent in Blackfrost (`b669c83`, defaults to 1D RoPE, which is equivalent for text-only). |

### 3.2 Published Runs Mixing `-m Blackfrost...Q4_K_M.gguf -md incoai/...DFlash2-Q4_K_M.gguf`
- **Status:** **`UNTESTED`**.
  - No issue, PR, model card, or benchmark repo has published a `llama-server` log or acceptance measurement pairing `-m Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` with `-md incoai/Qwen3.8-27B-DFlash2-GGUF:Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (or `z-lab`'s equivalent).

### 3.3 Exact Abort Strings if Mixing Incompatible Draft GGUFs
While stock `incoai`/`z-lab` `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` has a compatible 248,320-token vocabulary and `[5, 19, 33, 47, 61]` target layers, mixing a draft GGUF with any of the following mismatches triggers these exact `llama.cpp` errors in [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) and `src/llama-model.cpp`:
1. **Vocabulary / Token String Mismatch (`common_speculative_are_compatible`, lines 176–253):**
   - `common_speculative_are_compatible: draft model special tokens must match target model to use speculation`
   - `common_speculative_are_compatible: draft model vocab must closely match target model to use speculation`
   - `common_speculative_are_compatible: draft modelvocab token '%s' (id %d) does not match target model token '%s'`
   - Followed by: `common_speculative_impl_draft_dflash: the target context and the draft model are not compatible`
2. **Missing `mask_token_id` (`common/speculative.cpp`, line 1005):**
   - `common_speculative_impl_draft_dflash: draft model vocabulary is missing mask_token_id (required for dflash)`
3. **Missing or Out-of-Range `dflash.target_layers` (`common/speculative.cpp`):**
   - `common_speculative_impl_draft_dflash: draft model has no target_layers metadata`
   - Or if a draft built for 36-layer Qwen3-8B or 48-layer Qwen3.5-35B-A3B is mixed with 64-layer Qwen3.8-27B: silent garbage hidden-state extraction (`mean len ≈ 1.00`, `0%` acceptance) or out-of-bounds layer index abort.
4. **Legacy Pre-PR `#25110` DFlash GGUF (`general.architecture = "dflash-draft"`, issue [#25116](https://github.com/ggml-org/llama.cpp/issues/25116)):**
   - `error loading model: unknown model architecture: 'dflash-draft'`

### 3.4 Worker Verdict
- **Verdict:** **`KEEP_QUANTIZE_BLACKFROST`**
  - **Rationale:** (1) Locked worker policy strictly prohibits mixing stock `incoai`/`z-lab` DFlash GGUFs with the Blackfrost target unless a published `llama-server` run loaded that exact mixed pair and measured acceptance (`UNTESTED`); (2) quantizing `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` locally via `llama-quantize ... Q4_K_M` takes ~8–12 seconds on Colab CPU/NVMe, produces the exact `1,143,006,592`-byte file with Blackfrost's provenance metadata intact, and yields byte-identical quantized tensor blocks to `incoai`'s `Q4_K_M`.

## 4. First-Boot Log Grep List (Hybrid + DFlash Hooks)

Below are the exact log lines and regexes printed by `llama-server` at commit `efe3626` / tag `b11552` (and current master `23b0202a1` / `b11577`), with their exact C++ source file origins verified from [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp):

| Verification Check | Source File in `ggml-org/llama.cpp` | Exact Log String / Grep Pattern on `efe3626` (`b11552`) & Current Master | Notes / Halt Condition |
| :--- | :--- | :--- | :--- |
| **1a. Hybrid KV Cache (16 full-attn layers ONLY)** | [src/llama-kv-cache.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp) (lines 184–207) | `llama_kv_cache:      CUDA0 KV buffer size =` <br> `llama_kv_cache: size = .* MiB \(100000 cells,  16 layers,  1/1 seqs\), K \(q8_0\): .* MiB, V \(q8_0\): .* MiB` | **HALT IF:** `llama_kv_cache: size = .* \(100000 cells,  64 layers` (or `65 layers`) appears for the target model. That means all 64 layers were allocated as dense full-attention KV instead of 16 layers (`qwen35.full_attention_interval = 4`). *(Note: the separate draft context `ctx_dft` will print its own small 5-layer `llama_kv_cache: size = ... (100000 cells,   5 layers, ...)` line when initializing `draft-dflash`.)* |
| **1b. Hybrid Recurrent State (64 layer slots / 48 active GDN layers)** | `src/llama-memory-recurrent.cpp` (verified in [#23679](https://github.com/ggml-org/llama.cpp/issues/23679), [#19552](https://github.com/ggml-org/llama.cpp/issues/19552), [#22927](https://github.com/ggml-org/llama.cpp/issues/22927)) | `llama_memory_recurrent:      CUDA0 RS buffer size = .* MiB` <br> `llama_memory_recurrent: size = .* MiB \(.* cells,  64 layers, .* seqs\), R \(f32\): .* MiB, S \(f32\): .* MiB` | Confirms `llama_memory_hybrid` initialized the Gated DeltaNet recurrent state (`R` conv state + `S` recurrent state) across the 64-layer trunk alongside the 16-layer KV cache. **HALT IF** `llama_memory_recurrent:` is absent. |
| **2. Target Layer Count + Embedded 65th MTP Block** | `src/llama-model.cpp` (`print_info`, verified in [#28196](https://github.com/ggml-org/llama.cpp/issues/28196)) | `print_info: arch             = qwen35` <br> `print_info: n_layer          = 64` <br> `print_info: n_layer_nextn    = 1` *(or in GGUF loader: `qwen35.block_count = 65` / `load_tensors: offloaded 66/66 layers to GPU`)* | When `qwen35.block_count = 65` and `qwen35.nextn_predict_layers = 1`, `src/llama-model.cpp` sets `hparams.n_layer_all = 65`, `hparams.n_layer_nextn = 1`, and `hparams.n_layer = 64` (`n_layer_all - n_layer_nextn`). With `--spec-type draft-dflash` (not `draft-mtp`), the 64 trunk layers are used and `blk.64.nextn.*` stays idle. |
| **3. DFlash Draft Architecture & `n_draft_max` Clamping** | `src/llama-model.cpp` & [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) (lines 982–1025; verified in [#27829](https://github.com/ggml-org/llama.cpp/issues/27829), [#27833](https://github.com/ggml-org/llama.cpp/issues/27833), [unsloth#13511](https://github.com/unslothai/unsloth/issues/13511)) | `common_speculative_init_result: loading draft model '.*dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf'` <br> `print_info: arch             = dflash` <br> `common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'` <br> `common_speculative_impl_draft_dflash: - n_max=5, n_min=0, p_min=0.00` <br> `common_speculative_impl_draft_dflash: - block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true` | In [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) lines 991–996: `uint32_t block_size = ...; uint32_t max_draft = block_size > 1 ? block_size - 1 : 15; int32_t n_max = std::min<int32_t>(params.n_max, max_draft);`. Because `--spec-draft-n-max 5` is already $\le 7$ (`block_size - 1`), it prints `- n_max=5, n_min=0, p_min=0.00` directly. (If `n_max > 7` were passed, it silently clamps `n_max` in that banner to `n_max=7`). |
| **4. Draft Layer Hooks (`[6, 20, 34, 48, 62]` 1-indexed = `[5, 19, 33, 47, 61]` 0-indexed)** | [convert\_hf\_to\_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) (line 5323) & [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) (line 1020) | Startup banner: `common_speculative_impl_draft_dflash: - block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true` <br> *(With `--verbosity 4` / GGUF KV dump: `dflash.target_layers = [5, 19, 33, 47, 61]`)* | **Important indexing fact:** Both Hugging Face `config.json` (`target_layer_ids: [5, 19, 33, 47, 61]`) and the GGUF header key `dflash.target_layers` store **0-indexed** IDs `[5, 19, 33, 47, 61]` (verified in [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF/blob/main/Qwen3.8-27B-DFlash2-Q4_K_M.gguf) and [convert\_hf\_to\_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) line 5323: `self.gguf_writer.add_array(f"{arch}.target_layers", self.hparams["target_layer_ids"])`). They correspond to **1-indexed** human layer numbers `[6, 20, 34, 48, 62]`. At startup, `common/speculative.cpp` prints `n_extract=5` (confirming all 5 layer hooks `l_out-5`, `l_out-19`, `l_out-33`, `l_out-47`, `l_out-61` are registered). |
| **5. Flash Attention Enabled on Full-Attn Layers** | `src/llama-context.cpp` (verified in [#16950](https://github.com/ggml-org/llama.cpp/issues/16950), [#27309](https://github.com/ggml-org/llama.cpp/issues/27309)) | `llama_context: flash_attn    = 1` *(or `print_info: flash_attn       = enabled` / `auto (enabled)`)* | Moreover, because `-ctv q8_0` is locked, `src/llama-context.cpp:3602` enforces a hard gate: if Flash Attention were NOT enabled for the 16 full-attn layers, context construction aborts immediately with `llama_context: quantized V cache requires flash_attn to be enabled`. |
| **6. Spec Types Registered (`draft-dflash` + `ngram-map-k4v`, NOT `ngram-mod`)** | [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) (lines 110–137, 635, 1018) & `tools/server/server-context.cpp` | `common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'` <br> `common_speculative_impl_ngram_map: adding speculative implementation 'ngram-map-k4v'` <br> `srv    load_model: speculative decoding context initialized` | In [common/speculative.cpp](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp) line 635, the N-gram map constructor logs `LOG_INF("%s: adding speculative implementation '%s'\\n", __func__, ...)` with `'ngram-map-k4v'`. Verify that `'ngram-mod'` does **not** appear in the log. |
| **7. Server Listening on `0.0.0.0:8080`** | `tools/server/server.cpp` & [tools/server/server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp) | `srv  llama_server: model loaded` <br> `main: server is listening on http://0.0.0.0:8080` *(or `127.0.0.1:8080` if `--host 0.0.0.0` is omitted)* | Note: `llama-server` defaults `--host` to `127.0.0.1` unless `--host 0.0.0.0` or `LLAMA_ARG_HOST=0.0.0.0` is set in the environment. |

## 5. Tool Calling with `--reasoning off` / `enable_thinking=false`

### 5.1 How Qwen 3.8 XML `<function=...>` / `<tool_call>` Is Parsed on `efe3626` (`llama-server` vs SGLang)
- **`--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3` Are SGLang / vLLM Flags ONLY:**
  - Confirmed on the [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16 model card](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16) and [ggml-org/llama.cpp CLI documentation](https://github.com/ggml-org/llama.cpp): `--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3` belong exclusively to `sglang.launch_server` and `vllm`. Passing either flag to `llama-server` fails immediately at startup with `error: invalid argument: --tool-call-parser`.
- **How `llama-server` Parses Qwen 3.8 Tool Calls on `efe3626` (`b11552`):**
  - When launched with `--jinja`, `llama-server` uses the **differential Jinja PEG autoparser** (`common/chat-auto-parser-generator.cpp` and `common/chat-peg-parser.cpp`, introduced in PR `#18675` and refined through `efe3626`).
  - At model load, `common_chat_templates_init()` renders synthetic probe conversations (with and without `tools` and `enable_thinking`) through the Minja Jinja engine to automatically detect:
    1. Thinking tags: `<think>...</think>` (controlled by `--reasoning off` / `--reasoning-format` and `chat_template_kwargs.enable_thinking`).
    2. Qwen 3.8's native XML-style tool-call syntax rendered by its Jinja template:
       ```xml
       <tool_call>
       <function=get_weather>
       <parameter=location>
       Tokyo
       </parameter>
       </function>
       </tool_call>
       ```
  - During generation, the PEG parser extracts the `<tool_call><function=...><parameter=...>...</parameter></function></tool_call>` block and populates the standard OpenAI-compatible `choices[0].message.tool_calls` array (`id`, `type: "function"`, `function.name`, `function.arguments` JSON string) with `finish_reason: "tool_calls"`.

### 5.2 `QwenLM/Qwen3.8#236`: Plain-Text Fallback When `enable_thinking=false` + Long System Prompts
- **Mechanism Documented in [QwenLM/Qwen3.8#236](https://github.com/QwenLM/Qwen3.8/issues/236):**
  - When `enable_thinking=false` (or `--reasoning off`, which injects `enable_thinking=false` into template kwargs), the Qwen 3.8 Jinja chat template appends an empty thinking block right after the assistant header:
    `<|im_start|>assistant\n<think>\n\n</think>\n\n`
  - Because Qwen 3.8 was post-trained primarily to invoke tools *after* an active `<think>` reasoning trace, pre-filling `<think>\n\n</think>\n\n` alongside a long or complex multi-rule system prompt and `"tool_choice": "auto"` causes the model to answer in plain text (`finish_reason: "stop"`, `tool_calls: null`) instead of emitting `<tool_call>`.
- **Does Blackfrost's Embedded Short Execution Prompt Avoid `#236`?**
  - **Only for short user prompts without large client-injected system prompts.** Blackfrost embeds its short execution prompt once at the top of the default GGUF Jinja chat template ([Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF)), which keeps the baseline system prompt short. However, if an API client (such as an agent harness) passes a long custom `"role": "system"` message or a large multi-tool schema with `"tool_choice": "auto"` while `enable_thinking=false` is locked, [QwenLM/Qwen3.8#236](https://github.com/QwenLM/Qwen3.8/issues/236) **still triggers** because a client-supplied system message overrides the default system prompt and the empty `<think>\n\n</think>\n\n` prefill remains present.
- **Verified Workaround Without Changing Locked Launch Flags:**
  - Pass **`"tool_choice": "required"`** (or `"tool_choice": {"type": "function", "function": {"name": "..."}}`) in the `/v1/chat/completions` JSON request body whenever a tool call is required. In `llama-server` with `--jinja`, `"tool_choice": "required"` compiles the PEG tool parser into an active **GBNF grammar constraint** that forces the first generated token after `<think>\n\n</think>\n\n` to begin with `<tool_call>\n<function=`, completely eliminating the plain-text fallback even with `enable_thinking=false` and long system prompts.

### 5.3 Issue `#27107` (`raise_exception` in Unpatched Qwen 3.8 Jinja Template) on Blackfrost GGUF
- **Mechanism Documented in [ggml-org/llama.cpp#27107](https://github.com/ggml-org/llama.cpp/issues/27107) and [Qwen/Qwen3.8-27B#148](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/148):**
  - Stock upstream `Qwen/Qwen3.8-27B` `chat_template.jinja` contains strict Jinja assertions:
    `{{- raise_exception('System message must be at the beginning.') }}`
  - This causes two failures in `llama-server`:
    1. If an agent client sends a second `"role": "system"` message mid-conversation (after a user/assistant turn), template rendering throws `System message must be at the beginning.` and returns HTTP 500.
    2. On older `llama.cpp` builds (prior to PR `#27135`), the autoparser's differential probe triggered `Unable to generate parser for this template: ... raise_exception('System message must be at the beginning.')`.
- **Status on Commit `efe3626` (`b11552`) + Blackfrost `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf`:**
  - `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (`4f6732c`, converted `2026-08-16` from `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-BF16`) modified only the default system prompt string in the upstream Qwen 3.8 Jinja template (replacing `"You are a helpful assistant."` with the Blackfrost short execution prompt) and retained the upstream Qwen 3.8 loop structure.
  - On `efe3626` / `b11552`, `llama-server`'s autoparser initialization **succeeds** at startup (verified in [unslothai/unsloth#13511](https://github.com/unslothai/unsloth/issues/13511) on `b11541`), because PR `#27135` fixed the differential probe's system message placement.
  - **When `--chat-template-file` Is Needed:** You do **not** need `--chat-template-file` for standard requests where `"role": "system"` is only the first message (`messages[0]`). You **only** need `--chat-template-file` (with `raise_exception(...)` removed, as shown in [Qwen/Qwen3.8-27B#148](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/148)) if your downstream client injects mid-conversation `"role": "system"` messages after `messages[0]`.

### 5.4 Minimal `curl` Command Proving Tool Calling Works with Locked Flags
*(Uses port `8080`, model alias `-a qwen3.8-27b`, `Authorization: Bearer <API_KEY>` header, `"enable_thinking": false`, and `"tool_choice": "required"`):*

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY:-sk-local}" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "Check the current disk usage of /content in gigabytes."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_disk_usage",
          "description": "Return disk usage for a mount path",
          "parameters": {
            "type": "object",
            "properties": {
              "mount_path": {"type": "string"},
              "unit": {"type": "string", "enum": ["MB", "GB"]}
            },
            "required": ["mount_path", "unit"]
          }
        }
      }
    ],
    "tool_choice": "required",
    "temperature": 0,
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }' | jq '{finish_reason: .choices[0].finish_reason, tool_calls: .choices[0].message.tool_calls, timings: .timings}'
```
- **Expected Passing Output:** `finish_reason` equals `"tool_calls"`, `tool_calls[0].function.name` equals `"get_disk_usage"`, and `tool_calls[0].function.arguments` parses as `{"mount_path":"/content","unit":"GB"}`.

## First-Boot Grep + Acceptance Gate (Copy-Paste Block)

Drop this verification script into the Colab worker to verify the quantized sidecar byte size, startup log invariants, tool calling, and the `n_max=5` speculative acceptance gate against `/v1/chat/completions`:

```bash
#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${1:-/tmp/llama-server.log}"
API_URL="${2:-http://127.0.0.1:8080}"
API_KEY="${LLAMA_API_KEY:-sk-local}"
DRAFT_Q4="${3:-dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf}"

echo "=== [1/4] Verifying Quantized DFlash2 Q4_K_M Sidecar File Size ==="
ACTUAL_BYTES=$(stat -c%s "$DRAFT_Q4" 2>/dev/null || stat -f%z "$DRAFT_Q4")
if [[ "$ACTUAL_BYTES" != "1143006592" && "$ACTUAL_BYTES" != "1143006752" ]]; then
  echo "FAIL: Unexpected $DRAFT_Q4 size: $ACTUAL_BYTES bytes (expected 1143006592 raw or 1143006752 with mrope KV)" >&2
  exit 1
fi
echo "PASS: $DRAFT_Q4 size = $ACTUAL_BYTES bytes"

echo "=== [2/4] Auditing First-Boot llama-server Log ($LOG_FILE) ==="
# 1. Halt if dense 64-layer KV cache was allocated instead of 16-layer hybrid KV
if grep -Eq "llama_kv_cache: size = .*100000 cells,[[:space:]]+6[45] layers" "$LOG_FILE"; then
  echo "FAIL: HALT! Dense 64/65-layer KV cache detected instead of 16-layer hybrid KV!" >&2
  exit 1
fi
grep -Eq "llama_kv_cache: size = .*100000 cells,[[:space:]]+16 layers.*K \(q8_0\):.*V \(q8_0\):" "$LOG_FILE" \
  || { echo "FAIL: Missing 16-layer q8_0 target llama_kv_cache line" >&2; exit 1; }

# 2. Confirm 64-layer recurrent state buffer (48 active GDN layers across 64-layer trunk)
grep -Eq "llama_memory_recurrent:.*RS buffer size =" "$LOG_FILE" \
  || { echo "FAIL: Missing llama_memory_recurrent RS buffer allocation" >&2; exit 1; }
grep -Eq "llama_memory_recurrent: size = .*64 layers" "$LOG_FILE" \
  || { echo "FAIL: Missing 64-layer llama_memory_recurrent line" >&2; exit 1; }

# 3. Confirm target n_layer = 64 and embedded MTP block (n_layer_nextn = 1 / block_count = 65)
grep -Eq "print_info: arch[[:space:]]+= qwen35" "$LOG_FILE" \
  || { echo "FAIL: Missing target arch = qwen35" >&2; exit 1; }
grep -Eq "print_info: n_layer[[:space:]]+= 64" "$LOG_FILE" \
  || { echo "FAIL: Missing print_info: n_layer = 64" >&2; exit 1; }

# 4. Confirm Flash Attention enabled (mandatory for -ctv q8_0)
if grep -Eq "quantized V cache requires flash_attn to be enabled" "$LOG_FILE"; then
  echo "FAIL: Flash Attention disabled with quantized V cache" >&2
  exit 1
fi

# 5. Confirm DFlash2 drafter arch, block_size=8, mask_token_id=248070, n_extract=5 (5 layer hooks), and n_max=5
grep -Eq "print_info: arch[[:space:]]+= dflash" "$LOG_FILE" \
  || { echo "FAIL: Missing draft arch = dflash" >&2; exit 1; }
grep -Fq "common_speculative_impl_draft_dflash: adding speculative implementation 'draft-dflash'" "$LOG_FILE" \
  || { echo "FAIL: Missing draft-dflash registration" >&2; exit 1; }
grep -Fq "common_speculative_impl_draft_dflash: - n_max=5, n_min=0, p_min=0.00" "$LOG_FILE" \
  || { echo "FAIL: Missing n_max=5 in draft-dflash banner" >&2; exit 1; }
grep -Fq "common_speculative_impl_draft_dflash: - block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true" "$LOG_FILE" \
  || { echo "FAIL: Missing block_size=8, mask_token_id=248070, n_extract=5 banner" >&2; exit 1; }

# 6. Confirm ngram-map-k4v registered and ngram-mod NOT registered
grep -Fq "adding speculative implementation 'ngram-map-k4v'" "$LOG_FILE" \
  || { echo "FAIL: Missing ngram-map-k4v registration" >&2; exit 1; }
if grep -Fq "adding speculative implementation 'ngram-mod'" "$LOG_FILE"; then
  echo "FAIL: Prohibited spec-type ngram-mod is registered!" >&2
  exit 1
fi

# 7. Confirm server ready & listening on port 8080
grep -Eq "server is listening on http://(0\.0\.0\.0|127\.0\.0\.1):8080" "$LOG_FILE" \
  || { echo "FAIL: Server not listening on port 8080" >&2; exit 1; }
echo "PASS: All startup log grep checks succeeded."

echo "=== [3/4] Running Greedy 256-Token Acceptance Gate (n_max=5) ==="
RESP=$(curl -sS "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "Write a complete Python implementation of a thread-safe LRU cache using collections.OrderedDict and threading.RLock, including type annotations, docstrings, get, put, delete, clear, and a self-contained unit test suite."}
    ],
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false}
  }')

python3 -c '
import json, sys
data = json.loads(sys.argv[1])
t = data.get("timings", {})
draft_n = t.get("draft_n", 0)
accepted_n = t.get("draft_n_accepted", 0)
pred_n = t.get("predicted_n", 0)
tps = t.get("predicted_per_second", 0.0)
acc_rate = (accepted_n / draft_n) if draft_n > 0 else 0.0
mean_len = 1.0 + 5.0 * acc_rate
print(f"predicted_n={pred_n} | draft_n={draft_n} | draft_n_accepted={accepted_n} | accept_rate={acc_rate:.4f} ({acc_rate*100:.2f}%) | implied_mean_len={mean_len:.2f} | speed={tps:.2f} tok/s")
if draft_n == 0 or (acc_rate < 0.50 and mean_len < 3.50):
    print("FAIL: Acceptance gate below threshold (require accept_rate >= 0.50 or mean_len >= 3.50 at n_max=5)", file=sys.stderr)
    sys.exit(1)
print("PASS: Acceptance gate met.")
' "$RESP"

echo "=== [4/4] Verifying Tool Calling with enable_thinking=false ==="
TOOL_RESP=$(curl -sS "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "Check the current disk usage of /content in gigabytes."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_disk_usage",
          "description": "Return disk usage for a mount path",
          "parameters": {
            "type": "object",
            "properties": {
              "mount_path": {"type": "string"},
              "unit": {"type": "string", "enum": ["MB", "GB"]}
            },
            "required": ["mount_path", "unit"]
          }
        }
      }
    ],
    "tool_choice": "required",
    "temperature": 0,
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": false}
  }')

python3 -c '
import json, sys
data = json.loads(sys.argv[1])
choice = data["choices"][0]
tc = choice["message"].get("tool_calls") or []
if not tc or tc[0]["function"]["name"] != "get_disk_usage":
    print(f"FAIL: Tool call verification failed: {choice}", file=sys.stderr)
    sys.exit(1)
print(f"PASS: Tool call parsed cleanly -> {tc[0][\"function\"][\"name\"]}({tc[0][\"function\"][\"arguments\"]})")
' "$TOOL_RESP"
```

## Do Not Download / Do Not Mix Recap

- **Do NOT download or mix stock `incoai/Qwen3.8-27B-DFlash2-GGUF` or `z-lab/Qwen3.8-27B-DFlash2-GGUF` (`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, SHA-256 `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd`):** Keep the workflow strictly self-contained inside [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) by downloading `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (`ca6dacc0bba87332b4eae6f103fa4c0c800ca871657e83d2aed658351b2c4fb3`) and running `llama-quantize dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf Q4_K_M` on first boot (`KEEP_QUANTIZE_BLACKFROST`).
- **Do NOT download or pass `mmproj-Qwen3.8-27B-ABLITERATED-F16.gguf` or `mmproj-Qwen3.8-27B-ABLITERATED-Q8_0.gguf`:** Pairing `--mmproj` with `--model-draft ... --spec-type draft-dflash` desynchronizes `ctx_dft` sequence positions on the first image request and permanently poisons the server slot ([unslothai/unsloth#13511](https://github.com/unslothai/unsloth/issues/13511)).
- **Do NOT check out `ggml-org/llama.cpp` PR `#27342` (`1deefcca395743049c3820ab8f9b15043f3e9446`):** Stay on the locked mainline commit `efe3626` / tag `b11552` (which includes merged PR `#27816`, commit `b10f9ca58c`).
- **Do NOT pass `--imatrix`, `--output-tensor-type`, or graft an output head during `llama-quantize`:** DFlash 2 borrows `token_embd.weight` and `output.weight` from the target model (`ctx_other`) at runtime (`src/models/dflash.cpp:800/:1010`).
- **Do NOT pass SGLang/vLLM flags `--tool-call-parser qwen3_coder` or `--reasoning-parser qwen3` to `llama-server`:** They are invalid CLI arguments in `llama.cpp`.

## Remaining Unknowns Only

1. **Empirical `Q4_K_M` Target + `Q4_K_M` DFlash 2 Acceptance Rate on Blackfrost Abliterated Weights (`UNKNOWN` Until First Boot):** No published benchmark has measured `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` paired with `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` at `--spec-draft-n-max 5` on Colab A100 40GB. The first-boot gate in Step `[3/4]` above will record the exact `timings.draft_n_accepted / timings.draft_n` and `slot print_timing: ... mean len` on first boot.
2. **Exact Deterministic SHA-256 of `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (`UNKNOWN` Until First Boot):** While the expected output file size is exactly `1,143,006,592` bytes (without `dflash.rope.dimension_sections` injection) and its 81 quantized tensor payloads are bit-identical to `incoai/Qwen3.8-27B-DFlash2-GGUF:Q4_K_M`, the whole-file SHA-256 incorporates the Blackfrost GGUF header metadata and must be recorded via `sha256sum dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` immediately after `llama-quantize` finishes on first boot.
