# llama.cpp Serving Plan & Flag Audit: Qwen 3.8 27B + DFlash 2 Speculative Decoding

**Target Architecture:** Qwen 3.8 27B (`qwen35` architecture: 64 total layers = 48 Gated DeltaNet recurrent linear-attention layers + 16 full-attention layers at `full_attention_interval = 4`) paired with the 5-layer (1.92B parameter) **DFlash 2** block-diffusion speculative drafter (`dflash.block_size = 8`) and a 4-gram key / 4-token value runtime lookup sidecar (`ngram-map-k4v`).

**Deployment Targets:** Google Colab **A100 40GB** (`sm_80`), **A100 80GB** (`sm_80`), and **G4 96GB** (NVIDIA RTX PRO 6000 Blackwell Server Edition, `sm_120`).

**Audit Baseline:** Upstream [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` as of October 14, 2026 (server documentation commit [efe3626](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), common parameter definitions [e750b88](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h), and release build tag [b11552 / 23b0202a1](https://hexdocs.pm/llama_cpp_ex/changelog.html)).

## Recommended Commit / Tag SHA

* **Primary Recommendation (Current `master` HEAD, October 2026):** Check out upstream `master` at commit **`efe3626`** (2026-10-13, [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)) or build tag **`b11552`** (commit **`23b0202a1`**, `llama.cpp 0.6.0` / `ggml 0.26.0`, [LlamaCppEx Changelog](https://hexdocs.pm/llama_cpp_ex/changelog.html)).

* **Stable Semantic Release Pin:** Tag **`v0.5.0`** / **`b11146`** (commit **`7fe450e19`**, [AgenticWire llama-server Reference](https://www.agenticwire.news/article/llama-server)).

* **Minimum Master Commit Supporting DFlash 2:** Tag **`b10658`** (commit **`b10f9ca58c`**, merged **2026-08-27** via [PR #27816](https://github.com/ggml-org/llama.cpp/issues/26669); verified container pin **`b10666` / `4e97ac86`** in [lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP)).

* **Do NOT check out `pull/27342/head` (`5ecbe1ac` / `b10498`):** That pre-merge branch predates the upstream candidate-selector API refactoring in `b10f9ca58c`, lacks the August 27, 2026 `dflash.rope.dimension_sections` mrope metadata fix ([Issue #27862](https://github.com/ggml-org/llama.cpp/issues/27862)), and misses two months of Blackwell (`sm_120`) and server fixes.

## 1. Code Lineage

### 1.1 Status of Cited Pull Requests on `llama.cpp` `master`

| PR Number | Title / Scope | Status on `master` (Oct 2026) | Merge Date & Commit SHA | Verdict for Playbook A | 
| ----- | ----- | ----- | ----- | ----- | 
| [**PR #27816**](https://github.com/ggml-org/llama.cpp/issues/26669) | Merge `xsn/dflash2`: DFlash 2 block-diffusion speculative decoding (two-tap local dynamic convolution + candidate path selector API + `p_min > 0.0` selector gate) | **Merged to `master`** | **2026-08-27** — Commit **`b10f9ca58c`** (first release tag: **`b10658`**, [lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP)) | **Built into `master`.** No separate branch fetch needed. | 
| [**PR #27342**](https://www.reddit.com/r/LocalLLaMA/comments/1w00jes/spec_add_dflash2_support_local_convolution/) | `spec : add DFlash2 support (local convolution + candidate selector)` by `SubSir` (initial PR into `ggml-org` intermediate branch `xsn/dflash2`) | **Merged into `xsn/dflash2`, then superseded on `master` by PR #27816** | **2026-08-27** — PR head was **`5ecbe1ac`** (`5c094f254`, \~`b10498`); landed on `master` via **`b10f9ca58c`** ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)) | **Obsolete as a standalone branch.** Delete `git fetch origin pull/27342/head:pr-27342` and `git switch pr-27342`. | 
| [**PR #27173**](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) | One-decode chained MTP drafting (`LLAMA_SPEC_CHAIN=1`) + leading vocabulary slice (`LLAMA_SPEC_CHAIN_SUB=16384`) for `draft-mtp` | **Not merged (external feature branch for MTP)** | Unmerged PR branch ([cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF](https://huggingface.co/cdiamond/Qwen3.8-27B-iMatrix-NVFP4-MTP-GGUF)) | **Obsolete / Irrelevant for DFlash 2.** Chained MTP eliminates intermediate target decode passes during autoregressive 1-token-per-step MTP drafting (`draft-mtp`). Because DFlash 2 (`draft-dflash`) is a block-diffusion drafter that emits the entire draft block in a single forward pass, chain drafting has zero work to remove ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)). | 

> **GGUF Conversion Lineage Warning:** Any DFlash 2 `.gguf` sidecar converted prior to the August 27, 2026 merge (`b10f9ca58c`) fails on current `master` with `error loading model: tensor 'blk.0.attn_k.weight' not found` or `dflash.rope.dimension_sections` metadata errors ([Issue #27862](https://github.com/ggml-org/llama.cpp/issues/27862)). Always pull post-merge GGUF sidecars such as [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF), [incoai/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF) (re-uploaded Aug 27, 2026: `"Fix the bug where conversion fails after merging into the master branch"`), or [Akicou/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/Akicou/Qwen3.8-27B-DFlash2-GGUF).

### 1.2 Correct Checkout for a Colab CUDA Build in October 2026

Use **`master`** (or a current October 2026 release tag such as **`b11552`** / **`efe3626`**). Do **not** check out `pr-27342`. Since `b10658` (`b10f9ca58c`), `master` includes:

1. Full DFlash 2 support (`--spec-type draft-dflash`) with two-tap local dynamic convolution (`dflash_conv1d`) and the finalized candidate path selector API ([lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP)).

2. Upstream per-architecture MMVQ/MMQ switch points (`ggml_cuda_should_use_mmvq`) for Blackwell (`sm_120`, Colab G4) introduced in [PR #26079 (2b5621094ef383cdcd8428ef6d22efe5df976532)](https://github.com/ggml-org/llama.cpp/issues/28090).

3. Reworked differential PEG chat/tool autoparsers ([PR #30210](https://github.com/ggml-org/llama.cpp/issues/27107)) and unified load-mode flags (`--load-mode`).

### 1.3 CMake Flags Required for CUDA `llama-server` + `llama-quantize`

When compiling directly inside an active Colab GPU session (where `nvidia-smi` sees the attached A100 or G4), CMake defaults `CMAKE_CUDA_ARCHITECTURES` to `native` and compiles for the attached GPU automatically:

```
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release -j "$(nproc)" --target llama-server llama-quantize

```

If you pre-build a single cached binary in Colab to run across **both** Colab A100 (`sm_80`) and Colab G4 (RTX PRO 6000 Blackwell, `sm_120`), explicitly specify `80-real;120a-real`—because `llama.cpp`'s CMake configuration disables PTX JIT fallback on `120a-real` to unlock native Blackwell instructions ([zephel01 llama.cpp CLI & Blackwell Guide](https://note.com/zephel01/n/n02145d6d88ef?hl=en)):

```
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --config Release -j "$(nproc)" --target llama-server llama-quantize

```

## 2. Every Flag in Playbook A

### 2.1 Master Flag & Environment Variable Audit Table

All flag names, defaults, and environment variable bindings below are verified against [tools/server/README.md (efe3626)](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md), and [common/common.h (e750b88)](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h).

| Old / Playbook A Flag or Env | Current `master` Flag / Env | Status on `master` | Technical Notes & Playbook A Action | 
| ----- | ----- | ----- | ----- | 
| `GGML_MMVQ_MAX=2` | *(None on stock `master`)* | **Not upstream** (custom patch only) | Only exists in [hanxiao's 0001-mmvq-runtime-crossover.patch](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4). Ignored by stock `master`. Counterproductive on A100 (`sm_80`) and redundant on G4 (`sm_120`) ([Issue #28090](https://github.com/ggml-org/llama.cpp/issues/28090)). **Delete.** | 
| `LLAMA_SCHED_POOL=8` | *(None on stock `master`)* | **Not upstream** (no-op) | Not defined in `ggml` or `common/common.h` on upstream `master`; appeared only in `hanxiao`'s local test harness ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)). **Delete.** | 
| `-m FNAME` | `-m, --model FNAME` (`LLAMA_ARG_MODEL`) | **Valid** | Path to the Qwen 3.8 27B target GGUF ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). **Keep.** | 
| `-md FNAME` | `-md, --spec-draft-model, --model-draft FNAME` (`LLAMA_ARG_SPEC_DRAFT_MODEL`) | **Valid** | Short alias `-md` and canonical long form `--spec-draft-model` are both active ([docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)). **Keep.** | 
| `--spec-type draft-dflash,ngram-map-k4v` | `--spec-type TYPE` (`LLAMA_ARG_SPEC_TYPE`) | **Valid** | Comma-separated list from `none`, `draft-simple`, `draft-eagle3`, `draft-dflash`, `draft-dspark`, `draft-mtp`, `ngram-cache`, `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod` ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). **Keep exact string.** | 
| `--spec-draft-n-max 5` | `--spec-draft-n-max N` (`LLAMA_ARG_SPEC_DRAFT_N_MAX`) | **Valid** (replaces removed `--draft`, `--draft-n`, `--draft-max`) | Default is `3` (`common_params_speculative_draft::n_max = 3` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)). Setting `5` is the empirical throughput sweet spot for DFlash 2 ([lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP)). **Keep `5`.** | 
| `--spec-draft-n-min 1` (or `0` in older blueprint) | `--spec-draft-n-min N` (`LLAMA_ARG_SPEC_DRAFT_N_MIN`) | **Valid** (replaces removed `--draft-min`, `--draft-n-min`) | Default is `0` (`int32_t n_min = 0` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)). Discards any draft where `result.size() < n_min`. **Use `0` (default) or `1`.** | 
| `--spec-draft-p-min 0` (older blueprint) | `--spec-draft-p-min P` (`LLAMA_ARG_SPEC_DRAFT_P_MIN`) | **Valid** (replaces removed `--draft-p-min`) | Default is `0.00` (`float p_min = 0.0f` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)). In PR #27816 (`b10f9ca58c`), gates selector-confidence truncation when `p_min > 0.0`. At default `0.0`, it is a no-op. **Omit (leave at default `0.0`).** | 
| `-ngl -1` (or `-ngl all` in older blueprint) | `-ngl, --gpu-layers, --n-gpu-layers N` (`LLAMA_ARG_N_GPU_LAYERS`) | **Valid**, accept `N`, `auto` (`-1`), or `all` (`-2`) | Default is `auto` (`-1`). **TRAP:** Explicitly passing `-ngl -1` or `-ngl all` alongside `-c 100000` and `--fit on` causes `common_fit_params()` to abort with `n_gpu_layers already set by user to ..., abort` ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)). **Delete `-ngl` when using `--fit on` + `-c`.** | 
| `--spec-draft-ngl 99` / `-ngld all` | `--spec-draft-ngl, -ngld, --gpu-layers-draft, --n-gpu-layers-draft N` (`LLAMA_ARG_N_GPU_LAYERS_DRAFT`) | **Valid** (both `--spec-draft-ngl` and `-ngld` are active aliases) | Default is `auto` (`-1` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)); accepts number, `auto`, or `all` (`-2`). Use `--spec-draft-ngl all` (or `99`) to pin all 5 DFlash 2 layers to CUDA ([docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)). **Keep.** | 
| `-fa on` | `-fa, --flash-attn [on|off|auto]` (`LLAMA_ARG_FLASH_ATTN`) | **Valid** | Default is `auto`. Explicitly takes `on`, `off`, or `auto` ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). Required when `-ctv` is quantized (`q8_0`). **Keep `-fa on`.** | 
| `--fit on` | `-fit, --fit [on|off]` (`LLAMA_ARG_FIT`) | **Valid** | Default is `on` (`bool fit_params = true` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)). Adjusts unset `-c` and/or unset `-ngl` to fit device memory. **Keep `--fit on`.** | 
| `--fit-target 768` | `-fitt, --fit-target MiB0,MiB1,...` (`LLAMA_ARG_FIT_TARGET`) | **Valid** | Default is `1024` MiB per device ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). Preserves `768 MiB` of unallocated CUDA headroom for runtime scratch allocations. **Keep `--fit-target 768`.** | 
| `--ctx-checkpoints 4` | `-ctxcp, --ctx-checkpoints, --swa-checkpoints N` (`LLAMA_ARG_CTX_CHECKPOINTS`) | **Valid** | Default is `32` (`n_ctx_checkpoints = 32` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)). Caps host-RAM recurrent-state (GDN) rollback snapshots per slot (`4` snapshots × `~299 MiB` = `~1.2 GiB` instead of `32` × `299 MiB` = `~9.6 GiB`). **Keep `--ctx-checkpoints 4`.** | 
| `--checkpoint-min-step 16384` | `-cms, --checkpoint-min-step N` (`LLAMA_ARG_CHECKPOINT_MIN_SPACING_NT`) | **Valid** (replaces older `--checkpoint-every-n-tokens`) | Default is `8192` tokens (`0 = no minimum` in [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). Spaces GDN recurrent checkpoints every 16,384 tokens across a 100k context window. **Keep `--checkpoint-min-step 16384`.** | 
| `-b 2048` | `-b, --batch-size N` (`LLAMA_ARG_BATCH`) | **Valid** | Default is `2048` (`n_batch = 2048` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)). Logical maximum batch size for prompt ingestion. **Keep `-b 2048`.** | 
| `-ub 256` | `-ub, --ubatch-size N` (`LLAMA_ARG_UBATCH`) | **Valid** | Default is `512` (`n_ubatch = 512` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)). Controls physical microbatch size and CUDA compute buffer size. On 40–96 GB GPUs (A100/G4), default `-ub 512` is faster for 100k prompt prefill than `-ub 256` (which was a 24 GB L4 VRAM-saving cut). **Use `-ub 512` (or keep `-ub 256`).** | 
| `-ctk q8_0` | `-ctk, --cache-type-k TYPE` (`LLAMA_ARG_CACHE_TYPE_K`) | **Valid** | Default is `f16` ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). Sets target model K cache type (for the 16 full-attention layers). **Keep `-ctk q8_0` (or `f16` on 80–96 GB).** | 
| `-ctv q8_0` | `-ctv, --cache-type-v TYPE` (`LLAMA_ARG_CACHE_TYPE_V`) | **Valid** | Default is `f16` ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). Sets target model V cache type (requires `-fa on`). **Keep `-ctv q8_0` (or `f16` on 80–96 GB).** | 
| `-bs` | `-bs, --backend-sampling` (`LLAMA_ARG_BACKEND_SAMPLING`) | **Valid** | Default is `disabled` for the target model (`backend_sampling = false` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)). Keeps target-model verification sampling on the GPU instead of copying `248,320 × 4 B = 993 KB` of logits per verified token (`~6 MB` per block) over PCIe to host RAM ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)). **Keep `-bs`.** | 
| `--no-mmap` | **Renamed to `-lm none` / `--load-mode none`** (`LLAMA_ARG_LOAD_MODE`) | **Deprecated / Replaced** | `--mmap`, `--no-mmap`, and `-dio` were consolidated into `-lm, --load-mode [auto|none|mmap|mlock|mmap+mlock|dio]` ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [zephel01 CLI Guide](https://note.com/zephel01/n/n02145d6d88ef?hl=en)). **Replace `--no-mmap` with `--load-mode none`.** | 
| `--parallel 1` | `-np, --parallel N` (`LLAMA_ARG_N_PARALLEL`) | **Valid** | Default is `-1` (`auto`, which spawns up to 4 concurrent slots). Setting `--parallel 1` allocates a single dedicated slot with 1× recurrent GDN state ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). **Keep `--parallel 1`.** | 
| `--jinja` | `--jinja, --no-jinja` (`LLAMA_ARG_JINJA`) | **Valid** (now `enabled` by default) | Default is `true` (`use_jinja = true` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h)). Explicitly passing `--jinja` remains valid. **Keep `--jinja`.** | 
| `--reasoning off` | `-rea, --reasoning [on|off|auto]` (`LLAMA_ARG_REASONING`) | **Valid** | Default is `auto`. Passing `--reasoning off` sets `enable_reasoning = 0` (`enable_thinking=false` in Jinja + `--reasoning-budget 0`) ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). **Keep `--reasoning off`.** | 
| `--host 0.0.0.0` | `--host HOST` (`LLAMA_ARG_HOST`) | **Valid** | Default is `127.0.0.1` ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). **Keep `--host 0.0.0.0`.** | 
| `--port 8080` | `--port PORT` (`LLAMA_ARG_PORT`) | **Valid — Mandatory** | **CRITICAL CHANGE:** In October 2026 (`efe3626` / `b11552`), the upstream default port changed from `8080` to **`9931`** ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [LlamaCppEx Changelog](https://hexdocs.pm/llama_cpp_ex/changelog.html)). You **must** pass `--port 8080` explicitly to bind `8080`. | 
| `-c 100000` *(missing in Playbook A)* | `-c, --ctx-size N` (`LLAMA_ARG_CTX_SIZE`) | **Valid** | Default is `0` (loads `262144` from Qwen 3.8 GGUF metadata). Pass `-c 100000` (without `-ngl`) to lock a 100,000-token context window ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). **Add `-c 100000`.** | 

### 2.2 Detailed Answers to Flag Mechanics & Measured Traps

#### A. Exact Spelling of `--spec-type` and Why Stacking `ngram-mod` with `ngram-map-k4v` Fails on DFlash 2

* **Exact current spelling:** `--spec-type draft-dflash,ngram-map-k4v` (comma-separated, no spaces, verified in [docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) and [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).

* **Precedence rule in `llama.cpp`:** Whenever a draft-model implementation (`draft-dflash`) is paired with a draftless lookup implementation (`ngram-map-k4v`), the draftless implementation has **higher dispatch priority** ([docs/speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md): *"If a draft model is combined with a draftless decoding the draftless decoding has higher precedence"*). When `ngram-map-k4v` finds a 4-gram match with at least 4 drafted tokens inside the current sequence, `llama.cpp` skips the DFlash 2 GPU forward pass for that step and verifies the n-gram tokens directly.

* **Why stacking `ngram-mod,ngram-map-k4v` is wrong for DFlash 2:** Controlled 18-turn multi-phase coding benchmarks in [lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP (benchmark/bench_ngram.py)](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP) prove that while stacking `ngram-mod,ngram-map-k4v` helped Qwen 3.6 + DFlash 1 (where `block_size = 16`, `n_max = 15`), it **severely degrades Qwen 3.8 + DFlash 2** (`block_size = 8`):

  * `--spec-type draft-dflash,ngram-map-k4v` (single lookup table): **`304.92 tok/s`** in the build phase (**`4.68×`** over baseline), **`64.2%`** token acceptance rate, **`1.41`** drafts/token.

  * `--spec-type draft-dflash,ngram-mod,ngram-map-k4v` (stacked lookup tables): drops to **`245.84 tok/s`** (**`3.77×`**), **`55.6%`** token acceptance rate, **`1.59`** drafts/token—a **19.4% throughput loss** (even slower than pure `draft-dflash` with no n-gram at `254.99 tok/s`).

  * **Root cause:** With a tight 5-to-7 token draft window, registering two higher-priority n-gram drafters causes modulo-hash collisions in `ngram-mod` to preempt DFlash 2's high-accuracy block-diffusion forward pass with lower-precision n-gram guesses.

#### B. `--spec-draft-p-min`: No-Op on DFlash 2 or Not?

* **Version-dependent behavior:**

  1. **On pre-merge PR #27342 (`5ecbe1ac` / `b10498`):** `--spec-draft-p-min` was a **complete no-op** for DFlash 2. In `common/speculative.cpp` (lines 1219–1266), the `is_dflash2` branch extracted tokens from the candidate selector lattice and returned before reaching the `p_min` confidence threshold check ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4), [lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP)).

  2. **On current `master` after PR #27816 (`b10f9ca58c` / `b10658+`):** Upstream review moved DFlash 2's candidate selector to a dedicated API and **wired selector-confidence draft truncation into `--spec-draft-p-min`, gated on `p_min > 0.0`** ([lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP)).

  3. **Operational rule on `master`:** The upstream default in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h) is `float p_min = 0.0f` (`--spec-draft-p-min 0.00`). At `0.0`, the `p_min > 0.0` gate is **disabled (no-op)**, so DFlash 2 always emits its full `n_max` draft block (`5` tokens). Passing any positive `--spec-draft-p-min` (such as `0.75`) activates early selector truncation and reduces accepted tokens per verify step. Leave `--spec-draft-p-min` unset (default `0.0`).

#### C. `--spec-draft-n-min 0` vs `1`: Which Is Correct, and Does `n-min > draft_length` Silently Disable Speculation?

* **Silent disablement trap:** In `common/speculative.cpp`, after the active drafter (`draft-dflash` or `ngram-map-k4v`) finishes assembling `result`, `llama.cpp` evaluates:

  ```
  if (result.size() < (size_t) params.n_min) {
      result.clear();
  }
  
  ```

  When `result.size() < n_min`, the **entire draft vector is discarded** and the target model falls back to a standard 1-token autoregressive decode step with zero warning in standard logs ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)). Consequently, if you set `--spec-draft-n-min` larger than the number of tokens produced by the drafter (for instance, setting `--spec-draft-n-min 6` when `--spec-draft-n-max 5`), speculation is **100% silently disabled**.

* **`0` vs `1`:** With `p_min = 0.0`, DFlash 2 unconditionally emits `n_max = 5` tokens per round, and `ngram-map-k4v` emits up to 4–5 tokens on a hit. Therefore, both `0` and `1` pass when at least 1 token is drafted. However, **`--spec-draft-n-min 0` is the upstream default** in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h) (`int32_t n_min = 0`) and is the safest setting so a 1-token edge draft is never dropped.

#### D. `dflash.block_size = 8`: Clamping to `7` and the `need_n_rs_seq()` GDN VRAM Trap at `n = 8` or `9`

* **Why `n-max` clamps to `7`:** The Qwen 3.8 27B DFlash 2 GGUF metadata defines `dflash.block_size = 8` (representing `[1 anchor token + 7 mask positions]`). During speculative initialization in `common/speculative.cpp`, `llama.cpp` clamps the active draft depth to `n_draft_max = block_size - 1 = 7` ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)).

* **Why passing `n = 8` or `9` still wastes GDN VRAM via `need_n_rs_seq()`:** In [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h), `common_params_speculative::need_n_rs_seq()` computes the number of recurrent state sequence slots required by the target model's `llama_memory_recurrent` buffer **at target context creation time—before `common/speculative.cpp` clamps its local copy of `n_max` to `7`**:

  ```
  uint32_t need_n_rs_seq() const {
      // Sizes recurrent state slots directly from raw params.speculative.draft.n_max
  }
  
  ```

  Across Qwen 3.8 27B's **48 Gated DeltaNet layers**, each recurrent-state sequence slot (`R (f32)` conv state + `S (f32)` SSM state) occupies **`~299 MiB` of VRAM** ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)):

  * `--spec-draft-n-max 5` → allocates **6** GDN state slots (`1,796.25 MiB`), drafts **5** tokens.

  * `--spec-draft-n-max 7` → allocates **8** GDN state slots (`2,395.00 MiB`), drafts **7** tokens.

  * `--spec-draft-n-max 8` → allocates **9** GDN state slots (`2,694.38 MiB`), still clamps to **7** drafted tokens (**wastes `299.38 MiB` of VRAM**).

  * `--spec-draft-n-max 9` → allocates **10** GDN state slots (`2,993.75 MiB`), still clamps to **7** drafted tokens (**wastes `598.75 MiB` of VRAM**).

* **Why `--spec-draft-n-max 5` is locked:** In addition to saving **`598.75 MiB`** of recurrent buffer VRAM compared to `n = 7` (and avoiding 2 extra GDN state snapshot copies per verification step), [lukaLLM's draft-depth sweep (benchmark/speed_bench_v2.py)](https://github.com/lukaLLM/DFlash_Qwen3.6_27B_LlamaCPP) showed that `--spec-draft-n-max 5` achieves **`187.13 tok/s`** (`70.4%` acceptance) vs **`168.06 tok/s`** (`59.7%` acceptance) at `n = 7`—an **11.3% wall-clock speedup** because positions 6 and 7 suffer exponentially compounding step-to-step error while forcing larger target verification batches (`ne11 = 8` vs `ne11 = 6`).

#### E. Setting `-c 100000` Explicitly Together with `--fit on --fit-target 768`

* **The `common_fit_params()` mutual-exclusion rule:** In `llama.cpp`, `--fit on` can only auto-adjust parameters that the user left **unset** on the command line ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4), [zephel01 CLI Guide](https://note.com/zephel01/n/n02145d6d88ef?hl=en)):

  * If **both** `-c` and `-ngl` (even `-ngl -1` or `-ngl all`) are explicitly passed on the CLI, `common_fit_params()` logs:

    ```
    common_fit_params: context size set by user to 100000, abort
    common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to -1, abort
    
    ```

* **How to combine `-c 100000` with `--fit on --fit-target 768` cleanly:**

  * **Pass `-c 100000 --fit on --fit-target 768` and OMIT `-ngl` entirely** (leaving `n_gpu_layers` at its default `auto`).

  * At startup, `common_fit_params()` honors your explicit `-c 100000`, checks that total VRAM usage (target weights + draft weights + 100k KV cache + 6 GDN recurrent slots + compute buffers) leaves at least `768 MiB` free on the A100 40GB / A100 80GB / G4 96GB, and automatically assigns all 65/65 target layers to CUDA.

  * *(Alternative if you want `100000` as a floor while letting `--fit` expand up to 262,144 on 80–96 GB GPUs: omit `-c` and pass `-fitc 100000` / `--fit-ctx 100000`.)*

#### F. Setting Server-Default Sampling (`temperature`, `repetition_penalty`)

* **CLI flags exist for both (`tools/server/README.md`, `common/common.h`):** You do **not** need to rely solely on the HTTP request body; `llama-server` sets global default sampler parameters via CLI flags:

  * **Temperature:** `--temp N` or `--temperature N` (default: `0.80`, env: `LLAMA_ARG_TEMPERATURE`).

    * *Important GGUF Override Trap:* Unsloth and community Qwen 3.8 GGUFs embed `general.sampling.temp = 1.0` in GGUF metadata, which overrides `llama-server`'s `0.80` fallback unless `--temp` is explicitly provided on the CLI ([hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)). For Qwen 3.8 non-thinking mode, pass `--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0` (or `--temp 0.0` for greedy deterministic coding).

  * **Repetition penalty:** `--repeat-penalty N` (default: `1.00` = disabled, env: `LLAMA_ARG_REPEAT_PENALTY`) and `--repeat-last-n N` (default: `64`, env: `LLAMA_ARG_REPEAT_LAST_N`). Note the exact CLI spelling is `--repeat-penalty` (not `--repetition-penalty`).

  * **Presence / Frequency penalty:** `--presence-penalty N` (default: `0.00`) and `--frequency-penalty N` (default: `0.00`).

  * **Interaction with `-bs` (`--backend-sampling`):** Keep `--repeat-penalty 1.0` and `--presence-penalty 0.0` at their defaults unless needed, because active token-history penalties force a host-side penalty pass or degrade GPU backend-sampler fast paths ([Issue #24871](https://github.com/ggml-org/llama.cpp/issues/24871)). Any per-request value passed in `/v1/chat/completions` JSON (`"temperature"`, `"repeat_penalty"`, `"presence_penalty"`) overrides the CLI default for that request.

#### G. `--reasoning off` vs `chat_template` / Jinja for `enable_thinking=false` on Qwen 3.8

* **How `--reasoning off` works on current `master`:** Passing `--reasoning off` (or `-rea off`, `env: LLAMA_ARG_REASONING`) sets `params.enable_reasoning = 0` ([common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h), [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)). When `--jinja` is enabled, `--reasoning off` automatically injects `enable_thinking=false` into the Qwen 3.8 Jinja template evaluation context **and** sets `--reasoning-budget 0`.

* **What the Qwen 3.8 Jinja template emits when `enable_thinking=false`:** As verified in [Qwen/Qwen3.8-27B's chat_template.jinja](https://github.com/ggml-org/llama.cpp/issues/27107), when `enable_thinking is defined and enable_thinking is false`, the template appends an empty closed thinking block right after the assistant header:

  ```
  <|im_start|>assistant
  <think>
  
  </think>
  
  
  ```

  This forces Qwen 3.8 to begin generating the final response immediately without emitting chain-of-thought tokens, which is mandatory for high DFlash 2 acceptance (`~70%`).

* **Belt-and-suspenders CLI flag:** You can also pass `--chat-template-kwargs '{"enable_thinking":false}'` (`env: LLAMA_ARG_CHAT_TEMPLATE_KWARGS`, [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)) alongside `--reasoning off` so that even if a client sends `"reasoning_effort"` or custom template kwargs, `enable_thinking` defaults to `false`.

#### H. DFlash 2 Sidecar Quantization across 40–96 GB GPUs (`Q4_K_M` 1.14 GB vs `BF16` 3.86 GB)

* **Verdict:** **Keep `Q4_K_M` (`Qwen3.8-27B-DFlash2-Q4_K_M.gguf`, 1.14 GB) as the default DFlash 2 sidecar across all three GPU classes (A100 40GB, A100 80GB, and G4 96GB). Do NOT switch to the `BF16` (3.86 GB) sidecar.**

* **Why `BF16` is slower despite ample VRAM on 40–96 GB GPUs:**

  1. **Zero acceptance advantage (actually slightly worse):** In the official evaluation published by the DFlash authors on [z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF), mean accepted draft length ($\tau$) across 8 benchmarks is **`5.39` for `Q4_K_M` (1.14 GB)** vs **`5.28` for `BF16` (3.86 GB)** and **`5.13` for `Q8_0` (2.06 GB)**. Independent benchmarking in [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) confirmed `Q4_K_M` outperforms `Q8_0` in both acceptance and end-to-end throughput (`35.99` vs `35.39 tok/s`).

  2. **Drafting is strictly memory-bandwidth bound:** Every speculative verification cycle reads the entire 5-layer, 1.92B-parameter DFlash 2 sidecar once from GPU memory. Switching from `Q4_K_M` (`1.06 GiB`) to `BF16` (`3.60 GiB`) forces the GPU to stream **2.54 GiB of additional drafter weights on every single draft step** (`3.4×` more bytes per draft block), adding \~1.3–1.7 ms of pure memory-read overhead per block on A100 (1,555–2,039 GB/s) and G4 (1,792 GB/s) for zero additional accepted tokens.

  3. **Where to spend the extra 40–96 GB VRAM instead:** Spend the abundant VRAM of A100 80GB and G4 96GB on **unquantized `f16` target KV cache (`-ctk f16 -ctv f16`)** at `-c 100000` (which only costs **`6.10 GiB`** across Qwen 3.8's 16 full-attention layers and eliminates K/V quantization overhead during 100k prefill and verify), while keeping `-ctkd f16 -ctvd f16` (default) for the DFlash 2 sidecar.

| GPU Class | VRAM & Arch | Recommended Target Model (`-m`) | Recommended DFlash 2 Sidecar (`-md`) | Recommended KV Flags at `-c 100000` | Total Est. VRAM Footprint | 
| ----- | ----- | ----- | ----- | ----- | ----- | 
| **Colab A100 40GB** | 40 GB HBM2 (`sm_80`, 1,555 GB/s) | `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (\~16.5 GiB) | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (**1.14 GB**, [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)) | `-ctk q8_0 -ctv q8_0` (`3.25 GiB` target KV + `0.61 GiB` draft KV) | **\~23.8 GiB** (\~16 GiB free) | 
| **Colab A100 80GB** | 80 GB HBM2e (`sm_80`, 2,039 GB/s) | `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (\~16.5 GiB) | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (**1.14 GB**, [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)) | `-ctk f16 -ctv f16` (`6.10 GiB` target KV + `0.61 GiB` draft KV) | **\~26.7 GiB** (\~53 GiB free) | 
| **Colab G4 96GB** | 96 GB GDDR7 (RTX PRO 6000 Blackwell, `sm_120`, 1,792 GB/s) | `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (\~16.5 GiB) | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` (**1.14 GB**, [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)) | `-ctk f16 -ctv f16` (`6.10 GiB` target KV + `0.61 GiB` draft KV) | **\~26.7 GiB** (\~69 GiB free) | 

## 3. MMVQ Patch (`GGML_MMVQ_MAX=2`)

### 3.1 Stock Environment Variable vs. `hanxiao`'s Custom Patch

* **`GGML_MMVQ_MAX` is NOT a stock environment variable on `llama.cpp` `master`.** It exists solely in `patches/0001-mmvq-runtime-crossover.patch` inside [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4). Passing `GGML_MMVQ_MAX=2` to an unpatched `master` binary has zero effect.

### 3.2 Reproduction of `hanxiao`'s Patch

In [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4) (`patches/0001-mmvq-runtime-crossover.patch`), the patch modifies `ggml_cuda_should_use_mmvq()` in **`ggml/src/ggml-cuda/mmvq.cu`** (where `MMVQ_MAX_BATCH_SIZE` is defined as `8` in `ggml/src/ggml-cuda/mmvq.cuh`, [cyyself/tune-ggml-mmvq.py](https://gist.github.com/cyyself/e4b6b3351abd5f1d728369b130c7c488)) so that the batch-width (`ne11`) crossover from the vector-dot kernel (`mul_mat_vec_q` / **MMVQ**) to the tiled tensor-core matrix kernel (`mul_mat_q` / **MMQ**) can be lowered at runtime via `getenv("GGML_MMVQ_MAX")`:

* **File:** `ggml/src/ggml-cuda/mmvq.cu`

* **Function:** `bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11)`

* **Diff Reproduction:**

```
--- a/ggml/src/ggml-cuda/mmvq.cu
+++ b/ggml/src/ggml-cuda/mmvq.cu
@@ -78,5 +78,6 @@ bool ggml_cuda_should_use_mmvq(enum ggml_type type, int cc, int64_t ne11) {
-    return ne11 <= MMVQ_MAX_BATCH_SIZE;
+    static const int mmvq_max = getenv("GGML_MMVQ_MAX") ? atoi(getenv("GGML_MMVQ_MAX")) : MMVQ_MAX_BATCH_SIZE;
+    return ne11 <= mmvq_max;
 }

```

*(Note: `hanxiao`'s patch file also supports per-quant environment overrides `GGML_MMVQ_MAX_Q6K` and `GGML_MMVQ_MAX_IQ4XS`, and includes a second hunk in `common/speculative.cpp` changing the per-step draft acceptance trace from `LOG_TRC` to `LOG_INF`, [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)).*

### 3.3 Does Current `master` Already Route Verify Width $\ge 3$ to MMQ, and Is the Patch Needed on A100 / G4?

* **What current `master` does ([PR #26079, commit 2b5621094ef383cdcd8428ef6d22efe5df976532](https://github.com/ggml-org/llama.cpp/issues/28090)):**
  In `ggml/src/ggml-cuda/mmvq.cu`, `ggml_cuda_should_use_mmvq(type, cc, ne11)` now contains a compile-time `switch (cc)` table covering **Blackwell (`GGML_CUDA_CC_BLACKWELL`, `sm_120`)**, **DGX Spark (`sm_121`)**, **Ada Lovelace (`GGML_CUDA_CC_ADA_LOVELACE`, `sm_89`)**, and **CDNA1/CDNA2** ([cyyself/tune-ggml-mmvq.py](https://gist.github.com/cyyself/e4b6b3351abd5f1d728369b130c7c488), [Issue #28090](https://github.com/ggml-org/llama.cpp/issues/28090)):

  1. **On Colab G4 (RTX PRO 6000 Blackwell, `sm_120`):** Current `master` **already routes K-quants (`Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`) to MMQ at small speculative verify widths (`ne11 >= 2–4`)** via the `GGML_CUDA_CC_BLACKWELL` case in `ggml_cuda_should_use_mmvq()`. The patch is **completely unnecessary on G4**.

  2. **On Colab A100 40GB / 80GB (Ampere `sm_80`):** `sm_80` has no dedicated case in `ggml_cuda_should_use_mmvq()` on `master`, so it falls through to `return ne11 <= MMVQ_MAX_BATCH_SIZE;` (`8`). At `--spec-draft-n-max 5` (target verify width `ne11 = 6`), stock `master` on A100 runs **MMVQ** (`mul_mat_vec_q`).

  3. **Why you must NOT apply `GGML_MMVQ_MAX=2` on A100 (`sm_80`):** As demonstrated in the 28-table kernel benchmark across GPU architectures in [Issue #28090 (manunicholasjacob & zhihz)](https://github.com/ggml-org/llama.cpp/issues/28090), `hanxiao`'s `GGML_MMVQ_MAX=2` patch sped up the **L4 (`sm_89`)** only because the L4 has just 58 SMs throttled by a 72W TDP cap. On **A100 (`sm_80`, 108 SMs, 1,555–2,039 GB/s HBM2/e)**, MMVQ has ample SMs and memory bandwidth at `ne11 = 5..8`, whereas forcing MMQ at `ne11 = 6` is **measurably slower across every model and quant tested** (`force_mmq / selected` ratio at `ne11 = 5, 6, 7, 8` on A100-SXM4 is **`0.70, 0.80, 0.85, 0.92`** on `Q4_0` and **`0.85, 0.94, 0.96, 0.99`** on `Q8_0`—meaning forcing MMQ at `ne11 = 6` costs a **6% to 20% kernel slowdown** on A100).

* **Bottom Line:** **Delete the `GGML_MMVQ_MAX=2` source patch and env var.** Stock `master` already selects the optimal kernel path for both **G4 (`sm_120`, MMQ via PR #26079)** and **A100 (`sm_80`, MMVQ up to `ne11 <= 8`)**.

## 4. HTTP API & Log Verification

### 4.1 Health Endpoint & Ready JSON Shape

* **Supported paths:** **Both `/health` and `/v1/health` are served identically** ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): *"GET /health: Returns health check result. This endpoint is public (no API key check). /v1/health also works."*).

* **JSON shape when ready (`HTTP 200 OK`):**

  ```
  {"status":"ok"}
  
  ```

* **JSON shape while models are still loading (`HTTP 503 Service Unavailable`):**

  ```
  {"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}
  
  ```

### 4.2 OpenAI-Compatible Endpoints & Advertised Model ID String

* **Endpoints:**

  * `GET /v1/models` (OpenAI models list)

  * `POST /v1/chat/completions` (OpenAI chat completions, streaming and non-streaming)

  * `POST /v1/completions` (raw text completions)

  * `POST /v1/responses` (OpenAI Responses API)

  * `POST /v1/messages` (Anthropic Messages API compatible endpoint, [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md))

* **What model ID string does `llama-server` advertise?**

  * **Default (when `-a` / `--alias` is omitted):** `GET /v1/models` advertises the **exact file path passed to `-m`** (e.g., `"id": "/content/models/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf"`, or the `-hf` repo string if loaded via `-hf`) ([tools/server/README.md line 1203](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).

  * **Clean model ID recommendation:** Pass **`-a qwen3.8-27b`** (`--alias qwen3.8-27b`, `env: LLAMA_ARG_ALIAS`) on the command line so `GET /v1/models` advertises `"id": "qwen3.8-27b"`. *(Note: In single-model mode without `--models-dir`, `POST /v1/chat/completions` accepts any `"model"` string in the request body and routes to the loaded model).*

### 4.3 Authentication (`--api-key`)

* **Exact flags:**

  * **`--api-key KEY`** (`env: LLAMA_API_KEY`): Accepts a single key or a comma-separated list of valid API keys ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).

  * **`--api-key-file FNAME`** (`env: LLAMA_ARG_API_KEY_FILE`): Path to a file containing one API key per line.

* **Request header:** Clients pass `Authorization: Bearer <KEY>` (or `x-api-key: <KEY>` on `/v1/messages`). `/health` and `/v1/health` remain public (`200 OK` without a header), while `/v1/models`, `/v1/chat/completions`, `/props`, `/slots`, and `/metrics` return `401 Unauthorized` without a valid key.

### 4.4 Tool Calling for Qwen 3.8 on `llama-server`

* **Do `--tool-call-parser qwen3_coder` or `--reasoning-parser qwen3` exist in `llama-server`?**

  * **No.** Those flags belong exclusively to **SGLang / vLLM** ([QwenLM/Qwen3.8#236](https://github.com/QwenLM/Qwen3.8/issues/236)) and will cause `llama-server` to exit with `error: invalid argument`.

* **What is the `llama.cpp` equivalent?**

  * **`--jinja`** (enabled by default on `master`) + **`--reasoning-format auto`** (default, `env: LLAMA_ARG_THINK`, which maps to `deepseek` and extracts any `<think>...</think>` blocks into `message.reasoning_content`) + **`--reasoning off`** ([tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).

  * Under `--jinja`, `llama-server` uses its built-in **differential PEG autoparser** (`common/chat.h`, reworked in PR #18675 and [PR #30210](https://github.com/ggml-org/llama.cpp/issues/27107)), which inspects the model's Jinja template at startup (`Chat format: peg-constructed`) and automatically constructs a PEG grammar + stream parser for the model's exact tool-call tags.

  * *(Warning: Do **not** pass `--tools all` on `llama-server`—that flag enables built-in host shell/filesystem execution for the Web UI agent, not OpenAI API function calling!)*

* **XML `<function=...>` vs. Hermes JSON `<tool_call>` on Qwen 3.8:**

  * Unlike Qwen 2.5 / Qwen 3 (which used Hermes JSON inside `<tool_call>`), **Qwen 3.8 uses the Qwen3-Coder XML-inside-`<tool_call>` format** as defined in [Qwen/Qwen3.8-27B's chat_template.jinja](https://github.com/ggml-org/llama.cpp/issues/27107):

    ```
    <tool_call>
    <function=get_weather>
    <parameter=location>
    Tokyo
    </parameter>
    </function>
    </tool_call>
    
    ```

  * `llama-server`'s `peg-constructed` autoparser parses this XML `<tool_call><function=...><parameter=...>` output and translates it into standard OpenAI `tool_calls[].function.{name, arguments}` JSON in the `/v1/chat/completions` response.

* **Critical Qwen 3.8 Jinja `raise_exception` Bug ([Issue #27107](https://github.com/ggml-org/llama.cpp/issues/27107)):**

  * The stock `Qwen/Qwen3.8-27B` Jinja template contains five strict `{{ raise_exception(...) }}` clauses (such as `raise_exception('System message must be at the beginning.')`). When `llama-server`'s differential autoparser probes the template during startup or when an OpenAI client (like Claude Code or Cursor) injects mid-conversation system instructions, the stock template fails with `Unable to generate parser for this template: ... raise_exception('System message must be at the beginning.')`.

  * **Fix:** If your GGUF carries the stock unpatched Qwen 3.8 template (Unsloth GGUFs already patch this), strip the `raise_exception` lines into a local `qwen3.8-fixed.jinja` file and pass **`--chat-template-file qwen3.8-fixed.jinja`** ([Issue #27107](https://github.com/ggml-org/llama.cpp/issues/27107), [tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).

### 4.5 Flash-Attention & Hybrid 48 GDN + 16 Full-Attention Log Verification

* **How to confirm `-fa on` is active in logs:**
  During context construction (`llama_context: constructing llama_context`), `llama-server` prints:

  ```
  llama_context: flash_attn = enabled
  
  ```

* **How to confirm the model loaded as "48 GDN + 16 full-attn" vs. "dense 64-layer":**
  In `llama.cpp`, a dense 64-layer transformer allocates a 64-layer `llama_kv_cache` and **never** instantiates `llama_memory_recurrent`. A hybrid Qwen 3.8 27B (`qwen35`) model with `qwen35.block_count = 64` and `qwen35.full_attention_interval = 4` prints **both** a **16-layer** `llama_kv_cache` (for the $64 / 4 = 16$ full-attention layers) **and** a **64-layer** `llama_memory_recurrent` buffer (holding the `R (f32)` conv and `S (f32)` state tensors for the 48 Gated DeltaNet layers) ([Issue #20002](https://github.com/ggml-org/llama.cpp/issues/20002), [hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4](https://github.com/hanxiao/Qwen3.8-27B-UD-Q4_K_XL-L4)):

  ```
  llama_model_loader: - kv   0: general.architecture str = qwen35
  llama_model_loader: - kv   5: qwen35.block_count u32 = 64
  llama_model_loader: - kv  22: qwen35.full_attention_interval u32 = 4
  print_info: arch             = qwen35
  print_info: n_layer          = 64
  print_info: ssm_d_conv       = 4
  print_info: ssm_d_state      = 128
  llama_kv_cache:      CUDA0 KV buffer size =  3320.31 MiB
  llama_kv_cache: size =  3320.31 MiB (100000 cells,  16 layers,  1/1 seqs), K (q8_0): 1660.16 MiB, V (q8_0): 1660.16 MiB
  llama_memory_recurrent:      CUDA0 RS buffer size =  1796.25 MiB
  llama_memory_recurrent: size =  1796.25 MiB (     6 cells,  64 layers,  6 seqs), R (f32):   135.00 MiB, S (f32):  1661.25 MiB
  
  ```

  Seeing **`16 layers`** inside `llama_kv_cache: size = ...` paired with `llama_memory_recurrent: size = ... (6 cells, 64 layers, 6 seqs)` is the definitive proof that the model is running as **48 GDN + 16 full-attention** with **`--spec-draft-n-max 5`** (`5 + 1 = 6` recurrent state sequences).

## 5. Corrected Playbook A

### 5.1 Summary of Flag Changes

#### Flags & Steps to Delete from Old Playbook A

1. **`git fetch origin pull/27342/head:pr-27342 && git switch pr-27342`** — Delete; DFlash 2 is on `master` since commit `b10f9ca58c` (`b10658`, Aug 27, 2026).

2. **`GGML_MMVQ_MAX=2` patch and environment variable** — Delete; not a stock env var, redundant on G4 (`sm_120` via PR #26079), and 6–20% slower on A100 (`sm_80`) ([Issue #28090](https://github.com/ggml-org/llama.cpp/issues/28090)).

3. **`LLAMA_SCHED_POOL=8`** — Delete; not an upstream `llama.cpp` environment variable.

4. **`-ngl -1` (and `-ngl all`)** — Delete; explicitly setting `-ngl` alongside `-c 100000` causes `--fit on` (`common_fit_params`) to abort. Leave `-ngl` unset (`auto`).

5. **`--no-mmap`** — Delete; deprecated on `master` in favor of `--load-mode none` (`-lm none`).

#### Flags to Add / Update in Corrected Playbook A

1. **`-c 100000`** — Explicitly locks the 100,000-token context window (while `--fit on --fit-target 768` auto-places all 65/65 target layers on GPU with a 768 MiB safety reserve).

2. **`--load-mode none`** — Current `master` replacement for `--no-mmap`.

3. **`--spec-draft-n-min 0`** — Uses the upstream default (`0`) so no valid 1-token draft is discarded.

4. **`--ubatch-size 512` (`-ub 512`)** — Restores full 512-token microbatch prefill speed on 40–96 GB GPUs.

5. **`--temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --repeat-penalty 1.0`** — Explicitly overrides any GGUF-embedded `general.sampling.temp = 1.0` metadata with Qwen 3.8 non-thinking defaults.

6. **`--chat-template-kwargs '{"enable_thinking":false}'`** — Reinforces `--reasoning off` at the Jinja template level.

7. **`-a qwen3.8-27b` (`--alias qwen3.8-27b`)** — Advertises a clean model ID on `GET /v1/models` instead of a local filesystem path.

8. **`--api-key "${LLAMA_API_KEY}"`** — Secures `/v1/*` endpoints while keeping `/health` and `/v1/health` open for readiness probes.

### 5.2 Corrected Compile Block (`master` / `efe3626` / `b11552`)

```
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
# Checkout current master (or pin to verified October 2026 commit efe3626 / tag b11552)
git checkout master

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build --config Release -j "$(nproc)" --target llama-server llama-quantize

```

### 5.3 Corrected Launch Block (Colab A100 40GB, A100 80GB, and G4 96GB)

> **KV Cache Note:** On **A100 40GB**, keep `-ctk q8_0 -ctv q8_0` (`3.25 GiB` KV cache at `-c 100000`). On **A100 80GB** and **G4 96GB**, you can either run `-ctk q8_0 -ctv q8_0` or upgrade to `-ctk f16 -ctv f16` (`6.10 GiB` KV cache at `-c 100000`). Keep the **Q4_K_M** DFlash 2 sidecar (`1.14 GB`) on all three GPUs.

```
export LLAMA_API_KEY="sk-local-qwen38-secret"

./build/bin/llama-server \
  -m /content/models/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf \
  -md /content/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  -a qwen3.8-27b \
  --spec-type draft-dflash,ngram-map-k4v \
  --spec-draft-n-max 5 \
  --spec-draft-n-min 0 \
  --spec-draft-ngl all \
  -fa on \
  -c 100000 \
  --fit on --fit-target 768 \
  --ctx-checkpoints 4 --checkpoint-min-step 16384 \
  -b 2048 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  -bs --load-mode none --parallel 1 \
  --jinja --reasoning off --reasoning-format auto \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 \
  --api-key "${LLAMA_API_KEY}" \
  --host 0.0.0.0 --port 8080

```

*(If your target GGUF uses the unpatched stock Qwen 3.8 Jinja template and throws a `raise_exception` parser error on startup, append `--chat-template-file /content/models/qwen3.8-fixed.jinja` per [Issue #27107](https://github.com/ggml-org/llama.cpp/issues/27107)).*

## API Cheat Sheet (`health`, `models`, `chat`, `api-key`)

```
# 1. Health / Readiness Check (Public — no API key required; both /health and /v1/health work)
curl -s http://127.0.0.1:8080/health
# Expected ready response (HTTP 200): {"status":"ok"}
curl -s http://127.0.0.1:8080/v1/health
# Expected ready response (HTTP 200): {"status":"ok"}

# 2. List Models (Requires Bearer API Key; returns "id": "qwen3.8-27b" when -a qwen3.8-27b is set)
curl -s http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" | jq .

# 3. Chat Completion (Non-thinking mode enforced by server; includes speculative timings in response)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "system", "content": "You are a concise systems engineering assistant."},
      {"role": "user", "content": "Write a Python function to parse GGUF header magic bytes."}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }' | jq '{content: .choices[0].message.content, timings: .timings}'

# 4. OpenAI-Compatible Tool Calling (Parsed via llama-server --jinja PEG autoparser)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "Check the GPU temperature on node-01."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_gpu_temp",
          "description": "Get GPU temperature in Celsius for a cluster node",
          "parameters": {
            "type": "object",
            "properties": {
              "node": {"type": "string"}
            },
            "required": ["node"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }' | jq '.choices[0].message.tool_calls'

```

## Log Lines to Grep on First Boot

Run `grep -E "arch|full_attention_interval|flash_attn|llama_kv_cache: size|llama_memory_recurrent: size|speculative|thinking|Chat format|listening" server.log` on first boot and verify these exact lines:

1. **Hybrid 48 GDN + 16 Full-Attention Architecture Loaded:**

   ```
   llama_model_loader: - kv   0: general.architecture str = qwen35
   llama_model_loader: - kv   5: qwen35.block_count u32 = 64
   llama_model_loader: - kv  22: qwen35.full_attention_interval u32 = 4
   print_info: arch             = qwen35
   print_info: ssm_d_conv       = 4
   print_info: ssm_d_state      = 128
   
   ```

2. **16-Layer Full-Attention KV Cache + 6-Slot (`n_max=5`) Recurrent GDN State Buffer:**

   ```
   llama_kv_cache: size =  3320.31 MiB (100000 cells,  16 layers,  1/1 seqs), K (q8_0): 1660.16 MiB, V (q8_0): 1660.16 MiB
   llama_memory_recurrent: size =  1796.25 MiB (     6 cells,  64 layers,  6 seqs), R (f32):   135.00 MiB, S (f32):  1661.25 MiB
   
   ```

   *(If `llama_kv_cache` says `64 layers`, you loaded a dense model. If `llama_memory_recurrent` says `8 seqs` or `9 seqs`, `--spec-draft-n-max` was set above `5`).*

3. **Flash Attention Enabled on Both Target and DFlash 2 Drafter:**

   ```
   llama_context: flash_attn = enabled
   
   ```

   *(Must appear **twice**: once when constructing the target `qwen35` context and once when constructing the `dflash` draft context).*

4. **DFlash 2 Drafter Loaded (`arch = dflash`, `block_size = 8`) & Speculative Types Registered:**

   ```
   print_info: arch             = dflash
   print_info: n_layer          = 5
   common_speculative_init: ...
   
   ```

5. **Thinking Disabled (`thinking = 0`) & Differential PEG Tool Parser Active:**

   ```
   srv    init: init: chat template, thinking = 0
   Chat format: peg-constructed
   
   ```

6. **Server Ready and Listening on Port 8080:**

   ```
   main: server is listening on http://0.0.0.0:8080 - starting the main loop
   srv  update_slots: all slots are idle
   
   ```

## Remaining Unknowns

1. **Upstream Ampere (`sm_80`) Per-Quant MMVQ Switch Points ([Issue #28090](https://github.com/ggml-org/llama.cpp/issues/28090)):** Issue #28090 is actively tracking a follow-up PR to add dedicated `GGML_CUDA_CC_AMPERE` (`sm_80`) switch points into `ggml_cuda_should_use_mmvq()` in `ggml/src/ggml-cuda/mmvq.cu`. On A100-SXM4 (`sm_80`), the benchmark tables in `#28090` show MMVQ remains faster at `ne11 = 6` (`--spec-draft-n-max 5`) for `Q4_0`/`Q4_K`/`Q8_0`, but `IQ4_XS` crosses over around `ne11 = 5–6`. If your `ABLITERATED-Q4_K_M` GGUF mixes `IQ4_XS` tensors and a future commit merges an `sm_80` table, verify whether `sm_80` routing changes for `ne11 = 6`.

2. **Upstream Fix for `need_n_rs_seq()` Pre-Clamping Over-Allocation:** As of October 2026, `common_params_speculative::need_n_rs_seq()` in [common/common.h](https://github.com/ggml-org/llama.cpp/blob/master/common/common.h) still computes recurrent sequence slots before `common/speculative.cpp` reads `dflash.block_size` from the draft GGUF. Because we explicitly pass `--spec-draft-n-max 5` (which is `< 7`), no VRAM is wasted, but passing `n >= 8` remains a silent VRAM trap until `need_n_rs_seq()` is clamped against draft model metadata.

3. **Qwen 3.8 Tool-Call Reliability with `enable_thinking=false` on Very Long System Prompts ([QwenLM/Qwen3.8#236](https://github.com/QwenLM/Qwen3.8/issues/236)):** Community reports on `QwenLM/Qwen3.8#236` note that when `enable_thinking=false` (`--reasoning off`) is combined with multi-thousand-token agent system prompts (e.g., OpenCode / Claude Code), Qwen 3.8 27B can occasionally respond in plain text rather than emitting `<tool_call><function=...>`. If this occurs on specific agent turns, passing `"tool_choice": "required"` (or a named function) in the `/v1/chat/completions` request body forces grammar-constrained tool emission without needing to re-enable `<think>` tokens.