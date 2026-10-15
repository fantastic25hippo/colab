# A100-SXM4-40GB Speculative Decoding Speed Gates, Kernel Routing, Dual-Arch Fatbin, and Colab CLI Session Longevity

**Date:** October 14, 2026  
**Target Stack:** Blackfrost Qwen 3.8 27B ABLITERATED (`Q4_K_M`, 16.8 GB) + Quantized Blackfrost DFlash 2 (`Q4_K_M`, ~1.14 GiB) + `llama.cpp` (`master` commit `efe3626` / `b11552`) with `--spec-type draft-dflash,ngram-map-k4v`, `--spec-draft-n-max 5`, `--spec-draft-n-min 0`, thinking off, `-c 100000`, `-ctk q8_0 -ctv q8_0`, `-fa on`, `--parallel 1`  
**Target Runtime:** Google Colab AI Pro (200 CU/month) · Standard-shape `NVIDIA A100-SXM4-40GB` (`colab new -s qwen-worker --gpu A100`, ~5.37–5.40 CU/hr) · `google-colab-cli` v0.7.4

## Primary Sources

1. **lukaLLM DFlash 1 & DFlash 2 Benchmark Repository (July–August 2026):** [lukaLLM/DFlash\_Qwen3.6\_27B\_LlamaCPP](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP) — End-to-end `llama.cpp` benchmarks of `Qwen3.8-27B` (`Q4_K_M`) + `DFlash 2` (`Q4_K_M`) + `ngram-map-k4v` / `ngram-mod` on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition (`sm_120`, 97 GB, 1,792 GB/s).
2. **hanxiao Qwen 3.8 27B L4 Speculative Decoding & Kernel Study (August 2026):** [hanxiao/Qwen3.8-27B-UD-Q4\_K\_XL-L4](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4) — Cost model, active weight-read measurements (15.75 GiB), `DFlash 2` long-horizon decode retention (22K→174K), and `GGML_MMVQ_MAX` kernel crossover analysis on NVIDIA L4 (`sm_89`).
3. **Blackfrost Qwen 3.8 27B ABLITERATED GGUF Model Card (August 2026):** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF) — Standard K-quant ladder specification (confirming zero `IQ4_XS` / importance-matrix tensors) and `DFlash 2` measurements on an NVIDIA RTX PRO 6000 Blackwell Server Edition (`sm_120`, 96 GB).
4. **llama.cpp Issue #28090 — Ampere/Ada/Turing/Volta MMVQ Crossover Sweeps (August–October 2026):** [ggml-org/llama.cpp#28090](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F28090) — 28-table multi-architecture `force_mmq / selected` measurements (including `A100-SXM4-40GB` at `sm_80`, 108 SMs) and October 14, 2026 maintainer/author consensus not to alter the static cutoff table.
5. **llama.cpp Issue #27623 — Qwen 3.8 27B Long-Context Decode & Quantized KV Analysis (August–September 2026):** [ggml-org/llama.cpp#27623](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F27623) — Retraction of the reported >80K positional cliff (caused by prefill-inclusive timing), verified smooth 0→98K→262K decode curves on `sm_86`/`sm_120`, and `q8_0` vs `f16` CUDA FlashAttention conversion observations.
6. **llama.cpp Issue #29371 — CUDA FlashAttention Quantized KV Scratch Conversion (September 2026):** [ggml-org/llama.cpp#29371](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F29371) — Technical description of stock CUDA MMA FlashAttention requiring `f16` KV tiles and converting quantized KV via a scratch buffer.
7. **llama.cpp Issues #19662, #28790, and #30224 — Blackwell `120a-real` CMake Architecture Handling (February–October 2026):** [ggml-org/llama.cpp#19662](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F19662), [ggml-org/llama.cpp#28790](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F28790), [ggml-org/llama.cpp#30224](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F30224), and [ggml-org/llama.cpp#17822](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F17822) — Verification of `ggml/src/ggml-cuda/CMakeLists.txt` rewriting `120` to `120a-real` and `ptxas` failure modes.
8. **llama.cpp HTTP Server Documentation & Slot Context Implementation (`efe3626`):** [tools/server/README.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fblob%2Fmaster%2Ftools%2Fserver%2FREADME.md), [tools/server/server-context.cpp](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fblob%2Fmaster%2Ftools%2Fserver%2Fserver-context.cpp), [ggml-org/llama.cpp#29372](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F29372), [ggml-org/llama.cpp#30204](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F30204), [ggml-org/llama.cpp#26669](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F26669), [ggml-org/llama.cpp#23268](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F23268), and [ggml-org/llama.cpp#29527](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F29527).
9. **NVIDIA A100 & RTX PRO 6000 Blackwell Datasheets and Colab G4 Verification:** [NVIDIA A100 Tensor Core GPU Datasheet PDF](https://www.google.com/search?q=https%3A%2F%2Fwww.nvidia.com%2Fcontent%2Fdam%2Fen-zz%2FSolutions%2FData-Center%2Fa100%2Fpdf%2Fnvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf), [NVIDIA RTX PRO 6000 Blackwell Server Edition Product Page](https://www.google.com/search?q=https%3A%2F%2Fwww.nvidia.com%2Fen-us%2Fdata-center%2Frtx-pro-6000-blackwell-server-edition%2F), [PNY/NVIDIA RTX PRO 6000 Blackwell Server Edition Datasheet PDF](https://www.google.com/search?q=https%3A%2F%2Fwww.pny.com%2Fen-eu%2Ffile%2520library%2Fprofessional%2Fdatasheet%2Fdata%2520center%2520cards%2Fpny-nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf), [Lenovo ThinkSystem RTX PRO 6000 Blackwell Server Edition Guide](https://www.google.com/search?q=https%3A%2F%2Flenovopress.lenovo.com%2Flp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu), [VRLA Tech Workstation vs. Server Edition Comparison](https://www.google.com/search?q=https%3A%2F%2Fvrlatech.com%2Frtx-pro-6000-blackwell-workstation-vs-server-edition%2F), and [Colab G4 nvidia-smi Confirmation](https://www.google.com/search?q=https%3A%2F%2Fwww.reddit.com%2Fr%2FGoogleColab%2Fcomments%2F1r6pt7n%2Fcolab_g4_new_card_nvidia_rtx_pro_6000_blackwell%2F).
10. **Google Colab CLI v0.7.4 Documentation & Issues:** [docs/01\_session\_management.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2Fdocs%2F01_session_management.md), [docs/02\_execution\_and\_interactive.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2Fdocs%2F02_execution_and_interactive.md), [skills/colab-operator/SKILL.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2Fskills%2Fcolab-operator%2FSKILL.md), [CHANGELOG.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2FCHANGELOG.md), [googlecolab/google-colab-cli#14](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fissues%2F14), [googlecolab/google-colab-cli#106](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fissues%2F106), [googlecolab/google-colab-cli#160](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fissues%2F160), [googlecolab/colabtools#5950](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fcolabtools%2Fissues%2F5950), and [googlecolab/colabtools#6101](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fcolabtools%2Fissues%2F6101).
11. **2026 Google Colab Compute Unit (CU/hr) Benchmarks:** [Qiita Colab GPU Pricing \& Benchmark Study (2026-02-27)](https://www.google.com/search?q=https%3A%2F%2Fqiita.com%2Fjin0g%2Fitems%2F8fbfb209ac4e9343c13f) and [note.com Google AI Pro A100 Deployment Test (2026-10-11)](https://www.google.com/search?q=https%3A%2F%2Fnote.com%2Funco3%2Fn%2Fn47710f23e25a).

## Executive Findings

* **A100 40GB / 80GB DFlash 2 + `ngram-map-k4v` Published Tables Remain `UNKNOWN`:** No public benchmark table exists for `Qwen3.8-27B` (`Q4_K_M`, stock or Blackfrost) with `draft-dflash` + `ngram-map-k4v` (`n_max=5`) on `A100-SXM4-40GB` or `A100-SXM4-80GB` at short context, 32K, or 100K ([lukaLLM](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP); [llamabench.ai](https://www.google.com/search?q=https%3A%2F%2Fllamabench.ai%2F%3Fhardware%3Da100sxm40%26mode%3Dtune%26speed%3D0%26sort%3Ddecode)). All published `DFlash 2` numbers for `Qwen3.8-27B` were captured on RTX PRO 6000 Blackwell Workstation Edition (`1,792 GB/s`), RTX PRO 6000 Blackwell Server Edition (`1,597 GB/s`), or L4 (`300 GB/s`). Replace scaled estimates with the 3-minute first-boot smoke recipe in Section 2.
* **Colab G4 Uses the 1,597 GB/s Server Edition SKU, Making A100-SXM4-40GB 97.4% of G4 Bandwidth at 61.7% of the CU Cost:** Official NVIDIA, PNY, and Lenovo datasheets confirm that the passively cooled **NVIDIA RTX PRO 6000 Blackwell Server Edition** deployed in Colab `G4` has **1,597 GB/s** peak memory bandwidth ([NVIDIA Server Edition Page](https://www.google.com/search?q=https%3A%2F%2Fwww.nvidia.com%2Fen-us%2Fdata-center%2Frtx-pro-6000-blackwell-server-edition%2F); [PNY Datasheet PDF](https://www.google.com/search?q=https%3A%2F%2Fwww.pny.com%2Fen-eu%2Ffile%2520library%2Fprofessional%2Fdatasheet%2Fdata%2520center%2520cards%2Fpny-nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf)), whereas **1,792 GB/s** belongs to the active-cooled Workstation Edition used by `lukaLLM` ([VRLA Tech](https://www.google.com/search?q=https%3A%2F%2Fvrlatech.com%2Frtx-pro-6000-blackwell-workstation-vs-server-edition%2F)). Against `A100-SXM4-40GB` (**1,555 GB/s** HBM2 at **5.37 CU/hr** vs G4 at **8.71 CU/hr**, [Qiita](https://www.google.com/search?q=https%3A%2F%2Fqiita.com%2Fjin0g%2Fitems%2F8fbfb209ac4e9343c13f)), A100-40GB delivers **97.4%** of G4's raw memory bandwidth for **38.3% fewer CU/hr**.
* **Keep `-ctk q8_0 -ctv q8_0` on A100 40GB:** Although stock CUDA MMA FlashAttention converts `q8_0` KV tiles to `f16` via a scratch buffer ([llama.cpp#29371](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F29371); [llama.cpp#27623](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F27623)), no cited A100 measurement shows `f16` KV overcoming the **2.86 GiB** extra VRAM tax at 100K context (6.10 GiB `f16` vs 3.24 GiB `q8_0` across Qwen 3.8's 16 full-attention layers), which is needed as compute-buffer headroom across varied `ubatch` shapes ([hanxiao](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)).
* **Ampere MMVQ Verdict Is `KEEP_DELETED`:** No `GGML_CUDA_CC_AMPERE` or `sm_80` cutoff table has landed on `master` at or after `efe3626` (`b11552`). Direct measurements on `A100-SXM4-40GB` (`sm_80`, 108 SMs) in [llama.cpp#28090](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F28090) prove that forcing MMQ at `ne11 = 6` (`n_max = 5`) is **20% slower** (`0.80×` `measured`) than stock MMVQ on 8B `Q4_0` and never wins at any `ne11 <= 8`. Furthermore, Blackfrost `Q4_K_M` uses a pure K-quant ladder with **zero `IQ4_XS` tensors** ([Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)). Keep `GGML_MMVQ_MAX` and `LLAMA_SCHED_POOL` deleted.
* **Use `-DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real"` for the Dual-Arch Drive Binary:** On CUDA 13.0 / driver 580, `80-real;120a-real` embeds native SASS cubins for both A100 (`sm_80`) and G4 (`sm_120a` block-scaled MMA) with zero PTX bloat and no reliance on CMake regex rewrites ([llama.cpp#19662](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F19662); [llama.cpp#28790](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F28790)).
* **Session Longevity, Re-Attach, & `/slots` Watchdog:** An attached `colab exec --timeout 86400` emitting a 20–25s `flush=True` stdout heartbeat prevents both the 80s TFE tunnel freeze ([google-colab-cli#14](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fissues%2F14)) and Colab's idle-kernel reaper ([colabtools#5950](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fcolabtools%2Fissues%2F5950)). If the WebSocket drops, running a fresh `colab exec -s qwen-worker --timeout 86400 -f attach_heartbeat.py` refreshes `runtimeProxyInfo.token` and re-attaches to the persistent kernel and running `llama-server` / `cloudflared` PIDs without reloading weights ([01\_session\_management.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2Fdocs%2F01_session_management.md); [colab-operator SKILL.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2Fskills%2Fcolab-operator%2FSKILL.md)). Poll `GET /slots` and reset the 30-minute idle timer whenever top-level `id_task` changes OR (`is_processing == true` and `next_token[0].n_decoded` or `n_prompt_tokens_processed` advances) ([server-context.cpp](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fblob%2Fmaster%2Ftools%2Fserver%2Fserver-context.cpp)).

## 1. Measured A100 40GB (and 80GB) `llama.cpp` DFlash 2 + `ngram-map-k4v`

### 1.1 Published A100-SXM4-40GB / 80GB Tables for Qwen 3.8 27B `Q4_K_M` + `draft-dflash,ngram-map-k4v`

* **A100-SXM4-40GB and A100-SXM4-80GB Status:** **`UNKNOWN`** (`unknown` at short context, ~32K, and 100K).
* A systematic audit of [lukaLLM/DFlash\_Qwen3.6\_27B\_LlamaCPP](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP), [hanxiao/Qwen3.8-27B-UD-Q4\_K\_XL-L4](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4), [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF), `llama.cpp` issues/discussions, and crowd-sourced [llamabench.ai A100-SXM4-40GB records](https://www.google.com/search?q=https%3A%2F%2Fllamabench.ai%2F%3Fhardware%3Da100sxm40%26mode%3Dtune%26speed%3D0%26sort%3Ddecode) (which contain only two non-speculative runs: `Ornith-1.5-35B-A3B` at `148 tok/s` `measured` and `Ornith-1.5-9B` at `129 tok/s` `measured`) confirms that **no published table exists for `Qwen3.8-27B` `Q4_K_M` with `draft-dflash` and `ngram-map-k4v` on either A100-40GB or A100-80GB**.
* For reference only (never to be substituted for A100 measurements), the table below compiles the published `Qwen3.8-27B` `DFlash 2` measurements on adjacent GPUs:

| Source & Hardware (Peak BW) | Target + Drafter Quants | Speculative Flags & Context | Workload / Prompt Depth | Decode Speed (`tok/s`) | Evidence Label & Acceptance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A100-SXM4-40GB** (`1,555 GB/s`) or **80GB** (`2,039 GB/s`) | `Qwen3.8-27B Q4_K_M` + `DFlash 2 Q4_K_M` | `draft-dflash,ngram-map-k4v`, `n_max=5`, `-c 100000`, `q8_0` KV | Short, ~32K, and 100K | **`UNKNOWN`** | `unknown` — Must be captured via Section 2 first-boot smoke |
| [lukaLLM](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP) · RTX PRO 6000 Blackwell **Workstation** (`1,792 GB/s`) | `ggml-org Q4_K_M` (18 GB) + `incoai DFlash2 Q4_K_M` (1.1 GB) | `draft-dflash`, `n_max=5`, `-c 262144`, `f16` KV (`b10498`) | 8,192 prompt / 256 out (`speed_bench_v2.py`, cold prefill) | **187.13 tok/s** (vs **168.06 tok/s** at `n_max=7`) | `measured` (on Workstation Edition, 70.4% accept at `n_max=5` vs 59.7% at `n_max=7`) |
| [lukaLLM](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP) · RTX PRO 6000 Blackwell **Workstation** (`1,792 GB/s`) | `ggml-org Q4_K_M` + `incoai DFlash2 Q4_K_M` | `draft-dflash`, `n_max=7`, `-c 262144`, `f16` KV | LiveCodeBench × 100 single-turn coding (natural EOS) | **153.91 tok/s** (`+both lookups`: **155.83 tok/s**) | `measured` (on Workstation Edition, 2.26× vs **67.97 tok/s** no-spec base; 60% probe accept) |
| [lukaLLM](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP) · RTX PRO 6000 Blackwell **Workstation** (`1,792 GB/s`) | `ggml-org Q4_K_M` + `incoai DFlash2 Q4_K_M` | `draft-dflash,ngram-map-k4v`, `n_max=7`, `-c 262144`, `f16` KV | 18-turn cumulative coding (`bench_ngram.py`): Turns 1–9 build / All 18 turns | **304.92 tok/s** (Turns 1–9) / **343.52 tok/s** (All 18) | `measured` (on Workstation Edition, 4.68× / 6.03× vs **65.14** / **56.95 tok/s** base; +68% over DFlash2-only **181.89 tok/s**) |
| [Blackfrost-AI](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF) · RTX PRO 6000 Blackwell **Server Edition** (`1,597 GB/s`) | `Blackfrost Q4_K_M` (16.8 GB) + `DFlash2 BF16` (3.86 GB) | `draft-dflash` (no k4v), `n_max=3`, `-c 262144`, `q8_0` target KV, `f16` draft KV (`1deefcca`) | 3 short text prompts, 256 tokens out | **109.31 tok/s** median (`104.47` / `109.31` / `118.60`) | `measured` (on Server Edition / G4 SKU, 55.8% accept `478/856`; uses 3.86 GB BF16 drafter & `n_max=3`) |
| [hanxiao](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4) · NVIDIA L4 24GB (`300.05 GB/s`, 72 W) | `Unsloth UD-Q4_K_XL` (16.35 GiB) + `DFlash2 Q4_K_M` (1.14 GiB) | `draft-dflash`, `n_max=7`, `GGML_MMVQ_MAX=2`, `-c 226048`, `q4_0` KV | 7-workload mean (256 out) / 32K window / GSM8K-shaped | **37.4 tok/s** (226K) / **38.3 tok/s** (32K) / **55.34 tok/s** (GSM8K) | `measured` (on L4; mean accepted length 3.63 on 7-workload bench, 5.573 on GSM8K) |

### 1.2 No-Spec Baseline `tok/s` on A100-SXM4-40GB (~15.75 GiB Active Weight Read)

* **Direct A100-SXM4-40GB Measurement for Qwen 3.8 27B `Q4_K_M`:** **`UNKNOWN`** (`unknown` — no published `llama-bench` or `llama-server --spec-type none` run on A100-SXM4-40GB for `Qwen3.8-27B` `Q4_K_M`).
* **Active Weight-Read Verification & Cross-Card Measured Anchors:**
  * In [hanxiao/Qwen3.8-27B-UD-Q4\_K\_XL-L4 (Table 1)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4), the irreducible trunk weight read per single-token forward pass (total GGUF minus the 682 MiB `token_embd` row lookup and the 272 MiB `blk.64` MTP block, which is bypassed when `--spec-type draft-mtp` is not used) is measured at **15.75 GiB** (`16.912 GB`) on the initial `UD-Q4_K_XL` build and **15.35 GiB** (`16.482 GB`) on the 2026-08-19 rebuild. For `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (`16.8 GB` decimal on disk, [Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)), subtracting the embedded 65th MTP block and `token_embd` leaves an active trunk weight read of **~15.4–15.75 GiB** (`~16.5–16.9 GB`).
  * **Measured no-spec baselines on other cards for the same model:**
    * **NVIDIA L4 (`300.05 GB/s` peak):** **14.95 tok/s** `measured` (`66.9 ms/step`, achieving **252.8 GB/s** = **84.3%** of peak datasheet bandwidth, [hanxiao Table 1](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)).
    * **RTX PRO 6000 Blackwell Workstation Edition (`1,792 GB/s` peak):** **71.0 tok/s** `measured` (`llama-bench tg128`), **67.97 tok/s** `measured` (`llama-server` LCB N=100), and **67.57 tok/s** `measured` (`isl512_osl512` sweep) ([lukaLLM](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP)), representing **63.8%–67.0%** of 1,792 GB/s over 15.75 GiB.
    * **RTX 5090 (`1,792 GB/s` peak):** **76.48 tok/s** `measured` (`llama-bench tg32` at d=0 on `Qwen3.8-27B-UD-Q4_K_M.gguf`, [llama.cpp#27623 comment by @siavashnorouzi](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F27623)), representing **72.2%** of 1,792 GB/s over 15.75 GiB.
    * **2× RTX 3090 layer-split (`936 GB/s` per card, 250 W limit):** **40.92 ± 0.41 tok/s** `measured` (`llama-bench tg32` at d=0 on `Qwen3.8-27B-UD-Q4_K_XL.gguf`, [llama.cpp#27623 comment by @geoffreybyers](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F27623)).
  * **Correction to the Roofline Arithmetic on `A100-SXM4-40GB` (`1,555 GB/s` peak per [NVIDIA A100 Datasheet](https://www.google.com/search?q=https%3A%2F%2Fwww.nvidia.com%2Fcontent%2Fdam%2Fen-zz%2FSolutions%2FData-Center%2Fa100%2Fpdf%2Fnvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf)):**
    * Be careful with GiB ($2^{30}$ bytes) vs. GB ($10^9$ bytes): `15.75 GiB = 16.911 GB`.
    * At **80%–84%** of `1,555 GB/s` (`1,244–1,306 GB/s` effective), dividing by `16.911 GB` yields **73.6–77.2 tok/s** `scaled` (or **69.0–72.4 tok/s** `scaled` if dividing by the full 16.8 GiB file size before subtracting `token_embd` and `blk.64`).
    * The prompt's expected **~62–68 tok/s** `scaled` corresponds to **67.4%–74.0%** effective bandwidth utilization (`1,048–1,150 GB/s` over `16.911 GB`), which closely matches the **64%–72%** effective utilization measured on Blackwell GPUs (`67.97–76.48 tok/s`) once per-layer kernel launch overhead, Gated DeltaNet recurrence kernels, and `llama-server` HTTP/sampling overhead are included.

### 1.3 64K–100K Decode Curve Validity and Published Deep-Context Data

* **Why `lukaLLM`'s 65K and 131K Decode Cells Are Invalid:** Confirmed directly in [lukaLLM/DFlash\_Qwen3.6\_27B\_LlamaCPP (Long context section)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP): `scripts/run_context_scaling.sh` truncated concatenated LiveCodeBench problems to exact token boundaries (65,536 and 131,072), where the model interpreted the prompt as finished and emitted EOS after 1 token on **12 of 12** requests (`"At 65 536 and 131 072, zero of 12 generations produced a usable decode number... Prefill at those depths is valid; decode does not exist"`). Similarly, `speed_sweep.sh` at 36,864 (`ISL=OSL=36864`, **210.66 tok/s** `measured` DFlash2-only, **498.00 tok/s** `measured` DFlash2+ngram) used `ignore_eos: true` and `min_tokens: 36864`, which `lukaLLM` proved forces self-repetitive 2-line output loops (`98.3%` duplicate 12-grams) and cannot be used as a real decode curve.
* **What `hanxiao` Measured on L4 (22K → 174K Multi-Turn Conversation with Prefix Reuse):** Confirmed in [hanxiao/Qwen3.8-27B-UD-Q4\_K\_XL-L4 (Long-horizon behaviour and the context ceiling)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4) for `DFlash 2` (`n_max=7`, `q4_0` KV):
  * Turn 1 (`21,963` ctx): **34.12 tok/s** `measured` (accepted length `3.76`, VRAM `23,786 MiB`)
  * Turn 2 (`43,696` ctx): **33.13 tok/s** `measured` (accepted length `4.09`, `0.971×` retention)
  * Turn 3 (`65,429` ctx): **27.93 tok/s** `measured` (accepted length `3.76`, `0.819×` retention)
  * Turn 4 (`87,162` ctx): **29.76 tok/s** `measured` (accepted length `4.36`, `0.872×` retention)
  * Turn 5 (`108,895` ctx): **29.10 tok/s** `measured` (accepted length `4.57`, **`0.853×` retention vs 22K**)
  * Turn 6 (`130,628` ctx): **26.35 tok/s** `measured` (accepted length `4.47`, `0.772×` retention)
  * Turn 7 (`152,361` ctx): **22.17 tok/s** `measured` (accepted length `4.00`, `0.650×` retention)
  * Turn 8 (`174,094` ctx): **20.29 tok/s** `measured` (accepted length `3.92`, `0.595×` retention)
* **Are There Any Other Valid ~100K Decode Numbers on A100 or G4?**
  * **On A100 (40GB or 80GB) or Colab G4 (Server Edition) at ~100K resident KV:** **`UNKNOWN`** (`unknown`). Note that while [Blackfrost-AI](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF) allocated `-c 262144` (`q8_0` KV) on an RTX PRO 6000 Blackwell Server Edition and measured **109.31 tok/s** `measured`, those three prompts were short 256-token generations inside an empty 262K allocation (where attention only scans the resident ~300 tokens, not 100K tokens). And `lukaLLM`'s **241.44 tok/s** `measured` at `98,304` context on RTX PRO 6000 Workstation Edition was from the July 2026 study on `Qwen3.6-27B + DFlash 1` with reasoning enabled, not `Qwen3.8-27B + DFlash 2`.
  * **Valid ~65K–100K Non-Speculative (`--spec-type none`) Decode Curves on Other GPUs (from [llama.cpp#27623](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F27623)):** After `@cuiyuxin-gif` retracted the initial 91K cliff report (which had accidentally divided `completion_tokens` by `total_request_time` including 43 seconds of prompt processing), four independent server/bench runs in `#27623` proved that `Qwen3.8-27B` decode decays smoothly with context depth:

| Hardware & Backend (`llama.cpp#27623`) | Quant & KV Cache | Short / `d=0` (`tok/s`) | ~32K (`d=32768`) (`tok/s`) | ~65K–68K (`tok/s`) | ~81K (`d=81920`) (`tok/s`) | ~90K–98K (`tok/s`) | ~0→98K Retention |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RTX 5090** (`sm_120`, CUDA 13.3, `b10615`) | `UD-Q4_K_M`, `-fa on` | **76.48** `measured` | **70.05** `measured` | **64.26** `measured` | **61.69** `measured` | **59.34** `measured` (`tg32 @ d98304`) / **60.03** `measured` (`llama-server` @ 90K fill) | **0.776×–0.785×** |
| **2× RTX 3090** (`sm_86`, CUDA 13.0, `b10603`) | `UD-Q4_K_XL`, `-fa 1` | **40.92** `measured` | **37.40** `measured` | **33.97** `measured` | **32.49** `measured` | **31.13** `measured` (`tg32 @ d98304`; `28.68` @ 131K) | **0.761×** |
| **RTX 4080 SUPER** (`sm_89`, CUDA 13.1, `d222767c`) | `UD-Q3_K_XL`, `q4_0` KV, `-fa on` (`llama-server`) | **~38.0** `measured` | — | **33.1–36.1** `measured` | — | **32.4** `measured` (`llama-server` @ 91K fill) | **~0.853×** |
| **RX 7900 XTX** (RADV Vulkan, `b11065`) | `IQ3_XXS`, `q4_0` KV (`llama-server`) | **52.83** `measured` | — | **41.98** `measured` (68.4K) | **40.65** `measured` (80.6K) | **39.51** `measured` (91.8K) | **0.748×** |

### 1.4 `q8_0` vs `f16` KV Cache Effect on DFlash 2 Verify Speed at 100K on A100

* **Mechanism Documented in [llama.cpp#27623](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F27623) and [llama.cpp#29371](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F29371):**
  * During single-token non-speculative decode (`ne11 = 1`), CUDA FlashAttention can use the vector kernel (`fattn-vec`), which reads quantized KV natively. However, when batch width exceeds the vector kernel threshold (such as DFlash 2 verification at `n_max=5`, where verification width `ne11 = 6`) or when dispatching to the **CUDA MMA FlashAttention kernel (`fattn-mma-f16`)**, the stock CUDA MMA path supports `f16` KV tiles only.
  * Consequently, any quantized KV cache (`q8_0` or `q4_0`) flowing through the CUDA MMA FA path is first dequantized/converted to `f16` into a scratch buffer ([llama.cpp#29371](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F29371)). On 16 GB / 24 GB Ada cards (`sm_89`), `@jimdawdy-hub` (RTX 2000 Ada 16GB) and `@apresence` (RTX 4090) reported in [llama.cpp#27623](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F27623) that this `q8_0 → f16` FA conversion reduced short-context decode on `UD-IQ4_XS` from **13.3 tok/s** `measured` (`f16` KV) to **5.8–5.9 tok/s** `measured` (`q8_0` KV).
* **A100-SXM4-40GB 100K DFlash 2 Measurement Status:** **`UNKNOWN`** (`unknown` — no published A/B on A100-SXM4-40GB comparing `q8_0` vs `f16` KV under DFlash 2 at 100K).
* **Why `q8_0` KV Must Be Kept on A100 40GB:**
  1. **Hybrid Architecture Bounds the KV Size:** `Qwen3.8-27B` has 64 layers with `full_attention_interval = 4` (only **16 full-attention layers** hold a KV cache; the other 48 layers are Gated DeltaNet with a fixed-size `f32` recurrent state that does not scale with context length, [hanxiao Architecture](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)). Across 16 layers (4 KV heads, `head_dim = 256`), `f16` KV costs exactly **64 KiB/token** (**6.10 GiB** at `-c 100000`), whereas `q8_0` KV (`34 bytes / 32 elements`) costs **34 KiB/token** (**3.24 GiB** at `-c 100000`) — a **2.86 GiB** (~2.85 GiB) VRAM saving.
  2. **VRAM Margin Prevents Mid-Session Graph-Allocation Aborts:** As proven in [hanxiao (Memory and context \& Sweeping the ceiling)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4), `llama.cpp` instantiates a new CUDA graph and allocates `ggml_gallocr` compute buffers lazily inside `llama_context::decode` whenever a prompt hits a new `ubatch` remainder shape. Surrendering **2.86 GiB** of VRAM headroom on a 40 GiB A100 (where target + drafter + 100K `q8_0` KV + recurrent state checkpoints + scratch buffers already consume ~23.8–27.6 GiB, plus the `fattn` MMA scratch buffer itself) narrows the safety margin against fragmentation and varied batch shapes.
  3. **HBM2 Bandwidth Amortizes Conversion Across Draft Tokens:** Even if `fattn-mma` converts resident `q8_0` KV tiles to `f16` on `A100-SXM4-40GB`, `1,555 GB/s` HBM2 streams 3.24 GiB in ~2.2 ms at 100K full occupancy, and DFlash 2 amortizes each target verification pass across ~3.6–4.6 accepted tokens.
  * **Verdict:** **Keep `-ctk q8_0 -ctv q8_0` on A100 40GB.** No cited measurement shows `f16` KV is fast enough on A100 to justify losing 2.86 GiB of VRAM headroom.

### 1.5 NVIDIA RTX PRO 6000 Blackwell Server Edition Peak Memory Bandwidth (1,792 GB/s vs 1,597 GB/s)

* **Datasheet Verdict for Colab G4:** **1,597 GB/s** is the official datasheet peak memory bandwidth for the **NVIDIA RTX PRO 6000 Blackwell Server Edition** deployed in Google Colab's `G4` instances ([NVIDIA RTX PRO 6000 Blackwell Server Edition Page](https://www.google.com/search?q=https%3A%2F%2Fwww.nvidia.com%2Fen-us%2Fdata-center%2Frtx-pro-6000-blackwell-server-edition%2F); [PNY/NVIDIA Server Edition Datasheet PDF](https://www.google.com/search?q=https%3A%2F%2Fwww.pny.com%2Fen-eu%2Ffile%2520library%2Fprofessional%2Fdatasheet%2Fdata%2520center%2520cards%2Fpny-nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf); [Lenovo ThinkSystem Product Guide](https://www.google.com/search?q=https%3A%2F%2Flenovopress.lenovo.com%2Flp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu); [Colab G4 nvidia-smi Output](https://www.google.com/search?q=https%3A%2F%2Fwww.reddit.com%2Fr%2FGoogleColab%2Fcomments%2F1r6pt7n%2Fcolab_g4_new_card_nvidia_rtx_pro_6000_blackwell%2F)).
* **Why Both Numbers Appear in the Literature:** NVIDIA ships three distinct SKUs under the RTX PRO 6000 Blackwell name ([VRLA Tech Comparison](https://www.google.com/search?q=https%3A%2F%2Fvrlatech.com%2Frtx-pro-6000-blackwell-workstation-vs-server-edition%2F)):
  * **RTX PRO 6000 Blackwell Workstation Edition (600 W, active cooling):** **1,792 GB/s** (28 Gbps GDDR7 over 512-bit bus). This is the local desktop/workstation card used in [lukaLLM/DFlash\_Qwen3.6\_27B\_LlamaCPP](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP).
  * **RTX PRO 6000 Blackwell Max-Q Workstation Edition (300 W, blower):** **1,792 GB/s**.
  * **RTX PRO 6000 Blackwell Server Edition (400–600 W configurable, passive data-center cooling):** **1,597 GB/s** (25 Gbps GDDR7 over 512-bit bus). This is the card inside **Google Colab G4** and the card used in the [Blackfrost Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF) benchmark.
* **Corrected CU-per-Token Math (`A100-SXM4-40GB` vs Colab `G4`):**
  * **Bandwidth ratio:** $rac{1,555	ext{ GB/s (A100-SXM4-40GB)}}{1,597	ext{ GB/s (Colab G4 Server Edition)}} = \mathbf{0.9737}$ (**97.4%**, rather than the $0.8677$ ratio obtained if erroneously dividing by the Workstation Edition's `1,792 GB/s`).
  * **Burn-rate ratio:** $rac{5.37	ext{ CU/hr (A100-40GB)}}{8.71	ext{ CU/hr (G4)}} = \mathbf{0.6165}$ (**38.3% cheaper per hour** on A100-40GB, [Qiita 2026 Colab Benchmark](https://www.google.com/search?q=https%3A%2F%2Fqiita.com%2Fjin0g%2Fitems%2F8fbfb209ac4e9343c13f)).
  * Consequently, on a pure bandwidth-per-CU basis, `A100-SXM4-40GB` delivers **1.58× more memory bandwidth per Compute Unit** than Colab `G4` ($rac{0.9737}{0.6165} = 1.579	imes$).

## 2. First-Boot Speed Gates on A100 40GB (≤3-Minute Smoke Recipe)

### 2.1 Critical `llama-server` Behavior: Why Per-Request `"speculative.n_max": 0` Cannot Measure No-Spec

* Before running first-boot curls, note a verified `llama-server` limitation documented in [llama.cpp#30204](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F30204), [llama.cpp#26669](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F26669), and [llama.cpp#27852](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F27852):
  * Passing `"speculative.n_max": 0` (or `"speculative": {"n_max": 0}`) inside a `POST /v1/chat/completions` JSON body is **silently ignored** by current `llama-server` builds (`timings.draft_n` remains `> 0` and drafting stays active).
  * Therefore, if you want a true **no-spec baseline (`--spec-type none`)** without restarting `llama-server`, run a **12-second `llama-bench` one-liner** (`-p 512 -n 128 -r 1`) right before launching `llama-server` (or inspect `timings.predicted_ms / (timings.draft_n / 5 + 1)` on the live server to see the per-verify-step latency).
  * Against the live `llama-server` started with Playbook A (`--spec-type draft-dflash,ngram-map-k4v --spec-draft-n-max 5`), **Curl 1** (cold short coding prompt with `"cache_prompt": false`) isolates **one-shot `DFlash 2` (`n_max=5`)** because `ngram-map-k4v` requires 4-gram key/4-token value matches in existing context and drafts `0` tokens on a short cold prompt ([hanxiao Configurations not used](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4); [lukaLLM One change, four different answers](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP)). **Curl 2** sends a multi-turn code-edit follow-up with `"cache_prompt": true` that seeded the slot with a function and asks for a modified re-emission, exercising both prefix caching (`timings.cache_n > 0`) and `ngram-map-k4v`.

### 2.2 Evaluation of Proposed Numeric Gates

| Gate Metric | Proposed Threshold | Audit Verdict & Source Basis | Next Action on Failure |
| :--- | :--- | :--- | :--- |
| **Post-`/health` VRAM used** (`nvidia-smi` at `-c 100000` `q8_0` KV) | **≤ 31,000 MiB** | **SOUND (Derived from Measured Components + Safety Margin):** `16.8 GB` (`~16,020 MiB`) target + `~1,170 MiB` (`Q4_K_M`) DFlash 2 drafter + `~3,320 MiB` 100K `q8_0` KV + Gated DeltaNet recurrent states & compute buffers = **~23.8–27.6 GiB** (`~24,400–28,300 MiB`). `≤ 31,000 MiB` leaves `≥ 9,500 MiB` free on a `40,536 MiB` A100, preventing the lazy `cudaGraphInstantiate` / `ggml_gallocr` OOMs documented by [hanxiao](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4). | Verify `-ctk q8_0 -ctv q8_0` (not `f16` KV), `--parallel 1`, and `--spec-draft-n-max 5` (since `n_max > 7` inflates `need_n_rs_seq()` recurrent state by ~299 MiB per step, [hanxiao](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)). |
| **No-spec short decode** (`llama-bench tg128` or `--spec-type none`) | **≥ 50.0 tok/s** | **SOUND SANITY FLOOR (Conservative vs ~62–75 tok/s `scaled`):** At `50.0 tok/s`, a `15.75 GiB` (`16.91 GB`) active read corresponds to `845 GB/s` (**54.4%** of A100-SXM4-40GB's `1,555 GB/s` HBM2 peak). Any healthy A100-SXM4-40GB should clear `60+ tok/s`; falling below `50 tok/s` indicates severe clock throttling, CPU fallback, or wrong GPU SKU. | Stop and inspect `nvidia-smi -q -d PERFORMANCE,POWER,CLOCK` (check for thermal/power throttle, PCIe vs SXM4, or unoffloaded layers). |
| **One-shot spec (`n_max=5`) short coding prompt** (`timings.predicted_per_second`) | **≥ 95.0 tok/s** | **PARTIALLY ARBITRARY (Prompt-Dependent, Sound for Code, Too High for Prose):** To beat G4 on CU/token ($rac{5.37	ext{ CU/hr}}{8.71	ext{ CU/hr}} = 0.6165$), A100 only needs **≥ 61.7% of G4's speed** on the *same* prompt. On LiveCodeBench coding (`153.91 tok/s` `measured` on 1,792 GB/s Workstation Edition, [lukaLLM](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP)), $0.6165 	imes 153.91 = \mathbf{94.9	ext{ tok/s}}$ `scaled` — which is where the **95 tok/s** gate comes from. However, on short non-code prose (`109.31 tok/s` `measured` on 1,597 GB/s G4 Server Edition at `n_max=3`, [Blackfrost](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF)), the G4 CU-parity point is only $0.6165 	imes 109.31 = \mathbf{67.4	ext{ tok/s}}$ `scaled`. **Rule:** Only apply the `≥ 95 tok/s` gate to a **deterministic coding prompt** (like Curl 1 below), never to open-ended prose. | **Do NOT set `GGML_MMVQ_MAX=2`** (Section 3 proves MMQ is 20% slower on A100 at `ne11=6`). Instead, inspect `timings.draft_n_accepted / timings.draft_n` (if acceptance `< 45%`, the prompt was prose-heavy; if acceptance `≥ 55%` and speed `< 95 tok/s`, inspect SM clocks, thermal state, and SXM4 vs PCIe). |
| **Follow-up turn hitting `ngram-map-k4v`** (`timings.predicted_per_second`) | **> 1.25× Curl 1 tok/s** and **`cache_n > 0`** | **EMPIRICALLY GROUNDED:** On multi-turn code editing where prior code is in context, [lukaLLM](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2FlukaLLM%2FDFlash_Qwen3.6_27B_LlamaCPP) measured `+68%` (`304.92` vs `181.89 tok/s` `measured`) when adding `ngram-map-k4v` to `DFlash 2`. | Verify `--spec-type draft-dflash,ngram-map-k4v` is active and that the follow-up turn re-emits or edits spans ≥ 12 tokens (`--spec-ngram-map-k4v-size-n` defaults to `12`, [tools/server/README.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fblob%2Fmaster%2Ftools%2Fserver%2FREADME.md)). |

### 2.3 Exact JSON Fields to Record from `POST /v1/chat/completions`

Per [tools/server/README.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fblob%2Fmaster%2Ftools%2Fserver%2FREADME.md) and [llama.cpp#29372](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F29372), `llama-server` returns a top-level `timings` object in every non-streaming completion response (and in the final chunk of streaming responses when `timings_per_token: true` or `stream_options.include_usage: true` is set):

* `timings.predicted_per_second` *(float)*: **Primary decode throughput (`tok/s`)**, excluding prompt prefill time (`1000.0 * predicted_n / predicted_ms`).
* `timings.predicted_n` *(int)* and `timings.predicted_ms` *(float)*: Total generated tokens and pure decode wall time in milliseconds.
* `timings.prompt_per_second` *(float)*, `timings.prompt_n` *(int)*, and `timings.prompt_ms` *(float)*: Uncached prompt evaluation speed (`tok/s`), tokens evaluated, and prefill duration in milliseconds.
* `timings.cache_n` *(int)*: Prefix tokens reused from the slot KV cache without re-prefilling.
* `timings.draft_n` *(int)* and `timings.draft_n_accepted` *(int)*: Total tokens drafted and accepted across the completion (`acceptance_rate = 100.0 * draft_n_accepted / draft_n`).

*(See the standalone copy-paste block after Section 5 for the exact commands.)*

## 3. Ampere MMVQ Follow-Up ([llama.cpp#28090](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F28090)) Only

### 3.1 Has a `GGML_CUDA_CC_AMPERE` / `sm_80` Table Landed on `master` After `efe3626`?

* **No.** As of `master` commit `efe3626` (`b11552`, PR `#30383`, merged October 13, 2026, [tools/server/README.md commit history](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fblob%2Fmaster%2Ftools%2Fserver%2FREADME.md)), **no PR has been opened or merged** from [llama.cpp#28090](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F28090) (`Development: No branches or pull requests`), and `ggml_cuda_should_use_mmvq()` in `ggml/src/ggml-cuda/mmvq.cu` still has **no `sm_80`, `sm_86`, or `GGML_CUDA_CC_AMPERE` branch**. All Ampere GPUs continue to fall through to `return ne11 <= MMVQ_MAX_BATCH_SIZE;` (`8`).
* Moreover, on **October 14, 2026**, `@manunicholasjacob` and `@zhihz` completed clock-pinned (`810 MHz`) vs unpinned (`2520 MHz`) sweeps on L4 (`sm_89`, 58 SMs) and L20 (`sm_89`, 92 SMs) and explicitly concluded in [llama.cpp#28090](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F28090) that the `ne11 = 8` crossover on smaller cards is an artifact of power-cap SM clock throttling rather than compute capability or SM count:
  > *"Based on this, I don't think a static cutoff keyed to compute capability or SM count is supported by the current data. My preference would be to leave the table unchanged and document the finding."*

### 3.2 Blackfrost `Q4_K_M` Tensor Mix Check (`IQ4_XS` Presence) and `ne11 = 6` (`n_max = 5`) on A100-SXM4-40GB

1. **Blackfrost `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` Contains Zero `IQ4_XS` Tensors:**
   * Unlike Unsloth's `UD-Q4_K_XL` (which is an importance-matrix dynamic quant containing 70 `IQ4_XS` tensors and 191 `Q5_K` tensors, [hanxiao The 2026-08-19 rebuild](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4)), the [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF Model Card](https://www.google.com/search?q=https%3A%2F%2Fhuggingface.co%2FBlackfrost-AI%2FQwen3.8-27B-ABLITERATED-GGUF) explicitly states:
     > *"The complete standard K-quant ladder (`Q2_K` through `Q8_0`) and both vision projectors are included. **No IQ/IK or importance-matrix quants are used.**"*
   * Therefore, `Blackfrost-AI/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (`16.8 GB`) uses standard `Q4_K_M` tensor types (`Q4_K` and `Q6_K`, plus `Q8_0`/float metadata tensors) and has **0% `IQ4_XS`**.
2. **Stock MMVQ vs Forced MMQ at `ne11 = 6` (`n_max = 5`) on `A100-SXM4-40GB` (`sm_80`):**
   * In [llama.cpp#28090](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F28090), `@manunicholasjacob` ran a 6-round rotated-binary paired benchmark directly on **NVIDIA A100-SXM4-40GB (`sm_80`, 108 SMs, ~1400–1410 MHz SM clock)** comparing `force_mmq` (`ne11 <= 1`, which routes `ne11 = 2..8` to MMQ just like `hanxiao`'s `GGML_MMVQ_MAX=2`) against `selected` (stock MMVQ for `ne11 <= 8`):

| GPU (`llama.cpp#28090`) | Model & Pure Quant | `force_mmq / selected` at `ne11 = 5` | **`force_mmq / selected` at `ne11 = 6` (`n_max = 5`)** | `force_mmq / selected` at `ne11 = 7` | `force_mmq / selected` at `ne11 = 8` | First `ne11` Where MMQ Wins | Noise Floor (`null_rebuild`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A100-SXM4-40GB** (`sm_80`, 108 SMs) | Llama-3.1-8B `Q4_0` | `0.70×` (`-30%`) | **`0.80×` (`-20%` `measured`)** | `0.85×` (`-15%`) | `0.92×` (`-8%`) | **`never`** (through 8) | `0.38%` (`1404 MHz`) |
| **A100-SXM4-40GB** (`sm_80`, 108 SMs) | Llama-3.1-8B `Q8_0` | `0.85×` (`-15%`) | **`0.94×` (`-6%` `measured`)** | `0.96×` (`-4%`) | `0.99×` (`-1%`) | **`never`** (through 8) | `0.68%` (`1400 MHz`) |
| **A100-SXM4-40GB** (`sm_80`, 108 SMs) | Llama-3.2-3B `Q4_0` | `0.72×` (`-28%`) | **`0.78×` (`-22%` `measured`)** | `0.82×` (`-18%`) | `0.87×` (`-13%`) | **`never`** (through 8) | `0.12%` (`1410 MHz`) |
| **A100-SXM4-40GB** (`sm_80`, 108 SMs) | Llama-3.2-3B `Q8_0` | `0.84×` (`-16%`) | **`0.90×` (`-10%` `measured`)** | `0.92×` (`-8%`) | `0.95×` (`-5%`) | **`never`** (through 8) | `0.32%` (`1410 MHz`) |
| **A100-SXM4-40GB** (`sm_80`, 108 SMs) | Llama-3.2-1B `Q4_0` | `0.69×` (`-31%`) | **`0.76×` (`-24%` `measured`)** | `0.79×` (`-21%`) | `0.86×` (`-14%`) | **`never`** (through 8) | `0.80%` (`1410 MHz`) |

* **Why Stock MMVQ Wins on A100-SXM4-40GB while Losing on L4:** On L4 (`sm_89`, 58 SMs, 72 W hard slot power cap), unpacking K-quant scales on CUDA cores in `mul_mat_vec_q` hits the 72 W power cap and throttles SM clocks to ~806–1286 MHz ([hanxiao Power limit](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fhanxiao%2FQwen3.8-27B-UD-Q4_K_XL-L4); [llama.cpp#28090](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F28090)). On `A100-SXM4-40GB` (108 SMs, 400 W TDP), the GPU holds ~1404–1410 MHz boost with zero power throttling and has enough SMs to saturate `1,555 GB/s` HBM2 via MMVQ's streaming one-row-per-block layout, whereas MMQ (`mul_mat_q`) pays a fixed shared-memory staging admission penalty rounded up to an 8-column tile (`J = 8`).

### 3.3 Section 3 Verdict

* **Verdict:** **`KEEP_DELETED`**
* Keep both `GGML_MMVQ_MAX` and `LLAMA_SCHED_POOL` deleted. Do not revive `hanxiao`'s L4-only `0001-mmvq-runtime-crossover.patch` on A100-SXM4-40GB.

## 4. Dual-Arch Fatbin String (`sm_80` A100 + `sm_120` Blackwell G4)

### 4.1 How `ggml/src/ggml-cuda/CMakeLists.txt` Processes `80` and `120`

* In `llama.cpp` (`master` `efe3626`), `ggml/src/ggml-cuda/CMakeLists.txt` contains an unconditional loop outside `if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` that inspects `CMAKE_CUDA_ARCHITECTURES` and rewrites any plain `12X` / `120-real` entry to `12Xa` / `120a-real`, logging `-- Replacing 120 in CMAKE_CUDA_ARCHITECTURES with 120a` ([llama.cpp#28790 comment by @jahnclawdmonet](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F28790); [llama.cpp#19662 comment by @ORippler](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F19662)).
* **Why `120a-real` Is Mandatory on Blackwell (`sm_120`):** `llama.cpp` instantiates Blackwell hardware-accelerated block-scaled MMA / MXFP4 templates (`mmq-instance-mxfp4.cu`, using PTX features `.kind::mxf4`, `.block_scale`, `.scale_vec::2X`, and `mma with block scale`). In `nvcc` / `ptxas` (CUDA 12.8+ / 13.0+), those instructions are architecture-accelerated features valid **only** on `.target sm_120a` (and `120f`), **not** on base `.target sm_120` or forward-compatible virtual PTX (`compute_120`) ([llama.cpp#19662](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F19662)).

### 4.2 Comparison of the Three Candidate CMake Strings on CUDA 13.0 / Driver 580 (Colab 2026.07)

| Candidate String | Resulting `nvcc` `-gencode` Targets | Fatbin Size & Build Time | Verdict & Risk Profile |
| :--- | :--- | :--- | :--- |
| **`-DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real"`** | `arch=compute_80,code=sm_80` + `arch=compute_120a,code=sm_120a` (SASS cubins only; zero PTX) | **Smallest fatbin, fastest link** (no embedded PTX blobs) | **RECOMMENDED EXACT STRING:** Explicitly targets native A100 SASS (`sm_80`) and native Blackwell architecture-accelerated SASS (`sm_120a`) without relying on CMake string-replacement heuristics and without bloating the Google Drive binary with unused `compute_80` PTX ([llama.cpp#30224](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F30224); [llama.cpp#19662](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F19662)). |
| `-DCMAKE_CUDA_ARCHITECTURES="80;120"` | `80` expands to `sm_80` SASS + `compute_80` PTX; `120` is rewritten by `CMakeLists.txt` to `120a` (`sm_120a` SASS) | **Larger fatbin** (embeds redundant `compute_80` PTX across all CUDA kernels) | **Suboptimal:** Relies on `CMakeLists.txt` rewriting `120 → 120a` and embeds tens of megabytes of unnecessary `compute_80` PTX that slows down Drive sync/load on every Colab sitting. |
| `-DCMAKE_CUDA_ARCHITECTURES="80-real;120-real"` | `sm_80` SASS; `120-real` must be rewritten by `CMakeLists.txt` to `120a-real` | Same as `80-real;120a-real` *only if* rewrite fires | **Fragile:** Passing explicit `120-real` asks CMake for base `.target sm_120` SASS; if `CMakeLists.txt` does not rewrite `120-real` to `120a-real`, `ptxas` aborts with `Error 255` ([llama.cpp#19662](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F19662)). |

### 4.3 What Breaks If You Guess Wrong

1. **Compile-Time `ptxas` Fatal Error (`120`, `120-real`, or `120-virtual` reaching `ptxas` without the `a` suffix):**
   * Compilation aborts at `mmq-instance-mxfp4.cu.o` with `Error 255`:
     `ptxas fatal : Instruction 'mma with block scale' not supported on .target 'sm_120'` and `Feature '.kind::mxf4' not supported on .target 'sm_120'` ([llama.cpp#19662](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F19662)).
2. **Missing Cubin / Immediate Runtime Crash (`80-real` only run on G4, or `120a-real` only run on A100):**
   * Because `-real` omits PTX and `sm_120a` SASS cannot execute on `sm_80` (nor `sm_80` SASS on `sm_120` without PTX), `llama-server` aborts on device init or first matmul with `CUDA error: no kernel image is available for execution on the device` (`cudaErrorNoKernelImageForDevice`).
3. **Multi-Minute PTX JIT Stall + Severe Qwen3 Slowdown on G4 (`80` or `80-virtual` without `120a-real`):**
   * If the binary embeds `compute_80` / `compute_89` PTX but lacks `120a-real` cubins (for example, if compiled on an older toolkit `< 12.8` or with `120a-real` omitted), driver 580 on G4 falls back to **runtime PTX JIT compilation** (freezing startup for several minutes before `/health` responds) and executes non-Blackwell fallback kernels, causing severe throughput regressions on Qwen3 models ([llama.cpp#17822](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F17822)).

## 5. Attached `colab exec` Longevity on `google-colab-cli` 0.7.4

### 5.1 Does the Colab Runtime Proxy Drop an Already-Open `colab exec` WebSocket at 1h / 2h / Never While a 20s `iopub` Heartbeat Is Flowing?

* **Documented Token Refresh Mechanics in v0.7.4:**
  * Per [docs/01\_session\_management.md (Section 6, updated 2026-09-25 via PR #149)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2Fdocs%2F01_session_management.md) and [googlecolab/google-colab-cli#106](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fissues%2F106), `runtimeProxyInfo.token` has a server-reported TTL of `tokenExpiresInSeconds = 3600` (1 hour).
  * In `google-colab-cli` 0.7.4, `State.get_session(name)` checks `SessionState.token_expires_at` whenever a **new** CLI command starts; if the token is within 5 minutes of expiry (3300s), it calls `GET /tun/m/assignments`, extracts the fresh `runtimeProxyInfo.token` and `url`, and persists them to `~/.config/colab-cli/sessions.json`.
  * However, `01_session_management.md` Section 6 explicitly notes:
    > *"Long-lived connections (`repl`, `console`, `ssh`) only present the token on initial connect and aren't refreshed mid-session."*
* **Already-Open `colab exec` WebSocket Longevity Verdict:** **`UNKNOWN` for exact deterministic proxy cutoff (1h vs 2h vs never)**, with two bounded empirical behaviors in the primary issue tracker:
  1. **What v0.7.4 Fixed (`~60 min` discrete polling & `console` drops):** The reproducible 60-minute failure in [issue #106](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fissues%2F106) occurred when polling a session via periodic `colab exec` invocations (where the first new `exec` after 60 minutes sent the expired token in its HTTP/WebSocket handshake, received HTTP 401/404, and triggered `state.prune_session()`). Commenter `@moneythemoney999` in [issue #106](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fissues%2F106) also reported ~60-minute drops on `colab console`, which PR `#149` ("fix: refresh runtime credentials and stabilize console sessions", shipped in `0.7.4`) addressed.
  2. **Multi-Hour Tunnel / Proxy Resets (`~2h15m–3h49m`):** In [googlecolab/colabtools#6101](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fcolabtools%2Fissues%2F6101) and [googlecolab/colabtools#5793](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fcolabtools%2Fissues%2F5793), continuous GPU workloads with an active 30s heartbeat thread experienced unexpected proxy/connection drops clustered around **2h15m to 3h49m** of VM age.
  * Because no source in `google-colab-cli` or `colabtools` provides a formal specification proving whether the Google Colab runtime proxy (`*.prod.colab.dev`) forcibly closes an already-upgraded Jupyter `/api/kernels/<id>/channels` WebSocket at the 1h token boundary, at ~2–3.5h tunnel rotation, or never while 20s `iopub` frames are flowing, treat mid-session WebSocket drops as **expected operational events** and wrap the client command in the zero-reload reconnect loop below.

### 5.2 Documented Reconnect That Re-Attaches to Existing `llama-server` + `cloudflared` PIDs Without Reloading 16.8 GB Weights

* **Why Background Daemons and VRAM Survive a Dropped `colab exec` Socket:**
  * Per [skills/colab-operator/SKILL.md (Mental model)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2Fskills%2Fcolab-operator%2FSKILL.md):
    > *"Kernel state PERSISTS across `colab exec` / `colab repl` calls in the same session. Each invocation reattaches to the same kernel (the kernel ID is cached in local state) and only closes the websocket on exit — it does not shut the kernel down... (`colab stop` and `colab restart-kernel` are what actually reset it.)"*
  * Because `llama-server` (holding the 16.8 GB `Q4_K_M` target + 1.14 GiB `DFlash 2` weights in GPU VRAM) and `cloudflared` run as independent OS processes on the Colab VM, a dropped `colab exec` WebSocket on your laptop **does not terminate `llama-server` or `cloudflared`**.
* **Exact CLI Reconnect Recipe (`colab exec -s qwen-worker --timeout 86400 -f attach_heartbeat.py`):**
  * When the initial `colab exec` exits (due to token rotation, Wi-Fi blip, or proxy reset), immediately invoking a new `colab exec` triggers `State.get_session("qwen-worker")`, which fetches a **fresh 3600s `RuntimeProxyInfo.token`** via `GET /tun/m/assignments` ([01\_session\_management.md Section 6](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2Fdocs%2F01_session_management.md)) and opens a new WebSocket to the existing kernel.
  * Run a small idempotent `attach_heartbeat.py` script via `colab exec`:
    ```bash
    # Laptop-side auto-reconnect loop (re-mints proxy token via get_session on every reconnect):
    while true; do
      colab status -s qwen-worker || break
      colab exec -s qwen-worker --timeout 86400 -f attach_heartbeat.py
      echo "[reconnect] colab exec WebSocket closed at $(date -Is); re-attaching in 5s..."
      sleep 5
    done
    ```
  * Inside `attach_heartbeat.py`, check if `llama-server` (`curl -sf http://localhost:${PORT:-8080}/health`) and `cloudflared` (`pgrep -f cloudflared`) are already alive; if both are healthy, **skip compilation and model loading entirely** and enter the 20–25s `print(..., flush=True)` heartbeat + `/slots` watchdog loop.
  * **Alternative CLI attachment flags ([README.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli), [02\_execution\_and\_interactive.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2Fdocs%2F02_execution_and_interactive.md)):**
    * `cat attach_heartbeat.py | colab repl -s qwen-worker`: Piped non-interactive REPL execution on the same kernel.
    * `colab ssh -s qwen-worker`: Opens a direct SSH-over-WebSocket shell to the existing VM (do **not** pass `--rm`, though `--rm` only stops auto-created sessions).
    * `echo "pgrep -a llama-server; pgrep -a cloudflared" | colab console -s qwen-worker`: Runs a batch shell command inside the VM's `tmux`-backed `/colab/tty` session.
    * **Warning:** Never run `colab restart-kernel -s qwen-worker` or `colab stop -s qwen-worker` during a reconnect.

### 5.3 Confirmation: AI Pro Idle-Kernel Reaping vs. ~10–15 Minute Disconnected-Client Reclaim

* **Confirmed: Idle-Kernel Reaping Does Not Fire While the Heartbeat Cell Is Running:**
  * In [googlecolab/colabtools#5950](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fcolabtools%2Fissues%2F5950), `@colaboratory-team` confirmed that Colab's idle runtime deletion policy only terminates runtimes *"while there's no active/unacknowledged executions on those runtimes at the time of deletions."*
  * In [docs/01\_session\_management.md (Section 5)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fblob%2Fmain%2Fdocs%2F01_session_management.md), `google-colab-cli` removed its separate keep-alive daemon (PR `#144`, 2026-09-25) specifically because *"VM liveness is automatically maintained by the Colab backend based on active kernel executions and interactions."*
  * Meanwhile, the 20–25s `flush=True` stdout print loop prevents the ~80–86s Tunnel Frontend (TFE) silent-stream freeze documented in [googlecolab/google-colab-cli#14](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fissues%2F14).
* **Confirmed: Disconnected-Client Reclaim Remains ~10–15 Minutes If `colab exec` Dies Without Re-Attaching:**
  * Because Google AI Pro does not include Colab Pro+'s 24-hour detached background execution entitlement ([colabtools#5950](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fcolabtools%2Fissues%2F5950)), Colab requires an active client connection.
  * Collaborator `@EvanWiederspan` confirmed in [googlecolab/google-colab-cli#160](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fissues%2F160) that *"Colab mostly depends on checking for live connections to the VM to determine whether to shut down the VM"*, and `@bkowshik` measured in [googlecolab/google-colab-cli#14](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fgooglecolab%2Fgoogle-colab-cli%2Fissues%2F14) that *"during idle gaps between steps... A 10–15 min gap is enough to lose the runtime. The next step fails on reconnect with: `[colab] Session '...' appears to be lost (404/401). Cleaning up.`"*
  * Thus, the laptop-side `while true` reconnect loop in Section 5.2 must reconnect within `<< 10 minutes` (e.g., `sleep 5`) whenever `colab exec` drops.

### 5.4 `llama-server` `GET /slots` Exact JSON Schema & 30-Minute Idle Watchdog Reset Logic

* **Source-Verified Schema (`efe3626`):** In [tools/server/server-context.cpp (server\_slot::to\_json, lines 485–508)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fblob%2Fmaster%2Ftools%2Fserver%2Fserver-context.cpp) and [tools/server/README.md (GET /slots)](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fblob%2Fmaster%2Ftools%2Fserver%2FREADME.md), calling `GET /slots` (with `-H "Authorization: Bearer $LLAMA_API_KEY"`) returns a JSON array of slot objects:

```json
[
  {
    "id": 0,
    "n_ctx": 100000,
    "speculative": true,
    "is_processing": false,
    "id_task": 135,
    "n_prompt_tokens": 4218,
    "n_prompt_tokens_processed": 0,
    "n_prompt_tokens_cache": 0,
    "params": {
      "n_predict": -1,
      "speculative.n_max": 5,
      "speculative.n_min": 0,
      "speculative.p_min": 0.0
    },
    "next_token": [
      {
        "has_next_token": false,
        "has_new_line": false,
        "n_remain": -1,
        "n_decoded": 0
      }
    ]
  }
]
```

* **Exact State Transitions in `server-context.cpp` Between User Turns vs. During Generation:**
  1. **When a slot is actively processing a request (`SLOT_STATE_PROCESSING_PROMPT` or `SLOT_STATE_GENERATING`):**
     * Top-level `is_processing` is **`true`** (`state != SLOT_STATE_IDLE`, line 378).
     * Top-level `id_task` is set to the active `task->id`.
     * During prefill, `n_prompt_tokens_processed` (`stats.n_prompt_processed`) increments; during decode, `next_token[0].n_decoded` (`stats.n_gen`, line 502; note `next_token` is serialized as a 1-element JSON array in `server-context.cpp:498`, though some older README examples show it as an object—handle both in Python) increments with every accepted/verified token batch.
  2. **When a coding turn completes and the slot is idle between user turns (`SLOT_STATE_IDLE`):**
     * `server_slot::release()` (line 425) sets `state = SLOT_STATE_IDLE` (so `is_processing` becomes **`false`**) and calls `reset()` (line 286).
     * Inside `reset()`, `task_prev = std::move(task); task.reset(); stats = {};` (lines 310–314).
     * **Critical consequence:** Because `stats = {}` zeroes `stats.n_gen` and `stats.n_prompt_processed`, **`next_token[0].n_decoded` immediately resets to `0`** as soon as a turn finishes! However, because `to_json()` reads `const auto & ptask = task ? task : task_prev;` (line 492), **top-level `id_task` persists at the completed turn's `task_prev->id`**, and `n_prompt_tokens` (`prompt.tokens.size()`) persists at the cached conversation length!
* **Which Fields the 30-Minute (1800s) Idle Watchdog Must Reset On:**
  * Do **not** reset solely on `is_processing == true` or `n_decoded > 0`:
    * If a fast thinking-off coding turn takes 4 seconds (starting and finishing entirely between two 25-second heartbeat ticks), at both poll instants `is_processing` is `false` and `n_decoded` is `0`!
    * Conversely, if a slot hangs mid-generation (a known speculative decoding failure mode documented in [llama.cpp#23268](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F23268) and [llama.cpp#29527](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fissues%2F29527)), `is_processing` stays `true` forever while `n_decoded` freezes.
  * **Exact Watchdog Reset Rule:** On every 20–25s tick, extract from `slot = slots_json[0]`:
    ```python
    nt = slot.get("next_token", {})
    if isinstance(nt, list):
        nt = nt[0] if nt else {}
    cur_sig = (
        slot.get("id_task"),
        nt.get("n_decoded", 0),
        slot.get("n_prompt_tokens_processed", 0),
        slot.get("n_prompt_tokens", 0),
    )
    ```
    **Reset `last_progress_ts = time.monotonic()` if and only if `cur_sig != prev_sig`.**
  * **Why this handles both the 2-minute pause and a laptop crash:**
    * **2-minute coding pause:** While you read code or run tests for 2 minutes (120s), `cur_sig` stays constant (`elapsed = 120s < 1800s`). The moment your next turn starts (or even if it starts and finishes in 3 seconds between polls), `id_task` and `n_prompt_tokens` change, `cur_sig != prev_sig` fires, and the 1800s timer resets to `0`.
    * **Laptop crash / abandoned session / hung slot:** No new `id_task` arrives and `n_decoded` does not advance. If the client WebSocket stayed open (or re-attached) with no user work for `1800s` (or if `is_processing == true` with `n_decoded` frozen for `1800s`), `time.monotonic() - last_progress_ts >= 1800` triggers `from google.colab import runtime; runtime.unassign()`, terminating CU burn. (If the laptop crashed hard and killed `colab exec`, Colab's ~10–15 min disconnected-client reaper also reclaims the VM.)

### 5.5 120s vs 1800s Idle Unassign Timeout

* **Verdict:** **`PREFERENCE_ONLY`**
* Neither `llama-server` (where `--timeout` defaults to `3600`s and `--sleep-idle-seconds` defaults to `-1` disabled, [tools/server/README.md](https://www.google.com/search?q=https%3A%2F%2Fgithub.com%2Fggml-org%2Fllama.cpp%2Fblob%2Fmaster%2Ftools%2Fserver%2FREADME.md)) nor Google Colab imposes a technical requirement for `120s` vs `1800s` when an active heartbeat cell is running.
* A `120s` (2-minute) idle unassign risks tearing down the VM (and forcing a full re-provision + 16.8 GB model reload) during normal coding pauses (reading diffs, running local tests), whereas `1800s` (30 minutes) caps worst-case abandoned burn at `5.37 CU/hr × 0.5 hr = 2.69 CU` while preserving interactive coding flow.

### 5.6 Standard-Shape `--gpu A100` Silently Returning `A100-SXM4-80GB` at ~6.77 CU/hr: Keep or Stop?

* **Cited Billing & Hardware Evidence:**
  * In [Qiita's 2026 Colab GPU Benchmark (2026-02-27)](https://www.google.com/search?q=https%3A%2F%2Fqiita.com%2Fjin0g%2Fitems%2F8fbfb209ac4e9343c13f), Colab's standard `A100` (`40.0 GB` VRAM, `83.5 GB` host RAM) burned **5.37 CU/hr**, while `A100 HM` (`80.0 GB` VRAM, `167.1 GB` host RAM) burned **7.52 CU/hr** (and `G4` burned **8.71 CU/hr**).
  * In [note.com's Google AI Pro A100 Test (2026-10-11)](https://www.google.com/search?q=https%3A%2F%2Fnote.com%2Funco3%2Fn%2Fn47710f23e25a), an AI Pro user allocated an `NVIDIA A100-SXM4-80GB` (`80 GiB` VRAM, `~167 GiB` host RAM, `HIGH_RAM` shape) and recorded a live burn rate of **6.77 CU/hr** (`200 CU / 6.77 CU/hr = 29.54 hours/month`, compared to `200 CU / 5.37 CU/hr = 37.24 hours/month` at 5.37 CU/hr).
  * Per the [NVIDIA A100 Datasheet PDF](https://www.google.com/search?q=https%3A%2F%2Fwww.nvidia.com%2Fcontent%2Fdam%2Fen-zz%2FSolutions%2FData-Center%2Fa100%2Fpdf%2Fnvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf), `A100-SXM4-80GB` has **2,039 GB/s** HBM2e bandwidth vs **1,555 GB/s** on `A100-SXM4-40GB` (**+31.1% bandwidth** for **+26.1% CU/hr** at `6.77` vs `5.37 CU/hr`).
* **Billing-Grounded Operational Verdict:** **STOP IMMEDIATELY (`colab stop -s qwen-worker` / `runtime.unassign()`), per locked policy.**
  1. In Colab's fleet, `A100-SXM4-80GB` instances are coupled with the `~167 GiB` host RAM (`HIGH_RAM`) VM shape ([Qiita](https://www.google.com/search?q=https%3A%2F%2Fqiita.com%2Fjin0g%2Fitems%2F8fbfb209ac4e9343c13f); [note.com](https://www.google.com/search?q=https%3A%2F%2Fnote.com%2Funco3%2Fn%2Fn47710f23e25a)), which directly triggers the locked stop rule (`~80GB` VRAM or `HIGH_RAM` / `>110 GB` host RAM).
  2. Even if evaluated purely on `6.77 CU/hr` vs `5.37 CU/hr` billing: while active continuous decode gets +31.1% HBM2e bandwidth for +26.1% CU/hr, a sequential coding worker spends the majority of wall-clock session time between turns (reading, editing, testing) where extra VRAM (80 GB vs 40 GB) and extra bandwidth sit idle while burning **+26.1% more CU/hr** (`-7.7 hours` of monthly session time per 200 CU). Because `-c 100000` with `q8_0` KV already fits comfortably in `~23.8–27.6 GiB` on `A100-SXM4-40GB`, paying `6.77 CU/hr` (or `7.52 CU/hr`) buys unused capacity at the expense of monthly hours.

## First-Boot 3-Minute Smoke (Copy-Paste)

Run this block once your Playbook A `llama-server` is up. It assumes `$LLAMA_URL` (e.g., `http://localhost:${PORT:-8080}`) and `$LLAMA_API_KEY` are set in your shell environment, and does not include or modify your locked `llama-server` launch command.

```bash
# 0. GPU SKU, VRAM (<31,000 MiB gate), Clocks, Power, and Host RAM (<110 GB gate)
nvidia-smi --query-gpu=name,memory.used,memory.total,clocks.sm,clocks.mem,power.draw,power.limit --format=csv
free -g

# Optional (run BEFORE starting llama-server if you want a true 12s no-spec hardware baseline,
# because llama-server ignores per-request "speculative.n_max": 0 per llama.cpp#30204 / #26669):
# ./llama-bench -m "$TARGET_GGUF" -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 1

# 1. Health & Slot Readiness Check
curl -sf "${LLAMA_URL}/health" | jq .
curl -sf -H "Authorization: Bearer ${LLAMA_API_KEY}" "${LLAMA_URL}/slots" | jq '.[0] | {id, n_ctx, speculative, is_processing, id_task, params: {n_max: .params["speculative.n_max"], n_min: .params["speculative.n_min"]}}'

# 2. CURL 1 — One-Shot Spec (n_max=5) Short Coding Prompt (Cold Cache; Gate: predicted_per_second >= 95.0 tok/s)
curl -sf "${LLAMA_URL}/v1/chat/completions"   -H "Content-Type: application/json"   -H "Authorization: Bearer ${LLAMA_API_KEY}"   -d '{
    "model": "qwen-worker",
    "cache_prompt": false,
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [
      {"role": "user", "content": "Write a self-contained Python 3 module implementing an iterative AVL tree with insert, delete, rebalance, and level-order traversal. Code only, no prose."}
    ]
  }' | jq '{
    smoke_step: "curl_1_oneshot_spec_nmax5",
    predicted_per_second: .timings.predicted_per_second,
    predicted_n: .timings.predicted_n,
    predicted_ms: .timings.predicted_ms,
    prompt_per_second: .timings.prompt_per_second,
    prompt_n: .timings.prompt_n,
    cache_n: .timings.cache_n,
    draft_n: .timings.draft_n,
    draft_n_accepted: .timings.draft_n_accepted,
    accept_pct: (if (.timings.draft_n // 0) > 0 then (100.0 * .timings.draft_n_accepted / .timings.draft_n) else 0 end)
  }'

# 3. CURL 2 — Multi-Turn Code Edit Hitting ngram-map-k4v + Prefix Cache (Gate: cache_n > 0 and tok/s > Curl 1)
curl -sf "${LLAMA_URL}/v1/chat/completions"   -H "Content-Type: application/json"   -H "Authorization: Bearer ${LLAMA_API_KEY}"   -d '{
    "model": "qwen-worker",
    "cache_prompt": true,
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [
      {"role": "user", "content": "Write a Python class TokenBucketRateLimiter with methods allow_request(self, cost: int = 1) -> bool, _refill(self) -> None, and snapshot_state(self) -> dict."},
      {"role": "assistant", "content": "```python
import time

class TokenBucketRateLimiter:
    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0.0:
            self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)
            self.last_refill = now

    def allow_request(self, cost: int = 1) -> bool:
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def snapshot_state(self) -> dict:
        self._refill()
        return {"capacity": self.capacity, "tokens": self.tokens, "refill_rate": self.refill_rate}
```"},
      {"role": "user", "content": "Re-emit the complete TokenBucketRateLimiter class verbatim, renaming allow_request to consume_tokens and adding a reset_bucket(self) -> None method."}
    ]
  }' | jq '{
    smoke_step: "curl_2_followup_ngram_k4v",
    predicted_per_second: .timings.predicted_per_second,
    predicted_n: .timings.predicted_n,
    predicted_ms: .timings.predicted_ms,
    prompt_per_second: .timings.prompt_per_second,
    prompt_n: .timings.prompt_n,
    cache_n: .timings.cache_n,
    draft_n: .timings.draft_n,
    draft_n_accepted: .timings.draft_n_accepted,
    accept_pct: (if (.timings.draft_n // 0) > 0 then (100.0 * .timings.draft_n_accepted / .timings.draft_n) else 0 end)
  }'

# 4. OPTIONAL — Cheap ~32K Prefill & Deep-Context Decode Probe (~10–14s on A100)
python3 -c '
import json
chunk = "def compute_step(x: int, y: int) -> int:
    return ((x * 31) ^ (y + 17)) & 0xFFFFFFFF

"
body = {
    "model": "qwen-worker",
    "cache_prompt": false,
    "temperature": 0,
    "max_tokens": 128,
    "chat_template_kwargs": {"enable_thinking": False},
    "messages": [{"role": "user", "content": (chunk * 1350) + "
Write a unit test function test_compute_step() that asserts three distinct inputs."}]
}
print(json.dumps(body))
' | curl -sf "${LLAMA_URL}/v1/chat/completions"   -H "Content-Type: application/json"   -H "Authorization: Bearer ${LLAMA_API_KEY}"   -d @- | jq '{
    smoke_step: "optional_32k_prefill_and_decode",
    prompt_n: .timings.prompt_n,
    prompt_per_second: .timings.prompt_per_second,
    prompt_ms: .timings.prompt_ms,
    predicted_per_second: .timings.predicted_per_second,
    predicted_n: .timings.predicted_n,
    draft_n: .timings.draft_n,
    draft_n_accepted: .timings.draft_n_accepted
  }'
```

### Fields to Record in Your Run Log

* `nvidia-smi`: `name` (must be `NVIDIA A100-SXM4-40GB`), `memory.used` (`<= 31000 MiB`), `clocks.sm` (`~1410 MHz` under load), `power.limit` (`400 W`), and `free -g` total host RAM (`~83 GB`, `< 110 GB`).
* `timings.predicted_per_second`: Record separately for Optional `llama-bench` no-spec (`>= 50 tok/s`), Curl 1 one-shot `DFlash 2` `n_max=5` (`>= 95 tok/s`), Curl 2 multi-turn `DFlash 2 + ngram-map-k4v` (`> Curl 1`), and Optional 32K probe.
* `timings.draft_n_accepted` and `timings.draft_n`: Record acceptance percentage (`100 * draft_n_accepted / draft_n`) for Curl 1 vs. Curl 2.
* `timings.prompt_n`, `timings.cache_n`, and `timings.prompt_per_second`: Confirm `cache_n > 0` on Curl 2 and record `prompt_per_second` at ~32K.

## Remaining Unknowns Only

1. **Measured `A100-SXM4-40GB` (and `80GB`) `tok/s` for `Qwen3.8-27B` `Q4_K_M` + `DFlash 2` (`Q4_K_M`) + `ngram-map-k4v` (`n_max=5`) at Short Context, ~32K, and 100K:** Unpublished across all primary repositories and benchmark databases (`unknown` until captured by your first sitting's 3-minute smoke test).
2. **Measured `A100-SXM4-40GB` Non-Speculative (`--spec-type none`) Baseline `tok/s` for `Qwen3.8-27B` `Q4_K_M`:** Unpublished (`unknown`; bounded by `~62–75 tok/s` `scaled` from `15.75 GiB` active weight read at `67%–84%` of `1,555 GB/s` HBM2).
3. **Measured 64K–100K Speculative Decode Retention Curve on `A100-SXM4-40GB` or Colab `G4`:** Unpublished (`unknown`; `lukaLLM`'s 65K/131K decode cells emitted EOS after 1 token, leaving only `hanxiao`'s L4 `0.853×` retention from 22K→109K and `llama.cpp#27623`'s non-speculative `0.76×–0.85×` retention from 0→98K on RTX 5090 / 3090 / 4080S).
4. **Measured `q8_0` vs `f16` KV Cache Verify-Step Delta on `A100-SXM4-40GB` at 100K with `DFlash 2`:** Unpublished (`unknown`; `llama.cpp#27623` and `#29371` document the CUDA MMA FlashAttention `q8_0 -> f16` scratch conversion on `sm_89`, but no A100 `sm_80` measurement exists).
5. **Exact Colab Runtime Proxy WebSocket Cutoff (1h vs 2h vs Never) for a Single Already-Open `colab exec --timeout 86400` Channel Streaming 20s `iopub` Heartbeats on `google-colab-cli` 0.7.4:** Undocumented by Google (`unknown`; mitigated deterministically by the `colab exec` auto-reconnect wrapper in Section 5.2, which refreshes `RuntimeProxyInfo.token` via `GET /tun/m/assignments` and re-attaches to the persistent kernel and live `llama-server` / `cloudflared` PIDs in seconds).
