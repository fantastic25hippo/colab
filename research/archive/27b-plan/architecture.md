# Architecture

End-to-end plan for an uncensored Qwen 3.8 27B coding worker on Google Colab, paid from a [Google AI Pro](https://support.google.com/googleone/answer/16287313) 200 compute-unit (CU) monthly pool.

Orchestration is local. Weights and a prebuilt `llama-server` live on Google Drive. The GPU boots, serves, then shuts down.

Launch flags: [`inference.md`](inference.md). CLI and shutdown: [`ops.md`](ops.md). First sitting: [`first-boot.md`](first-boot.md).

---

## 1. Orchestrator → worker

| Layer | Where | CU cost | Job |
| :--- | :--- | :--- | :--- |
| **Orchestrator** | Local PC (Antigravity, Grok, or a raw OpenAI client) | 0 | Planning, prompt prep. Thinking stays here. |
| **Storage** | Google Drive `/content/drive/MyDrive/colab_llm/` | 0 | GGUF weights, compiled `llama-server` / `llama-quantize`, `cloudflared`. A Colab VM cannot read a home SSD. Drive copies over Google's backbone in 30–60 s. |
| **Worker** | Colab GPU via [`google-colab-cli`](https://github.com/googlecolab/google-colab-cli) 0.7.4 | Metered | Coding batch with thinking off. Session name `qwen-worker`. Always `colab stop` / `runtime.unassign()`. |

Qwen 3.8 27B is a dense 27.78B hybrid: 64 layers, hidden 5,120, **48 Gated DeltaNet + 16 full GQA** (`qwen35.full_attention_interval=4`). Every token streams the full ~16 GB of 4-bit weights. Only the 16 full-attn layers grow KV with context. The GDN recurrent state is the speculation bottleneck.

Blackfrost's GGUF also embeds a 65th NextN/MTP block (`qwen35.block_count=65`). Playbook A uses DFlash 2, so that head stays idle. Fallback if the sidecar fails: `--spec-type draft-mtp` with no `-md`.

---

## 2. Checkpoints

Primary for `llama.cpp`:

| File | Role | Size | SHA-256 |
| :--- | :--- | :--- | :--- |
| [Blackfrost `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf`](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) | Target | 16,810,716,384 B (16.8 GB) | `5d53637a59cfcd3a4d8354e254ffd44943e5a693da2405a3e228c62962355509` |
| [Blackfrost `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf`](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) | DFlash 2 source | 3,860,292,992 B (3.86 GB) | `ca6dacc0bba87332b4eae6f103fa4c0c800ca871657e83d2aed658351b2c4fb3` |
| `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` | DFlash 2 sidecar (quantize locally) | 1,143,006,592 B expected | Record on first boot |

Quantize the Blackfrost BF16 sidecar on first boot (`KEEP_QUANTIZE_BLACKFROST`). Do not download stock `incoai` / `z-lab` DFlash GGUFs. Skip `mmproj` (poisons DFlash slots). Need ≥ 22 GB free Drive.

The Aug 22 sidecar omits `dflash.rope.dimension_sections` and still **loads** on `efe3626` for text-only serving (`required=false`; 1D RoPE equals text-only M-RoPE). Skip injection on first boot. If injecting later, use `gguf_new_metadata.py`, not `gguf_set_metadata.py`.

Draft hooks in the GGUF are 0-indexed `[5, 19, 33, 47, 61]` (`n_extract=5`). Halt if target KV is 64/65 layers instead of 16.

Other abliterated checkpoints (OBLITERATUS, medismera, jenerallee78) stay on the shelf. medismera is the SGLang / vLLM path if Playbook B is ever revived.

---

## 3. Inference engine

Default is **llama.cpp / llama-server** with DFlash 2 + `ngram-map-k4v`. Compile once on Colab, cache the binary on Drive.

Pin:

- Repo: `ggml-org/llama.cpp`
- Commit `efe3626` or tag `b11552`
- DFlash 2 is on master via PR **#27816** (commit `b10f9ca58c`). Do not check out PR #27342.
- Guest CUDA: driver **580**, toolkit **13.0**
- Fatbin: `-DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real"` (A100 `sm_80` + G4 `sm_120a`). Plain `80;120` embeds unused PTX. `120-real` can fail `ptxas`.

vLLM and SGLang are deferred. Do not serve unquantized BF16 for speed.

---

## 4. GPU, speed, and CU

Colab bills **connected wall-clock**, including install, load, and idle.

| GPU | Usable VRAM | Bandwidth | Hourly CU | Role |
| :--- | :--- | :--- | :--- | :--- |
| **A100-SXM4-40GB** | ~39.6 GB | **1,555 GB/s** | **~5.37–5.40** | **First sitting.** `colab new -s qwen-worker --gpu A100` |
| G4 (RTX PRO 6000 Blackwell **Server Edition**) | ~95.6 GB | **1,597 GB/s** | **~8.71** | Optional if this account can assign it. Often grayed on AI Pro. |
| A100-SXM4-80GB / High-RAM | 80 GB | 2,039 GB/s | ~6.77–15 | **Barred.** Stop even at Standard-shape ~6.77 CU/hr. |
| L4 | 22.5 GB | 300 GB/s | ~1.71 | Rejected. Slow wall-clock makes a 100k-token sitting more expensive. |

A100 40GB is 97.4% of Colab G4 bandwidth at 61.7% of CU/hr. The 1,792 GB/s figure is the Workstation card lukaLLM used, not Colab G4.

Published DFlash 2 + k4v tables are from G4 / L4. A100 40GB tok/s is **unpublished**. Scaled no-spec is ~62–75 tok/s. First-boot gates live in [`first-boot.md`](first-boot.md).

**100K q8_0 KV FIT** on A100 40GB (~23.8–27.6 GiB used of ~39.6). Keep q8_0 (f16 costs +2.86 GiB; no A100 A/B proving it is worth it). After `/health`, `nvidia-smi` must show ≤ 31,000 MiB used.

Finish-or-4-hour cutoff: `colab stop` when the queue is empty, plus a 4-hour watchdog so an unattended session never runs overnight. Thirty minutes of idle on A100 40GB is ~2.69 CU.

---

## 5. Drive packing list

Path: `/content/drive/MyDrive/colab_llm/`

| Artifact | Action |
| :--- | :--- |
| `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` | Download; verify SHA |
| `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` | Download; quantize once |
| `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` | Output of `llama-quantize … Q4_K_M` |
| `bin/llama-server`, `bin/llama-quantize` | Compile on Colab, copy here |
| `bin/cloudflared-linux-amd64` | Cache once |

Warm boot target: under 45 s after Drive is mounted (copy binary, point `-m` at Drive weights, start tunnel). Cold boot (download + compile + quantize) is once.

---

## 6. Boot shape

1. `colab new -s qwen-worker --gpu A100` (no `--high-mem`).
2. Immediate SKU guard: stop if ~80 GB VRAM, `HIGH_RAM`, or host RAM > 110 GB.
3. `colab drivemount` → `/content/drive`. Browser `authuser=0` must be `zuluzain5@gmail.com`.
4. Keep `colab exec -s qwen-worker --timeout 86400 -f start_server.py` **attached**. Fire-and-exit is reaped in ~10–15 min on AI Pro.
5. In-guest: llama-server + cloudflared, 20–25 s heartbeat, 1800 s idle `runtime.unassign()`.
6. Local client talks OpenAI `/v1` at `https://<slug>.trycloudflare.com/v1` with `--api-key`.
7. Every exit path: `colab stop -s qwen-worker`.

Details in [`ops.md`](ops.md). Hermes, if used, gets an ephemeral config. Do not edit `~/.hermes/config.yaml`. Bend wiring is in [`bend-hands.md`](bend-hands.md).

---

## 7. Deferred

- Playbook B (SGLang + NVFP4 on G4)
- Bend Hands implementation
- Unsloth QLoRA of this 27B
- Mixing stock DFlash GGUFs with the Blackfrost target
