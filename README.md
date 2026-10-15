# Colab Qwen worker

Abliterated **Qwen 3.8 27B** coding worker on Google Colab, billed from a **Google AI Pro 200 CU/month** pool.

This folder is not a git repo. There is no `start_server.py` yet. Do not start a GPU, implement Bend Hands, or edit `~/.hermes/config.yaml` unless asked.

## Read in this order

| File | What it is |
| :--- | :--- |
| [`architecture.md`](architecture.md) | Model, GPU, Drive files, CU, engine pin |
| [`inference.md`](inference.md) | Compile, Playbook A launch, tools, sampling |
| [`ops.md`](ops.md) | CLI, Drive mount, tunnel, heartbeat, stop |
| [`first-boot.md`](first-boot.md) | First sitting checklist, log grep, smoke gates |
| [`bend-hands.md`](bend-hands.md) | Shelved Grok → Gemini → Hermes plan |
| [`research/`](research/README.md) | Ingested papers (evidence only) |

## Locked

- **Worker:** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) `Q4_K_M` + locally quantized Blackfrost DFlash 2 `Q4_K_M`.
- **Engine:** `llama.cpp` master `efe3626` / tag `b11552`, `--spec-type draft-dflash,ngram-map-k4v`, thinking off.
- **GPU:** `colab new -s qwen-worker --gpu A100` (A100-SXM4-40GB, ~5.37 CU/hr). Never `--high-mem`. Stop if the VM is ~80 GB VRAM, `HIGH_RAM`, or host RAM > 110 GB.
- **Context:** `-c 100000`, `-ctk q8_0 -ctv q8_0`, `-fa on`, `--parallel 1`.
- **Session:** keep `colab exec --timeout 86400` attached with a 20–25 s stdout heartbeat. Always `colab stop -s qwen-worker` on exit.
- **Idle watchdog:** 1800 s of no progress on `/slots`, then `runtime.unassign()`.
- **Drive:** `/content/drive/MyDrive/colab_llm/`. Unix LF in every file sent to the server.

## Open preferences

Answer these before the first sitting if you care; otherwise the docs already pick a default.

1. First client: raw OpenAI SDK, Antigravity, or Hermes.
2. Where `LLAMA_API_KEY` lives: laptop env (passed with `colab exec --env`) vs a Drive file.
3. Run a one-shot `colab new --gpu G4` probe (then stop), or stay A100-only.
4. Sampling override: keep Qwen non-thinking `--temp 0.7 --top-p 0.8 --top-k 20`, or force greedy `temp=0` on the server.
5. Idle unassign 1800 s (documented) vs 120 s.
