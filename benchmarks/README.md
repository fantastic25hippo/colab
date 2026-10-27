# Benchmarks

Measured results for running models on Colab GPUs and TPUs, written so a future Grok can bench a new model with no trial and error.

**Goal.** Sahil downloads a model, asks Grok to bench it, and Grok benches it by following this file. Nothing here is a guess unless it says `ESTIMATE` or `UNVERIFIED`.

## Where to look, where to write

- Read first: [`../ops.md`](../ops.md) (CLI, Drive, quota), [`../hardware.md`](../hardware.md) (accelerators, CU rates), [`../research/`](../research/README.md) (speed estimates, not measurements).
- Results: `benchmarks/<Model>/<GPU>/RESULTS.md`, model first, then GPU, e.g. `benchmarks/Spark-X2.6-4B/T4/RESULTS.md`. Newest entry on top. Optional raw logs sit beside it in `logs/`.
- Index: one line per entry appended to [`INDEX.md`](INDEX.md).
- Entry format: copy [`_TEMPLATE.md`](_TEMPLATE.md). Every field is required; write `n/a` and why if one cannot be filled.
- Names: `<Model>` = model name and size as on the card (`Spark-X2.6-4B`). `<GPU>` = `T4`, `L4`, `A100-40`, `A100-80`, `G4`, `TPU-v5e`, `TPU-v6e`. No empty folders: create a folder with its first entry.

## Rules

1. Never run a model from the Drive mount. Copy the weights to local `/tmp` first and time the copy.
2. Never re-download weights from Hugging Face when they are already in Drive. Sahil keeps them at `/content/drive/MyDrive/models/<name>.gguf/<name>.gguf` (the folder and the file share a name).
3. No keys, tokens, OAuth files or any account email in this repo, in logs or in commit messages. The repo is public.
4. One short run (about 10 minutes). If something fails, record the failure, do not improvise a bigger run.
5. Always stop the runtime at the end (`colab --auth=oauth2 stop -s <session>`), then verify `colab --auth=oauth2 sessions` is empty. Do this even when a step failed.
6. Spend CU only with Sahil's go. Bots never log in, never type Google codes, never approve Drive.
7. Label every number `MEASURED` (from your run) or `ESTIMATE`. Copy nothing from `research/` as measured.

## The Colab CLI (`colab`, version 0.7.5 on the box)

- Global flag first: `colab --auth=oauth2 <subcommand>`. Check with `colab version` and `colab --auth=oauth2 whoami`.
- `new -s NAME --gpu T4|L4|G4|H100|A100` or `--tpu v5e1|v6e1`; always pass `-s`. A wrong `--gpu` silently gives an A100. A 400 on `new` means no entitlement. (VERIFIED from `colab skill`.)
- `exec -s NAME -f FILE.py --timeout SECONDS` runs Python on the VM. Default timeout is 30 s, so always set it. Kernel state persists between `exec` calls. `--env KEY=VALUE` sets variables.
- `drivemount -s NAME` is interactive and needs the human's browser approval on every new VM. A bot must stop and ask Sahil. Run without a human it printed the approval link, then failed with HTTP 400 after 90 s (seen 2026-10-27).
- `usage` prints balance and hourly rate (`Current balance`, `Usage rate: N/hr`). `status -s NAME` shows hardware. `sessions` lists runtimes. `stop -s NAME` releases the VM.
- `colab run [--gpu T4] script.py` = new + exec + stop in one command, but has no Drive mount step (UNVERIFIED), so use `new` + `exec` for Drive runs.

## Measured once: T4 runtime, 2026-10-27 (IST), from an aborted proof attempt

- MEASURED: runtime up in 3.3 s (`new`). Rate 1.07 CU/hr (`usage`). Balance read 400.00 CU before.
- MEASURED: driver 580.82.07, CUDA 13.0, Tesla T4 15360 MiB, 2 CPU cores, Ubuntu 24.04.
- No benchmark ran: Drive mount stopped at the approval link. Other rates are ESTIMATE (`hardware.md`).

## Engines per accelerator

| Accelerator | Engine | Status |
| :--- | :--- | :--- |
| G4 | llama.cpp CUDA build with a GGUF. For Flint, `RecursiveSI/flint-nvfp4-gguf` is the only route its model card documents (tested on RTX PRO 6000 Blackwell). | VERIFIED for Flint only |
| T4, L4, A100, H100 | llama.cpp CUDA build with a GGUF | UNVERIFIED, check on the first run |
| TPU v5e, v6e | none chosen yet | UNVERIFIED |

Also VERIFIED for Flint: `RecursiveSI/flint-int4` is MLX (Apple only), and SGLang needs at least 141 GB. Flint support landed in llama.cpp on 2026-10-21 (PR 30928).

## Bench a new model (runbook)

1. Find the weights file in Drive (`/content/drive/MyDrive/models/<name>.gguf/<name>.gguf`). Note its name and size. Do not download from Hugging Face.
2. Check that the model fits the GPU: file size plus KV cache under the usable VRAM in `hardware.md`. Pick the cheapest GPU that fits.
3. Preflight (free): `colab version`, `colab --auth=oauth2 whoami`, `sessions` (must be empty), `usage` (balance and rate). Get Sahil's go.
4. `colab --auth=oauth2 new -s bench --gpu <GPU>`. Time it. Run `nvidia-smi` through `exec` and confirm the GPU, driver and CUDA.
5. `colab --auth=oauth2 drivemount -s bench`. STOP and ask Sahil to approve in his browser (profile signed in to his Colab account). Wait for his reply.
6. Over `exec --timeout 1800`: copy the GGUF to `/tmp`, timing it (`cp` or `shutil.copyfile`).
7. Install the engine and note the exact version/build. For llama.cpp use a build that matches the driver's CUDA, and record the install time. Do not build from source if a prebuilt binary or wheel works.
8. `llama-bench -m /tmp/<file> -ngl 99 -p 512 -n 128 -r 3 -o md` gives pp512 and tg128 with stddev. Then one real generation of about 350 tokens from a prompt of about 1K tokens through `llama-server` or `llama-cli`; check the output is coherent and quote a short sample. (Flags UNVERIFIED on Colab until the first entry proves them; correct them in the entry.)
9. Run `usage` for balance and rate, then stop the runtime and verify `sessions` is empty. Compute CU used = balance before minus balance after.
10. Write the entry from `_TEMPLATE.md`, append the `INDEX.md` row, grep for `@`, `key`, `token`, then commit and push to `main`.

CU per 100K tokens = (100000 / tg_tok_s) / 3600 x CU_per_hr.

## Failures and gotchas

- `exec` default timeout is 30 s. Expiry detaches the laptop but does not kill guest processes.
- Google AI Pro has no background execution: an idle or detached runtime is reaped in about 10 to 15 minutes. Keep `exec` attached and print a line every 20 s on long steps.
- `drivemount` times out after 120 s on the VM side (`ops.md`); do not re-run it blindly, first check `os.path.ismount('/content/drive')` inside the guest.
- Reading a model from the Drive mount is slow: always copy to `/tmp` first. Colab disk was 194 GB free on the T4 runtime (MEASURED 2026-10-27).
- A VM that is not stopped keeps burning CU. `runtime.unassign()` inside the guest also releases it; still run `colab stop` and check `sessions`.
- 0.7.2 has a broken `exec` and 0.7.3 was yanked (`ops.md`); use 0.7.4 or newer.
