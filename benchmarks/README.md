# Benchmarks

Measured results for running models on Colab GPUs and TPUs, written so an agent can benchmark a model with zero trial and error.

**Goal.** Nothing here is a guess unless explicitly labeled `ESTIMATE` or `UNVERIFIED`.

## Where to look, where to write

- **Read first:** [`../ops.md`](../ops.md) (CLI, quota), [`../hardware.md`](../hardware.md) (accelerators, measured CU/hr), [`../docs/WEIGHTS-LOADING.md`](../docs/WEIGHTS-LOADING.md) (sharded HF pulls into `/dev/shm`), [`TPU-v5e/RECIPE.md`](TPU-v5e/RECIPE.md) & [`TPU-v5e/FAST-COLDSTART.md`](TPU-v5e/FAST-COLDSTART.md) (TPU v5e-1 serving and fast cold start), and [`../research/00-INDEX.md`](../research/00-INDEX.md) (consolidated research).
- **Results layout:**
  - Model benchmarks on GPU: `benchmarks/<Model>/<GPU>/RESULTS.md` (e.g. [`LFM2.5-24B-A2B-Q4/L4/RESULTS.md`](LFM2.5-24B-A2B-Q4/L4/RESULTS.md)).
  - TPU v5e-1 runs: `benchmarks/TPU-v5e/<run>/RESULTS.md` (e.g. [`TPU-v5e/tpu-final/RESULTS.md`](TPU-v5e/tpu-final/RESULTS.md)).
  - Transfer sweeps: `benchmarks/HF-transfer/<GPU>/RESULTS.md`.
- **Index:** Append one row per entry (newest first) to [`INDEX.md`](INDEX.md).
- **Entry format:** Copy [`_TEMPLATE.md`](_TEMPLATE.md). Every field is required; write `n/a` and why if a field does not apply.

## Rules

1. **Stage weights in `/dev/shm` from Hugging Face Hub.** Never run a model from a Google Drive FUSE mount, never use `colab drivemount` (interactive on every new VM), and do not use GCS buckets (no billing account). Pull sharded weights directly into `/dev/shm` (`snapshot_download`, `max_workers=8..12`); see [`../docs/WEIGHTS-LOADING.md`](../docs/WEIGHTS-LOADING.md).
2. **No secrets.** Never put keys, tokens, OAuth files, or account emails in this repo, on CLI argv, in logs, or in commit messages. Upload `/content/hf_token` via `colab upload` when a private repo is required.
3. **One bounded run.** Keep runs short and deterministic. Always emit a stdout line every ~20 s so `colab exec` flushes and the Colab reaper does not kill the VM.
4. **Always stop the runtime** at the end (`colab --auth=oauth2 stop -s <session>`), then verify `colab --auth=oauth2 sessions` is empty. Wrap session lifecycle in `try: ... finally:`.
5. **Spend CU only with Sahil's explicit go.**
6. **Label every number** `MEASURED` (from a live run) or `ESTIMATE`.

## The Colab CLI (`colab`, version 0.7.5)

- Global flag first: `colab --auth=oauth2 <subcommand>`. Verify with `colab version` and `colab --auth=oauth2 whoami`.
- `new -s NAME --gpu T4|L4|G4|H100|A100` or `--tpu v5e1|v6e1`; always pass `-s`. Note: `v6e1` timed out repeatedly during provisioning on 2026-10-28 (capacity unavailable, 0 CU spent).
- `exec -s NAME -f FILE.py --timeout 86400` runs Python on the VM. Default timeout is 30 s, so always pass `--timeout 86400`.
- `usage` prints current balance and hourly rate (`Current balance`, `Usage rate: N/hr`). `sessions` lists active runtimes. `stop -s NAME` releases the VM.

## Verified Engines per Accelerator

| Accelerator | Engine & Pin | Status |
| :--- | :--- | :--- |
| **TPU v5e-1** | `vllm-tpu==0.32.0` + `tpu-inference==0.32.0` (`torch==2.10.0+cpu`, Python 3.12 venv in `/dev/shm`) | **VERIFIED** (`73.08 s` in-VM / `81.22 s` global cold start, `21.4 ms` TTFT, **86.99 tok/s** decode, `0.04 CU`; see [`TPU-v5e/FAST-COLDSTART.md`](TPU-v5e/FAST-COLDSTART.md)) |
| **L4, T4, A100, G4** | Prebuilt `llama.cpp` `b11954` CUDA 13.4 binary (`llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz`) | **PINNED** (CUDA 12.8 binary failed `/health` against Colab driver 580.82.07 / CUDA 13.0; never compile on Colab) |
| **G4 (Flint NVFP4)** | `llama.cpp` CUDA build with `RecursiveSI/flint-nvfp4-gguf` or `avgrocketman/flint-GGUF` | **UNVERIFIED on Colab** (documented on model card) |

## Subagent Execution & Live SSH Monitoring Protocol

When running Colab benchmarks or cold-start tests, the main agent prepares the scripts and delegates live execution + monitoring to a **`flash` subagent** (`invoke_subagent` with `Model="flash"`, reusing the same subagent via `send_message` across iterative runs):

1. **Preflight (0 CU):** Confirm `colab --auth=oauth2 sessions` is empty and record starting `colab --auth=oauth2 usage`.
2. **Background Driver Launch:** The subagent launches the host runner in the background (e.g. `python3 -u benchmarks/TPU-v5e/run_coldstart.py > /tmp/run_coldstart.log 2>&1`).
3. **Live SSH Inspection:** Once `colab new -s <session>` provisions the VM, the subagent connects via SSH proxy mode without disturbing the active `colab exec` stream:
   ```bash
   ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
     -o ProxyCommand="colab --auth=oauth2 ssh -s <session> --proxy-mode" \
     root@colab-<session> "tail -n 30 /content/build.log; df -h /dev/shm"
   ```
4. **Teardown & Verification:** When the driver finishes, its `finally:` block runs `colab --auth=oauth2 stop -s <session>`. The subagent verifies `colab --auth=oauth2 sessions` is empty (`0` active sessions), records the final `colab --auth=oauth2 usage` CU delta, and returns the structured JSON metrics to the main agent.

## Runbooks

- **TPU v5e-1 (`vLLM` / `tpu-inference`):** Follow [`TPU-v5e/FAST-COLDSTART.md`](TPU-v5e/FAST-COLDSTART.md) and [`TPU-v5e/RECIPE.md`](TPU-v5e/RECIPE.md).
- **GPU (`llama.cpp` GGUF):**
  1. Preflight (0 CU): check `colab --auth=oauth2 sessions` (empty) and `usage` (record starting CU). Get Sahil's go.
  2. Provision: `colab --auth=oauth2 new -s bench --gpu <GPU>`. Confirm GPU, driver, VRAM (`nvidia-smi`), and `/dev/shm` (`df -h /dev/shm`).
  3. Pull sharded GGUF into `/dev/shm` via `snapshot_download(..., max_workers=12)` and unpack prebuilt `llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz`.
  4. Run `llama-bench -m /dev/shm/<file> -ngl 99 -p 512 -n 128 -r 3 -o md` plus a 1K-in / 256-out generation check; verify `nvidia-smi` VRAM delta and `offloaded N/N layers`.
  5. Stop runtime (`colab --auth=oauth2 stop -s bench`), confirm `sessions` is empty, record ending `usage` CU, write `RESULTS.md`, and update [`INDEX.md`](INDEX.md).

$$\text{CU per 100K output tokens} = \frac{100{,}000}{\text{tg\_tok\_s} \times 3600} \times \text{CU\_per\_hr}$$
