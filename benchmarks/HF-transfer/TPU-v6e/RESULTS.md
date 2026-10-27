## 2026-10-28 00:43 IST: HF-transfer benchmark on TPU-v6e (SKIPPED / TIMEOUT)

**Model:** bartowski/Qwen2.5-7B-Instruct-GGUF (`Qwen2.5-7B-Instruct-Q4_K_M.gguf`, 4.68 GB) into /dev/shm; bartowski/Qwen2.5-14B-Instruct-GGUF (`Qwen2.5-14B-Instruct-Q8_0.gguf`, 15.70 GB) into disk (ESTIMATE)
**Weights:** HF transfer download via `hf_transfer`
**Hardware:** Cloud TPU v6e-1 (Trillium), 32.0 GB HBM (ESTIMATE from hardware.md)
**CLI:** `colab` 0.7.5, session `hf-v6e`, command `colab --auth=oauth2 new -s hf-v6e --tpu v6e1`
**Engine:** hf_transfer (Python huggingface_hub + hf_transfer)
**Flags:** `HF_HUB_ENABLE_HF_TRANSFER=1`
**Supported Engines / Frameworks:** JAX, PyTorch/XLA, vLLM TPU. Official doc: [Cloud TPU v6e (Trillium)](https://docs.cloud.google.com/tpu/docs/v6e) cited in `hardware.md`. llama.cpp GGUF has no native TPU backend.

### Results

| Test | tok/s | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| pp512 (prompt) | n/a | n/a | 0 | MEASURED |
| tg128 (generation) | n/a | n/a | 0 | MEASURED |

Tok/s is n/a: benchmark skipped due to backend provisioning timeout (no capacity available across retries).
- `colab new --help` options and timeout/retry:
  `colab new` options: `--session/-s`, `--tpu`, `--gpu`, `--high-mem`, `--help/-h`.
  No timeout or retry flags exist on `colab new`.
  TPU option text: `--tpu <str> TPU accelerator variant. Supported: v5e1, v6e1. Ignored for accelerators that only offer a single shape (L4, v5e1, v6e1).`
- Provisioning attempts (MEASURED):
  - Attempt 1: `colab --auth=oauth2 new -s hf-v6e --tpu v6e1` exited 1 after 9m 15s:
    `ReadTimeout: HTTPSConnectionPool(host='colab.research.google.com', port=443): Read timed out. (read timeout=120)`
    `colab sessions`: empty (`[colab] No active sessions found on server.`).
  - Attempt 2: `colab --auth=oauth2 new -s hf-v6e --tpu v6e1` exited 1 after 9m 17s:
    `ReadTimeout: HTTPSConnectionPool(host='colab.research.google.com', port=443): Read timed out. (read timeout=120)`
    `colab sessions`: empty (`[colab] No active sessions found on server.`).
- No VM was allocated; `colab sessions` remained empty.

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new` attempt 1) | timed out (>555 s) | MEASURED |
| Runtime up (`new` attempt 2) | timed out (>557 s) | MEASURED |

### Cost

- CU rate: ~2.50 – 3.80 CU/hr (ESTIMATE from `hardware.md`).
- CU used: 0.00 CU (balance before 398.44, after 398.44 CU) (MEASURED).

### Gotchas

- TPU v6e (Trillium) is listed as supported in CLI help (`--tpu v6e1`), but repeated session creation attempts timed out after 9+ minutes waiting for server response, confirming lack of available capacity in this tier/region.
- No CU was consumed since neither session entered READY state.
- Official engine URL: `https://docs.cloud.google.com/tpu/docs/v6e`.

**Runtime stopped:** yes, `sessions` verified empty at 00:43 IST. Spent 0.00 CU.
