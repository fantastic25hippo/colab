# Colab Inference Worker

Sahil's notes and measured runbooks for running open-source models on Google Colab GPUs and TPUs (git repository: [`fantastic25hippo/colab`](https://github.com/fantastic25hippo/colab)), billed against a **Google AI Pro (~400 CU observed)** compute-unit pool.

---

## Current Focus

- **Fast TPU v5e-1 & L4 Cold Starts + Single-Stream Coding Speed.**
  - **TPU v5e-1 (`Qwen/Qwen3-4B` bf16, `vllm-tpu==0.32.0`) [MEASURED]:** **`73.08 s` in-VM (`81.22 s` global from `colab new`)** cold-start to first token (`3.48x` faster than the `282.46 s` cold baseline, spending only **`0.04 CU`**), with **`21.4 ms` TTFT**, **`86.99 tok/s`** single-stream decode (`1,653.4 tok/s` at Batch 32), and **`2.18 s` prefill (`10,223 tok/s`) / `58.31 tok/s` decode at `22,286` tokens**. See [`benchmarks/TPU-v5e/FAST-COLDSTART.md`](benchmarks/TPU-v5e/FAST-COLDSTART.md), [`benchmarks/TPU-v5e/tpu-final/RUN5-FLOOR-COLDSTART.md`](benchmarks/TPU-v5e/tpu-final/RUN5-FLOOR-COLDSTART.md), and [`benchmarks/TPU-v5e/RECIPE.md`](benchmarks/TPU-v5e/RECIPE.md).
  - **L4 Sharded Loading (`LFM2.5-24B-A2B-Q4`) [MEASURED]:** 16 shards (14.88 GB) pulled from public Hugging Face Hub into `/dev/shm` in **16.60 s (896 MB/s)**. Decode benchmark pending `llama-b11954` CUDA 13.4 binary verification. See [`docs/lfm-l4.md`](docs/lfm-l4.md) and [`benchmarks/HF-transfer/L4/SWEEP.md`](benchmarks/HF-transfer/L4/SWEEP.md).
  - **Flint (39.8B MoE, ~2.5B active) [ESTIMATE]:** Public upstream `RecursiveSI/flint-nvfp4-gguf` / `unsloth/flint-GGUF`, unbenchmarked on Colab GPU.

---

## Repository Map (Newest First)

| Path | What it is | Date |
| :--- | :--- | :--- |
| [`research/00-INDEX.md`](research/00-INDEX.md) | **Consolidated Research Master Index (`00`–`11`).** Hardware, storage, HF Xet, L4 loading, Colab CLI, agent workflows, TPU v5e runtime/quantization/cold-start, LFM2.5, and conflict verification. | 2026-10-31 |
| [`benchmarks/TPU-v5e/FAST-COLDSTART.md`](benchmarks/TPU-v5e/FAST-COLDSTART.md) | **TPU v5e-1 Fast Cold-Start Runbook.** CPU-torch `/dev/shm` venv, static topology vars, in-process EngineCore, `jax.clear_caches` patch, and measured `tpu-final/` results. | 2026-10-29 |
| [`benchmarks/TPU-v5e/RECIPE.md`](benchmarks/TPU-v5e/RECIPE.md) | **Universal TPU v5e-1 Serving Recipe.** HBM memory budget formula, persistent XLA cache lifecycle, batching scaling (B=1 to 32), and context limits. | 2026-10-29 |
| [`benchmarks/INDEX.md`](benchmarks/INDEX.md) | **Master Benchmark Index.** All measured TPU v5e, L4, A100, G4, T4, and CPU runs with CU/hr and CU spent. | 2026-10-29 |
| [`docs/lfm-l4.md`](docs/lfm-l4.md) | **LFM2.5-24B-A2B on L4.** Sharded download sweeps into `/dev/shm`, failure analysis, and CUDA 13.4 resume recipe. | 2026-10-28 |
| [`docs/WEIGHTS-LOADING.md`](docs/WEIGHTS-LOADING.md) | **Weights onto a Fresh VM.** Sharded HF pulls into `/dev/shm`, `/dev/shm` capacity table across tiers, and GPU/TPU health checks. | 2026-10-28 |
| [`gguf/README.md`](gguf/README.md) | **GGUF Inventory.** Deposited `LFM2.5-24B-A2B` and `Flint` GGUF artifacts on Hugging Face Hub. | 2026-10-28 |
| [`ops.md`](ops.md) | **Colab Operations.** Account, `google-colab-cli==0.7.5`, non-interactive HF `/dev/shm` loading, heartbeat, and CU quota discipline. | 2026-10-28 |
| [`hardware.md`](hardware.md) | **Hardware Specifications.** Verified Colab AI Pro specs and measured CU/hr burn rates (T4, L4, A100, G4, H100, TPU v5e/v6e). | 2026-10-29 |
| [`docs/OPTMEM.md`](docs/OPTMEM.md) & [`OPTMEM-SEED.md`](OPTMEM-SEED.md) | **OptMem Memory Workflow.** Cross-session CLI memory rules and seed facts. | 2026-10-27 |
| [`research/2026_model_research/`](research/2026_model_research/README.md) & [`research/precision_format/`](research/precision_format/) | **Pre-Benchmark Estimates.** Early model speed/fit dossiers and per-accelerator precision notes (superseded where measured benchmarks exist). | 2026-10-24 |

> [!IMPORTANT]
> Measured numbers in [`benchmarks/`](benchmarks/INDEX.md) always override estimates in [`research/`](research/00-INDEX.md). Spend 0 CU without Sahil's explicit approval, and always stop runtimes immediately after testing.
