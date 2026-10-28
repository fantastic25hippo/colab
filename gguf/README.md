# GGUF Holder

Clean holder directory for Unsloth Dynamic GGUF weights deposited during the 2026-10-28 session.

## Overview & Index

- [MODELS.md](MODELS.md): Inventory of deposited Dynamic 4-bit GGUF models (`UD-Q4_K_XL`), destination repos, file sizes, SHA256 hashes, source repositories, and deposit methods.
- [SESSION.md](SESSION.md): Operational findings, storage architecture ranking, upload mechanics, local hygiene status, and compute unit (CU) accounting.

## Execution Model

These GGUF weights are formatted for inference via `llama.cpp` / `llama-server`.

Execution strategy on Colab:
- Direct download from private Hugging Face Hub directly into memory (`/dev/shm/bundle`) to bypass slow disk I/O.
- Run `llama-server` with `--no-mmap` to keep weights pinned in RAM/VRAM.
- Expose inference endpoint over SSH loopback tunnel.
- Scripted trap handler ensuring immediate runtime shutdown (`colab stop`) upon exit or interruption.

## Next Steps

To begin using these models (Sahil must explicitly say go before any CU expenditure):

1. **Private Hub Holds Weights**: Both GGUF files are already deposited in private Hugging Face repositories (see [MODELS.md](MODELS.md)).
2. **Missing Components**:
   - Precompiled `llama-server` tarball hosted on Hugging Face for fast retrieval.
   - Dedicated session launcher automating:
     - `colab new`
     - `/dev/shm` remount for size expansion
     - Hugging Face download from private repo directly into `/dev/shm/bundle`
     - `llama-server --no-mmap` launch
     - SSH loopback forwarding
     - Shell `trap` handler calling `colab stop` on exit/failure
3. **GPU Sizing & Selection**:
   - **LFM2.5-24B-A2B Q4 (~15 GB)**:
     - L4: Fits comfortable; cheap smoke tests.
     - A100-40: Best speed-per-CU balance for this GGUF.
     - A100-80 / G4: Fully supported.
   - **Flint Q4 (~25 GB)**:
     - Requires A100-40+ (KV cache is tight on 40 GB) or A100-80 / G4.
     - T4 cannot fit either model.
   - Note: GGUF `Q4_K` utilizes integer quantization kernels in llama.cpp and does not use NVFP4/FP8 tensor cores.
4. **Cost & Runtime Discipline**:
   - Always stop the runtime immediately after testing.
   - Spend zero CU without explicit approval from Sahil.
