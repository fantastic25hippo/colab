# GGUF Holder

Clean holder directory for Unsloth Dynamic GGUF weights deposited during the 2026-10-28 session. For L4 benchmarking and findings, see [docs/lfm-l4.md](../docs/lfm-l4.md).

## Model Inventory

Deposited Unsloth Dynamic 4-bit GGUF models (`UD-Q4_K_XL`) in llama.cpp format.

### 1. LFM2.5-24B-A2B

- **Destination Repository**: `avgrocketman/LFM2.5-24B-A2B-GGUF` (private)
- **File**: `LFM2.5-24B-A2B-UD-Q4_K_XL.gguf`
- **Size**: 14,876,901,632 bytes (~14.9 GB)
- **SHA256**: `a235d23638cc5549cb1ff61da7522c8a7e97f056f27093462a1e56f35aba91d8`
- **Source**: `unsloth/LFM2.5-24B-A2B-GGUF`
- **Deposit Method**: Colab CPU `hf_hub_download` + `upload_file`
- **Performance**: ~98.3 MB/s download, ~92.5 MB/s upload
- **Resource Usage**: 0 CU consumed; session `run-6a5375` auto-stopped upon completion.

### 2. Flint

- **Destination Repository**: `avgrocketman/flint-GGUF` (private)
- **File**: `flint-UD-Q4_K_XL.gguf`
- **Size**: 25,054,688,704 bytes (~25.1 GB)
- **SHA256**: `63c2ecc461ca545134b670c660e76fa20f47240f8d39b6510bfd21c4dd6e072f`
- **Source**: `unsloth/flint-GGUF`
- **Deposit Method**: Local `huggingface_hub.copy_files` (Hub-to-Hub server-side copy)
- **Performance**: Completed in 3.24 seconds; 0 bytes transferred through local machine
- **Resource Usage**: 0 CU consumed.

### Ingestion Protocol

- **Hub-to-Hub**: Use `copy_files` or `hf cp` whenever the target weights already reside on Hugging Face (server-side copy within the same cloud storage region).
- **External Sources**: Only use download + upload streaming if the upstream model is not hosted on Hugging Face Hub.

## Storage Hierarchy & Session Findings

- **Rank-1 Storage**: Private Hugging Face Hub.
- **Rank-2 Storage**: KaggleHub.
- **Dead Storage Plans / Anti-Patterns**:
  - Google Cloud bucket / `gcsfuse`: DEAD (no billing account; high latency and instability with large tensor files).
  - Google Drive mount: DEAD (requires a human click on every new VM; bandwidth throttling and sync delays).
  - `/content` double-copy on Colab instances: Anti-pattern (use `/dev/shm` directly).
- **Fast-Path Ingestion**: Server-side copy (`copy_files` / `hf cp`) is the preferred fast path when weights are already present on the Hub, executing in seconds without local bandwidth or compute consumption.
- **LFS Mechanics**: Hugging Face LFS uploads require computing the SHA256 upfront before the upload request starts. Consequently, single-blob streaming remains serial (`GET` then `PUT`). Multi-shard architectures can pipeline downloading shard `n+1` while uploading shard `n`.
- **Status of GGUF Benchmarks**:
  - `LFM2.5-24B-A2B-Q4`: Sharded into `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (public) and download-benchmarked on L4 (16.60 s into `/dev/shm`); `llama-server` decode pending CUDA 13.4 binary test (see [`../docs/lfm-l4.md`](../docs/lfm-l4.md)).
  - `Flint`: Deposited in `avgrocketman/flint-GGUF` (25.1 GB single blob); unsharded and unbenchmarked on Colab GPU.
- **Resource Accounting & Policy**:
  - Check live CU balance via `colab --auth=oauth2 usage` before and after every session.
  - Inference Gate: Do not start any L4, A100, or G4 instances for inference until Sahil explicitly gives approval.

## Execution Model (Planned for GGUF / `llama.cpp`)

These GGUF weights are formatted for inference via `llama.cpp` / `llama-server` (pin `llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz` on Colab CUDA 13.0 drivers; never compile from source).

Execution strategy on Colab:
- Direct download of sharded GGUF from Hugging Face Hub into `/dev/shm/bundle` (`snapshot_download`, `max_workers=12`).
- Run `llama-server` with `--no-mmap -ngl 99` to keep weights pinned in VRAM.
- Keep I/O on `colab exec` (with a 20 s heartbeat) or `colab ssh --proxy-mode`, and always run `colab stop` in a `finally:` block.

## Next Steps

To begin using these models (Sahil must explicitly say go before any CU expenditure):

1. **Missing Components**:
   - Precompiled `llama-server` tarball hosted on Hugging Face for fast retrieval.
   - Dedicated session launcher automating:
     - `colab new`
     - `/dev/shm` remount for size expansion
     - Hugging Face download from private repo directly into `/dev/shm/bundle`
     - `llama-server --no-mmap` launch
     - SSH loopback forwarding
     - Shell `trap` handler calling `colab stop` on exit/failure
2. **GPU Sizing & Selection**:
   - **LFM2.5-24B-A2B Q4 (~15 GB)**:
     - L4: Fits comfortable; cheap smoke tests.
     - A100-40: Best speed-per-CU balance for this GGUF.
     - A100-80 / G4: Fully supported.
   - **Flint Q4 (~25 GB)**:
     - Requires A100-40+ (KV cache is tight on 40 GB) or A100-80 / G4.
     - T4 cannot fit either model.
   - Note: GGUF `Q4_K` utilizes integer quantization kernels in llama.cpp and does not use NVFP4/FP8 tensor cores.
3. **Cost & Runtime Discipline**:
   - Always stop the runtime immediately after testing.
   - Spend zero CU without explicit approval from Sahil.
