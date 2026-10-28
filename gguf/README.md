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
- **Local Hygiene Status**:
  - `~/.config/colab-cli/token.json` permissions: Currently `664`; needs remediation to `600`.
  - SSH configuration: Missing `~/.ssh/config` entry for automated tunnels.
  - Binaries: Precompiled `llama-bin.tar.gz` bundle not yet staged on Hugging Face.
  - Automation scripts: `run_*_session.sh` orchestration script pending.
  - Lifecycle management: Trap-to-`colab stop` wrapper script pending implementation.
- **Resource Accounting & Policy**:
  - Compute Units Balance: 398.08 CU available (burn rate: 0 CU/hr; active sessions: 0 / empty).
  - Inference Gate: Do not start any L4, A100, or G4 instances for inference until Sahil explicitly gives approval.

## Execution Model

These GGUF weights are formatted for inference via `llama.cpp` / `llama-server`.

Execution strategy on Colab:
- Direct download from private Hugging Face Hub directly into memory (`/dev/shm/bundle`) to bypass slow disk I/O.
- Run `llama-server` with `--no-mmap` to keep weights pinned in RAM/VRAM.
- Expose inference endpoint over SSH loopback tunnel.
- Scripted trap handler ensuring immediate runtime shutdown (`colab stop`) upon exit or interruption.

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
