# GGUF Session Findings (2026-10-28)

Architectural and operational findings from model weight deposits and runtime preparations.

## Storage Hierarchy & Ingestion Patterns

- **Rank-1 Storage**: Private Hugging Face Hub.
- **Rank-2 Storage**: KaggleHub.
- **Anti-Patterns**:
  - No `gcsfuse` (high latency and instability with large tensor files).
  - No `/content` double-copy on Colab instances.
  - No Google Drive mounting for weight storage (bandwidth throttling and sync delays).
- **Fast-Path Ingestion**: Server-side copy (`copy_files` / `hf cp`) is the preferred fast path when weights are already present on the Hub, executing in seconds without local bandwidth or compute consumption.
- **LFS Mechanics**: Hugging Face LFS uploads require computing the SHA256 upfront before the upload request starts. Consequently, single-blob streaming remains serial (`GET` then `PUT`). Multi-shard architectures can pipeline downloading shard `n+1` while uploading shard `n`.

## Local Environment & Hygiene

Status of pending hygiene and tooling tasks:

- `~/.config/colab-cli/token.json` permissions: Currently `664`; needs remediation to `600`.
- SSH configuration: Missing `~/.ssh/config` entry for automated tunnels.
- Binaries: Precompiled `llama-bin.tar.gz` bundle not yet staged on Hugging Face.
- Automation scripts: `run_*_session.sh` orchestration script pending.
- Lifecycle management: Trap-to-`colab stop` wrapper script pending implementation.

## Resource Accounting & Policy

- **Compute Units Balance**: 398.08 CU available (burn rate: 0 CU/hr; active sessions: 0 / empty).
- **Inference Gate**: Do not start any L4, A100, or G4 instances for inference until Sahil explicitly gives approval.
