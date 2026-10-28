# LFM & L4 Colab Sweep Handoff Failures & Fixes

Summary of failures encountered during LFM sweeps on CPU and L4 instances, along with required operational fixes.

- **llama.cpp build failure on CPU Colab**:
  - *Failure*: GitHub `/releases/latest` API query failed because asset `bin-ubuntu-x64.zip` was not present, causing fallback to cloning and compiling llama.cpp with `cmake` on Colab CPU.
  - *Fix*: Always pin b11954: pin `llama-b11954-bin-ubuntu-x64.tar.gz` and `llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz`. Never query `/releases/latest`. Never run `cmake` or build from source on Colab.

- **Timeout race condition between grok-agy and colab CLI**:
  - *Failure*: `grok-agy --wait-secs 1800` equaled `colab --timeout 1800`. The job was terminated mid w=1 benchmark (job `20261028-163723-lfm-q4-shard-sweep-1db5`).
  - *Fix*: Ensure `wait_secs > timeout` strictly (e.g., set `--wait-secs 2400` vs `--timeout 1800`).

- **HF_TOKEN leakage via command line arguments**:
  - *Failure*: Passing `HF_TOKEN` in `colab run` command arguments leaked the secret into `events.jsonl` `CommandLine`.
  - *Fix*: Follow no-token-on-argv strictly. Never pass `HF_TOKEN` on argv. Upload the token via `colab upload _tasks/hf_token_upload hf_token -s SESSION` (relative remote name `hf_token`). Note that sandbox blocks `/tmp` and `/content/...` paths and `python -c`.

- **Log buffering and false hang assumptions**:
  - *Failure*: `colab log` remains empty until execution terminates (flush-at-end behavior), leading to assumptions that the script was hanging.
  - *Fix*: An empty log is not proof of a hang. Benchmark scripts must emit a heartbeat 20s (`heartbeat ~20s`) to ensure progress tracking.

- **Leftover active L4 session idle wait**:
  - *Failure*: Job `20261028-180811-lfm-l4-more-4cac` sat ACTIVE for ~16 min without an EXEC line until stopped manually with SIGTERM/KILL (exit 124), yielding no second RESULTS.
  - *Fix*: Do not idle-poll 20+ minutes waiting for execution without progress; inspect early and terminate orphaned sessions.

- **Unsuitable network hardware for sharding sweeps**:
  - *Failure*: CPU Colab network throughput (~98 MB/s) throttles shard and worker benchmarking.
  - *Fix*: CPU Colab NIC ~98 MB/s is the wrong machine for shard/worker sweeps. Use L4 GPU instances (~250 MB/s).

- **llama-server CUDA runtime version mismatch**:
  - *Failure*: Using llama-server with CUDA 12.8 against driver 13.0 caused `/health` `ready=False` timeout at 180s.
  - *Fix*: Pin b11954 with CUDA 13.4 (`llama-b11954-bin-ubuntu-cuda-13.4-x64.tar.gz`) matching driver requirements.

- **Missing CLI permissions in grok-agy runner**:
  - *Failure*: `grok-agy --tools code` denies execution of `colab` CLI commands.
  - *Fix*: Always pass `--tools code --allow-cmd colab` when running workflows requiring `colab` CLI.

- **CPU benchmark indexing classification**:
  - *Failure*: Tentative or partial CPU runs risking confusion with validated results.
  - *Fix*: The INDEX CPU TBD row must never be treated as MEASURED. Only verified, completed runs count as MEASURED.

- **Compute Unit (CU) spend governance**:
  - *Failure*: Risk of running up GPU CU costs without explicit confirmation.
  - *Fix*: Spend 0 CU without explicit user go. Keep sessions empty (0.00 CU/hr). Always run `colab stop` immediately after job completion.
