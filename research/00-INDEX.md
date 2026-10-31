# Research Master Index — Colab, GPU and TPU Notes

Consolidated 2026-10-31 from the top-level research files dated 2026-10-27 to 2026-10-29. This index plus the 12 master documents below replace those files. `README.md`, `precision_format/` and `2026_model_research/` were left untouched.

## Start here

| If you want to… | Read |
| :--- | :--- |
| Understand Colab hardware, RAM, `/dev/shm`, compute-unit costs and what fits where | [`01-Colab-Platform-Hardware-and-Costs.md`](01-Colab-Platform-Hardware-and-Costs.md) |
| Choose where to keep weights: HF Hub, Drive, Kaggle, GCS, R2, snapshots; auth without popups | [`02-Storage-Channels-and-Auth.md`](02-Storage-Channels-and-Auth.md) |
| Tune Hugging Face downloads: Xet, workers, shards, variance, rate limits, pricing | [`03-HuggingFace-Xet-Tuning.md`](03-HuggingFace-Xet-Tuning.md) |
| Load 15–75 GB weights on an L4: `/dev/shm`, disk, Run:ai streamer, Tensorizer, precision, GGUF | [`04-L4-Loading-Routes-and-Precision.md`](04-L4-Loading-Routes-and-Precision.md) |
| Automate `google-colab-cli`: secrets, hazards, SSH tunnels, ranked blueprints, timelines | [`05-Colab-CLI-Automation.md`](05-Colab-CLI-Automation.md) |
| Run an agent harness on Colab: Council, warm daemon, auth, compute-unit math, pipelining | [`06-Agent-Workflows-Council.md`](06-Agent-Workflows-Council.md) |
| Run vLLM or TorchTPU on a single TPU v5e-1: memory, XLA, flags, KV cache, traps | [`07-TPU-v5e-Runtime.md`](07-TPU-v5e-Runtime.md), [`07b-TPU-v5e-vLLM-and-TorchTPU.md`](07b-TPU-v5e-vLLM-and-TorchTPU.md) |
| Quantize for TPU v5e: W4A16 vs W8A8, support matrix, issues, benchmarks, flags | [`08-TPU-v5e-Quantization.md`](08-TPU-v5e-Quantization.md) |
| Cut vLLM-on-TPU cold start from 235 s: install, imports, precompile buckets, cache | [`09-TPU-vLLM-Cold-Start.md`](09-TPU-vLLM-Cold-Start.md) |
| Compare LFM2.5-24B-A2B checkpoints and estimate L4/TPU speeds | [`10-LFM2.5-24B-A2B.md`](10-LFM2.5-24B-A2B.md) |
| See contradictions, stale claims and what still needs live verification | [`11-Conflicts-and-Verification.md`](11-Conflicts-and-Verification.md) |

## Master documents

| # | File | Covers | Replaces |
| :-- | :--- | :--- | :--- |
| 01 | `01-Colab-Platform-Hardware-and-Costs.md` | Runtime tiers, CU burn rates, ephemerality, `/dev/shm`, FUSE failure, bandwidth physics, 75 GB tier qualification | Fast Headless §§Exec–4; Google AI Pro hardware parts |
| 02 | `02-Storage-Channels-and-Auth.md` | HF/Drive/Kaggle/GCS/R2/drivemount/upload/snapshot comparison, GCS costs, credentials, 20-pulls/day | Fast Headless §§1–2; Google AI Pro; “Read both” storage table |
| 03 | `03-HuggingFace-Xet-Tuning.md` | Xet architecture, worker tuning, deprecated flags, shard geometry, variance, quotas, pricing | `Hugging Face Download Optimization Research.pdf` + `HuggingFace-Optimization-Summary.md` |
| 04 | `04-L4-Loading-Routes-and-Precision.md` | L4 topology, three loading routes, 15/75 GB math, comparison table, FP8/AWQ/GPTQ/BF16, GGUF | `L4 GPU Model Loading Comparison.pdf` + `Summary.md` |
| 05 | `05-Colab-CLI-Automation.md` | `colab` auth, secret injection, six hazards, v6e timeout, ranked architectures, scripts, timelines | Fast Headless §§5–8 |
| 06 | `06-Agent-Workflows-Council.md` | Council harness, rclone/service account, warm daemon, compute math, pipelining, Drive quota | `Optimizing AI Agent Colab Workflows.pdf`; “Read both” harness parts |
| 07 | `07-TPU-v5e-Runtime.md` | v5e hardware, memory budget, XLA overhead, KV cache, acceleration, checklist | `TPU-V5e-Best-Practices.md` |
| 07b | `07b-TPU-v5e-vLLM-and-TorchTPU.md` | vLLM/TorchTPU install, launch flags, execution model, compile cache, traps | `TPU-V5e-Best-Practices.md` |
| 08 | `08-TPU-v5e-Quantization.md` | Format support matrix, issue #3540, W4A16 vs W8A8, benchmarks, KV warnings, version risks | `TPU v5e LLM Quantization Guide.pdf` + `TPU-V5e-LLM-Quantization-Guide.md` |
| 09 | `09-TPU-vLLM-Cold-Start.md` | 235 s breakdown, CPU torch, prebuilt venv, MDS polling, in-process engine, bucket pruning | `Accelerating vLLM Startup on TPU.pdf` + `.md` |
| 10 | `10-LFM2.5-24B-A2B.md` | Three checkpoint audit, L4 speed estimates, speculative decoding, TPU v5e/v6e feasibility | `LFM2.5-24B-A2B GPU Checkpoint Comparison.pdf` + `LFM2-5-24B-A2B-Summary.md` |
| 11 | `11-Conflicts-and-Verification.md` | Cross-document contradictions, stale claims, measured-vs-estimated, verification checklist | All of the above |

## Epistemic labels used in these notes

- **[M] Measured** — a number from a real run, fio/iperf3, or a vendor benchmark quoted as measured.
- **[I] Inferred** — derived from specs or measured rates; arithmetic, not a direct run.
- **[G] Guess / unverified** — planning estimate, community claim, or unconfirmed extrapolation.
- **[C] Confirmed** — backed by a primary source (GitHub issue/PR, official docs) in the original.
- **Reported** — a claim from a Gemini conversation that was not independently checked here.

Prefer **[M]** over **[I]** over **[G]**. The original documents mixed these labels; this set preserves them where they were explicit.

## Key conclusions at a glance

- **Colab VMs are stateless and wiped on stop.** Every `colab new` starts an empty container; there is no supported disk snapshot or persistent volume. Weights must be re-fetched or re-mounted every session.
- **The fastest free source is usually Hugging Face Hub, with Kaggle Models as backup.** HF Xet is peered with GCP; Kaggle resolves to signed GCS URLs. Google Drive is convenient but rate-limited and slow for 40–80 GB; GCS is fast but cross-region egress can be expensive.
- **On an L4, 15 GB is easy; 75 GB is not.** 15 GB fits `/dev/shm` (26 GB), host RAM (52 GB) and 23 GB VRAM. 75 GB exceeds VRAM 3.3×, `/dev/shm` 2.9× and host RAM 1.4×. Use pre-quantized weights or direct streaming; do not download 75 GB to `/dev/shm`.
- **The fastest L4 path without a paid bucket is sharded HF Hub → `/dev/shm`.** Measured at **16.60 s (896 MB/s)** for 14.88 GB (16 shards, `max_workers=12`; 3-rep mean **920.8 MB/s** in [`../benchmarks/HF-transfer/L4/SWEEP.md`](../benchmarks/HF-transfer/L4/SWEEP.md)), though Hub throttling can push slow reps to 75–147 s. With a co-located GCS bucket, Run:ai Model Streamer is reported at ~15–23 s for 15 GB.
- **On TPU v5e-1, `Qwen/Qwen3-4B` bf16 is measured at 84.92–88.58 tok/s (`B=1`) and 1,653.4 tok/s (`B=32`).** Prebuilt CPU-torch venv (`venv312_cpu.tar.zst`) + parallel weight download into `/dev/shm` (`mount -o remount,exec /dev/shm`) + `TPU_SKIP_MDS_QUERY=1` + `VLLM_ENABLE_V1_MULTIPROCESSING=0` + `jax.clear_caches` loader patch cuts non-compile startup from 185.2 s to **25.8 s** ([`../benchmarks/TPU-v5e/FAST-COLDSTART.md`](../benchmarks/TPU-v5e/FAST-COLDSTART.md)). For quantized models, W8A8 INT8 uses the native 393 TOPS INT8 MXU (reported ~1.4–1.7× faster than bf16 [I]); `--kv-cache-dtype fp8` may silently allocate bf16.
- **LFM2.5-24B-A2B on TPU v5e-1 at 200K context: Hybrid W8A8 Backbone (`0.66 GiB`) + W4A16 MoE (`10.69–12.02 GiB`) + K8V4 In-VMEM KV (`1.49–1.91 GiB`) fits in 15.75 GiB.** Pure W8A8 (`22.32 GiB`) and BF16 KV at 200K (`3.81 GiB`) OOM, and naive PyTorch KV dequant spills 410 MB/layer to HBM. Losslessly repacking `LFM2.5-24B-A2B-QAD-Q4_0.gguf` into mixed `compressed-tensors` (W8A8 INT8 on the 40 attention/conv/dense-FFN layers for 393 TOPS INT8 MXU + W4A16 on the 38 MoE layers) plus patching RPA v3 / Pallas for in-VMEM `K8V4` (`Q_int8 @ K_int8^T` on INT8 MXU) and XLA `ShortConv` runs 200K at **13.83–15.00 GiB** total HBM ([`10-LFM2.5-24B-A2B.md`](10-LFM2.5-24B-A2B.md)).
- **Compute units are the hidden cost.** G4 burns 8.90 CU/h; 200 CU/month is ~22.5 h (~45 min/day). Twenty 4.5-minute cold starts per day burn ~13.35 CU/day and exhaust 200 CU in ~15 days. Keep the VM warm when startup time exceeds debate/idle time.

## Where the old documents disagreed

The main conflicts are: HF `hf_transfer` vs Xet flags; Drive throughput; whether G4 standard RAM is 12.7 GB or 176 GB; whether `gcsfuse` is usable for mmap loading; XLA overhead 1.26 GB vs 4.5–6.5 GB; TPU `int8` vs `fp8` KV cache; A100/T4 CU rates; and 80 GB HF download time. See [`11-Conflicts-and-Verification.md`](11-Conflicts-and-Verification.md) for the side-by-side and which claim to trust.

## Cleanup note

The old top-level `.md` and `.pdf` files were removed after their content was consolidated here. Duplicate pairs removed: `Summary.md` (L4 PDF), `HuggingFace-Optimization-Summary.md` (HF PDF), `LFM2-5-24B-A2B-Summary.md` (LFM PDF), and the `.md`/`.pdf` pairs for Accelerating vLLM, TPU v5e quantization and Fast Headless. `README.md` and both subfolders were not modified.
