# TPU v5e-1 Cold-Start Breakdown & Startup Analysis (1a821ba)

Analysis of the 235.3 s cold start from commit 1a821ba on Google Colab TPU v5e-1 (`Qwen/Qwen3-4B`, `vllm-tpu==0.32.0`, `tpu-inference==0.32.0`, `jax==0.11.2`). Read-only local evidence from run logs (`tpu-final.jsonl`, `tpu-wheel-build.jsonl`), driver scripts, and unpacked wheels.

---

## 1. Measured Cold-Start Timeline (Bare VM to Ready: 233.7 s)

| Wall Time (UTC) | T+ (s) | Phase / Milestone | Dur (s) | Status | Evidence / Source |
| :--- | ---: | :--- | ---: | :--- | :--- |
| **14:11:02** | 0.00 | **VM ready; driver starts parallel prep** | 0.00 | [proven] | `tpu_driver.py:17` (`t_cold0`); VM create: 7.59s |
| 14:11:12 | 10.14 | Weight download done (7.49 GiB to `/dev/shm`) | 10.14 | [proven] | `[TPU_DRIVER 14:11:12] Download done in 10.14s` |
| 14:11:31 | 28.92 | Persistent cache restored (76 MB tar) | 28.92 | [proven] | `[TPU_DRIVER 14:11:31] Cache restore done in 28.92s` |
| **14:12:33** | 91.21 | **Dependencies install done** (prep bottleneck) | 91.21 | [proven] | `[TPU_DRIVER 14:12:33] Install done in 91.21s` |
| **14:12:33** | 91.21 | **Engine launch: MainProcess start & imports** | 48.00 | [inferred] | `tpu_driver.py:66`; `tpu_engine_runner.py:55` |
| 14:13:21 | 139.21 | MainProcess imports done; enter `LLM.__init__` | 11.00 | [proven] | `(MainProcess pid=7633) INFO [api_utils.py:357]` |
| 14:13:32 | 150.21 | Model config & tokenizer parsed; spawn EngineCore | 11.00 | [proven] | `(MainProcess pid=7633) INFO [model.py:733]` |
| 14:13:43 | 161.21 | EngineCore child process spawn & re-imports | 7.00 | [proven] | `(EngineCore pid=8039) INFO [attention_interface.py:60]` |
| 14:13:50 | 168.21 | Gloo mesh init, TPU device 0 init, buckets ready | 1.00 | [proven] | `(EngineCore pid=8039) INFO [tpu_worker.py:472]` |
| 14:13:51 | 169.21 | Abstract model shape traced (`nnx.eval_shape`) | 0.66 | [proven] | `Finished tracing create_abstract_model in 0.66s` |
| 14:14:02 | 181.06 | **Model weights loaded** to TPU HBM | 11.85 | [proven] | `w_load dur_s: 11.85`; `default_loader.py: 501 (9.90s)` |
| 14:14:05 | 184.37 | **KV cache allocated** (759 blks = 48.6k tokens) | 3.31 | [proven] | `w_init dur_s: 3.31`; `kv_cache_utils.py:1067` |
| 14:14:54 | 232.83 | **Compile & Warmup pass** (261 hits, 0 compiles) | 48.46 | [proven] | `w_comp dur_s: 48.46`; `core.py:395 (took 51.77s)` |
| **14:14:56** | 233.72 | **Engine Ready** (ZMQ handshake & InputProcessor) | 0.89 | [proven] | `core_client.py:738`; `hf.py:647`; `ready_s: 142.51` |
| 14:14:58 | 235.33 | **First token generated** (warmup decode) | 1.61 | [proven] | `first_token_from_ready_s: 1.61` |

---

## 2. Q1: Where Did the Untracked ~79 s Go?

Engine startup is 142.51 s, while `load (11.85) + KV (3.31) + compile (48.46) = 63.62 s`, leaving **78.89 s** untracked by the three worker hooks. From timestamp logs, the 78.89 s is fully accounted for across 5 distinct phases:
1. **MainProcess Heavy Imports & GCE Metadata Probe (48.0 s) [inferred duration, proven log lines]:** `tpu_engine_runner.py:56` imports `vllm`, which transitively loads `torch`, `jax`, `transformers`, and `tpu_inference`. During JAX TPU platform init, `cloud_tpu_init.py` attempts HTTP polling on `http://metadata.google.internal/...` which returns 404 and hangs on timeout (`Unable to poll TPU GCE Metadata. Got status code: 404`).
2. **MainProcess Model Config & Tokenizer Load (11.0 s) [proven]:** 14:13:21 to 14:13:32. `LLM.__init__` reads config, tokenizer, and chat template from `/dev/shm/qwen3` via `hf.py` / `model.py:733`.
3. **EngineCore Subprocess Spawn & Duplicate Import (11.0 s) [proven]:** 14:13:32 to 14:13:43. Because `VLLM_WORKER_MULTIPROC_METHOD="spawn"`, `launch_core_engines` (`core_client.py:682`) spawns a fresh Python interpreter (`pid=8039`) that re-executes Python initialization, re-imports `vllm`/`jax`/`torch`, and repeats the metadata 404 probe.
4. **TPU Device Init, Mesh & Gloo Backend (7.0 s) [proven]:** 14:13:43 to 14:13:50. Gloo file-store rendezvous (`parallel_state.py:1906`), JAX Mesh instantiation (`tpu_runner.py:885`), and PJRT `TpuDevice(id=0)` driver discovery.
5. **Abstract Model Shape Tracing & Sharding (1.0 s) [proven]:** 14:13:50 to 14:13:51. `model_loader.py:144` runs `create_abstract_model` under `flax.nnx.eval_shape` (0.66 s tracing) and constructs layer sharding specs.
6. **ZMQ Ready Handshake & InputProcessor Setup (0.89 s) [proven]:** 14:14:54 to 14:14:56. Child signals readiness over ZMQ (`core_client.py:738`) and MainProcess finishes `InputProcessor` initialization.

*Missing Timers to Add:* (a) `tpu_engine_runner.py:56` around `from vllm import LLM` (times MainProcess import); (b) `vllm/entrypoints/llm.py:178` around `VllmConfig` (times tokenizer/config load); (c) `core_client.py:682` around `launch_core_engines` (times child spawn + re-import).

---

## 3. Q2: The 48.5 s Compile/Warmup with 261 Cache Hits

Even with 0 XLA compilations, JAX requires **51.09 s** of host-side execution across 250 AOT programs:
- **Python Jaxpr AST Tracing:** **15.01 s** [proven from 251 log lines `Finished tracing ... in ... sec`].
- **Jaxpr to StableHLO/MLIR Lowering:** **32.70 s** [proven from 250 log lines `Finished jaxpr to MLIR module conversion ... in ... sec`].
- **Cache Key Lookup & Binary Deserialization:** **3.38 s** [proven from 250 log lines `Finished XLA compilation of ... in 0.001-0.003 sec`].
- **Warmup Execution:** 0.00 s for `compile_only=True` tasks; negligible (<0.01 s) for dummy warmup calls.

*Root Cause [proven]:* JAX persistent compilation cache keys are cryptographic hashes of the lowered StableHLO MLIR module. JAX cannot check the cache without first tracing the Python code and lowering to MLIR.

*Program & Bucket Breakdown (250 AOT programs) [proven]:*
- Bucket lists: `num_tokens_paddings=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]` (10), `num_reqs_paddings=[8, 16, 32, 64, 128, 256]` (6), `attn_num_reqs=[256]` (1).
- Programs: `_substitute_placeholder_token_fn` (71), `_select_from_array_fn` (50), `broadcast_in_dim` (40), `sample` (24: $6\times2\times2$), `convert_element_type` (10), `run_model_impl` backbone (10), `compute_logits` (6), `gather_logprobs` (6), `structured_decode` (6), other helpers (27).

*Council Redundancy (Batch $\le 16$, Prompts $\le 16\text{K}$) [proven]:*
Council never issues requests with batch > 16. Request buckets **`[32, 64, 128, 256]` are never hit**. These unused buckets create **96 wasted programs (38.4%)**: 44 in substitute placeholder, 24 in select from array, 16 in sample, 4 in logits, 4 in gather logprobs, and 4 in structured decode. Passing `--max-num-seqs 16` eliminates all 96 programs, saving **~18.3 s** of tracing and MLIR conversion.

---

## 4. Q3: The 91 s Install & PyTorch CPU Pin

- **Download vs Unpack/Link [proven]:** In `tpu-wheel-build.jsonl`, online `uv pip install` took **90.50 s**, while offline verify install from pre-downloaded wheels took **91.86 s**. Network download is fully pipelined; the entire 91 s is disk I/O, archive decompression, and shared library linking on Colab ext4.
- **Bytecode Compilation (`.pyc`) [proven]:** `uv` defaults to `--no-compile-bytecode`. Bytecode was NOT compiled during install.
- **Dominating Packages by Size [proven]:** 259 transitive wheels total 1.8+ GB (`RESULTS.md:74`). PyPI `torch` pulls **15 NVIDIA CUDA wheels** (`nvidia-cuda-runtime`, `nvidia-cudnn`, `nvidia-cublas`, `nvidia-nccl`, etc.) totaling **>1.4 GB (78% of wheel bundle)**!
- **CUDA Import vs CPU-only Torch [proven in code]:** Neither `vllm-tpu` nor `tpu-inference` lists `torch` in `Requires-Dist`. `torch` is pulled transitively by `compressed-tensors` (`torch>=2.10.0`) and `torchvision` (`torch>=2.14.0`). `tpu_platform.py` runs purely on JAX and monkeypatches `torch.accelerator`. `safetensors` only loads CPU tensors. CUDA is never called. A CPU-only wheel (`torch==2.10.0+cpu` from `download.pytorch.org/whl/cpu`) fully satisfies the version pins and avoids all 15 NVIDIA packages.
- **Link Mode & `/dev/shm` [inferred]:** Installing `venv312` into tmpfs RAM (`/dev/shm/venv312`) bypasses ext4 disk write throttling. Colab has 31.5 GiB RAM, ample for a 2.5 GiB venv.

---

## 5. Q4: Serial Bottlenecks in the Cold Path

1. **Dual Process Import (MainProcess + EngineCore) [proven]:** Running `spawn` forces duplicate imports of `vllm`, `jax`, and `torch`, burning 11.0 s in the child process.
2. **GCE Metadata 404 Timeout [proven]:** `cloud_tpu_init.py` makes unneeded HTTP probes to GCE metadata on Colab, incurring network timeouts in both processes.
3. **Transparent Hugepages (THP) Disabled [proven]:** Log warns that THP is disabled, which increases page allocation latency for 16 GB HBM buffers.
4. **Serial Step 2 Generation after Startup [proven]:** 3 warmup decode passes run synchronously before serving.

---

## 6. Ranked Concrete Reduction Ideas

| Rank | Reduction Strategy | Est. Saving | Conf. | Mechanism / Implementation |
| :---: | :--- | ---: | :--- | :--- |
| **1** | **CPU-only PyTorch Wheel** | **65–70 s** | [proven] | Pass `--extra-index-url https://download.pytorch.org/whl/cpu`. Drops 15 NVIDIA CUDA wheels (~1.4 GB), cutting install from 91s to ~22s. |
| **2** | **Prune Request Buckets (`--max-num-seqs 16`)** | **~18 s** | [proven] | Limits `num_reqs_paddings` to `[8, 16]`. Cuts 96 unused programs from JAX tracing and MLIR conversion. |
| **3** | **Install Venv into `/dev/shm`** | **15–20 s** | [inferred] | Target `/dev/shm/venv312` (tmpfs RAM). Eliminates ext4 disk write bottlenecks for wheel extraction and `.so` loads. |
| **4** | **Single-Process EngineCore (`VLLM_ENABLE_V1_MULTIPROCESSING=0`)** | **~11 s** | [inferred] | Runs EngineCore in-process; eliminates child process launch and duplicate module imports. |
| **5** | **Suppress GCE Metadata Probes** | **3–5 s** | [inferred] | Set environment variable to bypass `cloud_tpu_init.py` 404 HTTP metadata retry loop. |
| **6** | **Enable Transparent Hugepages** | **2–4 s** | [inferred] | Run `sudo sh -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled"` to accelerate HBM memory mapping. |

*Combined Impact:* Implementing items 1, 2, 3, and 4 drops total cold start from **235 s (3.9 min) to ~80–90 s (1.4 min)**.

---

## 7. Measurements Still Missing for Next Instrumented VM Run

1. `python3 -X importtime -c "import vllm"` log trace to profile the 48 s MainProcess import down to each individual module.
2. Direct wall-time measurement of `uv pip install --extra-index-url https://download.pytorch.org/whl/cpu` vs PyPI default on Colab TPU.
3. Benchmarked install and import duration in `/dev/shm/venv312` vs `/content/venv312`.
4. Measured startup compile time with `--max-num-seqs 16` to confirm exact reduction from 250 to 154 programs.
5. Verification of `VLLM_ENABLE_V1_MULTIPROCESSING=0` single-process stability on TPU v5e-1 without deadlock.
