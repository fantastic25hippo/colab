fantastic25hippo/colab (public) holds Sahil's notes for running open-source models on Google Colab GPUs and TPUs; work on the main branch only.
Goal: Sahil downloads a model, asks the agent to bench it, and it benches on Colab with no trial and error; fast cold-start and single-stream decode for coding work is the current focus.
Colab is billed against a Google AI Pro compute-unit (CU) pool; `colab usage` read ~400.00 CU starting balance (398.80 on 2026-10-27; 391.95 on 2026-10-29), so check `colab usage` before and after every run.
This repo is public: never put keys, tokens, the OAuth token file, memory contents or any Google account email in it.
Layout: README.md (focus and index), ops.md (Colab CLI, weights loading, tunnel, heartbeat, quota), hardware.md (the 7 accelerators with measured CU/hr), research/ (00-INDEX..11 consolidated notes + 2026_model_research), benchmarks/ (measured results).
Read benchmarks/README.md (and benchmarks/TPU-v5e/RECIPE.md for TPU) before benching anything. Results go in benchmarks/<Model>/<GPU>/RESULTS.md (or benchmarks/TPU-v5e/<run>/RESULTS.md) plus one row in benchmarks/INDEX.md.
Research files in research/ are estimates unless cited from benchmarks/; only numbers marked MEASURED in benchmarks/ came from a real run.
The Colab CLI is `colab` (google-colab-cli 0.7.5); put the global `--auth=oauth2` flag before the subcommand. `exec` times out after 30 s by default, so pass `--timeout 86400` and emit a ~20 s stdout heartbeat.
Never use `colab drivemount` (interactive on every new VM) or GCS buckets (no billing account). Pull sharded weights from Hugging Face Hub directly into `/dev/shm` (`snapshot_download`, `max_workers=8..12`); see docs/WEIGHTS-LOADING.md.
On TPU v5e-1 (2.92 CU/hr, 15.748 GiB HBM, 23 GB `/dev/shm`), use `mount -o remount,exec /dev/shm`, unpack prebuilt `venv312_cpu.tar.zst` (`torch==2.10.0+cpu`, `vllm-tpu==0.32.0`) + persistent XLA cache from `avgrocketman/tpu-compile-cache`, set `TPU_SKIP_MDS_QUERY=1` topology vars and `VLLM_ENABLE_V1_MULTIPROCESSING=0`, and apply the `jax.clear_caches` loader patch.
Always stop the Colab runtime when done (`colab --auth=oauth2 stop -s <session>`) and check that `colab --auth=oauth2 sessions` is empty; an idle runtime keeps burning CU.
Shared private OptMem store: fantastic25hippo/optmem-memory, synced by grok-agy-memo; nothing from it is stored in this repo. See docs/OPTMEM.md.
Keep every new file under 150 lines and every commit email on the GitHub noreply form.
