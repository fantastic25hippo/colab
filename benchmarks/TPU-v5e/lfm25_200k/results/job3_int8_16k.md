# Job 3 Report: TPU v5e Context 16K Run (STOP: Cache Miss Abort)

## 1. Executive Summary & Stop Trigger

STOP: driver aborted before LLM(). `LFM_ABORT_ON_CACHE_MISS` defaults to "1" (`lfm25_coldstart_driver.py:242-258`). Cache tar 404 on HF. No retry. No code change.

Session tpu-lfm25: created `--tpu v5e1` in 4.62s; kv_int8.py uploaded to /content; in-VM env LFM_MAX_MODEL_LEN=16384 LFM_KV_INT8=1 LFM_KV_SCALE_PROBE=1; LFM_K_SCALE and LFM_V_SCALE unset. Stopped after abort. `colab sessions`: No active sessions.

## 2. Preflight Checks (P1 - P4)

- P1 PASS: HEAD a44b6cb, no tracked changes.
- P2 PASS: tpu_v5e_patches.py sha256 155db072b3fb75141c836337a35382ff4d3f70fdd505e92bf0495760f2ae87f8
- P3 PASS: run_cpu_tests.sh 9 passed.
- P4 PASS: OFF ...42f6059b.tar ; INT8_16K ...8f6acabe.tar

## 3. Goals & Gates (G1 - G7)

- G1 FAIL: not reached (no `[LFM_KV_INT8] active kv_cache_dtype=fp8`).
- G2 FAIL: not reached (no RPA line).
- G3 FAIL: not reached (no KV page bytes / blocks / GiB).
- G4 FAIL: not reached (no context 16384 needle_pass).
- G5 FAIL: not reached (no CONVDBG).
- G6 FAIL: not reached (no `[LFM_KV_SCALE_PROBE]`).
- G7 FAIL: tar name would end in 8f6acabe.tar but was not published (404 miss, abort).

## 4. Run Metrics (MEASURED)

Metrics:
- session provision 4.62s MEASURED
- venv+zstd download 2.64s MEASURED
- weights 7.95s, 9 shards, 13125.7 MB, 1651.6 MB/s MEASURED
- venv unpack 7.52s MEASURED
- compile time: not reached
- engine-ready time: not reached
- TTFT at 16K: not reached
- decode tok/s: not reached
- HBM after init: not reached
- peak HBM: not reached
- per-layer scales: not reached

## 5. Last log lines

```
=== Job 3 int8 16K launch t0_global=1793642351.72 ===
=== Creating TPU v5e-1 session tpu-lfm25 ===
RUN: colab --auth=oauth2 new -s tpu-lfm25 --tpu v5e1
[colab] Creating session 'tpu-lfm25'...
[colab] Session READY.
=== Session provisioned in 4.62s ===
=== Uploading kv_int8.py to /content/kv_int8.py ===
RUN: colab --auth=oauth2 upload -s tpu-lfm25 /home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/lfm25_200k/kv_int8.py /content/kv_int8.py
[colab] Uploaded '/home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/lfm25_200k/kv_int8.py' to '/content/kv_int8.py'
=== Starting in-VM driver (LFM_MAX_MODEL_LEN=16384 LFM_KV_INT8=1 LFM_KV_SCALE_PROBE=1) ===
[LFM25_DRIVER 17:59:21] In-VM 16384 cold-start timer started at t_vm0=1793642361.70 (t_global0=1793642351.72)
[LFM25_DRIVER 17:59:21] Step 1: Remounting /dev/shm with exec...
[LFM25_DRIVER 17:59:21] Remount finished in 0.01s (rc=0)
[LFM25_DRIVER 17:59:22] Step 2: weights, venv and LFM compile cache all in flight; unpacks after...
[LFM25_DRIVER 17:59:22] [LFM_HF_ANON] download token=none avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU
[LFM25_DRIVER 17:59:22] [LFM_HF_ANON] download token=none tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar
[LFM25_DRIVER 17:59:22] [LFM_HF_ANON] download token=none venv312_cpu.tar.zst
[LFM25_DRIVER 17:59:22] [LFM_HF_ANON] download token=none zstd_bin
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[LFM25_DRIVER 17:59:22] cache miss: tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar
[LFM25_DRIVER 17:59:22] Compile cache tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar: {'hit': False, 'reason': '404 Client Error. (Request ID: Root=1-6ae8cf7a-43dd41164639d7b93812bb5a;c56278bd-e74d-437f-9539-e97cd737ab24)\n\nEntry Not Found for url: https://huggingface.co/avgrocketman/tpu-compile-cache/resolve/main/tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar.', 'cache_dl_s': 0.28, 't_dl_start': 1793642362.2155461, 't_dl_end': 1793642362.499277, 'filename': 'tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar', 'vllm': '0.32.0', 'key_args': 'max_model_len=100000;max_num_seqs=1;max_num_batched_tokens=8192;gpu_memory_utilization=0.92;prefix_caching=0;kv=auto;buckets=16|8192', 'dir': {'entries': 0, 'mb': 0}} (cache_dl_s=0.28, cache_unpack_s=None)
venv312_cpu.tar.zst: reconstructing file:   0%|          |  0.00B / 1.45GB      
zstd_bin: reconstructing file:   0%|          |  0.00B / 1.02MB            
venv312_cpu.tar.zst: downloading bytes:           |  0.00B            
zstd_bin: downloading bytes:           |  0.00B            
Downloading bytes:           |  0.00B            
Reconstructing (incomplete total...): |          |  0.00B /  0.00B            
Fetching 17 files:   0%|          | 0/17 [00:00<?, ?it/s]
[LFM25_DRIVER 17:59:24] Downloaded venv + zstd_bin in 2.64s (no Qwen XLA cache for LFM)
[LFM25_DRIVER 17:59:30] Weights ready in 7.95s (9 shards, 13125.7 MB, 1651.6 MB/s)
[LFM25_DRIVER 17:59:32] Unpacked venv in 7.52s
[LFM25_DRIVER 17:59:32] THP FAIL (sh: 1: cannot create /sys/kernel/mm/transparent_hugepage/enabled: Read-only file system)
[LFM25_DRIVER 17:59:32] ABORT ON CACHE MISS: filename=tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar reason=404 Client Error. (Request ID: Root=1-6ae8cf7a-43dd41164639d7b93812bb5a;c56278bd-e74d-437f-9539-e97cd737ab24)
Entry Not Found for url: https://huggingface.co/avgrocketman/tpu-compile-cache/resolve/main/tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar.
=== COLDSTART_RESULTS_JSONL ===
{"ok": false, "error": "compile cache miss: tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar (reason: 404 Client Error. (Request ID: Root=1-6ae8cf7a-43dd41164639d7b93812bb5a;c56278bd-e74d-437f-9539-e97cd737ab24)\n\nEntry Not Found for url: https://huggingface.co/avgrocketman/tpu-compile-cache/resolve/main/tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar.)", "cache_miss": true, "cache_restore": {"hit": false, "reason": "404 Client Error. (Request ID: Root=1-6ae8cf7a-43dd41164639d7b93812bb5a;c56278bd-e74d-437f-9539-e97cd737ab24)\n\nEntry Not Found for url: https://huggingface.co/avgrocketman/tpu-compile-cache/resolve/main/tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar.", "cache_dl_s": 0.28, "t_dl_start": 1793642362.2155461, "t_dl_end": 1793642362.499277, "filename": "tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar", "vllm": "0.32.0", "key_args": "max_model_len=100000;max_num_seqs=1;max_num_batched_tokens=8192;gpu_memory_utilization=0.92;prefix_caching=0;kv=auto;buckets=16|8192", "dir": {"entries": 0, "mb": 0}}, "filename": "tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar", "reason": "404 Client Error. (Request ID: Root=1-6ae8cf7a-43dd41164639d7b93812bb5a;c56278bd-e74d-437f-9539-e97cd737ab24)\n\nEntry Not Found for url: https://huggingface.co/avgrocketman/tpu-compile-cache/resolve/main/tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar."}
=== END_COLDSTART_RESULTS ===
An exception has occurred, use %tb to see the full traceback.
 [0;31mSystemExit [0m [0;31m: [0m 2
/usr/local/lib/python3.13/dist-packages/IPython/core/interactiveshell.py:3561: UserWarning: To exit: use 'exit', 'quit', or Ctrl-D.
  warn("To exit: use 'exit', 'quit', or Ctrl-D.", stacklevel=1)
=== DRIVER rc=0 ===
=== Downloading /content/build.log ===
RUN: colab --auth=oauth2 download -s tpu-lfm25 /content/build.log /tmp/job3_build.log
[colab] Downloaded '/content/build.log' to '/tmp/job3_build.log'
=== Stopping session tpu-lfm25 ===
RUN: colab --auth=oauth2 stop -s tpu-lfm25
[colab] Stopping session 'tpu-lfm25'...
[colab] Session terminated.
=== Confirming sessions ===
RUN: colab --auth=oauth2 sessions
[colab] No active sessions found on server.
```
