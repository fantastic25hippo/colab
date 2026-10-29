# Universal TPU v5e-1 Serving Recipe (vLLM / tpu-inference)

Final, model-independent runbook for deploying any LLM on Google Colab TPU v5e-1 (16 GB HBM2; 15.748 GiB driver limit).

---

## 1. Memory Budget Formula

At `gpu_memory_utilization=0.90`, vLLM allocates up to **14.17 GiB**. Net runtime overhead is **~0.38 GiB (380 MiB)**:

$$\text{Weights (GiB)} + \text{KV Cache (GiB)} \le \mathbf{13.8\text{ GiB}} \quad (\text{or } \mathbf{14.6\text{ GiB at } 0.95})$$

- **Weights Footprint:** $N_{\text{params}} \times \text{bytes/param}$ (e.g. 4B BF16 $\approx$ 7.49 GiB; 7B W4A16 $\approx$ 4.5 GiB).
- **KV Cache Footprint:** $\text{Tokens} \times (2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{dtype\_bytes})$.
  - Example (Qwen3-4B): $2 \times 36 \times 8 \times 128 \times 2 = 144.0\text{ KiB/token} \rightarrow 48,576\text{ tokens} = 6.695\text{ GiB}$ (at 0.90) or $54,336\text{ tokens} = 7.485\text{ GiB}$ (at 0.95).
- **Recommendation:** Use `gpu_memory_utilization=0.90` for multi-request serving / Batch 32 to absorb transient activation spikes. `0.95` is viable for single-stream / low-batch serving (+11.8% KV tokens, 406 MiB headroom).

---

## 2. Copy-Paste Fresh VM Bootstrap (Cold Start: ~3.9 min)

Run the three preparation tasks in parallel on a fresh VM:

```bash
# A. Create Python 3.12 venv and install TPU packages (takes ~90s)
uv venv --python 3.12 /content/venv312
uv pip install --python /content/venv312/bin/python \
    vllm-tpu==0.32.0 tpu-inference==0.32.0 hf-transfer huggingface_hub

# B. Download model weights to tmpfs RAM (/dev/shm) via hf_transfer (takes ~10s for 7.5 GB)
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c '
from huggingface_hub import snapshot_download
snapshot_download("ORG/MODEL", local_dir="/dev/shm/model", max_workers=12)
'

# C. Restore prebuilt persistent compilation cache (takes ~25s)
python3 -c '
from huggingface_hub import hf_hub_download; import subprocess
p = hf_hub_download(repo_id="avgrocketman/tpu-compile-cache", filename="tpu_compile_cache.tar", local_dir="/content")
subprocess.run(["tar", "-xf", p, "-C", "/content"], check=True)
'
```

---

## 3. Mandatory Weight Loading Speedup Patch

Neutralize redundant per-module `jax.clear_caches()` to drop load time from ~30s to **11.8s**:

```python
# Save as /content/hooks/sitecustomize.py and export PYTHONPATH=/content/hooks
import builtins, sys
orig_imp = builtins.__import__
def custom_imp(name, *args, **kwargs):
    res = orig_imp(name, *args, **kwargs)
    mod = sys.modules.get("tpu_inference.models.jax.utils.weight_utils")
    if mod and hasattr(mod, "jax"): mod.jax.clear_caches = lambda: None
    return res
builtins.__import__ = custom_imp
```

---

## 4. Engine Environment Flags & Serving

Set environment variables before starting vLLM:

```bash
export PYTHONPATH="/content/hooks"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export BENCH_LOAD_FIX="1"
export JAX_COMPILATION_CACHE_DIR="/content/tpu_compile_cache"
export VLLM_XLA_CACHE_PATH="/content/tpu_compile_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0"
```

Initialize engine in Python:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/dev/shm/model",
    max_model_len=17408,           # match workload or cache key
    gpu_memory_utilization=0.90,   # 0.90 for high batching; 0.95 for low batching
    trust_remote_code=True,
)
```

---

## 5. Building & Reusing the Persistent Compile Cache for a NEW Model

JAX XLA compilation on TPU v5e is 100% deterministic based on model layer shapes and flags.

1. **One-Time Paid Warm-up Run:** Start engine with `JAX_COMPILATION_CACHE_DIR=/content/tpu_compile_cache`. Allow default AOT precompile and send 1 warmup request per target batch size (1, 4, 8, 16, 32).
2. **Package Archive:** `tar -cf /content/tpu_compile_cache.tar -C /content tpu_compile_cache`.
3. **Upload & Key:** Deposit `tpu_compile_cache.tar` and `versions.json` (recording `max_model_len` and package pins) to your HF/GCS repo.
4. **Reuse:** On all future fresh VMs, restoring this tarball drops startup compilation from 147s to **48s** (0 new compiles, 100% cache hits).

---

## 6. Recommendations for Multi-Agent Council Workloads

- **Optimal Batch Size:** **16 to 32 concurrent requests**.
  - Batch 1: **84.8 tok/s** aggregate (84.9 tok/s/req, 16 ms TTFT).
  - Batch 8: **596.5 tok/s** aggregate (75.2 tok/s/req, 37 ms TTFT).
  - Batch 16: **1,050.1 tok/s** aggregate (66.2 tok/s/req, 40 ms TTFT).
  - Batch 32: **1,653.4 tok/s** aggregate (52.6 tok/s/req, 103 ms TTFT, 4.95s wall).
- **Context Capacity:** Handled **8 concurrent 8k** ($65,536$ tokens) and **4 concurrent 16k** ($65,536$ tokens) in 6.4s / 6.5s with **0 preemptions and 0 queued requests** via Chunked Prefill (`max_num_batched_tokens=8192`).
- **Cold Start Expectation:** **235 s (3.9 min)** from bare VM create to first generated token.

---

## 7. Known Traps to Avoid

1. **Never set `SKIP_JAX_PRECOMPILE=1` or restrict `compilation_sizes`:** Forces uncompiled shapes to JIT-compile on the fly during generation, collapsing decode speed from 85 tok/s down to 8–30 tok/s.
2. **Always warm up before timing:** Run 3 untimed requests to populate PJRT device buffers.
3. **No per-token execution hook I/O:** Synchronous disk writes inside forward steps throttle token delivery.
4. **Use Python 3.12 exclusively:** TPU inference packages are pinned to Python 3.12; Colab default Python 3.13 is incompatible.
5. **Do not use venv tarballs or offline wheel bundles:** Full venv tarball takes 201s to extract; 259-wheel offline bundles take 92s to unpack (identical to online `uv pip install`). Online `uv pip install` into tmpfs is the fastest and cleanest path.
