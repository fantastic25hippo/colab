# General TPU v5e-1 Serving Recipe (vLLM / tpu-inference)

Model-independent runbook for deploying any LLM on a Google Colab TPU v5e-1 chip (16 GB HBM2; 15.748 GiB driver limit).

---

## 1. Memory Budget Rule

At `gpu_memory_utilization=0.90`, vLLM allocates up to **14.17 GiB**. Net runtime overhead is **~0.38 GiB (380 MiB)**.

$$\text{Weights (GiB)} + \text{KV Cache (GiB)} \le \mathbf{13.8\text{ GiB}}$$

- **Weight Footprint:** $N_{\text{params}} \times \text{bytes/param}$ (e.g. 4B BF16 $\approx$ 7.49 GiB; 7B W4A16 $\approx$ 4.5 GiB).
- **KV Cache Footprint:** $\text{Tokens} \times (2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{dtype\_bytes})$.
  - Example (Qwen3-4B): $2 \times 36 \times 8 \times 128 \times 2 = 144.0\text{ KiB/token} \rightarrow 48,576\text{ tokens} = 6.69\text{ GiB}$.
- **Usable KV Tokens:** $\text{Max Tokens} = \frac{(14.17 - \text{Weight GiB}) \times 1024^3}{\text{Bytes per Token}}$.

---

## 2. Environment Setup & Fast Cold Start

Run in parallel on a fresh VM:

```bash
# A. Create Python 3.12 venv and install pinned TPU wheels
uv venv --python 3.12 /content/venv312
uv pip install --python /content/venv312/bin/python \
    vllm-tpu==0.32.0 tpu-inference==0.32.0 hf-transfer huggingface_hub

# B. Download model weights to tmpfs RAM (/dev/shm)
HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c '
from huggingface_hub import snapshot_download
snapshot_download("ORG/MODEL", local_dir="/dev/shm/model", max_workers=12)
'

# C. Restore prebuilt persistent compilation cache (if available)
tar -xf /content/tpu_compile_cache.tar -C /content
```

---

## 3. Weight Loading Speedup Patch

`tpu-inference` redundantly calls `jax.clear_caches()` after each layer. Neutralizing this drops load time from ~30s to ~11s:

```python
# Place in /content/hooks/sitecustomize.py and export PYTHONPATH=/content/hooks
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

## 4. Recommended Engine Flags & Execution

Set environment variables before starting vLLM:

```bash
export PYTHONPATH="/content/hooks"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export JAX_COMPILATION_CACHE_DIR="/content/tpu_compile_cache"
export VLLM_XLA_CACHE_PATH="/content/tpu_compile_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0"
export JAX_LOG_COMPILES="1"
```

Initialize engine in Python:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/dev/shm/model",
    max_model_len=16384,           # or matched to target workload
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
)
```

> [!IMPORTANT]
> **Never set `SKIP_JAX_PRECOMPILE=1` or restrict `compilation_sizes` in production.** Skipping precompilation forces JIT compiles into timed generation steps, dropping decode speed from 85 tok/s to 8–30 tok/s. Always perform 3 warmup requests before serving real traffic.

---

## 5. Building & Reusing the Persistent Compile Cache

JAX compilation on TPU v5e is 100% deterministic based on model shapes and flags.

1. **Build:** Start engine with `JAX_COMPILATION_CACHE_DIR=/content/tpu_compile_cache`. Allow default AOT compilation and run 1 warmup request per target batch size.
2. **Package:** `tar -cf /content/tpu_compile_cache.tar -C /content tpu_compile_cache`.
3. **Upload:** Deposit `tpu_compile_cache.tar` into your Hugging Face or GCS cache repository.
4. **Reuse:** On any fresh VM, extract the archive before launching the engine. AOT compilation drops from 147s to ~7s (0 compiled programs, 100% cache hits).

---

## 6. Recommended Settings for Multi-Agent Council

For running parallel worker agents / Council queries:

- **Batch Size:** **4 to 8 concurrent requests**.
  - Batch 4 delivers **320 tok/s** aggregate throughput (80 tok/s/req, 27 ms TTFT).
  - Batch 8 delivers **598 tok/s** aggregate throughput (75 tok/s/req, 31 ms TTFT).
  - Batch 16 reaches **1,050 tok/s** (66 tok/s/req, 40 ms TTFT).
- **Max Context:** **8,192 to 16,384 tokens**. Prefill TTFT remains sub-second (0.27s at 4k; 0.89s at 16k).
- **KV Cache Pressure:** 4 concurrent 4k-context requests require only 16,384 tokens (~34% of the 48k token KV cache pool). Handled with 0 preemptions and zero queueing.

---

## 7. What Changes for a New Model

1. **Model Weights:** Download target checkpoint to `/dev/shm`.
2. **Persistent Cache:** Use the model-specific `tpu_compile_cache.tar` (cache keys depend on layer shapes, head dimensions, and vocab size).
3. **Everything Else Stays Identical:** Python 3.12 venv, `vllm-tpu 0.32.0`, `libtpu 0.0.48`, loader cache patch, and engine flags remain 100% reusable across models.
