# Universal TPU v5e-1 Serving Recipe (vLLM / tpu-inference)

Final, model-independent runbook for deploying any LLM on Google Colab TPU v5e-1 (16 GB HBM2; 15.748 GiB driver limit, 2.92 CU/hr). For detailed stage-by-stage cold-start mechanics, see [`FAST-COLDSTART.md`](FAST-COLDSTART.md).

---

## 1. Memory Budget Formula

At `gpu_memory_utilization=0.90`, vLLM allocates up to **14.17 GiB** (leaving 1.188 GiB driver headroom). Measured net XLA runtime overhead (`peak - weights - KV`) is **~0.05–0.38 GiB (47–380 MiB)**:

$$\text{Weights (GiB)} + \text{KV Cache (GiB)} \le \mathbf{13.8\text{ GiB}} \quad (\text{or } \mathbf{14.6\text{ GiB at } 0.95})$$

- **Weights Footprint:** $N_{\text{params}} \times \text{bytes/param}$ (e.g. `Qwen3-4B` BF16 = **7.494 GiB** [MEASURED]; 7B W4A16 $\approx$ 4.5 GiB).
- **KV Cache Footprint:** $\text{Tokens} \times (2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{dtype\_bytes})$.
  - Example (`Qwen3-4B`): $2 \times 36 \times 8 \times 128 \times 2 = 144.0\text{ KiB/token} \rightarrow 48,576\text{ tokens} = 6.695\text{ GiB}$ (at 0.90) or $54,336\text{ tokens} = 7.485\text{ GiB}$ (at 0.95) [MEASURED].
- **Recommendation:** Use `gpu_memory_utilization=0.90` for multi-request serving / Batch 32 to absorb transient activation spikes. `0.95` is verified stable for single-stream / low-batch serving (+11.8% KV tokens, 406 MiB headroom, 84.91 tok/s).

---

## 2. Fast Fresh-VM Bootstrap (Prep: 19.85 s [MEASURED])

Colab mounts `/dev/shm` (23 GB tmpfs) with `noexec` by default. Remount it `exec` and pull the prebuilt CPU-torch Python 3.12 venv (`venv312_cpu.tar.zst`, 1.35 GB packed / 5.01 GB unpacked, 0 NVIDIA packages, `.pyc` precompiled), model weights, and persistent XLA cache into `/dev/shm` in parallel:

```bash
# 1. Strip noexec from /dev/shm so libtorch_global_deps.so can dlopen (0.08s)
mount -o remount,exec /dev/shm
mkdir -p /dev/shm/tpu_compile_cache /content/hooks

# 2. Parallel pull: prebuilt CPU-torch venv + model weights + XLA cache into /dev/shm (~20s total)
python3 -c '
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, snapshot_download
import subprocess

def pull_venv():
    p = hf_hub_download(repo_id="avgrocketman/tpu-compile-cache", filename="venv312_cpu.tar.zst", local_dir="/dev/shm")
    subprocess.run(["tar", "-I", "zstd", "-xf", p, "-C", "/dev/shm"], check=True)

def pull_cache():
    p = hf_hub_download(repo_id="avgrocketman/tpu-compile-cache", filename="tpu_compile_cache.tar", local_dir="/dev/shm")
    subprocess.run(["tar", "-xf", p, "-C", "/dev/shm/tpu_compile_cache", "--strip-components=1"], check=True)

def pull_model():
    snapshot_download("Qwen/Qwen3-4B", local_dir="/dev/shm/model", max_workers=12)

with ThreadPoolExecutor(max_workers=3) as ex:
    for f in [ex.submit(pull_venv), ex.submit(pull_cache), ex.submit(pull_model)]:
        f.result()
'
```

---

## 3. Mandatory Weight Loading Speedup Patch (`jax.clear_caches`)

Neutralize redundant per-module `jax.clear_caches()` in `tpu_inference` to cut weight loading from 29.4–40.5 s down to **10.7–11.8 s** [MEASURED]:

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

## 4. Fast Environment Flags & In-Process Engine Launch

Bypass GCE metadata 404 timeouts (**-9.65 s**, 15.56 s $\rightarrow$ 5.91 s) and run `EngineCore` in-process (**-11.0 s** vs `spawn`):

```bash
export PYTHONPATH="/content/hooks"
export TPU_SKIP_MDS_QUERY="1"
export TPU_TOPOLOGY="1x1"
export TPU_HOST_BOUNDS="1,1,1"
export TPU_CHIPS_PER_HOST_BOUNDS="1,1,1"
export TPU_WORKER_ID="0"
export TPU_WORKER_HOSTNAMES="localhost"
export JAX_PLATFORMS="tpu,cpu"
export VLLM_ENABLE_V1_MULTIPROCESSING="0"
export JAX_COMPILATION_CACHE_DIR="/dev/shm/tpu_compile_cache"
export VLLM_XLA_CACHE_PATH="/dev/shm/tpu_compile_cache"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="0"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES="0"
```

Initialize engine with `/dev/shm/vllm_env/bin/python`:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/dev/shm/model",
    max_model_len=17408,           # match the prebuilt XLA cache key (17408 or 32768)
    gpu_memory_utilization=0.90,   # 0.90 for Batch 32; 0.95 for low-batch (+5,760 KV tokens)
    trust_remote_code=True,
)
```

---

## 5. Multi-Agent Council & Batching Scaling [MEASURED]

- **Single-Stream & Batch Scaling (`Qwen3-4B`, 256 output tokens):**
  - Batch 1: **84.92 tok/s** (88.58 tok/s at 32K cfg, 16.3 ms TTFT).
  - Batch 4: **320.25 tok/s** aggregate (80.50 tok/s/req, 26.7 ms TTFT).
  - Batch 8: **596.53 tok/s** aggregate (75.16 tok/s/req, 37.1 ms TTFT).
  - Batch 16: **1,050.10 tok/s** aggregate (66.20 tok/s/req, 40.0 ms TTFT).
  - Batch 32: **1,653.41 tok/s** aggregate (52.63 tok/s/req, 103.5 ms TTFT, 4.95 s wall).
- **Concurrent Long Context:** Handled **8 concurrent 8k** ($65,536$ tokens, 321.0 tok/s) and **4 concurrent 16k** ($65,536$ tokens, 157.4 tok/s) in 6.4–6.5 s with **0 preemptions and 0 queued requests** via Chunked Prefill (`max_num_batched_tokens=8192`).

---

## 6. Known Traps to Avoid

1. **Never set `SKIP_JAX_PRECOMPILE=1` or restrict `compilation_sizes=[16, 512]`:** Forces uncompiled shapes to JIT-compile during generation, collapsing decode speed from 85–88 tok/s down to **7.6–29.9 tok/s** [MEASURED, `tpu-characterization/RESULTS.md`].
2. **Never skip `mount -o remount,exec /dev/shm`:** Running a venv in `/dev/shm` without `exec` fails with `OSError: failed to map segment from shared object` on `libtorch_global_deps.so` [MEASURED, `RUN1-VENV.md`].
3. **Always warm up before timing:** Send 1–3 untimed requests to populate PJRT device buffers.
4. **No per-token execution hook I/O:** Synchronous disk writes inside forward steps throttle token delivery.
5. **Use Python 3.12 exclusively:** `vllm-tpu==0.32.0` wheels target Python 3.12; Colab's default Python 3.13 fails to build from source (`gemma4-e4b-qat/RESULTS.md`).
