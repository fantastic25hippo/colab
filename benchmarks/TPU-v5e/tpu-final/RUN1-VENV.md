# Colab TPU v5e-1 Fast-Start Venv Build (Run 1 of 2)

Execution log and results for building a fast-start Python 3.12 venv with CPU-only PyTorch on Colab TPU v5e-1.

---

## 1. Measured Results

| Metric | Measured Value | Target / Baseline | Status |
| :--- | :--- | :--- | :--- |
| **Install Time** | **41.00 s** | 91.21 s (baseline) | **50.21 s saved (-55%)** |
| **Nvidia Packages Count** | **0** | 15 packages (~1.4 GB) | **Eliminated 100%** |
| **TPU Detection / Import** | **FAILED (Stopped at Step 3)** | TPU device 0 detected | See exact traceback below |
| **Venv Size (Unpacked)** | N/A (halted before pack) | Target ~1.8 GB | Stopped at Step 3 |
| **Packed Size (`.tar.zst`)** | N/A (halted before pack) | Target < 600 MB | Stopped at Step 3 |
| **Unpack Test Time** | N/A (halted before pack) | Target < 10 s | Stopped at Step 3 |
| **Upload Confirmation** | N/A (halted before pack) | `avgrocketman/tpu-compile-cache` | Stopped at Step 3 |
| **Compute Units Used** | **0.05 CU** (392.35 -> 392.30) | Budget <= 0.25 CU | **Within budget** |
| **Session Cleanup** | Clean (0 active sessions) | 0 active sessions | Stopped & verified |

---

## 2. Exact Import Failure & Root Cause

As instructed by Step 3 ("*If torch CPU breaks the vllm import, stop and report the exact error*"), the pipeline immediately halted and reported the error:

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/dev/shm/vllm_env/lib/python3.12/site-packages/vllm/__init__.py", line 14, in <module>
    import vllm.env_override
  File "/dev/shm/vllm_env/lib/python3.12/site-packages/vllm/env_override.py", line 150, in <module>
    import torch
  File "/dev/shm/vllm_env/lib/python3.12/site-packages/torch/__init__.py", line 430, in <module>
    _load_global_deps()
  File "/dev/shm/vllm_env/lib/python3.12/site-packages/torch/__init__.py", line 388, in _load_global_deps
    _preload_cuda_deps(err)
  File "/dev/shm/vllm_env/lib/python3.12/site-packages/torch/__init__.py", line 344, in _preload_cuda_deps
    raise err
  File "/dev/shm/vllm_env/lib/python3.12/site-packages/torch/__init__.py", line 366, in _load_global_deps
    ctypes.CDLL(global_deps_lib_path, mode=ctypes.RTLD_GLOBAL)
  File "/usr/lib/python3.12/ctypes/__init__.py", line 379, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /dev/shm/vllm_env/lib/python3.12/site-packages/torch/lib/libtorch_global_deps.so: failed to map segment from shared object
```

### Analysis & Next Step Options
1. **Root Cause:** `/dev/shm` on Colab is mounted `noexec`. Dynamic linker `mmap` with `PROT_EXEC` fails on shared libraries located in `/dev/shm`.
2. **Path Resolution for Run 2:**
   - **Option A:** Remount `/dev/shm` with exec permissions (`mount -o remount,exec /dev/shm`).
   - **Option B:** Build/extract venv at `/content/vllm_env` (ext4 with `exec` permitted).
