# Colab TPU v5e-1 Fast-Start Venv Build (Run 1 Complete)

Fast-start Python 3.12 venv with CPU-only PyTorch and vLLM-TPU 0.32.0 on Colab TPU v5e-1.

---

## 1. Measured Results

| Metric | Measured Value | Baseline / Target | Notes |
| :--- | :--- | :--- | :--- |
| **Initial `/dev/shm` Mount** | `(rw,nosuid,nodev,noexec,...)` | `noexec` present | Blocked `dlopen` in initial run |
| **Remount Exec** | **`mount -o remount,exec /dev/shm`** | `rc=0` | `noexec` stripped successfully |
| **Venv Build Path** | **/dev/shm/vllm_env** | tmpfs RAM | Ext4 disk write bottlenecks bypassed |
| **Install Time** | **42.08 s** | 91.21 s | **-49.13 s (-54%)** install speedup |
| **NVIDIA Packages** | **0** | 15 CUDA packages | 100% eliminated (~1.4 GB saved) |
| **TPU Detection / Import** | **PASSED (20.49 s)** | `[TpuDevice(id=0)]` | `import vllm, jax` fully verified |
| **Compileall Time** | **47.19 s** | Full `.pyc` compilation | Eliminates runtime JIT bytecode lag |
| **Venv Size (Unpacked)** | **5,132.45 MB (5.01 GB)** | ~9.9 GB (old venv) | 48% smaller than old bundle |
| **Packed Size (`.tar.zst`)** | **1,387.55 MB (1.35 GB)** | Level 3 zstd (`-T0`) | Packed in 10.36 s |
| **Unpack Time Test** | **6.17 s** | < 10 s target | **15x faster than 91s install** |
| **Hugging Face Upload** | **Confirmed** | `avgrocketman/tpu-compile-cache` | `venv312_cpu.tar.zst` & `versions.json` |
| **Compute Units Used** | **0.13 CU** (392.30 -> 392.17) | Budget <= 0.20 CU | **Within budget** |
| **Active Sessions** | **0** | Verified | Runtime stopped, timer killed |

---

## 2. Key Package Versions

- `vllm-tpu`: 0.32.0
- `tpu-inference`: 0.32.0
- `torch`: 2.10.0+cpu
- `jax` / `jaxlib`: 0.11.2
- `libtpu`: 0.0.48
- `transformers`: 4.57.1

---

## 3. Operational Cold-Start Bootstrap (Run 2 Ready)

```bash
mount -o remount,exec /dev/shm
python3 -c '
from huggingface_hub import hf_hub_download; import subprocess
p = hf_hub_download(repo_id="avgrocketman/tpu-compile-cache", filename="venv312_cpu.tar.zst")
subprocess.run(["tar", "-I", "zstd", "-xf", p, "-C", "/dev/shm"], check=True)
'
```
