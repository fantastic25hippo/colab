# Runbook: G4 Hardware and Entitlement Probe

Probe procedure for testing NVIDIA G4 entitlement, hardware specifications, host memory pinning (`mlock`), and hourly Compute Unit (CU) burn on Google Colab under the [Google AI Pro](https://support.google.com/googleone/answer/16287313) account (`zuluzain5@gmail.com`).

Primary references:
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- NVIDIA RTX PRO 6000 Server Edition: https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/
- SGLang Qwen 3.8 Flash-Next Cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
- Lenovo ThinkSystem G4 Server Documentation: https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu
- Benchmark & Meter Reference (Qiita): https://qiita.com/jin0g/items/8fbfb209ac4e9343c13f

---

## 1. Objectives and Scope

1. **Entitlement Probe**: Determine whether this Google AI Pro account (`zuluzain5@gmail.com`, 200 CU/month) can provision a `--gpu G4` VM.
2. **Hardware Verification**: Verify that the provisioned GPU is the Blackwell RTX PRO 6000 Server Edition with ~96 GB VRAM and ~177 GB host system RAM.
3. **Memory Pinning Probe**: Determine whether the unprivileged Colab user environment permits host memory pinning (`mlock` >= 64 GB) without Docker container privileges.
4. **Meter Verification**: Confirm the real-time hourly CU burn rate via `colab usage`.
5. **Fail-Fast Teardown**: Stop the instance immediately upon test completion to avoid unnecessary CU expenditure.

> [!IMPORTANT]
> **UNVERIFIED Status**:
> - **G4 Entitlement on AI Pro**: An independent Japanese report on `note.com` indicated G4 was grayed out on one AI Pro account. Whether `zuluzain5@gmail.com` can provision G4 remains **UNVERIFIED** until this probe executes.
> - **Colab Docker Availability**: Standard Colab VMs often do not provide Docker or do not have the Docker daemon running. Whether Docker is available and operational on the provisioned Colab VM remains **UNVERIFIED**.
> - **Colab Native mlock Without Docker**: SGLang `--ple-offload-embedding` requires pinning ~47.7 GiB of host RAM. The cookbook verified path is Docker `--ulimit memlock=-1`, but whether native Colab unprivileged runtimes permit pinning >= 64 GB remains **UNVERIFIED**.
> - **Server Edition Decode Speed**: Passively cooled Server Edition runs memory at 1,597 GB/s (~10.9% below Workstation Edition's 1,792 GB/s), giving an expected decode speed of ~130–135 tok/s (vs ~148 tok/s in cookbook). This remains **UNVERIFIED** on live Colab silicon.

---

## 2. Pre-Flight Check

On your local workstation terminal:

```bash
# Verify CLI version (must be 0.7.4; 0.6.0 drops sessions, 0.7.2 broken exec, 0.7.3 yanked)
colab --auth=oauth2 whoami
# Expected: Email: zuluzain5@gmail.com

# Verify zero lingering active sessions
colab --auth=oauth2 sessions

# Check existing CU balance and record timestamp before provisioning
date -u +"%Y-%m-%dT%H:%M:%SZ"
colab --auth=oauth2 usage
```

---

## 3. Provisioning Request

Request a G4 instance without `--high-mem`:

```bash
colab --auth=oauth2 new -s g4-probe --gpu G4
```

### Immediate Branching Logic:
- **Case A: Request Rejected or Gated**:
  If the CLI returns an entitlement error, quota denial, or unsupported accelerator message:
  1. Record the exact error string.
  2. Conclude G4 is unentitled for this AI Pro tier.
  3. G4 unavailable or mlock fails → Flash-Next on Colab is blocked; stop and report.
  4. Run `colab --auth=oauth2 sessions` to confirm nothing is left allocated.
- **Case B: Session Assigned**:
  Proceed immediately to Step 4.

---

## 4. Hardware Inspection & Stop Gates

If this `colab exec` fails or times out, go straight to §7 teardown. Do not continue to §5.

Run an immediate non-interactive hardware telemetry audit:

```bash
colab --auth=oauth2 exec -s g4-probe --timeout 120 -- bash -c '
echo "=== GPU TELEMETRY ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version,pstate,compute_cap --format=csv

echo "=== HOST MEMORY ==="
free -g
free -m

echo "=== CPU ARCHITECTURE ==="
lscpu | grep -E "Model name|CPU\(s\):|Thread\(s\) per core"

echo "=== FILESYSTEM STORAGE ==="
df -h / /content
'
```

### Strict SKU Stop Criteria:
Halt and unassign the VM immediately (`colab --auth=oauth2 stop -s g4-probe`) if ANY of the following conditions occur:
1. **GPU is not Blackwell G4**: Name does not identify as `NVIDIA RTX PRO 6000 Server Edition` (or Blackwell `sm_120`/`sm_120a`).
2. **VRAM is not ~96 GB**: `nvidia-smi` reports total VRAM significantly below ~95.6 GiB (97,896 MiB). If assigned an A100 (40 GB or 80 GB) or L4 (24 GB), STOP.
3. **Host RAM is not ~177 GB**: Total system RAM reported by `free -g` is in the ~53–84 GB range (standard A100/L4 shapes). Qiita measured G4 host RAM at 176.9 GB with 48 vCPUs. If host RAM < 150 GB, STOP.
4. **Storage Shortage**: Available disk on `/content` or `/` is < 150 GB (insufficient to stage the 132.7 GB Flash-Next checkpoint).

---

## 5. Host Memory Pinning (`mlock`) and Docker Probe

SGLang's `--ple-offload-embedding` offloads the 51.2B FP8 piece-level embedding table (~47.7 GiB) into pinned host RAM with asynchronous PCIe prefetching. Pinning requires `mlock` permissions. The official cookbook verified path uses Docker with `--ulimit memlock=-1`. However, Colab VMs often do not have Docker installed or running.

If this `colab exec` fails or times out, go straight to §7 teardown.

Execute the following in-guest script to probe both Docker availability and native `mlock` allocation limits:

```bash
colab --auth=oauth2 exec -s g4-probe --timeout 180 -- bash -c '
echo "=== DOCKER AVAILABILITY PROBE ==="
if command -v docker &>/dev/null; then
    docker --version
    if docker info &>/dev/null; then
        echo "DOCKER STATUS: Docker daemon is active and accessible."
    else
        echo "DOCKER STATUS: Docker installed, but daemon is NOT running / inaccessible."
    fi
else
    echo "DOCKER STATUS: Docker is NOT installed on this Colab VM."
fi

echo -e "\n=== RLIMIT_MEMLOCK & ULIMIT PROBE ==="
ulimit -l

python3 -c '\''
import os, sys, resource, ctypes

soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
print(f"Soft limit: {soft} bytes ({soft / (1024**3):.2f} GiB)")
print(f"Hard limit: {hard} bytes ({hard / (1024**3):.2f} GiB)")

# Attempt to lock a test buffer (50 GiB to cover the 47.7 GiB PLE table)
TARGET_GIB = 50
TARGET_BYTES = TARGET_GIB * 1024 * 1024 * 1024

print(f"\nAttempting allocation of {TARGET_GIB} GiB host memory...")
try:
    buf = bytearray(TARGET_BYTES)
    print("Allocation succeeded. Attempting mlock via libc...")
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    addr = (ctypes.c_char * len(buf)).from_buffer(buf)
    res = libc.mlock(ctypes.byref(addr), ctypes.c_size_t(len(buf)))
    if res == 0:
        print(f"SUCCESS: Successfully pinned {TARGET_GIB} GiB host RAM with mlock!")
        libc.munlock(ctypes.byref(addr), ctypes.c_size_t(len(buf)))
        del buf
    else:
        err = ctypes.get_errno()
        print(f"FAILED: libc.mlock returned {res} (errno {err}: {os.strerror(err)})")
        print("RESULT: Native mlock failed. Docker container with --ulimit memlock=-1 is required if Docker is available.")
        del buf
except MemoryError:
    print("FAILED: Host MemoryError during allocation.")
except Exception as e:
    print(f"FAILED: Exception encountered: {e}")
'\''

echo -e "\n=== CUDA HOST MEMORY PINNING (torch.pin_memory) PROBE ==="
# libc mlock alone does not prove CUDA page-locked memory registration
python3 -c "import torch,time;t=time.time();x=torch.empty(48*2**30,dtype=torch.uint8,pin_memory=True);print(\"pinned OK\",round(time.time()-t,1),\"s\");del x"
'
```

### Result Analysis:
- If `mlock` and `pin_memory` succeed: Native SGLang execution outside Docker is viable.
- If native pinning fails but Docker daemon is active: SGLang can be launched inside Docker using `docker run --gpus all --ulimit memlock=-1`.
- If both native pinning and Docker fail: G4 unavailable or mlock fails → Flash-Next on Colab is blocked; stop and report.

---

## 6. Real Metered Rate Verification

Check the metered hourly burn rate reported by the Colab backend, recording the exact timestamp:

```bash
date -u +"%Y-%m-%dT%H:%M:%SZ"
colab --auth=oauth2 usage
```

Expected metered burn for G4 Server Edition: **~8.71 CU/hr** (measured by Qiita at 8.71 CU/hr; early promotional/off-peak rates of 6.40–6.74 CU/hr are obsolete).

Cross-check via Web UI if browser profile `zuluzain5@gmail.com` is active:
1. Navigate to https://colab.research.google.com
2. Open active session `g4-probe`.
3. Check the Resources panel (RAM / Disk / Compute Units remaining).

---

## 7. Immediate Teardown

> [!CAUTION]
> **Stop/unassign must run on every path** (whether pass, fail, error, or timeout). Keep the probe strictly under **~10 minutes** (burning ≤1.45 CU total). Record timestamps and `colab usage` before and after.

```bash
# Record usage and timestamp immediately before teardown
date -u +"%Y-%m-%dT%H:%M:%SZ"
colab --auth=oauth2 usage

# Terminate session from workstation
colab --auth=oauth2 stop -s g4-probe

# Verify clean state and record post-teardown balance
colab --auth=oauth2 sessions
date -u +"%Y-%m-%dT%H:%M:%SZ"
colab --auth=oauth2 usage
```

Alternatively, from within the guest (runs on all exit paths in automation):
```python
from google.colab import runtime
runtime.unassign()
```

---

## 8. Probe Decision Matrix

| Probe Item | Expected Value | Pass Next Step | Fail Remediation |
| :--- | :--- | :--- | :--- |
| **G4 Provisioning** | Session created | Proceed to hardware check | G4 unavailable or mlock fails → Flash-Next on Colab is blocked; stop and report. |
| **GPU Name** | RTX PRO 6000 Server Edition | Proceed to VRAM check | Terminate session immediately (`colab stop`) |
| **VRAM Capacity** | ~95.6 GiB (`nvidia-smi`) | Proceed to Host RAM check | Stop immediately; SKU mismatch |
| **Host System RAM** | ~177 GB (`free -g`) | Proceed to mlock & docker probe | Stop immediately; insufficient RAM for PLE |
| **Docker Availability** | Daemon active & GPU accessible | Docker `--ulimit memlock=-1` path verified | Native `mlock` / `pin_memory` required |
| **Native mlock / CUDA Pinning** | 48 GiB pinned OK | Native SGLang permitted | Docker `--ulimit memlock=-1` required; if unavailable: G4 unavailable or mlock fails → Flash-Next on Colab is blocked; stop and report. |
| **Metered CU Burn** | ~8.71 CU/hr | Record for sitting budget | If > 12 CU/hr, re-evaluate sitting economics |
