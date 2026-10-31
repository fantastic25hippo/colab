# Weights loading (fresh VM per task)

Goal: put GGUF weights on a **new** Colab VM, prove the model is on the GPU, then
`colab stop`. A VM left up between tasks burns CU. Weights load non-interactively
via Hugging Face shards into `/dev/shm` (no Drive approval, no bucket).

Startup seconds are **ESTIMATE** unless marked MEASURED. Do not spend CU without
Sahil's go. Pointers: [`../ops.md`](../ops.md), [`../hardware.md`](../hardware.md),
[`../benchmarks/README.md`](../benchmarks/README.md).

## Measured path: sharded weights into `/dev/shm`

Download sharded GGUF or safetensors from Hugging Face into `/dev/shm` using `snapshot_download` with `max_workers=8..12` (see [`../research/03-HuggingFace-Xet-Tuning.md`](../research/03-HuggingFace-Xet-Tuning.md)):
- `HF_XET_HIGH_PERFORMANCE=1` (with `hf_xet >= 1.7.0`)
- `HF_HUB_ENABLE_HF_TRANSFER=1` (used in measured runs; note `research/03` & `11` state Xet is now the default backend)

### Measured throughput (L4 & TPU v5e-1)
Numbers from [`docs/lfm-l4.md`](lfm-l4.md), [`../benchmarks/HF-transfer/L4/SWEEP.md`](../benchmarks/HF-transfer/L4/SWEEP.md), and [`../benchmarks/TPU-v5e/FAST-COLDSTART.md`](../benchmarks/TPU-v5e/FAST-COLDSTART.md):
- **L4 winning path**: 14.88 GB, 16 shards into `/dev/shm`, `max_workers=12`: **16.60 s** (**896 MB/s** MEASURED; 3-rep sweep mean **920.8 MB/s**).
- **L4 worker counts (`SWEEP.md`)**: `max_workers=8..12` recommended. Across 3 reps, `w=4,8,12,16` all hit **940–1,120 MB/s** on unthrottled pulls and **190–316 MB/s** when Hub-throttled; the earlier single-rep slow timings at `w=8` (61.8 s) and `w=16` (82.3 s) were random throttling variance, not worker collapse.
- **L4 disk overlay**: 16 shards into disk `/tmp`: **104.82 s** (141.93 MB/s MEASURED). Disk `/tmp` is ~5–6× slower than `/dev/shm`.
- **TPU v5e-1**: `Qwen/Qwen3-4B` (7.49 GiB safetensors) into `/dev/shm`: **10.14–11.89 s** (**~645–755 MB/s** MEASURED), overlapped with prebuilt `venv312_cpu.tar.zst` download/unpack in **19.85 s total** (see [`../benchmarks/TPU-v5e/FAST-COLDSTART.md`](../benchmarks/TPU-v5e/FAST-COLDSTART.md)).

```python
import os
from huggingface_hub import snapshot_download

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"

snapshot_download(
    repo_id="avgrocketman/LFM2.5-24B-A2B-Q4-shards",
    allow_patterns=["shards/*"],
    local_dir="/dev/shm/bundle",
    max_workers=12,
)
```

## Dead plans

- **Google Cloud bucket**: Abandoned (no billing account).
- **gcsfuse**: Abandoned (FUSE stalls, no billing account).
- **Drive mount**: Abandoned (a human click on every new VM; slow transfer).

## `/dev/shm` sizes per GPU

| Hardware | `/dev/shm` size | Fits 14.88 GB LFM Q4? | Fits 25.1 GB Flint Q4? |
| :--- | :--- | :--- | :--- |
| T4 | 5.7 GB (MEASURED) | No | No |
| L4 | 26 GB (MEASURED) | Yes | Tight / No |
| A100-40 | 41 GB (MEASURED) | Yes | Yes |
| TPU-v5e | 23 GB (MEASURED) | Yes | No |
| G4 | 87 GB (MEASURED) | Yes | Yes |

## Health (Council)

1. Before and after load:
   `nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader`
   VRAM used must rise by at least 70% of the GGUF size (UNVERIFIED threshold).
2. llama.cpp log line `offloaded N/N layers to GPU` with N equal to the model.
   `offloaded 0/` is fatal. `torch.cuda.is_available()` is not a health signal.
3. Print tok/s about every 20 s (also keeps the Pro ~10–15 min reaper away).
   After warmup, tok/s near 0 with GPU util 0 is CPU fallback — abort.

## Failure modes

- Fresh-VM `drivemount` always prints a browser URL (MEASURED). Skip it per task.
- `exec` default timeout is 30 s; always pass `--timeout 86400`.
- Unstopped VM burns CU (T4 1.07 CU/hr MEASURED). Always `stop`, then `sessions`.
- FUSE mmap (Drive / gcsfuse): stalls over 30 s; tok/s can drop under 0.5.
- `colab upload` dies at 100 MiB, HTTP 400 (MEASURED).
- `scp` over `colab ssh` is 3–6 MB/s (MEASURED) — hours for 30–75 GB.
- Google Cloud bucket: no billing account; gcsfuse stalls (DEAD).
- 75 GB without `--high-mem`: Persistent Disk write throttle (UNVERIFIED).
- Report `colab new` ~10 s is UNVERIFIED; T4 MEASURED 3.3 s.

## Claims this design uses

| Claim | Status |
| :--- | :--- |
| New VM `drivemount` needs a browser grant | MEASURED |
| Same-VM remount is silent | MEASURED |
| `colab upload` 100 MiB HTTP 400 | MEASURED |
| `colab ssh` scp ~3–6 MB/s | MEASURED; also google-colab-cli |
| T4 1.07 CU/hr; ~194 GB disk; `new` 3.3 s | MEASURED |
| No headless `drivemount` | DOCUMENTED (google-colab-cli AGENTS.md, #103, #113, #165) |
| `exec --env` since 0.7.0; CLI 0.7.5 | DOCUMENTED (CLI / PyPI) |
| Drive rclone ~50–76 MiB/s | DOCUMENTED (IEEE rclone study) |
| gdown 24 h lockout | DOCUMENTED (Colab FAQ, gdown #26) |
| FUSE mmap stalls | DOCUMENTED (GKE Fluid, mmap write-up) |
| Paid CLI jobs allowed | DOCUMENTED (Colab FAQ) |
| Google Cloud bucket ~500–700 MB/s | DEAD (no billing account) |
| HF `hf_xet` ~450 MB/s | UNVERIFIED (Unsloth Colab logs) |
| VRAM delta ≥ 70% of file size | UNVERIFIED (report design) |
| `--high-mem` avoids 75 GB PD throttle | UNVERIFIED |

## Gemini chat claims: verified or wrong

| Claim | Status | Notes |
| :--- | :--- | :--- |
| 1. HF hf_transfer 500–800 MB/s on Colab | verified (shards) | Single blob peaked at 352.98 MB/s on A100; 16 shards into /dev/shm reached 896 MB/s on L4 (MEASURED) |
| 2. 80GB HF pull in 90–150s / under 2 min | wrong | INFERRED ~783 s (T4) down to ~227 s (A100) at 352.98 MB/s MEASURED (> 2 min) |
| 3. rclone copy into /dev/shm for a 75GB model | mixed | Checked T4 (5.7G), L4 (26G), A100-40 (41G), TPU-v5e (23G) — cannot fit; fits only on G4 (87G MEASURED). rclone unmeasured |
| 4. /dev/shm faster than disk for a 4GB copy | verified | /dev/shm (0.52–1.67 s, up to 8.3 GB/s; TPU-v5e 0.77 s, 5.6 GB/s) vs /tmp (11.83–36.39 s, 118–363 MB/s) MEASURED |
| 5. laptop HTTP to VM: ssh forward, exec log, cloudflared | mixed | (a) unmeasured/denied (ssh harness denial), (b) verified (exec log RTT 2.92 s MEASURED), (c) untried |
| 6. Colab FAQ/ToS sentence forbidding ssh/tunnels/web servers | verified | FAQ restricted activities quoted; paid + positive CU lifts SSH restriction |
| 7. Drive SA rclone with --transfers=12 etc. | unmeasured | Unmeasured (superseded by HF shards into `/dev/shm`) |
| 8. TPU weight loading & engine compatibility | verified | TPU v5e: 2.92 CU/hr, 23G `/dev/shm`, `vllm-tpu==0.32.0` + `tpu-inference==0.32.0` verified at **84.92 tok/s** on `Qwen/Qwen3-4B` (see [`../benchmarks/TPU-v5e/RECIPE.md`](../benchmarks/TPU-v5e/RECIPE.md)). TPU v6e (`--tpu v6e1`) timed out during provisioning on repeated attempts (0 CU spent). |

The MEASURED rows above match the report. The report's `new` ~10 s is the only
disagreement (slower than T4 MEASURED 3.3 s).
