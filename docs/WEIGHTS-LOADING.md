# Weights loading (fresh VM per task)

Goal: put GGUF weights on a **new** Colab VM, prove the model is on the GPU, then
`colab stop`. A VM left up between tasks burns CU. Weights load non-interactively
via Hugging Face shards into `/dev/shm` (no Drive approval, no bucket).

Startup seconds are **ESTIMATE** unless marked MEASURED. Do not spend CU without
Sahil's go. Pointers: [`../ops.md`](../ops.md), [`../hardware.md`](../hardware.md),
[`../benchmarks/README.md`](../benchmarks/README.md).

## Measured path: sharded GGUF into `/dev/shm`

Download sharded GGUF from a public Hugging Face repository into `/dev/shm` using `snapshot_download` with `max_workers=12` and the following flags set:
- `HF_HUB_ENABLE_HF_TRANSFER=1`
- `HF_XET_HIGH_PERFORMANCE=1`
- `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`

### Measured throughput (L4)
Numbers from `docs/lfm-l4.md`:
- **Winning path**: L4, 14.88 GB, 16 shards into `/dev/shm`, `max_workers=12`: **16.60 s** (**896 MB/s** MEASURED).
- **Disk overlay**: 16 shards into disk `/tmp`: **104.82 s** (141.93 MB/s MEASURED). Disk `/tmp` is significantly slower than `/dev/shm`.
- **Worker counts**: `max_workers=12` is the fastest confirmed setting. `max_workers=8` (61.80 s) and `max_workers=16` (82.30 s) were slower and unconfirmed.

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

| Hardware | `/dev/shm` size | Fits 14.88 GB model? |
| :--- | :--- | :--- |
| T4 | 5.7 GB (MEASURED) | No |
| L4 | 26 GB (MEASURED) | Yes |
| A100-40 | 41 GB (MEASURED) | Yes |
| TPU-v5e | 23 GB (MEASURED) | Yes |
| G4 | 87 GB (MEASURED) | Yes |

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
| 7. Drive SA rclone with --transfers=12 etc. | unmeasured | `scripts/drive_sa_pull.sh` prepared; unmeasured |
| 8. TPU weight loading & engine compatibility | verified | TPU v5e: 2.92 CU/hr, 23G /dev/shm, JAX preinstalled; GGUF/llama.cpp lacks native TPU support (requires JAX/Pallas, torch_xla, or vLLM TPU). TPU v6e (`--tpu v6e1`) timed out during provisioning on repeated attempts (capacity unavailable, 0 CU spent). |

The MEASURED rows above match the report. The report's `new` ~10 s is the only
disagreement (slower than T4 MEASURED 3.3 s).
