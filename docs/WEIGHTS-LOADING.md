# Weights loading (fresh VM per task)

Goal: put GGUF weights on a **new** Colab VM, prove the model is on the GPU, then
`colab stop`. A VM left up between tasks burns CU. At most **one** human Drive
approval, ever (the one-time GCS copy). Secrets stay in environment variables.

Startup seconds are **ESTIMATE** unless marked MEASURED. Do not spend CU without
Sahil's go. Pointers: [`../ops.md`](../ops.md), [`../hardware.md`](../hardware.md),
[`../benchmarks/README.md`](../benchmarks/README.md).

## Size classes — GCS `gcloud storage cp` into `/tmp`

| Class | GPU (see hardware.md) | Copy ESTIMATE | Load ESTIMATE | First token ESTIMATE |
| :--- | :--- | :--- | :--- | :--- |
| ~4 GB | T4 or larger | ~7 s | ~4 s | ~25 s (`new` 3.3 s MEASURED on T4) |
| ~30 GB | A100-40+ with `--high-mem` | ~51 s | ~14 s | ~80 s |
| ~75 GB | A100-80 / G4 / H100 with `--high-mem` | ~128 s | ~30 s | ~170 s |

Copy times assume ~600 MB/s (UNVERIFIED). Fallback with no GCS: private Hugging
Face repo and `HF_XET_HIGH_PERFORMANCE=1` (~450 MB/s ESTIMATE; T4 hf_transfer was 104.65 MB/s MEASURED).

## One-time setup (Sahil, once)

1. Create a `us-central1` bucket and a Service Account with `roles/storage.objectViewer`.
2. On the laptop: `export GCP_SA_B64` from the SA JSON. Never commit it.
3. CPU VM: `colab --auth=oauth2 new -s migrate-once`, then `drivemount` (**the only
   click**). Activate the SA and copy Drive `models/` into
   `gs://${BUCKET_NAME}/models/`. Optional: copy a CUDA llama.cpp binary too.
4. `colab --auth=oauth2 stop -s migrate-once` and check `sessions` is empty.

```bash
gcloud storage buckets create "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" \
  --location="us-central1" --uniform-bucket-level-access
gcloud iam service-accounts create "${SA_NAME}" --project="${PROJECT_ID}"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member="serviceAccount:${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/storage.objectViewer"
```

## Each task (Grok)

Free preflight: `colab version` (0.7.5), `colab --auth=oauth2 whoami`, empty
`sessions`, positive `usage`. Then:

```bash
# GPU=T4 | L4 | A100 | G4 | H100 ; add --high-mem for ~30 GB and ~75 GB
colab --auth=oauth2 new -s load --gpu "$GPU"
colab --auth=oauth2 exec -s load --timeout 86400 \
  --env GCP_SA_B64="$GCP_SA_B64" --env GCS_MODEL_URI="$GCS_MODEL_URI" <<'PY'
import base64, os, subprocess
open("/tmp/sa.json","wb").write(base64.b64decode(os.environ["GCP_SA_B64"]))
subprocess.check_call(["gcloud","auth","activate-service-account","--key-file=/tmp/sa.json"])
subprocess.check_call(["gcloud","storage","cp","--user-output-enabled=false",
                       os.environ["GCS_MODEL_URI"], "/tmp/model.gguf"])
subprocess.check_call(["nvidia-smi"])
PY
# install engine matching nvidia-smi CUDA (T4 was CUDA 13.0, MEASURED), then run
colab --auth=oauth2 stop -s load
colab --auth=oauth2 sessions   # must be empty
```

T4 disk was ~194 GB free (MEASURED). Use `gcloud storage cp`, not `gsutil`.

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
- Cross-region VM vs `us-central1` bucket: ~200–350 MB/s possible (UNVERIFIED).
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
| `gcloud storage cp` ~500–700 MB/s here | UNVERIFIED (report note 10 unnamed; GCP 1 GiB bench) |
| HF `hf_xet` ~450 MB/s | UNVERIFIED (Unsloth Colab logs) |
| VRAM delta ≥ 70% of file size | UNVERIFIED (report design) |
| `--high-mem` avoids 75 GB PD throttle | UNVERIFIED |

## Gemini chat claims: verified or wrong

| Claim | Status | Notes |
| :--- | :--- | :--- |
| 1. HF hf_transfer 500–800 MB/s on Colab | wrong | 104.65 MB/s MEASURED on T4 (15.70 GB in 143.09 s) |
| 2. 80GB HF pull in 90–150s / under 2 min | wrong | INFERRED ~783 s at 104.65 MB/s MEASURED |
| 3. rclone copy into /dev/shm for a 75GB model | wrong | /dev/shm capacity is 5.7G MEASURED; 75 GB cannot fit; rclone unmeasured |
| 4. /dev/shm faster than disk for a 4GB copy | verified | /dev/shm 17.32 s (248 MB/s) vs /tmp 29.29 s (147 MB/s) MEASURED (~1.69x) |
| 5. laptop HTTP to VM: ssh forward, exec log, cloudflared | mixed | (a) unmeasured/denied (ssh harness denial), (b) verified (exec log RTT 2.92 s MEASURED), (c) untried |
| 6. Colab FAQ/ToS sentence forbidding ssh/tunnels/web servers | verified | FAQ restricted activities quoted; paid + positive CU lifts SSH restriction |
| 7. Drive SA rclone with --transfers=12 etc. | unmeasured | `scripts/drive_sa_pull.sh` prepared; unmeasured |

The MEASURED rows above match the report. The report's `new` ~10 s is the only
disagreement (slower than T4 MEASURED 3.3 s).
