# 02 — Storage Channels, Costs and Headless Auth

Consolidated from `Fast Headless Colab Weight Loading` §§1–2, `Google AI Pro Cloud Benefits`, and the “Read both of these software…” conversation. The question: where do 15–80 GB of weights live, how fast can Colab pull them, and how do you authenticate without a browser click?

## 1. Storage/persistence comparison

| Mechanism | Headless no-click auth | Realistic Colab throughput | Limits | Cost without GCP billing | Main risk |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hugging Face Hub** (public/private, Xet/hf_transfer) | Yes, `--env HF_TOKEN=…` [C] | 252–353 MB/s to disk; 200–270 MB/s to `/dev/shm` [M] | 300 GB+ per repo free; <50 GB/shard recommended [G] | $0 [C] | None; native ML standard |
| **Google Drive + IAM service account** (rclone/Drive API) | Yes, static JSON key via `--env` [C] | 50–76 MiB/s baseline [C]; 60–140 MB/s multi-shard [G] | 15 GB free; 2 TB with Google AI Pro [G] | $0; Drive API needs no billing account [C] | High risk of HTTP 403 daily quota lockout [G] |
| **Google Drive + OAuth2 refresh token** (rclone) | Yes after one-time local `rclone authorize` [C] | 50–76 MiB/s baseline; 60–140 MB/s multi-shard [C/G] | Bound to personal Drive quota | $0 [C] | 7-day token expiry if OAuth app is in “Testing” [G] |
| **Google Drive shortcuts** | Yes with service account/OAuth [G] | 50–76 MiB/s [C] | Bypasses owner storage quota; points to third-party file [G] | $0 [G] | 24-hour 403 lockout on shared target [G] |
| **Kaggle Models/Datasets** (`kagglehub`/`kaggle.json`) | Yes, `--env KAGGLE_USERNAME/KEY` [C] | 80–150 MB/s legacy zip; **200–350 MB/s** uncompressed GCS shards [G] | 100 GB per dataset/model; 100 GB+ private quota [G] | $0 [C] | None major; backed by GCS signed URLs [G] |
| **Google Cloud Storage** (GCS, `gs://`) | Yes, `gcloud auth`/service account | 400–900+ MB/s inside GCP; 300–800 MB/s typical | Bucket quota/cost; region matters | $10/month AI Pro credit; overage billed | Cross-region egress can wipe the credit in a day |
| **Cloudflare R2 / Backblaze B2** (S3-compatible) | Yes, static S3 keys [C] | 200–400 MB/s via s5cmd/Run:ai [G] | 10 GB free cap (too small for 15–80 GB) [G] | Credit card required above 10 GB (~$1.05/mo for 80 GB) [G] | Fails strict “no billing card” constraint |
| **`colab drivemount`** (native ephemeral DFS) | **No** — browser click every VM [M] | 40–90 MB/s sequential; <5 MB/s random FUSE mmap [G] | Bound to personal Drive quota | $0 [C] | Unusable headless; blocks websocket on consent [C] |
| **`colab upload`** (Jupyter Contents API) | Yes [C] | 2–15 MB/s; laptop uplink + base64 JSON [G] | Fails/OOMs >500 MB–1 GB [G] | $0 [C] | Unusable for 15–80 GB |
| **Pre-built disk/container snapshot** | N/A | N/A | N/A | N/A | Colab backend does not support it [C] |

## 2. Hugging Face Hub

- In 2026 HF moved from Git LFS + `hf_transfer` to **Xet** (`hf_xet`, Rust `xet-core`): content-addressed ~64 KB chunks, deduplicated into ~64 MB Xorbs, adaptive byte-range concurrency. On Xet repos `hf_transfer` is superseded unless `HF_HUB_DISABLE_XET=1`.
- HF's CDN is heavily peered with GCP. Measured **252 MB/s on L4, 286 MB/s on G4, 353 MB/s on A100** to disk [M]. `/dev/shm` often plateaus at 200–270 MB/s due to tmpfs lock contention; tuned sequential writes can reach 350–450 MB/s [G].
- Public/private repo downloads cost the **owner $0 egress**; pricing page guarantees “Egress and CDN included at no extra cost.” Rate limits apply to the **downloader's** token/IP, not the model author. See 03 for quotas and pricing.

## 3. Google Drive

- A **service account** is a free automated Google identity. Create a GCP project, enable `drive.googleapis.com`, create an IAM service account, export `credentials.json` — no credit card or billing account required. Share a folder in your personal Drive with the service-account email as **Viewer**. The service account's own 15 GB quota is irrelevant because the folder belongs to your 2 TB AI Pro Drive.
- A one-time `rclone authorize "drive"` refresh token also works headlessly, but a “Testing” OAuth consent screen expires in 7 days; publish it or use a service account.
- Drive v3 `alt=media` enforces per-stream shaping: **50–76 MiB/s** documented; tuned `--transfers 8` across 4 GB shards plateaus at **60–140 MB/s** [C/G]. `--drive-chunk-size 128M` speeds uploads, not downloads.
- **Drive quotas:** Google enforces roughly **1 TB/day egress per project** plus hidden per-file frequency limits. Reading an 80 GB model 12 times = 960 GB, at the cliff; repeated reads trigger a **24-hour HTTP 403 `downloadQuotaExceeded`** ban after ~15–25 pulls [G]. A mounted file counts as a download because the bytes still leave Drive's servers.
- Google Drive is fine for **4B–14B** models (3–8 GB; 50 pulls/day stays under 25% of the cap) and for a **cold vault**. It is a poor primary store for an 80 GB model pulled many times a day.

## 4. Kaggle Models

- Kaggle stores model/dataset blobs in GCS buckets. `kagglehub.model_download()` resolves to **signed GCS URLs**, so inside Colab it traverses Google's internal network at **200–350 MB/s [G]**.
- Use a **Kaggle Model** with individual uncompressed `.safetensors`/`.gguf` shards. The legacy `kaggle datasets download` CLI wraps everything in one `.zip`, throttles to **80–150 MB/s**, and needs double disk for decompression [G].
- Set `KAGGLEHUB_CACHE=/dev/shm/kaggle`; the default `~/.cache/kagglehub` writes to the slow root disk.
- Kaggle is the strongest zero-cost fallback when HF CDN peering congests.

## 5. Google Cloud Storage (GCS) and the $10 credit

- Google AI Pro gives **$10/month Google Cloud credits** via the Google Developer Program, applicable to GCS storage, egress, and API operations. It is a promotional balance on a linked billing account; a credit/debit card is still required. Each monthly grant is valid 12 months, consumed first-to-expire. The 5 TB included storage is **Google One**, not GCS.
- What $10 buys (US/Asia, approximate): ~380–500 GB Standard storage at $0.020–0.026/GB/mo; ~650 GB–1 TB Nearline; ~80–120 GB internet egress after the first 100 GB/mo free; up to ~2M Class A requests.
- **The 80 GB × 10 downloads/day example:** 800 GB/day = 24 TB/month. Tiered egress ≈ **$2,230/month** (~$75–96/day); Standard Tier routing still ≈ $1,900/month. The $10 credit is consumed in ~1–2 hours of downloading. This is why GCS is only attractive when the Colab VM is **in the same region** and traffic stays on the internal VPC (~$0/GB).
- GCS itself can deliver **200 MB/s–1+ GB/s** for large objects with multi-stream tools. Your local home connection is the bottleneck for local downloads (100 Mbps ≈ 11–12 MB/s; 300 Mbps ≈ 35 MB/s; 1 Gbps ≈ 100–115 MB/s).
- **Inside Colab:** `gcloud storage cp -r gs://bucket/model /content/model` can reach **300–800+ MB/s** and pull 80 GB in ~1.5–3 min [Reported]. But Colab's region is not selectable; a cross-region bucket incurs $0.02–0.12/GB ($1.60–$9.60 per 80 GB pull), which can exceed the credit.
- **`gcsfuse` caveat:** the Google AI Pro conversation recommends mounting GCS via gcsfuse to avoid local disk. The Fast Headless benchmarks show FUSE + mmap is catastrophic for LLM loading (page-fault storm, <0.5 tok/s). Use gcsfuse for sequential copy, or use `gcloud storage cp`/Run:ai streamer instead.
- **Double-cache trap:** loading `/content/model` with `from_pretrained` can duplicate files into `~/.cache/huggingface`. Pass `local_files_only=True` and avoid symlink/cache copies.

## 6. Cloudflare R2 / Backblaze B2

- R2 charges ~$0.015/GB/month storage with **$0 egress**. 24 TB/month of downloads costs $0 egress versus >$2,000 on GCS. But the free tier is only **10 GB**, so an 80 GB model needs a credit card (~$1.05/month storage). Throughput via s5cmd or Run:ai streamer is ~200–400 MB/s [G].
- Good if you already have a card and want zero egress; fails a strict “no billing card” constraint.

## 7. Colab-native mechanisms that do not work headlessly

- `colab drivemount` calls `drive.mount(path, timeout_ms=600000)` and triggers `dfs_ephemeral` OAuth consent in a browser for every new VM. Credentials cannot be cached across assignments. It blocks the websocket on consent.
- `colab upload` uses `PUT /api/contents`, base64-encodes the payload in memory, and sends it over the laptop uplink. Fine for notebooks, useless for 15–80 GB.
- There is **no** supported disk snapshot or custom image; every session boots Google's immutable base image.

## 8. Headless authentication and secret injection

- `from google.colab import userdata; userdata.get("HF_TOKEN")` inside `colab exec`/`colab run` **fails with `TimeoutException`**: the web Secrets sidebar needs a browser-proxied channel that the CLI does not have [C].
- Control plane: run `colab --auth=oauth2` once on the laptop. It persists refresh/access tokens to `~/.config/colab-cli/token.json` with OS keyring support (PRs #148/#182) and refreshes automatically.
- Data plane: `colab run`/`colab exec --env KEY=VALUE` injects environment variables into the remote Jupyter kernel without writing secrets to remote disk or shell history (PR #65, Issue #48).
- Store the token locally in the OS keychain (`secret-tool`, `pass`) or a `chmod 600` file; pass it as `colab exec --env HF_TOKEN="$(cat ~/.config/colab-cli/hf_token)" -f run_task.py`.

| Credential | Default lifespan | What breaks it | How to make it permanent |
| :--- | :--- | :--- | :--- |
| HF access token (`hf_…`) | **Infinite** [C] | Manual revocation or optional expiry | Fine-grained read token, no expiry, scoped to private model repos |
| GCP IAM service-account key (`credentials.json`) | **Infinite** [C] | Manual key deletion | RSA private key never expires; base64-encode and pass via `--env SA_KEY_B64=…` |
| Kaggle API token (`kaggle.json`) | **Infinite** [C] | Replacement/revocation | Download once; pass `KAGGLE_USERNAME`/`KAGGLE_KEY` via `--env` |
| Google OAuth2 refresh token (rclone/colab-cli) | Infinite in production; **7 days** in “Testing” [G] | Testing consent screen, password change, 6 months inactivity | Use a service account for Drive; colab-cli's built-in client auto-refreshes |

## 9. Can you pull 80 GB twenty times a day (1.6 TB/day)?

| Service | Allows 20×/day? | Speed to Colab G4 (`/dev/shm`) | 80 GB cold start | Monthly cost (48 TB/mo) | Catch |
| :--- | :--- | ---: | ---: | ---: | :--- |
| **HF Hub** (Xet/hf_transfer) | **Yes (best)** | 286–450 MB/s | ~4.0–4.7 min | **$0**, no card | Shard into ~16×5 GB; set Xet env vars before import |
| **Kaggle Models** (kagglehub) | **Yes (best backup)** | 200–350 MB/s | ~4.5–5.0 min | **$0**, no card | Upload as uncompressed Kaggle Model, not legacy zip Dataset |
| **GCS** ($10 credit) | Yes, high billing risk | 400–900 MB/s (same region) | ~2.0–3.5 min | $2.08/mo same region; up to $190+/mo cross-region | Cannot pick Colab region; egress can exceed credit |
| **Cloudflare R2** | Yes (zero egress) | 200–400 MB/s | ~4.0–5.5 min | ~$1.05/mo storage, $0 egress | Needs card above 10 GB |
| **Google Drive** (SA + rclone) | **No** — hard quota ban | 60–140 MB/s | 11–18 min | $0 | 403 `downloadQuotaExceeded` after ~15–25 pulls |

**Bottom line:** HF Hub primary, Kaggle Models backup, GCS only if you can keep the bucket and VM in the same region, Drive only for small models or a cold vault. But even when the storage is free, 20 cold starts/day is a compute-unit disaster; see 01 §5 and 06.
