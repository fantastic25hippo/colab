# 06 — Agent Workflows on Colab: Council, Warm Daemons and Pipelining

Consolidated from `Optimizing AI Agent Colab Workflows` (2026-10-27) and the harness-related parts of the “Read both of these software…” conversation (2026-10-28). The setting: a local “Council” harness (Grok + Gemini debate, then dispatch a worker) uses a Colab GPU as the local-model worker, with 200 compute units/month and no GCP billing.

## 1. The original failure loop

- Council debates a plan, dispatches a Colab worker, the worker writes code, then **terminates the Colab session** to stop CU billing while Council reviews.
- On rejection, Council re-invokes the worker → **new OAuth prompt** and **new weight download/load** every round.
- `google.colab.drive.mount()` requires an interactive browser OAuth popup; `userdata` in a headless CLI also fails. This halts the pipeline.
- Downloading or mounting 40–80 GB per sub-task is both slow and quota-risky. The user's 4B model on L4/T4 ran at 40–50 tok/s once loaded; the problem was startup, not generation.

## 2. Zero-click Drive auth

Two free options, neither needs a billing account or credit card:

1. **rclone + Colab Secrets (fastest setup).** On the local Linux Mint machine, `rclone config` once for Google Drive; copy `~/.config/rclone/rclone.conf` (contains a persistent OAuth refresh token) into Colab Secrets as `RCLONE_CONFIG`, or have Council inject it. In the worker:
   ```python
   import os, subprocess
   from google.colab import userdata
   os.makedirs("/root/.config/rclone", exist_ok=True)
   with open("/root/.config/rclone/rclone.conf", "w") as f:
       f.write(userdata.get("RCLONE_CONFIG"))
   subprocess.run("apt-get install -y rclone && mkdir -p /content/drive", shell=True)
   subprocess.Popen(["rclone", "mount", "gdrive:", "/content/drive", "--vfs-cache-mode", "writes"])
   ```
   A 7-day expiry applies only if the GCP OAuth consent screen is left in “Testing”; publish it or use a service account.
2. **Service account.** Create a free GCP project, enable Drive API, create an IAM service account, download its JSON key. Share the Drive folder containing the weights with the service-account email (`...@....iam.gserviceaccount.com`) as Viewer. A static cryptographic key means no popup, no password reset, no 7-day expiry. The service account's own 15 GB quota does not matter because the folder belongs to the user's Drive.

## 3. “The model is in Drive” does not avoid the transfer

- Google Drive servers and Colab GPU servers are **different machines**. A GPU cannot execute weights sitting on Drive; every byte must be copied into VRAM on every new VM.
- The Drive mount is a **FUSE network shortcut**, not local disk. `from_pretrained("/content/drive/...")` and `rclone copy` both count as an 80 GB download from Drive's perspective.
- Drive FUSE random-read I/O is slow and rate-limited: **40–80 MB/s** typical; an 80 GB load takes **12–20 min**. A 4B model (3–8 GB) takes 15–45 s and feels fast; a 70B+ model is 10–20× larger.
- **Fix:** do not load directly from the mount. `rclone copy gdrive:model /dev/shm/model --transfers=8` (or tuned flags below) pulls to RAM first, then load from `/dev/shm`. This cuts Drive→GPU time ~4–5×.
- Colab never uses the user's home upload bandwidth: the transfer is Drive→Colab inside Google's data centers. The laptop only sends prompts and receives tokens.

## 4. Warm inference daemon with an idle kill-switch

Instead of kill/restart per sub-task:

1. **Startup once per work session:** boot the VM, pull weights once, load into VRAM, start an OpenAI-compatible server (vLLM, llama-cpp-python, FastAPI) with a `/health` endpoint.
2. **Expose it:** free Cloudflare Tunnel (`cloudflared`) or ngrok, or `colab ssh --proxy-mode` with a local port-forward. The harness gets a stable URL.
3. **Keep warm across debates:** when the worker returns code, leave the server running while Grok/Gemini review (seconds). On rejection, the model is already in VRAM: zero auth, zero reload, instant execution.
4. **Terminate only on final approval or timeout:** an internal Python timer (e.g., 5–30 min with no requests) calls `runtime.unassign()`; or Council calls it when the whole task is done. This stops CU billing immediately.

```python
from google.colab import runtime
runtime.unassign()   # disconnect and free the GPU from inside the code
```

**Idle watchdog:** reset a timer on every request; if no request for N minutes, `runtime.unassign()`. This protects the 200-CU budget during long human breaks while keeping the model warm during active work.

## 5. Compute-unit math: restart vs keep warm

Example: 3 revision rounds; 3 min load + 1 min generate each; Council debate 45 s (0.75 min) per round.

| Strategy | Billed GPU runtime | Total |
| :--- | :--- | ---: |
| **Terminate & restart every round** | 3 × (3 min load + 1 min generate) | **12.0 min** |
| **Keep warm via tunnel** | 3 min initial load + 3 × 1 min generate + 2 × 0.75 min idle debate | **7.5 min** |

Rule of thumb:
- **Keep GPU warm if model startup time > Council debate time.**
- **Terminate & restart if model startup time < Council debate time.**
- If a 4B model starts in 15 s and Council debates 90 s, restarting saves CUs. If a large model takes 3 min to load and Council debates 90 s, keeping warm saves ~40% even on A100/G4.
- At 8.90 CU/h, 20 cold starts/day × 4.5 min = 90 min/day = **13.35 CU/day**; 200 CU is exhausted in ~15 days before any generation. Batch into 2–3 sessions/day instead.

## 6. Rclone tuning and maximum model size on G4

```bash
rclone copy gdrive:model /dev/shm/model \
  --transfers=12 \
  --multi-thread-streams=8 \
  --multi-thread-cutoff=64M \
  --drive-chunk-size=128M
```

- `--transfers=12` = parallel files. Google Drive enforces roughly **10–12 file requests/s** per user; 32 or 64 triggers `403 Rate Limit Exceeded` and a cooldown, making it slower than 8. 12–16 is the sweet spot.
- `--multi-thread-streams=8` = parallel chunks inside one large `.gguf`; `--multi-thread-cutoff=64M` enables it for files above 64 MB.
- `--drive-chunk-size=128M` = fewer API calls (default 8 MB).
- G4 has 96 GB VRAM; reserve ~8–14 GB for CUDA context + KV cache, so realistic max weights **~84 GB** (70–72B FP8, or 110–123B 4/5-bit), 96 GB absolute ceiling.
- Tuned rclone inside GCP sustains **400–600 MB/s** from Drive before the per-account shaper. `/dev/shm` → G4 VRAM + engine init runs at ~3.5 GB/s effective (~25–30 s).

| Transfer stage | 84 GB realistic max | 96 GB absolute ceiling |
| :--- | ---: | ---: |
| Standard `drive.mount` (~65 MB/s) — do not use | ~22 min | ~25 min |
| Tuned rclone → `/dev/shm` @400 MB/s | 3 min 35 s (215 s) | 4 min 05 s (245 s) |
| Tuned rclone → `/dev/shm` @600 MB/s peak | 2 min 23 s (143 s) | 2 min 44 s (164 s) |
| `/dev/shm` → G4 VRAM + engine init | ~25 s | ~30 s |
| **Total cold start** | **~2.8–4.0 min** | **~3.2–4.6 min** |

Any model under ~40 GB (fits L4 24 GB or A100 40 GB) loads in under 2 min with this command. An 84–96 GB model on G4 takes 3–4 min, over a 2-min cutoff; do not restart a G4 between revision rounds.

## 7. Google Drive quota reality

- Google enforces a **1 TB/day egress limit per project** plus hidden per-file frequency limits. An 80 GB model loaded 12 times = **960 GB**, at the cliff. Repeated reads trigger a **24-hour HTTP 403 `downloadQuotaExceeded`** after ~15–25 pulls [G].
- Mounting Drive and loading directly **still counts** as an 80 GB download. Ten session restarts/day = 800 GB/day.
- Safe pattern: **one or two 80 GB pulls/day** (80–160 GB) stays well under 1 TB. That means one long warm session per coding block, not one session per sub-task.
- For 70B+ models pulled repeatedly, use **Hugging Face CDN**, not Drive: HF is built for cloud GPU nodes pulling 80 GB all day, has no daily download ban, uses 0 MB of home internet and 0 MB of Drive quota, and needs only a static `HF_TOKEN` for gated repos.
- A 4B–14B model in Drive is safe: 3–8 GB × 50 pulls/day is <25% of the 1 TB cap.

## 8. Drive vs Hugging Face

| Feature | Drive model folder | HF `hf_transfer`/Xet |
| :--- | :--- | :--- |
| Uses home internet? | No (0 MB) | No (0 MB) |
| Auth hassle | Rclone token / service account | None for public; static `HF_TOKEN` for gated |
| Ban/quota risk | High if >10 loads/day (1 TB/day) | Zero (Colab allows unlimited incoming CDN pulls) |
| 80 GB speed | ~3–4 min tuned rclone / ~20 min standard mount | ~1.5–2.5 min [Reported; newer docs say ~4–4.7 min cold start] |
| Best use | Custom fine-tuned weights, 4B–14B | Large open-weight models, 32B–120B |

## 9. Pipelined “Plan A → B → C” execution

The user's design: split the task into A/B/C; as soon as A is agreed, send it to the worker; while the GPU generates A, Council debates B/C; when A returns, immediately send B while evaluating A. This is **CPU/GPU double-buffering** and it removes GPU idle time.

- **Soft deadline** (e.g., turn 3 or 90 s): inject “debate timer at 75%; extract consensus into `PARTIAL_PLAN_1` so the worker can start; list only unresolved conflicts.”
- **Hard deadline** (e.g., turn 4 or 150 s): force a tie-breaker (single synthesizer prompt choosing the conservative option, or priority to the agent with the passing test spec) so a warm GPU never waits on a deadlock.
- **Pre-warm the GPU before agreement finishes:** the moment Council decides the task needs the heavy worker, boot the Colab VM in the background while the first debate runs. The model finishes loading as Part A locks in.
- **2-minute watchdog:** if the model is idle for 2 min, send a micro-task (file summary, AST/interface extraction, test scaffolding) to keep the GPU busy and the tunnel socket active; this also prevents the CLI's 10–15 min idle pruning.
- **Prefix caching:** Plan B/C share the same system prompt/repo context (10K–40K tokens). For llama-server use `--cache-reuse 256 -ctk q8_0 -ctv q8_0`; for vLLM use `--enable-prefix-caching --kv-cache-dtype fp8`. Plan B/C skip prefill and stream in <50 ms.

## 10. Extra safeguards from the “Read both” follow-up

- **More than 16 shards is slower, not faster.** 16×5 GB for 80 GB is the sweet spot. `hf_xet`/`hf_transfer` already parallelize inside each file; 50+ small shards add serialized HTTP/ETag handshakes and tmpfs lock contention. Use ~4.5–5.0 GB shards, `max_workers=8`, and sequential reconstruction.
- **180 GB MoE trap:** MoE reduces active compute, **not VRAM storage**. All experts must be resident because the router can pick any of them. A 180 GB BF16 model does not fit 96 GB VRAM or ~155 GB `/dev/shm`; CPU-offloading half the experts drops speed from >130 tok/s to <3 tok/s. Quantize to FP8 (~80–85 GB) or Q4_K_M/NVFP4 (~42–48 GB). FP8 fits G4 with 11–16 GB KV headroom at 130–175+ tok/s; Q4 (~45 GB, 9 shards) cold-starts in ~2 min and also fits A100-80.
- **GPU selector bug:** validate uppercase `T4`/`L4`/`A100`/`G4`; unrecognized strings silently map to A100 (5.30 CU/h).
- **Proxy token:** `runtimeProxyInfo.token` has a 1-hour TTL; refresh `colab sessions` every ~40 min on long pipelines.
- **Kaggle cache:** set `KAGGLEHUB_CACHE=/dev/shm/kaggle` before importing `kagglehub`; otherwise it writes to slow `/content`.

## 11. Caveats

- This document preserves the 2026-10-27 recommendations; some numbers conflict with the newer Fast Headless measurements: Drive 15–20 min vs 11–18 min; HF 80 GB 90–150 s vs 4.0–4.7 min; A100 “~15 CU/h” vs measured 5.30; T4 “~1.96 CU/h” vs measured 1.07. See 11.
- The warm-daemon architecture is sound; exact speeds depend on the model, shard layout, Colab region, and current Drive/HF throttling.
- `runtime.unassign()` is the correct in-code teardown; still verify the session is gone with `colab sessions`.
