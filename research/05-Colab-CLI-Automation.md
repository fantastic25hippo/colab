# 05 — Colab CLI Automation, Hazards and Ranked Blueprints

Consolidated from `Fast Headless Colab Weight Loading` §§5–8. Scope: driving `google-colab-cli` (v0.7.5) from a laptop for one-fresh-VM-per-task coding harnesses, with no GCP billing account. Authentication mechanics are in 02 §8; this document covers operational failure modes and end-to-end blueprints.

## 1. Engine cold-start and throughput comparison

| Engine | Install on fresh VM | Init/graph compile | Weight load (15 GB / 75 GB from `/dev/shm`) | 15 GB speed (L4 / A100 / G4) | 75 GB speed (G4 96 GB / A100-80) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **llama.cpp** (`llama-server` static binary) | **2–4 s** (pre-built CUDA tar.gz) [G] | **1–2 s** (zero JIT, instant CUDA init) [G] | **3 s / 14 s** (`--no-mmap` sequential DMA) [G] | **115–140 tok/s** MoE on L4; 95–220 on A100/G4 [G] | **120–160 tok/s** MoE; 50–70 dense + draft [G] |
| **vLLM** (`--load-format runai_streamer`) | **28–55 s** (`uv pip install vllm runai-model-streamer`) [G] | 35–80 s default (`torch.compile` + CUDA graphs); **8–12 s** with `--enforce-eager` [G] | **5 s / 18 s** (overlapped with download) [C/G] | **110–135 tok/s** MoE on L4; 100–250 on A100/G4 [G] | **130–175 tok/s** MoE + FP8/AWQ on G4 [G] |
| **SGLang** | **30–60 s** (`uv pip install sglang[all]`) [G] | 25–65 s (CUDA graph capture; faster with `--disable-cuda-graph`) [G] | **6 s / 22 s** from `/dev/shm` [G] | **115–145 tok/s** (RadixAttention + Eagle) on L4/A100 [G] | **135–185 tok/s** MoE + Eagle on G4 [G] |
| **TensorRT-LLM** | **60–120 s+** [G] | **300–900 s** unless a pre-built sm_120/sm_80 engine is cached [G] | **8 s / 25 s** (pre-built `.engine` only) [G] | **120–260 tok/s** [G] | **140–200 tok/s** (impractical due to engine build/size) [G] |
| **TPU engines** (`torch_tpu` vs `tpu-jax`) | 0–10 s (`torch_tpu` 0.2 pre-installed) [C] | **359 s** (`torch_tpu` defer-and-fuse first call); 25–60 s XLA compile [C/G] | Fails >16 GiB HBM compile peak (16.67 GB) [C] | **31.8–45.3 tok/s** (`torch_tpu`); **106–135 tok/s** (`tpu-jax` on smaller models) [C] | **Incompatible** (exceeds 16 GiB v5e-1 / 32 GB v6e-1) [C/G] |

For one-fresh-VM-per-task, **llama-server** and **vLLM with cached wheels + runai_streamer** give the fastest time-to-first-token. vLLM/SGLang are not pre-installed; pip install costs 30–60 s plus 35–80 s graph capture. Shipping a pre-compiled `llama-server` CUDA tarball (~60 MB) in the same HF repo as the GGUF weights lets the VM fetch binary + weights and start in under 4 s with zero Python dependency resolution.

## 2. The six operational hazards

| Hazard | Mechanism | Mandatory mitigation |
| :--- | :--- | :--- |
| **1. 5–6 min keep-alive 403 + 10–15 min idle pruning** (Issue #14) | CLI runs no background keep-alive between discrete `colab exec` calls. `keep_alive_assignment` sends `x-goog-user-project: 1014160490159`, which fails with `403 USER_PROJECT_DENIED`; idle sessions drop with 404/401 after 10–15 min. | Drive the model over a persistent `colab ssh --proxy-mode` tunnel, or run a background thread emitting `flush=True` heartbeats every 25 s. |
| **2. 30-second `colab exec` timeout + 100% CPU spin** (Issue #82) | `colab exec`/`colab run` default `--timeout 30`. On timeout, `jupyter-kernel-client` 0.9.0 clamps the deadline to `Event.wait(0)`, spinning one local core at 100% forever while the remote VM keeps burning CUs (fixed in PR #185). | Always pass `--timeout 600` (or higher), and print download/load progress every 5–10 s. |
| **3. Suppressed unassign errors on teardown** (CLI 0.7.4/0.7.5) | Ephemeral cleanup in `colab run` suppresses release errors; exit 0 or “Session terminated” is not proof that `POST /tun/m/unassign` succeeded. A leaked G4 drains 400 CU in 44.9 h. | Shell `trap` that runs `colab stop -s "$SESSION"` and verifies release via `colab sessions` (`GET /tun/m/assignments`). |
| **4. Silent hardware selector fallback** (Issue #47) | `resolve_runtime_options()` maps any unrecognized `--gpu` string to **A100** (5.30 CU/h) and any non-v5e1 `--tpu` string to **V6E1**. | Strictly validate `--gpu` in the wrapper: only uppercase `T4`, `L4`, `A100`, `G4`. |
| **5. 1-hour proxy token expiration** | `runtimeProxyInfo.token` authenticates VM requests and expires after `tokenExpiresInSeconds` (3,600 s). | For tasks approaching 50 min, refresh by polling `colab status`/`colab sessions`; refresh every ~40 min. |
| **6. Web-service hosting ToS vs SSH tunneling** | Colab FAQ prohibits public web-service hosting (ngrok/cloudflared) but permits SSH on paid plans with positive CUs. | Use `colab ssh -s <NAME> --proxy-mode` with OpenSSH `-L 18080:127.0.0.1:8080`, exposing the OpenAI API only on local loopback. |

Hazards 1 and 6 interact: discrete `colab exec` calls incur a **2.9 s polling round-trip per turn [M]** and leave the session vulnerable to 10–15 min idle pruning. A persistent SSH local port-forward keeps the WebSocket active, bypasses the 2.9 s exec overhead, streams tokens in real time, and complies with ToS.

## 3. Why `colab new --tpu v6e1` hangs for 9 minutes

- The CLI resolves the flags to `variant=TPU`, `accelerator=V6E1` and sends `POST https://colab.sandbox.google.com/tun/m/assign` (or `colab.research.google.com/tun/m/assign`).
- `AuthorizedSession` enforces a **120 s per-request HTTP timeout** (`COLAB_CLI_REQUEST_TIMEOUT=120`) with automatic retry + exponential backoff. **Four consecutive 120 s read timeouts plus backoff** produce the exact ~9-minute hang before abort.
- The Web UI dropdown renders **static tier entitlements**, not live capacity. Unlike H100 (gated behind Google AI Ultra at $99.99–$199.99/mo, immediately returns HTTP 400/412/503), v6e-1 requests for Pro/Pro+ enter a backend provisioning queue when Trillium slices are contended or cold-starting.
- CLI `POST /tun/m/assign` differs from the browser's two-step `GET /tun/m/assign` → XSRF token → `POST` handshake and sends CLI-specific headers (e.g., `x-goog-user-project`, Issue #14) that stall on newer accelerator pools.
- **Mitigation:** `export COLAB_CLI_REQUEST_TIMEOUT=25` before `colab new --tpu v6e1`, so capacity stalls fail in ~1.5 min. Or adopt a browser-initiated session: `GET https://colab.research.google.com/tun/m/assignments` returns endpoint + `runtime_proxy_info` (token, url, TTL); write those into `~/.config/colab-cli/sessions.json` and use `colab exec`. Even when provisioned, v6e-1 has only 32 GB HBM and requires XLA compilation on every fresh VM; L4/G4 are better for ephemeral coding.

## 4. Ranked end-to-end architectures

| Rank | Architecture | One-time setup | Per-run steps | Target hardware / model |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Private HF repo + pre-compiled `llama-server` + `/dev/shm` + `colab ssh` tunnel** | Create private HF repo + non-expiring fine-grained read token (`~/.config/colab-cli/hf_token`, `chmod 600`). Upload `llama-bin.tar.gz` (~60 MB) + 4.5 GB sharded `.gguf`. Verify `~/.ssh/id_ed25519.pub` for `colab ssh`. | `colab new -s $SESSION --gpu L4` (or G4); `colab exec --env HF_TOKEN=… --timeout 600`: remount `/dev/shm` (44G L4 / 150G G4), `snapshot_download` to shm, start `llama-server --no-mmap` on `127.0.0.1:8080`, unlink `.gguf` after `/health` 200. Tunnel `127.0.0.1:18080` via `colab ssh --proxy-mode`; run harness; `colab stop` + verify `colab sessions`. | **15 GB:** Qwen3-30B-A3B-Q4_K_M on **L4** (1.54 CU/h). **75 GB:** Llama-4-Scout-109B-A17B or Qwen-MoE on **G4** (8.90 CU/h). |
| **2** | **Private HF repo + vLLM + runai-model-streamer (zero-disk VRAM stream)** | Private HF repo + read token. Upload 4–5 GB `.safetensors` shards + a pre-packaged `vllm-wheels-cu128.tar` matching Colab's Python/PyTorch 2.11+cu128 image (avoids 45 s PyPI resolution). | Provision L4/G4/A100 `--high-mem`. `colab exec --env HF_TOKEN=… --timeout 600`: pull wheel bundle to shm, `uv pip install --no-index --find-links /dev/shm/wheels vllm runai-model-streamer`, launch `vllm serve <repo> --load-format runai_streamer --enforce-eager --port 8080`. Tunnel; run; teardown. | **15 GB:** MoE/Eagle `.safetensors` on **L4**. **75 GB:** MoE FP8/AWQ on **G4** or **A100 --high-mem**. |
| **3** | **Kaggle Models (`kagglehub`) + `/dev/shm` + `llama-server`** | Generate `kaggle.json`; upload uncompressed 4.5 GB `.gguf` shards + `llama-server` binary as a private **Kaggle Model** (not a zipped Dataset). | Provision L4/G4. `colab exec --env KAGGLE_USERNAME=… --env KAGGLE_KEY=…`: remount shm, `KAGGLEHUB_CACHE=/dev/shm/kaggle`, `kagglehub.model_download()`, start `llama-server --no-mmap`, unlink shards. Tunnel; teardown. | Same as Rank 1; independent GCS-backed fallback if HF CDN peering congests. |
| **4** | **Google Drive + IAM service account (rclone multi-shard to `/dev/shm`)** | Free GCP project, enable Drive API, create IAM service account, download `credentials.json`; upload 4 GB shards to personal Drive; share folder with service-account email as Viewer. | Provision L4/G4. Pass `SA_JSON_B64` via `colab exec --env`; remount shm; write `/dev/shm/sa.json`; `rclone copy --drive-service-account-file /dev/shm/sa.json --transfers 8 gdrive:models/ /dev/shm/models/`. Start `llama-server --no-mmap`; tunnel; teardown. | Viable for occasional **15 GB** runs on **L4**; not recommended for **75 GB** (11–18 min transfer + 403 quota bans). |

## 5. Reference implementation — Rank 1

```bash
#!/usr/bin/env bash
set -euo pipefail
SESSION="code-task-$(date +%s)"
GPU_TIER="${1:-L4}"          # "L4" (15 GB) or "G4" (75-80 GB)
SHM_SIZE="${2:-44G}"         # "44G" for L4 or "150G" for G4
MODEL_PREFIX="${3:-qwen3-30b-a3b}"
HF_TOKEN=$(cat ~/.config/colab-cli/hf_token)

cleanup() { colab stop -s "$SESSION" || true; colab sessions; }
trap cleanup EXIT

colab new -s "$SESSION" --gpu "$GPU_TIER"

colab exec -s "$SESSION" --env HF_TOKEN="$HF_TOKEN" --env SHM_SIZE="$SHM_SIZE" \
  --env MODEL_PREFIX="$MODEL_PREFIX" --timeout 600 << 'EOF'
import os, subprocess, time, urllib.request

# Set HF variables BEFORE importing huggingface_hub
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

subprocess.run(["sudo", "mount", "-o", f"remount,size={os.environ['SHM_SIZE']}", "/dev/shm"], check=True)
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="username/coding-models-private",
    local_dir="/dev/shm/bundle",
    allow_patterns=["llama-bin.tar.gz", f"{os.environ['MODEL_PREFIX']}*.gguf"],
    max_workers=8,
)
subprocess.run(["tar", "-xzf", f"{local_dir}/llama-bin.tar.gz", "-C", "/dev/shm/bundle"], check=True)

first_shard = sorted([f for f in os.listdir(local_dir) if f.endswith(".gguf")])[0]
proc = subprocess.Popen([
    "/dev/shm/bundle/llama-server", "-m", f"{local_dir}/{first_shard}",
    "-ngl", "999", "--no-mmap", "-fa", "--lookup-ngram-min", "3",
    "--host", "127.0.0.1", "--port", "8080",
], stdout=open("/tmp/llama.log", "w"), stderr=subprocess.STDOUT)

for _ in range(120):
    try:
        if urllib.request.urlopen("http://127.0.0.1:8080/health").status == 200:
            break
    except Exception:
        time.sleep(0.5)
for f in os.listdir(local_dir):
    if f.endswith(".gguf"):
        os.remove(os.path.join(local_dir, f))
print("SERVER_READY", flush=True)
EOF

ssh -f -N -L 18080:127.0.0.1:8080 \
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ProxyCommand="colab ssh -s $SESSION --proxy-mode" colab-vm

./run_coding_harness.py --api-base "http://127.0.0.1:18080/v1"
```

**Caveat:** this script uses `HF_HUB_ENABLE_HF_TRANSFER=1` and `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`, which the newer HF Xet document says are no-ops on Xet repos (see 03/11). For Xet, remove them and keep `HF_XET_HIGH_PERFORMANCE=1`; for non-Xet, keep `HF_HUB_DISABLE_XET=1` + `HF_HUB_ENABLE_HF_TRANSFER=1`.

## 6. Seconds from `colab new` to first token

| Ranked approach | VM boot | Env/binary/wheel setup | Weight transfer | Engine init + VRAM load | Total to first answer | Unproven |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **1A: HF + llama-server + shm, 15 GB on L4** | 4.7 s [M] | 2.5 s untar [G] | 55–65 s @~250 MB/s [M/G] | 4.5 s `--no-mmap` DMA [M/G] | **~67–77 s (1.1–1.3 min)** [G] | Exact hf_xet vs hf_transfer throughput on shm with sequential-write flag [C/G] |
| **1B: HF + llama-server + shm, 75 GB on G4** | 4.7 s [M] | 2.5 s untar [G] | 230–265 s @285–325 MB/s [M/G] | 14.0 s `--no-mmap` DMA [M/G] | **~251–286 s (4.2–4.8 min)** [G] | Sustained >100 tok/s requires MoE or draft speculation; dense 75 GB is ~19 tok/s [C/G] |
| **2: HF + vLLM + runai_streamer, 75 GB on G4/A100-80** | 4.7 s [M] | 10–15 s cached wheels [G] | 215–260 s streamed to VRAM [C/G] | 12.0 s `--enforce-eager` [G] | **~242–292 s (4.0–4.9 min)** [G] | Pre-built vLLM wheel compatibility with G4 CUDA 13.0/sm_120 without JIT fallback [C/G] |
| **3: Kaggle + shm, 15 GB L4 / 75 GB G4** | 4.7 s [M] | 3.0 s [G] | 65 s (15 GB) / 280 s (75 GB) [G] | 4.5 s / 14.0 s [G] | **~77 s / ~302 s** [G] | Multi-file GCS signed-URL concurrency limits for 16-shard models [G] |
| **4: Drive + service-account rclone, 15 GB L4 / 75 GB G4** | 4.7 s [M] | 6.0 s install rclone [G] | 150–240 s (15 GB) / 650–1,100 s (75 GB) [C/G] | 4.5 s / 14.0 s [G] | **~165–255 s / ~11–18 min** [G] | Number of 75 GB Drive pulls before 24 h `downloadQuotaExceeded` [G] |

## 7. Operational checklist

- [ ] `colab --auth=oauth2` once; store tokens in keyring; pass secrets via `--env`, never `userdata`.
- [ ] Validate `--gpu` is exactly uppercase `T4`/`L4`/`A100`/`G4`; validate `--tpu` is `v5e1` if intended.
- [ ] Always `--timeout 600` (or higher); heartbeat every 25 s; print progress every 5–10 s.
- [ ] Trap `EXIT` → `colab stop`; then `colab sessions` to confirm release.
- [ ] Use `colab ssh --proxy-mode` + local port-forward for long tasks; refresh proxy token every ~40 min.
- [ ] Remount `/dev/shm` before download; use `--no-mmap`; unlink shards after VRAM hydration.
- [ ] Prefer HF Hub → Kaggle → GCS (same region) → Drive; see 02.
- [ ] For v6e-1 probing, `COLAB_CLI_REQUEST_TIMEOUT=25`; expect capacity queues and XLA compile.
