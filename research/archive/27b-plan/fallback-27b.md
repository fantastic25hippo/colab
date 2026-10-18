# Runbook: Fallback Playbook A — Qwen 3.8 27B on Colab A100 40GB

Deterministic operational procedure for compiling, deploying, and serving Blackfrost `Qwen3.8-27B-ABLITERATED` on a Google Colab Standard `NVIDIA A100-SXM4-40GB` accelerator using `llama.cpp` + DFlash 2 + `ngram-map-k4v`.

Primary references:
- Blackfrost Model Card: https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF
- llama.cpp Repository: https://github.com/ggml-org/llama.cpp
- DFlash 2 Upstream Integration (PR #27816): https://github.com/ggml-org/llama.cpp/pull/27816
- NVIDIA A100 Datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
- Google Colab CLI: https://github.com/googlecolab/google-colab-cli

---

## 1. Stack Architecture & Target Files

| Component | Target Artifact / Pin | Role / Rationale |
| :--- | :--- | :--- |
| **Accelerator SKU** | `NVIDIA A100-SXM4-40GB` (~5.37 CU/hr, 1,555 GB/s HBM2) | 97.4% G4 bandwidth at 61.7% CU cost; verified fit for 100K context. |
| **Engine Commit** | `ggml-org/llama.cpp` @ `efe3626` / tag `b11552` | Master branch containing DFlash 2 block-diffusion support (PR #27816). |
| **Target Checkpoint** | `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (16,810,716,384 bytes) | SHA-256: `5d53637a59cfcd3a4d8354e254ffd44943e5a693da2405a3e228c62962355509`. |
| **Draft Model Source** | `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` (3,860,292,992 bytes) | SHA-256: `ca6dacc0bba87332b4eae6f103fa4c0c800ca871657e83d2aed658351b2c4fb3`. |
| **Quantized Drafter** | `dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` (1,143,006,592 bytes) | Quantized locally from BF16 sidecar; 81 blocks bit-identical to reference. |
| **Exact-Match Drafter** | `ngram-map-k4v` | Zero-VRAM prompt dictionary matching code edits; omit `ngram-mod`. |
| **Drive Storage** | `/content/drive/MyDrive/colab_llm/` | Shared persistent storage for weights, compiled binaries, and logs. |

---

## 2. Provisioning & Strict SKU Guard

Provision the session without `--high-mem`:

```bash
colab --auth=oauth2 new -s qwen-worker --gpu A100
```

### Strict SKU Guard (Immediate Halt Gate):
Execute an immediate environment check inside the container:

```bash
colab --auth=oauth2 exec -s qwen-worker --timeout 60 -- bash -c '
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
HOST_RAM_GB=$(free -g | awk "/^Mem:/ {print \$2}")

echo "GPU: ${GPU_NAME} (${VRAM_TOTAL} MiB)"
echo "Host RAM: ${HOST_RAM_GB} GB"

# Halt if assigned 80GB SXM4 or High-RAM host
if [[ "${VRAM_TOTAL}" -gt 45000 || "${HOST_RAM_GB}" -gt 110 ]]; then
  echo "CRITICAL SKU MISMATCH: Assigned High-RAM or 80GB SKU. Aborting." >&2
  exit 1
fi
'
```

Check metered rate on workstation:
```bash
colab --auth=oauth2 usage
```

**IMMEDIATE STOP CRITERIA**:
Halt session immediately (`colab --auth=oauth2 stop -s qwen-worker`) if:
- `nvidia-smi` reports ~80 GB (`A100-SXM4-80GB`).
- `colab status` shows `HIGH_RAM`.
- Host system RAM > 110 GB (A100 High-RAM is ~167 GB; Standard A100 is ~83.5 GB).
- `colab usage` reports ~6.77+ CU/hr burn rate (standard rate is ~5.37 CU/hr).

---

## 3. Persistent Drive Mount

Mount Google Drive interactively. Ensure browser profile is active as `zuluzain5@gmail.com` (`authuser=0`):

```bash
colab --auth=oauth2 drivemount -s qwen-worker /content/drive
```

Verify mount point:
```bash
colab --auth=oauth2 exec -s qwen-worker --timeout 30 -- python3 -c '
import os, sys
if not os.path.ismount("/content/drive"):
    sys.exit("Drive mount failed or hung.")
print("Drive successfully mounted at /content/drive.")
'
```

---

## 4. Compilation with Warm-Boot Portability

Fold in fixes from [`plan_audit_and_fixes.md`](plan_audit_and_fixes.md):
1. `-DBUILD_SHARED_LIBS=OFF`: Links `libllama` and `libggml` statically so precompiled binaries cached on Drive execute seamlessly on future warm VMs without `error while loading shared libraries`.
2. Fatbin targets: `-DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real"` (A100 `sm_80` and G4 `sm_120a`).
3. Binary targets: `--target llama-server llama-quantize llama-bench`.
4. Copy binaries to Drive and local `/usr/local/bin/`.

```bash
colab --auth=oauth2 exec -s qwen-worker --timeout 1800 -- bash -c '
set -euo pipefail

DRIVE_BIN=/content/drive/MyDrive/colab_llm/bin
mkdir -p "$DRIVE_BIN"

# Check if prebuilt binaries already exist on Drive
if [[ -f "$DRIVE_BIN/llama-server" && -f "$DRIVE_BIN/llama-quantize" && -f "$DRIVE_BIN/llama-bench" ]]; then
  echo "Found precompiled binaries on Drive. Skipping compilation."
  cp "$DRIVE_BIN/llama-server" "$DRIVE_BIN/llama-quantize" "$DRIVE_BIN/llama-bench" /usr/local/bin/
  chmod +x /usr/local/bin/llama-*
  exit 0
fi

echo "Compiling llama.cpp efe3626 with -DBUILD_SHARED_LIBS=OFF..."
git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
cd /tmp/llama.cpp
git checkout efe3626

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build --config Release -j "$(nproc)" \
  --target llama-server llama-quantize llama-bench

cp build/bin/llama-server build/bin/llama-quantize build/bin/llama-bench "$DRIVE_BIN/"
cp build/bin/llama-server build/bin/llama-quantize build/bin/llama-bench /usr/local/bin/
chmod +x /usr/local/bin/llama-*
echo "Compilation and installation complete."
'
```

---

## 5. Local Sidecar Quantization (Once)

Quantize the Blackfrost BF16 sidecar to `Q4_K_M`:

```bash
colab --auth=oauth2 exec -s qwen-worker --timeout 600 -- bash -c '
set -euo pipefail
DRIVE=/content/drive/MyDrive/colab_llm
BF16_PATH="$DRIVE/dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf"
Q4_PATH="$DRIVE/dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf"

if [[ -f "$Q4_PATH" ]]; then
  echo "Quantized sidecar already exists."
else
  echo "Quantizing BF16 sidecar to Q4_K_M..."
  /usr/local/bin/llama-quantize "$BF16_PATH" "$Q4_PATH" Q4_K_M
fi

ACTUAL_BYTES=$(stat -c%s "$Q4_PATH")
echo "Sidecar bytes: $ACTUAL_BYTES"
if [[ "$ACTUAL_BYTES" != "1143006592" && "$ACTUAL_BYTES" != "1143006752" ]]; then
  echo "ERROR: Unexpected sidecar byte length $ACTUAL_BYTES!" >&2
  exit 1
fi
sha256sum "$Q4_PATH"
'
```

---

## 6. Optional No-Spec Baseline via `llama-bench`

Run `llama-bench` before starting `llama-server` (`"speculative.n_max": 0` on a running server is ignored):

```bash
colab --auth=oauth2 exec -s qwen-worker --timeout 300 -- bash -c '
DRIVE=/content/drive/MyDrive/colab_llm
/usr/local/bin/llama-bench \
  -m "$DRIVE/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf" \
  -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 1
'
```

**Baseline Gate**: `≥ 50 tok/s` (healthy scaled band on A100 40GB is ~62–75 tok/s).

---

## 7. Server Launch & Process Isolation

Spawn `llama-server` and `cloudflared` using `start_new_session=True` with output redirected to file-backed logs (`/tmp/llama-server.log` and `/tmp/cloudflared.log`):

```bash
colab --auth=oauth2 exec -s qwen-worker --timeout 86400 \
  --env LLAMA_API_KEY="${LLAMA_API_KEY}" \
  -- python3 -c '
import subprocess, time, re, sys, os

DRIVE = "/content/drive/MyDrive/colab_llm"
LOG_SERVER = "/tmp/llama-server.log"
LOG_TUNNEL = "/tmp/cloudflared.log"
URL_FILE = "/tmp/tunnel_url.txt"

api_key = os.environ.get("LLAMA_API_KEY", "")

# 1. Start llama-server
# Flag notes:
# - Base -ngl is omitted: passing -ngl alongside --fit and -c aborts with "n_gpu_layers already set by user";
#   --fit on + -c 100000 automatically offloads all layers to A100 VRAM.
# - --spec-draft-ngl all pins all 5 DFlash 2 layers to GPU.
# - -bs is the official boolean flag for --backend-sampling (GPU-side logit sampling).
# - --no-warmup skips the initial dummy inference pass to accelerate startup.
# - Thinking-off flags (--reasoning off, --reasoning-format auto, chat-template-kwargs) are locked for non-thinking coding.
server_cmd = [
    "/usr/local/bin/llama-server",
    "-m", f"{DRIVE}/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf",
    "-md", f"{DRIVE}/dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf",
    "-a", "qwen3.8-27b",
    "--spec-type", "draft-dflash,ngram-map-k4v",
    "--spec-draft-n-max", "5",
    "--spec-draft-n-min", "0",
    "--spec-draft-ngl", "all",
    "-fa", "on",
    "-c", "100000",
    "--fit", "on", "--fit-target", "768",
    "--ctx-checkpoints", "4", "--checkpoint-min-step", "16384",
    "-b", "2048", "-ub", "512",
    "-ctk", "q8_0", "-ctv", "q8_0",
    "-bs", "--load-mode", "none", "--parallel", "1",
    "--no-warmup",
    "--jinja", "--reasoning", "off", "--reasoning-format", "auto",
    "--chat-template-kwargs", "{\"enable_thinking\":false}",
    "--temp", "0.0",
    "--api-key", api_key,
    "--host", "0.0.0.0", "--port", "8080"
]

server_out = open(LOG_SERVER, "w")
p_server = subprocess.Popen(
    server_cmd,
    stdout=server_out,
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print("Spawned llama-server (PID:", p_server.pid, ")")

# 2. Start cloudflared
tunnel_cmd = ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8080"]
tunnel_out = open(LOG_TUNNEL, "w")
p_tunnel = subprocess.Popen(
    tunnel_cmd,
    stdout=tunnel_out,
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print("Spawned cloudflared (PID:", p_tunnel.pid, ")")

# 3. Wait for tunnel URL
regex = re.compile(r"https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com")
tunnel_url = None
for _ in range(60):
    time.sleep(1)
    if os.path.exists(LOG_TUNNEL):
        with open(LOG_TUNNEL, "r") as f:
            match = regex.search(f.read())
            if match:
                tunnel_url = match.group(0)
                break

if not tunnel_url:
    sys.exit("Failed to parse trycloudflare URL.")

with open(URL_FILE, "w") as f:
    f.write(tunnel_url)

# Wait 8s for DNS propagation
time.sleep(8)
print(f"TUNNEL_READY_URL={tunnel_url}", flush=True)

# 4. Heartbeat loop
import urllib.request, json
start_time = time.time()
last_active = time.time()
idle_timeout = 1800  # 30 min
hard_timeout = 14400 # 4 hours hard cutoff
last_metrics = None

while True:
    time.sleep(25)
    now = time.time()
    
    # Check children
    if p_server.poll() is not None or p_tunnel.poll() is not None:
        print("Child process died. Exiting heartbeat.", flush=True)
        break
        
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8080/slots",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            slot = data[0] if isinstance(data, list) and len(data) > 0 else {}
            is_processing = slot.get("is_processing", False)
            metrics = (
                slot.get("id_task"),
                slot.get("n_decoded"),
                slot.get("n_prompt_tokens_processed"),
                slot.get("n_prompt_tokens"),
            )
            # Reset idle watchdog on is_processing OR when generation metrics change
            if is_processing or (last_metrics is not None and metrics != last_metrics):
                last_active = now
            last_metrics = metrics
    except Exception as e:
        pass

    idle_sec = now - last_active
    elapsed_sec = now - start_time
    print(f"HEARTBEAT: idle={idle_sec:.0f}s elapsed={elapsed_sec:.0f}s server_alive=True tunnel_alive=True", flush=True)

    if idle_sec > idle_timeout:
        print(f"IDLE TIMEOUT ({idle_timeout}s). Unassigning runtime...", flush=True)
        from google.colab import runtime
        runtime.unassign()
        break

    if elapsed_sec > hard_timeout:
        print(f"HARD CUTOFF ({hard_timeout}s). Unassigning runtime...", flush=True)
        from google.colab import runtime
        runtime.unassign()
        break
'
```

---

## 8. Log Grep & First-Boot Halt Conditions

Before sending traffic, audit `/tmp/llama-server.log`:

```bash
colab --auth=oauth2 exec -s qwen-worker --timeout 60 -- bash -c '
LOG_FILE=/tmp/llama-server.log

# 1. HALT if dense 64/65-layer KV
if grep -Eq "llama_kv_cache: size = .*100000 cells,[[:space:]]+6[45] layers" "$LOG_FILE"; then
  echo "FAIL: Dense 64/65-layer KV detected!" >&2
  exit 1
fi

# 2. Verify 16-layer KV allocation
grep -Eq "llama_kv_cache: size = .*100000 cells,[[:space:]]+16 layers.*K \(q8_0\):.*V \(q8_0\):" "$LOG_FILE"
grep -Eq "llama_memory_recurrent: size = .*64 layers" "$LOG_FILE"
grep -Eq "print_info: arch[[:space:]]+= qwen35" "$LOG_FILE"
grep -Eq "print_info: n_layer[[:space:]]+= 64" "$LOG_FILE"
grep -Eq "print_info: arch[[:space:]]+= dflash" "$LOG_FILE"
grep -Fq "common_speculative_impl_draft_dflash: - n_max=5, n_min=0, p_min=0.00" "$LOG_FILE"
grep -Fq "adding speculative implementation '"'"'ngram-map-k4v'"'"'" "$LOG_FILE"
grep -Fq "main: server is listening on http://0.0.0.0:8080" "$LOG_FILE"

# 3. HALT if ngram-mod is registered
if grep -Fq "adding speculative implementation '"'"'ngram-mod'"'"'" "$LOG_FILE"; then
  echo "FAIL: ngram-mod registered!" >&2
  exit 1
fi

# 4. Check VRAM usage (must be <= 31000 MiB)
USED_VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "Current VRAM used: ${USED_VRAM} MiB"
if [[ "${USED_VRAM}" -gt 31000 ]]; then
  echo "FAIL: VRAM used ${USED_VRAM} MiB exceeds 31,000 MiB gate!" >&2
  exit 1
fi

echo "ALL FIRST-BOOT LOG GATES PASSED."
'
```

---

## 9. Smoke Curls (Incorporating Plan Audit Fixes)

Set endpoint:
```bash
LLAMA_URL="http://127.0.0.1:8080"  # or "${TUNNEL_READY_URL}"
```

### Curl 1: One-Shot Speculative Code Generation (Cold Cache)
**Gate**: `predicted_per_second ≥ 95 tok/s`, `accept_pct ≥ 50%` (or mean draft length ≥ 3.50).

```bash
curl -sf "${LLAMA_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "cache_prompt": false,
    "temperature": 0.0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [
      {"role": "user", "content": "Write a self-contained Python 3 module implementing an iterative AVL tree with insert, delete, rebalance, and level-order traversal. Code only, no prose."}
    ]
  }' | jq '{
    smoke_step: "curl_1_oneshot",
    predicted_per_second: .timings.predicted_per_second,
    draft_n: .timings.draft_n,
    draft_n_accepted: .timings.draft_n_accepted,
    accept_pct: (if (.timings.draft_n // 0) > 0 then (100.0 * .timings.draft_n_accepted / .timings.draft_n) else 0 end)
  }'
```

### Curl 2: Restored Full Payload Testing `cache_n > 0` & `ngram-map-k4v`
Fold in fix from [`plan_audit_and_fixes.md`](plan_audit_and_fixes.md) §2.3:
- Preceded by a 1-token prefix seed turn to warm slot 0.
- Restores the full 28-line `TokenBucketRateLimiter` class in the assistant turn (eliminating the truncated `...` stub so `k4v` can extract 4-gram matches).
**Gate**: `predicted_per_second > Curl 1` and `cache_n > 0`.

```bash
# Step 2a: Seed prefix cache in slot 0
curl -sf "${LLAMA_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "cache_prompt": true,
    "temperature": 0.0,
    "max_tokens": 1,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [
      {"role": "user", "content": "Write a Python class TokenBucketRateLimiter with methods allow_request(self, cost: int = 1) -> bool, _refill(self) -> None, and snapshot_state(self) -> dict."}
    ]
  }' > /dev/null

# Step 2b: Follow-up turn hitting prefix cache and ngram-map-k4v
curl -sf "${LLAMA_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "cache_prompt": true,
    "temperature": 0.0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [
      {"role": "user", "content": "Write a Python class TokenBucketRateLimiter with methods allow_request(self, cost: int = 1) -> bool, _refill(self) -> None, and snapshot_state(self) -> dict."},
      {"role": "assistant", "content": "```python\nimport time\n\nclass TokenBucketRateLimiter:\n    def __init__(self, capacity: int, refill_rate: float) -> None:\n        self.capacity = capacity\n        self.refill_rate = refill_rate\n        self.tokens = float(capacity)\n        self.last_refill = time.monotonic()\n\n    def _refill(self) -> None:\n        now = time.monotonic()\n        elapsed = now - self.last_refill\n        if elapsed > 0.0:\n            self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)\n            self.last_refill = now\n\n    def allow_request(self, cost: int = 1) -> bool:\n        self._refill()\n        if self.tokens >= cost:\n            self.tokens -= cost\n            return True\n        return False\n\n    def snapshot_state(self) -> dict:\n        self._refill()\n        return {\"capacity\": self.capacity, \"tokens\": self.tokens, \"refill_rate\": self.refill_rate}\n```"},
      {"role": "user", "content": "Re-emit the complete TokenBucketRateLimiter class verbatim, renaming allow_request to consume_tokens and adding a reset_bucket(self) -> None method."}
    ]
  }' | jq '{
    smoke_step: "curl_2_k4v",
    predicted_per_second: .timings.predicted_per_second,
    cache_n: .timings.cache_n,
    draft_n: .timings.draft_n,
    draft_n_accepted: .timings.draft_n_accepted
  }'
```

### Curl 3: Function / Tool Calling

```bash
curl -sf "${LLAMA_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "Check the disk space of /content in gigabytes."}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_disk_space",
        "description": "Get disk usage for mount",
        "parameters": {
          "type": "object",
          "properties": {
            "mount": {"type": "string"}
          },
          "required": ["mount"]
        }
      }
    }],
    "tool_choice": "required",
    "temperature": 0.0,
    "max_tokens": 128
  }' | jq '{
    finish_reason: .choices[0].finish_reason,
    tool_calls: .choices[0].message.tool_calls
  }'
```
Pass when `finish_reason` is `"tool_calls"`.
