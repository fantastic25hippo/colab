# Ops

Account, Colab CLI, Drive, tunnel, heartbeat, client integration, and shutdown.
Runbooks:
- G4 Hardware & Entitlement Probe: [`runbooks/g4-probe.md`](runbooks/g4-probe.md)
- Qwen 3.8 Flash-Next via SGLang: [`runbooks/flash-next-sglang.md`](runbooks/flash-next-sglang.md)
- Fallback: abliterated Flash-Next on G4: [`runbooks/fallback-abliterated.md`](runbooks/fallback-abliterated.md)

---

## Account

| Item | Value |
| :--- | :--- |
| CLI identity | `zuluzain5@gmail.com` |
| Token | `~/.config/colab-cli/token.json` |
| Plan | Google AI Pro → **200 CU / month** pool |
| CLI | `google-colab-cli==0.7.4` at `/home/sahil/.local/bin/colab` |
| Auth | `--auth=oauth2`. Grant all OAuth scopes. |

Do not print or commit `token.json`. Keep existing CLI identity intact.

`"You are not subscribed"` in the Colab UI is expected: Google AI Pro does not badge as Colab Pro. Paid-plan GPUs draw directly from the positive 200 CU monthly balance.

---

## CLI upgrade (once)

Installed baseline was 0.6.0. Upgrade directly to **0.7.4**. Skip 0.7.2 (broken `exec`) and yanked 0.7.3.

```bash
uv tool install --upgrade "google-colab-cli==0.7.4" \
  || pip install --user --upgrade "google-colab-cli==0.7.4"

colab --auth=oauth2 whoami
# Email: zuluzain5@gmail.com
```

0.6.0 drops the local session after 1 h (`RuntimeProxyInfo.token` never refreshes) while the VM keeps billing. 0.7.4 refreshes the token 5 minutes before expiry and adds `colab usage` (`GET /tun/m/ccu-info`). Web Resources panel remains an independent check.

There is **no** `colab port-forward`. Hybrid loop: CLI for `new` / `exec` / `stop`; web for CCU; in-guest trycloudflare for OpenAI `/v1`.

---

## Client integration (Pi harness & council-go)

The Pi coding-agent harness is the client of this `/v1` endpoint; later council-go will use the same OpenAI-compatible `/v1` over the existing trycloudflare tunnel with an API key.

Clients connect to the ephemeral public URL emitted at tunnel startup:

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="https://<slug>.trycloudflare.com/v1",
    api_key=os.environ["LLAMA_API_KEY"],
)
```

Both clients interact strictly via standard OpenAI `/v1/chat/completions` and `/v1/models` requests authenticated via the pre-shared Bearer token. Neither client requires internal architectural changes to switch between stock and abliterated Flash-Next backends.

---

## Sitting loop

### Primary (G4 Flash-Next)
Follow [`runbooks/g4-probe.md`](runbooks/g4-probe.md) first to verify entitlement and host `mlock`. If entitled, launch via [`runbooks/flash-next-sglang.md`](runbooks/flash-next-sglang.md):

```bash
colab --auth=oauth2 new -s qwen-g4 --gpu G4 --high-mem
colab --auth=oauth2 drivemount -s qwen-g4 /content/drive
# Immediate guard: stop unless RTX PRO 6000 (~95.6 GiB VRAM), High RAM shape (~177 GB / ~48 vCPUs), metered ~8.71–8.90 CU/hr
```

### Fallback (Abliterated Flash-Next on G4)
Follow [`runbooks/fallback-abliterated.md`](runbooks/fallback-abliterated.md):
If refusal mitigation is required, deploy an abliterated Flash-Next candidate on G4 using SGLang (same G4 provisioning and flags as primary, pointing `--model-path` to the selected abliterated checkpoint). See [`runbooks/fallback-abliterated.md`](runbooks/fallback-abliterated.md).

`colab exec` default `--timeout` is 30 s. Expiry does **not** kill guest processes; it only detaches the laptop. Always pass `--timeout 86400`.

Google AI Pro has **no background execution**. Fire-and-exit exec is reaped in ~10–15 min even if the server is serving over trycloudflare. `start_server.py` must stay in a 20–25 s stdout heartbeat loop polling `/health` and `/slots`.

---

## Drive mount & caching to cut boot

Mount is interactive **every new VM** (`authtype=dfs_ephemeral`, `record=false`). Kernel timeout is **120 s**.

Browser `authuser=0` must be `zuluzain5@gmail.com`. Use a dedicated browser profile with only that Google account. Complete the grant within 60 s.

If `colab drivemount` hangs at `Resuming mount...`, open the session URL in that same profile (`colab url -s <session> --open`).

Do not blindly re-run `drivemount`. Inside the guest, check `os.path.ismount('/content/drive')` first.

Canonical storage path: `/content/drive/MyDrive/colab_llm/`.

### Drive caching strategy
Cold boots on Google Colab burn expensive metered CUs on every session initialization. Caching artifacts directly on Google Drive cuts warm-boot latency to under 45 seconds:
- **Abliterated Fallback on G4**: If an abliterated candidate is selected ([`runbooks/fallback-abliterated.md`](runbooks/fallback-abliterated.md)), its weights (~105–112 GB) should be cached on Drive under `/content/drive/MyDrive/colab_llm/checkpoints/` similarly to the primary checkpoint to eliminate repeated large downloads.
- **G4 Flash-Next**: The 132.7 GB safetensors checkpoint (`nvidia/Qwen3.8-Flash-Next-NVFP4`) is cached under `/content/drive/MyDrive/colab_llm/checkpoints/nvidia/Qwen3.8-Flash-Next-NVFP4/` to eliminate a 10–20 minute download penalty (~1.5–3.0 CU) on every session.

---

## Tunnel & API key

```bash
cloudflared tunnel --url http://localhost:8080 > /tmp/cloudflared.log 2>&1 &
```

Parse from file log `/tmp/cloudflared.log` (never attach directly to `subprocess.PIPE`):

```
https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com
```

Persist the parsed URL to `/tmp/tunnel_url.txt`. Wait 5–8 s (or resolve against 1.1.1.1) before printing or using the URL; trycloudflare NXDOMAIN caches for 30 min if queried too early.

`--api-key` on the server (`llama-server` or SGLang) is mandatory to lock the public tunnel. Pass the secret via `colab exec --env LLAMA_API_KEY="$LLAMA_API_KEY"` so it does not appear in process tables (`ps aux`). `/health` remains accessible for monitoring without auth.

---

## Heartbeat and reconnect

To survive ephemeral client disconnections and prevent Google AI Pro's ~10–15 minute reaper from killing the VM, `start_server.py` implements strict subprocess isolation and heartbeat logging:

1. **Process Isolation (`start_new_session=True`)**: Both the model engine (`llama-server` or SGLang) and `cloudflared` must be spawned with `start_new_session=True` so they become independent process group leaders.
2. **File-Backed Logs (No `subprocess.PIPE`)**: Redirect child `stdout` and `stderr` directly to disk files (`/tmp/llama-server.log` or `/tmp/sglang.log`, and `/tmp/cloudflared.log`). Never attach `stdout`/`stderr` to `subprocess.PIPE`—an undrained 64 KB Linux pipe buffer fills up and triggers `SIGPIPE` or deadlocks when the laptop client detaches.
3. **Tunnel URL Persistence**: Poll `/tmp/cloudflared.log` to extract the `trycloudflare.com` URL, wait 5–8 s for DNS propagation, write it to `/tmp/tunnel_url.txt`, and print `TUNNEL_READY_URL=https://...` with `flush=True`.
4. **Heartbeat Loop**: Every 20–25 s, poll `/health` and `/slots` (or `/v1/models`), confirm child processes are alive, and print a heartbeat line to stdout with `flush=True`.

If the attached `colab exec` WebSocket drops after 1–3.5 hours, reconnect **without reloading weights**:

```bash
colab exec -s <session> --timeout 86400 -f attach_heartbeat.py
```

`attach_heartbeat.py` reads `/tmp/tunnel_url.txt`, verifies `/health` and `cloudflared`, re-emits `TUNNEL_READY_URL=...`, and resumes the heartbeat/watchdog loop. Reconnect within << 10 minutes. Never execute `colab restart-kernel` or `colab stop` during reconnect.

---

## Idle watchdog & stop/unassign

Reset the idle timer on change of `(id_task, n_decoded, n_prompt_tokens_processed, n_prompt_tokens)`. Because `n_decoded` drops to 0 when a generation turn completes, an idle timer that only watches `n_decoded > 0` will prematurely kill a normal pause between agent prompts.

Default idle timeout is **1800 s** (30 min ≈ 2.69 CU on A100 40GB; ~4.36 CU on G4).

When the timer fires:

```python
from google.colab import runtime
runtime.unassign()
```

`runtime.unassign()` deletes the VM assignment from within the guest and halts billing immediately. The local wrapper on the laptop then runs `colab stop -s <session>` to clean up local `sessions.json`.

A 4-hour hard cutoff is enforced so unattended sessions never run overnight.

---

## SKU stop rules

Immediately execute `colab stop -s <session>` (or `runtime.unassign()` from the guest) if any hardware guard fails:

### G4 SKU Guard
Stop immediately if:
- GPU is not confirmed as `NVIDIA RTX PRO 6000 Server Edition` (`sm_120`/`sm_120a`).
- VRAM reported by `nvidia-smi` is not ~95.6 GiB (e.g., fallback to 24 GB L4 or 40 GB A100).
- High RAM shape was not applied: host RAM is not in the ~177 GB class or the VM reports ~24 vCPUs instead of ~48. The ~90 GB Standard G4 is **not** approved for the primary FP8-PLE launch (Qiita measured High RAM G4 at 176.9 GB, 48 vCPUs; stop if host RAM < 150 GB).
- Metered burn rate via `colab usage` exceeds the expected High RAM baseline of ~8.71–8.90 CU/hr.

---

## CU meter & quota management

`colab usage` is available directly in CLI 0.7.4. Cross-check against the web Resources panel before and after every sitting:

1. Open https://colab.research.google.com as `zuluzain5@gmail.com`.
2. Connect or inspect *Runtime → Change runtime type*.
3. Inspect the RAM / Disk bar for remaining units and hourly burn.

Do not provision a VM merely to inspect quota. Check quota beforehand.

At 0 CU, the backend terminates the VM immediately and the trycloudflare URL dies.

---

## Checklist before a sitting

1. Workstation identity check: `colab --auth=oauth2 whoami` → `zuluzain5@gmail.com`.
2. Clean session state: `colab --auth=oauth2 sessions` → empty.
3. Quota check: `colab --auth=oauth2 usage` → positive CU balance from 200 CU pool.
4. Browser profile: Dedicated profile where `authuser=0` is `zuluzain5@gmail.com` for interactive Drive grant.
5. Provision target GPU:
   - Run G4 probe ([`runbooks/g4-probe.md`](runbooks/g4-probe.md)) to test G4 **High RAM** entitlement and host `mlock`.
   - If G4 High RAM passes, serve via [`runbooks/flash-next-sglang.md`](runbooks/flash-next-sglang.md) (or [`runbooks/fallback-abliterated.md`](runbooks/fallback-abliterated.md) if refusal mitigation is required).
   - If G4 High RAM is unentitled, rejected, or fails host `mlock`, Flash-Next on Colab is blocked; do not substitute a Standard ~90 GB G4 and proceed silently; stop and report.
6. Verify first-boot log greps, VRAM ceilings, and smoke curl gates.
7. After work: Ensure session is unassigned and `colab --auth=oauth2 sessions` is empty.
