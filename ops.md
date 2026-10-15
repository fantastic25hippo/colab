# Ops

Account, Colab CLI, Drive, tunnel, heartbeat, and shutdown. Architecture: [`architecture.md`](architecture.md). Launch: [`inference.md`](inference.md). First sitting: [`first-boot.md`](first-boot.md).

---

## Account

| Item | Value |
| :--- | :--- |
| CLI identity | `zuluzain5@gmail.com` |
| Token | `~/.config/colab-cli/token.json` |
| Plan | Google AI Pro → **200 CU / month** |
| CLI | `google-colab-cli==0.7.4` at `/home/sahil/.local/bin/colab` |
| Auth | `--auth=oauth2`. Grant all OAuth scopes. |

Do not print or commit `token.json`.

`"You are not subscribed"` in the Colab UI is expected: AI Pro does not badge as Colab Pro. Paid-plan GPUs need a **positive** CU balance.

---

## CLI upgrade (once)

Installed baseline was 0.6.0. Upgrade directly to **0.7.4**. Skip 0.7.2 (broken `exec`) and yanked 0.7.3.

```bash
uv tool install --upgrade "google-colab-cli==0.7.4" \
  || pip install --user --upgrade "google-colab-cli==0.7.4"

colab --auth=oauth2 whoami
# Email: zuluzain5@gmail.com
```

0.6.0 drops the local session after 1 h (`RuntimeProxyInfo.token` never refreshes) while the VM keeps billing. 0.7.4 refreshes the token 5 minutes before expiry and adds `colab usage` (`GET /tun/m/ccu-info`). Web Resources panel remains a check.

There is **no** `colab port-forward`. Hybrid loop: CLI for `new` / `exec` / `stop`; web for CCU; in-guest trycloudflare for OpenAI `/v1`.

---

## Sitting loop

```bash
colab --auth=oauth2 whoami
colab --auth=oauth2 sessions
colab --auth=oauth2 usage          # also check the web Resources panel

colab --auth=oauth2 new -s qwen-worker --gpu A100
# never --high-mem

# SKU guard immediately:
colab --auth=oauth2 usage
# stop if burn > ~8 CU/hr, status HIGH_RAM, nvidia-smi ~80 GB, or host RAM > 110 GB

colab --auth=oauth2 drivemount -s qwen-worker /content/drive

colab --auth=oauth2 exec -s qwen-worker --timeout 86400 \
  --env LLAMA_API_KEY="$LLAMA_API_KEY" \
  -f start_server.py
# keep this attached

# every exit path, including Ctrl-C:
colab --auth=oauth2 stop -s qwen-worker
colab --auth=oauth2 sessions
```

`colab exec` default `--timeout` is 30 s. Expiry does **not** kill guest processes; it only detaches the laptop. Always pass `--timeout 86400`.

Google AI Pro has **no background execution**. Fire-and-exit exec is reaped in ~10–15 min even if llama-server is serving over trycloudflare. `start_server.py` must stay in a 20–25 s stdout heartbeat loop polling `/health` and `/slots`.

---

## Drive mount

Mount is interactive **every new VM** (`authtype=dfs_ephemeral`, `record=false`). Kernel timeout is **120 s**.

Browser `authuser=0` must be `zuluzain5@gmail.com`. Use a dedicated browser profile with only that Google account. Complete the grant within 60 s.

If `colab drivemount` hangs at `Resuming mount...`, open the session URL in that same profile (`colab url -s qwen-worker --open`).

Do not blindly re-run `drivemount`. Inside the guest, check `os.path.ismount('/content/drive')` first.

Canonical weights path: `/content/drive/MyDrive/colab_llm/`.

---

## Tunnel

```bash
cloudflared tunnel --url http://localhost:8080
```

Parse from stderr:

```
https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com
```

Wait 5–8 s (or resolve against 1.1.1.1) before printing the URL. trycloudflare NXDOMAIN caches for 30 min if you query too early.

Local client:

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://<slug>.trycloudflare.com/v1",
    api_key=os.environ["LLAMA_API_KEY"],
)
```

`--api-key` on llama-server is the documented way to lock the public tunnel. Prefer `LLAMA_API_KEY` via `colab exec --env`.

---

## Heartbeat and reconnect

`start_server.py` (when written) must:

1. Start llama-server and cloudflared as children.
2. Print `TUNNEL_READY_URL=https://….trycloudflare.com` with `flush=True`.
3. Every 20–25 s, poll `/health` and `/slots`, confirm both children are alive, print one heartbeat line with `flush=True`.

Attached exec + heartbeat prevents idle-kernel reaping. Exact 1 h WebSocket cutoff is unpublished; treat drops as expected. Reconnect **without reloading weights**:

```bash
colab exec -s qwen-worker --timeout 86400 -f attach_heartbeat.py
```

`attach_heartbeat.py` checks `/health` and `pgrep cloudflared`. If both are up, skip compile and load, and enter the heartbeat + watchdog loop. Reconnect within << 10 min. Never `colab restart-kernel` or `colab stop` during a reconnect.

---

## Idle watchdog

Reset the timer on change of `(id_task, n_decoded, n_prompt_tokens_processed, n_prompt_tokens)`. `n_decoded` goes to 0 when a turn ends, so a timer that only watches `n_decoded > 0` will kill a healthy pause.

Default idle is **1800 s** (30 min ≈ 2.69 CU on A100 40GB). 120 s kills normal coding pauses.

When the timer fires:

```python
from google.colab import runtime
runtime.unassign()
```

That deletes the VM assignment and stops CU burn. Laptop wrapper still runs `colab stop -s qwen-worker` so `sessions.json` is clean.

Also keep a 4-hour hard cutoff.

---

## SKU stop rules

Immediately `colab stop -s qwen-worker` (or `runtime.unassign()` from the guest) if any of:

- `nvidia-smi` name / memory shows ~80 GB (`A100-SXM4-80GB`)
- `colab status` shows `HIGH_RAM`
- host RAM > 110 GB (A100 High-RAM is ~167 GB; Standard A100 is ~83.5 GB)
- `colab usage` burn is High-RAM-class (~6.77+ CU/hr on an A100 request)

Silent Standard-shape 80 GB at ~6.77 CU/hr is still STOP. Extra VRAM is unused at 100K q8_0, and idle minutes cost more.

G4 is optional. `colab new --gpu G4` is often rejected on AI Pro. If it assigns, it is the Server Edition at 1,597 GB/s.

---

## CU meter

`colab usage` exists in 0.7.4. Web Resources panel is still a check:

1. Open [colab.research.google.com](https://colab.research.google.com/) as `zuluzain5@gmail.com`.
2. Connect or *Runtime → Change runtime type*.
3. RAM / Disk bar → remaining units and hourly burn.

Do not start a VM just to inspect quota. Recheck before any GPU sitting.

At 0 CU the backend kills the VM and the trycloudflare URL dies.

---

## Checklist before a sitting

1. `colab --auth=oauth2 whoami` → `zuluzain5@gmail.com`
2. `colab sessions` → none, unless you intend one
3. `colab usage` and Resources panel → positive CU, matching avatar
4. Dedicated browser profile is `authuser=0` for Drive
5. After work: `colab stop -s qwen-worker` and `colab sessions` empty
