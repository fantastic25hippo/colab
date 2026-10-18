# Bend Hands (shelved)

**Status:** DESIGN / IMPLEMENT LATER. Do not implement until asked.

**Target:** Linux Mint workstation + Colab GPU, Bend projects under `/home/sahil/Desktop/bend/projects/<project-name>/`.

GPU stack, launch, and shutdown defer to [`architecture.md`](architecture.md), [`inference.md`](inference.md), and [`ops.md`](ops.md).

**Out of scope:** "Bend inside of bend". Do not invent it here.

**Fallback if Colab is unused:** Bend work uses DeepSeek via API. No extra DeepSeek architecture or keys are specified.

---

## 1. Three CLIs

```
[ Sahil ]
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│ 1. Grok CLI (Build Lead)                                  │
│    Reads Bend planning docs, writes laws + objective list │
│    Hands off to Gemini, then STOPS                        │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ 2. Gemini CLI / agy (infra + auditor)                     │
│    Allocates Colab VM, boots llama-server, extracts URL   │
│    Spawns local Hermes with ephemeral config              │
│    Audits bend interpret / compile / checkup / PROOF      │
│    Unconditionally colab stop on every exit               │
│    FORBIDDEN from editing project files                   │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ 3. Hermes CLI + Colab Qwen 3.8 27B (Hands)                │
│    Brain: abliterated Qwen on Colab                       │
│    Tools run on local disk in projects/<name>/            │
│    Writes main.bend and PROOF.bend until objectives pass  │
└───────────────────────────────────────────────────────────┘
```

**Grok** plans and stops. No audit loop, no SuperGrok burn during coding.

**Gemini** never writes, patches, or modifies project files. If Gemini edits code, the split collapses.

**Hermes** is Hands only. Never Lead.

Rejected: Gemini as patcher, cutting Gemini, Colab bind-mount of local disk, High-RAM A100, raw API keys.

---

## 2. Bend paths

```
/home/sahil/Desktop/bend/          ← READ-ONLY snapshot (updated later)
├── docs/  examples/  getting-started/  reference/
├── README.md  WORKFLOW.md  INSTALL.md
└── projects/                      ← WRITABLE ROOT
    └── <project-name>/            ← HERMES CWD
        ├── LAWS.bend              ← human-owned, immutable
        ├── main.bend              ← worker implementation
        ├── PROOF.bend             ← worker proof
        └── <binary>               ← compile output stays here
```

Hermes `cwd` must be the project subdirectory so relative imports and `bend -o` stay there. Compiler: `~/.local/bin/bend`, runtime `~/.bend/` (GCC 13). `import Base` resolves from `~/.bend/current/bend2/base.bend`.

| Agent | Reads |
| :--- | :--- |
| Grok (plan) | `docs/guide.md`, `docs/laws-and-proofs.md`, `getting-started/agents.md`, `getting-started/quickstart.md` |
| Gemini (audit) | `getting-started/agents.md`, `INSTALL.md`, `WORKFLOW.md`, `docs/laws-and-proofs.md` |

---

## 3. GPU stack (when implementing)

Use the live worker from [`architecture.md`](architecture.md) and Playbook A from [`inference.md`](inference.md).

- Session: `qwen-worker`
- GPU: `colab new -s qwen-worker --gpu A100` (no `--high-mem`)
- Keep `colab exec --timeout 86400` attached with a 20–25 s heartbeat
- Idle watchdog 1800 s → `runtime.unassign()`
- Drive: `/content/drive/MyDrive/colab_llm/`

### Hermes ephemeral config

Do not edit `~/.hermes/config.yaml`. Launch with `HERMES_CONFIG=/tmp/hermes-colab.yaml`:

```yaml
model:
  default: "qwen3.8-27b"
  provider: "custom"
  base_url: "https://<subdomain>.trycloudflare.com/v1"
  context_length: 100000

compression:
  enabled: true
  threshold: 0.90
  target_ratio: 0.20
  protect_last_n: 20
  micro_compact: false
```

Compact at 90K. `micro_compact: false` so llama.cpp prefix cache survives across turns.

When a Hermes turn needs a tool, send `"tool_choice": "required"`.

---

## 4. Boot and shutdown

`colab exec -s qwen-worker --timeout 86400 -f start_server.py` stays attached:

1. Idempotent Drive mount (`os.path.ismount` before `drivemount`).
2. Start llama-server and cloudflared as children.
3. Print the trycloudflare URL, then heartbeat `/health` and `/slots` every 20–25 s.
4. Gemini waits on `GET /health` over the public URL, then spawns Hermes.

Two-tier stop:

1. Gemini `try/finally: colab stop -s qwen-worker` plus SIGINT/SIGTERM.
2. In-guest 1800 s idle watchdog → `runtime.unassign()`.

Reconnect without reload: `colab exec -s qwen-worker --timeout 86400 -f attach_heartbeat.py`.

CU rates and meter: [`ops.md`](ops.md).

---

## 5. File ownership and audit

| File | Owner | Rule |
| :--- | :--- | :--- |
| `LAWS.bend` | Human / Grok at spec time | Read-only for workers. No weakening laws to pass a proof. |
| `main.bend` | Hermes | Implementation. |
| `PROOF.bend` | Hermes | Inductive proofs that `main` satisfies `LAWS`. |

Prohibited in the delivered tree: `?TODO`, `@unsafe`, cyclic imports (`main` must not import `LAWS` or `PROOF`). Affine by default; multi-use `+x` with kind `Data`; erased `-x`. Use `!` at top-level parallel calls.

Gemini quality gate, inside `projects/<name>/`:

1. `bend main.bend`
2. `bend main.bend -o game && ./game`
3. `bend main.bend --checkup`
4. `bend PROOF.bend`

Non-zero exit → objective not met. Gemini nudges Hermes with the compiler output. Gemini still does not edit files.

---

## 6. Later: Unsloth QLoRA

4-bit QLoRA of this 27B is a future `colab exec` job that writes a merged GGUF into Drive. No training recipe in this spec. Outer Grok / Gemini / Hermes topology stays the same.

---

## 7. Implementation checklist (future session)

1. [ ] `colab whoami` and Resources panel (positive CU) — [`ops.md`](ops.md)
2. [ ] `start_server.py`: Drive mount, Playbook A from [`inference.md`](inference.md), 20 s heartbeat, 1800 s `runtime.unassign()`, print tunnel URL
3. [ ] Gemini harness: timeout ≥ 3600 s, `try/finally colab stop`, poll `/v1/health`, ensure `projects/<name>/` exists, write `/tmp/hermes-colab.yaml`, spawn Hermes with that cwd, audit until proofs pass
4. [ ] Confirm `LAWS.bend` untouched and `/home/sahil/Desktop/bend/` snapshot never written
