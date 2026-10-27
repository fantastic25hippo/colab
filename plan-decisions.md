# Plan Decisions: Qwen 3.8 Flash-Next on Colab G4

**Status:** Decision register / plan of record  
**Last updated:** 2026-10-22  
**Scope:** Every material decision made for the Google Colab Qwen 3.8 Flash-Next worker plan.  
**Precedence:** This register summarizes the canonical set: [`README.md`](README.md), [`decision.md`](decision.md), [`claims-ledger.md`](claims-ledger.md), [`ops.md`](ops.md), and [`runbooks/`](runbooks/). If this document conflicts with those, the canonical documents win. `research/` is archive-only.

**Status key**

| Status | Meaning |
| :--- | :--- |
| **Decided** | Locked plan of record. |
| **Gated** | Decided, but execution requires a named probe to pass. |
| **Unverified** | Cannot be resolved without live measurement. Not assumed. |
| **Rejected** | Evaluated and deliberately excluded. |

---

## 1. Decision Summary

| ID | Area | Decision | Status |
| :--- | :--- | :--- | :--- |
| **D01** | Purpose | Build a private OpenAI-compatible coding worker for the Pi harness, later council-go. | Decided |
| **D02** | Account | Use Google AI Pro (`zuluzain5@gmail.com`) with the 200 CU/month pool. | Decided |
| **D03** | Provider | Run on Google Colab. Do not build or rent separate GPU infrastructure. | Decided |
| **D04** | GPU | Use Colab **G4**: NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM. | Decided |
| **D05** | **Host shape** | **Select the G4 High RAM option (`--high-mem`) for the entire plan.** The ~90 GB Standard G4 shape is not approved. | **Decided** |
| **D06** | Topology | Single GPU, TP=1. Multi-VM tensor parallelism is impossible on Colab and is not part of the plan. | Decided |
| **D07** | Primary model | Serve `nvidia/Qwen3.8-Flash-Next-NVFP4`; `RadixArk/Qwen3.8-Flash-Next-NVFP4` is the accepted alternate. | Decided |
| **D08** | Precision | Use the native NVFP4 mixed-precision checkpoint. Do not pass `--quantization`. | Decided |
| **D09** | Runtime | Serve with SGLang only. TensorRT-LLM, vLLM-only, and llama.cpp are not the serving path. | Decided |
| **D10** | Memory strategy | Offload the ~47.7 GiB FP8 PLE table to pinned host RAM with `--ple-offload-embedding`. | Decided |
| **D11** | Speculation | Enable MTP via the NEXTN speculative flags. | Decided |
| **D12** | KV cache | Use FP8 KV cache. Native 262,144-token context fits and is approved. | Decided |
| **D13** | Launch path | Prefer Docker with `--ulimit memlock=-1`; allow native launch only if native `mlock`/CUDA pinning passes. If neither works, stop and report. | Gated |
| **D14** | Checkpoint cache | Cache checkpoints on Google Drive at `/content/drive/MyDrive/colab_llm/checkpoints/`. | Decided |
| **D15** | Fallback | Use an abliterated Flash-Next checkpoint only if stock refusals block required work. Three candidates are shortlisted, all unverified. | Gated |
| **D16** | API | Expose an OpenAI-compatible `/v1` endpoint over a trycloudflare tunnel with a Bearer API key. | Decided |
| **D17** | Clients | Pi harness and council-go connect via standard OpenAI-compatible requests. No client-specific backend changes. | Decided |
| **D18** | Session survival | Keep `colab exec` attached with a 20–25 s heartbeat; use file-backed logs and process isolation. | Decided |
| **D19** | Shutdown | Enforce a 30-minute idle watchdog and a 4-hour hard cutoff; always unassign/stop. | Decided |
| **D20** | SKU guards | Stop immediately on any hardware, RAM-shape, VRAM, or billing-rate mismatch. | Decided |
| **D21** | CLI | Use `google-colab-cli==0.7.4` with `--auth=oauth2`. | Decided |
| **D22** | Probe order | Run the G4 probe before any checkpoint staging or launch. Probe must stay under ~10 minutes / ≤1.45 CU. | Decided |
| **D23** | Economics | Budget at 8.71 CU/hr, giving ~22.96 hours from 200 CU and ~560–943 tasks per pool (170 with verbose thinking). | Decided |
| **D24** | Acceptance | Health, model listing, coding completion, and tool-calling smoke tests must pass. | Decided |
| **D25** | Rejected models | GLM-5.3-Flash-NVFP4, dense 27B, and multi-GPU clusters are out. | Rejected |
| **D26** | Rejected runtimes | TensorRT-LLM, llama.cpp + DFlash 2, and multi-VM TP are out. | Rejected |
| **D27** | Open items | Seven live unknowns remain; each has a pass/fail gate and cannot be assumed. | Unverified |

---

## 2. Objective and Workload

### D01 — Private coding worker
**Decided.** The target is a private, self-hosted, OpenAI-compatible inference worker for coding-agent use. The first client is the Pi coding-agent harness; council-go will later use the same endpoint.

- Traffic is operator-driven, not a public multi-tenant API.
- The worker must support chat completions and tool/function calling.
- The worker is not being built to maximize generic benchmark throughput; it is being built to serve agentic coding sessions economically on Colab.
- The endpoint contract is standard `/v1/chat/completions` and `/v1/models`, authenticated with a pre-shared Bearer token.

### D02 — Account and billing pool
**Decided.** Use the existing Google AI Pro account `zuluzain5@gmail.com` and its **200 Compute Unit (CU)/month** pool.

- The plan uses the existing account; no account migration or plan switch is in scope.
- The Colab UI message “You are not subscribed” is expected for Google AI Pro and does not mean the GPU pool is unavailable.
- The token at `~/.config/colab-cli/token.json` must never be printed or committed.

### D03 — Provider
**Decided.** Google Colab is the platform of record. The plan does not provision a standalone GPU host, a different cloud, or an on-premises machine.

---

## 3. Hardware Decisions

### D04 — G4 is the target GPU
**Decided.** Use Colab **G4**, which is the NVIDIA RTX PRO 6000 Blackwell Server Edition.

| Property | Decision basis |
| :--- | :--- |
| Architecture | GB202, `sm_120` / `sm_120a`, 5th-gen Tensor Cores |
| Usable VRAM | ~95.60 GiB (97,896 MiB reported) of 96 GB physical |
| Memory bandwidth | 1,597 GB/s GDDR7, 512-bit bus |
| FP4 compute | 2.0 PFLOPS dense / 4.0 PFLOPS with 2:4 sparsity |
| FP8 compute | 1,000 TFLOPS dense / 2.0 PFLOPS with 2:4 sparsity |
| Billing | ~8.71 CU/hr standard; 8.90 CU/hr observed in later High RAM samples |

The G4 is chosen because it is the only Colab GPU that combines 96 GB VRAM, native NVFP4 execution, and enough bandwidth to serve Flash-Next at interactive speed on one card.

### D05 — High RAM is mandatory for the entire plan
**Decided and non-negotiable within this plan.**

- Provision with `--gpu G4 --high-mem`; in the web UI, enable **High RAM**.
- Required shape: **~176.9 GB host RAM (~177 GB) and ~48 vCPUs**.
- Standard G4 shape is **~90 GB RAM and ~24 vCPUs** and is **not approved** for this workload.
- Reason: `--ple-offload-embedding` must pin ~47.7 GiB of the 51.2B-parameter FP8 PLE table in host RAM. The ~177 GB shape leaves >120 GB free; the ~90 GB shape does not provide the approved headroom.
- The High RAM shape also carries the expected billing rate of ~8.71–8.90 CU/hr.

**Hard stop:** if the VM reports less than ~160 GiB host RAM or ~24 vCPUs instead of ~48, it is the Standard shape. Stop the VM and report; do not continue under the ~177 GB assumption and do not silently substitute Standard G4.

**Hard stop:** if `--high-mem` is rejected, gated, or unavailable on this AI Pro account, the primary Flash-Next launch is blocked. Stop and report.

### D06 — Single GPU only
**Decided.** The deployment is strictly **TP=1** on one G4.

- Colab exposes isolated single-GPU VMs with no NVLink/RoCE cross-VM fabric.
- Multi-VM `tp=2` is not possible on Colab.
- The sample `--tensor-parallel-size 8` in NVIDIA's model card is an HGX B200 reference command, not a requirement for G4.
- Any model that requires more than one 96 GB GPU is out of scope for this plan.

---

## 4. Model Decisions

### D07 — Primary checkpoint
**Decided.** The primary model is:

- **`nvidia/Qwen3.8-Flash-Next-NVFP4`** — first choice.
- **`RadixArk/Qwen3.8-Flash-Next-NVFP4`** — accepted alternate with the same architecture and launch shape.

Why this model:

- 124.91B base MoE, 512 fine-grained routed experts, top-10 routing, 1 shared expert.
- ~6.0B parameters active per token (~2.6B MoE + ~3.4B shared/attention/embeddings).
- 51.2B-parameter FP8 N-gram PLE table (~47.7 GiB), offloadable to host RAM.
- 1-layer FP8 MTP head (~2.76B physical parameters).
- 48 layers: 36 Gated DeltaNet linear-recurrent layers + 12 full-attention layers with QSA.
- 262,144-token native context.
- Official ModelOpt v0.46.0 release; public and ungated.
- Verified single-RTX-PRO-6000 TP=1 recipe in the official SGLang cookbook.

Checkpoint staging facts:

- Download size: **132.68 GB** across 11 safetensors shards.
- 78.96 GB base backbone + 53.72 GB auxiliary PLE/MTP shards.
- Canonical cache path: `/content/drive/MyDrive/colab_llm/checkpoints/nvidia/Qwen3.8-Flash-Next-NVFP4/`.

### D08 — NVFP4 mixed precision, no `--quantization` flag
**Decided.** Use the checkpoint's native mixed-precision layout and let SGLang auto-detect it.

- NVIDIA checkpoint resolves to `modelopt_mixed`: linear attention/embeddings in FP8/BF16, routed MoE in NVFP4.
- Do **not** pass `--quantization` for either the NVIDIA or RadixArk export; passing it causes loader failure.
- NVFP4 is chosen because Blackwell has native block-scaled FP4 tensor cores and the checkpoint is built for them.
- The KV cache remains FP8; it is not quantized to 4-bit.

### D09 — SGLang is the only serving runtime
**Decided.** SGLang is the sole production engine for this plan.

- The official SGLang Qwen3.8-Flash-Next cookbook verifies TP=1 on one RTX PRO 6000 with `--ple-offload-embedding`.
- The container of record is `lmsysorg/sglang:dev-qwen38-next-local`, which includes PR #38121.
- Native installs must build from source at PR #38121 if the container is unavailable.

**Rejected runtimes**

| Runtime | Why rejected |
| :--- | :--- |
| **TensorRT-LLM** | `trtllm-build` takes 30–50+ minutes, creates a non-portable 70+ GB engine, and burns 3.3–5.4 CU before serving. |
| **llama.cpp + DFlash 2** | DFlash 2 was trained on Qwen 27B dense hidden states and does not transfer to Flash-Next. CPU-mmapping the 51B PLE table collapses decode to 1–5 tok/s. |
| **vLLM-only path** | Not selected for this deployment; SGLang is the verified single-GPU stack for Flash-Next. |
| **Multi-GPU TP** | Physically and commercially impossible on Colab. |

---

## 5. Serving Configuration

### D10 — PLE host offload
**Decided.** Launch SGLang with `--ple-offload-embedding`.

- The ~47.7 GiB FP8 PLE table lives in pinned host RAM.
- ~78 GiB of remaining checkpoint weights stay in VRAM.
- Resident VRAM after loader temporaries: **74.7 GiB without MTP**, **81.8 GiB with MTP**.
- Free VRAM on 95.60 GiB usable: **19.4 GiB without MTP**, **12.3 GiB with MTP** before static runtime slack.
- Asynchronous token lookups move only ~20.48 KB/token over PCIe 5.0 (<7.2 MB/s at 350 tok/s, <0.012% of x16 bandwidth).

### D11 — MTP speculative decoding
**Decided.** Enable the model's built-in MTP head through SGLang's NEXTN speculative path.

Approved flags:

```text
--speculative-algorithm NEXTN
--speculative-num-steps 3
--speculative-num-draft-tokens 4
--speculative-eagle-topk 1
```

- MTP is expected to take decode from ~83 tok/s to ~148 tok/s on the RTX PRO 6000 Workstation reference.
- Colab's Server Edition is expected to land around ~130–135 tok/s.
- `--mem-fraction-static 0.96` is the MTP launch setting; `0.93` is the non-MTP setting.

### D12 — KV cache and context
**Decided.** Plan the KV cache at FP8 and approve the full native context. Verify the effective KV dtype at launch; the canonical launch command does not add a separate KV dtype flag.

- Only 12 of 48 layers are full attention, with 2 KV heads and `head_dim=256`.
- FP8 KV cost: **12 KiB/token**; FP16 is 24 KiB/token.
- 262,144 tokens consume only **~3.07 GiB** of FP8 KV cache.
- GDN recurrent state is O(1) with sequence length: **58.8 MB/slot in FP16** for Flash-Next's 36 GDN layers.
- Total VRAM at 262K context is ~87 GB of 95.6 GB.
- Message compaction is not required to avoid OOM. It may still be used to reduce prefill latency, but it is not a memory necessity.

### D13 — Docker-first, native only as a gated fallback
**Decided.** The preferred launch is the cookbook-verified Docker path because it can grant unlimited memlock.

Docker launch requirements:

```text
--gpus all
--shm-size 32g
--ulimit memlock=-1
-e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
-p 30000:30000
-v /content/drive/MyDrive/colab_llm:/colab_llm
```

Canonical SGLang launch shape:

```text
python3 -m sglang.launch_server
  --model-path /colab_llm/checkpoints/nvidia/Qwen3.8-Flash-Next-NVFP4
  --tp 1
  --ple-offload-embedding
  --host 0.0.0.0
  --port 30000
  --api-key "${LLAMA_API_KEY}"
  --mem-fraction-static 0.96
  --speculative-algorithm NEXTN
  --speculative-num-steps 3
  --speculative-num-draft-tokens 4
  --speculative-eagle-topk 1
  --trust-remote-code
```

Gating:

- If Docker is present and the daemon is active, use Docker with `--ulimit memlock=-1`.
- If Docker is absent, use the native path only if the probe proves native `mlock` and CUDA `pin_memory` can pin at least 48 GiB.
- If neither path works, Flash-Next on Colab is blocked. Stop and report.

### D14 — Checkpoint caching
**Decided.** Cache the primary checkpoint on Google Drive and avoid re-downloading it on every ephemeral VM.

- Canonical path: `/content/drive/MyDrive/colab_llm/checkpoints/nvidia/Qwen3.8-Flash-Next-NVFP4/`.
- Download with `huggingface_hub.snapshot_download`, resumable, 8 workers.
- If `/content` has at least 150 GB free, copy the checkpoint from the Drive FUSE mount to local disk before launch to avoid a 20–40+ minute FUSE load.
- If an abliterated candidate is selected, cache it the same way under `/content/drive/MyDrive/colab_llm/checkpoints/<candidate_name>/`.
- Drive mount is interactive on every new VM (`authtype=dfs_ephemeral`); the kernel grant times out after 120 seconds.
- The browser profile used for the grant must have `authuser=0` set to `zuluzain5@gmail.com`; complete the grant within 60 seconds.
- Do not blindly re-run `drivemount`. Check `os.path.ismount('/content/drive')` inside the guest first. If it hangs at “Resuming mount...”, open the session URL in the same browser profile.

---

## 6. Fallback Decisions

### D15 — Abliterated Flash-Next is the model fallback, not a hardware fallback
**Decided.** The fallback is triggered only when stock checkpoint refusals block required sensitive systems/security work. It runs on the same G4 High RAM hardware with the same SGLang recipe.

Shortlisted candidates, all **unverified**:

| Candidate | Format | Size | Status |
| :--- | :--- | :---: | :--- |
| `edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` | NVFP4 / SGLang | 106 GB | Unverified; upload date predates its listed base checkpoint |
| `claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4` | NVFP4 / SGLang | 105.48 GB | Unverified |
| `Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF` | GGUF / llama.cpp | 112 GB | Unverified; requires llama.cpp support for the GDN + QSA hybrid |

**Fallback acceptance gates (all three must pass):**

1. **Load gate:** SGLang initializes without missing tensors, shape mismatches, or CUDA illegal-memory-access errors on `sm_120` for NVFP4 candidates. GGUF must load and offload without mmap thrashing.
2. **Reasoning gate:** Full GSM8K (1,319 problems) against `/v1`, or at least 500 problems, stays within **2.0 percentage points** of the stock 97.0–97.7% baseline.
3. **Refusal gate:** Sensitive security/systems prompts produce zero refusals and compliant technical output.

If no candidate passes, the stock checkpoint remains the only approved model, and the limitation is reported rather than assumed away.

The historical Blackfrost Qwen 3.8-27B abliterated A100 fallback is archived and is **not** part of this plan.

---

## 7. API, Networking, and Client Decisions

### D16 — OpenAI-compatible endpoint over trycloudflare
**Decided.** Expose SGLang on port 30000 and tunnel it with `cloudflared`.

- Start: `cloudflared tunnel --url http://127.0.0.1:30000`.
- Parse the URL from `/tmp/cloudflared.log`; do not attach a pipe.
- URL regex: `https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com`.
- Persist to `/tmp/tunnel_url.txt`.
- Wait 5–8 seconds before use; early queries can cache NXDOMAIN for 30 minutes.
- `/health` stays open for monitoring.
- `/v1` is protected by `--api-key`; pass the secret via `colab exec --env LLAMA_API_KEY=...` so it does not appear in `ps aux`.

### D17 — Client integration
**Decided.** Pi and council-go use the same standard OpenAI-compatible contract:

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://<slug>.trycloudflare.com/v1",
    api_key=os.environ["LLAMA_API_KEY"],
)
```

- Switching between stock and abliterated backends requires no client code changes.
- No custom protocol, sidecar proxy, or per-client adapter is planned.

---

## 8. Session and Operations Decisions

### D18 — Attached heartbeat session
**Decided.** Google AI Pro has no background execution; a fire-and-exit process can be reaped in ~10–15 minutes even while serving.

- Run `colab exec` attached with `--timeout 86400`; the CLI default is only 30 seconds and is insufficient.
- `start_server.py` must heartbeat every 20–25 seconds.
- Poll `/health` and `/slots` or `/v1/models`.
- Spawn SGLang and cloudflared with `start_new_session=True`.
- Redirect child stdout/stderr to files (`/tmp/sglang.log`, `/tmp/cloudflared.log`); never leave them on `subprocess.PIPE`.
- If the WebSocket drops, reconnect with `colab exec -s <session> --timeout 86400 -f attach_heartbeat.py` within 10 minutes. Do not reload weights, restart the kernel, or stop the session.

### D19 — Idle watchdog and hard cutoff
**Decided.** Protect the CU pool from unattended burn.

- Idle timeout: **1800 seconds (30 minutes)**.
- Reset the timer on changes to `(id_task, n_decoded, n_prompt_tokens_processed, n_prompt_tokens)`; watching only `n_decoded > 0` is forbidden because it kills normal pauses between prompts.
- On timeout, call `runtime.unassign()` from inside the guest, then `colab stop -s <session>` from the workstation.
- Hard cutoff: **4 hours** per session, so unattended sessions never run overnight.

### D20 — SKU and spend guards
**Decided.** Stop immediately if any guard fails:

- GPU is not `NVIDIA RTX PRO 6000 Server Edition` / Blackwell `sm_120`.
- VRAM is not ~95.6 GiB.
- High RAM shape is not applied (~177 GB / ~48 vCPUs).
- Host RAM is below ~160 GiB or reports ~24 vCPUs.
- Metered burn exceeds the ~8.71–8.90 CU/hr High RAM baseline.
- `/content` or `/` has less than 150 GB free for checkpoint staging.

### D21 — Colab CLI
**Decided.** Use `google-colab-cli==0.7.4` with `--auth=oauth2`.

- 0.6.0 drops the local session after 1 hour while the VM keeps billing.
- 0.7.2 has broken `exec`; 0.7.3 was yanked.
- 0.7.4 refreshes the token 5 minutes before expiry and provides `colab usage`.
- There is no `colab port-forward`; the hybrid model is CLI for `new`/`exec`/`stop`, web for quota, in-guest trycloudflare for `/v1`.

### D22 — Probe before launch
**Decided.** Run [`runbooks/g4-probe.md`](runbooks/g4-probe.md) before staging or launching.

The probe must verify, in order:

1. G4 `--gpu G4 --high-mem` entitlement.
2. Exact GPU name and ~95.6 GiB VRAM.
3. High RAM shape (~177 GB / ~48 vCPUs), not Standard.
4. At least 150 GB free disk.
5. Docker daemon availability.
6. Native `mlock` and `torch.pin_memory` for at least 48 GiB.
7. Metered `colab usage` rate.

Probe discipline:

- Complete in under ~10 minutes, burning no more than ~1.45 CU.
- Record `colab usage` and UTC timestamps before and after.
- Stop/unassign on every path: pass, fail, error, or timeout.

### CU meter management
**Decided.** Use `colab usage` before and after every sitting and cross-check the web Resources panel.

- Do not provision a VM merely to inspect quota; check the balance beforehand.
- At 0 CU, the Colab backend terminates the VM immediately and the trycloudflare URL dies.

### Pre-sitting checklist
**Decided.** Every sitting follows this order:

1. `colab --auth=oauth2 whoami` returns `zuluzain5@gmail.com`.
2. `colab --auth=oauth2 sessions` is empty.
3. `colab --auth=oauth2 usage` shows a positive balance from the 200 CU pool.
4. The dedicated browser profile has `authuser=0` set to `zuluzain5@gmail.com`.
5. The G4 High RAM probe has passed.
6. Provision with `colab --auth=oauth2 new -s qwen-g4 --gpu G4 --high-mem`.
7. Mount Drive, verify the mount, and confirm the hardware/SKU guards.
8. Stage or locate the cached checkpoint, launch SGLang, and run the smoke gates.
9. After work, unassign the VM and confirm `colab --auth=oauth2 sessions` is empty.

---

## 9. Economic Decisions

### D23 — Budget at the standard rate and realistic task mix
**Decided.** Use **8.71 CU/hr** as the budgeting rate, not the obsolete 6.4–6.7 CU/hr promotional rate.

| Metric | Decision |
| :--- | :--- |
| Monthly pool | 200 CU |
| Effective uptime | 200 / 8.71 = **22.96 hours** (~23 hours) |
| Task shape | 25-turn coding loop; 8K initial context; ~1K delta per turn; 500 output tokens/turn |
| Cache assumption | RadixAttention/APC active, so cached prefixes bind in <5 ms and delta prefill is ~0.12 s |
| Boot assumption | Cold boot amortized over a multi-task sitting, not charged per task |

Resulting capacity:

| Scenario | Decode | CU/task | Tasks per 200 CU |
| :--- | :---: | :---: | :---: |
| Headless loop, 10-task sitting | 148 tok/s with MTP | 0.357 | **~560** |
| Continuous warm session | 148 tok/s with MTP | 0.212 | **~943** |
| Verbose thinking, 2.5K tok/turn | 148 tok/s with MTP | 1.173 | **~170** |

**Explicitly rejected:** the prior “under 40 tasks per 200 CU” verdict. It compounded a cold boot on every task with 2,500+ thinking tokens on every turn.

---

## 10. Verification and Acceptance Decisions

### D24 — Required smoke gates
**Decided.** A launch is not accepted until all three smoke tests pass.

1. **Health and model discovery**
   - `GET /health` returns success without auth.
   - `GET /v1/models` returns the model with the Bearer token.
2. **Coding completion**
   - A self-contained AVL-tree implementation request returns code with `temperature: 0.0`.
3. **Tool calling**
   - A tool-calling request with `tool_choice: "required"` returns a valid `tool_calls` payload.

Runtime acceptance also requires:

- The G4 High RAM SKU guards in D20 pass.
- MTP is active and decode is measured; target is ~130–135 tok/s on Server Edition, versus ~148 tok/s on the Workstation reference.
- The real metered CU rate is recorded and consistent with ~8.71–8.90 CU/hr.

---

## 11. Rejected Alternatives and Superseded Assumptions

### Rejected models and platforms

| Item | Decision | Reason |
| :--- | :--- | :--- |
| **`nvidia/GLM-5.3-Flash-NVFP4`** | Rejected | 321.32B MoE, 204.46 GB safetensors, ~195–205 GB loaded VRAM. Needs ≥4× 96 GB GPUs. Cannot fit or cluster on Colab. |
| **Multi-GPU / TP=2+** | Rejected | Colab has no cross-VM interconnect. |
| **Standard G4** | Rejected | ~90 GB host RAM is not approved for the 47.7 GiB PLE offload; High RAM is required. |
| **A100 / L4 / H100 substitution** | Rejected | Not the approved target for this plan. The historical A100 + 27B fallback is archived. |
| **Dense 27B as the current fallback** | Superseded | The plan moved to Flash-Next primary and abliterated Flash-Next fallback. |
| **GGUF as the primary runtime** | Rejected | llama.cpp cannot serve this architecture/PLE combination at usable speed on Colab. |

### Superseded assumptions

| Old assumption | Corrected decision |
| :--- | :--- |
| Colab G4 host RAM is 53–64 GB and will OOM | G4 High RAM is ~176.9 GB, leaving >120 GB free. |
| Flash-Next cannot run on one G4 | Official SGLang cookbook verifies TP=1 with PLE offload. |
| NVIDIA mandates TP=8 | TP=8 was an HGX B200 sample command, not a G4 requirement. |
| Flash-Next context is capped at 32K–48K | 262K native context costs only ~3.07 GiB of FP8 KV. |
| G4 bills 6.4–6.7 CU/hr | Budget 8.71 CU/hr; 6.4–6.74 was promotional/off-peak. |
| 200 CU yields ~30.5 hours | 200 CU yields ~22.96 hours at 8.71 CU/hr. |
| A 200 CU pool yields under 40 tasks | Realistic task capacity is ~560–943 tasks. |
| FP8 decode is faster than NVFP4 on Blackwell | Rejected for the 27B comparison: it violates bandwidth math. NVFP4 moves fewer weight bytes; this plan uses the checkpoint's native mixed precision. |
| CPU offload is a viable way to fit larger models | PCIe offload collapses decode to 0.3–3.5 tok/s. |

---

## 12. Open Items and Their Gates

These are not decisions to be assumed; they are live unknowns with explicit failure behavior.

| ID | Open item | Status | Pass condition | Failure behavior |
| :--- | :--- | :---: | :--- | :--- |
| **U01** | AI Pro entitlement to G4 | Unverified | `colab new -s g4-probe --gpu G4 --high-mem` succeeds. | Blocked; stop and report. |
| **U02** | High RAM shape actually assigned | Unverified | ~177 GB host RAM and ~48 vCPUs. | Treat as Standard G4; stop. |
| **U03** | Docker availability | Unverified | Docker daemon active with GPU access. | Use native path only if native pinning passes. |
| **U04** | Native `mlock` / CUDA pinning ≥48 GiB | Unverified | `mlock` and `torch.pin_memory` both succeed. | If Docker is also unavailable, blocked; stop and report. |
| **U05** | Real cold-boot time and CU burn | Unverified | Measured and recorded; budget remains viable. | Recalculate sitting economics and report. |
| **U06** | Abliterated checkpoint load, GSM8K, and refusals | Unverified | All three fallback gates pass. | Do not deploy that candidate; stock remains. |
| **U07** | Real G4 Server Edition decode speed | Unverified | Measured against ~130–135 tok/s expectation. | Record actual; no speed claim without measurement. |

---

## 13. Plan of Record in One Paragraph

Run Qwen 3.8 Flash-Next NVFP4 (`nvidia`, or `RadixArk` as alternate) with SGLang TP=1 and `--ple-offload-embedding` on a **Colab G4 High RAM** VM (~177 GB host RAM, ~48 vCPUs, 96 GB VRAM), cache the 132.68 GB checkpoint on Google Drive, pin the ~47.7 GiB PLE table in host RAM, keep FP8 KV cache for the full 262K context, enable NEXTN MTP, expose `/v1` through a trycloudflare tunnel with a Bearer key, keep the session alive with an attached 20–25 s heartbeat, and enforce a 30-minute idle watchdog plus a 4-hour hard cutoff. If stock refusals block required work, qualify an abliterated Flash-Next fallback through load, GSM8K, and refusal gates. If G4 High RAM, Docker/memlock, or the fallback gates fail, stop and report rather than substituting hardware or assumptions.

---

## 14. Source Map

| Decision area | Canonical source |
| :--- | :--- |
| Architecture, model choice, economics | [`decision.md`](decision.md), [`claims-ledger.md`](claims-ledger.md) |
| G4 hardware and High RAM probe | [`runbooks/g4-probe.md`](runbooks/g4-probe.md) |
| SGLang launch and smoke tests | [`runbooks/flash-next-sglang.md`](runbooks/flash-next-sglang.md) |
| Abliterated fallback gates | [`runbooks/fallback-abliterated.md`](runbooks/fallback-abliterated.md) |
| Account, CLI, Drive, tunnel, heartbeat, shutdown | [`ops.md`](ops.md) |
| Repository-wide summary | [`README.md`](README.md) |
