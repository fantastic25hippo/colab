# Google Colab Worker Plan: Kernel Lifecycle, CLI Surface, Drive Mount, Tunneling, Shutdown & Guest CUDA (`kernel-and-cli.md`)

**Date:** 2026-10-14  
**Target Account & Environment:** Google AI Pro (200 Colab Compute Units/month), CLI identity `zuluzain5@gmail.com`  
**Installed CLI Baseline:** `google-colab-cli` `0.6.0` at `~/.local/bin/colab` vs. Current Latest `0.7.4` (released 2026-09-25)

---

## Primary Sources List

1. **Google Colab CLI Repository & README (`googlecolab/google-colab-cli`, `0.6.0`–`0.7.4`):** [github.com/googlecolab/google-colab-cli](https://github.com/googlecolab/google-colab-cli)
2. **Google Colab CLI PyPI Package Index (`0.1.0` through `0.7.4`, released 2026-09-25):** [pypi.org/project/google-colab-cli](https://pypi.org/project/google-colab-cli/)
3. **Google Colab CLI Changelog (`CHANGELOG.md`, `0.1.0`–`0.7.0`+):** [github.com/googlecolab/google-colab-cli/blob/main/CHANGELOG.md](https://github.com/googlecolab/google-colab-cli/blob/main/CHANGELOG.md)
4. **Google Colab CLI Agent & Contributor Architecture (`AGENTS.md`):** [github.com/googlecolab/google-colab-cli/blob/main/AGENTS.md](https://github.com/googlecolab/google-colab-cli/blob/main/AGENTS.md)
5. **Google Colab CLI Operator Skill Specification (`skills/colab-operator/SKILL.md`):** [github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md)
6. **Colab CLI Doc 01 — Session Management & Backend Architecture (`docs/01_session_management.md`, updated 2026-09-25 via PR #144 & PR #149):** [github.com/googlecolab/google-colab-cli/blob/main/docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)
7. **Colab CLI Doc 02 — Remote Code Execution & Interactive Environments (`docs/02_execution_and_interactive.md`):** [github.com/googlecolab/google-colab-cli/blob/main/docs/02\_execution\_and\_interactive.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/02_execution_and_interactive.md)
8. **Colab CLI Doc 03 — Remote File System Management (`docs/03_file_management.md`):** [github.com/googlecolab/google-colab-cli/blob/main/docs/03\_file\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/03_file_management.md)
9. **Colab CLI Doc 04 — Workspace Automation & Utilities (`docs/04_automation_and_utility.md`):** [github.com/googlecolab/google-colab-cli/blob/main/docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)
10. **Colab CLI Doc 05 — Ephemeral Execution Engine (`docs/05_run_command.md`):** [github.com/googlecolab/google-colab-cli/blob/main/docs/05\_run\_command.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/05_run_command.md)
11. **Colab CLI Recorded Terminal Demos (`docs/demos.md`):** [github.com/googlecolab/google-colab-cli/blob/main/docs/demos.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/demos.md)
12. **Colab CLI Issue #14 — Session Unresponsive / Idle Pruning / Keep-Alive RPC (`2026-05-05` to `2026-06-16`):** [github.com/googlecolab/google-colab-cli/issues/14](https://github.com/googlecolab/google-colab-cli/issues/14)
13. **Colab CLI Issue #47 — Runtime Shape Selection (`--high-mem`), G4 & A100 SKU Assignments (`2026-06-05` to `2026-08-09`):** [github.com/googlecolab/google-colab-cli/issues/47](https://github.com/googlecolab/google-colab-cli/issues/47)
14. **Colab CLI Issue #48 — Colab Secrets (`userdata.get`) Failure in Headless CLI Mode (`2026-06-05`):** [github.com/googlecolab/google-colab-cli/issues/48](https://github.com/googlecolab/google-colab-cli/issues/48)
15. **Colab CLI Issue #82 — Long `exec` Timeout Spins Local CPU Core Indefinitely (`2026-07-19`):** [github.com/googlecolab/google-colab-cli/issues/82](https://github.com/googlecolab/google-colab-cli/issues/82)
16. **Colab CLI Issue #94 — `0.6.0` Crash with `jupyter-kernel-client 1.0.0` (`2026-08-03`):** [github.com/googlecolab/google-colab-cli/issues/94](https://github.com/googlecolab/google-colab-cli/issues/94)
17. **Colab CLI Issue #102 — `0.6.0` Missing `ssh` Command on PyPI (`2026-08-08`):** [github.com/googlecolab/google-colab-cli/issues/102](https://github.com/googlecolab/google-colab-cli/issues/102)
18. **Colab CLI Issue #103 — `drivemount` Fails with `"Error propagating: 400"` on Multi-Account Browser (`authuser!=0`, `2026-08-09`):** [github.com/googlecolab/google-colab-cli/issues/103](https://github.com/googlecolab/google-colab-cli/issues/103)
19. **Colab CLI Issue #106 — Live Sessions Pruned Locally After 60 Minutes Due to Unrefreshed `RuntimeProxyInfo.token` (Fixed in `0.7.4` via PR #149, `2026-08-09` to `2026-09-25`):** [github.com/googlecolab/google-colab-cli/issues/106](https://github.com/googlecolab/google-colab-cli/issues/106)
20. **Colab CLI Issue #113 — `colab drivemount` 120s Timeout on Headless CLI-Only Sessions (`2026-08-19`):** [github.com/googlecolab/google-colab-cli/issues/113](https://github.com/googlecolab/google-colab-cli/issues/113)
21. **Colab CLI Issue #116 — Partial OAuth Consent Forces Re-Login Every 1 Hour (`2026-08-27`):** [github.com/googlecolab/google-colab-cli/issues/116](https://github.com/googlecolab/google-colab-cli/issues/116)
22. **Colab CLI Issue #137 — `0.7.2` Regression: `jupyter-kernel-client==0.8` Pin Breaks `colab exec` (`2026-09-25`):** [github.com/googlecolab/google-colab-cli/issues/137](https://github.com/googlecolab/google-colab-cli/issues/137)
23. **Colab CLI Issue #160 — Session Lifetime When Disconnected / Running in Background (`2026-09-30`):** [github.com/googlecolab/google-colab-cli/issues/160](https://github.com/googlecolab/google-colab-cli/issues/160)
24. **Google Developers Blog (`2026-09-22`) — *"Colab is now part of your Google AI plan"*:** [developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/)
25. **Google One Help — *Use Google AI Pro benefits* (`16476811`) & *Use Google AI Ultra benefits* (`16286513`):** [support.google.com/googleone/answer/16476811](https://support.google.com/googleone/answer/16476811?hl=en&co=GENIE.Platform=Android) | [support.google.com/googleone/answer/16286513](https://support.google.com/googleone/answer/16286513?hl=en)
26. **Google Colaboratory Official FAQ:** [research.google.com/colaboratory/faq.html](https://research.google.com/colaboratory/faq.html) | [research.google.com/colaboratory/intl/en-GB/faq.html](https://research.google.com/colaboratory/intl/en-GB/faq.html)
27. **Google Colaboratory Release Notes (`2026-03-25` G4 RTX PRO 6000 Blackwell Launch & `2022-09-12` `runtime.unassign()`):** [colab.research.google.com/notebooks/relnotes.ipynb](https://colab.research.google.com/notebooks/relnotes.ipynb)
28. **Google Colab Tools (`googlecolab/colabtools`) — `google/colab/drive.py` Source:** [github.com/googlecolab/colabtools/blob/main/google/colab/drive.py](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)
29. **Google Colab Tools Issue #2568 (`runtime.unassign()` specification, `2022-09-09`):** [github.com/googlecolab/colabtools/issues/2568](https://github.com/googlecolab/colabtools/issues/2568)
30. **Google Colab Tools Issue #5950 (`@colaboratory-team` statement on idle runtime deletion policy & unacknowledged execution requirement, `2026-04-15`):** [github.com/googlecolab/colabtools/issues/5950](https://github.com/googlecolab/colabtools/issues/5950)
31. **Google Colab Backend Info (`googlecolab/backend-info`, `2026.07` runtime image):** [github.com/googlecolab/backend-info](https://github.com/googlecolab/backend-info)
32. **llama.cpp HTTP Server Documentation (`tools/server/README.md`):** [github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
33. **Cloudflare Quick Tunnels Field Guide (`cloudflared 2026.9.1`, `2026-09-22`):** [flaviocopes.com/cloudflare-quick-tunnels](https://flaviocopes.com/cloudflare-quick-tunnels/)
34. **Colab CLI `v0.7.4` Source Code Audit (`issoh.co.jp`, `2026-10-09`):** [issoh.co.jp/tech/details/12438](https://www.issoh.co.jp/tech/details/12438/)
35. **Google AI Pro 200 CU Verification & Field Measurements (`2026-09-23` to `2026-10-11`):** [note.com/toolset](https://note.com/toolset/n/n67bae780888a?hl=en) | [note.com/gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en) | [note.com/unco3](https://note.com/unco3/n/n47710f23e25a) | [explainx.ai](https://explainx.ai/blog/google-ai-pro-colab-a100-h100-200-compute-units-claimed-vs-verified-2026) | [r/GeminiAI G4 on AI Pro](https://www.reddit.com/r/GeminiAI/comments/1x2uoz1/i_just_found_this_out/) | [r/GoogleColab G4 nvidia-smi 580.82.07 / CUDA 13.0](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/)

---

## Executive Findings

1. **The "fire-and-exit" `colab exec` boot pattern will get your VM killed in ~10–15 minutes (`Documented` & `Measured`):** While daemonizing `llama-server` and `cloudflared` via `Popen(..., start_new_session=True)` survives the immediate exit of `colab exec` ([SKILL.md](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md)), Google AI Pro does **not** include Colab background execution ([Google Developers Blog, 2026-09-22](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/)). Worse, as of `google-colab-cli` `0.7.4` (PR #144, 2026-09-25), the CLI removed its local keep-alive daemon ([docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)), and the Colab backend deletes runtimes whenever there is no active/unacknowledged kernel cell execution or live client connection ([colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950), [google-colab-cli#160](https://github.com/googlecolab/google-colab-cli/issues/160), [google-colab-cli#14](https://github.com/googlecolab/google-colab-cli/issues/14)). Traffic through `trycloudflare.com` bypasses Colab's Tunnel Frontend (TFE) and does not register as activity. **You must keep `colab exec` attached in a foreground watchdog/heartbeat loop that prints a line every 20–25 seconds.**
2. **Upgrade `google-colab-cli` from `0.6.0` directly to `0.7.4` (skip `0.7.2` and `0.7.3`) (`Documented`):** In `0.6.0`, the 1-hour `RuntimeProxyInfo.token` (`tokenExpiresInSeconds = 3600`) is never refreshed; after 60 minutes, any CLI command receives HTTP `401`/`404` and deletes the session from local `~/.config/colab-cli/sessions.json` while the VM continues running and burning CU ([google-colab-cli#106](https://github.com/googlecolab/google-colab-cli/issues/106)). Version `0.7.4` (2026-09-25) fixes this via automatic token refresh 5 minutes before expiry ([docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)) and adds `colab usage` (`GET /tun/m/ccu-info`). Do **not** install `0.7.2` (broken `colab exec` due to `jupyter-kernel-client==0.8` pin, [google-colab-cli#137](https://github.com/googlecolab/google-colab-cli/issues/137)) or `0.7.3` (yanked on PyPI, [PyPI](https://pypi.org/project/google-colab-cli/)).
3. **Avoiding High-RAM A100 and understanding `--gpu` flags (`Documented` & `Measured`):** Valid `--gpu` values are `T4`, `L4`, `G4`, `H100`, and `A100` ([SKILL.md](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md)). There is no `"A100 High-RAM"` GPU enum—unrecognized strings silently fall back to `A100` ([issoh.co.jp](https://www.issoh.co.jp/tech/details/12438/)). High-RAM (~167 GB system RAM, `A100-SXM4-80GB`, ~11–15 CU/hr) is requested via `--high-mem` (`shape=hm` in `0.7.0+`, [google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47)). Omit `--high-mem` to request Standard shape (`NVIDIA A100-SXM4-40GB`, ~83.5 GB system RAM, ~5.4–6.8 CU/hr), and enforce a startup check in `start_server.py` via `colab usage` or `nvidia-smi` + `psutil`. `G4` (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, 96 GB GDDR7, ~6.71–6.77 CU/hr) is confirmed working for some Google AI Pro 200 CU users ([r/GeminiAI](https://www.reddit.com/r/GeminiAI/comments/1x2uoz1/i_just_found_this_out/)) and grayed out for others by region ([note.com/gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en)).
4. **Drive mount requires `zuluzain5@gmail.com` to be `authuser=0` in the browser and never persists across VMs (`Documented` & `Measured`):** `colab drivemount` uses `authtype=dfs_ephemeral` with `record=false` ([docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)), requiring an interactive browser grant within 120 seconds on every new VM ([colabtools/drive.py](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)). If the browser's default account (`authuser=0`) is not `zuluzain5@gmail.com`, credential propagation fails with `[colab] Error propagating: 400` ([google-colab-cli#103](https://github.com/googlecolab/google-colab-cli/issues/103)).
5. **CUDA 13.0 / Driver 580 and multi-arch binary caching (`Measured` & `Documented`):** Current 2026 Colab GPU runtimes (`2026.07` image, Ubuntu 22.04.5 LTS, Python 3.12.13, [backend-info](https://github.com/googlecolab/backend-info)) run NVIDIA Driver `580.82.07` with CUDA Toolkit `13.0` (`13.0.88`, [r/GoogleColab](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/), [note.com/unco3](https://note.com/unco3/n/n47710f23e25a)). Compile `llama-server` either as two separate GPU-Cached binaries or as a single fatbin with `-DCMAKE_CUDA_ARCHITECTURES="80;120"` so both A100 (`sm_80`) and G4 Blackwell (`sm_120`) execute native SASS without JIT compilation delay.

---

## 1. Session Lifetime After `colab exec` Returns

### 1.1 Does a process started with `start_new_session=True` / `nohup` keep running after the `colab exec` Python/cell process exits on Google AI Pro?
* **Process-level survival: Yes (`Documented`).**  
  In `google-colab-cli`, `colab exec` connects to the remote VM's existing Jupyter kernel over a custom WebSocket subprotocol (`JupyterSubprotocol` via `ColabRuntime`), submits the code as a single cell execution (`execute_request`), streams `iopub` outputs until `status: idle` and `execute_reply` arrive, and then closes the local WebSocket client ([docs/02\_execution\_and\_interactive.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/02_execution_and_interactive.md), [SKILL.md](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md)). Closing `colab exec` does **not** terminate the remote Jupyter kernel process or send `SIGHUP`/`SIGTERM` to detached child process groups created with `subprocess.Popen(..., start_new_session=True)` (`setsid()`). As documented in [SKILL.md](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md): *"State persists across `colab exec` calls in the same session: variables, imports, files on `/content` remain available until `colab restart-kernel` or `colab stop`."*
* **VM-level survival after `colab exec` exits: No — the VM will be reaped as idle within ~10–15 minutes (`Documented` & `Measured`).**  
  Once `colab exec` returns, two critical things become true from Google's infrastructure perspective:
  1. There is **no active/unacknowledged cell execution** running on the Jupyter kernel (`kernel_state == idle`).
  2. There is **no open WebSocket client connection** through Colab's Tunnel Frontend (`colab.research.google.com/tun/m/<endpoint>`), because incoming LLM requests from your laptop travel through Cloudflare's edge directly to the in-guest `cloudflared` daemon (`127.0.0.1:PORT`), completely bypassing Colab's proxy and kernel telemetry.
  - **Documented (`google-colab-cli` `0.7.4`, PR #144, 2026-09-25):** In `0.6.0`, `colab new` spawned a detached local keep-alive daemon that polled `GET /tun/m/<endpoint>/keep-alive/` with header `X-Colab-Tunnel: Google` ([CHANGELOG.md](https://github.com/googlecolab/google-colab-cli/blob/main/CHANGELOG.md), [google-colab-cli#14](https://github.com/googlecolab/google-colab-cli/issues/14)). In `0.7.4` (PR #144, merged 2026-09-25), the maintainers deleted the local keep-alive daemon and documented in [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md):
    > *"VM liveness is automatically maintained by the Colab backend based on active kernel activity. The CLI runs no keep-alive daemon. A session persists until explicitly terminated via `colab stop` or until the backend reclaims it due to inactivity."*
  - **Documented (`colabtools#5950`, 2026-04-15 & `google-colab-cli#160`, 2026-09-30):** In [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950), `@colaboratory-team` confirmed that runtimes are *"deleted due to our idle runtime deletion policy while there's no active/unacknowledged executions on those runtimes at the time of deletions. This indicates that your code cell execution was completed and all the execution replies were fetched by your browser."* Similarly, in [google-colab-cli#160](https://github.com/googlecolab/google-colab-cli/issues/160) (2026-09-30), Colab CLI maintainer `@EvanWiederspan` confirmed that for background `tmux` jobs not using the Jupyter kernel:
    > *"Colab mostly depends on checking for live connections to the VM to determine whether to shut down the VM. As long as the SSH connection remains open, I believe that should count as an active connection and prevent the VM from shutting down due to being idle... If you shut down the SSH connection, the VM will likely be shut down soon after since we won't have a signal that the VM is still being used."*
  - **Measured (`google-colab-cli#14`, 2026-05-05):** When executing discrete `colab exec` commands without an active connection or kernel execution between calls, user `@bkowshik` measured that an idle gap of **10 to 15 minutes** caused the Colab backend to reclaim the VM, returning HTTP `404`/`401` on the next `colab exec` ([google-colab-cli#14](https://github.com/googlecolab/google-colab-cli/issues/14)).

### 1.2 Does Google AI Pro include Colab background execution? Comparison across tiers
* **Google AI Pro background execution status: No (`Documented` & `Measured`).**  
  Google's official announcement on **2026-09-22** (*"Colab is now part of your Google AI plan"*, [Google Developers Blog](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/)) and the [Google AI Ultra Help Center (`16286513`)](https://support.google.com/googleone/answer/16286513?hl=en) explicitly restrict uninterrupted background execution to **Google AI Ultra** (and standalone **Colab Pro+**), excluding Google AI Pro:
  > *"Starting today, Google AI subscribers get premium Google Colab benefits, including priority access to faster accelerators and more powerful machines. Google AI Ultra subscribers also unlock uninterrupted background execution and Premium GPU access, so long training runs can finish without keeping a browser tab open."* — [Google Developers Blog, 2026-09-22](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/)
  
  Independent tests on Google AI Pro (200 CU/month) accounts on **2026-09-23** ([note.com/gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en)) and **2026-10-03** ([note.com/toolset](https://note.com/toolset/n/n67bae780888a?hl=en), [explainx.ai](https://explainx.ai/blog/google-ai-pro-colab-a100-h100-200-compute-units-claimed-vs-verified-2026)) confirmed that Google AI Pro grants 200 monthly CUs with Colab Pro-tier session semantics—meaning **no background execution after client disconnect**.

| Attribute | Free Tier (No CU) | Colab Pro (Standalone) | Google AI Pro (Locked Account) | Colab Pro+ (Standalone) | Google AI Ultra |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Monthly Compute Units** | `0 CU` (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/faq.html)) | `100 CU/mo` (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/faq.html)) | **`200 CU/mo`** (`Documented`, [Google Blog 2026-09-22](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/)) | `500–600 CU/mo` (`Documented`, [Google Blog 2026-09-22](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/)) | `1,000–2,000 CU/mo` (`Measured` 1,000 CU in JP / `Documented` up to 4x Pro+, [Google Blog 2026-09-22](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/), [gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en)) |
| **Background Execution (Run after disconnect)** | **No** (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/faq.html)) | **No** (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/faq.html)) | **No** (`Documented`, [Google Blog 2026-09-22](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/); `Measured`, [toolset](https://note.com/toolset/n/n67bae780888a?hl=en)) | **Yes** — only while a code cell is actively executing (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/faq.html), [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)) | **Yes** — only while a code cell is actively executing (`Documented`, [Google Blog 2026-09-22](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/), [Google One Help](https://support.google.com/googleone/answer/16286513?hl=en)) |
| **Max Connected Lifetime** | **Up to 12 hours** (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/faq.html)) | **Up to 24 hours** (`Documented` "longer runtimes", [Colab FAQ](https://research.google.com/colaboratory/faq.html); `Measured` 24h cap) | **Up to 24 hours** (AI Pro docs silent on exact hours; inherits paid Pro 24h cap, `Measured`, [toolset](https://note.com/toolset/n/n67bae780888a?hl=en)) | **Up to 24 hours** (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/faq.html), [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)) | **Up to 24 hours** (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/faq.html)) |
| **Idle-Kernel Reaping (Client Connected)** | **~90 minutes** of no cell execution (`Measured`, [Colab Guide](https://www.qwe.edu.pl/tutorial/google-colab-ai-data-analysis/); FAQ notes idle timeouts apply) | **~90 minutes** of no cell execution (`Measured`; FAQ notes *"Idle timeouts only apply if code execution terminates"*) | **~90 minutes** with client open and no cell executing (`Measured`; AI Pro docs silent) | **~90 minutes** after cell execution terminates (`Documented` policy in [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)) | **~90 minutes** after cell execution terminates (`Documented` policy in [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)) |
| **Survives Client Disconnecting?** | **No** — reaped after **~10–15 min** (`Measured`, [google-colab-cli#14](https://github.com/googlecolab/google-colab-cli/issues/14)) | **No** — reaped after **~10–15 min** without open WebSocket/SSH (`Documented` & `Measured`, [#14](https://github.com/googlecolab/google-colab-cli/issues/14), [#160](https://github.com/googlecolab/google-colab-cli/issues/160)) | **No** — reaped after **~10–15 min** without open WebSocket/SSH (`Documented` & `Measured`, [#14](https://github.com/googlecolab/google-colab-cli/issues/14), [#160](https://github.com/googlecolab/google-colab-cli/issues/160)) | **Yes** (up to 24h if a cell is running; reaped in ~15–60m if cell exited, [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)) | **Yes** (up to 24h if a cell is running) |

### 1.3 Supported pattern to keep `llama-server` alive
* **Supported Pattern (`Documented` in [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md), [docs/02\_execution\_and\_interactive.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/02_execution_and_interactive.md), and [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)):**  
  Instead of having `start_server.py` spawn `llama-server` + `cloudflared` and exit, **keep `colab exec` attached in the foreground** running an in-cell Python supervisor loop for the entire lifetime of your coding session:
  1. Launch `colab exec -s <name> -f start_server.py --timeout 86400` on the laptop.
  2. Inside `start_server.py`, start `llama-server` and `cloudflared` as child processes, capture the `https://<slug>.trycloudflare.com` URL from `cloudflared`'s `stderr`, and print it to `stdout` with `flush=True` so your local wrapper script can immediately read it.
  3. Enter a `while True:` watchdog/heartbeat loop inside `start_server.py` that sleeps `20` seconds per tick, queries `http://127.0.0.1:8080/health` and `http://127.0.0.1:8080/slots` (to track whether any slot is actively generating tokens), and **prints a short status line with `flush=True` every 20 seconds**.
  - **Why this works:**
    - It keeps an **active, unacknowledged cell execution** running on the Jupyter kernel (`kernel_state == busy`), satisfying the backend liveness rule stated by `@colaboratory-team` in [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950) and [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md).
    - It keeps a **live WebSocket connection** open through Colab's TFE proxy, satisfying `@EvanWiederspan`'s live-connection check in [google-colab-cli#160](https://github.com/googlecolab/google-colab-cli/issues/160).
    - Emitting `iopub.stream` (`stdout`) every 20 seconds prevents `jupyter-kernel-client`'s 30-second silence ceiling from firing even if `--timeout` is left at default (`docs/04_automation_and_utility.md`).

### 1.4 What is the default `colab exec` timeout, and does expiry kill guest processes?
* **Default timeout: `30.0` seconds (`Documented`).**  
  Per [docs/02\_execution\_and\_interactive.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/02_execution_and_interactive.md), [docs/05\_run\_command.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/05_run_command.md), and [CHANGELOG.md](https://github.com/googlecolab/google-colab-cli/blob/main/CHANGELOG.md) (PR #43, commit `889d09f`, 2026-06-04, *"Increase default execution timeout to 30 seconds"*), the default `--timeout` for `colab exec` and `colab run` is **`30.0` seconds** (while `colab drivemount`, `colab gcloud-auth`, and `colab install` override it to `INTERACTIVE_AUTOMATION_TIMEOUT_SEC = 600` seconds, [docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)).
* **Does timeout expiry kill guest processes? No (`Documented` & `Measured`).**  
  `--timeout` is enforced strictly on the client side inside `jupyter_kernel_client.KernelClient.execute_interactive()`. When it expires, the CLI does **not** send an `interrupt_request` or kill signal to the remote kernel or its child processes. As documented in [google-colab-cli#82](https://github.com/googlecolab/google-colab-cli/issues/82) (2026-07-19):
  > *"The remote Colab execution remains independent and can continue running; the busy loop is local to the client."*
* **Critical `0.6.0` client bug on timeout expiry (`Measured`, [google-colab-cli#82](https://github.com/googlecolab/google-colab-cli/issues/82)):**  
  In `google-colab-cli` `0.6.0` paired with `jupyter-kernel-client` `0.9.0`, when a cell runs longer than `--timeout`, `execute_interactive()` clamps the remaining wait time to `0` (`timeout = max(0, deadline - time.monotonic())`) and calls `threading.Event.wait(0)` inside a `while` loop without checking if `deadline` passed (`datalayer/jupyter-kernel-client#41`). Instead of exiting cleanly with a timeout error, the local `colab exec` process spins **one local laptop CPU core at 100% indefinitely** ([google-colab-cli#82](https://github.com/googlecolab/google-colab-cli/issues/82)). Upgrading to `0.7.4` (which uses the patched `googlecolab/jupyter-kernel-client` fork via PR #125/#138) and passing `--timeout 86400` eliminates this bug.

### 1.5 What does CLI `0.6.0` "local session drop after ~1 hour while the VM still bills" mean in current docs/issues? Did `0.7.x` fix it?
* **Root Cause #1 — Unrefreshed 1-Hour `RuntimeProxyInfo.token` (`Documented` & `Measured` in [google-colab-cli#106](https://github.com/googlecolab/google-colab-cli/issues/106) and [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md) Section 6):**  
  When `colab new` assigns a VM, the backend returns a `RuntimeProxyInfo` object containing a proxy `token` and `url` with a strict **1-hour TTL** (`tokenExpiresInSeconds: "3600"`, [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)).
  - In `0.6.0`, `src/colab_cli/client.py` parsed `token_expires_in_seconds` into `RuntimeProxyInfo`, but **never persisted or refreshed it** ([google-colab-cli#106](https://github.com/googlecolab/google-colab-cli/issues/106), filed 2026-08-09).
  - After 60 minutes, the cached proxy token in `~/.config/colab-cli/sessions.json` expired. The very next `colab exec`, `colab status`, or `colab stop` received HTTP `401 Unauthorized` or `404 Not Found` from the runtime proxy.
  - In `src/colab_cli/common.py` (line 71), `session_command` caught the error, called `is_terminal_error(e)` in `src/colab_cli/utils.py` (which treated any `401`, `404`, or `502` as a permanently dead VM), and unconditionally executed `state.prune_session(name)`—printing:
    ```text
    [colab] Session 'worker' is no longer active on the server. Pruning local state.
    ```
  - Because `prune_session()` deleted the local entry in `sessions.json` while the VM was still alive on Google's backend, the VM became an **untracked orphan** (`[?]` in `colab sessions`) that **continued burning Compute Units at the full hourly rate** until manually killed via the web UI.
* **Did `0.7.x` fix it? Yes, in `0.7.4` (released 2026-09-25, PR #149) (`Documented`):**  
  Maintainer `@sethtroisi` closed [google-colab-cli#106](https://github.com/googlecolab/google-colab-cli/issues/106) on **2026-09-25**: *"Fixed by #149 which is in 0.7.4"*. As documented in [docs/01\_session\_management.md Section 6 (Token Refresh)](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md):
  > *"The `RuntimeProxyInfo.token` issued at session creation has a 1-hour TTL (`token_expires_in_seconds = 3600`). `SessionState` stores `token_expires_at` alongside the token. On every `State.get_session(name)` call, if the token is within 5 minutes of expiring (or already expired), the CLI silently calls `list_assignments()` (`GET /tun/m/assignments`) and updates `token`, `url`, and `token_expires_at` in `sessions.json` before returning the session. This makes `exec`, `push`, `pull`, `ls`, `cat`, `rm`, and `status` work seamlessly across the 1-hour boundary. Long-lived connections (`repl`, `console`, `ssh`) only present the token on initial connect and aren't refreshed mid-session."*
* **Root Cause #2 — Granular OAuth2 Scope Mismatch After 1 Hour (`Documented` in [google-colab-cli#116](https://github.com/googlecolab/google-colab-cli/issues/116), 2026-08-27):**  
  If a user authenticating via `--auth=oauth2` unchecks optional scopes (such as Google Drive or Cloud Platform) on the Google OAuth consent screen, Google issues an access token with a 1-hour expiry and a narrower scope list. When `auth.py` reloads `~/.config/colab-cli/token.json` after 1 hour, it passes hardcoded `PUBLIC_SCOPES` (all 6 scopes) to `Credentials.from_authorized_user_file()`. The background refresh request fails with `RefreshError: ('invalid_scope: ...')`, wiping the token and forcing an interactive browser login after 1 hour. **Fix:** Always grant **all requested boxes** on the OAuth consent screen when logging in as `zuluzain5@gmail.com`.

---

## 2. CLI Surface (Current Latest `0.7.4` vs. `0.6.0`)

### 2.1 Latest `google-colab-cli` version on PyPI / GitHub releases
* **Current Latest Release: `0.7.4`, published `2026-09-25` (`Documented`, [PyPI](https://pypi.org/project/google-colab-cli/)).**

| Version | Release Date | Distribution Status | Key Changes & Operational Impact |
| :--- | :--- | :--- | :--- |
| **`0.6.0`** | `2026-06-16` | PyPI & GitHub (`Installed Baseline`) | Added `--remote` copy-paste OAuth2 flow (#54) and local TFE keep-alive ping (#14, #61). **Bugs:** Prunes live sessions after 1 hour (#106); crashes if `jupyter-kernel-client >= 1.0.0` is installed (#94); spins local CPU core on `--timeout` expiry (#82); lacks `usage`, `ssh`, and `--high-mem`. ([CHANGELOG.md](https://github.com/googlecolab/google-colab-cli/blob/main/CHANGELOG.md)) |
| **`0.7.0`** | `2026-09-03` | GitHub tag only (Rejected by PyPI) | Added `colab ssh` (#88), `colab usage`, `--env KEY=VALUE` on `exec`/`run` (#65), `--high-mem` on `new`/`run`/`ssh` (#105), friendly HTTP 412 error message (#112). Failed PyPI upload due to Hatchling `1.32.0` Core Metadata `2.5` ([CHANGELOG.md](https://github.com/googlecolab/google-colab-cli/blob/main/CHANGELOG.md), [#102](https://github.com/googlecolab/google-colab-cli/issues/102)). |
| **`0.7.1`** | `2026-09-14` | GitHub tag only | Pinned `core-metadata-version = "2.4"` in `pyproject.toml` (#129). |
| **`0.7.2`** | `2026-09-22` | PyPI (**Broken `exec`**) | **Do not use.** `pyproject.toml` declared `jupyter-kernel-client==0.8`, causing `pip`/`uv` to install upstream PyPI `0.8.0` instead of the `googlecolab` fork; `colab exec` crashes with `AttributeError: module 'jupyter_kernel_client' has no attribute 'JupyterSubprotocol'` ([google-colab-cli#137](https://github.com/googlecolab/google-colab-cli/issues/137)). |
| **`0.7.3`** | `2026-09-25` | PyPI (**Yanked**) | Yanked on `2026-09-25` with reason: `"bad version pin"` ([PyPI](https://pypi.org/project/google-colab-cli/)). |
| **`0.7.4`** | `2026-09-25` | PyPI (**Current Latest**) | **Recommended upgrade target.** Restores `jupyter-kernel-client` compatibility (#125, #138), adds automatic 1-hour runtime proxy token refresh (#149 / [#106](https://github.com/googlecolab/google-colab-cli/issues/106)), and removes the local keep-alive daemon (#144). ([PyPI](https://pypi.org/project/google-colab-cli/), [issoh.co.jp](https://www.issoh.co.jp/tech/details/12438/)) |

### 2.2 Exact commands and output of `whoami`, `sessions`, `usage`, `new`, `exec`, `stop`, `drivemount`
All global flags (`--auth=oauth2`, `--verbose`) must precede the subcommand (`colab --auth=oauth2 <subcommand> ...`) ([SKILL.md](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md)).

1. **`colab --auth=oauth2 whoami` (`Documented`, [docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md), [issoh.co.jp](https://www.issoh.co.jp/tech/details/12438/)):**  
   Hidden diagnostic command (`hidden=True` in `cli.py`, added `2026-05-07`) that decodes the active OAuth token via `https://oauth2.googleapis.com/tokeninfo`:
   ```text
   Auth provider: oauth2
   Email: zuluzain5@gmail.com
   Audience: 764086051850-6qr4p6gpi6hn506pt8ejuq83di341hur.apps.googleusercontent.com
   Expires in: 52m
   Scopes:
     - email
     - https://www.googleapis.com/auth/cloud-platform
     - https://www.googleapis.com/auth/colaboratory
     - https://www.googleapis.com/auth/drive.file
     - https://www.googleapis.com/auth/userinfo.email
     - openid
   ```
2. **`colab --auth=oauth2 sessions` (`Documented`, [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md), [docs/demos.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/demos.md)):**  
   Queries `GET https://colab.research.google.com/tun/m/assignments?authuser=0`, prunes dead local entries from `~/.config/colab-cli/sessions.json`, and prints tracked `[<name>]` plus untracked `[?]` web/orphan VMs:
   ```text
   [worker] m-s-2b3k9x1p4q8r | Hardware: A100 | Variant: GPU
   [?] m-s-9z8y7x6w5v4u | Hardware: T4 | Variant: GPU
   ```
   When no VMs exist on the server:
   ```text
   [colab] Pruned 1 stale local session(s).
   [colab] No active sessions found on server.
   ```
3. **`colab --auth=oauth2 usage` (`Documented`, [docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md), [SKILL.md](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md); `Measured`, [note.com/unco3](https://note.com/unco3/n/n47710f23e25a)):**  
   *Availability:* **Does NOT exist in `0.6.0`**; added on `2026-08-10` and available in `0.7.4`. Calls `GET https://colab.research.google.com/tun/m/ccu-info` (`get_ccu_info()`):
   ```text
   Current balance: 195.61 compute units
   Usage rate: 6.77/hr
   Active assignments: 1
   ```
   *(Note: Your locked workflow checks remaining CU in the Colab web Resources panel, which is required on `0.6.0`; upgrading to `0.7.4` also exposes this exact endpoint via `colab usage`.)*
4. **`colab --auth=oauth2 new [-s NAME] [--gpu T4|L4|G4|H100|A100] [--tpu v5e1|v6e1] [--high-mem]` (`Documented`, [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md), [docs/demos.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/demos.md)):**  
   Performs the 2-step XSRF handshake (`GET` then `POST /tun/m/assign?nbh=<uuid>&variant=GPU&accelerator=A100`), waits for the Jupyter kernel to reach `idle`, and registers the session in `sessions.json`:
   ```text
   [colab] Creating session 'worker'...
   [colab] Session READY.
   ```
   If the account lacks entitlement or capacity for the requested GPU (HTTP `412 Precondition Failed` or `503`), `0.7.4` prints (`CHANGELOG.md` #112):
   ```text
   [colab] Failed to assign VM (412 Precondition Failed). The requested hardware (H100) may require a higher Colab tier or has no capacity.
   ```
5. **`colab --auth=oauth2 exec [-s NAME] [-f FILE] [--timeout SECONDS] [--env KEY=VALUE] [--output-image PATH] [CODE]` (`Documented`, [docs/02\_execution\_and\_interactive.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/02_execution_and_interactive.md)):**  
   *Note:* `--env KEY=VALUE` was added in `0.7.0` (#65) and is absent in `0.6.0`. For a `.py` file or inline code, `colab exec` emits no wrapper banner—it streams the remote cell's `stdout` and `stderr` directly to local `stdout`/`stderr` and exits with code `0` on success or `1` on remote unhandled exception (`execute_reply.status == "error"`).
6. **`colab --auth=oauth2 stop [-s NAME]` (`Documented`, [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md), [docs/demos.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/demos.md)):**  
   Performs the 2-step XSRF teardown (`GET /tun/m/unassign/<endpoint>?authuser=0` to acquire the XSRF token, then `POST /tun/m/unassign/<endpoint>?authuser=0` with `X-Goog-Colab-Token`) and removes the session from `sessions.json`:
   ```text
   [colab] Stopping session 'worker'...
   [colab] Session terminated.
   ```
7. **`colab --auth=oauth2 drivemount [-s NAME] [PATH]` (`Documented`, [docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md), [google-colab-cli#103](https://github.com/googlecolab/google-colab-cli/issues/103)):**  
   Mounts Google Drive via FUSE to `PATH` (default `/content/drive`):
   ```text
   [colab] Mounting Google Drive to '/content/drive' on worker...
   [colab] Intercepted Drive Auth Request. Connecting to https://colab.research.google.com...
   [colab] REQUIRED: Google Drive Authorization needed.
   Please visit: https://accounts.google.com/o/oauth2/v2/auth?client_id=...&redirect_uri=https%3A%2F%2Fcolab.research.google.com%2Ftun%2Fm%2Fauthorize-for-drive-credentials-ephem&response_type=none%2Bgsession&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive&login_hint=zuluzain5%40gmail.com...
   Press Enter after you have granted access...
   [colab] Authorizing VM...
   [colab] Credentials propagated. Resuming mount...
   Mounted at /content/drive
   ```

### 2.3 GPU enum for `colab new --gpu`: Which of `T4`, `L4`, `A100`, `A100 High-RAM`, `G4`, `H100` are valid on Google AI Pro?
* **Valid CLI `--gpu` enum strings (`Documented`, [SKILL.md](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md), [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md), [issoh.co.jp](https://www.issoh.co.jp/tech/details/12438/)):**  
  The `Gpu` enum in `src/colab_cli/models.py` defines exactly five values (matched case-insensitively): **`T4`**, **`L4`**, **`G4`**, **`H100`**, **`A100`**.
* **Warning on `"A100 High-RAM"` (`Documented`, [issoh.co.jp](https://www.issoh.co.jp/tech/details/12438/), [google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47)):**  
  `A100 High-RAM` is **not** a valid `--gpu` value. In `src/colab_cli/commands/session.py`, if `--gpu` is passed a string not in `{T4, L4, G4, H100, A100}`, the CLI silently falls back to `Gpu.A100` (`variant=GPU&accelerator=A100`). High-RAM is controlled separately via `--high-mem` (`shape=hm`, added in `0.7.0`).

| Requested Flag | Valid in CLI Enum? | Valid on Google AI Pro (200 CU)? | Evidence & Hardware Assigned |
| :--- | :--- | :--- | :--- |
| `--gpu T4` | **Yes** (`Documented`) | **Yes** (`Documented` & `Measured`) | `NVIDIA Tesla T4` (16 GB GDDR6, ~12.7 GB system RAM, ~1.88–1.96 CU/hr, [toolset](https://note.com/toolset/n/n67bae780888a?hl=en)). |
| `--gpu L4` | **Yes** (`Documented`) | **Yes** (`Documented` & `Measured`) | `NVIDIA L4` (22.5–24 GB GDDR6, ~53–62.8 GB system RAM, ~3.92–4.82 CU/hr, [toolset](https://note.com/toolset/n/n67bae780888a?hl=en), [google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47)). *(Not default per locked plan.)* |
| `--gpu A100` (no `--high-mem`) | **Yes** (`Documented`) | **Yes** (`Documented` & `Measured`) | Sends `variant=GPU&accelerator=A100` (`Shape.STANDARD`). Assigns `NVIDIA A100-SXM4-40GB` (~83.5 GB system RAM, ~5.39–6.77 CU/hr; occasionally 80GB at ~6.77 CU/hr if 40GB pool is full, [toolset](https://note.com/toolset/n/n67bae780888a?hl=en), [gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en)). |
| `--gpu A100 --high-mem` (`0.7.0+`) | **Yes** (`--gpu A100` + `--high-mem`) | **Yes on backend, BARRED by plan** | Sends `variant=GPU&accelerator=A100&shape=hm` (`Shape.HIGH_RAM`). Assigns `NVIDIA A100-SXM4-80GB` (~167 GB system RAM, ~11–15 CU/hr, [google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47), [toolset](https://note.com/toolset/n/n67bae780888a?hl=en)). **Barred by locked plan.** |
| `--gpu "A100 High-RAM"` | **No** (Invalid string) | N/A | Silently falls back to `Gpu.A100` (`Shape.STANDARD`) in `commands/session.py` ([issoh.co.jp](https://www.issoh.co.jp/tech/details/12438/)). |
| `--gpu G4` | **Yes** (`Documented`) | **Region / Pool Dependent (`Measured`)** | Assigns `NVIDIA RTX PRO 6000 Blackwell Server Edition` (96 GB GDDR7 VRAM, 176 GB system RAM, ~6.71–6.77 CU/hr, [r/GeminiAI](https://www.reddit.com/r/GeminiAI/comments/1x2uoz1/i_just_found_this_out/), [r/GoogleColab](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/)). Grayed out on some JP AI Pro accounts ([gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en)). |
| `--gpu H100` | **Yes** (`Documented`) | **No** (`Documented` & `Measured`) | Gated to Google AI Ultra / Colab Pro+ ("Premium GPU access", [Google Blog 2026-09-22](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/); confirmed grayed out on AI Pro in [toolset](https://note.com/toolset/n/n67bae780888a?hl=en) and [gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en)). |

### 2.4 When you pass `--gpu A100`, which SKU is assigned (40GB Standard, 80GB, High-RAM)? How do you force 40GB Standard and avoid High-RAM (~11–13 CU/hr)?
* **How Colab's `/tun/m/assign` API maps `--gpu A100` (`Documented`, [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md), [google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47)):**  
  Colab's assignment endpoint accepts three orthogonal query parameters: `variant=GPU`, `accelerator=A100`, and an optional `shape=hm`.
  1. **When you run `colab new -s worker --gpu A100` (without `--high-mem`):** The CLI sends `variant=GPU&accelerator=A100` with **no `shape` parameter** (`Shape.STANDARD`). On Google's backend, Standard-shape A100 maps to a Google Cloud `a2-highgpu-1g` slice (**`NVIDIA A100-SXM4-40GB`**, 12 vCPUs, ~83.5 GB system RAM, ~5.39–6.77 CU/hr, [google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47), [toolset](https://note.com/toolset/n/n67bae780888a?hl=en)).
  2. **When you run `colab new -s worker --gpu A100 --high-mem` (in `0.7.0+`):** The CLI sends `variant=GPU&accelerator=A100&shape=hm` (`Shape.HIGH_RAM`). This maps to an `a2-ultragpu-1g` / high-memory slice (**`NVIDIA A100-SXM4-80GB`**, ~167 GB system RAM) that burns ~11–15 CU/hr ([google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47), [toolset](https://note.com/toolset/n/n67bae780888a?hl=en)).
  3. **Can `--gpu A100` (without `--high-mem`) ever assign an 80GB A100?** **Yes (`Measured`, [gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en), 2026-09-23).** During testing of a newly upgraded Google AI Pro (200 CU) account, selecting Standard A100 assigned an `NVIDIA A100-SXM4-80GB` at a burn rate of **`6.77 CU/hr`** (not the High-RAM ~13–15 CU/hr rate), because Colab's backend does not expose a separate `40GB` vs `80GB` accelerator parameter—only `accelerator=A100` and `shape=hm`.
* **How to force Standard shape and strictly prevent High-RAM (~11–15 CU/hr) burn:**
  1. **Never pass `--high-mem`** to `colab new` (on `0.6.0`, `--high-mem` does not exist; on `0.7.4`, omitting `--high-mem` guarantees `shape=hm` is never sent, [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)).
  2. **Add an immediate pre-flight guard in `start_server.py` (and/or `colab usage`):** Have `start_server.py` inspect system RAM (`psutil.virtual_memory().total`) on boot. On A100, Standard shape has **~83.5 GB system RAM**, whereas High-RAM A100 has **~167 GB system RAM** (`> 120 GB`, [google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47)). (Note: G4 Standard has ~176 GB system RAM at ~6.71–6.77 CU/hr, so only fail the RAM check when `"A100"` is in the GPU name!) Even better, on `0.7.4`, run `colab --auth=oauth2 usage` right after `colab new` and abort with `colab stop` if `Usage rate:` exceeds `8.0/hr`.

### 2.5 Is Colab G4 (`RTX PRO 6000 Blackwell`) actually available on Google AI Pro 200 CU, or only on a higher tier?
* **Official Documentation Status: Silent on per-SKU tier gating (`Documented`).**  
  Google's [2026-03-25 Colab Release Notes](https://colab.research.google.com/notebooks/relnotes.ipynb) announced *"Nvidia RTX Pro 6000 Blackwell Server Edition GPUs on G4 machines"* (96 GB GDDR7 VRAM, 960 BF16 TFLOPs), and the [2026-09-22 Google Developers Blog](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/) states that Google AI Pro includes *"priority access to faster accelerators"* while Google AI Ultra unlocks *"Premium GPU access"*, without publishing an explicit GPU-to-tier table.
* **Field Measurements: Available on Google AI Pro in US/UK/EU pools at ~6.71–6.77 CU/hr, but grayed out on some JP accounts (`Measured`).**  
  - **Positive measurements on Google AI Pro (200 CU):** On **2026-10-11**, Google AI Pro subscribers in [r/GeminiAI](https://www.reddit.com/r/GeminiAI/comments/1x2uoz1/i_just_found_this_out/) verified live allocation of the **G4 GPU (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, 96 GB VRAM, 176 GB System RAM, 235 GB disk)** under the 200 CU plan, reporting ~30 hours/month of runtime (**~6.71–6.77 CU/hr**, matching the `6.71–6.74 CU/hr` measured in [r/GoogleColab](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/) and [google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47)).
  - **Negative measurement on Google AI Pro (200 CU):** On **2026-09-23**, Japanese tester [gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en) observed that `T4`, `L4`, and `A100` were selectable on AI Pro, while `G4` and `H100` were grayed out in that region.
  - **Practical implication for your worker script:** Try `colab --auth=oauth2 new -s worker --gpu G4` first (96 GB GDDR7 VRAM at ~6.74 CU/hr—the same burn rate as Standard A100 40GB, with 2.4x the VRAM for 100K-context KV cache!). If your regional pool rejects `G4` with HTTP `412 Precondition Failed` or `503`, fall back immediately to `colab --auth=oauth2 new -s worker --gpu A100`.

### 2.6 Breaking changes `0.6.0` → `0.7.4` that affect this boot path
1. **Removal of the local keep-alive background process (`0.7.4`, PR #144, 2026-09-25, `Documented`):** `0.6.0` spawned a local daemon pinging `/tun/m/<endpoint>/keep-alive/` every 60s. `0.7.4` removes this daemon completely ([docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)). Your boot path **cannot** rely on `colab new` keeping an idle VM alive in the background; `colab exec` must remain connected.
2. **Automatic 1-hour proxy token refresh (`0.7.4`, PR #149, 2026-09-25, `Documented`):** `0.6.0` drops the session locally after 60 minutes ([google-colab-cli#106](https://github.com/googlecolab/google-colab-cli/issues/106)), making `colab stop` fail after a >1h coding session unless you manually kill the VM in the web UI. `0.7.4` refreshes `RuntimeProxyInfo.token` automatically when within 5 minutes of expiry ([docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)).
3. **`jupyter-kernel-client` version sensitivity (`0.6.0` vs `0.7.2` vs `0.7.4`, `Documented`):**
   - `0.6.0` crashes with `TypeError: KernelHttpClient.__init__() got an unexpected keyword argument 'headers'` if `jupyter-kernel-client` is upgraded to `1.0.0` ([google-colab-cli#94](https://github.com/googlecolab/google-colab-cli/issues/94)).
   - `0.7.2` crashes with `AttributeError: module 'jupyter_kernel_client' has no attribute 'JupyterSubprotocol'` on standard PyPI installs ([google-colab-cli#137](https://github.com/googlecolab/google-colab-cli/issues/137)).
   - `0.7.4` fixes the dependency pin.
4. **New flags and commands in `0.7.4` (`Documented`, [CHANGELOG.md](https://github.com/googlecolab/google-colab-cli/blob/main/CHANGELOG.md)):** Adds `colab usage`, `colab ssh`, `--high-mem`, and `--env KEY=VALUE` on `colab exec` (allowing you to pass `--env LLAMA_API_KEY=...` directly from the CLI instead of hardcoding secrets in `start_server.py`).

---

## 3. Drive Mount (`colab drivemount`)

### 3.1 Exact `colab drivemount` flow, including the browser grant
* **Step-by-step protocol (`Documented` in [docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md), [colabtools/google/colab/drive.py](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py), and [google-colab-cli#103](https://github.com/googlecolab/google-colab-cli/issues/103)):**
  1. **Kernel Injection (`timeout=600`):** Running `colab --auth=oauth2 drivemount -s worker /content/drive` attaches `drivefs_hook` to `ColabRuntime.colab_request_hook` and executes `from google.colab import drive; drive.mount('/content/drive')` inside the remote Jupyter kernel with a 600-second CLI timeout ([docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)).
  2. **Ephemeral Auth Request (`timeout_sec=120` in kernel):** Inside `google/colab/drive.py`, `mount()` calls `_mount(..., ephemeral=True)`, which sends a Jupyter `colab_request` message (`request_auth` with `{'authType': 'dfs_ephemeral'}`) over the WebSocket `iopub`/`stdin` channel and blocks for **120 seconds** (`timeout_ms = 120000` in [drive.py line 111](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)).
  3. **Dry-Run Pre-Flight Check:** `drivefs_hook` intercepts the message and sends:
     ```http
     GET https://colab.research.google.com/tun/m/credentials-propagation/<endpoint>?authuser=0&authtype=dfs_ephemeral&version=2&dryrun=true&propagate=true&record=false
     ```
     with header `X-Colab-Tunnel: Google` and your CLI OAuth Bearer token.
  4. **Interactive Browser Grant:** If the backend responds with `"action": "PROMPT"`, it returns an XSRF `token` and a Google OAuth URL containing `redirect_uri=https://colab.research.google.com/tun/m/authorize-for-drive-credentials-ephem`, `response_type=none+gsession`, `scope=https://www.googleapis.com/auth/drive`, and `login_hint=zuluzain5@gmail.com`. The CLI prints the URL and blocks reading from `/dev/tty` (`Press Enter after you have granted access...`, [google-colab-cli#100](https://github.com/googlecolab/google-colab-cli/issues/100)).
  5. **Credential Propagation & FUSE Startup:** Once you click **Allow** in the browser (which redirects to `colab.research.google.com/tun/m/authorize-for-drive-credentials-ephem` to bind a Drive `gsession` to your Colab backend session) and press **Enter** in the terminal **within 120 seconds**, the CLI sends:
     ```http
     POST https://colab.research.google.com/tun/m/credentials-propagation/<endpoint>?authuser=0&authtype=dfs_ephemeral&version=2&dryrun=false&propagate=true&record=false
     ```
     with header `X-Goog-Colab-Token: <token>`. Upon HTTP `200` (`{"success": true}`), the CLI sends `colab_reply` over the kernel `stdin` channel. `drive.py` then spawns `/opt/google/drive/drive --features=...` (DriveFS FUSE daemon) and prints `Mounted at /content/drive` ([drive.py](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)).

### 3.2 Failure mode when the browser is signed into a different Google account (`authuser`), and how to pin `zuluzain5@gmail.com`
* **Exact Failure Mode (`Documented` & `Measured` in [google-colab-cli#103](https://github.com/googlecolab/google-colab-cli/issues/103), 2026-08-09):**  
  If your default browser has multiple Google accounts signed in and `zuluzain5@gmail.com` is **not** `authuser=0` (for example, your primary browser profile is `authuser=0` for a personal/work account and `zuluzain5@gmail.com` is `authuser=1`), approving the OAuth screen and pressing Enter causes `POST /tun/m/credentials-propagation/<endpoint>` to fail with HTTP `400 Bad Request`:
  ```text
  [colab] Authorizing VM...
  [colab] Error propagating: 400
  ```
  Subsequently, the remote cell fails or hangs until the 120s kernel timeout expires (`ValueError: Mount failed` / `Timed out receiving message from renderer`).
* **Why it happens (`Measured` in [google-colab-cli#103](https://github.com/googlecolab/google-colab-cli/issues/103)):**  
  Even though the CLI's OAuth URL includes `login_hint=zuluzain5%40gmail.com`, the redirect endpoint (`https://colab.research.google.com/tun/m/authorize-for-drive-credentials-ephem`) binds the resulting `gsession` against the browser's active `colab.research.google.com` cookie context (`authuser=0`). When the CLI (authenticated via Bearer token as `zuluzain5@gmail.com`) calls `POST /tun/m/credentials-propagation/<endpoint>?authuser=0`, the backend sees no pending `dfs_ephemeral` grant for `zuluzain5@gmail.com` on that endpoint and rejects it with `400`. Maintainer debugging in [#103](https://github.com/googlecolab/google-colab-cli/issues/103) confirmed that simply changing `authuser` on the CLI request does not solve the mismatch.
* **Secondary Headless Timeout Failure (`Measured` in [google-colab-cli#113](https://github.com/googlecolab/google-colab-cli/issues/113), 2026-08-19):**  
  Even after `Credentials propagated. Resuming mount...` succeeds, if you took longer than **120 seconds** in the browser—or if the local DriveFS daemon waits for web-UI tunnel mediation—`drive.mount()` raises a 120s timeout inside the cell despite `colab drivemount` having a 600s outer timeout.
* **How to pin `zuluzain5@gmail.com` reliably:**
  1. Run `colab --auth=oauth2 whoami` before `colab new` to verify `Email: zuluzain5@gmail.com`.
  2. Maintain a **dedicated Chrome/Brave browser profile** (or Firefox Multi-Account Container) where **`zuluzain5@gmail.com` is the ONLY signed-in Google account (`authuser=0`)**, and where `https://colab.research.google.com/` is already logged in as `zuluzain5@gmail.com`.
  3. Always open the `colab drivemount` authorization URL inside that single-account browser profile and complete the click-through + `Enter` keystroke **within 60 seconds** (well inside `drive.py`'s 120s internal ceiling).
  4. If `colab drivemount` ever hangs at `Resuming mount...` (Issue [#113](https://github.com/googlecolab/google-colab-cli/issues/113)), run `colab --auth=oauth2 url -s worker --open` in that same browser profile so the browser tab completes the DriveFS handshake.

### 3.3 Does a previous grant persist across new VMs, or is every VM interactive?
* **Every new CLI VM is interactive (`Documented`, [docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md), [colabtools/drive.py](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py), [Colab FAQ](https://research.google.com/colaboratory/faq.html)).**  
  Two mechanisms prevent persistence across new CLI VMs:
  1. `google.colab.drive.mount()` hardcodes `ephemeral=True` (`authtype=dfs_ephemeral`), which scopes the credential strictly to the single runtime `<endpoint>` ID (`m-s-...`).
  2. While the [Colab FAQ](https://research.google.com/colaboratory/faq.html) notes that the web UI can remember Drive consent for a specific saved notebook ID (`nbh`) owned and unedited by any other user, `colab new` generates a **brand-new random UUID v4 converted to a 44-character web-safe notebook hash (`nbh`) on every session creation** ([docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)) and `drivefs_hook` explicitly passes `record=false` ([docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)). Consequently, **every `colab new` VM requires one interactive browser grant** to mount `/content/drive`.

### 3.4 Is mount idempotent if `/content/drive` is already mounted?
* **Idempotent at the filesystem level, but `colab drivemount` still fires the auth request hook first (`Documented` in [colabtools/google/colab/drive.py lines 105–138](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)).**  
  Inspecting the exact execution order in `google/colab/drive.py`:
  1. `_mount()` **first** executes `_message.blocking_request('request_auth', request={'authType': 'dfs_ephemeral'}, timeout_sec=120)` on line 110—**before** checking whether `/content/drive` is already mounted!
  2. Only after `blocking_request` returns does `_mount()` check `already_mounted = _os.path.isdir(_os.path.join(mountpoint, 'My Drive'))` on line 127.
  3. If `already_mounted` is `True` and `force_remount=False`, `drive.py` prints:
     ```text
     Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
     ```
     and returns early without spawning a second `drive` FUSE process.
  - **Operational rule for your worker loop:** Because `drive.mount()` calls `request_auth` before checking `already_mounted`, **do not blindly re-run `colab drivemount`** if you aren't sure whether Drive is mounted. Check inside Python (`os.path.ismount('/content/drive')` or `os.path.isdir('/content/drive/MyDrive')`) before invoking `colab drivemount`.

---

## 4. Tunnel (`cloudflared` / `trycloudflare`)

### 4.1 Are outbound `cloudflared tunnel --url http://localhost:PORT` / `trycloudflare` URLs allowed on a paid Colab VM with positive CU?
* **Yes, for single-operator interactive compute while your CU balance is positive (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)).**  
  The official [Google Colaboratory FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html) explicitly distinguishes between prohibited usage on **all** Colab runtimes (such as torrenting, crypto mining, password cracking, or public multi-user web hosting unrelated to interactive compute) and restrictions that apply **only** to unpaid runtimes without a positive Compute Unit balance:
  > *"The following are disallowed from managed Colab runtimes running free of charge, without a positive Colab compute unit balance and may be terminated at any time without warning:*  
  > *- remote control such as SSH shells, remote desktops*  
  > *- bypassing the notebook UI to interact primarily via a web UI*  
  > *... You can remove these types of restrictions by purchasing one of our paid plans here [and maintaining a positive compute unit balance]."* — [Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)
  
  In fact, `google-colab-cli` `0.7.0+` itself ships a built-in `colab ssh` command that downloads `cloudflared` onto the Colab VM and runs an outbound Cloudflare tunnel (`cloudflared access ssh`) to expose `sshd` ([docs/02\_execution\_and\_interactive.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/02_execution_and_interactive.md), [issoh.co.jp](https://www.issoh.co.jp/tech/details/12438/)). At the network layer, outbound connections from a Colab VM to Cloudflare's edge (`*:7844` QUIC/HTTP2 and `https://api.trycloudflare.com`) are unrestricted.

### 4.2 What happens to that tunnel on free-tier or when CU hits zero?
* **Policy violation & unwarned runtime termination (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html); `Measured`, [Stack Overflow #76860422](https://stackoverflow.com/questions/76860422/need-help-understanding-a-warning-from-google-colab-disallowed-code-execution)).**  
  - When a paid account's CU balance hits `0`, or on a Free-tier account, the runtime operates *"without a positive Colab compute unit balance"* ([Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)).
  - Running `cloudflared` or exposing an API/web UI without positive CUs triggers Colab's automated abuse scanner (which inspects running process names/command lines and tunnel traffic patterns).
  - In the web UI, Colab displays the warning banner (`Measured`, [Stack Overflow #76860422](https://stackoverflow.com/questions/76860422/need-help-understanding-a-warning-from-google-colab-disallowed-code-execution)):
    > *"You may be executing code that is disallowed which may terminate your runtime without warning. Colab prioritizes interactive notebook compute and disallows some types of usage when executing code without compute units... Your compute unit balance is 0."*
  - Shortly thereafter, the backend forcibly terminates the VM (`SIGKILL` / container deletion), severing the `trycloudflare.com` tunnel (`HTTP 530 / Error 1033 Argo Tunnel error` on the laptop client). Repeated violations at `0 CU` can trigger temporary or permanent GPU eligibility bans on the Google account ([Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)).

### 4.3 Current `cloudflared` URL pattern on `stdout`/`stderr` (exact regex)
* **Stream & Format (`Documented` & `Measured` on `cloudflared version 2026.9.1 (built 2026-09-17)`, [Flavio Copes, 2026-09-22](https://flaviocopes.com/cloudflare-quick-tunnels/)):**  
  `cloudflared tunnel --url http://127.0.0.1:8080` writes all operational logs—including the generated Quick Tunnel URL—to **`stderr`** (not `stdout`, unless redirected with `stderr=subprocess.STDOUT`):
  ```text
  2026-09-22T12:41:08Z INF Requesting new quick Tunnel on trycloudflare.com...
  2026-09-22T12:41:12Z INF +--------------------------------------------------------------------------------------------+
  2026-09-22T12:41:12Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
  2026-09-22T12:41:12Z INF |  https://kingston-inside-best-graphic.trycloudflare.com                                    |
  2026-09-22T12:41:12Z INF +--------------------------------------------------------------------------------------------+
  ```
* **Exact Python Regex (`Documented` & `Measured`):**  
  ```python
  TRYCLOUDFLARE_URL_RE = re.compile(
      r"https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com"
  )
  ```
  - **Why `(?!api\.)` and `(?:-[a-z0-9]+)+` matter:** Quick Tunnel hostnames are always 4 hyphen-separated lowercase English dictionary words (e.g., `kingston-inside-best-graphic.trycloudflare.com`). Excluding `api.trycloudflare.com` prevents false matches if `cloudflared` logs an error or debug line referencing its registration endpoint `https://api.trycloudflare.com/tunnel`.
* **DNS Propagation & Negative Caching Trap (`Measured`, [Flavio Copes, 2026-09-22](https://flaviocopes.com/cloudflare-quick-tunnels/)):**  
  After `cloudflared` prints the URL to `stderr`, global DNS registration takes **3 to 33 seconds** to complete. Furthermore, the `trycloudflare.com` SOA record sets a **30-minute negative caching TTL (`1800s`)**; if your local laptop queries the hostname via its system DNS resolver before Cloudflare finishes registering the record, your laptop's OS/router DNS cache may cache `NXDOMAIN` for up to 30 minutes! **Mitigation:** In `start_server.py`, after matching the regex from `stderr`, wait **5–8 seconds** (or verify DNS resolution inside `start_server.py` against `1.1.1.1`) before printing `TUNNEL_READY_URL=<url>` to `stdout`.

### 4.4 Documented way to require an API key on `llama-server` behind `trycloudflare`
* **Official `llama-server` Authentication Flags (`Documented`, [ggml-org/llama.cpp/blob/master/tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)):**  
  `llama-server` provides built-in Bearer token authentication via three mechanisms:
  1. **Command-line flag `--api-key KEY`:** Set one API key (or a comma-separated list of keys) required on all restricted endpoints.
  2. **Environment variable `LLAMA_API_KEY`:** Equivalent to `--api-key`. Prefer this over `--api-key` on the command line so the key is not exposed in `ps aux` output inside the container (and in `0.7.4`, you can inject it via `colab exec --env LLAMA_API_KEY="$LLAMA_API_KEY"`).
  3. **Key file `--api-key-file FNAME` (or env `LLAMA_ARG_API_KEY_FILE`):** Path to a file containing API keys separated by newlines (e.g., `/content/drive/MyDrive/colab-worker/llama_api_key.txt`).
* **Protected vs. Public Endpoints (`Documented`, [llama.cpp/tools/server/README.md](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)):**  
  When `LLAMA_API_KEY` / `--api-key` is set, every request to `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/props`, and `/slots` must include the HTTP header:
  ```http
  Authorization: Bearer <YOUR_API_KEY>
  ```
  Unauthenticated requests receive `HTTP 401 Unauthorized`. Meanwhile, `GET /health` remains publicly accessible without an API key so local/in-guest health probes can poll readiness (`200 OK` vs `503 Loading model`).

---

## 5. Shutdown and Idle Behavior

### 5.1 Does `from google.colab import runtime; runtime.unassign()` still work on AI Pro, and does it stop CU burn?
* **Yes to both (`Documented`, [Colab Release Notes 2022-09-12](https://colab.research.google.com/notebooks/relnotes.ipynb), [colabtools#2568](https://github.com/googlecolab/colabtools/issues/2568)).**  
  Introduced in September 2022 ([colabtools#2568](https://github.com/googlecolab/colabtools/issues/2568)), `from google.colab import runtime; runtime.unassign()` is part of the pre-installed `google.colab` Python package on all Colab runtimes (Free, Pro, AI Pro, Pro+, Ultra).
  - As documented by Colab engineer `@peap` in [colabtools#2568](https://github.com/googlecolab/colabtools/issues/2568):
    > *"The new `unassign()` function marks the currently connected runtime for deletion and disconnects any notebook sessions."*
  - Unlike `!kill -9 -1` or `os.kill(os.getpid(), 9)`—which only kills the container processes and causes the kernel manager to restart the kernel while the VM remains assigned and continues burning Compute Units ([colabtools#2568](https://github.com/googlecolab/colabtools/issues/2568))—`runtime.unassign()` instructs the Colab runtime service to **delete the VM assignment itself**, immediately stopping Compute Unit burn.
  - **Clean cleanup note:** Even when an in-guest watchdog calls `runtime.unassign()`, your local wrapper script should still run `colab --auth=oauth2 stop -s worker` (or `colab --auth=oauth2 sessions`) on exit so `~/.config/colab-cli/sessions.json` is cleaned up locally.

### 5.2 Can an in-guest 30-minute idle watchdog call `unassign()` if the original `colab exec` has already returned?
* **Dependency on Question 1 (`Documented` & `Measured`):**
  1. **If you keep `colab exec` attached in the foreground (the required pattern from Section 1.3):** **Yes, 100% reliable (`Documented`).** Because the watchdog runs *inside* the active `colab exec` Python cell, calling `from google.colab import runtime; runtime.unassign()` when `idle_seconds >= 1800` executes inside the active IPython kernel context, cleanly unassigning the VM and terminating the attached `colab exec` process on your laptop.
  2. **If the original `colab exec` has already returned:** **No, it is both unreliable and moot (`Documented` & `Measured`).**
     - First, as established in Section 1.1, on Google AI Pro (and `0.7.4` with no keep-alive daemon), once `colab exec` returns and no kernel cell is executing, the Colab backend's own idle runtime deletion policy will reap the VM after **~10–15 minutes** ([google-colab-cli#14](https://github.com/googlecolab/google-colab-cli/issues/14), [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950))—long before a 30-minute watchdog timer fires, and even while `llama-server` is actively generating over `trycloudflare`!
     - Second, `google.colab.runtime.unassign()` relies on the IPython kernel environment (`TBE_RUNTIME_ADDR` / kernel message bridge). A detached background daemon (`Popen(..., start_new_session=True)`) running outside the IPython kernel process cannot reliably invoke `google.colab` frontend/kernel RPCs unless it injects code back into the local Jupyter kernel via `jupyter_client` or calls the internal `TBE_RUNTIME_ADDR` HTTP endpoint directly.
     - Therefore, **run the 30-minute idle watchdog directly inside the foreground `start_server.py` loop** invoked by `colab exec --timeout 86400`.

### 5.3 Documented max session length on AI Pro, and what kills an unattended VM
* **Max Session Length on Google AI Pro: Up to 24 hours (`Documented` on Colab paid tiers; `Measured` on AI Pro, [Colab FAQ](https://research.google.com/colaboratory/faq.html), [note.com/toolset](https://note.com/toolset/n/n67bae780888a?hl=en)).**  
  Google AI Pro's dedicated Help Center page ([`16476811`](https://support.google.com/googleone/answer/16476811?hl=en&co=GENIE.Platform=Android)) is silent on exact runtime hours, while the [Colab FAQ](https://research.google.com/colaboratory/faq.html) states: *"Notebooks can run for at most 12 hours, depending on availability and your usage patterns. Colab Pro and Pro+ offer longer runtimes than the free of charge version of Colab"* (up to 24 hours, confirmed in [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)).
* **The 6 mechanisms that kill an unattended VM (`Documented` & `Measured`):**
  1. **No Active Kernel Execution + Disconnected Client (~10–15 minutes, `Measured`, [google-colab-cli#14](https://github.com/googlecolab/google-colab-cli/issues/14), [#160](https://github.com/googlecolab/google-colab-cli/issues/160), [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)):** If `colab exec` exits (or your laptop sleeps/loses Wi-Fi) on Google AI Pro (which lacks background execution), the backend reclaims the VM within ~10–15 minutes.
  2. **Idle Kernel with Open Client (~90 minutes, `Measured`, [Colab Guide](https://www.qwe.edu.pl/tutorial/google-colab-ai-data-analysis/), [Colab FAQ](https://research.google.com/colaboratory/faq.html)):** If a browser tab or client is connected but no cell is executing (`kernel_state == idle`), the idle runtime deletion policy terminates the VM after ~90 minutes.
  3. **24-Hour Hard Wall-Clock Cap (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/faq.html), [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)):** Every Colab VM has a hard maximum lifetime lease (up to 24 hours on paid tiers), after which it is forcibly recycled regardless of activity.
  4. **Container Unhealthy / System RAM or VRAM OOM (`Documented`, [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)):** As `@colaboratory-team` noted in [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950) (2026-04-15), if the container health check fails due to memory exhaustion or kernel death, the backend deletes the runtime as "unhealthy".
  5. **Compute Unit Exhaustion + Disallowed Free-Tier Process (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)):** If your CU balance drops to `0.00` while running `cloudflared` (see Section 5.4).
  6. **2026 Multi-Hour GPU Preemption / Unhealthy Regression (`Measured`, [colabtools#5793](https://github.com/googlecolab/colabtools/issues/5793), [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950), [colabtools#6101](https://github.com/googlecolab/colabtools/issues/6101)):** Users in Jan–Sep 2026 have measured intermittent backend disconnects after 2.5–5 hours of continuous GPU generation even with positive CUs and >40 GB free RAM; checkpointing / fast restarts from Drive cache mitigate this.

### 5.4 What happens to a running GPU VM when paid CU balance hits 0? What do `colab sessions` and the web UI show?
* **Backend & Runtime Behavior when CU hits `0` (`Documented`, [Colab FAQ](https://research.google.com/colaboratory/faq.html); `Measured`, [Stack Overflow #74709472](https://stackoverflow.com/questions/74709472/exhausted-compute-units-google-colab), [#76860422](https://stackoverflow.com/questions/76860422/need-help-understanding-a-warning-from-google-colab-disallowed-code-execution)):**  
  Per the [Colab FAQ](https://research.google.com/colaboratory/faq.html): *"All users can access Colab resources based on availability. If you do not have compute units, you can only use Colab resources reserved for non-paying users."*
  - **On a Premium GPU (`A100` or `G4`) and/or running `cloudflared`:** Because `A100`/`G4` are not in the free-tier resource pool AND `cloudflared` violates free-tier usage rules once `CU == 0`, the running VM is **disconnected and terminated** shortly after the balance reaches `0.00` (whereas a standard T4 notebook doing normal interactive Python without tunnels may occasionally be allowed to finish a short grace window under free-tier quotas before reclamation).
* **What the Web UI shows (`Measured`, [Stack Overflow #74709472](https://stackoverflow.com/questions/74709472/exhausted-compute-units-google-colab), [#76860422](https://stackoverflow.com/questions/76860422/need-help-understanding-a-warning-from-google-colab-disallowed-code-execution)):**
  - The **Resources panel** shows: `Available: 0.00 compute units` and a prompt to purchase Pay As You Go compute units or upgrade.
  - If a tunnel/web UI is running when CU hits `0`, a warning modal appears: *"You may be executing code that is disallowed which may terminate your runtime without warning... Your compute unit balance is 0."*
  - Once reclaimed, the top-right status changes to **`Reconnect`** with modal **`Runtime disconnected`** (and attempting to reconnect with `A100` or `G4` selected fails with an unavailable/upgrade prompt until CUs replenish or runtime type is switched to `T4` / `CPU`).
* **What the CLI (`colab sessions` and `colab usage`) shows (`Documented`, [docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md), [docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)):**
  - Once the VM is reclaimed by the backend, running `colab --auth=oauth2 sessions` sees that the endpoint is absent from `GET /tun/m/assignments` and prints:
    ```text
    [colab] Pruned 1 stale local session(s).
    [colab] No active sessions found on server.
    ```
  - In `0.7.4`, running `colab --auth=oauth2 usage` prints:
    ```text
    Current balance: 0.00 compute units
    Usage rate: 0.00/hr
    Active assignments: 0
    ```
  - Subsequent `colab new --gpu A100` or `--gpu G4` at `0.00 CU` fails with HTTP `412 Precondition Failed` ([CHANGELOG.md #112](https://github.com/googlecolab/google-colab-cli/blob/main/CHANGELOG.md)).

---

## 6. Guest CUDA on Colab A100 and G4 (Not L4)

### 6.1 Current Colab CUDA toolkit and NVIDIA driver for A100 and for G4, and whether `sudo` works
* **OS, Python, and Container User (`Documented`, [googlecolab/backend-info](https://github.com/googlecolab/backend-info) `2026.07` image, [docs/demos.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/demos.md), [SKILL.md](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md)):**
  - **Base OS:** `Ubuntu 22.04.5 LTS` (Jammy), Linux kernel `6.6.113+ x86_64`, `glibc 2.35`.
  - **Python & PyTorch:** `Python 3.12.13`, `PyTorch 2.11.0` ([googlecolab/backend-info](https://github.com/googlecolab/backend-info)).
  - **Does `sudo` work? Yes (`Documented`).** Every Colab runtime container executes directly as **`root`** (`uid=0(root) gid=0(root)`, visible in [drive.py line 183](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py) `root@<hostname>` and [SKILL.md](https://github.com/googlecolab/google-colab-cli/blob/main/skills/colab-operator/SKILL.md) `User root`). Commands run with full root privileges inside the container both directly and when prefixed with passwordless `sudo`.
* **NVIDIA Driver and CUDA Toolkit on A100 and G4 (`Documented` & `Measured`):**  
  All Colab GPU VMs (`A100` and `G4`) boot from the same unified Colab container image (`us-docker.pkg.dev/colab-images/public/runtime`), while the host kernel driver is injected by the underlying GCP host node:

| Component | Colab A100 (`A100-SXM4-40GB` / `80GB`) | Colab G4 (`RTX PRO 6000 Blackwell Server Edition`) | Source & Evidence Label |
| :--- | :--- | :--- | :--- |
| **CUDA Compute Capability** | **`sm_80`** (`8.0`, Ampere) | **`sm_120`** (`12.0`, Blackwell workstation/server) | `Documented` (NVIDIA CUDA GPUs spec) & `Measured` ([r/GoogleColab](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/)) |
| **Guest CUDA Toolkit (`nvcc --version`)** | **CUDA `13.0` (`13.0.88`)** *(upgraded in 2026 from CUDA `12.5` / `12.8`)* | **CUDA `13.0` (`13.0.88`)** *(Blackwell `sm_120` requires CUDA `>= 12.8`)* | `Measured` on `2026-10-11` (`cuda=13.0.88`, [note.com/unco3](https://note.com/unco3/n/n47710f23e25a)) and `2026-02-17` (`CUDA Version: 13.0`, [r/GoogleColab](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/)) |
| **Host NVIDIA Driver (`nvidia-smi`)** | **`580.82.07`** (R580 branch supporting CUDA 13.0; older A2 host nodes may occasionally report R550/R570, [colabtools#5061](https://github.com/googlecolab/colabtools/issues/5061)) | **`580.82.07`** (`NVIDIA-SMI 580.82.07`, Driver Version `580.82.07`) | `Measured` verbatim from `nvidia-smi` on Colab G4 ([r/GoogleColab, 2026-02-17](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/)) |
| **VRAM & System RAM (Standard Shape)** | `40,960 MiB` (`40 GB` HBM2e) / `~83.5 GB` System RAM (or `80 GB` / `~83.5 GB` when 80GB assigned at Standard rate) | `97,887 MiB` (`96 GB` GDDR7 ECC) / `~176 GB` System RAM | `Measured` ([google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47), [r/GoogleColab](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/), [r/GeminiAI](https://www.reddit.com/r/GeminiAI/comments/1x2uoz1/i_just_found_this_out/)) |

### 6.2 Will a CUDA `llama-server` binary compiled on Colab and cached on Drive work on the next VM of the same GPU class? Must A100 and G4 have separate binaries (`sm_80` vs `sm_120`)?
* **Will a Drive-cached binary work on the next VM of the same GPU class? Yes, with two packaging rules (`Documented`):**  
  Because every Colab VM runs the exact same container image (`Ubuntu 22.04.5 LTS`, `/usr/local/cuda`) until Google bumps the runtime image (`googlecolab/backend-info`), a `llama-server` binary compiled on Colab and cached in `/content/drive/MyDrive/colab-worker/bin/` works immediately on subsequent VMs **provided you follow two rules**:
  1. **Either link `ggml`/`llama` statically (`-DBUILD_SHARED_LIBS=OFF`) or bundle `libllama.so` / `libggml*.so`:** By default, recent `llama.cpp` CMake builds produce shared libraries (`libggml.so`, `libggml-cuda.so`, `libllama.so`). Passing `-DBUILD_SHARED_LIBS=OFF` compiles a single self-contained `llama-server` executable that only dynamically links against system libraries (`libc.so.6`, `libcuda.so.1`, `libcudart.so`, `libcublas.so`, `libcublasLt.so`).
  2. **Copy the binary from `/content/drive/...` to local `/usr/local/bin/llama-server` and `chmod 755` before executing:** Executing binaries directly off the DriveFS FUSE mount (`/content/drive`) incurs FUSE latency or permission issues; always `shutil.copy2()` to `/usr/local/bin/llama-server` on VM startup.
* **Must A100 (`sm_80`) and G4 (`sm_120`) have separate binaries? No—if you compile with `-DCMAKE_CUDA_ARCHITECTURES="80;120"`, one fatbin works on both; if you compile with `native`, they WILL fail on each other (`Documented`):**
  - If you build `llama.cpp` with `-DGGML_CUDA=ON` and leave `CMAKE_CUDA_ARCHITECTURES` at its default (`native`), CMake detects only the GPU attached to the current build VM (e.g., `sm_80` on A100) and embeds SASS for `sm_80` only. Running that binary on a G4 (`sm_120`) fails immediately with `CUDA error: no kernel image is available for execution on the device`.
  - **Solution A (Single Dual-Arch Fatbin — Recommended):** Compile once on Colab (CUDA 13.0) with:
    ```bash
    cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_CUDA_ARCHITECTURES="80;120"
    cmake --build build --config Release -t llama-server -j $(nproc)
    ```
    This embeds native `sm_80` (Ampere A100) SASS, native `sm_120` (Blackwell G4) SASS, and PTX fallback into a single `llama-server` binary (~140–180 MB) that starts with **zero PTX JIT delay** on both A100 and G4.
  - **Solution B (Per-Arch Cache Directory):** If you prefer smaller binaries (~85 MB each), key your Drive cache path by compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` → `8.0` vs `12.0`), e.g., `/content/drive/MyDrive/colab-worker/bin/llama-server-sm_80` and `llama-server-sm_120`.

### 6.3 How to detect a broken cached binary after a Colab image bump
When Google updates the Colab container image ([googlecolab/backend-info](https://github.com/googlecolab/backend-info)), a cached `llama-server` binary can break in three ways:
1. **CUDA Major Version Bump (Missing `.so` SONAME):** E.g., `libcudart.so.12` → `libcudart.so.13` or `libcublas.so.12` → `libcublas.so.13`.
2. **glibc / libstdc++ Symbol Mismatch:** If Google upgrades the container base from Ubuntu 22.04 to 24.04.
3. **CUDA Driver / Runtime Incompatibility (`cudaErrorInsufficientDriver` or `cudaErrorNoKernelImageForDevice`):** `ldd` and `--version` succeed, but CUDA initialization fails when probing the GPU.

* **3-Step Automated Verification Check in `start_server.py` (takes <1.5 seconds on boot):**
  1. **Check 1 — Shared Library Resolution (`ldd`):** Run `ldd /usr/local/bin/llama-server`. If output contains `"not found"` or `"version ... not found"`, mark the cached binary invalid.
  2. **Check 2 — Binary & Linker Execution (`--version`):** Run `/usr/local/bin/llama-server --version` (timeout `10s`, expects exit code `0`).
  3. **Check 3 — Live GPU Backend & Compute Capability Probe (`--list-devices`):** Run `/usr/local/bin/llama-server --list-devices` (timeout `15s`). This forces `ggml-cuda` to initialize the NVIDIA driver, query the CUDA runtime, and enumerate `CUDA0: NVIDIA A100...` or `CUDA0: NVIDIA RTX PRO 6000 Blackwell...`. Verify that exit code is `0` and `"CUDA0"` appears in `stdout`/`stderr`.
  4. **Optional Sidecar Metadata (`build-info.json`):** Save `{ "cuda_soname": "libcudart.so.13", "archs": ["80", "120"], "llama_commit": "<sha>" }` alongside the binary on Drive; if any check fails, trigger an automatic rebuild on the VM and overwrite the Drive cache.

---

## Implications for `start_server.py`

1. **You MUST keep `colab exec` attached — do NOT exit after `Popen(..., start_new_session=True)`:**
   - Because Google AI Pro does **not** include Colab background execution ([Google Developers Blog, 2026-09-22](https://developers.googleblog.com/colab-is-now-part-of-your-google-ai-plan/)) and `google-colab-cli` `0.7.4` runs no client-side keep-alive daemon ([docs/01\_session\_management.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)), letting `start_server.py` exit leaves the VM with no active kernel execution and no live client connection. The Colab backend will delete the VM in **~10–15 minutes** ([google-colab-cli#14](https://github.com/googlecolab/google-colab-cli/issues/14), [#160](https://github.com/googlecolab/google-colab-cli/issues/160), [colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)), even while `llama-server` is busy serving requests over `trycloudflare.com`.
2. **A 20-second `stdout` heartbeat loop is mandatory inside `start_server.py`:**
   - Keep `start_server.py` running in the foreground inside `colab exec --timeout 86400`.
   - Every **20 seconds**, `start_server.py` should poll `http://127.0.0.1:8080/health` and `http://127.0.0.1:8080/slots`, verify that `llama-server` and `cloudflared` (`proc.poll() is None`) are still alive, and print a single heartbeat line with `flush=True` (e.g., `[heartbeat] up=1420s busy_slots=1 idle=0s url=https://...trycloudflare.com`).
   - This simultaneously satisfies: (a) the kernel active-execution requirement ([colabtools#5950](https://github.com/googlecolab/colabtools/issues/5950)), (b) the TFE live-connection requirement ([google-colab-cli#160](https://github.com/googlecolab/google-colab-cli/issues/160)), and (c) `jupyter-kernel-client`'s iopub activity monitor ([docs/04\_automation\_and\_utility.md](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)).
3. **An in-guest 30-minute (`1800s`) idle watchdog calling `runtime.unassign()` is completely supported:**
   - Because `start_server.py` stays running inside the IPython kernel cell, it can query `GET http://127.0.0.1:8080/slots` (with header `Authorization: Bearer <LLAMA_API_KEY>`) every 20 seconds.
   - Whenever any slot has `"is_processing": true` (or `n_decoded > last_n_decoded`), reset `last_active_ts = time.monotonic()`.
   - If `time.monotonic() - last_active_ts >= 1800` (30 minutes of zero token generation), print `[watchdog] 30m idle threshold reached; unassigning runtime...`, terminate `llama-server` and `cloudflared`, and call:
     ```python
     from google.colab import runtime
     runtime.unassign()
     ```
   - This immediately releases the VM and stops CU burn even if your laptop crashed or lost network connectivity.
4. **Which `--gpu` flags are safe on Google AI Pro (and how to block High-RAM A100):**
   - **Safe & Recommended Primary Flags:**
     - `--gpu G4` (no `--high-mem`): Assigns `NVIDIA RTX PRO 6000 Blackwell Server Edition` (96 GB GDDR7 VRAM, 176 GB System RAM) at **~6.71–6.77 CU/hr** where regionally available on AI Pro ([r/GeminiAI](https://www.reddit.com/r/GeminiAI/comments/1x2uoz1/i_just_found_this_out/), [r/GoogleColab](https://www.reddit.com/r/GoogleColab/comments/1r6pt7n/colab_g4_new_card_nvidia_rtx_pro_6000_blackwell/)).
     - `--gpu A100` (no `--high-mem`): Assigns Standard shape `NVIDIA A100-SXM4-40GB` (~83.5 GB System RAM; occasionally 80GB at Standard rate) at **~5.39–6.77 CU/hr** ([toolset](https://note.com/toolset/n/n67bae780888a?hl=en), [gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en)).
   - **Unsafe / Barred Flags:**
     - **Never pass `--high-mem` with `--gpu A100`**: Triggers `shape=hm` (`A100 High-RAM`, ~167 GB System RAM, burning **~11–15 CU/hr**, [google-colab-cli#47](https://github.com/googlecolab/google-colab-cli/issues/47)).
     - **Do not pass `--gpu H100`**: Gated to Ultra/Pro+; fails with `412 Precondition Failed`.
   - **Startup Hardware & Burn-Rate Guard in `start_server.py`:** Immediately check GPU name (`nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader`) and system RAM (`psutil.virtual_memory().total`). If `"A100"` is detected and `system_ram_gb > 110` (indicating an accidental High-RAM assignment), immediately call `runtime.unassign()` so you never burn ~13 CU/hr.

---

## CLI Cheat Sheet (Copy-Paste Commands)

### 0. One-Time Upgrade from `0.6.0` to `0.7.4` (Fixes 1-Hour Session Drop `#106` & Timeout CPU Spin `#82`)
```bash
# Check current version at ~/.local/bin/colab
colab version

# Upgrade directly to 0.7.4 (skip broken 0.7.2 and yanked 0.7.3)
uv tool install --upgrade "google-colab-cli==0.7.4" || pip install --user --upgrade "google-colab-cli==0.7.4"

# Verify CLI identity is pinned to zuluzain5@gmail.com
colab --auth=oauth2 whoami
```

### 1. Check Active Sessions & Compute Unit Balance
```bash
# List all active VMs on backend (including untracked [?] orphans)
colab --auth=oauth2 sessions

# Check remaining Compute Units and live burn rate (0.7.4+; also verify in Web UI Resources panel)
colab --auth=oauth2 usage
```

### 2. Provision High-Bandwidth GPU Session (Try G4 96GB First, Fallback to Standard A100 40GB)
```bash
# Attempt G4 (RTX PRO 6000 Blackwell 96GB, ~6.74 CU/hr) — NO --high-mem flag!
# Falls back to Standard A100 40GB (~5.4–6.8 CU/hr) if G4 returns 412/503 in your region.
colab --auth=oauth2 new -s llm-worker --gpu G4 \
  || colab --auth=oauth2 new -s llm-worker --gpu A100

# Verify assigned hardware and confirm burn rate is < 8.0 CU/hr (not High-RAM ~13 CU/hr)
colab --auth=oauth2 status -s llm-worker
colab --auth=oauth2 usage
```

### 3. Mount Google Drive (Interactive Browser Grant Pinned to `zuluzain5@gmail.com`)
```bash
# IMPORTANT: Open the printed URL in the browser profile where zuluzain5@gmail.com is authuser=0
# and press Enter within 60 seconds (kernel internal timeout is 120s).
colab --auth=oauth2 drivemount -s llm-worker /content/drive
```

### 4. Push & Run `start_server.py` (Keep `exec` Attached with 24h Timeout & API Key Env)
```bash
export LLAMA_API_KEY="sk-colab-$(openssl rand -hex 24)"
echo "Using LLAMA_API_KEY=${LLAMA_API_KEY}"

# Run start_server.py and KEEP THIS TERMINAL ATTACHED for the entire coding session.
# On 0.7.4, --env passes LLAMA_API_KEY cleanly; --timeout 86400 allows up to 24h.
colab --auth=oauth2 exec -s llm-worker \
  -f start_server.py \
  --timeout 86400 \
  --env "LLAMA_API_KEY=${LLAMA_API_KEY}"
```
*(If running on `0.6.0` before upgrading, omit `--env` and read `LLAMA_API_KEY` from `/content/drive/MyDrive/colab-worker/llama_api_key.txt` inside `start_server.py`.)*

### 5. Call the Public OpenAI-Compatible `/v1` Endpoint from Laptop
```bash
# Wait ~8 seconds after TUNNEL_READY_URL is printed so Cloudflare DNS propagates
export COLAB_OPENAI_BASE="https://<printed-4-word-slug>.trycloudflare.com/v1"

curl -sS "${COLAB_OPENAI_BASE}/models" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" | jq .
```

### 6. Mandatory Teardown on Exit (Stop CU Burn Immediately)
```bash
# Always stop the session and verify 0 active assignments on server
colab --auth=oauth2 stop -s llm-worker
colab --auth=oauth2 sessions
colab --auth=oauth2 usage
```

---

## Remaining Unknowns

1. **Exact server-side idle timer threshold on Google AI Pro when `colab exec` disconnects mid-run (`Unknown` in official docs; `Measured` at ~10–15 minutes):**  
   Google's official documentation ([Colab FAQ](https://research.google.com/colaboratory/faq.html), [Google One AI Pro Help](https://support.google.com/googleone/answer/16476811?hl=en&co=GENIE.Platform=Android)) deliberately omits exact minute thresholds for disconnected-client reclamation, stating only that timeouts vary dynamically based on capacity and usage patterns. Empirical CLI measurements place disconnected-client reclamation at **10–15 minutes** ([google-colab-cli#14](https://github.com/googlecolab/google-colab-cli/issues/14)).
2. **Mid-session WebSocket behaviour across the 1-hour `RuntimeProxyInfo.token` boundary (`Documented` caveat, `Unknown` exact server cut-off):**  
   While `google-colab-cli` `0.7.4` refreshes `RuntimeProxyInfo.token` before any new CLI command within 5 minutes of the 3600s TTL ([docs/01\_session\_management.md Section 6](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)), the docs note that already-open WebSocket connections only present the token during the initial HTTP-to-WebSocket upgrade handshake. Whether the runtime proxy ever drops an established `colab exec` WebSocket after multiple hours—requiring the laptop wrapper script to automatically re-attach (`colab exec` in a reconnect mode that attaches to existing `llama-server` + `cloudflared` PIDs if already running)—should be validated during your first >90-minute session.
3. **Regional G4 availability for `zuluzain5@gmail.com` (`Unknown` until probed on account):**  
   Because `G4` (`RTX PRO 6000 Blackwell`) is enabled for Google AI Pro 200 CU users in some regional pools ([r/GeminiAI](https://www.reddit.com/r/GeminiAI/comments/1x2uoz1/i_just_found_this_out/)) and disabled in others ([note.com/gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en)), whether `zuluzain5@gmail.com` can allocate `--gpu G4` can only be confirmed by running `colab --auth=oauth2 new -s test --gpu G4`.
4. **Silent 40GB vs. 80GB assignment on Standard `--gpu A100` (`Measured` variance, `Unknown` backend allocation rule):**  
   Google's `/tun/m/assign` API does not expose a separate parameter to request `A100 40GB` vs `A100 80GB` independently of `shape=hm` (`--high-mem`). While omitting `--high-mem` normally assigns `A100-SXM4-40GB` at ~5.4–6.8 CU/hr, users have occasionally received `A100-SXM4-80GB` at the Standard ~6.77 CU/hr rate ([note.com/gakushiai](https://note.com/gakushiai/n/n71ca7ca09d1b?hl=en)). Checking `colab usage` (`Usage rate`) right after `colab new` is the only deterministic way to verify the exact CU/hr rate assigned to that specific VM.
