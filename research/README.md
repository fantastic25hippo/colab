# Research Archive

This directory tree is strictly an evidence and research archive. Shipping documentation at the repository root ([`README.md`](../README.md), [`decision.md`](../decision.md), [`claims-ledger.md`](../claims-ledger.md), [`ops.md`](../ops.md)) and in [`runbooks/`](../runbooks/) ([`g4-probe.md`](../runbooks/g4-probe.md), [`flash-next-sglang.md`](../runbooks/flash-next-sglang.md), [`fallback-abliterated.md`](../runbooks/fallback-abliterated.md)) are the canonical source of truth. When any archived paper or research note conflicts with a shipping document, the shipping document wins.

## Archive Index

### 1. Historical Research Notes (`research/*.md`)
Ingested 2026-10-14. Foundational exploration and technical notes:
- [`a100_speed_and_session.md`](a100_speed_and_session.md): CMake fatbin targets, G4 bandwidth, reconnect loops, `/slots` watchdog, smoke gates.
- [`blueprint.md`](blueprint.md): Original orchestrator → Drive → Colab worker concept.
- [`busy_gpu.md`](busy_gpu.md): A100 40GB deployment notes.
- [`drive_files.md`](drive_files.md): Checkpoint and binary packing list for Google Drive.
- [`kernel_and_cli.md`](kernel_and_cli.md): Colab CLI 0.7.4, attached exec + heartbeat, Drive mount, `runtime.unassign()`.
- [`llamacpp_flags.md`](llamacpp_flags.md): Deep audit of llama-server CLI flags and traps.
- [`qwen38_27b_max_tokens_per_second_deep_research.md`](qwen38_27b_max_tokens_per_second_deep_research.md): DFlash 2 and speculative throughput analysis.
- [`sidecar_and_verify.md`](sidecar_and_verify.md): DFlash 2 sidecar quantization, tools PEG parser, first-boot grep gates.

### 2. 27B Plan Archive (`research/archive/27b-plan/`)
Historical planning documents for the 27B worker superseded by current shipping runbooks:
- [`architecture.md`](archive/27b-plan/architecture.md): Initial 27B worker architecture and design.
- [`inference.md`](archive/27b-plan/inference.md): Original inference parameters and launch flags.
- [`first-boot.md`](archive/27b-plan/first-boot.md): Initial first-sitting checklists and verification gates.
- [`plan_audit_and_fixes.md`](archive/27b-plan/plan_audit_and_fixes.md): Technical audit identifying compile flags, Curl 2 fixes, and session handling.
- [`bend-hands.md`](archive/27b-plan/bend-hands.md): Shelved Grok → Gemini → Hermes orchestrator plan.

### 3. Flash-Next Audit Dossier (`research/archive/flash-next-audit/`)
Ten-paper research dossier auditing hardware, checkpoints, runtime stack, KV cache, economics, and claims for Qwen 3.8 Flash-Next:
- [`README.md`](archive/flash-next-audit/README.md) & [`CLAIMS.md`](archive/flash-next-audit/CLAIMS.md): Dossier index and claims ledger.
- Papers `01_colab_g4_hardware_and_cu.md` through `10_claims_ledger_and_verdict.md`.

### 4. Flash-Next Chat Trace (`research/archive/flash-next-chat.md`)
- [`flash-next-chat.md`](archive/flash-next-chat.md): Raw chat transcript exploring Qwen Flash-Next deployment options on Colab.

---

> [!IMPORTANT]
> **Precedence Rule:** Shipping docs win on conflict. Do not use archived notes or plans as operational launch instructions.
