# Research Archive

This directory tree is strictly an evidence and research archive. Shipping documentation at the repository root ([`README.md`](../README.md), [`decision.md`](../decision.md), [`claims-ledger.md`](../claims-ledger.md), [`ops.md`](../ops.md)) and in [`runbooks/`](../runbooks/) ([`g4-probe.md`](../runbooks/g4-probe.md), [`flash-next-sglang.md`](../runbooks/flash-next-sglang.md), [`fallback-abliterated.md`](../runbooks/fallback-abliterated.md)) are the canonical source of truth. When any archived paper or research note conflicts with a shipping document, the shipping document wins.

For the full archive index covering foundational research notes as well as these archived materials, link up to [`../README.md`](../README.md).

---

> [!IMPORTANT]
> **Precedence Rule:** Shipping docs at repo root ([`README.md`](../README.md), [`decision.md`](../decision.md), [`claims-ledger.md`](../claims-ledger.md), [`ops.md`](../ops.md), and [`runbooks/`](../runbooks/)) are the source of truth and win on conflict. Do not use archived notes or plans as operational launch instructions.

---

## Archive Index

### 1. Historical Research Notes (`research/27b-notes/`, formerly `research/*.md`)
Ingested 2026-10-14. Foundational exploration and technical notes:
- [`a100_speed_and_session.md`](27b-notes/a100_speed_and_session.md): CMake fatbin targets, G4 bandwidth, reconnect loops, `/slots` watchdog, smoke gates.
- [`blueprint.md`](27b-notes/blueprint.md): Original orchestrator → Drive → Colab worker concept.
- [`busy_gpu.md`](27b-notes/busy_gpu.md): A100 40GB deployment notes.
- [`drive_files.md`](27b-notes/drive_files.md): Checkpoint and binary packing list for Google Drive.
- [`kernel_and_cli.md`](27b-notes/kernel_and_cli.md): Colab CLI 0.7.4, attached exec + heartbeat, Drive mount, `runtime.unassign()`.
- [`llamacpp_flags.md`](27b-notes/llamacpp_flags.md): Deep audit of llama-server CLI flags and traps.
- [`qwen38_27b_max_tokens_per_second_deep_research.md`](27b-notes/qwen38_27b_max_tokens_per_second_deep_research.md): DFlash 2 and speculative throughput analysis.
- [`sidecar_and_verify.md`](27b-notes/sidecar_and_verify.md): DFlash 2 sidecar quantization, tools PEG parser, first-boot grep gates.

### 2. 27B Plan Archive (`research/27b-plan/`)
Historical planning documents for the 27B worker superseded by current shipping runbooks:
- [`architecture.md`](27b-plan/architecture.md): Initial 27B worker architecture and design.
- [`inference.md`](27b-plan/inference.md): Original inference parameters and launch flags.
- [`first-boot.md`](27b-plan/first-boot.md): Initial first-sitting checklists and verification gates.
- [`plan_audit_and_fixes.md`](27b-plan/plan_audit_and_fixes.md): Technical audit identifying compile flags, Curl 2 fixes, and session handling.
- [`bend-hands.md`](27b-plan/bend-hands.md): Shelved Grok → Gemini → Hermes orchestrator plan.
- [`fallback-27b.md`](27b-plan/fallback-27b.md): Historical step-by-step launch procedure for uncensored 27B worker via `llama.cpp` on A100 (superseded by abliterated Flash-Next fallback).

> [!NOTE]
> Any leftover `file:///` paths within archived audit notes are historical artifacts referencing original authoring workspaces. Valid relative links point to corresponding archive locations.

### 3. Flash-Next Audit Dossier (`research/flash-next-audit/`)
Ten-paper research dossier auditing hardware, checkpoints, runtime stack, KV cache, economics, and claims for Qwen 3.8 Flash-Next:
- [`README.md`](flash-next-audit/README.md) & [`CLAIMS.md`](flash-next-audit/CLAIMS.md): Dossier index and claims ledger.
- Papers `01_colab_g4_hardware_and_cu.md` through `10_claims_ledger_and_verdict.md`: Detailed audits spanning Colab G4 hardware specifications, checkpoints, speed rooflines, runtime stacks, KV cache context headroom, Compute Unit economics, model quality, and the synthesized claims verdict.

### 4. Flash-Next Chat Trace (`research/flash-next-chat.md`)
- [`flash-next-chat.md`](flash-next-chat.md): Raw chat transcript exploring Qwen Flash-Next deployment options on Colab that seeded the audit dossier.

---

## Flattening & Legacy Path Map

The directory structure was flattened by removing the legacy `research/archive/` wrapper and reorganizing loose notes:
- Former `research/archive/README.md` merged into `research/README.md`.
- Former `research/archive/27b-plan/` moved to `research/27b-plan/`.
- Former `research/archive/flash-next-audit/` moved to `research/flash-next-audit/`.
- Former `research/archive/flash-next-chat.md` moved to `research/flash-next-chat.md`.
- Former loose notes (`research/a100_speed_and_session.md`, `research/blueprint.md`, `research/busy_gpu.md`, `research/drive_files.md`, `research/kernel_and_cli.md`, `research/llamacpp_flags.md`, `research/qwen38_27b_max_tokens_per_second_deep_research.md`, `research/sidecar_and_verify.md`) moved to `research/27b-notes/`.
