# Evidence Archive

This directory is the evidence archive containing historical planning documents, deep research audits, and exploratory chat transcripts:
- [`27b-plan/`](27b-plan/)
- [`flash-next-audit/`](flash-next-audit/)
- [`flash-next-chat.md`](flash-next-chat.md)

For the full archive index covering foundational research notes as well as these archived materials, link up to [`../README.md`](../README.md).

---

> [!IMPORTANT]
> **Precedence Rule:** Shipping docs at repo root ([`README.md`](../../README.md), [`decision.md`](../../decision.md), [`claims-ledger.md`](../../claims-ledger.md), [`ops.md`](../../ops.md), and [`runbooks/`](../../runbooks/)) are the source of truth and win on conflict. Do not use archived notes or plans as operational launch instructions.

---

## Archive Map

### 1. 27B Plan Archive (`27b-plan/`)
Historical planning documents for the 27B worker, superseded by current shipping runbooks:
- [`architecture.md`](27b-plan/architecture.md): Initial 27B worker architecture and design.
- [`inference.md`](27b-plan/inference.md): Original inference parameters and launch flags.
- [`first-boot.md`](27b-plan/first-boot.md): Initial first-sitting checklists and verification gates.
- [`plan_audit_and_fixes.md`](27b-plan/plan_audit_and_fixes.md): Technical audit identifying compile flags, Curl 2 fixes, and session handling.
- [`bend-hands.md`](27b-plan/bend-hands.md): Shelved Grok → Gemini → Hermes orchestrator plan.
- [`fallback-27b.md`](27b-plan/fallback-27b.md): Historical step-by-step launch procedure for uncensored 27B worker via `llama.cpp` on A100 (superseded by abliterated Flash-Next fallback).

> [!NOTE]
> Any leftover `file:///` paths within archived audit notes are historical artifacts referencing original authoring workspaces. Valid relative links point to corresponding archive locations.

### 2. Flash-Next Audit Dossier (`flash-next-audit/`)
Ten-paper research dossier auditing hardware, checkpoints, runtime stack, KV cache, economics, and claims for Qwen 3.8 Flash-Next:
- [`README.md`](flash-next-audit/README.md) & [`CLAIMS.md`](flash-next-audit/CLAIMS.md): Dossier index and claims ledger.
- Papers `01_colab_g4_hardware_and_cu.md` through `10_claims_ledger_and_verdict.md`: Detailed audits spanning Colab G4 hardware specifications, checkpoints, speed rooflines, runtime stacks, KV cache context headroom, Compute Unit economics, model quality, and the synthesized claims verdict.

### 3. Flash-Next Chat Trace (`flash-next-chat.md`)
- [`flash-next-chat.md`](flash-next-chat.md): Raw chat transcript exploring Qwen Flash-Next deployment options on Colab that seeded the audit dossier.
