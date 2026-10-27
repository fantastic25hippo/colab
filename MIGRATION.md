# Documentation Migration & Deduplication Map

This document tracks the repository documentation reorganization from base commit `56943ac` on branch `council-docs-reorg`. It maps every removed, merged, split, or renamed file (old path -> new path(s) + section), quotes every old path verbatim, and details each deduplication.

---

## 1. File Renames & Moves

| Old Path (Verbatim) | New Path | Section / Target | Notes |
| :--- | :--- | :--- | :--- |
| `Google Colab AI Pro Runtime Specifications & Comparison.md` | `hardware.md` | Full file verbatim (Sections 1–5, Tables 3.1 & 3.2) | Renamed to concise, standard filename. Full body preserved verbatim. Added to `README.md` Document Map and cross-linked from `decision.md` §3 and `ops.md`. |
| `research/archive/README.md` | `research/README.md` | §Archive Index & §Flattening & Legacy Path Map | Deleted and merged into parent research index. Retains keys `fallback-27b.md`, `file:///`, `../README.md`, and legacy path strings. |
| `research/archive/flash-next-chat.md` | `research/flash-next-chat.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/27b-plan/architecture.md` | `research/27b-plan/architecture.md` | Full file (relative links updated) | Flattened out of `research/archive/` wrapper; updated `../../../ops.md` to `../../ops.md`. |
| `research/archive/27b-plan/bend-hands.md` | `research/27b-plan/bend-hands.md` | Full file (relative links updated) | Flattened out of `research/archive/` wrapper; updated `../../../ops.md` to `../../ops.md`. |
| `research/archive/27b-plan/fallback-27b.md` | `research/27b-plan/fallback-27b.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/27b-plan/first-boot.md` | `research/27b-plan/first-boot.md` | Full file (relative links updated) | Flattened out of `research/archive/` wrapper; updated `../../../ops.md` to `../../ops.md`. |
| `research/archive/27b-plan/inference.md` | `research/27b-plan/inference.md` | Full file (relative links updated) | Flattened out of `research/archive/` wrapper; updated `../../../ops.md` to `../../ops.md`. |
| `research/archive/27b-plan/plan_audit_and_fixes.md` | `research/27b-plan/plan_audit_and_fixes.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/01_colab_g4_hardware_and_cu.md` | `research/flash-next-audit/01_colab_g4_hardware_and_cu.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/02_qwen38_flash_next_checkpoint.md` | `research/flash-next-audit/02_qwen38_flash_next_checkpoint.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/03_qwen38_27b_nvfp4_checkpoint.md` | `research/flash-next-audit/03_qwen38_27b_nvfp4_checkpoint.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/04_glm53_flash_nvfp4.md` | `research/flash-next-audit/04_glm53_flash_nvfp4.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/05_speed_roofline_and_mtp.md` | `research/flash-next-audit/05_speed_roofline_and_mtp.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/06_runtime_stack.md` | `research/flash-next-audit/06_runtime_stack.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/07_kv_cache_and_context.md` | `research/flash-next-audit/07_kv_cache_and_context.md` | §Conflicts with architecture.md / prior research/ | Flattened out of `research/archive/` wrapper; updated link to `../27b-notes/qwen38_27b_max_tokens_per_second_deep_research.md#L94`. |
| `research/archive/flash-next-audit/08_colab_cu_economics.md` | `research/flash-next-audit/08_colab_cu_economics.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/09_quality_abliteration_agent_fit.md` | `research/flash-next-audit/09_quality_abliteration_agent_fit.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/10_claims_ledger_and_verdict.md` | `research/flash-next-audit/10_claims_ledger_and_verdict.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/CLAIMS.md` | `research/flash-next-audit/CLAIMS.md` | Full file verbatim | Flattened out of `research/archive/` wrapper. |
| `research/archive/flash-next-audit/README.md` | `research/flash-next-audit/README.md` | §Source & Overview | Flattened out of `research/archive/` wrapper; updated historical links `../../../README.md` to `../../README.md` and `../../README.md` to `../README.md`. |
| `research/a100_speed_and_session.md` | `research/27b-notes/a100_speed_and_session.md` | Full file verbatim | Moved loose 27B note into dedicated `research/27b-notes/` subdirectory. |
| `research/blueprint.md` | `research/27b-notes/blueprint.md` | Full file verbatim | Moved loose 27B note into dedicated `research/27b-notes/` subdirectory. |
| `research/busy_gpu.md` | `research/27b-notes/busy_gpu.md` | Full file verbatim | Moved loose 27B note into dedicated `research/27b-notes/` subdirectory. |
| `research/drive_files.md` | `research/27b-notes/drive_files.md` | Full file verbatim | Moved loose 27B note into dedicated `research/27b-notes/` subdirectory. |
| `research/kernel_and_cli.md` | `research/27b-notes/kernel_and_cli.md` | Full file verbatim | Moved loose 27B note into dedicated `research/27b-notes/` subdirectory. |
| `research/llamacpp_flags.md` | `research/27b-notes/llamacpp_flags.md` | Full file verbatim | Moved loose 27B note into dedicated `research/27b-notes/` subdirectory. |
| `research/qwen38_27b_max_tokens_per_second_deep_research.md` | `research/27b-notes/qwen38_27b_max_tokens_per_second_deep_research.md` | Full file verbatim | Moved loose 27B note into dedicated `research/27b-notes/` subdirectory. |
| `research/sidecar_and_verify.md` | `research/27b-notes/sidecar_and_verify.md` | Full file verbatim | Moved loose 27B note into dedicated `research/27b-notes/` subdirectory. |

---

## 2. Deduplications & Canonical Consolidations

### Deduplication 1: First-Principles Economic Recalculation
- **Canonical Location:** [`claims-ledger.md §3`](claims-ledger.md#3-first-principles-economic-recalculation) ("First-Principles Economic Recalculation").
- **Deduplicated File:** `decision.md §4` ("First-Principles Economic Recalculation").
- **Details:** Consolidated all unique economic keys into `claims-ledger.md §3`, including:
  - `8–18 minute cold boot`, `1.16–2.61 CU`, `560 to 943` tasks per 200 CU pool.
  - Kept existing 6-row table, `1,377.7 minutes`, `5.37 CU/hr`, `187.13 tok/s`, and historical 27B rows.
  - Replaced the duplicate section in `decision.md §4` with a concise operational recap pointing to `claims-ledger.md §3`.

### Deduplication 2: Five Open UNVERIFIED Probes
- **Canonical Location:** [`claims-ledger.md §5`](claims-ledger.md#5-unverified-remaining) ("UNVERIFIED Remaining").
- **Deduplicated Files:**
  - `decision.md §5` ("Open Empirical Probes (UNVERIFIED Status)")
  - `runbooks/flash-next-sglang.md §6` ("Active UNVERIFIED Probes to Track")
- **Details:** Consolidated all unique probe keys into `claims-ledger.md §5`, including:
  - `https://note.com/gakushiai/n/n71ca7ca09d1b`
  - `colab new -s qwen-worker --gpu G4` and `colab new --gpu G4`
  - `132.68 GB` and `132.7 GB`
  - `Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF at 112 GB` (`https://huggingface.co/Alexsi98/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF`)
  - `https://huggingface.co/claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4` at `105.48 GB`
  - `Docker daemon`, `>= 64 GB`, `--ulimit memlock=-1`, `51.2B`, `sm_120f`
  - Replaced duplicate probe lists in `decision.md §5` and `runbooks/flash-next-sglang.md §6` with concise cross-links to `claims-ledger.md §5`.

### Deduplication 3: Evidence Archive Index Consolidation
- **Canonical Location:** [`research/README.md`](research/README.md) ("Research Archive").
- **Deduplicated File:** `research/archive/README.md` ("Evidence Archive").
- **Details:** Merged the redundant `research/archive/README.md` file into `research/README.md`, preserving all unique keys (`fallback-27b.md`, `file:///`, `../README.md`, and all legacy `research/archive/` path strings). The empty directory wrapper `research/archive/` was completely removed.
- **Link updates:** Updated former 3-level relative links (`../../../ops.md`, `../../../README.md`) to 2-level links (`../../ops.md`, `../../README.md`).
