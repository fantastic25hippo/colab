# Research archive

Ingested 2026-10-14. Public research is complete. Shipping docs at the folder root are the live plan. These files are the evidence trail; do not treat them as launch instructions.

| File | Settled |
| :--- | :--- |
| [`blueprint.md`](blueprint.md) | Original orchestrator → Drive → Colab worker plan (Oct 14 18:20). L4-first and fire-and-exit exec are superseded. |
| [`qwen38_27b_max_tokens_per_second_deep_research.md`](qwen38_27b_max_tokens_per_second_deep_research.md) | DFlash 2 + k4v speed story. Launch block and PR #27342 checkout are superseded by [`llamacpp_flags.md`](llamacpp_flags.md). |
| [`kernel_and_cli.md`](kernel_and_cli.md) | CLI 0.7.4, attached exec + heartbeat, Drive mount, trycloudflare, `runtime.unassign()`. |
| [`llamacpp_flags.md`](llamacpp_flags.md) | Playbook A flags on `efe3626` / `b11552`. Trust §5.3 for the launch. |
| [`busy_gpu.md`](busy_gpu.md) | A100 40GB first sitting. §4b sample launch is stale. |
| [`drive_files.md`](drive_files.md) | Blackfrost packing list. Draft-hook indexing is corrected in [`sidecar_and_verify.md`](sidecar_and_verify.md). |
| [`sidecar_and_verify.md`](sidecar_and_verify.md) | Quantize Blackfrost BF16, skip rope injection, tools PEG, first-boot grep. |
| [`a100_speed_and_session.md`](a100_speed_and_session.md) | CMake fatbin, G4 1,597 GB/s, reconnect, `/slots` watchdog, smoke gates. |

When a root doc and a paper disagree, the root doc wins.
