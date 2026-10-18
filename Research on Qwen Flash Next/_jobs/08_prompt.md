Read `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/_jobs/SHARED_PROTOCOL.md` and follow it exactly.

You are Job 08. Colab Compute Unit economics.

Write ONLY this file:
`/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/08_colab_cu_economics.md`

Own claims **E1–E18** in CLAIMS.md.

## What you must settle

Recompute every CU figure from first principles.

```
hours = wall_seconds / 3600
CU    = hours × CU_per_hour
```

Use **two** G4 CU/hr rates in every table, because H15 is contested:

- chat rate 6.55 CU/hr (mid of 6.4–6.7)
- architecture.md rate 8.71 CU/hr

Also show L4 at both 2.1 (chat) and 1.71 (architecture.md).

Recalculate:

1. E2: 60k **output** tokens at 140 vs 300 tok/s. The chat ignores prefill. Add a note.
2. E3: interactive hour = full hourly burn. True if billing is wall-clock.
3. E4: 25 turns × (prefill + decode). Chat uses 14–18 s vs 4–6 s. Rebuild from plausible prefill (4k–12k @ X tok/s) + 800 decode tokens @ Y tok/s. Show sensitivity.
4. E7: G4 vs L4 20-turn. Rebuild. The L4 row uses a **14B dense** model, not Flash-Next — that is an apples-to-oranges comparison. Call it out. Also compute Flash-Next-on-L4 **if it even fits** (probably not at 70 GB).
5. E11: 37M hit + 2M miss + 1M out. **Do not accept 100k–200k tok/s cache-hit bind uncritically.** Give:
   - chat’s own arithmetic reproduced
   - a corrected arithmetic if cache hits are nearly free (prefix skip) vs if they still scan
6. E15–E16: 200 CU / rate = hours. Then how many 40M-token blocks fit at 320/400/500/600 decode tok/s with the chat’s 10.1 min fixed prompt overhead. Reproduce the table, then a corrected table if prompt overhead or CU/hr changes.
7. E13: $15–$35 API vs <$1 Colab. Convert 7 CU to USD using Google AI Pro pricing (the 200 CU pool is bundled in the subscription — **marginal CU cost may be $0** until you buy extra). This is a common accounting trick. Handle it.
8. E18: confirm 200 CU/month Google AI Pro still works this way in Oct 2026.

Race-to-sleep is valid **only** for automated loops that `runtime.unassign()` when done. Idle human sessions do not save CUs with a faster model.

Do not re-derive GPU TFLOPS. You may assume Job 05 speed bands, but also show a “if decode is only 150 tok/s” sensitivity because the 320–380 Flash-Next code speed may be optimistic.
