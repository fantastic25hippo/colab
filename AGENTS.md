# AGENTS.md

Map for AI agents working in this repo (public; no keys, tokens or account emails, ever).

- Start with `README.md` (current focus and index), then `ops.md` and `hardware.md`.
- Benchmarking a model on Colab: read `benchmarks/README.md` first and follow its runbook. Write results to `benchmarks/<Model>/<GPU>/RESULTS.md` and add a row to `benchmarks/INDEX.md`.
- Research under `research/` is estimates, not measurements. Only `MEASURED` numbers in `benchmarks/` come from real runs.
- Memory: OptMem, see `docs/OPTMEM.md` and `OPTMEM-SEED.md`. Run `grok-agy-memo wake` first in every session.
- Work on `main`. Keep every new file under 150 lines. Spend no CU without Sahil's go; always stop the runtime afterwards.
