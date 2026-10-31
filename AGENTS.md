# AGENTS.md

Map for AI agents working in this repo (public; no keys, tokens or account emails, ever).

- Start with `README.md` (current focus and index), then `ops.md` and `hardware.md`.
- Benchmarking a model on Colab: read `benchmarks/README.md` first and follow its runbook (including the **Flash Subagent + Live SSH Monitoring Protocol**). Write results to `benchmarks/<Model>/<GPU>/RESULTS.md` and add a row to `benchmarks/INDEX.md`.
- Fast TPU v5e-1 Cold-Start Harness (`73.08 s` in-VM / `81.22 s` global): read `benchmarks/TPU-v5e/FAST-COLDSTART.md` and reuse the 3-script reference pipeline (`benchmarks/TPU-v5e/run_coldstart.py`, `tpu_coldstart_driver.py`, `tpu_engine_subrunner.py`).
- Research under `research/` is estimates, not measurements. Only `MEASURED` numbers in `benchmarks/` come from real runs.
- Memory: OptMem, see `docs/OPTMEM.md` and `OPTMEM-SEED.md`. Run `grok-agy-memo wake` first in every session.
- Work on `main`. Keep every new file under 150 lines. Spend no CU without Sahil's go; always stop the runtime afterwards.
