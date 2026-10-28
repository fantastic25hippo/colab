# Benchmark index

One row per entry, newest first. Format and rules: [`README.md`](README.md). Numbers are tok/s, MEASURED only.

| Date (IST) | Model | Quant | GPU | Engine + build | pp512 | tg128 | CU/hr | CU used | Entry |
| :--- | :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 2026-10-28 | LFM2.5-24B-A2B | Q4 (sharded) | L4 | hf_transfer (llama-server did not load) | n/a | n/a | 1.54 | 0.32 | [entry](LFM2.5-24B-A2B-Q4/L4/RESULTS.md) |
| 2026-10-28 | HF-transfer | sharded (LFM-24B-Q4) | CPU | not measured | n/a | n/a | 0.00 | 0.00 | [entry](HF-transfer/CPU/RESULTS.md) |
| 2026-10-28 | HF-transfer | sharded (Qwen2.5-7B) | L4 | hf_transfer | n/a | n/a | 1.54 | 0.08 | [entry](HF-transfer/L4/RESULTS.md) |
| 2026-10-28 | HF-transfer | n/a | TPU-v6e | skipped (timeout) | n/a | n/a | n/a | 0.00 | [entry](HF-transfer/TPU-v6e/RESULTS.md) |
| 2026-10-28 | HF-transfer | n/a | TPU-v5e | n/a | n/a | n/a | 2.92 | 0.23 | [entry](HF-transfer/TPU-v5e/RESULTS.md) |
| 2026-10-27 | HF-transfer | n/a | T4 | n/a | n/a | n/a | 1.07 | 0.00 | [entry](HF-transfer/T4/RESULTS.md) |
