## 2026-10-27 21:47 IST: HF-transfer benchmark on T4

**Model:** HF-transfer speed test, quant n/a, file n/a, n/a GB (ESTIMATE)
**Weights:** HF transfer test to local disk/shm, not downloaded.
**Hardware:** T4, driver n/a, CUDA n/a, VRAM n/a MiB, CPU cores n/a
**CLI:** `colab` 0.7.5, session `bench`, command `colab --auth=oauth2 new -s bench --gpu T4` (denied)
**Engine:** n/a, exact version or build n/a, install method n/a
**Flags:** `n/a`

### Results

| Test | tok/s | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| pp512 (prompt) | n/a | n/a | 0 | ESTIMATE |
| tg128 (generation) | n/a | n/a | 0 | ESTIMATE |

Real generation: n/a. No bench ran because tool call was denied by harness pre-tool hook (`tool call denied by pre-tool hook: shell not allowed: colab; next: edit_file`). Session never started.
> n/a

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new`) | n/a | ESTIMATE |
| Drive mount (human approval included) | n/a | ESTIMATE |
| Drive copy to /tmp | n/a | ESTIMATE |
| Engine install | n/a | ESTIMATE |
| Model load | n/a | ESTIMATE |

### Cost

- CU rate: 1.07/hr (MEASURED from previous T4 run). Balance before 398.89 (MEASURED), after 398.89 (MEASURED), CU used 0.00 (MEASURED).
- CU per 100K tokens = n/a (session never started).

### Commands that worked (copy-paste, in order)

```
# Pre-tool hook denied running colab command:
# tool call denied by pre-tool hook: shell not allowed: colab; next: edit_file
# CU spent: 0.00. No commands executed on Colab.
```

### Gotchas

- Colab execution denied by environment harness hook (`shell not allowed: colab`).
- Session never started; 0 CU spent.
- HF hf_transfer speed claims (500–800 MB/s) remain unmeasured.

**Runtime stopped:** yes, `sessions` empty at 21:47 IST (session never started).
