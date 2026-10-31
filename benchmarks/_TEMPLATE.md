# Entry template

Copy the block below to the TOP of `benchmarks/<Model>/<GPU>/RESULTS.md` (or `benchmarks/TPU-v5e/<run>/RESULTS.md`, newest first) and fill every field.
Label each number `MEASURED` or `ESTIMATE`. Times are IST. Never write a key, token, or account email.

````markdown
## YYYY-MM-DD HH:MM IST: <Model> <quant> on <GPU/TPU>

**Model:** `<repo_id>`, quant `<bf16 / Q4_K_XL / W8A8 / ...>`, size `<GiB / GB>` (MEASURED)
**Weights:** Hugging Face `<repo_id>` pulled into `/dev/shm/<dir>` (`<n>` shards, `max_workers=<w>`).
**Hardware:** `<T4|L4|A100-40|A100-80|G4|TPU-v5e|TPU-v6e>`, VRAM/HBM `<x>` GiB, host RAM `<x>` GB, `/dev/shm` `<x>` GB, CPU cores `<n>` (MEASURED)
**CLI:** `colab` 0.7.5, session `<name>`, command `colab --auth=oauth2 new -s <name> --gpu <GPU>` (or `--tpu v5e1`)
**Engine:** `<vllm-tpu 0.32.0 / llama.cpp b11954 cuda-13.4>`, install method `<prebuilt /dev/shm venv / prebuilt binary>`
**Flags:** `<exact engine flags and environment variables>`

### Results

| Test | tok/s (or TTFT ms) | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| pp512 / TTFT | | | 3 | MEASURED |
| tg128 / tg256 (decode) | | | 3 | MEASURED |

Sample generation (`<prompt tokens>` in, `<output tokens>` out, coherent: `<yes/no>`):
> <quote a short sample>

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new`) | | MEASURED |
| Venv / binary + weights pull to `/dev/shm` (parallel) | | MEASURED |
| Engine import & init | | MEASURED |
| Weight load + KV cache alloc | | MEASURED |
| Compile / warmup to first token | | MEASURED |
| **Total VM start to first token** | | **MEASURED** |

### Cost

- CU rate: `<x>`/hr from `colab usage` (MEASURED). Balance before `<x>`, after `<x>`, CU used `<x>` (MEASURED).
- CU per 100K tokens = `(100000 / tg_tok_s) / 3600 * CU_per_hr` = `<x>` (derived from decode tok/s)

### Gotchas

- <anything that failed, was slow, or required a workaround>

**Runtime stopped:** yes, `sessions` empty at `<time>` IST.
````

Index row, appended to `benchmarks/INDEX.md`:
`| YYYY-MM-DD | <Model> | <quant> | <GPU> | <engine+build> | <pp512 / TTFT> | <tg128 / decode> | <CU/hr> | <CU used> | [entry](<path>/RESULTS.md) |`
