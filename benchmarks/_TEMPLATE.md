# Entry template

Copy the block below to the TOP of `benchmarks/<Model>/<GPU>/RESULTS.md` (newest first) and fill every field.
Label each number `MEASURED` or `ESTIMATE`. Times are IST. Never write a key, token or account email.

````markdown
## YYYY-MM-DD HH:MM IST: <Model> <quant> on <GPU>

**Model:** <name>, quant <Q8_0 / NVFP4 / ...>, file `<name>.gguf`, <size in GB> (MEASURED)
**Weights:** Drive `/content/drive/MyDrive/models/<name>.gguf/<name>.gguf`, copied to `/tmp/<name>.gguf`. Not downloaded.
**Hardware:** <T4|L4|A100-40|A100-80|G4|TPU-v5e|TPU-v6e>, driver <x>, CUDA <x>, VRAM <x> MiB, CPU cores <n>
**CLI:** `colab` <version>, session `<name>`, command `colab --auth=oauth2 new -s <name> --gpu <GPU>`
**Engine:** <llama.cpp / other>, exact version or build <bN or commit>, install method <prebuilt / wheel / source>
**Flags:** `<exact llama-bench and server flags>`

### Results

| Test | tok/s | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| pp512 (prompt) | | | 3 | MEASURED |
| tg128 (generation) | | | 3 | MEASURED |

Real generation: <~1K-token prompt, ~350 tokens out>, <tok/s>, coherent: <yes/no>. Sample (short):
> <quote a few lines>

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new`) | | MEASURED |
| Drive mount (human approval included) | | MEASURED |
| Drive copy to /tmp | | MEASURED |
| Engine install | | MEASURED |
| Model load | | MEASURED |

### Cost

- CU rate: <x>/hr from `colab usage` (MEASURED). Balance before <x>, after <x>, CU used <x> (MEASURED).
- CU per 100K tokens = (100000 / tg_tok_s) / 3600 x CU_per_hr = <x> (derived from tg128)

### Commands that worked (copy-paste, in order)

```
<exact commands, including exec file contents or one-liners>
```

### Gotchas

- <anything that failed, was slow, or needs a human step>

**Runtime stopped:** yes, `sessions` empty at <time> IST.
````

Index row, appended to `benchmarks/INDEX.md`:
`| YYYY-MM-DD | <Model> | <quant> | <GPU> | <engine+build> | <pp512> | <tg128> | <CU/hr> | <CU used> | [entry](<Model>/<GPU>/RESULTS.md) |`
