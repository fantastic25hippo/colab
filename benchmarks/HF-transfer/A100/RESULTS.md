## 2026-10-27 23:36 IST: HF-transfer benchmark on A100

**Model:** bartowski/Qwen2.5-7B-Instruct-GGUF (`Qwen2.5-7B-Instruct-Q4_K_M.gguf`, 4.68 GB) into /dev/shm; bartowski/Qwen2.5-14B-Instruct-GGUF (`Qwen2.5-14B-Instruct-Q8_0.gguf`, 15.70 GB) into disk (MEASURED)
**Weights:** HF transfer download via `hf_transfer`
**Hardware:** NVIDIA A100-SXM4-40GB 40960 MiB, CPU cores 12 (MEASURED); /dev/shm 41G, /tmp and /content overlay 236G (194G avail), free 83G (MEASURED)
**CLI:** `colab` 0.7.5, session `hf-a100`, command `colab --auth=oauth2 new -s hf-a100 --gpu A100`
**Engine:** hf_transfer (Python huggingface_hub + hf_transfer)
**Flags:** `HF_HUB_ENABLE_HF_TRANSFER=1`

### Results

| Test | tok/s | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| pp512 (prompt) | n/a | n/a | 0 | MEASURED |
| tg128 (generation) | n/a | n/a | 0 | MEASURED |

Tok/s is n/a: transfer bench measuring Hugging Face download speed and disk/shm throughput.
- 4 GB HF download to /dev/shm: 4,683,074,240 bytes in 17.93 s = 249.12 MB/s (MEASURED).
- 15 GB HF download to disk: 15,701,598,336 bytes in 42.42 s = 352.98 MB/s (MEASURED).
- Disk vs SHM (4 GB dd conv=fsync):
  - `/dev/shm`: 1.66 s = 2.6 GB/s (MEASURED)
  - `/tmp`: 11.83 s = 363 MB/s (MEASURED)
- 75 GB model cannot fit in /dev/shm (41G capacity).

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new`) | 4.58 | MEASURED |
| 4 GB HF download (/dev/shm) | 17.93 | MEASURED |
| 15 GB HF download (disk) | 42.42 | MEASURED |

### Cost

- CU rate: 5.30/hr from `colab usage` (MEASURED). Balance before 398.74, after 398.67, CU used 0.07 (MEASURED).
- CU per 100K tokens = n/a.

### Gotchas

- /dev/shm is 41G on A100 (40 GB SXM4 standard); 75 GB model cannot fit in /dev/shm.
- HF download to disk reached 352.98 MB/s (15.70 GB downloaded in 42.42 s).
- Disk dd write speed was 363 MB/s (identical to L4, both capped by overlay disk throttle).

**Runtime stopped:** yes, `sessions` empty at 23:38 IST. Spent 0.07 CU.
