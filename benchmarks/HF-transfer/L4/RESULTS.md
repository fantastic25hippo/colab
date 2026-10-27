## 2026-10-27 23:34 IST: HF-transfer benchmark on L4

**Model:** bartowski/Qwen2.5-7B-Instruct-GGUF (`Qwen2.5-7B-Instruct-Q4_K_M.gguf`, 4.68 GB) into /dev/shm; bartowski/Qwen2.5-14B-Instruct-GGUF (`Qwen2.5-14B-Instruct-Q8_0.gguf`, 15.70 GB) into disk (MEASURED)
**Weights:** HF transfer download via `hf_transfer`
**Hardware:** NVIDIA L4 23034 MiB, CPU cores 12 (MEASURED); /dev/shm 26G, /tmp and /content overlay 236G (194G avail), free 52G (MEASURED)
**CLI:** `colab` 0.7.5, session `hf-l4`, command `colab --auth=oauth2 new -s hf-l4 --gpu L4`
**Engine:** hf_transfer (Python huggingface_hub + hf_transfer)
**Flags:** `HF_HUB_ENABLE_HF_TRANSFER=1`

### Results

| Test | tok/s | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| pp512 (prompt) | n/a | n/a | 0 | MEASURED |
| tg128 (generation) | n/a | n/a | 0 | MEASURED |

Tok/s is n/a: transfer bench measuring Hugging Face download speed and disk/shm throughput.
- 4 GB HF download to /dev/shm: 4,683,074,240 bytes in 18.05 s = 247.42 MB/s (MEASURED).
- 15 GB HF download to disk: 15,701,598,336 bytes in 59.35 s = 252.29 MB/s (MEASURED).
- Disk vs SHM (4 GB dd conv=fsync):
  - `/dev/shm`: 1.65 s = 2.6 GB/s (MEASURED)
  - `/tmp`: 11.85 s = 362 MB/s (MEASURED)
- 75 GB model cannot fit in /dev/shm (26G capacity).

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new`) | 4.64 | MEASURED |
| 4 GB HF download (/dev/shm) | 18.05 | MEASURED |
| 15 GB HF download (disk) | 59.35 | MEASURED |

### Cost

- CU rate: 1.54/hr from `colab usage` (MEASURED). Balance before 398.74, after 398.74, CU used 0.00 (MEASURED).
- CU per 100K tokens = n/a.

### Gotchas

- /dev/shm is 26G on L4 (significantly larger than T4's 5.7G, but 75 GB model still cannot fit).
- Disk dd write speed jumped from 191 MB/s on T4 to 362 MB/s on L4 (~1.89x).
- HF download to disk reached 252.29 MB/s (vs 104.65 MB/s on T4, ~2.4x speedup).

**Runtime stopped:** yes, `sessions` empty at 23:36 IST. Spent 0.00 CU.
