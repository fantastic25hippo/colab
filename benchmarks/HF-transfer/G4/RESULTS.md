## 2026-10-27 23:38 IST: HF-transfer benchmark on G4

**Model:** bartowski/Qwen2.5-7B-Instruct-GGUF (`Qwen2.5-7B-Instruct-Q4_K_M.gguf`, 4.68 GB) into /dev/shm; bartowski/Qwen2.5-14B-Instruct-GGUF (`Qwen2.5-14B-Instruct-Q8_0.gguf`, 15.70 GB) into disk (MEASURED)
**Weights:** HF transfer download via `hf_transfer`
**Hardware:** NVIDIA RTX PRO 6000 Blackwell Server Edition 97887 MiB, CPU cores 48 (MEASURED); /dev/shm 87G, /tmp and /content overlay 236G (194G avail), free 176G (MEASURED)
**CLI:** `colab` 0.7.5, session `hf-g4`, command `colab --auth=oauth2 new -s hf-g4 --gpu G4`
**Engine:** hf_transfer (Python huggingface_hub + hf_transfer)
**Flags:** `HF_HUB_ENABLE_HF_TRANSFER=1`

### Results

| Test | tok/s | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| pp512 (prompt) | n/a | n/a | 0 | MEASURED |
| tg128 (generation) | n/a | n/a | 0 | MEASURED |

Tok/s is n/a: transfer bench measuring Hugging Face download speed and disk/shm throughput.
- 4 GB HF download to /dev/shm: 4,683,074,240 bytes in 16.52 s = 270.36 MB/s (MEASURED).
- 15 GB HF download to disk: 15,701,598,336 bytes in 52.33 s = 286.16 MB/s (MEASURED).
- Disk vs SHM (4 GB dd conv=fsync):
  - `/dev/shm`: 0.52 s = 8.3 GB/s (MEASURED)
  - `/tmp`: 13.48 s = 319 MB/s (MEASURED)
- 75 GB model CAN fit in /dev/shm (87G capacity MEASURED).

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new`) | 4.48 | MEASURED |
| 4 GB HF download (/dev/shm) | 16.52 | MEASURED |
| 15 GB HF download (disk) | 52.33 | MEASURED |

### Cost

- CU rate: 8.90/hr from `colab usage` (MEASURED). Balance before 398.67, after 398.67, CU used 0.00 (MEASURED).
- CU per 100K tokens = n/a.

### Gotchas

- /dev/shm is 87G on G4 (48 vCPU / 176G host RAM shape), allowing a 75 GB model to fit directly in RAM (/dev/shm).
- /dev/shm dd write achieved 8.3 GB/s (0.52 s for 4 GB).
- HF download capped around 270–286 MB/s across both shm and disk targets, indicating network throughput bottleneck rather than disk I/O.

**Runtime stopped:** yes, `sessions` empty at 23:40 IST. Spent 0.00 CU.
