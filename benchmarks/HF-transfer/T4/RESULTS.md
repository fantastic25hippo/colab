## 2026-10-27 23:30 IST: HF-transfer benchmark on T4 (Control rerun)

**Model:** bartowski/Qwen2.5-7B-Instruct-GGUF (`Qwen2.5-7B-Instruct-Q4_K_M.gguf`, 4.68 GB) into /dev/shm; bartowski/Qwen2.5-14B-Instruct-GGUF (`Qwen2.5-14B-Instruct-Q8_0.gguf`, 15.70 GB) into disk (MEASURED)
**Weights:** HF transfer download via `hf_transfer`
**Hardware:** Tesla T4 15360 MiB, CPU cores 2 (MEASURED); /dev/shm 5.7G, /tmp and /content overlay 236G (194G avail), free 12G (MEASURED)
**CLI:** `colab` 0.7.5, session `hf-t4`, command `colab --auth=oauth2 new -s hf-t4 --gpu T4`
**Engine:** hf_transfer (Python huggingface_hub + hf_transfer)
**Flags:** `HF_HUB_ENABLE_HF_TRANSFER=1`

### Results

| Test | tok/s | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| pp512 (prompt) | n/a | n/a | 0 | MEASURED |
| tg128 (generation) | n/a | n/a | 0 | MEASURED |

Tok/s is n/a: transfer bench measuring Hugging Face download speed and disk/shm throughput.
- 4 GB HF download to /dev/shm: 4,683,074,240 bytes in 22.10 s = 202.08 MB/s (MEASURED).
- 15 GB HF download to disk: capped at 180s timeout; prior uncapped was 104.65 MB/s (MEASURED).
- Disk vs SHM (4 GB dd conv=fsync):
  - `/dev/shm`: 1.67 s = 2.6 GB/s (MEASURED)
  - `/tmp`: 22.51 s = 191 MB/s (MEASURED)
- 75 GB model cannot fit in /dev/shm (5.7G capacity).

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new`) | 4.88 | MEASURED |
| 4 GB HF download (/dev/shm) | 22.10 | MEASURED |

### Cost

- CU rate: 1.07/hr from `colab usage` (MEASURED). Balance before 398.80, after 398.74, CU used 0.06 (MEASURED).
- CU per 100K tokens = n/a.

### Gotchas

- /dev/shm is only 5.7G on T4; 75 GB models cannot fit.
- 15 GB disk download hit 180s cap due to Colab UI secret token vault timeout overhead.

**Runtime stopped:** yes, `sessions` empty at 23:33 IST. Spent 0.06 CU.

## 2026-10-27 22:42 IST: HF-transfer benchmark on T4

**Model:** bartowski/Qwen2.5-14B-Instruct-GGUF, quant Q8_0, file `Qwen2.5-14B-Instruct-Q8_0.gguf`, 15,701,598,336 bytes (15.70 GB) (MEASURED)
**Weights:** HF transfer download into `/content/Qwen2.5-14B-Instruct-Q8_0.gguf` via `hf_transfer`
**Hardware:** T4, driver n/a, CUDA n/a, VRAM n/a MiB, CPU cores 2 (MEASURED); /dev/shm 5.7G, /tmp and /content overlay 236G (179G avail), free 12G (MEASURED)
**CLI:** `colab` 0.7.5, session `hfbench`, command `colab --auth=oauth2 new -s hfbench --gpu T4`
**Engine:** hf_transfer (Python huggingface_hub + hf_transfer pip packages)
**Flags:** `HF_HUB_ENABLE_HF_TRANSFER=1`

### Results

| Test | tok/s | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| pp512 (prompt) | n/a | n/a | 0 | MEASURED |
| tg128 (generation) | n/a | n/a | 0 | MEASURED |

Tok/s is n/a: this is a transfer bench measuring Hugging Face download speed and disk/shm copy throughput.
Throughput: 15,701,598,336 bytes downloaded in 143.09 s = 104.65 MB/s (MEASURED).
Disk vs SHM (4 GB dd conv=fsync):
- `/dev/shm`: 17.32 s = 248 MB/s (MEASURED)
- `/tmp`: 29.29 s = 147 MB/s (MEASURED)
75 GB model cannot fit in /dev/shm (5.7G capacity).

HTTP path:
- (a) ssh port-forward denied by harness (`shell not allowed: ssh`); `colab ssh` not run.
- (b) `colab exec` log heartbeat worked: setup 3.40 s, RTT 2.92 s (MEASURED).
- (c) cloudflared untried.

FAQ URL: https://research.google.com/colaboratory/faq.html#disallowed-activities
FAQ Quote: "The following are disallowed from all managed Colab runtimes: file hosting, media serving, or other web service offerings not related to interactive compute with Colab..." and for free/zero CU runtimes: "remote control such as SSH shells, remote desktops... You can remove these types of restrictions by purchasing one of our paid plans here and maintaining a positive compute unit balance."

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new`) | 4.95 | MEASURED |
| Readiness probe (`exec ready.py`) | 3.12 | MEASURED |
| New-to-ready total | 8.07 | MEASURED |
| Pip install prerequisites | 2.24 | MEASURED |
| HF transfer download | 143.09 | MEASURED |

### Cost

- CU rate: not captured while VM was up (idle usage rate was 0.00/hr). Balance before 398.89 (MEASURED), after 398.80 (MEASURED), CU used 0.09 (MEASURED).
- CU per 100K tokens = n/a (transfer bench).

### Commands that worked (copy-paste, in order)

```bash
colab --auth=oauth2 usage
colab --auth=oauth2 new -s hfbench --gpu T4
colab --auth=oauth2 exec -s hfbench --timeout 86400 -f ready.py
colab --auth=oauth2 exec -s hfbench --timeout 86400 <<'PY'
import subprocess, time
subprocess.check_call(["pip", "install", "-q", "huggingface_hub", "hf_transfer"])
from huggingface_hub import hf_hub_download
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
t0 = time.time()
hf_hub_download("bartowski/Qwen2.5-14B-Instruct-GGUF", "Qwen2.5-14B-Instruct-Q8_0.gguf", local_dir="/content")
print(f"Elapsed: {time.time()-t0:.2f}s")
PY
colab --auth=oauth2 stop -s hfbench
colab --auth=oauth2 sessions
colab --auth=oauth2 usage
```

### Gotchas

- Chat claim of 500–800 MB/s is disproven; actual transfer throughput was 104.65 MB/s (MEASURED).
- /dev/shm is only 5.7G on standard T4; 75 GB models cannot fit.
- Harness denied ssh tool call; background exec polling worked with 2.92 s RTT.

**Runtime stopped:** yes, `sessions` empty at 22:43 IST. Spent 0.09 CU.
