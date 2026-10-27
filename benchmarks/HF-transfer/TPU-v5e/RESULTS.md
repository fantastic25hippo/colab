## 2026-10-28 00:02 IST: HF-transfer benchmark on TPU-v5e

**Model:** bartowski/Qwen2.5-7B-Instruct-GGUF (`Qwen2.5-7B-Instruct-Q4_K_M.gguf`, 4.68 GB) into /dev/shm; bartowski/Qwen2.5-14B-Instruct-GGUF (`Qwen2.5-14B-Instruct-Q8_0.gguf`, 15.70 GB) into disk (MEASURED)
**Weights:** HF transfer download via `hf_transfer`
**Hardware:** Cloud TPU v5e-1 (ViperLite), 1 chip `TPU v5 lite`, HBM 16.0 GB (16909336064 bytes limit), CPU cores 24 (MEASURED); /dev/shm 23G, /tmp and /content overlay 226G (207G avail), system RAM 47G total (40G avail) (MEASURED)
**CLI:** `colab` 0.7.5, session `hf-v5e`, command `colab --auth=oauth2 new -s hf-v5e --tpu v5e1`
**Engine:** hf_transfer (Python huggingface_hub + hf_transfer)
**Flags:** `HF_HUB_ENABLE_HF_TRANSFER=1`
**Supported Engines / Frameworks:** JAX (preinstalled v0.11.2 with native TPU backend), PyTorch/XLA (torch 2.14.1+cpu preinstalled; torch_xla not preinstalled), vLLM TPU. Official doc: [Cloud TPU v5e (ViperLite)](https://docs.cloud.google.com/tpu/docs/v5e) cited in `hardware.md`. llama.cpp GGUF has no native TPU backend.

### Results

| Test | tok/s | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| pp512 (prompt) | n/a | n/a | 0 | MEASURED |
| tg128 (generation) | n/a | n/a | 0 | MEASURED |

Tok/s is n/a: transfer bench measuring Hugging Face download speed and disk/shm throughput.
- 4 GB HF download to /dev/shm: 4,683,074,240 bytes in 20.07 s = 222.58 MB/s (MEASURED).
- 15 GB HF download to disk: 15,701,598,336 bytes in 140.06 s = 106.91 MB/s (MEASURED).
- Disk vs SHM (4 GB dd conv=fsync):
  - `/dev/shm`: 0.77 s = 5.6 GB/s (4294967296 bytes) (MEASURED)
  - `/tmp`: 36.39 s = 118 MB/s (4294967296 bytes) (MEASURED)
- 75 GB model cannot fit in /dev/shm (23G capacity).

### Raw System Details (MEASURED)
- `colab new --help` TPU option text:
  `--tpu               <str>  TPU accelerator variant. Supported: v5e1, v6e1.`
  `Ignored for accelerators that only offer a single shape (L4, v5e1, v6e1).`
- `df -h /dev/shm /tmp /content`:
  shm 23G (12K used, 23G avail)
  overlay 226G (19G used, 207G avail)
- `free -g`: total 47, used 6, free 16, buff/cache 24, available 40
- `nproc`: 24
- TPU device count: 1 (`jax.devices()`: `[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]`)
- TPU chip type: `TPU v5 lite`, platform `tpu`
- TPU HBM: 16909336064 bytes (~16.0 GB)

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new`) | 4.61 | MEASURED |
| 4 GB HF download (/dev/shm) | 20.07 | MEASURED |
| 15 GB HF download (disk) | 140.06 | MEASURED |

### Cost

- CU rate: 2.92/hr from `colab usage` (MEASURED). Balance before 398.67, after 398.44, CU used 0.23 (MEASURED).
- CU per 100K tokens = n/a.

### Gotchas

- /dev/shm is 23G on TPU v5e (large, but cannot fit a 75 GB model).
- Disk dd write speed on /tmp is 118 MB/s (slower than L4's 362 MB/s).
- HF download to disk is throttled to 106.91 MB/s by disk write speed, while download to /dev/shm reaches 222.58 MB/s.
- `torch_xla` is not preinstalled in the default image; `jax` 0.11.2 is preinstalled with TPU support.

**Runtime stopped:** yes, `sessions` empty at 00:02 IST. Spent 0.23 CU.
