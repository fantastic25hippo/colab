# google/gemma-4-E4B-it-qat-w4a16-ct on TPU v5e-1

Measurement of whether `google/gemma-4-E4B-it-qat-w4a16-ct` downloads, loads and runs on a Colab TPU v5e-1 with vLLM's TPU backend.

## Summary Table

| Stage | Status | Seconds | Details |
| :--- | :--- | ---: | :--- |
| env | Completed | n/a | Python `3.13.18`, RAM `47` GB, `/dev/shm` `23G`, TPU `["TPU_0(process=0,(0,0,0,0))"]` |
| install | Failed | 258.2362480163574 | `vllm-tpu` & `tpu-inference vllm` build failed (exit 1) |
| download | Completed | 20.182583808898926 | 11513861828 bytes (~11.51 GB), 570.4850249611397 MB/s |
| memory_before | Completed | n/a | `bytes_in_use`: 26624 B (0.0000247955 GB), `peak_bytes_in_use`: 26624 B (0.0000247955 GB) |
| load | Failed | not reached | `ModuleNotFoundError: No module named 'vllm'` |
| memory_after_load | Not reached | not reached | not reached |
| first_generation | Not reached | not reached | not reached |
| decode run 1 | Not reached | not reached | not reached |
| decode run 2 | Not reached | not reached | not reached |
| decode run 3 | Not reached | not reached | not reached |

## Metrics & Exact Values

- **Download Speed:** `570.4850249611397` MB/s (`20.182583808898926` seconds for `11513861828` bytes)
- **Memory Before Load:**
  - `bytes_in_use`: 26624 bytes (0.0000247955 GB)
  - `peak_bytes_in_use`: 26624 bytes (0.0000247955 GB)
- **Memory After Load:** not reached
- **Packed Ratio:** not reached
- **Packed / Unpacked Verdict:** not reached
- **Compile Overhead:** not reached
- **Decode Runs:**
  - Run 1: not reached
  - Run 2: not reached
  - Run 3: not reached

## Compute Unit (CU) Accounting

- **CU Before:** 396.84
- **CU After:** 396.66
- **CU Used (Difference):** 0.18

## Failure Details

### Install Error (`vllm-tpu` & `tpu-inference vllm`)
```text
vllm-tpu failed (code 1): error: subprocess-exited-with-error
  
  × Getting requirements to build wheel did not run successfully.
  │ exit code: 1
  ╰─> No available output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'vllm-tpu' when getting requirements to build wheel
tpu-inference vllm failed (code 1): error: subprocess-exited-with-error
  
  × installing build dependencies for vllm did not run successfully.
  │ exit code: 1
  ╰─> No available output.
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'vllm' when installing build dependencies for vllm
```

### Load Error
```text
Traceback (most recent call last):
  File "/tmp/ipykernel_7136/16632466.py", line 76, in <cell line: 0>
    import vllm
ModuleNotFoundError: No module named 'vllm'
```
