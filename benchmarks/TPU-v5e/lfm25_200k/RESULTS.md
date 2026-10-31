# LFM2.5-24B-A2B 200K on TPU v5e-1

## 2026-10-31 21:29 IST: LFM2.5-24B-A2B W8A8+W4A16 on TPU-v5e

**Model:** `LiquidAI/LFM2.5-24B-A2B` (`Lfm2MoeForCausalLM`), quant W8A8 backbone + W4A16 MoE, checkpoint `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` (9 shards).
**Weights:** Hugging Face pull into `/dev/shm/lfm25_200k` (`max_workers=12`), 11.60 s (MEASURED).
**Hardware:** TPU v5e-1, HBM limit 15.748 GiB, in use 14.738 GiB after init, peak 14.821 GiB (MEASURED).
**CLI:** `colab` 0.7.5, session `tpu-lfm25`, `colab --auth=oauth2 new -s tpu-lfm25 --tpu v5e1`. Provision 4.77 s.
**Engine:** vllm-tpu 0.32.0 + tpu-inference 0.32.0, torch 2.10.0+cpu, Python 3.12 venv in `/dev/shm`. `MODEL_IMPL_TYPE=vllm`.
**Flags:** `max_model_len=200000`, `max_num_seqs=1`, `max_num_batched_tokens=8192`, `kv_cache_dtype=fp8` (`float8_e4m3fn` attn), `gpu_memory_utilization=0.92`, `enable_prefix_caching=False`, temperature 0. One rep each. B=1.

### Results

| Test | Value | stddev | Reps | Label |
| :--- | ---: | ---: | ---: | :--- |
| Short TTFT | 0.1528 s | n/a | 1 | MEASURED |
| Short decode, 64 tok | 170.26 tok/s (0.3759 s) | n/a | 1 | MEASURED |
| TTFT @ 32,704 tok | 2.342 s (13,964 tok/s) | n/a | 1 | MEASURED |
| 64 tok @ 32,704 tok prompt | 23.87 tok/s over 2.68 s wall | n/a | 1 | MEASURED |
| TTFT @ 199,936 tok | 32.578 s (6,137 tok/s) | n/a | 1 | MEASURED |
| 64 tok @ 199,936 tok prompt | 1.93 tok/s over 33.10 s wall | n/a | 1 | MEASURED |

The 23.87 and 1.93 figures are `64 / generate_wall`. Prefix caching is off, so that wall includes a second full prefill (progress bars: 12,226 tok/s in at 32K, 6,041 tok/s in at 200K). They are not steady-state decode. The 170.26 tok/s figure is the short prompt, where prefill is tiny.

Sample (short, 64 tok, temperature 0), coherent: no. It repeats and misspells the prompt:
> The fundamental principles of astphysics and stellar evolution are the fundamental principles of astphysics and stellar evolution.

### Startup times (seconds)

| Step | Seconds | Label |
| :--- | ---: | :--- |
| Runtime up (`new`) | 4.77 | MEASURED |
| Venv download / unpack | 9.59 / 6.42 | MEASURED |
| Weights pull (overlaps TPU init) | 11.60 | MEASURED |
| TPU init | 20.20 | MEASURED |
| Engine init (load, compile, KV) | 269.29 | MEASURED |
| **In-VM start to first token** | **310.91** | **MEASURED** |
| **Global start to first token** | **320.99** | **MEASURED** |

### HBM

- After weights, JIT cleared: 12.45 / 15.75 GiB (MEASURED, `tpu_runner.py:1335`).
- Compact-mamba: 13,361 attn blocks, 213,724 tokens. 10 attn layers × 16,384 B = 2.04 GiB. 30 short-conv layers × 9 slots, state shape `(9, 2, 2048)` bf16. KV-init HBM 14.49 / 15.75 GiB (MEASURED).
- Steady through the 200K generates: 14.738 GiB in use, peak 14.821, limit 15.748 (MEASURED).

### Cost

- CU rate while the assignment was active: 3.09/hr (MEASURED). Balance before 391.68, after 391.46. Displayed CU used 0.22 (MEASURED). Rate after stop: 0.00/hr.
- CU per 100K short-context decode tokens = `(100000 / 170.26) / 3600 * 3.09` = 0.50 (derived from the 170.26 tok/s decode).

### Gotchas

- `jax.clear_caches` ran (`Cleared JIT caches after weight loading`). The 2.04 GiB KV cache fit after that.
- `KVCacheManager.get_kv_cache_spec` returned normally. The prior run died on a missing `cache_config`.
- `utils.py` still logs the full padding ladder `[16 … 8192]` before the wrapper returns. The list the runner stored was not logged on its own.
- Every `generate` logs `Should not schedule a request that does nothing!` and still returns tokens.
- Interpreter shutdown raises `AttributeError: 'VllmModelWrapper' object has no attribute 'modules'` in an atexit hook. The results JSON was already written. Host driver exit code 0.
- One rep only. The short sample is repetitive.

**Runtime stopped:** yes. `tpu-lfm25` terminated, `sessions` empty at 21:35 IST.
