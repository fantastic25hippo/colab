# LFM2.5-24B-A2B 200K on TPU v5e-1

## 2026-11-01 18:43 IST: short-context diagnostic — chat template vs raw completion (4096 ctx)

**Model:** `LiquidAI/LFM2.5-24B-A2B` (`Lfm2MoeForCausalLM`), W8A8 backbone + W4A16 MoE, checkpoint `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` (same tree as the 18:06 run).
**Flags:** `max_model_len=4096`, `max_num_batched_tokens=4096`, `max_num_seqs=1`, `enable_prefix_caching=False`, `trust_remote_code=True`, **no `kv_cache_dtype`** (vLLM resolved `auto`), `LFM_DIAG=1` (paddings `[16, 4096]`), `LFM_CONV_DEBUG=1`. B=1. One short prompt only.
**Cost/time (MEASURED):** session 18:36:52–18:43:47 IST, 0.10 CU (391.15 → 391.05), under the 0.25 CU cap.
**Startup (MEASURED):** provision 5.28 s; venv dl/unpack 7.60 / 6.52 s; weights 18.39 s; engine init **277.82 s**; first token (Run B) 318.72 s in-VM / 330.23 s global. HBM 14.732 GiB in use, 14.767 peak, 15.748 limit.

### Runs

| Run | Prompt path | Prompt ids | BOS 124894 | Sampling | Out | Coherent? |
| :--- | :--- | ---: | ---: | :--- | ---: | :---: |
| A | `llm.chat`, `add_generation_prompt=True` | 22 (expected) | 1 (expected) | temp 0.1, top_k 50, rep 1.05, 128 | — | **FAILED** |
| B | `llm.chat`, `add_generation_prompt=True` | 22 | 1 | greedy, rep 1.0, 64 | 64 | **YES** |
| C | raw `llm.generate` | 12 | 0 | greedy, `ignore_eos`, 64 | 64 | **NO** |

**Run A failure (MEASURED):** `ValueError: JAX does not support per-request seed.` The harness passed `seed=0`, which vLLM's JAX backend rejects. `seed=0` was removed afterwards; **Run A was not rerun**. The 22-token chat prompt and BOS count above are from Run B's actual `prompt_token_ids`.

**Run B (MEASURED, coherent):**
- prompt first 40: `[124894, 124899, 5922, 207, 89509, 278, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 22, 124900, 207, 124899, 63514, 207, 124901]`
- out ids (64): `[597, 4695, 20589, 34, 496, 89509, 278, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 2426, 969, 355, 267, 5939, 5439, 22, 1715, 1976, 401, 13816, 302, 5030, 37933, 1003, 61552, 309, 29460, 7216, 22, 27450, 736, 685, 1976, 267, 13999, 302, 5030, 37933, 1003, 9896, 34, 14124, 20, 38847, 20, 5030, 37933, 1003, 11051, 1171, 9418, 20, 28909, 6192, 20, 29460, 6900]`
- text: `'The user asks: "Explain the fundamental principles of astrophysics and stellar evolution." This is a broad request. They want an explanation of astrophysics fundamentals and stellar evolution. Likely they want a summary of astrophysics principles: physics, astronomy, astrophysics topics like radiation, gravitational forces, stellar formation'`
- wall 0.592 s.

**Run C (MEASURED, reproduces the bug):**
- prompt first 40: `[89509, 278, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 22]`
- out ids (64): `[440, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 437, 278, 9761, 9896, 302, 5030, 29544, 1003, 309, 29460, 7216, 22, 440, 9761, 9896, 302, 5030, 29544, 1003, 309, 29460, 7216, 437, 278, 9761, 9896, 302, 5030, 29544, 1003, 309, 29460, 7216, 22, 440, 9761, 9896, 302, 5030, 29544, 1003, 309, 29460, 7216, 437, 278, 9761, 9896, 302, 5030, 29544, 1003, 309, 29460]`
- text: `' The fundamental principles of astrophysics and stellar evolution are the fundamental principles of astphysics and stellar evolution. The fundamental principles of astphysics and stellar evolution are the fundamental principles of astphysics and stellar evolution. The fundamental principles of astphysics and stellar evolution are the fundamental principles of astphysics and stellar'`
- wall 0.376 s.

### CONVDBG (verbatim)

```
CONVDBG layer=model.layers.0.short_conv kind=prefill slot=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(int32[1])) n=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(int32[])) keep=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(bool[])) seq_lens=JitTracer(int32[8]) has_initial_states=missing prev_abs_sum=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(bfloat16[]))
CONVDBG layer=model.layers.0.short_conv kind=decode slot=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(int32[1])) n=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(int32[])) keep=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(bool[])) seq_lens=JitTracer(int32[8]) has_initial_states=missing prev_abs_sum=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(bfloat16[]))
CONVDBG layer=model.layers.0.short_conv kind=decode slot=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(int32[1])) n=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(int32[])) keep=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(bool[])) seq_lens=JitTracer(int32[8]) has_initial_states=missing prev_abs_sum=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(bfloat16[]))
CONVDBG layer=model.layers.0.short_conv kind=decode slot=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(int32[1])) n=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(int32[])) keep=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(bool[])) seq_lens=JitTracer(int32[8]) has_initial_states=missing prev_abs_sum=Tensor(<class 'jax._src.interpreters.partial_eval.DynamicJaxprTracer'> JitTracer(bfloat16[]))
```

All four lines are emitted during torchax tracing: `slot`, `n`, `keep`, `seq_lens` and `prev_abs_sum` are `JitTracer`/`DynamicJaxprTracer` reprs, not runtime scalars. `has_initial_states=missing` means the metadata object exposed no such attribute at trace time. The three `kind=decode` lines are separate trace invocations classified by the fallback path (`n` unreadable), not three runtime decode steps. `slot` and `n` values are therefore **not measured**; only the code path is.

### Verdict: template/prompt formatting, not conv/state and not the re-quant

- **MEASURED:** B is coherent and C loops with the same weights, engine, conv patch and greedy 64-token budget. The only differences are chat template + BOS (22 ids, BOS 1) vs raw completion (12 ids, BOS 0).
- **INFERRED:** the `ShortConv` replacement and conv cache work well enough to produce coherent text for 64 decode tokens in B. A conv/state bug is not required to explain C.
- **INFERRED:** the raw prompt is the trigger; greedy `ignore_eos` then holds the model on the prompt-echo path for 64 tokens. Sampling is not required for coherence: B uses no repetition penalty and is still coherent.
- **MEASURED:** this run used `kv_cache_dtype=auto` (bf16 KV) and C still looped, so fp8 KV is not the cause.
- **MEASURED:** the same W8A8/W4A16 checkpoint produces coherent Run B, so a Q4_0 re-quant defect is not supported by this evidence; the llama.cpp GGUF cross-check is not needed on this data.
- **UNMEASURED:** Run A (card sampling). A corrected Run A is a ~6 min / ~0.10 CU follow-up if wanted.

**Teardown:** TPU session stopped by the driver's `finally:` block. `colab ssh --proxy-mode` auto-created a **CPU** session after teardown (the known gotcha); it was stopped immediately. `colab sessions` empty, balance 391.05, rate 0.00/hr.

---


## 2026-11-01 18:06 IST: rerun after five RCA fixes (fp8 KV fallback)

**Model:** `LiquidAI/LFM2.5-24B-A2B` (`Lfm2MoeForCausalLM`), W8A8 backbone + W4A16 MoE, checkpoint `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` (17 files).
**Weights:** Hugging Face pull into `/dev/shm/lfm25_200k` (`max_workers=12`), 10.81 s (MEASURED).
**Hardware:** TPU v5e-1, HBM limit 15.748 GiB, in use 14.97 GiB after the 200K tests, peak 15.053 GiB (MEASURED).
**CLI:** `colab` 0.7.5, session `tpu-lfm25`, `colab --auth=oauth2 new -s tpu-lfm25 --tpu v5e1`. Provision 5.23 s.
**Engine:** vllm-tpu 0.32.0 + tpu-inference 0.32.0, torch 2.10.0+cpu, Python 3.12 venv in `/dev/shm`. `MODEL_IMPL_TYPE=vllm`.
**Flags:** `max_model_len=200000`, `max_num_seqs=1`, `max_num_batched_tokens=8192`, `kv_cache_dtype=fp8`, `gpu_memory_utilization=0.92`, `enable_prefix_caching=False`, temperature 0. One rep each. B=1.
**int8 KV:** plain `"int8"` was rejected at `LLM()` startup by vLLM 0.32.0 `CacheConfig` (`ValidationError: cache_dtype ... literal_error`; allowed list has `fp8`, `int8_per_token_head`, not `int8`). Switched to `fp8` per instruction, so the int8 KV path is still untested.
**Fixes present:** conv-cache metadata/`has_initial_states`, tpu_runner warning filter, int8 KV fallback, `[16, 4096, max_token_size]` paddings + JAX persistent cache env, `VllmModelWrapper.modules`.

### Old vs new (MEASURED)

| Metric | Old (2026-10-31 21:29) | New (2026-11-01 18:06) | Delta |
| :--- | ---: | ---: | ---: |
| Short TTFT | 0.1528 s | 0.1605 s | +0.0077 s (+5.0%) |
| Short decode, 64 tok | 170.26 tok/s | 165.91 tok/s | -4.35 tok/s (-2.6%) |
| TTFT @ 32,704 tok | 2.342 s | 2.3433 s | +0.0013 s (+0.1%) |
| 64 tok @ 32,704 tok prompt | 23.87 tok/s | 23.81 tok/s | -0.06 tok/s (-0.3%) |
| TTFT @ 199,936 tok | 32.5779 s | 32.3805 s | -0.1974 s (-0.6%) |
| 64 tok @ 199,936 tok prompt | 1.93 tok/s | 1.94 tok/s | +0.01 tok/s (+0.5%) |
| Engine init | 269.29 s | 388.73 s | +119.44 s (+44.3%) |
| In-VM start to first token | 310.91 s | 425.10 s | +114.19 s (+36.7%) |
| Global start to first token | 320.99 s | 435.63 s | +114.64 s (+35.7%) |
| HBM in use @ 200K | 14.738 GiB | 14.97 GiB | +0.232 GiB |
| HBM peak @ 200K | 14.821 GiB | 15.053 GiB | +0.232 GiB |
| Runtime up (`new`) | 4.77 s | 5.23 s | +0.46 s |
| Weights pull | 11.60 s | 10.81 s | -0.79 s |
| TPU init | 20.20 s | 21.19 s | +0.99 s |
| Venv download / unpack | 9.59 / 6.42 s | 3.04 / 6.48 s | -6.55 / +0.06 s |

### Five checks

| Check | Result | Evidence |
| :--- | :---: | :--- |
| Short 64-token sample coherent | **NO** | Still loops: `The fundamental principles of astphys and astphys are the fundamental principles of astphys and astphys.` No `astphysics`, but still degenerate. |
| `Should not schedule a request that does nothing!` gone | **YES** | 0 occurrences in the full run log. |
| atexit `AttributeError ... 'modules'` gone | **YES** | 0 occurrences; clean subrunner exit, host rc=0. |
| Engine init dropped from 269.29 s | **NO** | 388.73 s, +119.44 s (+44.3%). |
| 200K TTFT dropped from 32.578 s | **YES** (marginal) | 32.3805 s, -0.1974 s (-0.6%), near run-to-run noise. |

### HBM and cost

- After weights, JIT cleared: 12.45 / 15.75 GiB. Compact-mamba: 13,361 attn blocks, 213,724 tokens; 10 attn layers (fp8) + 30 short-conv layers (9 slots, `(9, 2, 2048)` bf16). KV-init HBM 14.49 / 15.75 GiB.
- Steady through the 200K generates: 14.97 GiB in use, peak 15.053, limit 15.748 (MEASURED). ~0.7 GiB headroom left.
- CU: balance 391.46 before, 391.15 after, **0.31 CU used** (MEASURED). The rejected-int8 attempt and a stray CPU monitor session both showed 0.00 balance delta. Rate after stop 0.00/hr.
- Active rate was not re-sampled this run; using the prior v5e observation of 3.09/hr gives `(100000 / 165.91) / 3600 * 3.09` = 0.52 CU per 100K short-context decode tokens (derived).
- Session stopped; `colab sessions` empty; no push.

### What still looks wrong

- **Coherence is not fixed.** The short sample still repeats a corrupted prompt fragment (`astphys and astphys`). The conv-cache metadata fix removed `astphysics` but not the looping.
- **Engine init regressed.** 388.73 s vs 269.29 s: weight load 119.32 s, compilation 228.24 s, including a 101.86 s warm-up pass. Buckets 16/4096/8192 still cost 9.64/38.57/38.45 s despite the JAX persistent cache, so pruning did not buy init time (likely cache misses from the changed graph/shape set).
- **Padding log is still misleading.** `utils.py:225` prints the full `[16 ... 8192]` ladder even though only 16/4096/8192 were precompiled. The pruned list did take effect for compilation; only the log is stale.
- **int8 KV remains unvalidated.** vLLM 0.32.0 rejects plain `int8`, so this run used fp8. The int8 fix needs a vLLM-side dtype extension before it can be tested.
- **200K TTFT win is marginal.** -0.6% is within noise; 32K TTFT and decode rates are unchanged, and HBM rose 0.23 GiB. This is not a meaningful speedup.
- **SSH monitor gotcha.** After teardown, `colab ssh -s tpu-lfm25 --proxy-mode` auto-created a CPU session. It was stopped immediately, cost 0.00 CU, and `sessions` is empty; only SSH while the VM is confirmed active.

---

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
