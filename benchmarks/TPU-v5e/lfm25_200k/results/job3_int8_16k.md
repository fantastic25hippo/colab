# Job 3 Report: TPU v5e Context 16K Run (STOP: ValueError)

## 1. Executive Summary & Stop Trigger

- **STOP**: Context 16384 ValueError `invalid literal for int() with base 10: 'input_ids'` at prompt_len=16320. No scale-probe lines. Session `tpu-lfm25` stopped.
- **int8 NOT applied**: `kv_cache_dtype=auto`; `regular_attn_dtype=bfloat16`; `32768 B/page`; `RPA q_bfloat16_kv_bfloat16`. `LLM()` never passed `kv_cache_dtype=int8`.

## 2. Preflight Checks (P1 - P5)

- **P1 PASS**: HEAD `ee02df3`.
- **P2 PASS**: sha `155db072b3fb75141c836337a35382ff4d3f70fdd505e92bf0495760f2ae87f8`.
- **P3 PASS**: 6 tests.
- **P4 PASS**: laptop keys `42f6059b` (off len100000) and `afa6a03f` (int8 16K str hash).
- **P5 PASS**: launch on `tpu-lfm25`.

## 3. Run Metrics & Engine Profiling (MEASURED)

- **Engine ready**: 27.13s
  - `weight_load`: 17.556s
  - `kv_alloc`: 0.132s
  - `compile_warmup`: 0.008s
- **TTFT**: 53.7384s (torch.compile 9.62s)
- **KV Allocation**: 6682 attn blocks × 32768 B = 2.04 GiB.
- **HBM Usage**:
  - HBM after init: 14.488/15.748 GiB
  - At fail: in_use 14.735 GiB, peak 14.786 GiB

## 4. Verification & Generation

- **Golden MATCH**: first 13 ids `[597, 4695, 20589, 34, 496, 89509, 278, 9761, 9896, 302, 5030, 37933, 1003]`.
- **16K generate**: never ran.
- **CONVDBG**: prefill line present (`JitTracer n=`).
- **Probe**: missing (no scale-probe lines reached).

## 5. Cache Tar & Artifacts

- **VM cache tar**: `tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__9535cd10.tar` (97.2 MB published).
- **Cache hashing**: Driver hashed patches as bytes under int8 (`9535cd10`); laptop str hash stays `afa6a03f`.

## 6. Compute Units

- **CU API**: 389.28→389.28 (delta 0.00, likely lag).

## 7. Execution Log Trace (Last Lines)

```
[LFM25_SUBRUNNER 16:32:18] Chat 64 tok: 17.419s; ids=[597, 4695, 20589, 34, ...]; text='The user asks: "Explain the fundamental principles of astrophysics and stellar evolution." ...'
[LFM25_SUBRUNNER 16:32:19] Decode 64 tok (ignore_eos): 0.7463s (85.75 tok/s)
[LFM25_SUBRUNNER 16:32:19] Context 16384 FAILED: ValueError: invalid literal for int() with base 10: 'input_ids'
[LFM25_SUBRUNNER 16:32:19] [WATERFALL] total_vm_s=96.67s accounted_s=90.77s unaccounted_s=5.91s
[LFM25_DRIVER 16:32:30] === Host WATERFALL ===
=== COLDSTART_RESULTS_JSONL ===
{"run_label": "job3_int8_16k", "diag": false, "in_vm_cold_start_to_first_token_s": 111.93, "total_cold_start_to_first_token_s": 126.55, "cache_publish": {"filename": "tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__9535cd10.tar", "tar_s": 0.04, "tar_mb": 97.2, "upload_s": 7.16, "verified_listed": true}, "subrunner_rc": 0}
=== END_COLDSTART_RESULTS ===
=== Coldstart driver finished with rc=0 ===
```

