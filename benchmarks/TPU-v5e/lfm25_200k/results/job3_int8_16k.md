# Job 3 Report: LFM2.5 on TPU-v5e with int8 KV at 16K

## 1. Executive Summary & Stop Trigger

- **Status**: STOPPED AT PREFLIGHT P5
- **Reason**: ARCHITECTURE.md has no launch for Colab session tpu-lfm25 (only a timeout citation of lfm25_coldstart_driver.py). STOP. No TPU run.
- **Execution Date**: 2026-11-02
- **Environment**: Colab TPU v5e (session target: `tpu-lfm25`)
- **Model**: `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU`

---

## 2. Preflight Checks (P1 - P5)

- **P1 PASS**: `git rev-parse --short HEAD` == 5d256a7; no tracked changes [MEASURED].
- **P2 PASS**: sha256sum tpu_v5e_patches.py == 155db072b3fb75141c836337a35382ff4d3f70fdd505e92bf0495760f2ae87f8 [MEASURED].
- **P3 PASS**: ./run_cpu_tests.sh -> 5 passed in 4.09s [MEASURED].
- **P4 PASS [MEASURED]**:
  - LFM_KV_INT8 off: filename `tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len100000__b16-8192__42f6059b.tar`, key `42f6059b`.
  - LFM_KV_INT8=1 at 16K: filename `tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__afa6a03f.tar` contains `__kvint8`, key `afa6a03f` (not 42f6059b).
- **P5 FAIL**: ARCHITECTURE.md has no launch for Colab session tpu-lfm25 (only a timeout citation of lfm25_coldstart_driver.py). STOP. No TPU run.

---

## 3. Cache Naming and Keys

- **LFM_KV_INT8 off**: filename `tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len100000__b16-8192__42f6059b.tar`, key `42f6059b` [MEASURED].
- **LFM_KV_INT8=1 at 16K**: filename `tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__afa6a03f.tar` contains `__kvint8`, key `afa6a03f` (not 42f6059b) [MEASURED].

---

## 4. Run Metrics (Stopped at P5)

Because preflight failed at P5, no remote TPU session was launched:

- **Compile time**: NOT RUN.
- **KV memory**: NOT RUN.
- **Blocks**: NOT RUN.
- **Peak HBM**: NOT RUN.

---

## 5. Needle In A Haystack (16K Context)

- **Needle expected vs got**: NOT RUN.

---

## 6. P2 Check (ShortConv query_start_loc Patch)

- **P2 log line**: NOT RUN.

---

## 7. Scale Probe (Per-Layer max |K| and max |V|)

- **Per-layer max|K|/max|V|**: NOT RUN.
- **Suggested scales**: NOT RUN.

---

## 8. Conclusion and Next Steps

1. Preflight verified code integrity, patches checksum, CPU unit tests, and compile cache isolation (P1–P4 passed).
2. Preflight halted at P5: ARCHITECTURE.md has no launch for Colab session tpu-lfm25 (only a timeout citation of lfm25_coldstart_driver.py). STOP. No TPU run.
3. Needle expected vs got, P2 log line, per-layer max|K|/max|V|, suggested scales, compile time, KV memory, blocks, peak HBM were NOT RUN and no values were invented.
4. Next step: Document the launch protocol for Colab session `tpu-lfm25` in ARCHITECTURE.md before attempting a TPU run.
