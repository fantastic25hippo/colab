# 07b — TPU v5e-1: vLLM/tpu-inference and TorchTPU Execution

Continuation of `07-TPU-v5e-Runtime.md`. Covers the two execution paths, launch flags, TorchTPU compilation, and the three single-stream traps. Source: `TPU-V5e-Best-Practices.md` (distilled from `TPU_v5e_LLM_Inference_Analysis.pdf` and `MoE_Models_on_TPU_v5e-1.pdf`).

## 1. vLLM / tpu-inference (vllm-jax) best practices

External report cold start on fresh Colab: **502.84–503.54 s total, 408.59–413.41 s (81–91%) is XLA compilation** (worst 738 s within 986 s). **In-repo measured (`../benchmarks/TPU-v5e/tpu-final/RESULTS.md`):** `Qwen/Qwen3-4B` bf16 on `vllm-tpu==0.32.0` took **388 s cold** (147.5 s compile, 38%) and **235.33 s with a warm XLA cache** (48.5 s precompile, 21%), while the `/dev/shm` CPU-torch pipeline cuts non-compile prep to **25.8 s** (`../benchmarks/TPU-v5e/FAST-COLDSTART.md`).

**Verified launch flags:**
```bash
export VLLM_TPU_USING_PATHWAYS=0
export TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export VLLM_USE_V1=1
export TPU_INFERENCE_MOE_BACKEND="gmm_v2"
export TPU_INFERENCE_ENABLE_SWA_RING_KV=1

vllm serve /content/model-w4a16-emb4-ct \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --kv-cache-dtype int8 \
  --kv-cache-memory 6442450944 \
  --max-model-len 200000 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 2048 \
  --enable-chunked-prefill \
  --gpu-memory-utilization 0.92 \
  --num-scheduler-steps 8 \
  --no-enable-prefix-caching
```

Why each flag:
- `--gpu-memory-utilization 0.92` → 14.49 GiB ceiling, 1.26 GiB left for XLA.
- `--kv-cache-memory` pinned → bypasses the startup profiler crash.
- `--max-num-batched-tokens 2048` → trims the bucket ladder.
- `--kv-cache-dtype int8` → true 8-bit compression. **Bug in v0.30.0:** `--kv-cache-dtype fp8` logs fp8 but silently allocates BF16 in HBM. Use symmetric int8 `torch.int8/jnp.int8` with BF16 scales in a ring buffer; cast int8→bf16 on the VPU at zero HBM overhead.
- `--num-scheduler-steps 8` → buffers 8 tokens in HBM per host readback, bypassing the 31.8–37.2 tok/s host-sync cap.
- `TPU_INFERENCE_ENABLE_SWA_RING_KV=1` → mandatory for SWA models; stock disables sliding windows and balloons KV (e.g. 1.356 GiB → 5.341 GiB).

**Conflict:** the newer quantization guide (08) says `--kv-cache-dtype int8` is **not supported by the TPU Ragged Paged Attention kernel**, and only fp8 is wired (but may silently fall back to bf16). This doc's int8 claim is likely specific to the TorchTPU custom ring path, not vLLM RPA. Verify token counts in `kv_cache_utils.py` logs; see 11.

## 2. TorchTPU path (fastest startup for a single-user notebook)

This is the most direct path for single-user `B=1` interactive use — it avoids the 400 s+ vLLM multi-bucket precompile.

- **Eager is fatal:** eager/naive lazy ≈ **5.0 tok/s** due to per-operator dispatch + continuous graph tracing.
- **Compile:** `torch.compile(model, backend="tpu", mode="reduce-overhead", fullgraph=True)` → up to **98.69× speedup**.
- **Compile only two static shapes:** one bucketed prefill chunk `[1, 2048]` + one decode `[1, 1]`.
  - Micro-graphs: 3.00 s.
  - Full hybrid MoE decode: 15–45 s.
  - With pinned KV + caching: 75–110 s / 90–140 s.

**Static Ring KV Cache pattern:** pre-allocate `k_int8`/`v_int8` shaped `[layers,1,kv_heads,max_len,head_dim]` as `torch.int8` on `device="tpu"`, plus `k_scale`/`v_scale` shaped `[layers,1,kv_heads,max_len,1]` as `torch.bfloat16`. Keep shapes invariant across 200K steps.

**Critical:** pass the step index as a 0-D tensor on TPU:
```python
pos_tensor = torch.tensor([pos], device="tpu", dtype=torch.int32)
ring_idx = pos % max_len
```
TorchTPU 0.2.0 forces `dynamic=False` by default and XLA Bounded Dynamism (SymInts) is incomplete, so **growing KV by 1 token per step triggers 115–341 ms XLA recompilation on every token**.

## 3. The three single-stream runtime traps

### Trap 1 — compilation and shape recompilations
- Cause: XLA requires static dims; dynamic seq-len = full recompilation (seconds to minutes).
- Fix vLLM: pre-allocated fixed block pool + power-of-two bucket ladder compiled once at startup.
- Fix TorchTPU: wrap in `torch.compile` with a static upper-bound `StaticCache`/ring buffer so XLA slices logical dims inside one executable.

### Trap 2 — true HBM retention / silent HBM unpacking
- Cause: standard HF `compressed-tensors` linear layers and unpatched tpu-inference fallbacks unpack 4-bit → persistent BF16 HBM buffers on first forward, or throw `NotImplementedError: compressed-tensors scheme unimplemented`. Unpacking 4× memory → immediate OOM for 12B–20B models on 16 GB v5e, and **3.8–4.0× decode slowdown**.
- Fix:
  - Never `model.to("tpu")` after `from_pretrained`. Use `device_map="tpu"` or load packed int32 shards directly onto TPU.
  - vLLM: route MoE experts through the Tokamax `gmm_v2` kernel — retains int4 packed tensors in HBM, unpacks tiles into BF16 strictly inside 32 MB VMEM before MXU.
  - TorchTPU: Packed-INT4 Gather-Then-Unpack — hold all experts as `W_packed [E, out, in//8]`, gather only top-k active experts (e.g. 4 of 64) across the HBM bus, bitwise shift-and-scale unpack inside a fused VMEM kernel: `((packed >> shifts) & 0xF) - 8`.

### Trap 3 — host-sync cap (31.8–37.2 tok/s in TorchTPU / naive loops) and 49% BW floor
- Cause (in TorchTPU / naive step loops): streaming token IDs back to the Python host token-by-token over PCIe Gen4 introduces a ~27–31 ms per-step sync barrier. (**Note:** In-repo measured `vllm-tpu==0.32.0` on `Qwen3-4B` bf16 is **not** capped at 37 tok/s — it achieves **84.92–88.58 tok/s** out of the box without `--num-scheduler-steps 8`, sustaining ~683 GB/s or **83% of peak 819 GB/s HBM bandwidth**.)
- Fix: multi-token on-chip buffering.
  - Pre-allocate `onchip_token_buf[step]` in HBM.
  - Transfer chunks to host every 64–128 steps (TorchTPU) or every 8 steps via `--num-scheduler-steps 8` / StepPool async / a `jax.lax.scan` N-step loop inside the compiled XLA graph before readback.

## 4. Which path to choose

| Situation | Path | Why |
| :--- | :--- | :--- |
| Single-user, batch 1, short/medium context, fastest startup | TorchTPU 0.2.0 with two static shapes | Avoids 400 s+ multi-bucket precompile; 25–60 s XLA compile; 106–135 tok/s on small models |
| Serving vLLM, long context, paged attention, MoE | vLLM + tpu-inference JAX path | Bucketed precompile + RPA + gmm_v2; slower start but production semantics |
| Need W4A16 compressed-tensors on TPU | vLLM/torchax path (PR #3487) or JAX path with #3653/#3660 | See 08 for the exact support matrix |
| Need W8A8 INT8 | vLLM path (`MODEL_IMPL_TYPE=vllm`) | Native 393 TOPS INT8; see 08 |

**Reality check:** vLLM on v5e-1 spends 81–91% of cold start compiling XLA. TorchTPU with two shapes is the only way to get a notebook-class startup, but it gives up paged attention and continuous batching.
