# 01 — Colab Platform, Hardware and Compute-Unit Costs

Consolidated from `Fast Headless Colab Weight Loading` (measured tier table), `Google AI Pro Cloud Benefits`, `L4 GPU Model Loading Comparison` and `Optimizing AI Agent Colab Workflows`. Numbers tagged as in the index.

## 1. The runtime model

- `colab new` provisions a **stateless container from a pre-warmed pool in 4.5–4.9 s [M]**. Root filesystem is empty; `/content`, `~/.cache` and `/dev/shm` are fresh.
- `colab stop` (`/tun/m/unassign`) **wipes disk and RAM**. Colab does not support custom disk snapshots, persistent volumes, or booting a user image; `/tun/m/assign` only boots Google-managed images. Every session must re-fetch or re-mount weights.
- A **leaked VM keeps burning CUs**. A G4 leak can drain 400 CU in ~44.9 h [M/G]. Always trap teardown and verify with `colab sessions`.
- Colab does **not let you choose the GCP region** of the VM. This matters for GCS egress and for co-locating a bucket with the runtime.

## 2. Runtime tiers (measured unless noted)

| Tier | CU/hour | System RAM | Default `/dev/shm` | Accelerator memory | hf_transfer → disk | hf_transfer → `/dev/shm` | 400 CU buys |
| :--- | ---: | ---: | ---: | :--- | ---: | ---: | ---: |
| **T4** | 1.07 [M] | 12 GB | 5.7 GB | 15–16 GB GDDR6 | 105 MB/s | 200–270 MB/s | 373.8 h [G] |
| **L4** | 1.54 [M] | 52 GB | 26.0 GB | 22.5 GB GDDR6 (24 GB physical, ECC) | 252 MB/s | 200–270 MB/s | 259.7 h [G] |
| **TPU v5e-1** | 2.92 [M] | 47 GB | 23.0 GB | 16.0 GiB HBM | 107 MB/s | 223 MB/s | 137.0 h [G] |
| **A100** | 5.30 [M] | 83 GB | 41.0 GB | 40 GB (80 GB with `--high-mem`) | 353 MB/s | 200–270 MB/s | 75.5 h [G] |
| **A100 `--high-mem`** | 5.30 [M] | ~170 GB | ~85 GB | 80 GB HBM2e | 353 MB/s | 200–270 MB/s | 75.5 h [G] |
| **G4** | 8.90 [M] | 176 GB | 87.0 GB | 96 GB GDDR7 (RTX PRO 6000 Blackwell, sm_120) | 286 MB/s | 200–270 MB/s | 44.9 h [G] |

Notes:
- **L4:** Ada Lovelace SM89, ~300 GB/s GDDR6, 48 MB L2, ~121 TFLOPS BF16/FP16, ~242.5 TFLOPS FP8/INT8.
- **G4:** native 176 GB host RAM and 87 GB `/dev/shm`; it does **not** need the High-RAM toggle for an 80 GB model. This contradicts the Google AI Pro conversation (see 11).
- **A100:** `--high-mem` is the only way to get 80 GB VRAM and ~170 GB RAM. Standard A100 is 40 GB VRAM / 83 GB RAM.
- **TPU v5e-1:** 15.75 GiB visible to XLA/PjRt; see 07.

## 3. Disk vs `/dev/shm` (the first bottleneck)

- On **T4** and **TPU v5e-1**, root-disk writes are throttled to **105–107 MB/s**; `/dev/shm` more than doubles that to **200–270 MB/s [M]**.
- On **L4 / G4 / A100**, disk writes are **252 / 286 / 353 MB/s [M]**, matching or beating default `/dev/shm` downloads.
- Local copy of 4 GB: **12–36 s on disk (111–333 MB/s)** versus **0.5–1.7 s in `/dev/shm` (2.35–8.0 GB/s) [M]**. A 75 GB disk→RAM read adds an avoidable **225–675 s [G]**.
- Writing 75 GB to disk then reading it back is the classic double-copy penalty. Prefer direct-to-RAM or direct-to-VRAM streaming.

## 4. `/dev/shm` mechanics

- Linux mounts tmpfs at **50% of physical RAM** by default: 26 GB on L4, 41 GB on A100, 87 GB on G4 [M].
- Colab gives passwordless sudo inside the container, so you can raise the ceiling in ~50 ms without reboot:

```bash
sudo mount -o remount,size=45G  /dev/shm   # L4, 52 GB RAM
sudo mount -o remount,size=72G  /dev/shm   # A100 standard, 83 GB RAM
sudo mount -o remount,size=155G /dev/shm   # G4, 176 GB RAM
```

- `size=` is a **virtual ceiling**, not pre-allocation. Physical RAM is consumed only as bytes are written; `rm` returns it immediately.
- When loading from `/dev/shm` into VRAM, pass `--no-mmap` to llama.cpp (or the equivalent) and **unlink the shm files after VRAM allocation**. mmap from tmpfs gives no I/O advantage and can double host-memory pressure.
- **Do not use FUSE mounts for mmap weight loading.** `mmap()` completes instantly, then the first forward pass triggers thousands of synchronous 4–128 KB page faults across the FUSE boundary; loading stalls >30 s and generation can collapse below **0.5 tok/s [M/G]**. This applies to `colab drivemount`, `rclone mount` and `gcsfuse` when the engine mmaps the files. Sequential `rclone copy`/`gcloud storage cp` to local disk or `/dev/shm` is different and can be fast.

## 5. Compute-unit economics

- Google AI Pro includes **200 Colab compute units/month** and **$10/month Google Cloud credits**; the 5 TB storage is Google One (Drive/Gmail/Photos), not a GCS bucket quota.
- At 8.90 CU/h, 200 CU is **22.47 h of G4** (~45 min/day). At 1.54 CU/h, it is **129.9 h of L4**; at 1.07 CU/h, **186.9 h of T4**.
- Cold starts are billed at the full accelerator rate. A 4.5-minute G4 cold start costs **~0.67 CU**. Twenty per day = **90 min/day = 13.35 CU/day**; 200 CU is gone in ~15 days before generating a token.
- **4.5-minute break-even rule [G]:** if the gap between runs is shorter than the cold-start time, keeping the VM warm is cheaper. If startup is 15 s and the debate is 90 s, restarting is cheaper.
- Example from the Council conversation: 3 revision rounds, 3 min load + 1 min generate each, 45 s debate. Restart every round = **12.0 billed minutes**; keep warm = **7.5 billed minutes**. Rule: keep warm if `model startup > debate time`; restart if `startup < debate time`.
- The same logic says: **batch work into 2–3 sessions/day** rather than 20 isolated one-shot VMs.

## 6. Bandwidth physics (why tok/s is not a compute problem)

- Batch-1 autoregressive decode is **memory-bandwidth bound**: each token reads all active weights plus the KV cache. `tok/s ≈ effective_bandwidth / active_bytes_per_token`.
- L4: ~300 GB/s → a dense 15 GB model reads 15 GB/token, capping theory at ~20 tok/s and reality at **~15–17 tok/s [G]**. G4: ~1,792 GB/s; A100-80: ~2,039 GB/s. A dense 75 GB model caps at **~18 tok/s on G4 / ~21 tok/s on A100-80 [G]**.
- To exceed 100 tok/s on one GPU you must reduce **active bytes per token**: MoE (activate 1.5–10 GB of a 15–80 GB model), speculative decoding (verify 3–5 tokens per pass), or native FP4/NVFP4 on Blackwell.
- Examples: Qwen3-30B-A3B Q4_K_M (~16.5 GB total, ~1.8 GB active) on L4 → **115–140 tok/s [G]**; Llama-4-Scout-109B-A17B or Qwen-MoE Q4 on G4 → **120–160 tok/s [G]**.
- Dense 15 GB on A100/G4: 78–105 tok/s native, 180–280 with speculation [G]. Dense 75 GB on G4: ~18 native, 55–75 with a draft model [G]. MoE + N-gram can exceed 150 tok/s [G].

## 7. Can a 75 GB model download + load in under 5 minutes?

| Tier | 75 GB in VRAM? | Download to disk | Disk→VRAM penalty | Download + load via `/dev/shm` | Meets <300 s? |
| :--- | :--- | ---: | ---: | ---: | :--- |
| T4 (16 GB) | No | 714 s (11.9 min) [G] | N/A (OOM) | N/A (12 GB RAM) | **No** |
| TPU v5e-1 (16 GiB) | No | 701 s (11.7 min) [G] | N/A (OOM) | N/A (47 GB RAM) | **No** |
| L4 (22.5 GB) | No (CPU offload <3 tok/s) | 298 s (5.0 min) [G] | +225–675 s | N/A (52 GB RAM) | **No** |
| A100 standard (40 GB) | No (CPU offload) | **212 s** (3.5 min) [G] | +225–400 s | ~225 s with remount | **Download yes, VRAM no** |
| A100 `--high-mem` (80 GB) | Yes (tight: 75 GB + 5 GB KV) | **212 s** [G] | +225–400 s | ~230–265 s | **Yes** |
| G4 (96 GB) | Yes (96 GB VRAM + 176 GB RAM) | **262 s** (4.4 min) [G] | +225–400 s | ~245–285 s | **Yes (best tier)** |

Only **G4** and **A100 `--high-mem`** can download and load a 75 GB model into accelerator memory in under 5 minutes. L4 is a 15 GB-class machine; 75 GB requires quantization or multi-GPU/offload.

## 8. Practical fit guide

| Model size / format | Best Colab tier | Why |
| :--- | :--- | :--- |
| ≤8B BF16 (~15 GB) | L4 (1.54 CU/h) | Fits 22.5 GB VRAM; `/dev/shm` holds it; ~78–128 s cold start from HF |
| 8–16B FP8 / 4-bit (~8–16 GB) | L4 | Pre-quantized; leaves KV headroom; see 04 |
| 24B 4-bit MoE (~13–15 GB) | L4 | LFM2.5-24B-A2B class; 130–140+ tok/s [G] |
| 30B MoE Q4 (~16–18 GB) | L4 | Qwen3-30B-A3B class; 115–140 tok/s [G] |
| 70–80B FP8 / Q4 (~40–80 GB) | G4 or A100-80 | 96 GB / 80 GB VRAM; 130–175 tok/s MoE [G] |
| 180 GB BF16 MoE | **Does not fit** | 96 GB VRAM and 155 GB `/dev/shm` both too small; quantize to FP8 (~80–85 GB) or Q4 (~42–48 GB) |

## 9. Caveats

- The tier table is a mix of measured values and one source's guesses; treat CU-hour figures as measured but 400-CU hours as arithmetic.
- The 75 GB timeline table is explicitly a planning estimate; the only measured anchors are the 212 s A100 download and 262 s G4 download.
- G4 “High-RAM” confusion: the Google AI Pro conversation claims standard G4 has 12.7 GB RAM and will OOM. The measured Fast Headless table says 176 GB. Trust the measured table; see 11.
- FUSE can be fine for a sequential copy. The failure mode is specifically mmap-backed loading by the inference engine.
