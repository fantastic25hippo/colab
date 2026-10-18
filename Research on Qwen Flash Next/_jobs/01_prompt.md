Read `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/_jobs/SHARED_PROTOCOL.md` and follow it exactly.

You are Job 01. Hardware and Colab billing.

Write ONLY this file:
`/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/01_colab_g4_hardware_and_cu.md`

Own claims **H1–H16** in CLAIMS.md.

## What you must settle

1. What GPU is Colab `--gpu G4`? Marketing name, SKU (Workstation vs Server Edition), compute capability, VRAM.
2. Memory bandwidth: 1,597 vs 1,792 GB/s. Which number is the Colab card? GDDR7 vs HBM? NVIDIA datasheet + Colab/GCP page.
3. Peak tensor TFLOPS/PFLOPS at NVFP4 / FP8 / FP16. The chat says 4 PFLOPS FP4 and ~1000+ TFLOPS FP8. Confirm or correct.
4. Colab Compute Unit burn rates for G4, L4, A100 40GB, A100 80GB, High-RAM. The chat says G4 ~6.4–6.7 CU/hr and L4 ~2.1. The existing plan (`architecture.md`) says G4 ~8.71 and L4 ~1.71 and A100 40GB ~5.37. **This conflict is the most important H-claim.** Find official or recently measured CU/hr tables (Google AI Pro / Colab docs, community tables dated 2026).
5. Can this Google AI Pro 200 CU plan actually assign G4, or is it often grayed out?
6. Is Colab single-GPU only (no tp=2 across VMs)?
7. Billing: wall-clock of the connected runtime, including idle.
8. L4 specs: 24 GB, 300 GB/s, ~242 TFLOPS FP8 — confirm. Ada L4 does not have native FP4/FP8 tensor cores the same way Blackwell does; do not copy the chat’s 16× tensor-compute comparison uncritically.
9. Confirm Blackwell `sm_120` / `sm_120a` native NVFP4.

## Required calculations

- tok/s hardware ceiling later jobs will use: bytes/s ÷ bytes/weight-payload. You only need to pin bandwidth and usable VRAM.
- Price multiplier G4/L4 from the CU/hr numbers you accept.
- Compare chat 6.4–6.7 vs architecture.md 8.71: which is current as of Oct 2026?

## Sources to start from (then find better)

- NVIDIA RTX PRO 6000 Blackwell datasheet (Workstation vs Server)
- Google Colab GPU list / AI Pro compute units
- `architecture.md` in this folder (treat as a competing claim, not truth)
- GCP A2/A3/G4 instance docs if Colab maps onto them

Do not research model architecture or tok/s tables. That is Jobs 02–05.
