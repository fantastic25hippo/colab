# Research on Qwen Flash Next

Deep research audit of the Gemini chat captured in
[`../Running Qwen flash next on Colab.md`](../Running%20Qwen%20flash%20next%20on%20Colab.md).

**Question:** are the hardware, model, speed, VRAM, and Colab Compute-Unit
claims in that document true enough to switch the Colab worker from
abliterated Qwen 3.8 27B (`llama.cpp` / A100) to
`nvidia/Qwen3.8-Flash-Next-NVFP4` on a Colab G4 (RTX PRO 6000 Blackwell)?

**Method:** nine parallel Antigravity CLI (`agy`) sessions, each on
**Gemini 3.8 Flash (Medium)**, each owning a disjoint claim slice.
A tenth synthesizer session runs after the nine finish.

Do not treat this folder as launch instructions. Shipping docs stay at the
repo root until a later sitting promotes a verdict.

## Source

- Chat dump: [`../Running Qwen flash next on Colab.md`](../Running%20Qwen%20flash%20next%20on%20Colab.md)
- Numbered claims: [`CLAIMS.md`](CLAIMS.md)
- Existing 27B plan (context only): [`../README.md`](../README.md), [`../architecture.md`](../architecture.md), [`../plan_audit_and_fixes.md`](../plan_audit_and_fixes.md)
- Prior 27B research archive: [`../research/`](../research/README.md)

## Job map

| Job | Owner file | Claim IDs | Scope |
| :--- | :--- | :--- | :--- |
| 01 | [`01_colab_g4_hardware_and_cu.md`](01_colab_g4_hardware_and_cu.md) | H1–H16 | Colab G4 SKU, VRAM, bandwidth, TFLOPS, CU/hr vs L4/A100, single-GPU |
| 02 | [`02_qwen38_flash_next_checkpoint.md`](02_qwen38_flash_next_checkpoint.md) | F1–F14 | `nvidia/Qwen3.8-Flash-Next-NVFP4` identity, MoE, experts, MTP, sizes |
| 03 | [`03_qwen38_27b_nvfp4_checkpoint.md`](03_qwen38_27b_nvfp4_checkpoint.md) | D1–D12 | `nvidia/Qwen3.8-27B-NVFP4` identity, hybrid GDN, sizes vs Blackfrost |
| 04 | [`04_glm53_flash_nvfp4.md`](04_glm53_flash_nvfp4.md) | G1–G8 | `nvidia/GLM-5.3-Flash-NVFP4` identity, VRAM, Colab impossibility |
| 05 | [`05_speed_roofline_and_mtp.md`](05_speed_roofline_and_mtp.md) | S1–S28 | Decode/prefill/TTFT/MTP/cache-hit numbers, independent roofline |
| 06 | [`06_runtime_stack.md`](06_runtime_stack.md) | R1–R12 | vLLM / SGLang / TRT-LLM / llama.cpp support, MTP, FlashInfer, Colab install |
| 07 | [`07_kv_cache_and_context.md`](07_kv_cache_and_context.md) | K1–K10 | Weight + KV + GDN + expert memory, 32K–128K headroom |
| 08 | [`08_colab_cu_economics.md`](08_colab_cu_economics.md) | E1–E18 | Race-to-sleep, agent-loop CU, 200 CU/month token math, API vs Colab |
| 09 | [`09_quality_abliteration_agent_fit.md`](09_quality_abliteration_agent_fit.md) | Q1–Q10 | Coding quality, abliteration, tools, thinking, vs current 27B worker |
| 10 | [`10_claims_ledger_and_verdict.md`](10_claims_ledger_and_verdict.md) | all | Synthesizer. Runs only after 01–09 exist. |

Job prompts live in [`_jobs/`](_jobs/). Do not edit them once a session has started.

## Verdict labels each paper must use

| Label | Meaning |
| :--- | :--- |
| **CONFIRMED** | Primary source (datasheet, model card, official docs, measured benchmark) matches the claim. |
| **PARTIALLY TRUE** | Directionally right, but a number, name, or condition is off. |
| **FALSE** | Primary source contradicts the claim. |
| **UNVERIFIED** | No primary source found. Do not upgrade a guess to CONFIRMED. |
| **INDEPENDENT CALC** | Recalculated from first principles (bytes/token × bandwidth, CU = hours × rate). Show the arithmetic. |

Every non-UNVERIFIED claim needs a real URL. Google-search wrapper links are not citations.

## Launch (already run)

```bash
MODEL='Gemini 3.8 Flash (Medium)'
ROOT='/home/sahil/Desktop/sahil/colab'
OUT="$ROOT/Research on Qwen Flash Next"
JOBS="$OUT/_jobs"

for n in 01 02 03 04 05 06 07 08 09; do
  # Flash (Medium) already encodes effort; --effort is rejected by agy.
  agy -p "$(cat "$JOBS/${n}_prompt.md")" \
    --model "$MODEL" \
    --dangerously-skip-permissions \
    --print-timeout 50m \
    --log-file "/tmp/qwen-flash-next-research/agy-${n}.log" \
    > "/tmp/qwen-flash-next-research/stdout-${n}.log" 2>&1 &
done
```

Working directory: `$ROOT`. Model: Gemini 3.8 Flash (Medium). Date: 2026-10-17.
