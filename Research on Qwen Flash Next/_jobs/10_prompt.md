Read `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/_jobs/SHARED_PROTOCOL.md` and follow it exactly.

You are Job 10, the synthesizer. The nine worker papers already exist. You do
not re-research the web unless a worker left a claim UNVERIFIED and a 30-second
fetch would close it. Prefer the worker papers.

Write ONLY this file:
`/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/10_claims_ledger_and_verdict.md`

Read, in order:

1. `Running Qwen flash next on Colab.md`
2. `Research on Qwen Flash Next/CLAIMS.md`
3. `Research on Qwen Flash Next/01_colab_g4_hardware_and_cu.md` through `09_quality_abliteration_agent_fit.md`
4. `architecture.md` and `README.md` (current locked 27B plan)

## Output shape

### 1. Executive verdict (one page)

Can the operator switch the Colab worker to Qwen 3.8 Flash-Next NVFP4 on G4?
Answer yes / no / not yet, with the three strongest reasons.

### 2. Master claims ledger

One table for every claim ID in CLAIMS.md:

| ID | Chat claim (short) | Worker job | Label | Corrected value | Best source |

Labels must be copied from the worker paper unless you have a documented
reason to override (worker contradiction). If two workers disagree, say so
and pick.

### 3. Internal contradictions in the Gemini chat

Especially the three 27B speed tables, G4 CU/hr 6.4–6.7 vs architecture.md 8.71,
bandwidth 1597 vs 1792, cache-hit 100k–200k tok/s.

### 4. Recalculated economics the operator can trust

One short table: CU per automated 25-turn task, CU per 1M output tokens,
tokens per 200 CU/month — using the **corrected** CU/hr and **corrected** tok/s,
not the chat’s.

### 5. Recommended next sitting

What to actually try on Colab, in order, without pretending the chat is a plan.

Do not implement anything. Do not start a GPU.
