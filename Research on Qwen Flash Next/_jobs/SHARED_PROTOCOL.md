# Shared protocol for every Qwen Flash Next research job

You are one of nine parallel Antigravity research sessions. You own ONE paper.
You do not implement code, start a GPU, edit root shipping docs, or write
anyone else's file.

## Read first

1. `/home/sahil/Desktop/sahil/colab/Running Qwen flash next on Colab.md`
2. `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/CLAIMS.md`
3. `/home/sahil/Desktop/sahil/colab/Research on Qwen Flash Next/README.md`
4. `/home/sahil/Desktop/sahil/colab/architecture.md` (context; do not treat as gospel)
5. Your assigned claim IDs only.

## Write exactly one file

The path is given in your job prompt. Overwrite it with the finished paper.
Do not create extra files. Do not edit `_jobs/`, `CLAIMS.md`, or `README.md`.

## Research method

- Fetch **primary sources**: Hugging Face model cards and `config.json` /
  `README.md` raw files, NVIDIA datasheets, Google Colab / AI Pro compute-unit
  docs, vLLM / SGLang / TensorRT-LLM docs, llama.cpp issues, measured
  benchmarks with hardware named.
- Use web search, then open the actual page. A Google-search URL is not a
  citation.
- Recalculate every numeric claim that is arithmetic (tok/s ceiling, GB,
  minutes, CU). Show the formula.
- If two sources disagree, report both and pick one with a reason.
- If you cannot find a primary source, label **UNVERIFIED**. Do not invent.
- Date-stamp the paper **2026-10-17**.
- Prefer sources from 2026 when the model family is Qwen 3.8 / Flash-Next.

## Paper shape

```
# <title>
Researched: 2026-10-17 | Engine: Gemini 3.8 Flash (Medium) | Job: NN

## Verdict in one paragraph
## Claim table (ID | chat claim | label | corrected value | source URL)
## Findings (sections)
## Independent calculations
## Conflicts with architecture.md / prior research/
## What remains unverified
## Sources
```

Labels: CONFIRMED, PARTIALLY TRUE, FALSE, UNVERIFIED, INDEPENDENT CALC.

Cite real URLs. Quote config fields and datasheet numbers. Keep the paper
self-contained; the synthesizer will not re-fetch.
