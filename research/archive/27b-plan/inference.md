# Inference

Playbook A for abliterated Qwen 3.8 27B on Colab A100 40GB: `llama.cpp` + DFlash 2 + `ngram-map-k4v`.

Architecture and files: [`architecture.md`](architecture.md). Session loop: [`ops.md`](ops.md). First-boot gates: [`first-boot.md`](first-boot.md).

Colab bills wall-clock, so tokens per second is CU savings.

---

## 1. What we are running

DFlash 2 is a 5-layer ~1.92B block-diffusion drafter. It reads hidden states from layers `[5, 19, 33, 47, 61]` and predicts a block in one pass. `dflash.block_size=8` clamps drafts to 7; we lock **n-max 5** (n=7 is slower; n=8/9 still drafts 7 and wastes GDN VRAM).

`ngram-map-k4v` is a zero-VRAM lookup that fires when the prompt already contains the next tokens (code edits). Stack only that one table with DFlash. `ngram-mod` plus k4v drops throughput ~19% on this model.

Unix **LF** only. CRLF in the prompt vs LF in the model kills n-gram matching.

Thinking is off. The orchestrator already planned.

---

## 2. Compile (once, cache on Drive)

Guest CUDA is driver 580 / toolkit 13.0. Pin master `efe3626` or tag `b11552`. DFlash 2 is already on master (PR #27816). Do not fetch PR #27342.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout efe3626

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build --config Release -j "$(nproc)" --target llama-server llama-quantize
```

Copy `build/bin/llama-server` and `build/bin/llama-quantize` to `/content/drive/MyDrive/colab_llm/bin/`.

Do not set `GGML_MMVQ_MAX` or `LLAMA_SCHED_POOL` (`KEEP_DELETED`). Forcing MMQ at verify width 6 is ~20% slower on A100-SXM4. Blackfrost Q4_K_M has zero IQ4_XS tensors. If first-boot one-shot coding is under 95 tok/s, inspect clocks and SKU; do not apply the MMVQ patch.

---

## 3. Quantize the sidecar (once)

```bash
DRIVE=/content/drive/MyDrive/colab_llm

./build/bin/llama-quantize \
  "$DRIVE/dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf" \
  "$DRIVE/dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf" \
  Q4_K_M

stat -c%s "$DRIVE/dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf"
sha256sum "$DRIVE/dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf"
```

Expected size **1,143,006,592** bytes (1,143,006,752 if mrope KV was injected). Record the local SHA. Keep the Q4 sidecar even with spare VRAM.

---

## 4. Launch (A100 40GB)

Paths are Drive. Alias advertised on `/v1/models` is `qwen3.8-27b`. Upstream default port is now **9931**, so `--port 8080` is mandatory.

```bash
DRIVE=/content/drive/MyDrive/colab_llm

"$DRIVE/bin/llama-server" \
  -m "$DRIVE/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf" \
  -md "$DRIVE/dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf" \
  -a qwen3.8-27b \
  --spec-type draft-dflash,ngram-map-k4v \
  --spec-draft-n-max 5 \
  --spec-draft-n-min 0 \
  --spec-draft-ngl all \
  -fa on \
  -c 100000 \
  --fit on --fit-target 768 \
  --ctx-checkpoints 4 --checkpoint-min-step 16384 \
  -b 2048 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  -bs --load-mode none --parallel 1 \
  --jinja --reasoning off --reasoning-format auto \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --repeat-penalty 1.0 \
  --api-key "${LLAMA_API_KEY}" \
  --host 0.0.0.0 --port 8080
```

### Flag notes

| Choice | Why |
| :--- | :--- |
| Omit `-ngl` | Passing `-ngl` with `--fit` + `-c` aborts (`n_gpu_layers already set`). |
| `--load-mode none` | Replaces deprecated `--no-mmap`. |
| `--spec-draft-n-min 0` | Upstream default. n-min larger than draft length silently disables speculation. |
| `-ub 512` | Prefill microbatch on 40 GB. `-ub 256` was an L4 cut. |
| `--repeat-penalty` | The flag name. `--repetition-penalty` is wrong. |
| `--temp 0.7 …` | Overrides GGUF-embedded `general.sampling.temp=1.0`. Qwen non-thinking defaults. Client may send `temperature: 0` for a greedy turn. |
| No `--chat-template-file` | Blackfrost GGUF is fine unless the client injects mid-conversation system messages. |
| `--api-key` / `LLAMA_API_KEY` | Locks the public tunnel. `/health` stays open. Prefer `--env` on `colab exec` so the key is not in `ps`. |

MTP fallback if the sidecar will not load: drop `-md` and use `--spec-type draft-mtp`. Same n-max 5 is DFlash-only; MTP uses its own depth.

Per-request `"speculative.n_max": 0` is **silently ignored**. A no-spec baseline needs `llama-bench` before server start.

---

## 5. Tools

`--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3` are **SGLang / vLLM only**. They abort llama-server with `error: invalid argument`.

llama-server uses `--jinja` PEG autoparser (`Chat format: peg-constructed`). It translates Qwen 3.8 XML:

```xml
<tool_call>
<function=get_disk_usage>
<parameter=mount_path>
/content
</parameter>
</function>
</tool_call>
```

into OpenAI `tool_calls[]`.

When a tool is required, the client sends `"tool_choice": "required"` (QwenLM/Qwen3.8#236). With thinking off, `"tool_choice": "auto"` plus a long system prompt can answer in plain text. `"required"` compiles a GBNF constraint so the first tokens are `<tool_call>`.

Do not pass `--tools all` (that enables the Web UI host agent, not API function calling).

---

## 6. API cheat sheet

```bash
# Health (no API key)
curl -s http://127.0.0.1:8080/health
# {"status":"ok"}

# Models
curl -s http://127.0.0.1:8080/v1/models \
  -H "Authorization: Bearer ${LLAMA_API_KEY}"

# Chat (thinking off is already a server default)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "Write a Python function to parse GGUF header magic bytes."}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }' | jq '{content: .choices[0].message.content, timings: .timings}'
```

Acceptance lives in the completion JSON: `timings.draft_n` and `timings.draft_n_accepted`. `GET /slots` has `is_processing` / `id_task` / `n_decoded` and does **not** include draft counts.

Gate at n_max=5: accept ≥ 50% **or** mean draft length ≥ 3.50.

Tool-call smoke:

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "messages": [
      {"role": "user", "content": "Check the current disk usage of /content in gigabytes."}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_disk_usage",
        "description": "Return disk usage for a mount path",
        "parameters": {
          "type": "object",
          "properties": {
            "mount_path": {"type": "string"},
            "unit": {"type": "string", "enum": ["MB", "GB"]}
          },
          "required": ["mount_path", "unit"]
        }
      }
    }],
    "tool_choice": "required",
    "temperature": 0,
    "max_tokens": 128
  }' | jq '{finish_reason: .choices[0].finish_reason, tool_calls: .choices[0].message.tool_calls}'
```

Pass when `finish_reason` is `"tool_calls"` and `tool_calls[0].function.name` is `"get_disk_usage"`.

---

## 7. Traps

1. Default port is 9931. Always pass `--port 8080`.
2. `-ngl` plus `--fit` plus `-c` aborts.
3. n-max 8 or 9 wastes ~299–599 MiB of GDN state and still drafts 7.
4. `"speculative.n_max": 0` in a request does nothing.
5. Synthetic benches (`ignore-eos` + `min_tokens`) let n-gram copy at fake hundreds of tok/s. Quote one-shot and iterative edits separately.
6. q4 KV hurts speculative acceptance. Keep q8_0 for 32K–100K coding.
7. `-c 262144` still OOMs via lazy CUDA graphs. Leave `--fit-target 768`.
8. Skip DSpark, `ik_llama.cpp`, and stacking `ngram-mod` with k4v.
