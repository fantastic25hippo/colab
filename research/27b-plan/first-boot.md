# First boot

One sitting to compile, quantize, verify logs, and measure A100 40GB speed. Do not start a GPU until asked.

Launch flags: [`inference.md`](inference.md). Session loop: [`ops.md`](../../ops.md).

---

## 0. Before the GPU

- CLI is `google-colab-cli==0.7.4`.
- Drive has ≥ 22 GB free.
- Target GGUF SHA `5d53637a…5509` and BF16 sidecar SHA `ca6dacc0…4fb3` are on `/content/drive/MyDrive/colab_llm/` (or download during this sitting).
- `LLAMA_API_KEY` is in the laptop env.
- Browser profile: `zuluzain5@gmail.com` is `authuser=0`.

---

## 1. Provision

```bash
colab --auth=oauth2 new -s qwen-worker --gpu A100
```

Stop immediately if VRAM ~80 GB, `HIGH_RAM`, or host RAM > 110 GB.

Mount Drive, keep exec attached (`--timeout 86400`).

---

## 2. Compile and quantize

Follow [`inference.md`](inference.md) §2–3. Pin `efe3626`, fatbin `"80-real;120a-real"`, quantize Blackfrost BF16 → Q4_K_M.

Record:

```
sidecar bytes:          (expect 1143006592)
sidecar sha256:
llama-server path:
cmake arch string:      80-real;120a-real
```

Skip `dflash.rope.dimension_sections` injection.

---

## 3. Optional no-spec baseline (before llama-server)

`"speculative.n_max": 0` on a live server is ignored. If you want a true no-spec number:

```bash
DRIVE=/content/drive/MyDrive/colab_llm
"$DRIVE/bin/llama-bench" \
  -m "$DRIVE/Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf" \
  -fa 1 -ctk q8_0 -ctv q8_0 -p 512 -n 128 -r 1
```

Gate: **≥ 50 tok/s**. Scaled healthy band is ~62–75 tok/s.

---

## 4. Start Playbook A

Launch from [`inference.md`](inference.md) §4. Log to a file so grep works.

---

## 5. Log grep (halt conditions)

```bash
LOG_FILE=/tmp/llama-server.log
DRAFT_Q4=/content/drive/MyDrive/colab_llm/dflash-Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf

ACTUAL_BYTES=$(stat -c%s "$DRAFT_Q4")
if [[ "$ACTUAL_BYTES" != "1143006592" && "$ACTUAL_BYTES" != "1143006752" ]]; then
  echo "FAIL: unexpected sidecar size $ACTUAL_BYTES" >&2
  exit 1
fi

# HALT if dense 64/65-layer KV
if grep -Eq "llama_kv_cache: size = .*100000 cells,[[:space:]]+6[45] layers" "$LOG_FILE"; then
  echo "FAIL: dense 64/65-layer KV" >&2
  exit 1
fi

grep -Eq "llama_kv_cache: size = .*100000 cells,[[:space:]]+16 layers.*K \(q8_0\):.*V \(q8_0\):" "$LOG_FILE"
grep -Eq "llama_memory_recurrent: size = .*64 layers" "$LOG_FILE"
grep -Eq "print_info: arch[[:space:]]+= qwen35" "$LOG_FILE"
grep -Eq "print_info: n_layer[[:space:]]+= 64" "$LOG_FILE"
grep -Eq "print_info: arch[[:space:]]+= dflash" "$LOG_FILE"
grep -Fq "common_speculative_impl_draft_dflash: - n_max=5, n_min=0, p_min=0.00" "$LOG_FILE"
grep -Fq "common_speculative_impl_draft_dflash: - block_size=8, mask_token_id=248070, n_extract=5, sample_from_anchor=true" "$LOG_FILE"
grep -Fq "adding speculative implementation 'ngram-map-k4v'" "$LOG_FILE"
grep -Fq "main: server is listening on http://0.0.0.0:8080" "$LOG_FILE"

if grep -Fq "adding speculative implementation 'ngram-mod'" "$LOG_FILE"; then
  echo "FAIL: ngram-mod registered" >&2
  exit 1
fi
```

Also confirm `thinking = 0` and `Chat format: peg-constructed`. Recurrent banner should show **6 seqs** at n-max 5. 8 or 9 seqs means n-max was set too high.

---

## 6. Smoke (≤ 3 min)

Set `LLAMA_URL` (guest `http://127.0.0.1:8080` or the trycloudflare `/v1` origin without the path). Model id is `qwen3.8-27b`.

```bash
nvidia-smi --query-gpu=name,memory.used,memory.total,clocks.sm,clocks.mem,power.draw,power.limit --format=csv
free -g
# expect: NVIDIA A100-SXM4-40GB, memory.used <= 31000 MiB, host RAM ~83 GB

curl -sf "${LLAMA_URL}/health" | jq .
curl -sf -H "Authorization: Bearer ${LLAMA_API_KEY}" "${LLAMA_URL}/slots" \
  | jq '.[0] | {id, n_ctx, speculative, is_processing, id_task, params: {n_max: .params["speculative.n_max"], n_min: .params["speculative.n_min"]}}'
```

**Curl 1 — one-shot spec, cold cache. Gate: `predicted_per_second` ≥ 95 tok/s** (coding prompt only).

```bash
curl -sf "${LLAMA_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "cache_prompt": false,
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [
      {"role": "user", "content": "Write a self-contained Python 3 module implementing an iterative AVL tree with insert, delete, rebalance, and level-order traversal. Code only, no prose."}
    ]
  }' | jq '{
    smoke_step: "curl_1_oneshot",
    predicted_per_second: .timings.predicted_per_second,
    draft_n: .timings.draft_n,
    draft_n_accepted: .timings.draft_n_accepted,
    accept_pct: (if (.timings.draft_n // 0) > 0 then (100.0 * .timings.draft_n_accepted / .timings.draft_n) else 0 end)
  }'
```

If under 95 tok/s with accept ≥ 55%, inspect SM clocks and SKU. Do not apply `GGML_MMVQ_MAX`.

Acceptance gate: `draft_n_accepted / draft_n` ≥ 50% **or** mean draft length ≥ 3.50.

**Curl 2 — follow-up hitting k4v + prefix cache. Gate: tok/s > Curl 1 and `cache_n > 0`.**

```bash
curl -sf "${LLAMA_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "cache_prompt": true,
    "temperature": 0,
    "max_tokens": 256,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [
      {"role": "user", "content": "Write a Python class TokenBucketRateLimiter with methods allow_request(self, cost: int = 1) -> bool, _refill(self) -> None, and snapshot_state(self) -> dict."},
      {"role": "assistant", "content": "class TokenBucketRateLimiter:\n    def allow_request(self, cost: int = 1) -> bool: ...\n    def _refill(self) -> None: ...\n    def snapshot_state(self) -> dict: ...\n"},
      {"role": "user", "content": "Re-emit the complete TokenBucketRateLimiter class verbatim, renaming allow_request to consume_tokens and adding a reset_bucket(self) -> None method."}
    ]
  }' | jq '{
    smoke_step: "curl_2_k4v",
    predicted_per_second: .timings.predicted_per_second,
    cache_n: .timings.cache_n,
    draft_n: .timings.draft_n,
    draft_n_accepted: .timings.draft_n_accepted
  }'
```

Then run the tool-call curl in [`inference.md`](inference.md) §6.

Optional: ~32K prefill probe. Record `prompt_per_second` and decode tok/s.

---

## 7. Record

| Field | Value |
| :--- | :--- |
| GPU name | |
| `memory.used` MiB | (≤ 31,000) |
| Host RAM GB | (~83, < 110) |
| CU/hr | (~5.37) |
| No-spec tok/s | |
| Curl 1 tok/s / accept % | (≥ 95) |
| Curl 2 tok/s / `cache_n` | (> Curl 1, cache_n > 0) |
| Sidecar SHA | |
| WebSocket lasted | (1 h / 2–3.5 h / never dropped) |

Always `colab stop -s qwen-worker` at the end.

---

## 8. Still unknown (measure, do not re-research)

- A100 40GB DFlash 2 + k4v tok/s at short, 32K, and 100K
- Blackfrost Q4 DFlash acceptance at n_max=5
- Local SHA of the quantized sidecar
- Whether `zuluzain5@gmail.com` can assign G4
- Whether this account's attached `colab exec` WebSocket lasts past 1 h / ~2–3.5 h
- q8_0 vs f16 verify delta on this A100 at 100K
