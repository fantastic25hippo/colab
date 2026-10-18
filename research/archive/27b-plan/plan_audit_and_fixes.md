# Colab Qwen 3.8 27B Worker Plan: Technical Audit, Performance Leaks, and Required Doc Fixes

**Target Repository:** Colab Qwen Worker (`README.md`, `architecture.md`, `inference.md`, `ops.md`, `first-boot.md`, `bend-hands.md`)  
**Target Stack:** [Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF](https://huggingface.co/Blackfrost-AI/Qwen3.8-27B-ABLITERATED-GGUF) (`Q4_K_M`) + Quantized Blackfrost DFlash 2 (`Q4_K_M`) + `ngram-map-k4v` on [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) (`efe3626` / `b11552`)  
**Target Runtime:** Google Colab `NVIDIA A100-SXM4-40GB` via [google-colab-cli](https://github.com/googlecolab/google-colab-cli) `0.7.4` on [Google AI Pro](https://support.google.com/googleone/answer/16287313) (200 CU/month)

## 1. Architecture and Economics Verdict: The Core Plan Is Sound

The foundational design across the six root documents is well-reasoned and avoids the most expensive Google Colab and `llama.cpp` operational traps:

| Architectural Decision | Locked Setting in Root Docs | Why It Holds Up Under Audit |
| :--- | :--- | :--- |
| **Primary GPU SKU** | `colab new -s qwen-worker --gpu A100` (Standard 40 GB, ~5.37 CU/hr) | Delivers 97.4% of Colab G4 Server Edition's memory bandwidth (`1,555 GB/s` vs `1,597 GB/s`) at 61.7% of the hourly CU burn (`~5.37` vs `~8.71 CU/hr`), fitting 100K context in `~24.1–27.6 GiB` (`≤ 31,000 MiB` gate). |
| **Speculative Stack** | `--spec-type draft-dflash,ngram-map-k4v` with `--spec-draft-n-max 5` | Caps draft depth at the `n=5` coding throughput sweet spot (`187.13 tok/s` vs `168.06 tok/s` at `n=7`), avoids the `need_n_rs_seq()` GDN recurrent-state VRAM waste at `n >= 8`, and omits `ngram-mod` (which drops speed ~19.4% when stacked with `k4v`). |
| **Ampere Kernel Routing** | `KEEP_DELETED` for `GGML_MMVQ_MAX` and `LLAMA_SCHED_POOL` | Stock `MMVQ` (`mul_mat_vec_q`) on A100's 108 SMs (`sm_80`, 400 W TDP) beats forced `MMQ` by ~20% at verification width `ne11 = 6` (`n_max = 5`), and Blackfrost `Q4_K_M` contains zero `IQ4_XS` tensors. |
| **Sidecar Provenance** | `KEEP_QUANTIZE_BLACKFROST` (`BF16` → `Q4_K_M`, `1,143,006,592` B) | Quantizing `dflash-Qwen3.8-27B-ABLITERATED-BF16.gguf` locally produces 81 quantized tensor blocks bit-identical to `incoai` `Q4_K_M` while preserving Blackfrost metadata. Skipping `dflash.rope.dimension_sections` injection is safe because `required = false` and 1D NeoX RoPE equals text-only M-RoPE `[11, 11, 10, 0]`. |
| **Session Survival & Idle Reaping** | Attached `colab exec --timeout 86400` + 20–25 s stdout heartbeat + 1800 s `/slots` watchdog | Prevents both the ~10–15 minute disconnected-client VM reaper on AI Pro and the ~80 s Tunnel Frontend silent-stream freeze, while resetting idle time on `(id_task, n_decoded, n_prompt_tokens_processed, n_prompt_tokens)` transitions. |

## 2. Four Breaking Bugs in the Current Root Docs block First Boot and Reconnect

Before starting a GPU sitting or writing `start_server.py`, an agent updating this repository must fix four concrete bugs in `inference.md`, `first-boot.md`, and `ops.md`.

### 2.1 Missing `-DBUILD_SHARED_LIBS=OFF` in `inference.md` §2 Breaks Warm-Boot Binary Caching

* **The Bug:** `inference.md` §2 runs `cmake -B build` without `-DBUILD_SHARED_LIBS=OFF` and instructs the operator to copy only `build/bin/llama-server` and `build/bin/llama-quantize` to `/content/drive/MyDrive/colab_llm/bin/`. By default, `llama.cpp` builds `libllama.so`, `libggml.so`, `libggml-base.so`, and `libggml-cuda.so` as shared libraries inside `build/bin/`. On the next Colab VM (warm boot), executing the cached `llama-server` binary will immediately crash with `error while loading shared libraries: libllama.so: cannot open shared object file: No such file or directory`.
* **Required Fix in `inference.md` §2:** Add `-DBUILD_SHARED_LIBS=OFF` to the CMake configuration so `llama-server`, `llama-quantize`, and `llama-bench` link `ggml` and `llama` statically into standalone executables (only dynamically linking against container system libraries `libc.so.6`, `libcuda.so.1`, `libcudart.so`, `libcublas.so`, and `libcublasLt.so`).

### 2.2 Missing `llama-bench` Target in `inference.md` §2 Breaks `first-boot.md` §3

* **The Bug:** `first-boot.md` §3 runs `"$DRIVE/bin/llama-bench"` to measure the true no-spec baseline (`≥ 50 tok/s` gate) before starting `llama-server` (since `llama-server` silently ignores per-request `"speculative.n_max": 0`). However, `inference.md` §2 only builds `--target llama-server llama-quantize` and never copies `llama-bench` to `$DRIVE/bin/`.
* **Required Fix in `inference.md` §2:** Change the build target list to `--target llama-server llama-quantize llama-bench` and copy all three binaries to `/content/drive/MyDrive/colab_llm/bin/`.

```bash
# Corrected compile block for inference.md §2
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout efe3626

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="80-real;120a-real" \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF

cmake --build build --config Release -j "$(nproc)" --target llama-server llama-quantize llama-bench

mkdir -p /content/drive/MyDrive/colab_llm/bin
cp build/bin/llama-server build/bin/llama-quantize build/bin/llama-bench /content/drive/MyDrive/colab_llm/bin/
```

### 2.3 Truncated Assistant Stub in `first-boot.md` §6 Breaks Both Curl 2 Gates (`cache_n > 0` and `ngram-map-k4v`)

* **The Bug (Part A — `ngram-map-k4v` Starvation):** When `first-boot.md` §6 condensed the Curl 2 payload from `a100_speed_and_session.md`, it replaced the 28-line `TokenBucketRateLimiter` implementation inside the `"assistant"` message with 4 stub lines ending in `...`. Because `ngram-map-k4v` requires matching 4-gram keys with at least 12-token context spans (`--spec-ngram-map-k4v-size-n 12` default) in the existing prompt to draft code edits, replacing the class body with `...` leaves `ngram-map-k4v` nothing to copy when the user asks to re-emit the complete class verbatim.
* **The Bug (Part B — `cache_n > 0` Prefix Cache Miss):** Curl 1 sends an AVL tree prompt with `"cache_prompt": false`. Curl 2 then sends a single request with a brand-new `TokenBucketRateLimiter` conversation that has never been cached in slot 0. Because slot 0 has no prior `TokenBucketRateLimiter` tokens cached, `timings.cache_n` will be `0` (or at most the ~15-token system header if a prior request cached it).
* **Required Fix in `first-boot.md` §6:** Restore the full 28-line `TokenBucketRateLimiter` implementation in the `"assistant"` turn of Curl 2, and precede Curl 2 with a fast seed turn (`max_tokens: 1`, `"cache_prompt": true`) or evaluate `cache_n > 0` on a back-to-back follow-up turn sharing the exact prefix.

```bash
# Corrected Curl 2 block for first-boot.md §6 (seeds prefix, then tests k4v + cache_n > 0)
# Step 2a: Seed the prefix cache in slot 0
curl -sf "${LLAMA_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "qwen3.8-27b",
    "cache_prompt": true,
    "temperature": 0,
    "max_tokens": 1,
    "chat_template_kwargs": {"enable_thinking": false},
    "messages": [
      {"role": "user", "content": "Write a Python class TokenBucketRateLimiter with methods allow_request(self, cost: int = 1) -> bool, _refill(self) -> None, and snapshot_state(self) -> dict."}
    ]
  }' > /dev/null

# Step 2b: Follow-up turn hitting both prefix cache (cache_n > 0) and ngram-map-k4v
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
      {"role": "assistant", "content": "```python\nimport time\n\nclass TokenBucketRateLimiter:\n    def __init__(self, capacity: int, refill_rate: float) -> None:\n        self.capacity = capacity\n        self.refill_rate = refill_rate\n        self.tokens = float(capacity)\n        self.last_refill = time.monotonic()\n\n    def _refill(self) -> None:\n        now = time.monotonic()\n        elapsed = now - self.last_refill\n        if elapsed > 0.0:\n            self.tokens = min(float(self.capacity), self.tokens + elapsed * self.refill_rate)\n            self.last_refill = now\n\n    def allow_request(self, cost: int = 1) -> bool:\n        self._refill()\n        if self.tokens >= cost:\n            self.tokens -= cost\n            return True\n        return False\n\n    def snapshot_state(self) -> dict:\n        self._refill()\n        return {\"capacity\": self.capacity, \"tokens\": self.tokens, \"refill_rate\": self.refill_rate}\n```"},
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

### 2.4 Child Process Pipe/Session Handling in `ops.md` Will Kill `llama-server` on WebSocket Drop

* **The Bug:** `ops.md` specifies that `start_server.py` starts `llama-server` and `cloudflared` as children, and if the `colab exec` WebSocket drops after 1–3.5 hours, `attach_heartbeat.py` reconnects without reloading weights. However, if `start_server.py` spawns those processes without `start_new_session=True`, or attaches their `stdout`/`stderr` to `subprocess.PIPE` (for example, to parse the `trycloudflare.com` URL from `cloudflared`'s `stderr`), then when `start_server.py` detaches or stops draining the pipe, the 64 KB Linux pipe buffer fills up or closes—delivering `SIGPIPE` or `SIGHUP` and killing `llama-server` or `cloudflared`. Additionally, `llama-server` writes its startup and slot logs to `stderr`, so redirecting only `stdout` to `/tmp/llama-server.log` will cause the `first-boot.md` §5 log greps to fail.
* **Required Fix in `ops.md` and `start_server.py`:**
  1. Spawn both `llama-server` and `cloudflared` with `start_new_session=True`.
  2. Redirect `llama-server` `stdout` and `stderr` (`stderr=subprocess.STDOUT`) directly to a disk file (`/tmp/llama-server.log`), never to `subprocess.PIPE`.
  3. Redirect `cloudflared` `stdout` and `stderr` directly to `/tmp/cloudflared.log`, poll `/tmp/cloudflared.log` from Python to extract the `https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com` URL, wait 5–8 seconds for DNS propagation, and persist the URL to `/tmp/tunnel_url.txt` so `attach_heartbeat.py` can re-emit `TUNNEL_READY_URL=...` upon reconnect.
  4. Only terminate `llama-server` and `cloudflared` when the 1800 s idle watchdog or 4-hour hard cutoff fires right before `runtime.unassign()`, **not** on a normal `start_server.py` exit/disconnect.

## 3. Three Performance and CU Leaks to Plug

### 3.1 Server-Default `--temp 0.7` Degrades Speculative and N-Gram Coding Speed by 20%–35%

* **Where We Lose Speed:** `inference.md` §4 sets `--temp 0.7 --top-p 0.8 --top-k 20` as the server default (`README.md` Open Preference #4), while `bend-hands.md` §3 (`HERMES_CONFIG=/tmp/hermes-colab.yaml`) does not set `temperature: 0`. Whenever Hermes or another client omits `temperature` in a request, `llama-server` uses stochastic sampling (`temp=0.7`). Under stochastic sampling, `llama.cpp` rejection sampling rejects draft tokens whenever $u > p_{\text{target}}(x) / p_{\text{draft}}(x)$, which severely penalizes `ngram-map-k4v` (a draftless exact-token lookup) and drops DFlash 2 acceptance at `n_max=5` from `~70.4%` (`mean len 4.52` at `temp=0`) down toward `~40%–46%` (`mean len ~3.0`).
* **Actionable Fix:** Because this worker only executes deterministic coding and formal proof tasks with thinking off (`--reasoning off`), change `--temp 0.7` to **`--temp 0.0`** in `inference.md` §4 (and explicitly add `temperature: 0.0` to `/tmp/hermes-colab.yaml` in `bend-hands.md` §3). Any client that genuinely wants non-greedy sampling on a specific turn can still pass `"temperature": 0.7` in the request JSON.

### 3.2 Executing Binaries Directly Off `/content/drive` FUSE (`"$DRIVE/bin/llama-server"`)

* **Where We Lose Speed / Stability:** `inference.md` §4 and `first-boot.md` §3 execute `"$DRIVE/bin/llama-server"` and `"$DRIVE/bin/llama-bench"` directly from the DriveFS FUSE mount (`/content/drive/MyDrive/colab_llm/bin/`). While pointing `-m` and `-md` at Drive with `--load-mode none` is fast (sequential read into VRAM), executing ELF binaries directly over DriveFS FUSE risks `Permission denied` (`noexec` mount flags) and FUSE page-fault stalls during execution.
* **Actionable Fix:** In `start_server.py` and `first-boot.md`, always `shutil.copy2()` `llama-server`, `llama-bench`, and `cloudflared-linux-amd64` from `/content/drive/MyDrive/colab_llm/bin/` to local `/usr/local/bin/`, set `chmod 0o755`, and run `/usr/local/bin/llama-server --list-devices` (1.5 s check) before loading weights to confirm the cached binary is compatible with the container's CUDA runtime.

### 3.3 Leaving 12.6 GiB of A100 VRAM Idle Instead of Testing `-ctk f16 -ctv f16`

* **Where We Might Lose Speed:** `inference.md` §4 locks `-ctk q8_0 -ctv q8_0` (`3.24 GiB` target KV cache at `-c 100000`), bringing total A100 40GB VRAM usage to `~24.1 GiB` and leaving `~15.5 GiB` of HBM2 completely unused. However, during width-6 speculative verification (`n_max=5`), `llama.cpp`'s CUDA MMA FlashAttention kernel (`fattn-mma-f16`) requires `f16` KV tiles and must dequantize `q8_0` KV into an `f16` scratch buffer on the fly (`llama.cpp#27623`, `#29371`). Switching to `-ctk f16 -ctv f16` (`6.10 GiB` target KV cache across Qwen 3.8's 16 full-attention layers) raises total VRAM usage to `~27.0 GiB` (`27,604 MiB`)—which still passes the `≤ 31,000 MiB` gate and leaves `+12.63 GiB` of free VRAM (16× the `--fit-target 768` MiB reserve).
* **Actionable Fix:** During the first sitting (`first-boot.md` §8), run the optional 32K prefill/decode probe once with `-ctk q8_0 -ctv q8_0` and once with `-ctk f16 -ctv f16`. If `f16` KV improves verify speed or draft acceptance at 32K+ while staying `≤ 31,000 MiB`, promote `-ctk f16 -ctv f16` to the A100 40GB default.

## 4. Three Workflow Blind Spots in `inference.md` and `bend-hands.md`

### 4.1 Unpatched Jinja `raise_exception` Will Crash Hermes on Mid-Conversation System Messages

* **The Blind Spot:** As proven in `sidecar_and_verify.md` §5.3, Blackfrost's `Qwen3.8-27B-ABLITERATED-Q4_K_M.gguf` retains upstream Qwen 3.8's strict Jinja assertion: `{{- raise_exception('System message must be at the beginning.') }}`. `inference.md` §4 omits `--chat-template-file` on the assumption that `messages[0]` is the only system message. However, in `bend-hands.md` §3, Hermes runs with `compression.enabled: true` (`threshold: 0.90`) and receives compiler audit nudges from Gemini. Many CLI agent harnesses inject context-compaction summaries or tool-error reminders as mid-conversation `"role": "system"` messages after `messages[0]`, which will cause `llama-server` to return **HTTP 500** mid-session.
* **Actionable Fix:** On first boot, extract the Jinja template from the GGUF (or save a patched `qwen3.8-fixed.jinja` to `/content/drive/MyDrive/colab_llm/qwen3.8-fixed.jinja` with the `raise_exception('System message must be at the beginning.')` clause removed) and pass `--chat-template-file "$DRIVE/qwen3.8-fixed.jinja"` in `inference.md` §4.

### 4.2 Hardcoding `"tool_choice": "required"` on Every Hermes Turn Prevents Task Completion

* **The Blind Spot:** `inference.md` §5 and `bend-hands.md` §3 note that passing `"tool_choice": "required"` works around `QwenLM/Qwen3.8#236` (where `enable_thinking=false` plus a long system prompt causes `"tool_choice": "auto"` to reply in plain text instead of calling a tool). However, in `llama-server`, `"tool_choice": "required"` compiles a GBNF grammar that forces the first generated token of the turn to be `<tool_call>`. If Hermes sends `"tool_choice": "required"` on **every** turn of an autonomous loop, the model is grammar-blocked from ever emitting a final plain-text completion once `main.bend` and `PROOF.bend` pass!
* **Actionable Fix:** Document in `bend-hands.md` §3 that `"tool_choice": "required"` must only be used either: (a) when Hermes exposes a terminal tool (such as `finish_task` / `submit`) that the model calls to end the loop, or (b) on Turn 1 and as an automatic retry override when a turn with `"tool_choice": "auto"` returns plain text before all 4 `bend` audit gates pass.

### 4.3 Solving Inductive Proofs (`PROOF.bend`) with `--reasoning off` Requires Diagnostic Audit Nudges

* **The Blind Spot:** In `bend-hands.md` §1 and §5, Grok writes `LAWS.bend` and stops *before* `main.bend` is written, Gemini is strictly forbidden from editing project files, and Hermes + Qwen 3.8 27B writes both `main.bend` and `PROOF.bend` with `--reasoning off` (`enable_thinking: false`). While `--reasoning off` is optimal for code generation when the orchestrator has already planned the logic, formal inductive proofs in `PROOF.bend` often fail on subtle base/inductive step mismatches or affine variable linearity (`+x` vs `-x`) that only surface after `main.bend` is written. With `<think>\n\n</think>` forced empty, Qwen 3.8 27B has zero chain-of-thought tokens to work out a multi-step proof repair on its own and can burn CUs looping on the same compiler error.
* **Actionable Fix:** Add an explicit rule in `bend-hands.md` §5: while Gemini is forbidden from editing `main.bend` or `PROOF.bend`, when `bend main.bend --checkup` or `bend PROOF.bend` fails, **Gemini must do the reasoning in its audit nudge**—analyzing the compiler traceback and handing Hermes the exact inductive step structure or lemma signature in natural language, rather than merely pasting raw stderr back to a non-thinking worker.

## 5. Summary Checklist of File Edits for Your Agent

| File to Edit | Section | Exact Change to Apply |
| :--- | :--- | :--- |
| `README.md` | `## Open preferences` | Lock Preference #4 to `--temp 0.0` on the server for greedy coding/proof acceptance, and lock Preference #2 to `LLAMA_API_KEY` in laptop env passed via `colab exec --env`. |
| `inference.md` | `## 2. Compile` | Add `-DBUILD_SHARED_LIBS=OFF` to `cmake -B build`, add `llama-bench` to `--target`, and copy all three binaries (`llama-server`, `llama-quantize`, `llama-bench`) to `/content/drive/MyDrive/colab_llm/bin/`. |
| `inference.md` | `## 4. Launch` | Copy `llama-server` to `/usr/local/bin/llama-server` before launch, change `--temp 0.7` to `--temp 0.0`, add `--chat-template-file "$DRIVE/qwen3.8-fixed.jinja"` (with `raise_exception` stripped), and redirect output via `> /tmp/llama-server.log 2>&1`. |
| `ops.md` | `## Heartbeat and reconnect` | Specify that `start_server.py` must spawn `llama-server` and `cloudflared` with `start_new_session=True` and file-backed logs (`/tmp/llama-server.log`, `/tmp/cloudflared.log`, `/tmp/tunnel_url.txt`) rather than `subprocess.PIPE`, so children survive WebSocket drops. |
| `first-boot.md` | `## 6. Smoke` | Restore the full 28-line `TokenBucketRateLimiter` implementation in Curl 2's assistant message and add the 1-token prefix-seed request before Curl 2 so both `cache_n > 0` and `ngram-map-k4v` are genuinely tested. |
| `first-boot.md` | `## 8. Still unknown` | Add the 60-second A/B test comparing `-ctk q8_0 -ctv q8_0` vs `-ctk f16 -ctv f16` at 32K/100K on A100 40GB. |
| `bend-hands.md` | `## 3. GPU stack` & `## 5. File ownership` | Add `temperature: 0.0` to `/tmp/hermes-colab.yaml`, clarify that `"tool_choice": "required"` requires either a terminal tool or retry-only fallback, and instruct Gemini to include the proof/induction diagnosis in its compiler audit nudges. |
