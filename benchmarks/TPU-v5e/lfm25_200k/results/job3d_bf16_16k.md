# Job 3d Report: bf16 Control at 16K (Same Needle, int8 OFF)

## 1. Executive Summary

- **Model Under Test**: `LFM2.5-24B-A2B` (`avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU`).
- **Command Run** (session `tpu-lfm25`, TPU v5e-1):
  ```bash
  cd /content && LFM_ABORT_ON_CACHE_MISS=0 LFM_MAX_MODEL_LEN=16384 python3 lfm25_coldstart_driver.py
  ```
  *(No `LFM_KV_INT8` set; no code edits).*
- **Session Teardown**: `colab --auth=oauth2 stop -s tpu-lfm25` confirmed (`No active sessions found on server`).
- **Verdict**: Under `bf16` KV (`q_bfloat16_kv_bfloat16`), the 16K needle evaluation produces the **exact same failure** as Job 3 (`needle_pass=False`, `output_text = "\n" + "</think>" * 63`). Therefore, the Job 3 `needle_pass=False` failure is **not caused by KV quantization**; it is caused by the thinking chat template (`<think>`) combined with `sp_dec = SamplingParams(max_tokens=64, temperature=0.0, ignore_eos=True)`.

---

## 2. Preflight Checks (P1–P3)

- **P1 PASS** (`MEASURED`): `HEAD` at `a7bcb70`, no tracked changes.
- **P2 PASS** (`MEASURED`): `tpu_v5e_patches.py` sha256 `155db072b3fb75141c836337a35382ff4d3f70fdd505e92bf0495760f2ae87f8`.
- **P3 PASS** (`MEASURED`): `./run_cpu_tests.sh` reports `9 passed in 4.31s`.

---

## 3. RPA Line & Context 16384 Line (`MEASURED`)

### RPA Line (`MEASURED`)
```text
(MainProcess pid=9879) WARNING 11-02 18:53:37 [tuned_block_sizes_hd64.py:456] Couldn`t find tuned sizes for the RPA v3 kernel with ('TPU v5e', 16, 'q_bfloat16_kv_bfloat16', 'q_head-32_kv_head-8_head-64', 'max_model_len-16384-sw-None')
(MainProcess pid=9879) INFO 11-02 18:53:37 [tuned_block_sizes_hd64.py:477] RPA v3 kernel tuned block sizes for ('TPU v5e', 16, 'q_bfloat16_kv_bfloat16', 'q_head-32_kv_head-8_head-64', 'max_model_len-16384-sw-None'): bkv_p=128, bq=32
```
- Also logged for `bq=16` on 64-token chat:
  ```text
  (MainProcess pid=9879) INFO 11-02 18:54:29 [tuned_block_sizes_hd64.py:477] RPA v3 kernel tuned block sizes for ('TPU v5e', 16, 'q_bfloat16_kv_bfloat16', 'q_head-32_kv_head-8_head-64', 'max_model_len-16384-sw-None'): bkv_p=128, bq=16
  ```
- KV Cache Allocation (`MEASURED`): `kv_cache_dtype_resolved = "auto"` (`bfloat16`), `32,768 B/page`, `6,682` attention blocks (`10 layers x 6,682 blocks x 32,768 B = 2.04 GiB`).

### Context 16384 Line (`MEASURED`)
```text
[LFM25_SUBRUNNER 18:54:49] Context 16384 (p=16320, bos=1, echo_bos=1, decode ignore_eos=True): TTFT=1.0359s, Decode=45.58 tok/s, needle_pass=False, HBM={'in_use_gib': 14.735, 'peak_gib': 14.817, 'limit_gib': 15.748}
```
- **First 200 chars of `output_text`** (`MEASURED`, `len(output_text) == 505`, `64` output tokens: `"\n"` + `63 × "</think>"`):
  ```text
  '\n</think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think'
  ```

---

## 4. Thinking Chat Template & Last 40 Decoded Prompt Tokens (`MEASURED` / `INFERRED`)

- **Is the chat template a thinking template?** **Yes (`MEASURED`)**.
  - In `/dev/shm/lfm25_200k/chat_template.jinja`, when `add_generation_prompt=True` is passed, the template appends:
    ```jinja
    {%- if add_generation_prompt -%}
        {{- "<|im_start|>assistant\n<think>" -}}
    {%- endif -%}
    ```
- **Does the prompt end inside `<think>`?** **Yes (`MEASURED`)**.
  - `empty = _chat_ids(tok, "")` ends with `tail_n = 6` tokens (`tail_ids = [124900, 207, 124899, 63514, 207, 124901]`), which decode to:
    `["<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>"]` (`"<|im_end|>\n<|im_start|>assistant\n<think>"`).
  - Therefore, generation starts **immediately inside the open `<think>` block**. On the 16K needle prompt, the model emits an empty reasoning block (`"\n</think>"`), and because `sp_dec` sets `ignore_eos=True` (`lfm25_engine_subrunner.py:274,450`) or repeats the closing tag token `124902` (`</think>`), all 63 remaining tokens are `</think>`.
- **Last 40 Decoded Prompt Tokens (`ids[-40:]`, including `q_ids` and `tail_n=6` region)**:
  - In `lfm25_engine_subrunner.py:311-334`, `build_long_prompt_ids` takes `body = list(full[:16314])` (`MEASURED` on VM), overwrites the last 7 tokens of `body` (`body[16307:16314]`) with `q_ids = _encode_text(tok, "\nWhat is the secret code?")` = `[207, 2992, 355, 278, 6654, 4713, 39]` (`INFERRED` from local `LiquidAI/LFM2.5-24B-A2B/tokenizer.json` + `MEASURED` VM body tokens), and appends `tail_ids` (`tail_n = 6`, `MEASURED` on VM):
  - **Token IDs (`ids[-40:]`)**:
    `[302, 5030, 37933, 1003, 309, 29460, 7216, 22, 42847, 278, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 22, 42847, 278, 9761, 9896, 302, 5030, 37933, 207, 2992, 355, 278, 6654, 4713, 39, 124900, 207, 124899, 63514, 207, 124901]`
  - **Per-Token Decoded List (`40` tokens)**:
    `[" of", " ast", "rophys", "ics", " and", " stellar", " evolution", ".", " Explain", " the", " fundamental", " principles", " of", " ast", "rophys", "ics", " and", " stellar", " evolution", ".", " Explain", " the", " fundamental", " principles", " of", " ast", "rophys", "\n", "What", " is", " the", " secret", " code", "?", "<|im_end|>", "\n", "<|im_start|>", "assistant", "\n", "<think>"]`
  - **Concatenated String of Last 40 Tokens**:
    `' of astrophysics and stellar evolution. Explain the fundamental principles of astrophysics and stellar evolution. Explain the fundamental principles of astrophys\nWhat is the secret code?<|im_end|>\n<|im_start|>assistant\n<think>'`

---

## 5. Root Cause Analysis: Why `kv_int8.py` Produced `float8_e4m3fn` Instead of `int8` (`INFERRED` from Source Inspection)

1. **What `kv_int8.apply()` Attempts to Patch vs. What Actually Happened**:
   - `kv_int8.py:9-16` (`patch_module`) targets `tpu_inference.utils._DTYPE_STR_ALIAS_TO_JAX_DTYPE` (`tpu_inference/utils.py:37-43`), attempting to mutate `d["fp8"] = d["int8"] = d["int8_per_token_head"] = jnp.int8.dtype`.
   - However, `kv_int8.py:72-76` (`_scan_sys_modules`) records every `m_name.startswith("tpu_inference")` in `_SEEN_INT8` **as soon as it appears in `sys.modules`**, without checking whether the module is still mid-import (`getattr(mod.__spec__, "_initializing", False)`).
2. **The Nested-Import Race Condition (`file:line`)**:
   - When `tpu_inference.utils` begins importing, Python places the partially-initialized module object into `sys.modules["tpu_inference.utils"]`.
   - At `tpu_inference/utils.py:27`, before line 37 is reached, `utils.py` executes a nested import:
     `from tpu_inference.layers.common.utils import general_device_put`.
   - That nested import invokes the hooked `builtins.__import__` (`_imp` at `kv_int8.py:87-91`), which calls `_scan_sys_modules()` (`kv_int8.py:72-76`).
   - `_scan_sys_modules()` sees `"tpu_inference.utils"` in `sys.modules`, **adds `"tpu_inference.utils"` to `_SEEN_INT8` (`kv_int8.py:75`)**, and calls `patch_module("tpu_inference.utils", mod)` (`kv_int8.py:76`).
   - Because `tpu_inference/utils.py` is paused at line 27, `_DTYPE_STR_ALIAS_TO_JAX_DTYPE` (`tpu_inference/utils.py:37`) **does not exist yet** (`getattr(mod, "_DTYPE_STR_ALIAS_TO_JAX_DTYPE", None)` at `kv_int8.py:12` returns `None`).
   - When `tpu_inference/utils.py` resumes and executes lines `37-43`, it defines `_DTYPE_STR_ALIAS_TO_JAX_DTYPE = {"fp8": jnp.float8_e4m3fn.dtype, ...}`. Because `"tpu_inference.utils"` is already in `_SEEN_INT8` (`kv_int8.py:74`), `_scan_sys_modules()` skips it on all subsequent imports, leaving `_DTYPE_STR_ALIAS_TO_JAX_DTYPE["fp8"]` unpatched as `jnp.float8_e4m3fn.dtype`.
   - *(Contrast: `tpu_v5e_patches.py:611-612` explicitly checks `if getattr(spec, "_initializing", False): continue` before marking a module as seen).*
3. **Which Dict/Module the KV Allocator & RPA Kernel Read (`file:line`)**:
   - `tpu_inference/runner/tpu_runner.py:842` calls `self.kv_cache_dtype = to_torch_dtype(cache_dtype)` -> `to_jax_dtype("fp8")` (`tpu_inference/utils.py:108-112`), which reads the unpatched `tpu_inference.utils._DTYPE_STR_ALIAS_TO_JAX_DTYPE["fp8"]` (`jnp.float8_e4m3fn.dtype` -> `torch.float8_e4m3fn`).
   - `tpu_inference/runner/kv_cache_manager.py:149,1002` creates the JAX KV cache arrays with `cache_dtype = t2j_dtype(layer_spec.dtype)` = `jnp.float8_e4m3fn`.
   - The RPA v3 kernel (`tpu_inference/kernels/ragged_paged_attention/v3/kernel_hd64.py:1455`) and backend (`tpu_inference/layers/vllm/backends/flash_attn.py:127`) read `kv_cache.dtype` directly from the allocated KV cache tensor (`jnp.float8_e4m3fn`) and pass it to `get_tuned_block_sizes` (`tpu_inference/kernels/ragged_paged_attention/v3/tuned_block_sizes_hd64.py:456,477`), formatting `f'q_{q_dtype}_kv_{kv_dtype}'` -> `'q_bfloat16_kv_float8_e4m3fn'`.
