# LFM2.5-24B-A2B on TPU v5e — load/run audit

HEAD `ede7172e11174cd73d4ea29cfed12f3c8bf8cf73` (matches `ede7172`, ahead of `origin/main` by 1). Read-only except this file. No runtime started.

Labels: **CODE** = this repo. **MEASURED** = `run_results.json` / `RESULTS.md` / a file read here. **UPSTREAM** = vLLM tag `v0.32.0` source. **INFERRED** = not logged.

## 1. Model identity

- Logical id `LiquidAI/LFM2.5-24B-A2B`. Weights actually loaded: `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` into `/dev/shm/lfm25_200k` (`lfm25_coldstart_driver.py:80-81`). `snapshot_download` passes no `revision`.
- Hub main at audit time is `accb27a6180bb9ba5bbc9690b2abe4a526cdb8cd`, `lastModified` 2026-10-31T14:02:01Z, 17 files (9 `model-*.safetensors` plus index, config, tokenizer, `chat_template.jinja`, `generation_config.json`). That timestamp is before both runs, so both pulls are the same tree unless main moved and moved back. The run JSON does not record a SHA. **INFERRED** equality.
- Built by `repack_lfm25_gguf` from `LiquidAI/LFM2.5-24B-A2B-GGUF` file `LFM2.5-24B-A2B-QAD-Q4_0.gguf` (`repack_guest_driver.py:32-52`). Repack uploads shards, index, and `config.json` only (`repack_checkpoint.py:113-117`). Tokenizer sidecars are on the Hub repo and are not produced by this script.
- dtype: config `bfloat16` (`config_builder.py:95-96`; Hub `config.json`). `LLM()` does not pass `dtype` (`lfm25_engine_subrunner.py:54-59`), so vLLM `dtype="auto"` (**UPSTREAM** `vllm/entrypoints/llm.py:191`).
- Quantization: compressed-tensors, not an `LLM(quantization=)` arg. Backbone W8A8 int8 (channel weights, dynamic per-token activations). MoE experts W4A16 symmetric group 128. Embeddings, norms, conv kernel, router, `lm_head` ignored (`config_builder.py:25-64`). Source GGUF is Q4_0 (group 32) dequantized to bf16 then re-quantized (`repack_checkpoint.py:99-111`).
- Tokenizer: Hub `tokenizer.json` + `tokenizer_config.json` (`tokenizer_class` `TokenizersBackend`, bos `<|startoftext|>` id 124894). `trust_remote_code=True` explicit (`lfm25_engine_subrunner.py:58`). Written config has no `auto_map`, so no custom modeling file is requested by this repo.
- Official LiquidAI `config.json` matches architecture and `layer_types` (attn at 2,6,…,38) and `rope_theta` 1e7, but `max_position_embeddings` is **131072**. Ours is **200000** (`config_builder.py:100`). `transformers_version` in the config is `5.12.1`.

## 2. Launch chain

1. `run_lfm25_tpu.py` → `colab usage`, `colab new -s tpu-lfm25 --tpu v5e1`, then `colab exec --timeout 86400 -f lfm25_coldstart_driver.py` with `T0_GLOBAL`, three base64 blobs, `HF_TOKEN` (`run_lfm25_tpu.py:33-47`).
2. Driver writes `/content/lfm25_engine_subrunner.py`, `tpu_v5e_patches.py`, `k8v4_pallas_kernel.py` (`lfm25_coldstart_driver.py:20-26`). The kernel file is never imported.
3. Driver: remount `/dev/shm` exec; download `venv312_cpu.tar.zst`, `tpu_compile_cache_b1_32k.tar`, `zstd_bin` from `avgrocketman/tpu-compile-cache`; unpack; `snapshot_download` weights (`:51-91`). If `layer_types` is missing it writes the 10-attn pattern (`:85-87`). Hub config already has it, so that write does not run.
4. Subprocess: `/dev/shm/vllm_env/bin/python -u /content/lfm25_engine_subrunner.py` (`:115-116`).
5. Subrunner calls `install_all_patches()` then `LLM(...)` (`lfm25_engine_subrunner.py:10-13`, `:54-59`). `VLLM_ENABLE_V1_MULTIPROCESSING=0`, so no EngineCore child. `LLM()` is the engine.
6. Host `finally`: `colab stop`, `sessions`, `usage` (`run_lfm25_tpu.py:64-71`).

`k8v4_pallas_kernel.py` is not on this path. Attention is stock tpu-inference with `kv_cache_dtype=fp8`.

## 3. Engine arguments

Passed in `LLM()` (`lfm25_engine_subrunner.py:54-59`):

| Arg | Value | Mark |
| :--- | :--- | :--- |
| `model` | `/dev/shm/lfm25_200k` | explicit |
| `max_model_len` | 200000 | explicit |
| `max_num_seqs` | 1 | explicit (vLLM LLM-class default would be 256; TPU v5e batched-token default 1024 is overridden) |
| `max_num_batched_tokens` | 8192 | explicit |
| `kv_cache_dtype` | `fp8` | explicit (default `"auto"`) |
| `gpu_memory_utilization` | 0.92 | explicit; equals v0.32 `LLM` default |
| `enable_prefix_caching` | False | explicit (CacheConfig default is True) |
| `trust_remote_code` | True | explicit (default False) |

Not passed, v0.32 `LLM`/`EngineArgs` defaults (**UPSTREAM** `llm.py:178-227`, `config/cache.py:72-141`, `config/model.py:193-272`, `config/parallel.py:141-146`, `engine/arg_utils.py:2987-2994`):

- `dtype` `"auto"` → config bf16, then the quant config. `quantization` None.
- `tensor_parallel_size` 1, `data_parallel_size` 1, `pipeline` 1.
- `enforce_eager` False. `seed` 0 (engine). Sampling seed stays None.
- `block_size` None → `CacheConfig.DEFAULT_BLOCK_SIZE` 16. **MEASURED** narrative “13,361 attn blocks, 213,724 tokens” (`RESULTS.md:46`) is 13361×16=213776, 52 tokens off, so 16 is the default, not a logged field.
- `enable_chunked_prefill` None, then `model_config.is_chunked_prefill_supported`. Not printed. Short prompt is under 8192 either way. Long prompts chunk only if that default is true. **INFERRED** for Lfm2Moe.
- `compilation_config` default. `LLM.__init__` sets `disable_log_stats=True` when omitted (`llm.py:226-227`).
- `mamba_cache_dtype` `"auto"`. **MEASURED** conv state stayed bf16 `(9, 2, 2048)` (`RESULTS.md:46`).
- No `chat_template=` argument (default None). `generate` does not apply one (section 6).

Token buckets are not an `LLM` arg. The patch replaces them (section 5).

## 4. Environment

Driver → subrunner (`lfm25_coldstart_driver.py:106-112`): `MODEL_IMPL_TYPE=vllm`, `TPU_SKIP_MDS_QUERY=1`, `TPU_TOPOLOGY=1x1`, `TPU_HOST_BOUNDS=1,1,1`, `TPU_CHIPS_PER_HOST_BOUNDS=1,1,1`, `TPU_WORKER_ID=0`, `TPU_WORKER_HOSTNAMES=localhost`, `GCE_METADATA_HOST/IP/ROOT=127.0.0.1:1`, `NO_GCE_CHECK=true`, `JAX_PLATFORMS=tpu,cpu`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `VLLM_LOGGING_LEVEL=INFO`, `PYTHONPATH=/content:…`.

Also in the driver process: `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_HIGH_PERFORMANCE=1` (`:45-46`). `HF_TOKEN` is passed into `colab exec` and written to the Hub token files (`run_lfm25_tpu.py:45`, driver `:28-31`). Value not recorded here. `SUBRUNNER_B64` / `PATCHES_B64` / `PALLAS_B64` are popped before the subrunner (`:104`).

Subrunner (`lfm25_engine_subrunner.py:15-20`): `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_XLA_CACHE_PATH=/dev/shm/tpu_compile_cache`, `JAX_COMPILATION_CACHE_DIR=/dev/shm/tpu_compile_cache`, `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`, `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1`, `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all`.

No `XLA_FLAGS`, `JAX_PLATFORMS` override inside the subrunner, or `VLLM_USE_V1`. The restored cache tarball is the Qwen `tpu_compile_cache_b1_32k.tar`, not an LFM cache (`lfm25_coldstart_driver.py:56-57`).

## 5. Patches (`tpu_v5e_patches.py`)

`install_all_patches` (`:234-237`) swaps `builtins.__import__`, installs a log filter, and walks `sys.modules`. The subrunner swallows a failed import (`lfm25_engine_subrunner.py:10-13`). Nov 1 checks show the filter and `modules` fix ran, so the import succeeded.

- `reference_short_conv_forward` (`:14-30`): CPU reference. Not installed.
- `sd_causal_conv` (`:32-39`): depthwise L=3 conv, state layout `(L-1, D)`. Used by the replacement forward. **Conv.**
- `_as_torch` / `_resolve_attn_metadata` (`:41-60`): torchax view and `get_forward_context().attn_metadata`. **Reads attention metadata for the conv cache.**
- `_tpu_xla_short_conv_forward` (`:62-81`): replaces `ShortConv.forward`. Gate, SD conv, SiLU, write state back into `wctx.kv_caches`. **Hybrid conv layers. Not the attention kernel.**
- `_lookup_conv_cache` (`:83-130`): slot from `state_indices_tensor` or `mamba_state_indices`, else slot 0. `n` from `query_start_loc`, else `T` (padded graph length). Zeros `prev` unless `has_initial_states*` is true **and** `seq_lens > n`. **Conv cache. Wrong `n` or `keep` drops or replays conv state.**
- `_wrap_paddings` / `_patch_padding_modules` (`:134-157`): if the stock list is longer than 2, return `{16, 4096, max_token_size}`. **Compile shape, not attention math.**
- `_DropEmptyScheduleFilter` (`:159-170`): drops the log line `Should not schedule a request that does nothing!`. No scheduling change.
- `_patch_vllm_model_wrapper` (`:172-176`): `VllmModelWrapper.modules` → `self.model.modules()`. Atexit only.
- `_patch_loaded_module` (`:178-218`): installs the conv forward; forces `supports_kv_cache_dtype` True for `fp8`/`int8`/`k8v4` (**KV dtype gate**); `ragged_paged_attention.v3.static_validate_inputs` swallows dtype `ValueError` (**attention validation**); `get_tuned_block_sizes` on int8/uint8 falls back to bf16 sizes (**KV block choice**); `_precompile_structured_decoding` becomes a no-op.
- `k8v4_pallas_kernel.py` is never called. The K8V4 path in `research/10-LFM2.5-24B-A2B.md:104-106` is not this run.

## 6. Prompt

Short prompt is the raw string at `lfm25_engine_subrunner.py:63`, passed to `llm.generate`, not `llm.chat`. **UPSTREAM** `generate` → `_run_completion` (`llm.py:418-482`). Chat template is applied only by `chat()` (`llm.py:617`). The template’s generation prompt is `<|im_start|>assistant\n<think>` and it prepends `bos_token` (`chat_template.jinja:1`, `:123-124`). That string is not built.

Completion tokenization uses `add_special_tokens=True` (**UPSTREAM** `vllm/renderers/base.py:486-490`). Chat tokenization uses False (`:501-505`) because the template already inserts BOS. The json post-processor is ByteLevel with `add_prefix_space`, and it does **not** insert BOS. Offline encode of the raw prompt with that `tokenizer.json` (no HF wrapper) is 12 ids, no 124894:

`[89509, 278, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 22]`

tokens: `Explain`, `Ġthe`, `Ġfundamental`, `Ġprinciples`, `Ġof`, `Ġast`, `rophys`, `ics`, `Ġand`, `Ġstellar`, `Ġevolution`, `.`

**MEASURED** from the Hub tokenizer file, not from the TPU log. The run never prints ids. Whether vLLM’s HF wrapper then prepends BOS 124894 was not executed here. The template is not also applied, so a second BOS from the template does not happen. `tok.encode((p_short+" ")*25000)` (`:82`) is a separate call used only for the 32K/200K sweeps.

## 7. Sampling

`SamplingParams` (**CODE** `:64`, `:71`). **UPSTREAM** defaults from `vllm/sampling_params.py` at tag v0.32.0. `temperature < 1e-5` forces greedy and sets `top_p=1`, `top_k=0`, `min_p=0` (`__post_init__`).

| Field | TTFT call | 64-tok call | Default |
| :--- | :--- | :--- | :--- |
| `temperature` | 0.0 explicit | 0.0 explicit | 1.0 |
| `top_p` / `top_k` / `min_p` | forced 1 / 0 / 0 | same | 1.0 / 0 / 0.0 |
| `repetition_penalty` | 1.0 default | 1.0 default | 1.0 (not cleared by greedy) |
| `max_tokens` | 1 explicit | 64 explicit | 16 |
| `min_tokens` | 1 explicit | 64 explicit | 0 |
| `ignore_eos` | True explicit | True explicit | False |
| `stop` / `stop_token_ids` | empty | empty | None → `[]` |
| `seed` | None | None | None |

`generation_config.json` on the checkpoint says `temperature` 0.1, `top_k` 50, `repetition_penalty` 1.05. `update_from_generation_config` only copies EOS ids, and `ignore_eos=True` skips even that (`sampling_params.py:772-800`). Card params are not in effect. Official card also wants the chat template.

## 8. Versions

**MEASURED** in `RESULTS.md:9` for this run: vllm-tpu 0.32.0, tpu-inference 0.32.0, torch 2.10.0+cpu, Python 3.12. The driver unpacks `venv312_cpu.tar.zst`. That artifact’s build note (`tpu-final/RUN1-VENV.md:29-34`) also pins jax 0.11.2, jaxlib 0.11.2, libtpu 0.0.48, transformers 4.57.1. The LFM log does not reprint jax/libtpu/transformers.

Mismatch with the model card / config: `transformers_version` 5.12.1 vs venv 4.57.1. `tokenizer_class` is `TokenizersBackend` (transformers 5). The run still produced text, so some tokenizer loaded. **INFERRED** that 4.57.1 did not load that class by name and fell back to `tokenizer.json`.

## 9. Oct 31 (`4f345a0`) vs `ede7172`

`git diff 4f345a0..ede7172` touches `tpu_v5e_patches.py`, `lfm25_engine_subrunner.py`, tests, `RESULTS.md`, `run_results.json`. Driver and `LLM()` numeric args did not change. `63db413` set `kv_cache_dtype="int8"`; `ede7172` put it back to `fp8` after vLLM 0.32 rejected plain `int8` (`run_results.json:14-15`). The measured Nov 1 engine is fp8, same as Oct 31.

| Item | 2026-10-31 21:29 | 2026-11-01 18:06 |
| :--- | :--- | :--- |
| Engine init | 269.29 s | 388.73 s |
| Short sample | `astphysics` loop | `astphys` loop |
| Short TTFT / 64-tok | 0.1528 s / 170.26 tok/s | 0.1605 s / 165.91 tok/s |
| 200K TTFT | 32.5779 s | 32.3805 s |
| HBM in use @ 200K | 14.738 GiB | 14.97 GiB |
| JAX min entry bytes | `"0"` | `"-1"` plus `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all` |
| Paddings | `{first, last}` of the stock list | `{16, 4096, max_token_size}` |
| Conv state | `shared_attn_metadata` (commit says it was always None → slot 0, `n=T`) | forward-context metadata, `has_initial_states` mask |
| Warning / atexit | present | gone |

**MEASURED** phase split exists only for the new run (`RESULTS.md:55`): weight load 119.32 s, compile 228.24 s, warmup 101.86 s, buckets 16/4096/8192 = 9.64/38.57/38.45 s. Baseline JSON has no phase split. Hub weights did not change between the runs (section 1).

## 10. Likely causes

### Incoherent short output

1. **Raw completion, no chat template.** Both runs echo the user sentence. The template and `<think>` are never applied (section 6). `astrophysics` is the BPE pieces `Ġast` + `rophys` + `ics`. The samples corrupt that middle piece (`astphysics`, then `astphys`) and repeat. Cheap test: one `llm.chat` user turn with `add_generation_prompt=True`, same checkpoint, 64 tokens. If the echo disappears, this was sufficient.
2. **Replaced `ShortConv.forward`.** The misspelling changed when only the conv-state lookup changed, and the echo stayed. Baseline used `n=T` (padded length) because shared metadata was None. The new mask can still keep uninitialized state when `seq_lens > n` on the first chunk, or drop it when `has_initial_states` is missing. Cheap test: print `slot`, `n`, `keep`, and `prev.abs().sum()` on the first prefill and the first decode for one conv layer.
3. **Greedy loop with no repetition penalty.** `temperature=0`, `repetition_penalty=1.0`, `ignore_eos`, `min_tokens=64`. Card values 0.1 / top_k 50 / 1.05 are not applied. This keeps a bad first token going for 64 tokens. Cheap test: same raw ids, card sampling, and read the first 8 tokens. If they are still `ast`+`phys`, sampling only sustains the loop.
4. **fp8 KV on v5e, with the dtype check swallowed.** Both runs. `supports_kv_cache_dtype` returns True for fp8, and `static_validate_inputs` ignores a dtype `ValueError` (`tpu_v5e_patches.py:186-200`). Commit `63db413` says v5e has no native FP8. Weak for a 12-token prompt, because little KV is read. Cheap test: `kv_cache_dtype` left at `auto`, `max_model_len=4096`, same prompt.
5. **Q4_0 re-quant.** Experts are reblocked from Q4_0 group 32 to group 128; backbone is Q4_0→bf16→W8A8 (`repack_checkpoint.py:99-111`). Same checkpoint both runs. Cheap test: llama.cpp, temperature 0, on `LFM2.5-24B-A2B-QAD-Q4_0.gguf` with the same 12 ids. Coherent text there puts the bug in the TPU quant/conv/KV path. `max_position_embeddings` 200000 vs the card’s 131072 does not explain a 12-token prompt.

### Init 269.29 s → 388.73 s

1. **Third compile bucket.** New paddings are 16/4096/8192 and cost 9.64/38.57/38.45 s (**MEASURED**). Baseline kept only the first and last stock sizes (**CODE**). The extra 4096 bucket is about 39 s of the +119 s.
2. **New conv graph, cold XLA.** The metadata/`torch.where` forward is a different program. The unpacked cache is the Qwen 32K tarball, so LFM misses. **MEASURED** compile+warmup 228.24+101.86 s inside 388.73 s. Baseline has no matching split.
3. **Persistent-cache flags write more.** `MIN_ENTRY_SIZE_BYTES=-1` and `ENABLE_XLA_CACHES=all` serialize every program into a cold directory (**CODE**). Time share **INFERRED**.
4. Not the rejected `int8` start: `init_s` is the successful fp8 `LLM()` (`lfm25_engine_subrunner.py:50-60`). Not the Hub download (11.60 s → 10.81 s) and not a weight revision (section 1).
