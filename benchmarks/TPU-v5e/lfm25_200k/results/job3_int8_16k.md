# Job 3 Report: TPU v5e Context 16K Run (STOP at G4)

## 1. Executive Summary & Stop Trigger

STOP at G4. No retry. No code change.

Launch: new -s tpu-lfm25 --tpu v5e1 (4.83s); kv_int8.py uploaded; env LFM_ABORT_ON_CACHE_MISS=0 LFM_MAX_MODEL_LEN=16384 LFM_KV_INT8=1 LFM_KV_SCALE_PROBE=1; K/V scales unset.

Session terminated; No active sessions found on server.

## 2. Preflight Checks (P1 - P5)

- P1 PASS HEAD db31164, no tracked changes.
- P2 PASS sha256 155db072b3fb75141c836337a35382ff4d3f70fdd505e92bf0495760f2ae87f8
- P3 PASS 9 passed.
- P4 PASS OFF …42f6059b.tar ; INT8_16K …8f6acabe.tar
- P5 PASS. Quote driver:242-258: abort_on_miss = os.environ.get("LFM_ABORT_ON_CACHE_MISS", "1") != "0" then if abort_on_miss and miss: sys.exit(2). With =0, miss falls through to t_spawn. MEASURED: 8f6acabe.tar 404, no ABORT ON CACHE MISS line.

## 3. Goals & Gates (G1 - G7)

- G1 PASS `[LFM_KV_INT8] active kv_cache_dtype=fp8`
- G2 PASS `RPA v3 kernel tuned block sizes for ('TPU v5e', 16, 'q_bfloat16_kv_float8_e4m3fn', 'q_head-32_kv_head-8_head-64', 'max_model_len-16384-sw-None'): bkv_p=128, bq=32`
- G3 PASS page 16384; 13364 attn blocks; 2.04 GiB. Quote: `attn=10 layers x 13364 blocks x 16384 B = 2.04 GiB`
- G4 FAIL `Context 16384 (p=16320, bos=1, echo_bos=1, decode ignore_eos=True): TTFT=1.0159s, Decode=46.67 tok/s, needle_pass=False` output_text is `\n` plus `</think>` x63.
- G5 FAIL no CONVDBG.
- G6 FAIL no `[LFM_KV_SCALE_PROBE]` layer lines. Scales: none.
- G7 FAIL cache_publish null (engine.ok false). Expected tar ends in 8f6acabe.tar.

## 4. Run Metrics (MEASURED)

Metrics MEASURED: compile_warmup 0.009s; torch.compile+warmup 9.76s; engine-ready 26.51s; TTFT 16K 1.0159s; decode 46.67 tok/s; HBM init 14.488 GiB; peak HBM 14.818 GiB; short TTFT 53.5814s; in-VM first token 109.96s; provision 4.83s; weights 8.82s / 13125.7 MB / 1487.7 MB/s.
Per-layer scales: not reached.

## 5. Last log lines

```
[LFM25_SUBRUNNER 18:09:13] Engine ready in 26.51s; weight_load=16.62s kv_alloc=0.129s compile_warmup=0.009s; kv_dtype=fp8; HBM={'in_use_gib': 14.488, 'peak_gib': 14.488, 'limit_gib': 15.748}
Rendering conversations:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering conversations: 100%|██████████| 1/1 [00:00<00:00, 1077.40it/s]
Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s](MainProcess pid=7506) WARNING 11-02 18:09:13 [decorators.py:694] Detected eager backend, disabling AOT compile.
(MainProcess pid=7506) INFO 11-02 18:09:13 [monitor.py:40] [backbone] Starting torch.compile
/dev/shm/vllm_env/lib/python3.12/site-packages/torchax/tensor.py:183: UserWarning: Explicitly requested dtype int64 requested in astype is not available, and will be truncated to dtype int32. To enable more dtypes, set the jax_enable_x64 configuration option or the JAX_ENABLE_X64 shell environment variable. See https://github.com/jax-ml/jax#current-gotchas for more.
  res = jax_function(self._elem, *args, **kwargs)
(MainProcess pid=7506) WARNING 11-02 18:09:13 [tuned_block_sizes_hd64.py:456] Couldn`t find tuned sizes for the RPA v3 kernel with ('TPU v5e', 16, 'q_bfloat16_kv_float8_e4m3fn', 'q_head-32_kv_head-8_head-64', 'max_model_len-16384-sw-None')
(MainProcess pid=7506) INFO 11-02 18:09:13 [tuned_block_sizes_hd64.py:477] RPA v3 kernel tuned block sizes for ('TPU v5e', 16, 'q_bfloat16_kv_float8_e4m3fn', 'q_head-32_kv_head-8_head-64', 'max_model_len-16384-sw-None'): bkv_p=128, bq=32
{"stage": "hb", "t": 1793642957.1115353}
(MainProcess pid=7506) INFO 11-02 18:09:23 [monitor.py:62] [backbone] torch.compile and initial profiling/warmup run together took 9.76 s in total
{"stage": "hb", "t": 1793642972.112216}
{"stage": "hb", "t": 1793642987.1128857}
{"stage": "hb", "t": 1793643002.1135106}
Processed prompts: 100%|██████████| 1/1 [00:53<00:00, 53.58s/it, est. speed input: 0.41 toks/s, output: 0.02 toks/s]
[LFM25_SUBRUNNER 18:10:07] Short TTFT (chat, 22 ids): 53.5814s
Rendering conversations: 100%|██████████| 1/1 [00:00<00:00, 985.74it/s]
Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]/dev/shm/vllm_env/lib/python3.12/site-packages/jax/_src/interpreters/mlir.py:1453: UserWarning: Some donated buffers were not usable: int32[16], int32[16].
See an explanation at https://docs.jax.dev/en/latest/faq.html#buffer-donation.
  warnings.warn("Some donated buffers were not usable:"
(MainProcess pid=7506) INFO 11-02 18:10:07 [tuned_block_sizes_hd64.py:477] RPA v3 kernel tuned block sizes for ('TPU v5e', 16, 'q_bfloat16_kv_float8_e4m3fn', 'q_head-32_kv_head-8_head-64', 'max_model_len-16384-sw-None'): bkv_p=128, bq=16
{"stage": "hb", "t": 1793643017.114208}
Processed prompts: 100%|██████████| 1/1 [00:17<00:00, 17.21s/it, est. speed input: 1.28 toks/s, output: 3.72 toks/s]
[LFM25_SUBRUNNER 18:10:24] Chat 64 tok: 17.207s; ids=[597, 4695, 20589, 34, 496, 89509, 278, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 2426, 969, 355, 267, 5939, 5439, 22, 1715, 1976, 401, 13816, 302, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 22, 27450, 736, 685, 1976, 401, 18948, 302, 5030, 37933, 1003, 11051, 34, 14124, 9896, 1171, 17627, 20, 9418, 20, 8502, 47235, 20, 4222, 1348, 309, 29460]; text='The user asks: "Explain the fundamental principles of astrophysics and stellar evolution." This is a broad request. They want an explanation of fundamental principles of astrophysics and stellar evolution. Likely they want an overview of astrophysics topics: physics principles like gravity, radiation, thermodynamics, etc., and stellar'
Rendering conversations: 100%|██████████| 1/1 [00:00<00:00, 741.04it/s]
Processed prompts:   0%|          | 0/1 [00:00<?, ?it/s, est. speed input: 0.00 toks/s, output: 0.00 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.35it/s, est. speed input: 29.59 toks/s, output: 86.08 toks/s]
Processed prompts: 100%|██████████| 1/1 [00:00<00:00,  1.34it/s, est. speed input: 29.59 toks/s, output: 86.08 toks/s]
[LFM25_SUBRUNNER 18:10:25] Decode 64 tok (ignore_eos): 0.7457s (85.83 tok/s)
Rendering prompts:   0%|          | 0/1 [00:00<?, ?it/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 786.78it/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.01s/it, est. speed input: 16096.75 toks/s, output: 0.99 toks/s]
Rendering prompts: 100%|██████████| 1/1 [00:00<00:00, 637.72it/s]
Processed prompts: 100%|██████████| 1/1 [00:01<00:00,  1.37s/it, est. speed input: 11922.24 toks/s, output: 46.75 toks/s]
[LFM25_SUBRUNNER 18:10:27] Context 16384 (p=16320, bos=1, echo_bos=1, decode ignore_eos=True): TTFT=1.0159s, Decode=46.67 tok/s, needle_pass=False, HBM={'in_use_gib': 14.736, 'peak_gib': 14.818, 'limit_gib': 15.748}
[LFM25_SUBRUNNER 18:10:27] [WATERFALL] total_vm_s=95.85s accounted_s=89.64s unaccounted_s=6.21s
[LFM25_SUBRUNNER 18:10:27]      0.01s -   11.97s ( 11.96s): imports_and_tpu_init
[LFM25_SUBRUNNER 18:10:27]     15.76s -   21.97s (  6.21s): config_setup
[LFM25_SUBRUNNER 18:10:27]     19.23s -   19.40s (  0.17s): create_engine_config
[LFM25_SUBRUNNER 18:10:27]     23.89s -   40.51s ( 16.62s): weight_load
[LFM25_SUBRUNNER 18:10:27]     40.55s -   40.68s (  0.13s): kv_alloc
[LFM25_SUBRUNNER 18:10:27]     40.68s -   40.69s (  0.01s): compile_warmup
[LFM25_SUBRUNNER 18:10:27]     41.08s -   42.22s (  1.13s): tokenizer
[LFM25_SUBRUNNER 18:10:27]    114.86s -  168.44s ( 53.58s): ttft
[LFM25_SUBRUNNER 18:10:27] Wrote /content/engine_results.json
[LFM25_SUBRUNNER 18:10:27] Engine benchmarking complete
(MainProcess pid=7506) WARNING 11-02 18:10:28 [parallel_state.py:2557] torch.accelerator.empty_host_cache() only available in Pytorch >=2.9
[LFM25_DRIVER 18:10:31] === Host WATERFALL ===
[LFM25_DRIVER 18:10:31] Phase                    | Start (s)  | Dur (s)    | End (s)    | Origin  
[LFM25_DRIVER 18:10:31] ------------------------------------------------------------------------
[LFM25_DRIVER 18:10:31] EXEC start               | 0.000      | 10.290     | 10.290     | host    
[LFM25_DRIVER 18:10:31] weights dl               | 10.824     | 8.823      | 19.647     | host     | parallel=true
[LFM25_DRIVER 18:10:31] cache dl                 | 10.826     | 0.268      | 11.094     | host     | parallel=true
[LFM25_DRIVER 18:10:31] venv dl                  | 10.827     | 2.465      | 13.292     | host     | parallel=true
[LFM25_DRIVER 18:10:31] venv unpack              | 13.292     | 8.033      | 21.325     | host    
[LFM25_DRIVER 18:10:31] imports_and_tpu_init     | 24.404     | 11.963     | 36.367     | in-vm   
[LFM25_DRIVER 18:10:31] config_setup             | 40.162     | 6.210      | 46.372     | in-vm   
[LFM25_DRIVER 18:10:31] create_engine_config     | 43.626     | 0.174      | 43.800     | in-vm   
[LFM25_DRIVER 18:10:31] weight_load              | 48.292     | 16.620     | 64.912     | in-vm   
[LFM25_DRIVER 18:10:31] kv_alloc                 | 64.946     | 0.129      | 65.075     | in-vm   
[LFM25_DRIVER 18:10:31] compile_warmup           | 65.075     | 0.009      | 65.084     | in-vm   
[LFM25_DRIVER 18:10:31] tokenizer                | 65.482     | 1.131      | 66.613     | in-vm   
[LFM25_DRIVER 18:10:31] ttft                     | 139.258    | 53.581     | 192.839    | in-vm   
[LFM25_DRIVER 18:10:31] accounted_s = 110.434 s (sum of non-overlapping phase durations)
[LFM25_DRIVER 18:10:31] unaccounted_s = 9.819 s (total = 120.253 s)
[LFM25_DRIVER 18:10:31] unaccounted_gaps = [{'after': 'EXEC start', 'before': 'weights dl', 'gap_s': 0.534}, {'after': 'venv unpack', 'before': 'imports_and_tpu_init', 'gap_s': 3.079}, {'after': 'imports_and_tpu_init', 'before': 'config_setup', 'gap_s': 3.795}, {'after': 'config_setup', 'before': 'weight_load', 'gap_s': 1.92}, {'after': 'weight_load', 'before': 'kv_alloc', 'gap_s': 0.034}, {'after': 'compile_warmup', 'before': 'tokenizer', 'gap_s': 0.398}, {'after': 'tokenizer', 'before': 'ttft', 'gap_s': 72.645}]
=== COLDSTART_RESULTS_JSONL ===
{"run_label": "job3_int8_16k", "diag": false, "startup": {"remount_s": 0.01, "venv_dl_s": 2.46, "venv_unpack_s": 8.03, "weights_submit_rel_s": 0.53, "prep_total_s": 10.5, "subrunner_spawn_rel_s": 11.04, "cache_restore": {"hit": false, "reason": "404 Client Error. (Request ID: Root=1-6ae8d191-6d8308562a15d4241d23b2c5;33c390a2-befd-43e2-b6a4-36d538924503)\n\nEntry Not Found for url: https://huggingface.co/avgrocketman/tpu-compile-cache/resolve/main/tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar.", "cache_dl_s": 0.27, "t_dl_start": 1793642897.645305, "t_dl_end": 1793642897.9132838, "filename": "tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len16384__b16-8192__kvint8__8f6acabe.tar", "vllm": "0.32.0", "key_args": "max_model_len=100000;max_num_seqs=1;max_num_batched_tokens=8192;gpu_memory_utilization=0.92;prefix_caching=0;kv=auto;buckets=16|8192", "dir": {"entries": 0, "mb": 0}}, "cache_dl_s": 0.27, "cache_unpack_s": null, "weights": {"shards": 9, "total_mb": 13125.7, "per_shard_mb": {"model-00001.safetensors": 1611.8, "model-00002.safetensors": 1612.3, "model-00003.safetensors": 1611.1, "model-00004.safetensors": 1611.7, "model-00005.safetensors": 1612.3, "model-00006.safetensors": 1612.3, "model-00007.safetensors": 1611.4, "model-00008.safetensors": 1611.7, "model-00009.safetensors": 231.3}, "largest": "model-00005.safetensors", "dl_s": 8.82, "t_start": 1793642897.6435437, "t_end": 1793642906.4661162, "mb_s": 1487.7, "shm_read_probe": {"read_mb": 1073.7, "s": 0.127, "mb_s": 8430.7}}}, "weights_dl_s": 8.82, "in_vm_cold_start_to_first_token_s": 109.96, "total_cold_start_to_first_token_s": 120.25, "accounted_s": 110.434, "unaccounted_s": 9.819, "unaccounted_gaps": [{"after": "EXEC start", "before": "weights dl", "gap_s": 0.534}, {"after": "venv unpack", "before": "imports_and_tpu_init", "gap_s": 3.079}, {"after": "imports_and_tpu_init", "before": "config_setup", "gap_s": 3.795}, {"after": "config_setup", "before": "weight_load", "gap_s": 1.92}, {"after": "weight_load", "before": "kv_alloc", "gap_s": 0.034}, {"after": "compile_warmup", "before": "tokenizer", "gap_s": 0.398}, {"after": "tokenizer", "before": "ttft", "gap_s": 72.645}], "waterfall": [{"name": "EXEC start", "start_rel": 0.0, "dur_s": 10.29, "end_rel": 10.29, "origin": "host", "parallel": false}, {"name": "weights dl", "start_rel": 10.824, "dur_s": 8.823, "end_rel": 19.647, "origin": "host", "parallel": true}, {"name": "cache dl", "start_rel": 10.826, "dur_s": 0.268, "end_rel": 11.094, "origin": "host", "parallel": true}, {"name": "venv dl", "start_rel": 10.827, "dur_s": 2.465, "end_rel": 13.292, "origin": "host", "parallel": true}, {"name": "venv unpack", "start_rel": 13.292, "dur_s": 8.033, "end_rel": 21.325, "origin": "host", "parallel": false}, {"name": "imports_and_tpu_init", "start_rel": 24.404, "dur_s": 11.963, "end_rel": 36.367, "origin": "in-vm", "parallel": false}, {"name": "config_setup", "start_rel": 40.162, "dur_s": 6.21, "end_rel": 46.372, "origin": "in-vm", "parallel": false}, {"name": "create_engine_config", "start_rel": 43.626, "dur_s": 0.174, "end_rel": 43.8, "origin": "in-vm", "parallel": false}, {"name": "weight_load", "start_rel": 48.292, "dur_s": 16.62, "end_rel": 64.912, "origin": "in-vm", "parallel": false}, {"name": "kv_alloc", "start_rel": 64.946, "dur_s": 0.129, "end_rel": 65.075, "origin": "in-vm", "parallel": false}, {"name": "compile_warmup", "start_rel": 65.075, "dur_s": 0.009, "end_rel": 65.084, "origin": "in-vm", "parallel": false}, {"name": "tokenizer", "start_rel": 65.482, "dur_s": 1.131, "end_rel": 66.613, "origin": "in-vm", "parallel": false}, {"name": "ttft", "start_rel": 139.258, "dur_s": 53.581, "end_rel": 192.839, "origin": "in-vm", "parallel": false}], "engine": {"ok": false, "init_s": 26.51, "phases": {"import_and_tpu_init_s": 11.96, "wait_for_weights_s": 0.0, "config_setup_s": 6.21, "llm_init_s": 26.51, "weight_load_s": 16.62, "kv_alloc_s": 0.129, "compile_warmup_s": 0.009, "warmup_pass_total_s": null, "warmup_per_bucket_s": {}, "warmup_tasks": [], "timeline": {"compile_per_bucket_s": {}, "compile_runs_per_bucket": {}, "warmup_pass": null, "weight_shard_loads": [], "tracing_lines": 0, "persistent_cache_misses": 0}, "log_lines_seen": 2, "llm_init_unaccounted_s": 3.55}, "engine_args": {"max_model_len": 16384, "max_num_seqs": 1, "max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.92, "enable_prefix_caching": false, "kv_cache_dtype": "auto (vLLM default, bf16)", "buckets": [16, 8192]}, "cache": {"before": {"entries": 0, "mb": 0}, "after": {"entries": 60, "mb": 0.9}, "new_entries": 60}, "hbm_after_init": {"in_use_gib": 14.488, "peak_gib": 14.488, "limit_gib": 15.748}, "kv_cache_dtype_resolved": "fp8", "first_token_ts": 1793643007.0716496, "ttft_short_s": 53.5814, "decode_tok_s": 85.83, "sample_output": "The user asks: \"Explain the fundamental principles of astrophysics and stellar evolution.\" This is a broad request. They want an explanation of fundamental principles of astrophysics and stellar evolution. Likely they want an overview of astrophysics topics: physics principles like gravity, radiation, thermodynamics, etc., and stellar", "short": {"ttft_s": 53.5814, "ttft_prompt_len": 22, "ttft_bos": 1, "chat": {"wall_s": 17.207, "prompt_len": 22, "prompt_bos": 1, "out_ids": [597, 4695, 20589, 34, 496, 89509, 278, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 2426, 969, 355, 267, 5939, 5439, 22, 1715, 1976, 401, 13816, 302, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 22, 27450, 736, 685, 1976, 401, 18948, 302, 5030, 37933, 1003, 11051, 34, 14124, 9896, 1171, 17627, 20, 9418, 20, 8502, 47235, 20, 4222, 1348, 309, 29460], "text": "The user asks: \"Explain the fundamental principles of astrophysics and stellar evolution.\" This is a broad request. They want an explanation of fundamental principles of astrophysics and stellar evolution. Likely they want an overview of astrophysics topics: physics principles like gravity, radiation, thermodynamics, etc., and stellar", "ignore_eos": false}, "decode": {"wall_s": 0.7457, "tok_s": 85.83, "tok_s_excl_prefill": 63000000.0, "n_out": 64, "ignore_eos": true, "note": "ignore_eos=True: throughput timing only, text not judged; tok_s=64/wall, tok_s_excl_prefill=63/(wall-ttft)"}}, "context_evals": {"16384": {"prompt_len": 16320, "ttft_s": 1.0159, "decode_tok_s": 46.67, "decode_wall_s": 1.3712, "decode_tok_s_excl_prefill": 177.32, "prompt_bos_count": 1, "engine_prompt_bos_count": 1, "prompt_ids_first40": [124894, 124899, 5922, 207, 89509, 278, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 22, 42847, 278, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 22, 42847, 278, 9761, 9896, 302, 5030, 37933, 1003, 309, 29460, 7216, 22], "n_out_decode": 64, "needle_pass": false, "needle_ok": false, "output_text": "\n</think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think></think>", "build": {"target": 16320, "tail_n": 6, "rendered_ids": 16439, "needle": "The secret code is 7-4-1-9-ORCHID.", "needle_pos": 8148, "question": "What is the secret code?"}, "hbm": {"in_use_gib": 14.736, "peak_gib": 14.818, "limit_gib": 15.748}}}, "hbm_stats": {"in_use_gib": 14.736, "peak_gib": 14.818, "limit_gib": 15.748}, "log_timeline": [{"t": 0.24, "log": "vllm", "msg": "Failed to import from vllm._C: ModuleNotFoundError(\"No module named 'vllm._C'\")"}], "notes": ["all prompts use the tokenizer chat template (add_generation_prompt=True)", "short chat prompt is 22 ids -> pads to the 8192 bucket; the 2026-11-01 18:06 run used the raw 12-id prompt (bucket 16)", "chat coherence run: greedy, max_tokens=64, no ignore_eos/min_tokens/seed", "64-token decode runs set ignore_eos=True: throughput timing only, text not judged", "kv_cache_dtype is vLLM's default (auto); fp8 is not enabled"], "waterfall": [{"name": "imports_and_tpu_init", "start_rel": 0.006, "dur_s": 11.963, "end_rel": 11.969}, {"name": "config_setup", "start_rel": 15.764, "dur_s": 6.21, "end_rel": 21.974}, {"name": "create_engine_config", "start_rel": 19.228, "dur_s": 0.174, "end_rel": 19.402}, {"name": "weight_load", "start_rel": 23.894, "dur_s": 16.62, "end_rel": 40.514}, {"name": "kv_alloc", "start_rel": 40.548, "dur_s": 0.129, "end_rel": 40.677}, {"name": "compile_warmup", "start_rel": 40.677, "dur_s": 0.009, "end_rel": 40.686}, {"name": "tokenizer", "start_rel": 41.084, "dur_s": 1.131, "end_rel": 42.215}, {"name": "ttft", "start_rel": 114.86, "dur_s": 53.581, "end_rel": 168.441}], "t_vm0": 1793642911.2174423, "accounted_s": 89.643, "unaccounted_s": 6.211, "unaccounted_gaps": [{"after": 11.969, "before": 15.764, "gap_s": 3.795}, {"after": 21.974, "before": 23.894, "gap_s": 1.92}, {"after": 40.686, "before": 41.084, "gap_s": 0.398}, {"after": 42.215, "before": 114.86, "gap_s": 72.645}]}, "cache_publish": null, "subrunner_rc": 0}
=== END_COLDSTART_RESULTS ===
=== DRIVER rc=0 ===
=== Downloading /content/build.log ===
RUN: colab --auth=oauth2 download -s tpu-lfm25 /content/build.log /tmp/job3_build.log
[colab] Downloaded '/content/build.log' to '/tmp/job3_build.log'
=== Stopping session tpu-lfm25 ===
RUN: colab --auth=oauth2 stop -s tpu-lfm25
[colab] Stopping session 'tpu-lfm25'...
[colab] Session terminated.
=== Confirming sessions ===
RUN: colab --auth=oauth2 sessions
[colab] No active sessions found on server.
```
