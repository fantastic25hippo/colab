import argparse, json, os, sys, time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mode", required=True)
    p.add_argument("--load-fix", type=int, default=1)
    p.add_argument("--skip-precompile", type=int, default=0)
    p.add_argument("--compilation-sizes", default=None)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    st_p = "/tmp/tpu_stats.json"
    if os.path.exists(st_p): os.remove(st_p)
    os.environ["BENCH_MODE"] = args.mode
    os.environ["BENCH_LOAD_FIX"] = str(args.load_fix)
    os.environ["SKIP_JAX_PRECOMPILE"] = str(args.skip_precompile)
    if args.compilation_sizes:
        os.environ["BENCH_COMPILATION_SIZES"] = args.compilation_sizes
    if args.cache_dir:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = args.cache_dir
        os.environ["VLLM_XLA_CACHE_PATH"] = args.cache_dir
        os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
        os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"
        os.environ["JAX_LOG_COMPILES"] = "1"
        os.environ["JAX_EXPLAIN_CACHE_MISSES"] = "1"
        os.makedirs(args.cache_dir, exist_ok=True)

    t_p0 = time.time()
    from vllm import LLM, SamplingParams
    try:
        llm = LLM(model=args.model, max_num_seqs=1, max_num_batched_tokens=512,
                  max_model_len=4096, gpu_memory_utilization=0.90, trust_remote_code=True)
        t_startup = time.time() - t_p0
    except Exception:
        import traceback
        err = traceback.format_exc()
        res = {"mode": args.mode, "status": "failed", "stage": "startup", "error": err}
        with open(args.out, "w") as f: json.dump(res, f)
        print("RUN_FAILED:\n" + err, flush=True); sys.exit(1)

    try:
        prompt = "The exploration of deep space has fascinated humanity for centuries. Astronomers observe distant stars and galaxies through powerful telescopes, searching for clues about cosmic microwave background and spacetime dimensions. "
        sp = SamplingParams(max_tokens=128, min_tokens=128, temperature=0.0, ignore_eos=True)
        t_g0 = time.time()
        out = llm.generate([prompt], sp)[0]
        gen_wall, out_text, toks = time.time() - t_g0, out.outputs[0].text, len(out.outputs[0].token_ids)
        ttft, m = None, getattr(out, "metrics", None)
        if m:
            ft = getattr(m, "first_token_time", None)
            ar = getattr(m, "arrival_time", None) or getattr(m, "first_scheduled_time", None)
            if ft is not None and ar is not None: ttft = ft - ar
            elif getattr(m, "time_to_first_token", None) is not None: ttft = m.time_to_first_token
        dec_tps = (toks - 1) / (gen_wall - ttft) if (ttft and gen_wall > ttft) else (toks / gen_wall if gen_wall > 0 else 0)
    except Exception:
        import traceback
        err = traceback.format_exc()
        res = {"mode": args.mode, "status": "failed", "stage": "generate", "error": err}
        with open(args.out, "w") as f: json.dump(res, f)
        print("GEN_FAILED:\n" + err, flush=True); sys.exit(1)

    stage_st = {}
    if os.path.exists(st_p):
        try:
            with open(st_p) as f: stage_st = json.load(f)
        except Exception: pass

    w_bytes = stage_st.get("after_load", {}).get("bytes_in_use", 0)
    kv_in_use = stage_st.get("after_kv", {}).get("bytes_in_use", w_bytes)
    kv_bytes = max(0, kv_in_use - w_bytes)
    c_st = stage_st.get("after_compile", {})
    g_st = stage_st.get("after_generation", {})
    peak_bytes = max(c_st.get("peak_bytes_in_use", 0), g_st.get("peak_bytes_in_use", 0))
    overhead_bytes = max(0, peak_bytes - w_bytes - kv_bytes) if peak_bytes else 0
    lim_bytes = stage_st.get("after_load", {}).get("bytes_limit", 0)

    blocks = stage_st.get("after_kv", {}).get("num_gpu_blocks")
    b_size = stage_st.get("after_kv", {}).get("block_size", 256)
    total_toks = (blocks * b_size) if blocks else 0
    cache_info = stage_st.get("cache", {})

    res = {
        "run_name": args.mode, "status": "success",
        "flags": {
            "load_fix": bool(args.load_fix), "skip_jax_precompile": bool(args.skip_precompile),
            "compilation_sizes": json.loads(args.compilation_sizes) if args.compilation_sizes else None,
            "cache_dir": args.cache_dir
        },
        "phase_times": {
            "load_s": round(stage_st.get("after_load", {}).get("dur_s", 0.0), 2),
            "compile_s": round(c_st.get("dur_s", 0.0), 2),
            "time_to_first_token_s": round(ttft, 4) if ttft else None,
            "total_startup_s": round(t_startup, 2),
        },
        "HBM": {
            "bytes_limit_gib": round(lim_bytes / (1024**3), 3),
            "weight_gib": round(w_bytes / (1024**3), 3),
            "kv_cache_gib": round(kv_bytes / (1024**3), 3),
            "peak_gib": round(peak_bytes / (1024**3), 3),
            "overhead_gib": round(overhead_bytes / (1024**3), 3),
            "at_a_after_load": stage_st.get("after_load"),
            "at_b_after_kv": stage_st.get("after_kv"),
            "at_c_after_compile": c_st,
            "at_d_after_generation": g_st,
        },
        "kv_cache": {
            "num_gpu_blocks": blocks, "block_size": b_size, "total_tokens": total_toks,
            "kv_cache_gib": round(kv_bytes / (1024**3), 3),
            "gpu_memory_utilization": stage_st.get("after_kv", {}).get("gpu_memory_utilization", 0.90)
        },
        "cache_stats": {
            "programs_compiled": cache_info.get("compiled", 0),
            "programs_cached": cache_info.get("cached", 0)
        },
        "generation": {
            "tokens": toks, "wall_s": round(gen_wall, 3), "decode_tok_s": round(dec_tps, 2),
            "first_100_chars": out_text[:100]
        }
    }
    with open(args.out, "w") as f: json.dump(res, f)
    print("RUN_RESULT_JSON:" + json.dumps(res), flush=True)

if __name__ == "__main__": main()
