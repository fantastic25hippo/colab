import argparse, json, os, sys, time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--mode", required=True, choices=["base", "opt", "warm"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    t_p0 = time.time()
    st_p = "/tmp/tpu_stats.json"
    if os.path.exists(st_p): os.remove(st_p)
    os.environ["BENCH_MODE"] = args.mode

    if args.mode in ("opt", "warm"):
        os.environ["JAX_COMPILATION_CACHE_DIR"] = "/content/tpu_compile_cache"
        os.environ["VLLM_XLA_CACHE_PATH"] = "/content/tpu_compile_cache"
        os.makedirs("/content/tpu_compile_cache", exist_ok=True)
    else:
        os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
        os.environ.pop("VLLM_XLA_CACHE_PATH", None)

    from vllm import LLM, SamplingParams
    try:
        t_eng0 = time.time()
        llm = LLM(model=args.model, max_num_seqs=1, max_num_batched_tokens=512,
                  max_model_len=4096, gpu_memory_utilization=0.90, trust_remote_code=True)
        t_startup = time.time() - t_p0
    except Exception:
        import traceback
        err = traceback.format_exc()
        res = {"model": args.model, "mode": args.mode, "status": "failed", "stage": "startup", "error": err}
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
        res = {"model": args.model, "mode": args.mode, "status": "failed", "stage": "generate", "error": err}
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
    peak_bytes = stage_st.get("after_compile", {}).get("peak_bytes_in_use", 0)
    overhead_bytes = max(0, peak_bytes - w_bytes - kv_bytes) if peak_bytes else 0
    lim_bytes = stage_st.get("after_load", {}).get("bytes_limit", 0)

    t_load = stage_st.get("after_load", {}).get("t_stamp", t_eng0) - t_p0
    c_dur = stage_st.get("after_compile", {}).get("dur_s")

    res = {
        "model": args.model, "mode": args.mode, "status": "success",
        "phase_times": {
            "start_to_weights_loaded_s": round(t_load, 2),
            "compile_time_s": round(c_dur, 2) if c_dur else None,
            "time_to_first_token_s": round(ttft, 4) if ttft else None,
            "total_startup_s": round(t_startup, 2),
        },
        "HBM": {
            "bytes_limit": lim_bytes, "bytes_limit_gib": round(lim_bytes / (1024**3), 2),
            "weight_bytes": w_bytes, "weight_gib": round(w_bytes / (1024**3), 3),
            "kv_cache_bytes": kv_bytes, "kv_cache_gib": round(kv_bytes / (1024**3), 3),
            "peak_bytes_in_use": peak_bytes, "peak_gib": round(peak_bytes / (1024**3), 3),
            "overhead_bytes": overhead_bytes, "overhead_gib": round(overhead_bytes / (1024**3), 3),
            "num_gpu_blocks": stage_st.get("after_kv", {}).get("num_gpu_blocks"),
            "gpu_memory_utilization": stage_st.get("after_kv", {}).get("gpu_memory_utilization", 0.90),
            "stage_stats": stage_st,
        },
        "generation": {
            "tokens": toks, "wall_s": round(gen_wall, 3), "decode_tok_s": round(dec_tps, 2),
            "first_100_chars": out_text[:100],
        }
    }
    with open(args.out, "w") as f: json.dump(res, f)
    print("RUN_RESULT_JSON:" + json.dumps(res), flush=True)

if __name__ == "__main__": main()
