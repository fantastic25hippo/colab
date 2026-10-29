import argparse, json, os, sys, time

def p95(v):
    if not v: return 0.0
    s = sorted(v)
    k = (len(s) - 1) * 0.95
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[f] * (1 - (k - f)) + s[c] * (k - f), 4)

def wait_hbm_stage(stage):
    with open("/tmp/record_hbm_stage.txt", "w") as f: f.write(stage)
    for _ in range(25):
        if not os.path.exists("/tmp/record_hbm_stage.txt"): break
        time.sleep(0.04)

def get_stats():
    st_p = "/tmp/tpu_stats.json"
    if os.path.exists(st_p):
        try:
            with open(st_p) as f: return json.load(f)
        except Exception: pass
    return {}

def add_req(eng, rid, p, sp):
    try: eng.add_request(rid, p, sp)
    except TypeError:
        try: eng.add_request(request_id=rid, prompt=p, sampling_params=sp)
        except TypeError: eng.add_request(request_id=rid, inputs=p, params=sp)

def run_batch_gen(llm, prompts, sp):
    t0 = time.perf_counter()
    req_ids = [f"r_{int(time.time()*1000)}_{i}" for i in range(len(prompts))]
    ttft, fin, toks = {}, {}, {}
    try:
        eng = llm.llm_engine
        for rid, p in zip(req_ids, prompts): add_req(eng, rid, p, sp)
        while eng.has_unfinished_requests():
            outs = eng.step()
            now = time.perf_counter()
            for o in outs:
                rid = o.request_id
                if rid not in ttft and len(o.outputs[0].token_ids) > 0:
                    ttft[rid] = now - t0
                if o.finished:
                    fin[rid] = now - t0
                    toks[rid] = len(o.outputs[0].token_ids)
    except Exception:
        outs = llm.generate(prompts, sp)
        now = time.perf_counter()
        for i, o in enumerate(outs):
            rid = req_ids[i]
            ttft[rid] = getattr(o.metrics, "first_token_time", None) or 0.01
            fin[rid] = now - t0
            toks[rid] = len(o.outputs[0].token_ids)
    wall = time.perf_counter() - t0
    return {"wall": wall, "ttft": ttft, "fin": fin, "toks": toks, "req_ids": req_ids}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--out", default="/content/tpu_engine_results.json")
    args = p.parse_args()

    t_start = time.time()
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=17408, gpu_memory_utilization=0.90, trust_remote_code=True)
    t_ready = time.time()

    base_p = "The exploration of deep space has fascinated humanity for centuries. Astronomers observe distant stars and galaxies through powerful telescopes. "
    sp_w = SamplingParams(max_tokens=128, min_tokens=128, temperature=0.0, ignore_eos=True)
    sp_256 = SamplingParams(max_tokens=256, min_tokens=256, temperature=0.0, ignore_eos=True)

    t_first_tok = None
    for _ in range(3):
        r = run_batch_gen(llm, [base_p], sp_w)
        if t_first_tok is None and r["ttft"]: t_first_tok = time.time()

    # Step 2: Real decode speed (5 timed single requests)
    c_b2 = get_stats().get("cache", {}).get("compiled", 0)
    s2_runs = []
    for _ in range(5):
        r = run_batch_gen(llm, [base_p], sp_256)
        rid = r["req_ids"][0]
        tf, td, nt = r["ttft"].get(rid, 0.0), r["fin"].get(rid, r["wall"]), r["toks"].get(rid, 256)
        s2_runs.append({"ttft_s": round(tf, 4), "decode_tok_s": round((nt - 1) / max(0.001, td - tf), 2), "wall_s": round(r["wall"], 4), "tokens": nt})
    wait_hbm_stage("after_step2_decode")
    c_a2 = get_stats().get("cache", {}).get("compiled", 0)

    # Step 3: Batching at 1, 4, 8, 16
    s3_res = {}
    for b in [1, 4, 8, 16]:
        run_batch_gen(llm, [base_p] * b, sp_w)
        cb = get_stats().get("cache", {}).get("compiled", 0)
        r = run_batch_gen(llm, [base_p] * b, sp_256)
        ca = get_stats().get("cache", {}).get("compiled", 0)
        tot_tok = sum(r["toks"].values())
        req_tps = [round((r["toks"][rid] - 1) / max(0.001, r["fin"][rid] - r["ttft"].get(rid, 0)), 2) for rid in r["req_ids"]]
        m_ttft = round(sum(r["ttft"].values()) / len(r["ttft"]), 4) if r["ttft"] else 0
        s3_res[str(b)] = {"batch_size": b, "total_throughput_tok_s": round(tot_tok / r["wall"], 2) if r["wall"] > 0 else 0,
                          "mean_per_req_tok_s": round(sum(req_tps)/len(req_tps), 2), "mean_ttft_s": m_ttft,
                          "p95_ttft_s": p95(list(r["ttft"].values())), "compiles": ca - cb, "wall_s": round(r["wall"], 3)}
    wait_hbm_stage("after_step3_batching")

    # Step 4: Context lengths
    chunk = "Artificial intelligence and distributed computing continue to revolutionize modern science. "
    ctx_prompts = {"512": chunk * 36, "4096": chunk * 290, "16384": chunk * 1100}
    s4_res = {}
    for clen, cp in ctx_prompts.items():
        r = run_batch_gen(llm, [cp], sp_256)
        rid = r["req_ids"][0]
        tf, td, nt = r["ttft"].get(rid, 0.0), r["fin"].get(rid, r["wall"]), r["toks"].get(rid, 256)
        wait_hbm_stage(f"ctx_{clen}")
        s4_res[clen] = {"prompt_target": int(clen), "prefill_ttft_s": round(tf, 4), "decode_tok_s": round((nt - 1) / max(0.001, td - tf), 2), "wall_s": round(r["wall"], 3)}

    # 4 concurrent at 4096
    r4k = run_batch_gen(llm, [ctx_prompts["4096"]] * 4, sp_256)
    s4_res["4_concurrent_4k"] = {"wall_s": round(r4k["wall"], 3), "throughput_tok_s": round(sum(r4k["toks"].values()) / r4k["wall"], 2), "mean_ttft_s": round(sum(r4k["ttft"].values())/len(r4k["ttft"]), 4)}
    wait_hbm_stage("after_step4_context")

    final_st = get_stats()
    res = {
        "engine_startup": {
            "load_s": round(final_st.get("after_load", {}).get("dur_s", 0), 2),
            "compile_s": round(final_st.get("after_compile", {}).get("dur_s", 0), 2),
            "ready_s": round(t_ready - t_start, 2),
            "first_token_from_ready_s": round(t_first_tok - t_ready, 2) if t_first_tok else None,
            "cache_compiled": final_st.get("cache", {}).get("compiled", 0),
            "cache_hits": final_st.get("cache", {}).get("cached", 0),
        },
        "step2_decode": {"runs": s2_runs, "mean_decode_tok_s": round(sum(x["decode_tok_s"] for x in s2_runs)/len(s2_runs), 2), "mean_ttft_s": round(sum(x["ttft_s"] for x in s2_runs)/len(s2_runs), 4), "jit_compiles": c_a2 - c_b2},
        "step3_batching": s3_res,
        "step4_context": s4_res,
        "hbm_snapshots": final_st
    }
    with open(args.out, "w") as f: json.dump(res, f, indent=2)
    print("ENGINE_BENCH_COMPLETE", flush=True)

if __name__ == "__main__": main()
