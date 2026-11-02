import base64, json, os, subprocess, sys, time

def run_cmd(cmd, capture=True):
    shown = ["HF_TOKEN=<redacted>" if str(c).startswith("HF_TOKEN=") else c for c in cmd]
    print(f"RUN: {' '.join(shown)}", flush=True)
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if res.returncode != 0 and capture:
        print(f"ERR ({res.returncode}):\n{res.stderr}\n{res.stdout}", flush=True)
    return res

def get_usage():
    res = run_cmd(["colab", "--auth=oauth2", "usage"], capture=True)
    return res.stdout.strip()

def encode_file(path):
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode("ascii")

def main():
    argv = sys.argv[1:]
    diag = "--diag" in argv
    label = argv[argv.index("--label") + 1] if "--label" in argv else ("diag" if diag else "run")
    sess = "tpu-lfm25"
    if diag: print("=== DIAG MODE: diag_short_subrunner.py (4096 ctx, 30 min cap) ===", flush=True)
    print("=== Checking Initial Compute Units Usage ===", flush=True)
    u_start = get_usage()
    print("USAGE_START:\n" + u_start, flush=True)

    token_p = os.path.expanduser("~/.config/colab-cli/hf_token")
    hf_tok = open(token_p).read().strip() if os.path.exists(token_p) else ""
    here = "benchmarks/TPU-v5e/lfm25_200k"
    sub_file = "diag_short_subrunner.py" if diag else "lfm25_engine_subrunner.py"
    blobs = {
        "SUBRUNNER_B64": f"{here}/{sub_file}",
        "PATCHES_B64": f"{here}/tpu_v5e_patches.py",
        "PALLAS_B64": f"{here}/k8v4_pallas_kernel.py",
        "PROMPT_B64": f"{here}/lfm25_prompt.py",
        "STARTUP_B64": f"{here}/lfm25_startup.py",
        "CACHE_B64": f"{here}/lfm25_cache.py",
        "CACHE_NAMING_B64": "benchmarks/TPU-v5e/cache_naming.py",
    }

    t0_global = time.time()
    results = []
    try:
        print(f"=== Creating TPU v5e-1 Session: {sess} ===", flush=True)
        r_new = run_cmd(["colab", "--auth=oauth2", "new", "-s", sess, "--tpu", "v5e1"], capture=False)
        if r_new.returncode != 0: raise RuntimeError(f"Failed to create session {sess}")
        t_prov = time.time() - t0_global
        print(f"=== TPU session provisioned in {t_prov:.2f}s ===", flush=True)

        exec_cmd = ["colab", "--auth=oauth2", "exec", "-s", sess,
                    "--timeout", "1800" if diag else "86400",
                    "--env", f"T0_GLOBAL={t0_global}",
                    "--env", f"HF_TOKEN={hf_tok}",
                    "--env", f"LFM_RUN_LABEL={label}"]
        for k, p in blobs.items(): exec_cmd += ["--env", f"{k}={encode_file(p)}"]
        if diag: exec_cmd += ["--env", "LFM_DIAG=1"]
        if "--force-cold" in argv or os.environ.get("LFM_CACHE_FORCE_COLD") == "1": exec_cmd += ["--env", "LFM_CACHE_FORCE_COLD=1"]
        if "--no-upload" in argv: exec_cmd += ["--env", "LFM_NO_UPLOAD=1"]
        exec_cmd += ["-f", f"{here}/lfm25_coldstart_driver.py"]

        proc = subprocess.Popen(exec_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines = []
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lines.append(line)
        proc.wait()

        in_b = False
        for l in lines:
            s = l.strip()
            if "=== COLDSTART_RESULTS_JSONL ===" in s: in_b = True; continue
            if "=== END_COLDSTART_RESULTS ===" in s: in_b = False; continue
            if in_b:
                try: results.append(json.loads(s))
                except Exception: pass
        if results: results[0]["vm_provision_s"] = round(t_prov, 2)
        if proc.returncode != 0: print(f"=== DRIVER FAILED rc={proc.returncode} (results kept) ===", flush=True)
    finally:
        print(f"=== Stopping session {sess} ===", flush=True)
        run_cmd(["colab", "--auth=oauth2", "stop", "-s", sess], capture=False)
        print("=== Confirming active sessions ===", flush=True)
        run_cmd(["colab", "--auth=oauth2", "sessions"], capture=False)
        print("=== Final Compute Units Usage ===", flush=True)
        u_end = get_usage()
        print("USAGE_END:\n" + u_end, flush=True)

    if results:
        print("=== FINAL_LFM25_TPU_RESULTS ===", flush=True)
        print(json.dumps(results[0], indent=2), flush=True)
        out_name = "run_results_diag.json" if diag else ("run_results.json" if label == "run" else f"run_results_{label}.json")
        with open(f"benchmarks/TPU-v5e/lfm25_200k/{out_name}", "w") as f:
            json.dump(results[0], f, indent=2)

if __name__ == "__main__":
    main()
