import base64, json, os, subprocess, sys, time

def run_cmd(cmd, capture=True):
    print(f"RUN: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if res.returncode != 0 and capture:
        print(f"ERR ({res.returncode}):\n{res.stderr}\n{res.stdout}", flush=True)
    return res

def get_usage():
    res = run_cmd(["colab", "--auth=oauth2", "usage"], capture=True)
    return res.stdout.strip()

def parse_blocks(lines, tag):
    res, in_b = [], False
    for l in lines:
        s = l.strip()
        if f"=== {tag} ===" in s: in_b = True; continue
        if f"=== END_{tag.split('_')[0]} ===" in s or f"=== END_{tag} ===" in s: in_b = False; continue
        if in_b:
            try: res.append(json.loads(s))
            except Exception: pass
    return res

def main():
    sess = "tpu-coldstart"
    print("=== Checking Initial Compute Units Usage ===", flush=True)
    u_start = get_usage()
    print("USAGE_START:\n" + u_start, flush=True)

    token_p = os.path.expanduser("~/.config/colab-cli/hf_token")
    hf_tok = open(token_p).read().strip() if os.path.exists(token_p) else ""
    with open("benchmarks/TPU-v5e/tpu_engine_subrunner.py", "rb") as f:
        sub_b64 = base64.b64encode(f.read()).decode("ascii")

    t0_global = time.time()
    results = []
    try:
        print(f"=== Creating TPU v5e-1 Session: {sess} ===", flush=True)
        r_new = run_cmd(["colab", "--auth=oauth2", "new", "-s", sess, "--tpu", "v5e1"], capture=False)
        if r_new.returncode != 0: raise RuntimeError(f"Failed to create session {sess}")
        t_provision_s = time.time() - t0_global
        print(f"=== Session provisioned in {t_provision_s:.2f}s (0 pre-exec uploads) ===", flush=True)

        t0_exec = time.time()
        print(f"=== Streaming execution of tpu_coldstart_driver.py (t0_global={t0_global:.2f}, t0_exec={t0_exec:.2f}) ===", flush=True)
        exec_cmd = [
            "colab", "--auth=oauth2", "exec", "-s", sess, "--timeout", "86400",
            "--env", f"T0_GLOBAL={t0_global}",
            "--env", f"SUBRUNNER_B64={sub_b64}",
            "--env", f"HF_TOKEN={hf_tok}",
            "-f", "benchmarks/TPU-v5e/tpu_coldstart_driver.py"
        ]
        proc = subprocess.Popen(exec_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines = []
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lines.append(line)
        proc.wait()
        if proc.returncode != 0:
            print(f"Driver execution failed with returncode {proc.returncode}", flush=True)

        results = parse_blocks(lines, "COLDSTART_RESULTS_JSONL")
        if results:
            results[0]["vm_provision_s"] = round(t_provision_s, 2)
    finally:
        print(f"=== Stopping session {sess} ===", flush=True)
        run_cmd(["colab", "--auth=oauth2", "stop", "-s", sess], capture=False)
        print("=== Confirming active sessions ===", flush=True)
        run_cmd(["colab", "--auth=oauth2", "sessions"], capture=False)
        print("=== Final Compute Units Usage ===", flush=True)
        u_end = get_usage()
        print("USAGE_END:\n" + u_end, flush=True)

    if results:
        print("=== FINAL_COLDSTART_RESULTS_RECORD ===", flush=True)
        print(json.dumps(results[0], indent=2), flush=True)
        with open("benchmarks/TPU-v5e/tpu-final/run5_floor_coldstart_results.json", "w") as f:
            json.dump(results[0], f, indent=2)

if __name__ == "__main__":
    main()
