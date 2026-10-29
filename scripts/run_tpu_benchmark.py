import json, os, subprocess, sys, time

def run_cmd(cmd, capture=True):
    print(f"RUN: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if res.returncode != 0 and capture:
        print(f"ERR ({res.returncode}):\n{res.stderr}\n{res.stdout}", flush=True)
    return res

def upload_auth_and_scripts(sess, files):
    token_p = "/home/sahil/.config/colab-cli/hf_token"
    if os.path.exists(token_p):
        run_cmd(["colab", "--auth=oauth2", "upload", "-s", sess, token_p, "/content/hf_token"])
    for src, dst in files:
        run_cmd(["colab", "--auth=oauth2", "upload", "-s", sess, src, dst])

def stream_exec(sess, script_path):
    proc = subprocess.Popen(
        ["colab", "--auth=oauth2", "exec", "-s", sess, "--timeout", "86400", "-f", script_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush(); lines.append(line)
    proc.wait()
    return proc.returncode, lines

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

def get_usage():
    res = run_cmd(["colab", "--auth=oauth2", "usage"], capture=True)
    return res.stdout.strip()

def main():
    out_dir = "benchmarks/TPU-v5e/tpu-characterization"
    os.makedirs(out_dir, exist_ok=True)
    all_results = []

    print("=== Checking Initial Compute Units Usage ===", flush=True)
    u_start = get_usage()
    print("USAGE_START:\n" + u_start, flush=True)

    sess = "tpu-char"
    t_vm0 = time.time()
    try:
        r_new = run_cmd(["colab", "--auth=oauth2", "new", "-s", sess, "--tpu", "v5e1"], capture=False)
        if r_new.returncode != 0:
            print("Failed to create TPU session. Exiting.", flush=True)
            sys.exit(1)
        t_vm_ready = time.time() - t_vm0
        print(f"VM_CREATE_TO_READY_S: {t_vm_ready:.2f}", flush=True)
        time.sleep(3)

        upload_auth_and_scripts(sess, [
            ("scripts/tpu_bench_hook.py", "/content/tpu_bench_hook.py"),
            ("scripts/tpu_engine_runner.py", "/content/tpu_engine_runner.py"),
            ("scripts/tpu_driver.py", "/content/tpu_driver.py"),
        ])

        rc, lines = stream_exec(sess, "scripts/tpu_driver.py")
        recs = parse_blocks(lines, "BENCH_RESULTS_JSONL")
        if recs:
            for r in recs:
                if "cold_start_breakdown" in r:
                    r["cold_start_breakdown"]["vm_create_to_shell_ready_s"] = round(t_vm_ready, 2)
            all_results.extend(recs)
    finally:
        print("=== Stopping VM Session ===", flush=True)
        run_cmd(["colab", "--auth=oauth2", "stop", "-s", sess], capture=False)
        print("=== Checking Final Compute Units Usage ===", flush=True)
        u_stop = get_usage()
        print("USAGE_STOP:\n" + u_stop, flush=True)
        print("=== Verifying 0 Active Sessions ===", flush=True)
        run_cmd(["colab", "--auth=oauth2", "sessions"], capture=False)

    jsonl_p = os.path.join(out_dir, "results.jsonl")
    with open(jsonl_p, "w") as f:
        for r in all_results: f.write(json.dumps(r) + "\n")
    print(f"SAVED {len(all_results)} records to {jsonl_p}", flush=True)

if __name__ == "__main__": main()
