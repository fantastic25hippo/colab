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
    out_dir = "benchmarks/TPU-v5e/qwen3-4b-final"
    os.makedirs(out_dir, exist_ok=True)
    all_results = []

    print("=== VM1: Start & Pre-Usage ===", flush=True)
    u_vm1_start = get_usage()
    print("USAGE_VM1_START:\n" + u_vm1_start, flush=True)
    s1 = "tpu-bench"
    try:
        r1 = run_cmd(["colab", "--auth=oauth2", "new", "-s", s1, "--tpu", "v5e1"], capture=False)
        if r1.returncode != 0: sys.exit(1)
        time.sleep(3)
        upload_auth_and_scripts(s1, [
            ("scripts/tpu_bench_hook.py", "/content/tpu_bench_hook.py"),
            ("scripts/tpu_run_single.py", "/content/tpu_run_single.py"),
            ("scripts/tpu_vm_driver.py", "/content/tpu_vm_driver.py"),
        ])
        rc1, lines1 = stream_exec(s1, "scripts/tpu_vm_driver.py")
        vm1_recs = parse_blocks(lines1, "BENCH_RESULTS_JSONL")
        all_results.extend(vm1_recs)
    finally:
        run_cmd(["colab", "--auth=oauth2", "stop", "-s", s1], capture=False)
        u_vm1_stop = get_usage()
        print("USAGE_VM1_STOP:\n" + u_vm1_stop, flush=True)

    print("=== VM2: Fresh Cold Start ===", flush=True)
    u_vm2_start = get_usage()
    print("USAGE_VM2_START:\n" + u_vm2_start, flush=True)
    s2 = "tpu-bench-cold"
    try:
        r2 = run_cmd(["colab", "--auth=oauth2", "new", "-s", s2, "--tpu", "v5e1"], capture=False)
        if r2.returncode != 0: sys.exit(1)
        time.sleep(3)
        upload_auth_and_scripts(s2, [
            ("scripts/tpu_bench_hook.py", "/content/tpu_bench_hook.py"),
            ("scripts/tpu_run_single.py", "/content/tpu_run_single.py"),
            ("scripts/tpu_cold_driver.py", "/content/tpu_cold_driver.py"),
        ])
        rc2, lines2 = stream_exec(s2, "scripts/tpu_cold_driver.py")
        vm2_recs = parse_blocks(lines2, "COLD_BENCH_RESULTS_JSONL")
        all_results.extend(vm2_recs)
    finally:
        run_cmd(["colab", "--auth=oauth2", "stop", "-s", s2], capture=False)
        u_vm2_stop = get_usage()
        print("USAGE_VM2_STOP:\n" + u_vm2_stop, flush=True)

    run_cmd(["colab", "--auth=oauth2", "sessions"], capture=False)
    jsonl_p = os.path.join(out_dir, "results.jsonl")
    with open(jsonl_p, "w") as f:
        for r in all_results: f.write(json.dumps(r) + "\n")
    print(f"SAVED {len(all_results)} records to {jsonl_p}", flush=True)

if __name__ == "__main__": main()
