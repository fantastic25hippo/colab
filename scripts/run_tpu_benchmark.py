import json, os, re, subprocess, sys, time

def run(cmd, capture=True):
    print(f"RUNNING: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if res.returncode != 0 and capture:
        print(f"ERROR ({res.returncode}):\n{res.stderr}\n{res.stdout}", flush=True)
    return res

def main():
    session_id = "tpu-bench"
    print("=== Step 1: Create TPU v5e-1 VM ===", flush=True)
    res_new = run(["colab", "--auth=oauth2", "new", "-s", session_id, "--tpu", "v5e1"], capture=False)
    if res_new.returncode != 0:
        print("Failed to create session. Exiting.", flush=True)
        sys.exit(1)

    try:
        print("=== Step 2: Upload benchmark scripts ===", flush=True)
        time.sleep(3)
        run(["colab", "--auth=oauth2", "upload", "-s", session_id, "scripts/tpu_bench_hook.py", "/content/tpu_bench_hook.py"])
        run(["colab", "--auth=oauth2", "upload", "-s", session_id, "scripts/tpu_run_single.py", "/content/tpu_run_single.py"])
        run(["colab", "--auth=oauth2", "upload", "-s", session_id, "scripts/tpu_vm_driver.py", "/content/tpu_vm_driver.py"])

        print("=== Step 3: Execute Benchmark Driver on VM ===", flush=True)
        proc = subprocess.Popen(
            ["colab", "--auth=oauth2", "exec", "-s", session_id, "--timeout", "86400", "-f", "scripts/tpu_vm_driver.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        captured_lines = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            captured_lines.append(line)
        proc.wait()
        print(f"Driver finished with code {proc.returncode}", flush=True)

        # Parse results
        results = []
        in_results = False
        for line in captured_lines:
            line_str = line.strip()
            if "=== BENCH_RESULTS_JSONL ===" in line_str:
                in_results = True
                continue
            if "=== END_BENCH_RESULTS ===" in line_str:
                in_results = False
                continue
            if in_results:
                try: results.append(json.loads(line_str))
                except Exception: pass

        out_dir = "benchmarks/TPU-v5e/overhead-two-models"
        os.makedirs(out_dir, exist_ok=True)
        jsonl_path = os.path.join(out_dir, "results.jsonl")
        with open(jsonl_path, "w") as f:
            for r in results: f.write(json.dumps(r) + "\n")
        print(f"Saved {len(results)} records to {jsonl_path}", flush=True)

    finally:
        print("=== Step 4: Stop VM immediately ===", flush=True)
        run(["colab", "--auth=oauth2", "stop", "-s", session_id], capture=False)
        print("=== Step 5: Verify 0 active sessions ===", flush=True)
        run(["colab", "--auth=oauth2", "sessions"], capture=False)

    print("=== Step 6: Check Final CU Usage ===", flush=True)
    res_usage = run(["colab", "--auth=oauth2", "usage"], capture=True)
    print(res_usage.stdout, flush=True)

if __name__ == "__main__":
    main()
