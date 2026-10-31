import base64, io, json, os, subprocess, sys, tarfile, time

def run_cmd(cmd, capture=True):
    print(f"RUN: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if res.returncode != 0 and capture:
        print(f"ERR ({res.returncode}):\n{res.stderr}\n{res.stdout}", flush=True)
    return res

def get_usage():
    res = run_cmd(["colab", "--auth=oauth2", "usage"], capture=True)
    return res.stdout.strip()

def package_modules():
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tf:
        for f in ["__init__.py", "config_builder.py", "repack_q4_utils.py", "repack_checkpoint.py"]:
            p = os.path.join("benchmarks/TPU-v5e/lfm25_200k", f)
            tf.add(p, arcname=f"lfm25_200k/{f}")
    return base64.b64encode(bio.getvalue()).decode("ascii")

def main():
    sess = "repack-cpu"
    print("=== Checking Initial Compute Units Usage ===", flush=True)
    u_start = get_usage()
    print("USAGE_START:\n" + u_start, flush=True)

    token_p = os.path.expanduser("~/.config/colab-cli/hf_token")
    hf_tok = open(token_p).read().strip() if os.path.exists(token_p) else ""
    pkg_b64 = package_modules()

    t0_global = time.time()
    results = []
    try:
        print(f"=== Creating CPU Session: {sess} ===", flush=True)
        r_new = run_cmd(["colab", "--auth=oauth2", "new", "-s", sess], capture=False)
        if r_new.returncode != 0: raise RuntimeError(f"Failed to create session {sess}")
        t_prov = time.time() - t0_global
        print(f"=== CPU session provisioned in {t_prov:.2f}s ===", flush=True)

        exec_cmd = [
            "colab", "--auth=oauth2", "exec", "-s", sess, "--timeout", "86400",
            "--env", f"HF_TOKEN={hf_tok}",
            "--env", f"PACKAGE_B64={pkg_b64}",
            "-f", "benchmarks/TPU-v5e/lfm25_200k/repack_guest_driver.py"
        ]
        proc = subprocess.Popen(exec_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines = []
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lines.append(line)
        proc.wait()
        if proc.returncode != 0: raise RuntimeError(f"Repack driver failed with rc={proc.returncode}")

        in_b = False
        for l in lines:
            s = l.strip()
            if "=== REPACK_RESULTS_JSONL ===" in s: in_b = True; continue
            if "=== END_REPACK_RESULTS ===" in s: in_b = False; continue
            if in_b:
                try: results.append(json.loads(s))
                except Exception: pass
    finally:
        print(f"=== Stopping session {sess} ===", flush=True)
        run_cmd(["colab", "--auth=oauth2", "stop", "-s", sess], capture=False)
        print("=== Confirming active sessions ===", flush=True)
        run_cmd(["colab", "--auth=oauth2", "sessions"], capture=False)
        print("=== Final Compute Units Usage ===", flush=True)
        u_end = get_usage()
        print("USAGE_END:\n" + u_end, flush=True)

    if results:
        print("=== FINAL_REPACK_RESULT ===", flush=True)
        print(json.dumps(results[0], indent=2), flush=True)

if __name__ == "__main__":
    main()
