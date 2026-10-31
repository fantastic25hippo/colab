import json, os, subprocess, sys, time

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
    sess = "tpu-fast-venv"
    print("=== Checking Initial Compute Units Usage ===", flush=True)
    u_start = get_usage()
    print("USAGE_START:\n" + u_start, flush=True)

    results = []
    try:
        print(f"=== Creating TPU v5e-1 Session: {sess} ===", flush=True)
        r_new = run_cmd(["colab", "--auth=oauth2", "new", "-s", sess, "--tpu", "v5e1"], capture=False)
        if r_new.returncode != 0: raise RuntimeError(f"Failed to create session {sess}")
        time.sleep(5)

        print("=== Checking SSH connectivity ===", flush=True)
        ssh_key = os.path.expanduser("~/.ssh/colab_ed25519")
        ssh_cmd = [
            "ssh", "-i", ssh_key,
            "-o", f"ProxyCommand=colab --auth=oauth2 ssh --proxy-mode -s {sess} -i {ssh_key}",
            "-o", "StrictHostKeyChecking=no", "root@colab", "echo SSH_CONNECTED && uptime"
        ]
        r_ssh = run_cmd(ssh_cmd, capture=True)
        print("SSH_TEST_OUTPUT:\n" + r_ssh.stdout.strip(), flush=True)

        print("=== Uploading scripts & auth token ===", flush=True)
        token_p = os.path.expanduser("~/.config/colab-cli/hf_token")
        if os.path.exists(token_p):
            run_cmd(["colab", "--auth=oauth2", "upload", "-s", sess, token_p, "/content/hf_token"])
        run_cmd(["colab", "--auth=oauth2", "upload", "-s", sess, "benchmarks/TPU-v5e/tpu_venv_builder.py", "/content/tpu_venv_builder.py"])

        print("=== Streaming execution of tpu_venv_builder.py ===", flush=True)
        proc = subprocess.Popen(
            ["colab", "--auth=oauth2", "exec", "-s", sess, "--timeout", "86400", "-f", "benchmarks/TPU-v5e/tpu_venv_builder.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        lines = []
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lines.append(line)
        proc.wait()
        if proc.returncode != 0:
            print(f"Script execution failed with returncode {proc.returncode}", flush=True)

        results = parse_blocks(lines, "VENV_BUILD_RESULTS_JSONL")
    finally:
        print(f"=== Stopping session {sess} ===", flush=True)
        run_cmd(["colab", "--auth=oauth2", "stop", "-s", sess], capture=False)
        print("=== Confirming active sessions ===", flush=True)
        run_cmd(["colab", "--auth=oauth2", "sessions"], capture=False)
        print("=== Final Compute Units Usage ===", flush=True)
        u_end = get_usage()
        print("USAGE_END:\n" + u_end, flush=True)

    if results:
        print("=== FINAL_RESULTS_RECORD ===", flush=True)
        print(json.dumps(results[0], indent=2), flush=True)
        with open("benchmarks/TPU-v5e/tpu-final/run1_venv_results.json", "w") as f:
            json.dump(results[0], f, indent=2)

if __name__ == "__main__":
    main()
