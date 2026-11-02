#!/usr/bin/env python3
"""Host-side TPU v5e-1 Job 6 runner: build and warm start with LFM compile cache."""
import argparse, base64, os, shutil, subprocess, sys, threading, time

REPO = "/home/sahil/Desktop/sahil/colab"
HERE = f"{REPO}/benchmarks/TPU-v5e/lfm25_200k"
SESS = "tpu-lfm25"
MAX_LEN = "100000"
CACHE_TAR = "lfm25-24b-a2b_v5e-1_vllm0.32.0_tpuinf0.32.0_ml100000_patches-c75139b.tar.gz"

def run_cmd(cmd, capture=True):
    shown = ["HF_TOKEN=<redacted>" if str(c).startswith("HF_TOKEN=") else c for c in cmd]
    print(f"RUN: {' '.join(shown)}", flush=True)
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if res.returncode != 0 and capture:
        print(f"ERR ({res.returncode}):\n{res.stderr}\n{res.stdout}", flush=True)
    return res

def encode_file(path):
    with open(path, "rb") as f: return base64.b64encode(f.read()).decode("ascii")

def prepare_subrunner():
    dst = f"/tmp/lfm25_engine_subrunner_{MAX_LEN}.py"
    shutil.copyfile(f"{HERE}/lfm25_engine_subrunner.py", dst)
    return dst

def make_vm_driver():
    code = '''#!/usr/bin/env python3
import base64, os, subprocess, sys
driver_code = base64.b64decode(os.environ["DRIVER_B64"]).decode("utf-8")
driver_path = "/dev/shm/lfm25_coldstart_driver.py"
with open(driver_path, "w") as f: f.write(driver_code)
p_drv = subprocess.Popen([sys.executable, "-u", driver_path], env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for l in p_drv.stdout: sys.stdout.write(l); sys.stdout.flush()
p_drv.wait()
if p_drv.returncode != 0: print(f"[JOB6_ERROR] Driver failed rc={p_drv.returncode}", flush=True); sys.exit(p_drv.returncode)
'''
    dst = "/tmp/job6_vm_driver.py"
    open(dst, "w").write(code)
    return dst

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["build", "warm"], default="build")
    args = p.parse_args()
    log_p = f"{HERE}/job6_{args.phase}.log"
    print(f"=== Job 6 Phase: {args.phase} (max_model_len={MAX_LEN}) ===", flush=True)
    run_cmd(["colab", "--auth=oauth2", "ls"], capture=False)
    u_start = run_cmd(["colab", "--auth=oauth2", "usage"], capture=True)
    print("USAGE_START:\n" + (u_start.stdout or "").strip(), flush=True)
    hf_tok = open(os.path.expanduser("~/.config/colab-cli/hf_token")).read().strip()
    blobs = {
        "DRIVER_B64": f"{HERE}/lfm25_coldstart_driver.py",
        "SUBRUNNER_B64": prepare_subrunner(),
        "PATCHES_B64": f"{HERE}/tpu_v5e_patches.py",
        "PALLAS_B64": f"{HERE}/k8v4_pallas_kernel.py",
        "PROMPT_B64": f"{HERE}/lfm25_prompt.py",
        "STARTUP_B64": f"{HERE}/lfm25_startup.py",
        "CACHE_B64": f"{HERE}/lfm25_cache.py",
        "CACHE_NAMING_B64": f"{REPO}/benchmarks/TPU-v5e/cache_naming.py",
    }
    vm_drv = make_vm_driver()
    t0_global = time.time()
    try:
        r_new = run_cmd(["colab", "--auth=oauth2", "new", "-s", SESS, "--tpu", "v5e1"], capture=False)
        if r_new.returncode != 0: raise RuntimeError(f"Failed to create session {SESS}")
        exec_cmd = [
            "colab", "--auth=oauth2", "exec", "-s", SESS, "--timeout", "86400",
            "--env", f"T0_GLOBAL={t0_global}", "--env", f"HF_TOKEN={hf_tok}",
            "--env", f"JOB6_PHASE={args.phase}", "--env", "SKIP_JAX_PRECOMPILE=0",
            "--env", "LFM_SHORT_REQUEST=1", "--env", "LFM_RUN_LABEL=bearings_job6",
            "--env", "LFM_BEARINGS=1",
            "--env", f"LFM_MAX_MODEL_LEN={MAX_LEN}",
        ]
        if os.environ.get("LFM_CACHE_FORCE_COLD") == "1": exec_cmd += ["--env", "LFM_CACHE_FORCE_COLD=1"]
        for k, path in blobs.items(): exec_cmd += ["--env", f"{k}={encode_file(path)}"]
        exec_cmd += ["-f", vm_drv]
        proc = subprocess.Popen(exec_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        last_line, last_time, done = ["(started)"], [time.time()], threading.Event()
        def mon():
            while not done.wait(60.0):
                now = time.time(); sil = now - last_time[0]
                ll = last_line[0].strip()[:97]
                print(f"[STATUS t={now-t0_global:.1f}s silence={sil:.1f}s] last_line={ll}", flush=True)
                if sil > 300.0:
                    print(f"STUCK ALERT: silence={sil:.1f}s > 300s. Terminating...", flush=True)
                    proc.terminate(); break
        threading.Thread(target=mon, daemon=True).start()
        with open(log_p, "w") as f:
            for l in proc.stdout:
                last_line[0] = l; last_time[0] = time.time()
                sys.stdout.write(l); sys.stdout.flush(); f.write(l); f.flush()
        done.set(); rc = proc.wait()
        print(f"=== EXEC finished rc={rc} wall={time.time()-t0_global:.1f}s ===", flush=True)
    finally:
        run_cmd(["colab", "--auth=oauth2", "stop", "-s", SESS], capture=False)
        run_cmd(["colab", "--auth=oauth2", "ls"], capture=False)
        u_end = run_cmd(["colab", "--auth=oauth2", "usage"], capture=True)
        print("USAGE_END:\n" + (u_end.stdout or "").strip(), flush=True)

if __name__ == "__main__": main()
