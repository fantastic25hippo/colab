#!/usr/bin/env python3
"""Host-side TPU v5e-1 Job 10b runner: warm run (42f6059b)."""
import base64, os, shutil, subprocess, sys, threading, time
REPO, SESS, MAX_LEN = "/home/sahil/Desktop/sahil/colab", "tpu-lfm25", "100000"
CACHE_TAR = "tpu_compile_cache__lfm2.5-24b-a2b__vllm0.32.0__len100000__b16-8192__42f6059b.tar"
HERE = f"{REPO}/benchmarks/TPU-v5e/lfm25_200k"

def run_cmd(cmd, capture=True):
    print(f"RUN: {' '.join('HF_TOKEN=<redacted>' if str(c).startswith('HF_TOKEN=') else c for c in cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if res.returncode != 0 and capture: print(f"ERR ({res.returncode}):\n{res.stderr}\n{res.stdout}", flush=True)
    return res

def make_vm_driver():
    code = '''#!/usr/bin/env python3
import base64, os, subprocess, sys
dp = "/dev/shm/lfm25_coldstart_driver.py"; open(dp, "w").write(base64.b64decode(os.environ["DRIVER_B64"]).decode("utf-8"))
p = subprocess.Popen([sys.executable, "-u", dp], env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for l in p.stdout: sys.stdout.write(l); sys.stdout.flush()
p.wait()
if p.returncode != 0: print(f"[JOB10B_ERROR] Driver failed rc={p.returncode}", flush=True); sys.exit(p.returncode)
'''
    dst = "/tmp/job10b_vm_driver.py"; open(dst, "w").write(code); return dst

def main():
    log_p = sys.argv[1] if len(sys.argv) > 1 else f"{HERE}/job10b_warm.log"
    run_cmd(["colab", "--auth=oauth2", "ls"], capture=False)
    print("USAGE_START:\n" + (run_cmd(["colab", "--auth=oauth2", "usage"]).stdout or "").strip(), flush=True)
    hf_tok = open(os.path.expanduser("~/.config/colab-cli/hf_token")).read().strip()
    sub_dst = f"/tmp/lfm25_engine_subrunner_{MAX_LEN}.py"; shutil.copyfile(f"{HERE}/lfm25_engine_subrunner.py", sub_dst)
    blobs = {"DRIVER_B64": f"{HERE}/lfm25_coldstart_driver.py", "SUBRUNNER_B64": sub_dst, "PATCHES_B64": f"{HERE}/tpu_v5e_patches.py",
             "PALLAS_B64": f"{HERE}/k8v4_pallas_kernel.py", "PROMPT_B64": f"{HERE}/lfm25_prompt.py", "STARTUP_B64": f"{HERE}/lfm25_startup.py",
             "CACHE_B64": f"{HERE}/lfm25_cache.py", "CACHE_NAMING_B64": f"{REPO}/benchmarks/TPU-v5e/cache_naming.py"}
    vm_drv, t0 = make_vm_driver(), time.time()
    t_session_connect0 = t0
    try:
        if run_cmd(["colab", "--auth=oauth2", "new", "-s", SESS, "--tpu", "v5e1"], capture=False).returncode != 0: raise RuntimeError("new failed")
        t_upload0 = time.time()
        cmd = ["colab", "--auth=oauth2", "exec", "-s", SESS, "--timeout", "86400",
               "--env", f"T0_GLOBAL={t0}", "--env", f"T_SESSION_CONNECT0={t_session_connect0}",
               "--env", f"T_UPLOAD0={t_upload0}",
               "--env", f"HF_TOKEN={hf_tok}",
               "--env", "JOB8_PHASE=warm", "--env", "SKIP_JAX_PRECOMPILE=0", "--env", "LFM_SHORT_REQUEST=1", "--env", "LFM_RUN_LABEL=bearings_job10b",
               "--env", "LFM_BEARINGS=1", "--env", "LFM_DIAG=0", "--env", f"LFM_MAX_MODEL_LEN={MAX_LEN}", "--env", "LFM_CACHE_PARALLEL=1",
               "--env", "LFM_EARLY_IMPORT=1", "--env", "LFM_THP=1", "--env", "LFM_PHASE_DETAIL=1",
               "--env", "LFM_INPROC_INSPECT=1", "--env", "LFM_PATCH_FAST=1", "--env", "LFM_HF_ANON=1",
               "--env", "LFM_ABORT_ON_CACHE_MISS=1"]
        for k, p in blobs.items(): cmd += ["--env", f"{k}={base64.b64encode(open(p, 'rb').read()).decode('ascii')}"]
        t_exec0 = time.time()
        cmd += ["--env", f"T_EXEC0={t_exec0}"]
        proc = subprocess.Popen(cmd + ["-f", vm_drv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        last_line, last_time, done = ["(started)"], [time.time()], threading.Event()
        def mon():
            while not done.wait(60.0):
                now, sil = time.time(), time.time() - last_time[0]
                print(f"[STATUS t={now-t0:.1f}s silence={sil:.1f}s] last_line={last_line[0].strip()[:97]}", flush=True)
                if sil > 300.0: proc.terminate(); break
        threading.Thread(target=mon, daemon=True).start()
        with open(log_p, "w") as f:
            for l in proc.stdout:
                last_line[0] = l; last_time[0] = time.time(); sys.stdout.write(l); sys.stdout.flush(); f.write(l); f.flush()
        done.set(); print(f"=== EXEC finished rc={proc.wait()} wall={time.time()-t0:.1f}s ===", flush=True)
    finally:
        run_cmd(["colab", "--auth=oauth2", "stop", "-s", SESS], capture=False); run_cmd(["colab", "--auth=oauth2", "ls"], capture=False)
        print("USAGE_END:\n" + (run_cmd(["colab", "--auth=oauth2", "usage"]).stdout or "").strip(), flush=True)
if __name__ == "__main__": main()
