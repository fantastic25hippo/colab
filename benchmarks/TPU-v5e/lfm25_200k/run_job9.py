#!/usr/bin/env python3
"""Host-side TPU v5e-1 Job 9 runner: warm diagnostic run only (LFM_DIAG=1)."""
import base64, io, os, shutil, subprocess, sys, tarfile, threading, time
REPO, SESS, MAX_LEN = "/home/sahil/Desktop/sahil/colab", "tpu-lfm25", "100000"
HERE, PFX = f"{REPO}/benchmarks/TPU-v5e/lfm25_200k", "=== JOB9_STACKS_TAR_GZ_B64:"

def run_cmd(cmd, capture=True):
    print(f"RUN: {' '.join('HF_TOKEN=<redacted>' if str(c).startswith('HF_TOKEN=') else c for c in cmd)}", flush=True)
    res = subprocess.run(cmd, capture_output=capture, text=True)
    if res.returncode != 0 and capture: print(f"ERR ({res.returncode}):\n{res.stderr}\n{res.stdout}", flush=True)
    return res

def make_vm_driver():
    code = '''#!/usr/bin/env python3
import base64, io, os, shutil, subprocess, sys, tarfile
print(f"[JOB9_VM] sys py-spy: {shutil.which('py-spy') or 'not in /usr/bin (checked /dev/shm/vllm_env/bin/py-spy in subrunner)'}", flush=True)
dp = "/dev/shm/lfm25_coldstart_driver.py"; open(dp, "w").write(base64.b64decode(os.environ["DRIVER_B64"]).decode("utf-8"))
env = os.environ.copy(); env.pop("LFM_DIAG", None); env["LFM_JOB9_DIAG"] = "1"
p = subprocess.Popen([sys.executable, "-u", dp], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for l in p.stdout: sys.stdout.write(l); sys.stdout.flush()
p.wait(); sdir = "/content/job9_stacks"
if os.path.isdir(sdir):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for fn in sorted(os.listdir(sdir)): tf.add(os.path.join(sdir, fn), arcname=fn)
    print(f"=== JOB9_STACKS_TAR_GZ_B64:{base64.b64encode(buf.getvalue()).decode('ascii')} ===", flush=True)
sys.exit(p.returncode)
'''
    dst = "/tmp/job9_vm_driver.py"; open(dst, "w").write(code); return dst

def main():
    log_p, sdir = f"{HERE}/job9_warm.log", f"{HERE}/job9_stacks"
    if os.path.isdir(sdir): shutil.rmtree(sdir)
    os.makedirs(sdir, exist_ok=True); run_cmd(["colab", "--auth=oauth2", "ls"], capture=False)
    print("USAGE_START:\n" + (run_cmd(["colab", "--auth=oauth2", "usage"]).stdout or "").strip(), flush=True)
    hf_tok = open(os.path.expanduser("~/.config/colab-cli/hf_token")).read().strip()
    sub_dst = f"/tmp/lfm25_engine_subrunner_{MAX_LEN}.py"; shutil.copyfile(f"{HERE}/lfm25_engine_subrunner.py", sub_dst)
    blobs = {"DRIVER_B64": f"{HERE}/lfm25_coldstart_driver.py", "SUBRUNNER_B64": sub_dst, "PATCHES_B64": f"{HERE}/tpu_v5e_patches.py",
             "PALLAS_B64": f"{HERE}/k8v4_pallas_kernel.py", "PROMPT_B64": f"{HERE}/lfm25_prompt.py", "STARTUP_B64": f"{HERE}/lfm25_startup.py",
             "CACHE_B64": f"{HERE}/lfm25_cache.py", "CACHE_NAMING_B64": f"{REPO}/benchmarks/TPU-v5e/cache_naming.py"}
    vm_drv, t0 = make_vm_driver(), time.time()
    try:
        if run_cmd(["colab", "--auth=oauth2", "new", "-s", SESS, "--tpu", "v5e1"], capture=False).returncode != 0: raise RuntimeError("new failed")
        cmd = ["colab", "--auth=oauth2", "exec", "-s", SESS, "--timeout", "86400", "--env", f"T0_GLOBAL={t0}", "--env", f"HF_TOKEN={hf_tok}",
               "--env", "JOB8_PHASE=warm", "--env", "SKIP_JAX_PRECOMPILE=0", "--env", "LFM_SHORT_REQUEST=1", "--env", "LFM_RUN_LABEL=bearings_job9",
               "--env", "LFM_BEARINGS=1", "--env", "LFM_DIAG=1", "--env", f"LFM_MAX_MODEL_LEN={MAX_LEN}", "--env", "LFM_CACHE_PARALLEL=1",
               "--env", "LFM_EARLY_IMPORT=1", "--env", "LFM_THP=1", "--env", "LFM_PHASE_DETAIL=1"]
        for k, p in blobs.items(): cmd += ["--env", f"{k}={base64.b64encode(open(p, 'rb').read()).decode('ascii')}"]
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
                if l.startswith(PFX):
                    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(l[len(PFX):].split(" ===")[0].strip())), mode="r:gz") as tf: tf.extractall(sdir)
                    continue
                last_line[0] = l; last_time[0] = time.time(); sys.stdout.write(l); sys.stdout.flush(); f.write(l); f.flush()
        done.set(); print(f"=== EXEC finished rc={proc.wait()} wall={time.time()-t0:.1f}s ===", flush=True)
    finally:
        run_cmd(["colab", "--auth=oauth2", "stop", "-s", SESS], capture=False); run_cmd(["colab", "--auth=oauth2", "ls"], capture=False)
        print("USAGE_END:\n" + (run_cmd(["colab", "--auth=oauth2", "usage"]).stdout or "").strip(), flush=True)
if __name__ == "__main__": main()
