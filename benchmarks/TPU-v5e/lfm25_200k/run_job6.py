#!/usr/bin/env python3
"""Host-side TPU v5e-1 Job 6 runner: build and warm start with LFM compile cache."""
import argparse, base64, os, shutil, subprocess, sys, threading, time

REPO = "/home/sahil/Desktop/sahil/colab"
HERE = f"{REPO}/benchmarks/TPU-v5e/lfm25_200k"
SESS = "tpu-lfm25"
MAX_LEN = "100000"
CACHE_TAR = "lfm25-24b-a2b_v5e-1_vllm0.32.0_tpuinf0.32.0_ml100000_patches-c75139b.tar.gz"
HF_REPO = "avgrocketman/tpu-compile-cache"

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
    src = open(f"{HERE}/lfm25_engine_subrunner.py").read()
    src = src.replace("max_model_len=200000", f"max_model_len={MAX_LEN}").replace('"max_model_len": 200000', f'"max_model_len": {MAX_LEN}')
    dst = f"/tmp/lfm25_engine_subrunner_{MAX_LEN}.py"
    open(dst, "w").write(src)
    return dst

def make_vm_driver():
    code = f'''#!/usr/bin/env python3
import base64, os, shutil, subprocess, sys, time
phase = os.environ.get("JOB6_PHASE", "build")
tar_name = "{CACHE_TAR}"
tar_path = f"/content/{{tar_name}}"
cache_dir = "/content/jax_cache_lfm25"
hf_tok = os.environ.get("HF_TOKEN", "")

if phase == "warm":
    print(f"[JOB6_WARM] Downloading {{tar_name}} from {HF_REPO}...", flush=True)
    t0 = time.time()
    try:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download(repo_id="{HF_REPO}", filename=tar_name, local_dir="/content", token=hf_tok or None)
        if f != tar_path: shutil.copyfile(f, tar_path)
    except Exception as e:
        cmd = ["curl", "-s", "-L", "-f"]
        if hf_tok: cmd += ["-H", f"Authorization: Bearer {{hf_tok}}"]
        cmd += [f"https://huggingface.co/{HF_REPO}/resolve/main/{{tar_name}}", "-o", tar_path]
        subprocess.run(cmd, check=True)
    t_dl = time.time() - t0
    sz = os.path.getsize(tar_path)
    print(f"[JOB6_WARM] Downloaded {{tar_name}} ({{sz}} bytes, {{sz/(1024*1024):.2f}} MB) in {{t_dl:.2f}}s", flush=True)
    t0 = time.time()
    os.makedirs(cache_dir, exist_ok=True)
    subprocess.run(["tar", "-xzf", tar_path, "-C", "/content"], check=True)
    t_ext = time.time() - t0
    cnt = len(os.listdir(cache_dir)) if os.path.exists(cache_dir) else 0
    print(f"[JOB6_WARM] Extracted {{cnt}} files to {{cache_dir}} in {{t_ext:.2f}}s (total cache prep: {{t_dl+t_ext:.2f}}s)", flush=True)

driver_code = base64.b64decode(os.environ["DRIVER_B64"]).decode("utf-8")
driver_path = "/dev/shm/lfm25_coldstart_driver.py"
with open(driver_path, "w") as f: f.write(driver_code)
p_drv = subprocess.Popen([sys.executable, "-u", driver_path], env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for l in p_drv.stdout: sys.stdout.write(l); sys.stdout.flush()
p_drv.wait()
if p_drv.returncode != 0: print(f"[JOB6_ERROR] Driver failed rc={{p_drv.returncode}}", flush=True); sys.exit(p_drv.returncode)

if phase == "build":
    entries = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
    print(f"[JOB6_BUILD] Found {{len(entries)}} files in {{cache_dir}}", flush=True)
    t0 = time.time(); subprocess.run(["tar", "-czf", tar_path, "-C", "/content", "jax_cache_lfm25"], check=True)
    t_tar = time.time() - t0; sz = os.path.getsize(tar_path)
    print(f"[JOB6_BUILD] Created {{tar_path}}: {{sz}} bytes ({{sz/(1024*1024):.2f}} MB) in {{t_tar:.2f}}s", flush=True)
    print(f"[JOB6_BUILD] Uploading {{tar_name}} to {HF_REPO}...", flush=True)
    t0 = time.time()
    from huggingface_hub import HfApi
    HfApi(token=hf_tok or None).upload_file(path_or_fileobj=tar_path, path_in_repo=tar_name, repo_id="{HF_REPO}")
    t_up = time.time() - t0
    print(f"[JOB6_BUILD] Upload complete: {{tar_name}} ({{sz}} bytes) uploaded in {{t_up:.2f}}s", flush=True)
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
            "--env", "JAX_COMPILATION_CACHE_DIR=/content/jax_cache_lfm25",
            "--env", "VLLM_XLA_CACHE_PATH=/content/jax_cache_lfm25",
            "--env", "LFM_SHORT_REQUEST=1", "--env", "LFM_RUN_LABEL=bearings_job6",
            "--env", "LFM_BEARINGS=1", "--env", "LFM_CACHE_FORCE_COLD=1",
            "--env", "LFM_NO_UPLOAD=1", "--env", f"LFM_MAX_MODEL_LEN={MAX_LEN}",
        ]
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
