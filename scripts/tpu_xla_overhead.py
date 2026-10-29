import os, sys, time, json, shutil, threading, subprocess

t_inst0, hb_stop = time.time(), threading.Event()
def hb():
    while not hb_stop.wait(20):
        print(json.dumps({"stage": "hb_bootstrap", "time": time.time()}), flush=True)
threading.Thread(target=hb, daemon=True).start()

uv_bin = shutil.which("uv")
if not uv_bin:
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    uv_bin = shutil.which("uv") or "/root/.local/bin/uv"
subprocess.run([uv_bin, "venv", "--python", "3.12", "/content/venv312"], check=True)
subprocess.run([uv_bin, "pip", "install", "--python", "/content/venv312/bin/python",
                "vllm-tpu==0.32.0", "hf-transfer", "huggingface_hub", "py-spy"], check=True)
t_install = time.time() - t_inst0; print(f"INSTALL_SECONDS: {t_install:.2f}", flush=True)

WORKER_CODE = '''import os, sys, time, json, glob, threading, subprocess, importlib.metadata
def main():
    hb_stop, hb_stage = threading.Event(), ["start"]
    def hb_worker():
        while not hb_stop.wait(20):
            print(json.dumps({"stage": "hb", "cur": hb_stage[0], "time": time.time()}), flush=True)
    threading.Thread(target=hb_worker, daemon=True).start()
    pkgs = ["vllm", "vllm-tpu", "tpu-inference", "jax", "jaxlib", "libtpu", "torch"]
    pkg_vers = {}
    for p in pkgs:
        try: pkg_vers[p] = importlib.metadata.version(p)
        except Exception:
            try: pkg_vers[p] = importlib.metadata.version(p.replace("-", "_"))
            except Exception: pkg_vers[p] = "not installed"
    print("PKG_VERSIONS:", json.dumps(pkg_vers), flush=True)
    hb_stage[0] = "download"
    from huggingface_hub import snapshot_download
    t_dl0 = time.time()
    snapshot_download("google/gemma-4-E4B-it-qat-w4a16-ct", local_dir="/dev/shm/gemma", max_workers=12)
    t_dl = time.time() - t_dl0
    file_sizes = {os.path.relpath(p, "/dev/shm/gemma"): os.path.getsize(p) for p in glob.glob("/dev/shm/gemma/**", recursive=True) if os.path.isfile(p)}
    st_bytes = sum(sz for fn, sz in file_sizes.items() if fn.endswith(".safetensors"))
    print("DOWNLOAD_INFO:", json.dumps({"seconds": t_dl, "safetensors_bytes": st_bytes, "files": file_sizes}), flush=True)
    hb_stage[0] = "compile"
    from vllm import LLM, SamplingParams
    t_eng0 = time.time()
    try:
        llm = LLM(model="/dev/shm/gemma", max_num_seqs=1, max_num_batched_tokens=512, max_model_len=8192, gpu_memory_utilization=0.92)
        t_compile = time.time() - t_eng0; print(f"COMPILE_SECONDS: {t_compile:.2f}", flush=True)
    except Exception:
        import traceback; print("ENGINE_FAILED:", traceback.format_exc(), flush=True); hb_stop.set(); sys.exit(1)
    hb_stage[0] = "generate"
    try:
        try: tok = llm.get_tokenizer()
        except Exception:
            from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained("/dev/shm/gemma")
        text = "The exploration of deep space has fascinated humanity for centuries. Astronomers observe distant stars and galaxies through powerful telescopes, searching for clues about cosmic microwave background and spacetime dimensions. " * 8
        toks = tok.encode(text)[:200]
        prompt_str = tok.decode(toks)
        print(f"PROMPT_TOKENS: {len(toks)}", flush=True)
        sp = SamplingParams(max_tokens=128, min_tokens=128, temperature=0.0, ignore_eos=True)
        t_g0 = time.time()
        out = llm.generate([prompt_str], sp)[0]
        gen_wall, gen_toks = time.time() - t_g0, len(out.outputs[0].token_ids)
        ttft, m = None, getattr(out, "metrics", None)
        if m:
            ft, ar = getattr(m, "first_token_time", None), (getattr(m, "arrival_time", None) or getattr(m, "first_scheduled_time", None))
            if ft is not None and ar is not None: ttft = ft - ar
            elif getattr(m, "time_to_first_token", None) is not None: ttft = m.time_to_first_token
        dec_tps = (gen_toks - 1) / (gen_wall - ttft) if (ttft and gen_wall > ttft) else (gen_toks / gen_wall if gen_wall > 0 else 0)
        print("GEN_INFO:", json.dumps({"wall_sec": gen_wall, "tokens": gen_toks, "ttft": ttft, "decode_tok_s": dec_tps}), flush=True)
    except Exception:
        import traceback; print("GENERATION_FAILED:", traceback.format_exc(), flush=True)
        gen_wall, gen_toks, ttft, dec_tps = 0, 0, None, 0
    hb_stage[0] = "save_cache"
    try:
        c_dir = "/content/tpu_compile_cache"
        c_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(c_dir) for f in fn]
        c_bytes = sum(os.path.getsize(f) for f in c_files)
        print(f"CACHE_DIR_INFO: count={len(c_files)} bytes={c_bytes}", flush=True)
        tar_p = "/content/tpu_compile_cache.tar"
        subprocess.run(["tar", "-cf", tar_p, "-C", c_dir, "."], check=True)
        v_info = {"packages": pkg_vers, "engine_settings": {"max_num_seqs": 1, "max_num_batched_tokens": 512, "max_model_len": 8192, "gpu_memory_utilization": 0.92, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}}
        with open("/content/versions.json", "w") as f: json.dump(v_info, f, indent=2)
        hf_tok = os.environ.get("HF_TOKEN")
        if not hf_tok:
            try: hf_tok = __import__("google.colab.userdata", fromlist=["userdata"]).userdata.get("HF_TOKEN")
            except Exception: pass
        if not hf_tok:
            try: hf_tok = __import__("huggingface_hub").HfFolder.get_token()
            except Exception: pass
        if not hf_tok:
            print("CACHE_UPLOAD_STATUS: skipped (no HF_TOKEN found)", flush=True)
        else:
            try:
                from huggingface_hub import HfApi
                api = HfApi(token=hf_tok)
                t_up0, rid = time.time(), "avgrocketman/tpu-compile-cache"
                api.create_repo(repo_id=rid, repo_type="model", private=True, exist_ok=True)
                pfx = "gemma4-e4b-qat/vllm-tpu-0.32.0_jax-0.11.2"
                api.upload_file(path_or_fileobj=tar_p, path_in_repo=f"{pfx}/cache.tar", repo_id=rid, repo_type="model")
                api.upload_file(path_or_fileobj="/content/versions.json", path_in_repo=f"{pfx}/versions.json", repo_id=rid, repo_type="model")
                print(f"CACHE_UPLOAD_SUCCESS: seconds={time.time()-t_up0:.2f}", flush=True)
            except Exception as e: print(f"CACHE_UPLOAD_ERROR: {type(e).__name__}: {e}", flush=True)
    except Exception:
        import traceback; print("CACHE_SAVE_FAILED:", traceback.format_exc(), flush=True)
    hb_stop.set()
    results = {"versions": pkg_vers, "download": {"seconds": t_dl, "safetensors_bytes": st_bytes},
               "compile_seconds": t_compile if "t_compile" in locals() else None,
               "gen": {"wall_sec": gen_wall, "tokens": gen_toks, "ttft": ttft, "decode_tok_s": dec_tps},
               "cache": {"files": len(c_files) if "c_files" in locals() else 0, "bytes": c_bytes if "c_bytes" in locals() else 0}}
    print("FINAL_OVERHEAD_JSON:", json.dumps(results), flush=True)
if __name__ == "__main__":
    main()
'''

with open("/content/worker.py", "w") as f:
    f.write(WORKER_CODE)

os.makedirs("/content/tpu_compile_cache", exist_ok=True)
env = os.environ.copy()
env.update({"JAX_COMPILATION_CACHE_DIR": "/content/tpu_compile_cache", "VLLM_XLA_CACHE_PATH": "/content/tpu_compile_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
for k in ["VLLM_USE_V1", "VLLM_ENABLE_V1_MULTIPROCESSING", "SKIP_JAX_PRECOMPILE"]:
    env.pop(k, None)
if "HF_TOKEN" not in env:
    try:
        tok = __import__("google.colab.userdata", fromlist=["userdata"]).userdata.get("HF_TOKEN")
        if tok: env["HF_TOKEN"] = tok
    except Exception: pass

proc = subprocess.Popen(["/content/venv312/bin/python", "-u", "/content/worker.py"],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
last_line_time = [time.time()]
def watchdog():
    while proc.poll() is None:
        time.sleep(10)
        if time.time() - last_line_time[0] > 900:
            print("WATCHDOG: 15m silence. Running py-spy dump...", flush=True)
            pyspy = shutil.which("py-spy") or "/content/venv312/bin/py-spy" or "/root/.local/bin/py-spy"
            try:
                out = subprocess.run([pyspy, "dump", "--pid", str(proc.pid)], capture_output=True, text=True, timeout=30)
                print("PY_SPY_DUMP:\n" + out.stdout + out.stderr, flush=True)
            except Exception as e: print(f"PY_SPY_ERROR: {e}", flush=True)
            proc.kill()
            break
threading.Thread(target=watchdog, daemon=True).start()

for line in proc.stdout:
    last_line_time[0] = time.time()
    sys.stdout.write(line); sys.stdout.flush()
proc.wait(); hb_stop.set(); sys.exit(proc.returncode)
