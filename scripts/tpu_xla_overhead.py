import os, sys, time, json, shutil, threading, subprocess

hb_stop = threading.Event()
def hb():
    while not hb_stop.wait(20):
        print(json.dumps({"stage": "hb_bootstrap", "time": time.time()}), flush=True)
threading.Thread(target=hb, daemon=True).start()

uv_bin = shutil.which("uv")
if not uv_bin:
    subprocess.run([sys.executable, "-m", "pip", "install", "uv"], check=True)
    uv_bin = shutil.which("uv") or "/root/.local/bin/uv" or "uv"
subprocess.run([uv_bin, "venv", "--python", "3.12", "/content/venv312"], check=True)
subprocess.run([uv_bin, "pip", "install", "--python", "/content/venv312/bin/python",
                "vllm-tpu", "hf-transfer", "huggingface_hub"], check=True)

WORKER_CODE = '''import os, sys, time, json, glob, logging, threading, subprocess, importlib.metadata
hb_stop, hb_stage = threading.Event(), ["start"]
def hb_worker():
    while not hb_stop.wait(20):
        print(json.dumps({"stage": "hb", "cur": hb_stage[0], "time": time.time()}), flush=True)
threading.Thread(target=hb_worker, daemon=True).start()
raw_logs = []
class Tee:
    def __init__(self, orig): self.orig = orig
    def write(self, s):
        self.orig.write(s)
        for l in s.splitlines():
            if l.strip(): raw_logs.append(l.strip())
    def flush(self): self.orig.flush()
    def fileno(self): return getattr(self.orig, "fileno", lambda: 1)()
    def isatty(self): return getattr(self.orig, "isatty", lambda: False)()
sys.stdout, sys.stderr = Tee(sys.stdout), Tee(sys.stderr)
pkgs = ["vllm", "vllm-tpu", "tpu-inference", "jax", "jaxlib", "libtpu", "libtpu-nightly", "torch"]
pkg_vers = {p: "not installed" for p in pkgs}
for p in pkgs:
    try: pkg_vers[p] = importlib.metadata.version(p)
    except Exception: pass
print("PKG_VERSIONS:", json.dumps(pkg_vers), flush=True)
hb_stage[0] = "download"
from huggingface_hub import snapshot_download
t_dl0 = time.time()
snapshot_download("google/gemma-4-E4B-it-qat-w4a16-ct", local_dir="/dev/shm/gemma", max_workers=12)
t_dl = time.time() - t_dl0
file_sizes = {os.path.relpath(p, "/dev/shm/gemma"): os.path.getsize(p)
              for p in glob.glob("/dev/shm/gemma/**", recursive=True) if os.path.isfile(p)}
st_bytes = sum(sz for fn, sz in file_sizes.items() if fn.endswith(".safetensors"))
print("DOWNLOAD_INFO:", json.dumps({"seconds": t_dl, "safetensors_bytes": st_bytes, "files": file_sizes}), flush=True)
try: import tpu_inference
except Exception: pass
import jax
m0 = jax.devices()[0].memory_stats()
print("M0_MEMORY:", json.dumps(m0), flush=True)
hb_stage[0] = "compile"
from vllm import LLM, SamplingParams
t_eng0 = time.time()
try:
    llm = LLM(model="/dev/shm/gemma", max_num_seqs=1, max_num_batched_tokens=512,
              max_model_len=8192, limit_mm_per_prompt={"image": 0, "audio": 0},
              gpu_memory_utilization=0.92)
    t_compile = time.time() - t_eng0
    print(f"COMPILE_SECONDS: {t_compile}", flush=True)
except Exception:
    import traceback
    print("ENGINE_FAILED:", traceback.format_exc(), flush=True)
    hb_stop.set()
    sys.exit(1)
m3 = jax.devices()[0].memory_stats()
print("M3_MEMORY:", json.dumps(m3), flush=True)
hb_stage[0] = "generate"
tok = llm.get_tokenizer()
text = ("The exploration of deep space has fascinated humanity for centuries. Astronomers observe distant stars, "
        "planets, and galaxies through powerful telescopes, searching for clues about the origins of the cosmos. "
        "Each discovery unveils new mysteries about dark matter, black holes, and the cosmic microwave background. "
        "Scientists formulate mathematical models to test gravitational theories across expanding spacetime dimensions. "
        "Deep space probes journey toward interstellar space carrying messages of our civilization. ")
toks = tok.encode(text)
while len(toks) < 200: toks += toks
prompt_str = tok.decode(toks[:200])
actual_p_len = len(tok.encode(prompt_str))
print(f"PROMPT_TOKENS: {actual_p_len}", flush=True)
sp = SamplingParams(max_tokens=128, min_tokens=128, temperature=0.0, ignore_eos=True)
t_g0 = time.time()
out = llm.generate([prompt_str], sp)[0]
t_g1 = time.time()
gen_wall, gen_toks = t_g1 - t_g0, len(out.outputs[0].token_ids)
ttft, m = None, getattr(out, "metrics", None)
if m:
    ft, ar = getattr(m, "first_token_time", None), (getattr(m, "arrival_time", None) or getattr(m, "first_scheduled_time", None))
    if ft is not None and ar is not None: ttft = ft - ar
    elif getattr(m, "time_to_first_token", None) is not None: ttft = m.time_to_first_token
dec_tps = (gen_toks - 1) / (gen_wall - ttft) if (ttft and gen_wall > ttft) else (gen_toks / gen_wall if gen_wall > 0 else 0)
print("GEN_INFO:", json.dumps({"wall_sec": gen_wall, "tokens": gen_toks, "ttft": ttft, "decode_tok_s": dec_tps}), flush=True)
m4 = jax.devices()[0].memory_stats()
print("M4_MEMORY:", json.dumps(m4), flush=True)
hb_stop.set()
for l in raw_logs:
    if any(k in l.lower() for k in ["weight", "kv cache", "gpu block", "kv block", "memory", "profil"]):
        print("RELEVANT_LOG:", l, flush=True)
results = {"versions": pkg_vers, "download": {"seconds": t_dl, "safetensors_bytes": st_bytes, "files": file_sizes},
           "m0": m0, "compile_seconds": t_compile, "m3": m3,
           "gen": {"wall_sec": gen_wall, "tokens": gen_toks, "ttft": ttft, "decode_tok_s": dec_tps},
           "m4": m4}
print("FINAL_OVERHEAD_JSON:", json.dumps(results), flush=True)
'''

with open("/content/worker.py", "w") as f:
    f.write(WORKER_CODE)

os.makedirs("/content/tpu_compile_cache", exist_ok=True)
env = os.environ.copy()
env.update({"JAX_COMPILATION_CACHE_DIR": "/content/tpu_compile_cache",
            "VLLM_XLA_CACHE_PATH": "/content/tpu_compile_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "0",
            "VLLM_USE_V1": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
hb_stop.set()
res = subprocess.run(["/content/venv312/bin/python", "-u", "/content/worker.py"], env=env)
sys.exit(res.returncode)
