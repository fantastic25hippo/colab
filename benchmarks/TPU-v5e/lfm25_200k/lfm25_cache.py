"""LFM2.5-only XLA compile-cache naming, restore and publish.

Key = vLLM version + CACHE_KEY_ARGS of the subrunner + sha of the patches file.
The Qwen `tpu_compile_cache_b1_32k.tar` is never restored for LFM: its program
hashes do not match the LFM graphs, so it only wastes a download.

HF is imported lazily so this module imports (and its key logic is testable)
without vLLM, a token, or a network.
"""

import hashlib, os, re, shutil, subprocess, sys, time

_bench_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)
from cache_naming import build_cache_name

REPO_ID = "avgrocketman/tpu-compile-cache"
CACHE_DIR = "/dev/shm/tpu_compile_cache"


def read_patch_content(path=None):
    """Read patch content as bytes from standard locations, file path, str, or bytes."""
    if isinstance(path, bytes):
        return path
    if path is None:
        for candidate in (
            "/content/tpu_v5e_patches.py",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "tpu_v5e_patches.py"),
        ):
            if os.path.exists(candidate):
                path = candidate
                break
    elif isinstance(path, str):
        if not os.path.exists(path):
            candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
            if os.path.exists(candidate):
                path = candidate
    if path and isinstance(path, str) and os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    if isinstance(path, str):
        return path.encode("utf-8")
    return b""


def cache_filename(vllm_version, args_str, patch_content=None, model_slug="lfm2.5-24b-a2b", max_model_len=200000, buckets="16-4096-8192"):
    """Delegate to canonical naming helper."""
    patch_bytes = read_patch_content(patch_content)
    if os.environ.get("LFM_KV_INT8") == "1":
        kv_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kv_int8.py")
        kv_h = hashlib.sha256(open(kv_p, "rb").read()).hexdigest()[:8] if os.path.exists(kv_p) else "0"
        patch_bytes = patch_bytes + f"__kvint8_{kv_h}".encode("utf-8")
        buckets = f"{buckets}__kvint8"
    return build_cache_name(
        model_slug=model_slug,
        vllm_version=vllm_version,
        max_model_len=max_model_len,
        buckets=buckets,
        llm_args=args_str,
        patch_content=patch_bytes,
    )


def extract_cache_key_args(src):
    """Pull CACHE_KEY_ARGS (possibly split across string literals) out of a source."""
    m = re.search(r"CACHE_KEY_ARGS\s*=\s*\((.*?)\)", src, re.S)
    return "".join(re.findall(r'"([^"]*)"', m.group(1))) if m else ""


def read_vllm_version(venv_root="/dev/shm/vllm_env"):
    """Version of the installed vllm package (falls back to vllm-tpu, then unknown)."""
    import glob
    found = {}
    for p in glob.glob(f"{venv_root}/lib/python*/site-packages/vllm*dist-info/METADATA"):
        name = ver = None
        with open(p, errors="replace") as f:
            for line in f:
                if line.startswith("Name:"): name = line.split(":", 1)[1].strip()
                elif line.startswith("Version:"): ver = line.split(":", 1)[1].strip()
                elif line.strip() == "" and ver: break
        if name and ver: found[name] = ver
    return found.get("vllm") or found.get("vllm-tpu") or next(iter(found.values()), "unknown")


def clean_dir(path=CACHE_DIR):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def dir_stats(path=CACHE_DIR):
    n = mb = 0
    for name in os.listdir(path) if os.path.isdir(path) else []:
        n += 1
        try:
            mb += os.path.getsize(os.path.join(path, name)) / 1e6
        except OSError:
            pass
    return {"entries": n, "mb": round(mb, 1)}


def restore_cache(filename, repo_id=REPO_ID, root="/dev/shm"):
    """Download + unpack the LFM cache if it exists. Any failure means cold."""
    t0 = time.time()
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=root)
    except Exception as e:
        return {"hit": False, "reason": type(e).__name__, "probe_s": round(time.time() - t0, 2)}
    dl, mb = round(time.time() - t0, 2), round(os.path.getsize(p) / 1e6, 1)
    t1 = time.time()
    subprocess.run(["tar", "-xf", p, "-C", root], check=True)
    try:
        os.remove(p)
    except OSError:
        pass
    return {"hit": True, "dl_s": dl, "unpack_s": round(time.time() - t1, 2), "tar_mb": mb}


def publish_cache(filename, repo_id=REPO_ID, root="/dev/shm"):
    """Tar the restored/cold cache dir, upload it, then list the repo to verify."""
    tar_p = os.path.join(root, filename)
    t0 = time.time()
    subprocess.run(["tar", "-cf", tar_p, "-C", root, "tpu_compile_cache"], check=True)
    tar_s, mb = round(time.time() - t0, 2), round(os.path.getsize(tar_p) / 1e6, 1)
    from huggingface_hub import HfApi
    api, t1 = HfApi(), time.time()
    api.upload_file(path_or_fileobj=tar_p, path_in_repo=filename, repo_id=repo_id)
    up_s = round(time.time() - t1, 2)
    verified = filename in set(api.list_repo_files(repo_id))
    try:
        os.remove(tar_p)
    except OSError:
        pass
    return {"filename": filename, "tar_s": tar_s, "tar_mb": mb,
            "upload_s": up_s, "verified_listed": verified}


if __name__ == "__main__":
    _src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lfm25_engine_subrunner.py")).read()
    _a = extract_cache_key_args(_src)
    os.environ.pop("LFM_KV_INT8", None)
    print("OFF:", cache_filename("0.32.0", _a, read_patch_content(), max_model_len=100000, buckets="16-8192"))
    os.environ["LFM_KV_INT8"] = "1"
    print("INT8_16K_BYTES:", cache_filename("0.32.0", _a, read_patch_content(), max_model_len=16384, buckets="16-8192"))
    print("INT8_16K_STR:  ", cache_filename("0.32.0", _a, "tpu_v5e_patches.py", max_model_len=16384, buckets="16-8192"))

