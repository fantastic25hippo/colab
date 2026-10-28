"""Worker-count sweep: download the public LFM Q4 shard repo with different max_workers.
Args: [WORKERS=4,8,12,16,24] [REPS=2] [DESTS=shm,disk] (shm=/dev/shm, disk=/tmp; aliases because the agy hook blocks /tmp as an argument) (colab run forwards script args, not env). Prints one JSON line per run, then a table."""
import json, os, shutil, subprocess, sys, threading, time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"
try:
    import hf_transfer  # noqa: F401
    from huggingface_hub import HfApi, snapshot_download
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])
    from huggingface_hub import HfApi, snapshot_download

REPO = "avgrocketman/LFM2.5-24B-A2B-Q4-shards"
WORKERS = [int(x) for x in (sys.argv[1:2] or ["4,8,12,16,24"])[0].split(",")]
REPS = int((sys.argv[2:3] or ["2"])[0])
ALIAS = {"shm": "/dev/shm", "disk": "/tmp"}
DESTS = [ALIAS.get(x, x) for x in (sys.argv[3:4] or ["shm,disk"])[0].split(",")]
stage, t_stage, stop = "init", time.time(), threading.Event()


def heartbeat():
    while not stop.wait(20):
        print(f"[heartbeat] {stage} {time.time() - t_stage:.0f}s", flush=True)


threading.Thread(target=heartbeat, daemon=True).start()
info = HfApi().model_info(REPO, files_metadata=True)
total = sum(s.size or 0 for s in info.siblings)
print(json.dumps({"repo": REPO, "files": len(info.siblings), "bytes": total, "cpus": os.cpu_count()}), flush=True)
runs = []
for dest in DESTS:
    free = shutil.disk_usage(dest).free
    if free < total * 1.1:
        print(json.dumps({"dest": dest, "skipped": "not enough space", "free": free}), flush=True)
        continue
    for rep in range(REPS):
        order = WORKERS[rep % len(WORKERS):] + WORKERS[:rep % len(WORKERS)]
        if rep % 2:
            order = order[::-1]
        for w in order:
            d = os.path.join(dest, "sweep")
            shutil.rmtree(d, ignore_errors=True)
            stage, t_stage = f"{dest} w={w} rep={rep}", time.time()
            t0 = time.time()
            try:
                snapshot_download(REPO, local_dir=d, cache_dir=os.path.join(d, ".c"), max_workers=w)
                sec = time.time() - t0
                row = {"dest": dest, "workers": w, "rep": rep, "sec": round(sec, 2), "mb_s": round(total / 1e6 / sec, 1)}
            except Exception as e:
                row = {"dest": dest, "workers": w, "rep": rep, "error": repr(e)[:200]}
            runs.append(row)
            print(json.dumps(row), flush=True)
            shutil.rmtree(d, ignore_errors=True)
stop.set()
print("\ndest workers mean_mb_s n")
for dest in DESTS:
    for w in sorted(set(WORKERS)):
        v = [r["mb_s"] for r in runs if r["dest"] == dest and r["workers"] == w and "mb_s" in r]
        if v:
            print(dest, w, round(sum(v) / len(v), 1), len(v))
