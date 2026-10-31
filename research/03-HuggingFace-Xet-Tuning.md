# 03 — Hugging Face Hub Transfer and Xet Tuning

Consolidated from `Hugging Face Download Optimization Research.pdf` (18 pages) and its `HuggingFace-Optimization-Summary.md`. Scope: 15–80 GB checkpoints pulled onto ephemeral cloud GPUs such as a Colab L4 (12 vCPU, ~52 GB RAM). Claims retain the original measured/documented/inferred labels where given.

## 1. What changed in 2025–2026

- HF moved from **monolithic Git LFS + `hf_transfer`** (legacy HTTP range-request accelerator) to **content-addressed, chunk-deduplicated storage** via Rust `hf-xet` / `xet-core`.
- Xet splits files into **~64 KB chunks**, hashes them, aggregates unique chunks into **~64 MB immutable Xorbs**, and keeps **30–50 MB metadata shards** in a CAS index.
- `hf_xet` starts at **1 stream** and adaptively scales to **64 concurrent streams** per transfer based on bandwidth. Official docs say the default already saturates data-center links.
- Consequence: **legacy tuning heuristics now degrade throughput**. More Python workers do not equal more bandwidth; they add GIL, lock, and tmpfs contention.

## 2. Two-tier concurrency

- **Outer layer:** `snapshot_download()` / `hf download` exposes `max_workers`, default **8**. One thread = one file, via `thread_map(max_workers=8)` in `src/huggingface_hub/_snapshot_download.py`.
- **Inner layer:** when the repo is Xet-backed and `hf_xet` is installed, each file is handed to Rust `xet-core`, which manages byte-range concurrency internally (1→64 streams).
- `max_workers=4` is the conservative recommendation from upstream Xet docs (`2–4` safe range on overlayfs). **In-repo L4 measurement (`../benchmarks/HF-transfer/L4/SWEEP.md`, 16 shards, 14.88 GB into `/dev/shm`):** `max_workers=12` achieved the highest 3-rep mean (**920.8 MB/s**, peak **1,119.7 MB/s**, best single run **16.60 s / 896 MB/s**), while `w=4,8,12,16` all hit ~940–1,120 MB/s on unthrottled reps and 190–316 MB/s on throttled reps. When staging into tmpfs `/dev/shm` (bypassing overlayfs lock stalls), **`max_workers=8..12`** is verified safe and fast.

## 3. Deprecated / no-op variables (important)

| Variable | Status | Detail |
| :--- | :--- | :--- |
| `HF_HUB_ENABLE_HF_TRANSFER=1` | **Deprecated no-op** on Xet repos | Hub now always uses Xet. Verified with `ss -tnp`: traffic goes to GCS `34.107.x.x`/`34.149.x.x` even with the flag. A `constants.py` logic bug + hidden `DeprecationWarning` made the no-op silent; fixed in PR #4220 (2026-05-15). Still wrongly recommended by some 2026 blogs/notebooks. |
| `HF_XET_NUM_CONCURRENT_RANGE_GETS` | **Removed in hf_xet 1.3.0** | Superseded by adaptive concurrency. Values like 1024 (Tensorfuse), 32 (NVIDIA forums), 1 (Medium) have zero effect. |
| `HF_XET_CHUNK_CACHE_SIZE_BYTES` | **Removed in 1.3.0** | Local chunk caching removed (`xet-core#715`). |
| `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY` | **Removed/no-op in 1.3.0** | Sequential write became mandatory. Some older docs (Fast Headless) still recommend it; see 11. |

**Exception:** if a specific repo is not Xet-backed, or if empirical testing shows `hf_transfer` wins, set **both** `HF_HUB_DISABLE_XET=1` and `HF_HUB_ENABLE_HF_TRANSFER=1`. Otherwise omit both.

## 4. `HF_XET_HIGH_PERFORMANCE=1` and the 52 GB RAM threshold

- Raises internal concurrency bounds, uses all CPU cores for decompression/reconstruction, and expands in-memory buffers.
- **hf_xet < 1.7.0:** requires **≥64 GB RAM**; can OOM-kill (SIGKILL) a 52 GB Colab VM due to large fixed buffers. Confirmed by an AWS EC2 case study.
- **hf_xet ≥ 1.7.0:** buffer sizes are dynamic and respect container memory limits. Safe on a 52 GB Colab VM, including with `HF_XET_HIGH_PERFORMANCE=1`.
- Check the installed version before enabling: `pip show hf_xet`.

## 5. Why more workers hurts

| Phenomenon | Source | Key finding |
| :--- | :--- | :--- |
| **GIL callback slowdown (4.2×)** | `huggingface_hub#3859` | 16.1 GB on 25 Gbps: pure Rust `hf_xet.download_files(progress_updater=None)` = **3,119 MB/s**; `snapshot_download(max_workers=8)` + tqdm = **750 MB/s**. Batching progress via `AggregatingProgressUpdater` (200 ms; `xet-core#668`) restored **2,395 MB/s** (<1% overhead). |
| **Container `.lock` stalls** | `huggingface_hub#2543` | overlayfs with 100+ layers (K8s/Colab): `max_workers=10` caused **170–510+ s** lock contention in `~/.cache/huggingface/hub/.locks`; a 3 GB model took 8–10 min. Lower to 1–2. |
| **Socket starvation** | Scribd guide / Medium | Aggressive parallel downloads can stall at **12 KB/s**. Recommend `max_workers=1 or 2`. |
| **Dataset stalls** | `datasets#6439` | `load_dataset` with 64 workers was 20× slower than `snapshot_download` and stalled >7 days. |

## 6. Shard size and geometry

- **Legacy:** `max_shard_size="5GB"` bounded peak RAM during sequential loading.
- **Transformers v5.x (Dec 2025, PR #42734):** default raised to **50 GB**. Maintainers (@Wauplin, @ArthurZucker): with Xet + safetensors mmap parallel loading, fewer shards simplify management; peak memory is the overall model size.
- **Diffusers:** default **10 GB**. **Hub hard cap:** **50 GB per file**. **<20 GB per file** recommended for CloudFront CDN cache retention on `/resolve/` (forum #128442).
- **Does sharding matter for deduplication? No.** Xet chunks and Xorbs are identical whether 15 GB is one file or 32×468 MB shards.
- **Does it matter for overhead? Yes.** 16–32 shards create 16–32 separate `/resolve/` metadata requests (`HF_HUB_ETAG_TIMEOUT`), 16–32 `.lock` acquisitions on overlayfs, and 8 reconstruction contexts contending for the GIL. A single large file avoids this and stays under 20 GB CDN.
- **Save-time exception:** `save_pretrained()` on a disk-offloaded/quantized model allocates a CPU staging buffer equal to `max_shard_size` (50 GB default) before writing → OOM on unified-memory hosts (`transformers#47336`). Use `max_shard_size="2GB"` or `"5GB"` **only when saving**, not downloading.

## 7. Run-to-run variance: 13–15 s vs 75–147 s for 14.88 GB

### Mechanism A — GCE Persistent Disk write-burst exhaustion

- Colab L4 runs on GCE Persistent Disk (`pd-balanced`/`pd-standard`/`pd-ssd`, 100–200 GiB).
- Official formula: **sustained = baseline + per-GiB × GiB**. Per-instance ceiling **1,200 MiB/s (~1,258 MB/s)**, requiring ≥2,000 GiB volume. Small disks allow short bursts above sustained caps.
- Linux page cache absorbs writes into RAM before async flush.
- **Fast path** (clean cache + burst credits): ~1,000–1,145 MB/s → 14.88 GB in **13–15 s**.
- **Slow path** (dirty-page threshold + credits exhausted): baseline **140 MiB/s (146.8 MB/s)**; per-GiB **0.28 MiB/s pd-balanced**, **0.48 pd-ssd**, **0.12 pd-standard**.
  - 100 GiB pd-balanced sustained: **~176.2 MB/s → 84.4 s** for 14.88 GB.
  - 100 GiB pd-ssd: **288 MiB/s (~302 MB/s → ~49 s)**, throttled to baseline + flush contention = **101–147 s**.
  - pd-standard capped **240–400 MB/s** per instance.
- Hypervisor-enforced; cannot be fixed by `max_workers`.

### Mechanism B — CDN/Xet edge cache hits vs misses

- Hub uses multi-cloud: AWS CloudFront `*.cdn.hf.co`, Google Cloud CDN `us.gcp.cdn.hf.co`, Xet CAS `cas-server.xethub.hf.co`.
- **HIT** (`x-hub-cache: HIT`): pulls across Google internal fabric at **>1 GB/s**.
- **MISS** (`cache: Miss from cloudfront`, `x-hub-cache: MISS`): higher latency, lower origin fill (`ai-dynamo/dynamo#10710`).
- Cache key is the **byte-range slice** (`xorb__0-62796437`), not the logical Xorb hash. Different adaptive boundaries cause misses even for the same Xorb.
- Upstream bug **`xet-core#800` “Xet stalls on GCP”** intermittently pushes transfers into the 75–147 s range.

### Mechanism C — concurrent `hf_xet` fallback bug + lock contention

- **Fallback bug** (fixed: “Concurrent downloads always use hf_xet”): some threads treated `hf_xet` as missing and fell back to plain single-stream HTTP at 10–20 MB/s. In a 16-shard download, 14 shards finish in 13 s but 2 stragglers take 75–147 s.
- **overlayfs filelock:** blocks shard threads for dozens of seconds regardless of `max_workers`.

### Per-IP rate limiting is not the cause

Hub does **not** shape bandwidth to 100–200 MB/s. It enforces **request count per 5-minute window** on `/resolve/`:

| Tier | `/resolve/` limit |
| :--- | ---: |
| Anonymous (shared IP) | 3,000 / 5 min |
| Free authenticated | 5,000 / 5 min |
| PRO | 12,000 / 5 min |
| Team ($20/mo) | 20,000 / 5 min |
| Enterprise ($50/mo) | 50,000 / 5 min |
| Enterprise Plus | 100,000 / 5 min |

Exceeding returns **HTTP 429 Too Many Requests**, not TCP throttling. An 80 GB model in 16 shards uses ~16–32 resolver requests per boot (~320–640/day at 20 runs), far below even the free bucket.

## 8. Public repos, egress and pricing

- **Zero egress cost for the owner.** Downloading a public/private repo does not count toward the owner's quota. Pricing page: “Egress and CDN included at no extra cost.”
- Storage: Public base (<50 TB) **$12/TB/mo**; Private base **$18/TB/mo** (AWS S3 ~$23/TB/mo; Backblaze Overdrive ~$15/TB/mo).
- Public volume discounts: **$10/TB (50 TB+)**, **$9/TB (200 TB+)**, **$8/TB (500 TB+)**. Private 500+ TB: **$12/TB**.
- Private overage: **$25/TB** over the included 1 TB tier — stored capacity only, never download bandwidth.
- Rate limits are enforced against the **requester's** `HF_TOKEN` or outbound NAT IP, not the model author. A shared proxy token saturates its own bucket; the owner is unaffected.

## 9. Recommended Colab L4 configuration

```python
import os
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"      # only if hf_xet >= 1.7.0
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from huggingface_hub import snapshot_download

model_path = snapshot_download(
    repo_id="org/model-name",
    max_workers=4,                                # range 2–4
    allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
)
```

Prerequisites and rules:
- `pip install -U "huggingface_hub[hf_xet]"`; ensure **hf_xet ≥ 1.7.0** (dynamic RAM), progress batching fix `xet-core#668`, and the concurrent-downloads fallback fix.
- `max_workers=4` is optimal; `hf_xet` auto-scales to 64 range streams internally.
- `HF_XET_HIGH_PERFORMANCE=1` is safe on 52 GB **only** with hf_xet ≥ 1.7.0; it enables 12-vCPU utilization.
- Authenticate with `HF_TOKEN` to move from 3,000 to 5,000–12,000+ requests/5 min.
- **Omit** `HF_HUB_ENABLE_HF_TRANSFER=1` and `HF_XET_NUM_CONCURRENT_RANGE_GETS` (no-ops on Xet).
- For repeated benchmarks, flush the kernel page cache between runs or stage to tmpfs `/dev/shm` (~25 GB) to bypass GCE 140–176 MB/s sustained throttling.

## 10. Shard recommendations

- **15 GB model:** one **15 GB** `.safetensors` (or at most 3×5 GB). Deduplicated to ~64 MB Xorbs, 64 concurrent range streams, <20 GB CDN optimal, matches Transformers v5 default, avoids 15 lock acquisitions.
- **80 GB model:** **2×~40 GB** (use 48 GB to stay below the 50 GB hard cap) or **4×~20 GB** for max CDN retention.
- Smaller 2 GB/5 GB shards are only needed when **saving** an offloaded model whose CPU RAM cannot hold a 50 GB staging buffer.
- For the “20 pulls/day” comparison and why 16×5 GB beats 50+ tiny shards, see 02 §9 and 11.

## 11. Key sources

- HF docs: models-downloading, environment_variables, mlx; HF pricing.
- Google Cloud: Persistent Disk performance.
- GitHub: `huggingface_hub#4219`, `#3859`, `#2543`; `transformers#42555`, `#44484`, `#47336`; `xet-core#668`, `#800`, `#715`; `ai-dynamo/dynamo#10710`.
- Blogs: JFrog Artifactory (June 2026), openxet.com, ai-muninn.com, Tensorfuse, Spheron.

**Caveat:** the Fast Headless document (2026-10-28) still recommends `HF_HUB_ENABLE_HF_TRANSFER=1` and `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`; this HF document (2026-10-29) says both are no-ops on current Xet. For Xet repos, trust this document; verify with `ss -tnp` and `hf_xet` version. See 11.
