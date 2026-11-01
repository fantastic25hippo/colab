# Fastest Measured Method: Multi-sharded (16 shards) `snapshot_download` to tmpfs `/dev/shm` with `max_workers=12` and `HF_XET_HIGH_PERFORMANCE=1` (Peak 1,119.7 MB/s / 3-rep mean 920.8 MB/s on L4; 6.03 s for 8.06 GB = 1,337 MB/s on TPU v5e-1, 0.00 s net wait).

---

## Comprehensive Hugging Face Download & Weight Staging Audit

### Summary Comparison Table

| Method / Configuration | Exact Settings (Workers, Env Flags, Tool) | Model & Size | VM Target | Measured Time | Throughput | Source File:Line | Status |
| :--- | :--- | :--- | :--- | ---: | ---: | :--- | :--- |
| **Multi-Shard Shm (L4 Sweep Winner)** | `max_workers=12`, `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_HIGH_PERFORMANCE=1`, `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`, `snapshot_download` | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards, 14.88 GB) | Colab L4 (12 vCPU) | **16.60 s** (single run); 13.28 s (best rep) | **896.13 MB/s** (single); **1,119.7 MB/s** (peak); **920.8 MB/s** (3-rep mean) | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:30](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L30), [benchmarks/HF-transfer/L4/SWEEP.md:23](file:///home/sahil/Desktop/sahil/colab/benchmarks/HF-transfer/L4/SWEEP.md#L23) | **MEASURED** |
| **Multi-Shard Shm (L4 Sweep w=4)** | `max_workers=4`, same flags | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards, 14.88 GB) | Colab L4 (12 vCPU) | 17.51 s (single run); 14.94 s (best rep) | 849.53 MB/s (single); 995.4 MB/s (peak); 695.0 MB/s (3-rep mean) | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:28](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L28), [benchmarks/HF-transfer/L4/SWEEP.md:21](file:///home/sahil/Desktop/sahil/colab/benchmarks/HF-transfer/L4/SWEEP.md#L21) | **MEASURED** |
| **Multi-Shard Shm (L4 Sweep w=8)** | `max_workers=8`, same flags | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards, 14.88 GB) | Colab L4 (12 vCPU) | 61.80 s (RESULTS.md 1-pass); 13.55 s (sweep best rep) | 240.72 MB/s (1-pass); 1,097.7 MB/s (peak); 746.3 MB/s (3-rep mean) | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:29](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L29), [benchmarks/HF-transfer/L4/SWEEP.md:22](file:///home/sahil/Desktop/sahil/colab/benchmarks/HF-transfer/L4/SWEEP.md#L22) | **MEASURED** |
| **Multi-Shard Shm (L4 Sweep w=16)** | `max_workers=16`, same flags | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards, 14.88 GB) | Colab L4 (12 vCPU) | 82.30 s (RESULTS.md 1-pass); 14.95 s (sweep best rep) | 180.76 MB/s (1-pass); 995.1 MB/s (peak); 460.9 MB/s (3-rep mean) | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:31](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L31), [benchmarks/HF-transfer/L4/SWEEP.md:24](file:///home/sahil/Desktop/sahil/colab/benchmarks/HF-transfer/L4/SWEEP.md#L24) | **MEASURED** |
| **Multi-Shard Shm (L4 Sweep w=24)** | `max_workers=24`, same flags | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards, 14.88 GB) | Colab L4 (12 vCPU) | 27.89 s (best rep) to 146.86 s (slow rep) | 533.3 MB/s (peak); 276.4 MB/s (3-rep mean) | [benchmarks/HF-transfer/L4/SWEEP.md:25](file:///home/sahil/Desktop/sahil/colab/benchmarks/HF-transfer/L4/SWEEP.md#L25) | **MEASURED** |
| **Multi-Shard Shm (L4 Sweep w=1, 2)** | `max_workers=1, 2`, same flags | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards, 14.88 GB) | Colab L4 (12 vCPU) | w=1: 48.50 s; w=2: 26.08 s | w=1: 306.73 MB/s; w=2: 570.38 MB/s | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:26-27](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L26-L27) | **MEASURED** |
| **Multi-Shard Disk (/tmp)** | `max_workers=12`, target `/tmp/w_best`, same flags | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards, 14.88 GB) | Colab L4 (12 vCPU) | 104.82 s | 141.93 MB/s (6.3x slower than `/dev/shm`) | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:39](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L39) | **MEASURED** |
| **Flag Ablation: HF_TRANSFER=0** | `max_workers=12`, `HF_HUB_ENABLE_HF_TRANSFER=0` to `/dev/shm` | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards, 14.88 GB) | Colab L4 (12 vCPU) | 76.62 s | 194.18 MB/s (vs 896.13 MB/s with =1) | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:40](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L40) | **MEASURED** (1 rep, confounded by Hub throttling) |
| **Flag Ablation: HIGH_PERF=0** | `max_workers=12`, `HF_XET_HIGH_PERFORMANCE=0` to `/dev/shm` | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards, 14.88 GB) | Colab L4 (12 vCPU) | 72.80 s | 204.34 MB/s (vs 896.13 MB/s with =1) | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:41](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L41) | **MEASURED** (1 rep, confounded) |
| **Flag Ablation: WRITE_SEQ=0** | `max_workers=12`, `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=0` | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (16 shards, 14.88 GB) | Colab L4 (12 vCPU) | 71.32 s | 208.60 MB/s (vs 896.13 MB/s with =1) | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:42](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L42) | **MEASURED** (1 rep, confounded) |
| **Single Blob to `/dev/shm` (L4)** | `hf_hub_download`, `HF_HUB_ENABLE_HF_TRANSFER=1` | `LFM2.5-24B-A2B-UD-Q4_K_XL.gguf` (1 blob, 14.88 GB) | Colab L4 (12 vCPU) | 23.61 s | 630.12 MB/s | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:17](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L17) | **MEASURED** |
| **Single Blob to Disk (`/tmp`) (L4)** | `hf_hub_download`, `HF_HUB_ENABLE_HF_TRANSFER=1` | `LFM2.5-24B-A2B-UD-Q4_K_XL.gguf` (1 blob, 14.88 GB) | Colab L4 (12 vCPU) | 44.71 s | 332.75 MB/s | [benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md:18](file:///home/sahil/Desktop/sahil/colab/benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md#L18) | **MEASURED** |
| **Single Blob to `/dev/shm` (4 GB)** | `hf_hub_download`, `HF_HUB_ENABLE_HF_TRANSFER=1` | `bartowski/Qwen2.5-7B-Instruct-GGUF` (4.68 GB) | T4 / L4 / A100 / TPU v5e / G4 | 22.10 s (T4); 18.05 s (L4); 17.93 s (A100); 20.07 s (v5e); 16.52 s (G4) | 202.08 MB/s (T4); 247.42 MB/s (L4); 249.12 MB/s (A100); 222.58 MB/s (v5e); 270.36 MB/s (G4) | [benchmarks/HF-transfer/*/RESULTS.md](file:///home/sahil/Desktop/sahil/colab/benchmarks/HF-transfer/T4/RESULTS.md#L18) | **MEASURED** |
| **Single Blob to Disk (15 GB)** | `hf_hub_download`, `HF_HUB_ENABLE_HF_TRANSFER=1` | `bartowski/Qwen2.5-14B-Instruct-GGUF` (15.70 GB) | T4 / L4 / A100 / TPU v5e / G4 | 143.09 s (T4); 59.35 s (L4); 42.42 s (A100); 140.06 s (v5e); 52.33 s (G4) | 104.65 MB/s (T4); 252.29 MB/s (L4); 352.98 MB/s (A100); 106.91 MB/s (v5e); 286.16 MB/s (G4) | [benchmarks/HF-transfer/*/RESULTS.md](file:///home/sahil/Desktop/sahil/colab/benchmarks/HF-transfer/A100/RESULTS.md#L19) | **MEASURED** |
| **CPU VM Download Sweep** | `scripts/hf_sweep.py 1,4,8,12,16 2 disk` | `avgrocketman/LFM2.5-24B-A2B-Q4-shards` (14.88 GB) | Colab CPU (2 vCPU) | Aborted after ~15 min | No throughput measured | [benchmarks/HF-transfer/CPU/RESULTS.md:3](file:///home/sahil/Desktop/sahil/colab/benchmarks/HF-transfer/CPU/RESULTS.md#L3) | **MEASURED** (Aborted; file says no number is measured) |
| **TPU v6e Provisioning** | `colab new -s hf-v6e --tpu v6e1` | `Qwen2.5-7B/14B` | TPU v6e-1 | Timed out (>555 s, exit 1) | 0 MB/s (no VM allocated) | [benchmarks/HF-transfer/TPU-v6e/RESULTS.md:20-30](file:///home/sahil/Desktop/sahil/colab/benchmarks/HF-transfer/TPU-v6e/RESULTS.md#L20-L30) | **MEASURED** (Failed) |
| **TPU v5e Floor Overlap (Qwen3-4B)** | `snapshot_download`, `max_workers=12`, into `/dev/shm/qwen3` (3 shards) | `Qwen/Qwen3-4B` (13 files, 8.06 GB; 3 safetensors = 8.04 GB / 7.49 GiB) | TPU v5e-1 (24 vCPU) | **6.03 s** (net wait **0.00 s** overlapped with imports) | **1,337 MB/s** (1,275 MiB/s) | [benchmarks/TPU-v5e/tpu-final/RUN5-FLOOR-COLDSTART.md:27](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu-final/RUN5-FLOOR-COLDSTART.md#L27), [run5_floor_coldstart_results.json:9](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu-final/run5_floor_coldstart_results.json#L9) | **MEASURED** (rate from current Hub tree 8,060,926,626 B) |
| **TPU v5e Parallel Prep (Qwen3-4B)** | `snapshot_download`, `max_workers=12`, concurrent with venv | `Qwen/Qwen3-4B` (8.06 GB) | TPU v5e-1 (24 vCPU) | **11.89 s** (in parallel with 19.85 s venv prep) | **678 MB/s** (646 MiB/s) | [benchmarks/TPU-v5e/tpu-final/RUN2-COLDSTART.md:14](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu-final/RUN2-COLDSTART.md#L14), [run2_coldstart_results.json:8](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu-final/run2_coldstart_results.json#L8) | **MEASURED** (rate from current Hub tree) |
| **TPU v5e Baseline (Qwen3-4B)** | `snapshot_download`, `max_workers=12`, into `/dev/shm` | `Qwen/Qwen3-4B` (8.06 GB) | TPU v5e-1 (24 vCPU) | 10.14 s to 12.66 s | 637–795 MB/s | [benchmarks/TPU-v5e/tpu-final/RESULTS.md:10](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu-final/RESULTS.md#L10), [benchmarks/TPU-v5e/tpu-characterization/RESULTS.md:23](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu-characterization/RESULTS.md#L23) | **MEASURED** (rate from current Hub tree) |
| **TPU v5e Multi-Shard (LFM2.5)** | `snapshot_download`, `max_workers=12`, into `/dev/shm` (9 shards) | `avgrocketman/LFM2.5-24B-A2B-W8A8-W4A16-TPU` (17 files, 13.15 GB / 12.24 GiB; 9 safetensors = 13.13 GB) | TPU v5e-1 (24 vCPU) | **10.81 s** (2026-11-01); 11.60 s (2026-10-31); 18.39 s (diag) | **1,216 MB/s** (10.81 s); **1,133 MB/s** (11.60 s); **715 MB/s** (diag) | [benchmarks/TPU-v5e/lfm25_200k/RESULTS.md:60, 84](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/lfm25_200k/RESULTS.md#L60), [run_results.json:44](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/lfm25_200k/run_results.json#L44), [run_results_diag.json:8](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/lfm25_200k/run_results_diag.json#L8) | **MEASURED** (rate from current Hub tree 13,145,607,027 B; `HF_STORAGE_AUDIT.md` labeled 12.24 GiB as "12.24 GB") |
| **TPU v5e Multi-Shard (Gemma4)** | `snapshot_download`, `max_workers=12`, into `/dev/shm` | `google/gemma-4-E4B-it-qat-w4a16-ct` (11.51 GB) | TPU v5e-1 (24 vCPU) | 20.18 s | 570.49 MB/s | [benchmarks/TPU-v5e/gemma4-e4b-qat/RESULTS.md:11](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/gemma4-e4b-qat/RESULTS.md#L11) | **MEASURED** |
| **Prebuilt Tar/Zstd Venv (`hf_hub_download`)** | `hf_hub_download` (3 files concurrently) + `zstd -d -T0 \| tar -xf` | `venv312_cpu.tar.zst` (1.35 GB packed / 5.01 GB unpacked) + cache + zstd_bin | TPU v5e-1 (24 vCPU) | Download: **2.60 s** (Run 5) to 13.50 s; Unpack: **6.17 s** to 6.51 s | Total prep: **9.11 s** (Run 5) vs 91.21 s pip baseline | [benchmarks/TPU-v5e/tpu-final/RUN5-FLOOR-COLDSTART.md:10-18](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu-final/RUN5-FLOOR-COLDSTART.md#L10-L18), [RUN1-VENV.md:18-21](file:///home/sahil/Desktop/sahil/colab/benchmarks/TPU-v5e/tpu-final/RUN1-VENV.md#L18-L21) | **MEASURED** |
| **Transformers v5 / Xet 50 GB Single-Shard Claim** | Recommend single 15–50 GB shard, `max_workers=4` | Theoretical 15–80 GB checkpoints | Any cloud GPU / Colab | Claims fewer shards eliminate 16–32 `.lock` acquisitions and ETag overhead | Stated without in-repo run; contradicted by in-repo shard sweep | [research/03-HuggingFace-Xet-Tuning.md:16, 46-52, 129-133](file:///home/sahil/Desktop/sahil/colab/research/03-HuggingFace-Xet-Tuning.md#L16) | **CONTRADICTED** |
| **Pure Rust `hf-xet` CLI (no progress bar)** | `hf_xet.download_files(progress_updater=None)` | 16.1 GB checkpoint | 25 Gbps cloud instance | Claimed 3,119 MB/s vs 750 MB/s for `snapshot_download` | 3,119 MB/s claimed upstream in `huggingface_hub#3859` | [research/03-HuggingFace-Xet-Tuning.md:40](file:///home/sahil/Desktop/sahil/colab/research/03-HuggingFace-Xet-Tuning.md#L40) | **CLAIMED** (Upstream) |
| **CoreWeave Tensorizer (HTTP/S3 libcurl)** | `--load-format tensorizer` direct to GPU VRAM | 15 GB to 60 GB models | 40GbE cloud instance | Claimed ~3–8.22 s (CoreWeave) / 37.36 s (S3) | Zero-disk VRAM streaming; Plaid mode <1–2 GB RAM | [research/04-L4-Loading-Routes-and-Precision.md:55, 87](file:///home/sahil/Desktop/sahil/colab/research/04-L4-Loading-Routes-and-Precision.md#L55) | **CLAIMED** (External) |
| **Run:ai Model Streamer (S3/GCS)** | `--load-format runai_streamer` direct to GPU VRAM | 15 GB checkpoint | Cloud GPU | Claimed 4.88 s (S3) / ~15–23 s (GCS) | Ring buffer 2 GB RAM; direct PCIe to VRAM | [research/04-L4-Loading-Routes-and-Precision.md:54, 86](file:///home/sahil/Desktop/sahil/colab/research/04-L4-Loading-Routes-and-Precision.md#L54) | **CLAIMED** (External) |
| **Aria2c / Parallel curl Range Gets** | `aria2c -x 16 -s 16` into `/dev/shm` | 15 GB checkpoint | Colab L4 | Mentioned as generic DRAM tmpfs route | Not tested in repo | [research/04-L4-Loading-Routes-and-Precision.md:20](file:///home/sahil/Desktop/sahil/colab/research/04-L4-Loading-Routes-and-Precision.md#L20) | **CLAIMED** (Unmeasured) |
| **Google Drive via rclone / drivemount** | `colab drivemount` or rclone OAuth2 | 15 GB checkpoint | Colab VM | `06` claimed 400–600 MB/s tuned; `02` baseline 50–76 MiB/s rclone | Interactive browser OAuth click every VM; 24h 403 quota lockout | [research/06-Agent-Workflows-Council.md:82](file:///home/sahil/Desktop/sahil/colab/research/06-Agent-Workflows-Council.md#L82), [research/02-Storage-Channels-and-Auth.md:10](file:///home/sahil/Desktop/sahil/colab/research/02-Storage-Channels-and-Auth.md#L10), [research/11-Conflicts-and-Verification.md:18](file:///home/sahil/Desktop/sahil/colab/research/11-Conflicts-and-Verification.md#L18) | **CONTRADICTED** (Dead) |

---

### 1. Document Claims Inventory (File:Line Citations)

The codebase makes extensive claims regarding download methods, environment flags, and storage targets:

1. **`research/03-HuggingFace-Xet-Tuning.md`:**
   - **L14–16:** Two-tier concurrency: outer `snapshot_download()` thread pool (`max_workers=8` default; one thread per file), inner Rust `xet-core` adaptive range gets (1→64 streams per file). L16 also records upstream `max_workers=4` and the in-repo L4 sweep (`w=12` mean **920.8 MB/s**).
   - **L18–26:** `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_NUM_CONCURRENT_RANGE_GETS`, `HF_XET_CHUNK_CACHE_SIZE_BYTES`, and `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY` are deprecated no-ops on current Xet.
   - **L28–34:** `HF_XET_HIGH_PERFORMANCE=1` buffer safety on `hf_xet >= 1.7.0` (dynamic RAM).
   - **L40:** Cites `huggingface_hub#3859` for 4.2× GIL slowdown (3,119 MB/s Rust vs 750 MB/s `snapshot_download` + tqdm on a 25 Gbps host).
   - **L46–52 & L129–133:** Transformers v5 philosophy: single 15–50 GB `.safetensors` files to cut `.lock` acquisitions and ETag overhead.
   - **L54–75:** 13–15 s vs 75–147 s variance attributed to GCE PD write-burst exhaustion and CDN HIT vs MISS.
   - **L142:** Caveat that Fast Headless still recommends the no-op flags.

2. **`research/04-L4-Loading-Routes-and-Precision.md`:**
   - **L12:** GCE vNIC 5,325–8,110 Mbps (665–1,013 MB/s) multi-thread TCP.
   - **L19–27:** Route 1: download into `/dev/shm`. Single-thread HTTP (35 MB/s avg), `hf_transfer` (125–250 MB/s), in-repo 16 shards at `max_workers=12` (**16.60 s / 896 MB/s**).
   - **L11, L37:** Route 2 disk: Colab Persistent Disk ~180 MiB/s sustained clamp.
   - **L54–60 & L87:** Route 3: Run:ai Model Streamer (4.88 s S3) and CoreWeave Tensorizer (8.22 s median HTTP; 37.36 s AWS S3).
   - **L115:** `curl | llama-server /dev/stdin` fails (`ESPIPE Illegal seek` on GGUF headers).

3. **`benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md`:**
   - **L17–18:** MEASURED single blob to `/dev/shm` (23.61 s / 630.12 MB/s) vs `/tmp` (44.71 s / 332.75 MB/s).
   - **L26–31:** MEASURED 16-shard 1-pass (`w=1` 306.73 MB/s, `w=2` 570.38 MB/s, `w=4` 849.53 MB/s, `w=8` 240.72 MB/s, `w=12` **896.13 MB/s**, `w=16` 180.76 MB/s).
   - **L39–42:** MEASURED 16 shards to disk (104.82 s / 141.93 MB/s) and flag ablations (71.32–76.62 s; 1 rep each).

4. **`benchmarks/HF-transfer/L4/SWEEP.md`:**
   - **L21–30:** 3-rep rotation across `w=4,8,12,16,24`. No worker count collapses by itself. `w=12` mean **920.8 MB/s** (peak 1,119.7 MB/s); `w=8` mean 746.3 MB/s (peak 1,097.7 MB/s); `w=4` mean 695.0 MB/s (peak 995.4 MB/s). Pick 8–12 is UNCONFIRMED which is best.

5. **`benchmarks/HF-transfer/*/RESULTS.md`:**
   - Standardized single-blob transfer benchmarks across 6 accelerators (T4, L4, A100-40, TPU-v5e, G4, CPU).

6. **`benchmarks/TPU-v5e/tpu-final/RUN1-VENV.md`, `RUN2-COLDSTART.md`, `RUN5-FLOOR-COLDSTART.md`:**
   - **RUN1 (L18–20):** Prebuilt venv `venv312_cpu.tar.zst` (1.35 GB packed / 5.01 GB unpacked) unpacks in 6.17 s.
   - **RUN2 (L12–14):** Parallel download of venv (19.85 s total: 13.50 s dl + 6.35 s unpack) and Qwen3-4B (11.89 s to `/dev/shm`).
   - **RUN5 (L15, L27):** Floor optimization: concurrent download of venv + XLA cache + `zstd_bin` in 2.60 s; unpack 6.51 s; total prep 9.11 s; Qwen3-4B download during Python imports **6.03 s, 0.00 s net wait**.

7. **`benchmarks/TPU-v5e/lfm25_200k/RESULTS.md` & `run_results.json`:**
   - **RESULTS.md:60, 84:** LFM2.5 W8A8/W4A16 download **10.81 s** (2026-11-01 18:06 IST) and **11.60 s** (2026-10-31). Those files record wall time only, not MB/s.
   - **run_results_diag.json:8:** Diagnostic run 18.39 s.
   - Current Hub tree (anonymous `repo_info`, 2026-11-01): 17 files, **13,145,607,027 B** (13.15 GB / 12.24 GiB). Nine safetensors = 13,125,680,160 B. Rates above use this tree. `HF_STORAGE_AUDIT.md` printed 12.24 GiB as "12.24 GB".
   - **RESULTS.md:100:** `12.45 / 15.75 GiB` is HBM after weights load, not download size.

---

### 2. Method Analysis & Contradictions Resolved

#### Contradiction 1: Shard Layout (Single 50 GB Blob vs 16 Shards)
- **Claim A (`research/03` L46–52, L129–133):** Upstream Transformers v5 maintainers recommend single 15–50 GB blobs. Claimed that Xet's inner range-get streaming (up to 64 streams) makes multi-sharding obsolete, while multi-sharding incurs 16–32 `.lock` acquisitions on overlayfs and extra ETag requests.
- **Claim B / Measured Truth (`benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md`, `benchmarks/HF-transfer/L4/SWEEP.md`):** When targeting tmpfs `/dev/shm` (which eliminates overlayfs lock stalls), **16 shards dramatically beat a single blob**:
  - Single blob to `/dev/shm`: **23.61 s (630.12 MB/s)**.
  - 16 shards to `/dev/shm`: **16.60 s (896.13 MB/s)** single run; **1,119.7 MB/s** peak; **920.8 MB/s** mean.
  - Multi-sharding achieves ~1.4×–1.8× higher throughput because `snapshot_download`'s outer Python `ThreadPoolExecutor` runs one **thread per file** (up to `max_workers`), each with its own Xet session. Inner `xet-core` still opens 1→64 range-get streams **per file**, so a single blob is already parallel. The extra outer files help, and they do not replace inner concurrency.

#### Contradiction 2: Environment Variables (`HF_HUB_ENABLE_HF_TRANSFER`, sequential write, HIGH_PERFORMANCE)
- **Claim A (`research/03-HuggingFace-Xet-Tuning.md` section 3, L142):** `HF_HUB_ENABLE_HF_TRANSFER=1` and `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY` are no-ops on Xet repos.
- **Claim B (`benchmarks/LFM2.5-24B-A2B-Q4/L4/RESULTS.md` section 3):** each flag set to 0 gave 71-77 s (194-209 MB/s), against 16.60 s with all three on.
- **Caveat (my correction):** each ablation is ONE rep, run after the sweep. `SWEEP.md` shows that unchanged settings also produce 75-147 s runs from Hub throttling. All three ablations landed in that same slow band, so the data cannot separate "flag removed" from "throttled". The "4x slowdown" reading is **not established**. Status: UNRESOLVED. Docs disagree, and the measurement is confounded.
- Practical answer: keep the flags set (harmless). The cheap A/B in the test list below settles it.

#### Contradiction 3: Recommended Worker Count (`max_workers=4` vs `8..12`)
- **Claim A (`research/03` L16):** Upstream Xet docs recommend `max_workers=4` (conservative range 2–4) to prevent lock and socket contention.
- **Claim B / Measured Truth (`benchmarks/HF-transfer/L4/SWEEP.md`):**
  - `w=4`: Mean 695.0 MB/s.
  - `w=8`: Mean 746.3 MB/s (peak 1,097.7 MB/s).
  - `w=12`: Mean **920.8 MB/s** (peak 1,119.7 MB/s).
  - `w=16`: Mean 460.9 MB/s; `w=24`: Mean 276.4 MB/s.
  - *Resolution:* On a 12-vCPU L4 with tmpfs `/dev/shm`, `SWEEP.md` picks **8 to 12** and marks which is best as UNCONFIRMED. `w=4,8,12,16` all hit ~1,000 MB/s on an unthrottled rep. The 61.80 s (`w=8`) and 82.30 s (`w=16`) rows are the earlier 1-pass `RESULTS.md` run, Hub/CDN noise in that file's reading. `w=24` has the lowest mean and still had a 533.3 MB/s rep.

---

### 3. Driver Code Audit (`lfm25_coldstart_driver.py` & Qwen Drivers)

A strict audit of `benchmarks/TPU-v5e/lfm25_200k/lfm25_coldstart_driver.py` and `benchmarks/TPU-v5e/tpu_coldstart_driver.py` shows these implementation differences:

1. **Gap 1: Environment Variables Set AFTER Module Imports (Import-Order Bug):**
   - In `lfm25_coldstart_driver.py`:
     ```python
     54: from huggingface_hub import hf_hub_download, snapshot_download
     55: os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
     56: os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
     ```
   - In `tpu_coldstart_driver.py`:
     ```python
     40: from huggingface_hub import hf_hub_download, snapshot_download
     41: os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
     42: os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
     ```
   - **The Flaw:** `huggingface_hub.constants` evaluates `HF_HUB_ENABLE_HF_TRANSFER` at first import, so setting `os.environ` afterwards may be too late. `scripts/hf_sweep.py:5-13` sets the env vars **before** importing `huggingface_hub`. `scripts/hf_bench_lfm_l4.py` does **not**: it imports at line 2 and sets env at lines 14–15. That L4 script is the 16.60 s / 896 MB/s winner, so post-import env still produced the fastest L4 run. Whether the flags bind at import is **unverified**.

2. **Gap 2: Missing Progress Bar Suppression (`HF_HUB_DISABLE_PROGRESS_BARS=1`):**
   - Neither driver sets `HF_HUB_DISABLE_PROGRESS_BARS = "1"`.
   - As documented in `research/03` L40 (citing `huggingface_hub#3859` on a 25 Gbps host), tqdm progress callbacks with concurrent workers dropped 16.1 GB from 3,119 MB/s (pure Rust) to 750 MB/s (`snapshot_download` + tqdm). That is an upstream measurement, not a Colab run.

3. **Gap 3: Missing Sequential Reconstruction Flag:**
   - Neither driver exports `os.environ["HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY"] = "1"`. That flag was on in the L4 winning config (`RESULTS.md` §3). The 71.32 s / 208.60 MB/s ablation is one confounded rep (Contradiction 2), so this gap is a missing flag, not a proven 4× slowdown.

4. **Gap 4: Shard Layout on Qwen:**
   - `tpu_coldstart_driver.py` calls `snapshot_download("Qwen/Qwen3-4B", local_dir="/dev/shm/qwen3", max_workers=12)`.
   - Public Hub tree (2026-11-01): **3 safetensors** (`model-00001-of-00003.safetensors` 3.96 GB, `00002` 3.99 GB, `00003` 99.6 MB) plus 10 small files. Outer `max_workers` is one thread per file; the two large shards dominate. Inner Xet still range-gets inside each file.

5. **Gap 5: Overlapped Pipeline Discrepancy:**
   - In `tpu_coldstart_driver.py` (Run 5): Step 2a downloads venv + cache; Step 2b unpacks them into `/dev/shm` uncontested; Step 3 starts `dl_weights()` in the background *during* Python engine imports (`import jax; from vllm import LLM`). Weights download (6.03 s) finishes completely inside the 19.39 s import window (**0.00 s net weight wait**).
   - In `lfm25_coldstart_driver.py`: Step 2 launches `dl_weights()` and `dl_venv()` *simultaneously*, so the 1.35 GB venv download shares the NIC with the 13.15 GB weight download before unpack.

---

### 4. Reference Runs Comparison: Qwen (8.06 GB @ 11.89 s) vs LFM (13.15 GB @ 10.81 s)

Times are MEASURED. Byte counts are the current public Hub trees (anonymous `HfApi.repo_info(files_metadata=True)`, 2026-11-01), because the run JSON stores only wall time.

- **Qwen Run 2:** 11.89 s for `Qwen/Qwen3-4B` (`RUN2-COLDSTART.md:14`, `run2_coldstart_results.json:8`). Hub tree 8,060,926,626 B (13 files). Three safetensors = 8,044,982,000 B (3.96 + 3.99 + 0.100 GB). Rate **678 MB/s**. `8,046,819,328` in `tpu-characterization/RESULTS.md:100` is **HBM after weights load**, not a download counter. `11.85 s` in `tpu-final/RESULTS.md:15` is **engine weight load**, not Hub download.
- **LFM Run:** `RESULTS.md:100` `12.45 / 15.75 GiB` is **HBM after weights load**. Hub tree is 13,145,607,027 B = **13.15 GB / 12.24 GiB** (17 files). Nine safetensors = 13,125,680,160 B: eight ~1.61 GB files plus `model-00009.safetensors` at 231 MB. Download **10.81 s → 1,216 MB/s**; 11.60 s → 1,133 MB/s; diag 18.39 s → 715 MB/s.

#### Why was LFM 1,216 MB/s in 10.81 s while Qwen Run 2 was 678 MB/s in 11.89 s?

The ~1.8× rate gap is **not proven**. Three factors sit in the evidence; none is isolated:

1. **Outer file count (evidence, not proof):**
   - Qwen has **two large shards** plus a 100 MB remainder. Outer `max_workers=12` cannot open 12 fat TCP files.
   - LFM has **eight ~1.61 GB shards** plus a 231 MB remainder. More outer files, still with inner Xet range-gets per file.
   - Run 5 Qwen, same 3-shard tree, hit **1,337 MB/s in 6.03 s** with no venv overlap. Three shards are enough to exceed LFM's 1,216 MB/s when the NIC is free. Inner Xet (1→64 streams/file) is why.

2. **NIC sharing in Qwen Run 2 (MEASURED overlap, rate inferred):**
   - Run 2 pulled weights beside `venv312_cpu.tar.zst` (1.35 GB) and the compile cache.
   - Run 5 started weights after venv unpack, during imports: **6.03 s, 0.00 s net wait, 1,337 MB/s**.

3. **Hub/CDN noise (MEASURED times, inferred cause):**
   - `research/03` L69–75: HIT vs MISS. LFM 10.81 s vs diag 18.39 s on the same repo. L4 sweep unchanged flags also produced 13 s and 75–147 s runs. Different days, large throttling noise. Cache HIT for the 10.81 s LFM run is inferred.

---

### 5. Documented But Never Measured Methods (Ranked by Payoff)

Below are methods proposed or documented in the repo that have never been benchmarked, ranked by expected latency payoff:

#### 1. Streaming Zstd Extraction Pipeline (`zstd -d -T0 -c | tar -xf -`) for Model Bundles
- **Concept:** Package model weights into an 8–12 shard `.tar.zst` or streaming chunk archive. As shards download into memory/FIFO, pipe them directly through multi-threaded zstd into `/dev/shm`, skipping post-download reconstruction (`research/09:72` describes this pipe for the venv archive, not a model bundle).
- **Expected Payoff:** **High** (~4–8 s saved on cold start; eliminates file-by-file Hub lock overhead).
- **Test Design:**
  - *VM Target:* Colab L4 GPU (`colab --auth=oauth2 new -s bench-pipe --gpu L4`).
  - *Execution:* Pull 8 shards of a tar.zst bundle from HF Hub into a named pipe: `mkfifo /dev/shm/p; zstd -d -T0 -c /dev/shm/p | tar -xf - -C /dev/shm & snapshot_download(..., local_dir="/dev/shm/p")`.
  - *Metrics to Record:* Combined download + extraction wall time (s), peak RAM usage in `/dev/shm`, CPU core utilization.
  - *Rough CU:* ~0.02 CU (~1 minute on L4 @ 1.54 CU/hr).

#### 2. CoreWeave Tensorizer (Zero-Disk libcurl HTTP Streaming Direct to VRAM)
- **Concept:** Convert model to `.tensors` format on HF Hub and stream layers directly over HTTPS via libcurl into GPU VRAM using Plaid Mode (staging buffer <1–2 GB in DRAM) (`research/04:55, 87`).
- **Expected Payoff:** **High** (Eliminates `/dev/shm` capacity bottleneck entirely; allows loading 30–70 GB models that exceed Colab host RAM).
- **Test Design:**
  - *VM Target:* Colab L4 GPU (`--gpu L4`).
  - *Execution:* Convert a 7B–14B model to `.tensors`, upload to Hub, install `tensorizer`, execute `TensorDeserializer(stream)` targeting `cuda:0`.
  - *Metrics to Record:* Network streaming time (s), host DRAM peak (GB), GPU VRAM allocated (GB), first-token latency.
  - *Rough CU:* ~0.05 CU (~2 minutes on L4 @ 1.54 CU/hr).

#### 3. Pure Rust `hf-xet` CLI / `download_files(progress_updater=None)`
- **Concept:** Bypass Python's `snapshot_download` and GIL/tqdm callbacks entirely by invoking Rust `hf_xet` directly (`research/03:46`, `huggingface_hub#3859`).
- **Expected Payoff:** **Medium-High** (Upstream #3859: 3,119 MB/s Rust vs 750 MB/s Python+tqdm on a 25 Gbps host; Colab NIC is lower).
- **Test Design:**
  - *VM Target:* Colab L4 GPU (`--gpu L4`).
  - *Execution:* Compare `python -c "import hf_xet; hf_xet.download_files(..., progress_updater=None)"` against `snapshot_download(..., max_workers=12)` on 16 shards of LFM Q4.
  - *Metrics to Record:* Download wall time (s), effective MB/s, Python GIL lock contention / CPU wait time.
  - *Rough CU:* ~0.02 CU (<1 minute on L4).

#### 4. Multi-Connection Range Requests via `aria2c` / Parallel `curl`
- **Concept:** Use `aria2c -x 16 -s 16 -k 10M` against HF Hub resolve URLs to download single monolithic blobs (e.g. 14.88 GB GGUF) into `/dev/shm` using 16 concurrent HTTP byte-range connections (`research/04:20`).
- **Expected Payoff:** **Medium** (Enables single blobs to reach multi-shard speeds without requiring manual pre-sharding on Hub).
- **Test Design:**
  - *VM Target:* Colab L4 GPU (`--gpu L4`).
  - *Execution:* `apt-get install -y aria2 && aria2c -x 16 -s 16 -d /dev/shm -o blob.gguf "https://huggingface.co/avgrocketman/LFM2.5-24B-A2B-Q4-shards/resolve/main/LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"`.
  - *Metrics to Record:* Total elapsed seconds, average MB/s, comparison against single-blob `hf_hub_download` (23.61 s / 630 MB/s).
  - *Rough CU:* ~0.02 CU (<1 minute on L4).

#### 5. Shard Sizing Granularity Sweep (32x 468 MB vs 16x 930 MB vs 8x 1.86 GB)
- **Concept:** Determine the mathematically optimal shard count for 12–24 vCPU instances (`scripts/hf_bench_lfm_l4.py:88-96`).
- **Expected Payoff:** **Medium** (Identifies exact tradeoff between per-file HTTP ETag overhead and TCP connection saturation).
- **Test Design:**
  - *VM Target:* Colab L4 GPU (`--gpu L4`).
  - *Execution:* Stage the same 14.88 GB model as 8 shards (1.86 GB), 16 shards (930 MB), and 32 shards (468 MB); run 3-rep rotation at `w=12` into `/dev/shm`.
  - *Metrics to Record:* Mean MB/s, request latency overhead, variance across repetitions.
  - *Rough CU:* ~0.08 CU (~3 minutes on L4).

---

### Caveats on this audit
- TPU Qwen/LFM MB/s figures are derived from current Hub tree sizes (anonymous `repo_info` on 2026-11-01) divided by MEASURED wall times. The run JSON does not store bytes transferred.
- `HF_STORAGE_AUDIT.md` printed the LFM TPU repo as "12.24 GB"; the Hub total is 12.24 GiB = 13.15 GB. Rates that used 12.24e9 bytes were ~13% low.
- Import-order gap is unverified. Both TPU drivers and `hf_bench_lfm_l4.py` (the 16.60 s L4 winner) set `HF_*` env after importing `huggingface_hub`. `hf_sweep.py` sets them before.
- Cheap extra test (about 0.05 CU, L4): 3 reps each with env flags on vs off, interleaved, to resolve Contradiction 2.
- Qwen3-4B has 3 safetensors (two ~4 GB, one 100 MB). Outer `max_workers=12` cannot open 12 fat files; inner Xet still parallelizes inside each file.
- Nothing was run on Colab, nothing committed, and no token was printed.
