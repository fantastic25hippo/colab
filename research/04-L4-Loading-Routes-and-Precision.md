# 04 — L4 Weight-Loading Routes, Precision and GGUF
Consolidated from `L4 GPU Model Loading Comparison.pdf` (23 pages, 68 references) and its `Summary.md`. This is the deep technical report on loading 15–75 GB checkpoints onto a Colab NVIDIA L4. Labels follow the original (Measured / Inferred / Spec).

## 1. L4 hardware topology and hard limits
| Subsystem | Spec | Capacity / ceiling | Throughput | Basis |
| :--- | :--- | :--- | :--- | :--- |
| **GPU** | NVIDIA L4 Ada Lovelace SM89 | 24 GB GDDR6 (~22.5–23.0 GB usable with ECC) | ~300 GB/s internal GDDR6 | Spec/Measured |
| **Host RAM** | Dual-channel DDR4/DDR5, 6 vCPU/12 thread Xeon/EPYC slice | ~52–53 GB total | ~25–40 GB/s DRAM | Measured |
| **`/dev/shm`** | Linux tmpfs, default 50% of RAM | ~26 GB max | ~20–30 GB/s in-memory | Kernel default |
| **CPU–GPU** | PCIe Gen4 x16 (sometimes Gen3 x16) | 31.5 GB/s theoretical Gen4 | ~12–14 GB/s unpinned mmap bounce; ~24–25 GB/s pinned DMA | PCIe spec + CUDA DMA |
| **Local disk** | GCE Network Block Storage + Docker overlayfs | ~100–200+ GB root (`/content`, `~/.cache`) | Standard PD: **179 MiB/s read, 180 MiB/s write** (fio 64k); SSD PD: **671/675 MiB/s** | fio on Colab GCE |
| **VM network** | GCE vNIC (virtio-net/gVNIC) | 5–10 Gbps virtual link | **5,325 Mbps (~665 MB/s) to 8,110 Mbps (~1,013 MB/s)** multi-thread TCP | iperf3 |
| **HF Hub ingress** | Public HTTPS CDN/LFS/Xet | Per-stream throttled | Standard HTTP **10.4–50 MB/s (35 avg)**; Git LFS **55 MB/s**; hf_transfer **125–250+ MB/s** | Measured |

**Foundational constraint:** a 15 GB checkpoint (8B FP16/BF16 or 27B 4-bit) fits in 26 GB shm, 52 GB host RAM and 23 GB VRAM. A 75 GB checkpoint (35–40B BF16 or 70B FP8) is **3.26× L4 VRAM, 2.88× `/dev/shm`, 1.44× host RAM**. Scaling 15→75 GB is non-linear: ENOSPC, page-cache thrashing, CUDA OOM unless streaming + online quantization/offloading.

**Post-Xet caveat:** the HF throughput figures above predate Xet. On Xet repos `HF_HUB_ENABLE_HF_TRANSFER=1` is a no-op; measured Xet throughput is 252–353 MB/s to disk and 200–270 MB/s to `/dev/shm`. See `03` and `11`.

## 2. Route 1 — download into `/dev/shm`, then GPU
Mechanism: `huggingface_hub`/`hf_transfer`/`aria2c`/`curl` writes to tmpfs backed by DRAM; PyTorch/vLLM/SGLang loads from the shm path over PCIe.

**15 GB typical:**
- Default single-thread Python HTTP: **35 MB/s avg (10.4–50)** → 15,360 MB = **307–439 s**.
- `HF_HUB_ENABLE_HF_TRANSFER=1` Rust multi-thread: **125 MB/s standard, 200–250+ MB/s uncongested** → **76.8–122.9 s**.
- Co-located GCS `gs://` or high-speed mirror saturating 665–1,013 MB/s NIC → **15.2–23.1 s**.
- Once in shm: 15 GB `.safetensors` → L4 via `safetensors.torch.load_file(..., device="cuda")` = **~1.1 s raw PCIe @14 GB/s unpinned**, **~2.5–5.0 s** with CUDA alloc + Python overhead.
- **End-to-end 15 GB:** **~78–128 s** via single-blob hf_transfer at 125–200 MB/s; **16.60 s download [M]** (896 MB/s; 3-rep mean 920.8 MB/s in `../benchmarks/HF-transfer/L4/SWEEP.md`) when pulling **16 shards into `/dev/shm` with `max_workers=12`**.

**75 GB scaling:** immediate catastrophic failure. `/dev/shm` capped at 26 GB → aborts at **34.6%** with `OSError: [Errno 28] No space left on device`. Even `mount -o remount,size=80G /dev/shm` fails: only ~52 GB physical DRAM + zero swap → Linux OOM killer. Viable only via rolling shard-by-shard: download 1–3 shards (5–15 GB) into shm, load + dynamically quantize to 4-bit onto 23 GB L4, `os.unlink()` to free shm, fetch next. 5×15 GB → **~390–640 s** via hf_transfer (384–614 s network + 5.5–15 s PCIe+quant).

**Limiting factors:** 15 GB — network (>95% wall time; HF per-connection throttling); load phase is CPU bounce-buffer memcpy + PCIe because mmap from shm is pageable/unpinned (12–14 GB/s vs 25 GB/s pinned ceiling). 75 GB — host RAM + 26 GB shm + 23 GB VRAM are hard blockers.

## 3. Route 2 — download to disk (`/content` or `~/.cache/huggingface/hub`), then GPU
Local disk = Docker overlayfs on GCE Persistent Disk. Distinguish **warm OS page cache** (fits host RAM) from **cold disk** (cache evicted/overwhelmed).

**15 GB:**
- Download: standard Colab PD write **180 MiB/s** clamp. hf_transfer at 200 MB/s → **~85.3 s** inferred, or **~122.9 s** at 125 MB/s. Azure A100 benchmark vLLM v0.23.0 phi-4 ~29 GB: download **~113 s (~256 MB/s)** → scales to **~58.4 s** for 15 GB.
- Load — **warm cache:** 52 GB DRAM retains the 15 GB written to `~/.cache`; mmap faults satisfied from RAM, zero physical disk reads → **PCIe 1.1–5.0 s**, total **~86.4 s** at 180–200 MB/s.
- Load — **cold disk:** single-threaded mmap triggers small sequential reads and cannot saturate SSD queue depth. NVIDIA/Run:ai measured Llama 3 8B 15 GB default HF safetensors loader: **47.99 s on AWS GP3** (1,000 MiB/s max), **47.00 s on IO2** (4,000 MiB/s max); vLLM ready **66.13 s GP3 / 62.69 s IO2**. Azure measured **~74 s disk→GPU for 29 GB** → **~38.3 s for 15 GB**. Colab standard PD 179 MiB/s read → **61.4–85.8 s cold load**, total cold-start **146.8–208.7 s**.

**75 GB scaling — super-linear degradation (3-stage thrashing cliff):**
1. **Warm cache vanishes:** 75 GB > ~52 GB host RAM (6–10 GB occupied by OS/Docker/Python/CUDA). Writing 75 GB forces `kswapd` to evict early shards; by byte 75 GB, shard 1 is gone from RAM.
2. **Synchronous write clamping:** 15 GB can buffer 5–10 GB dirty pages before `vm.dirty_ratio`; 75 GB spends >85% of its duration throttled to physical disk write **180 MiB/s standard PD** → **~426.7 s (~7.1 min)** just to commit, or **~614.4 s** at 125 MB/s hf_transfer baseline.
3. **Cold read back:** ~**240 s** at measured single-threaded SSD mmap ~312 MB/s, or ~**429 s** on 179 MiB/s Colab standard PD.
4. Plus mandatory layer-wise quant (`--load-format layered`) or disk/CPU offloading (`accelerate device_map="auto"`) because 75 GB cannot fit 23 GB VRAM → total **11–17 min (660–1040+ s)**.

**Limiting factors:** 15 GB — `min(network ingress, disk write)`; load is PCIe + CPU memcpy if warm, single-threaded mmap disk latency (~312 MB/s cloud SSD ceiling; 179 MiB/s Colab PD) if cold. 75 GB — persistent disk write throughput, host RAM capacity/page-cache thrashing, single-threaded disk read IOPS, 23 GB VRAM ceiling.

## 4. Route 3 — direct load from Hub/object storage to GPU (no staging copy)
Four mechanisms, carefully distinguished:

1. **Default vLLM/SGLang with HF Hub ID** (`vllm serve org/model`): **not streaming**. It invokes `huggingface_hub.snapshot_download()` to fetch all `.safetensors` into `~/.cache/huggingface/hub` first. Architecturally identical to Route 2.
2. **safetensors direct with `device="cuda"`:** `safetensors.torch.load_file(..., device="cuda")` requires a seekable local POSIX path for `mmap()`, cannot accept an HTTP stream. `safetensors.torch.load(bytes)` accepts an in-memory buffer fetched over HTTP but only targets CPU (no device arg), so the whole file must materialize in host DRAM first (GitHub Issue #200). True zero-CPU-staging device-aware network loading is future PyTorch Foundation roadmap.
3. **NVIDIA Run:ai Model Streamer** (`--load-format runai_streamer`): genuine zero-disk streaming. Phase 1: tiny config/tokenizer to local cache. Phase 2: **16–32 concurrent HTTP byte-range streams** pull `.safetensors` chunks into a bounded host-RAM ring buffer (`memory_limit`, e.g. 2 GB) and async-copy across PCIe to VRAM. Supports S3, GCS, Azure Blob, local/shared POSIX — **not direct HF Hub repo IDs**; weights must live in an S3-compatible bucket/GCS mirror to bypass disk.
4. **CoreWeave Tensorizer** (`--load-format tensorizer`): genuine zero-disk streaming via libcurl, serialized tensors directly from HTTP/HTTPS, S3 (`s3://`), Redis into GPU. Supports raw HTTP/HTTPS URLs, so it can stream directly from an HF Hub `resolve` URL if the checkpoint is pre-converted to bespoke `.tensors` single-file format (tensor metadata headers at front). **Plaid Mode** allocates one host staging buffer sized to the largest individual tensor, overwriting each transfer → host RAM <1–2 GB regardless of checkpoint size.

**Typical load time, 15 GB:**
- **Run:ai on AWS g5.12xlarge:** 15 GB Llama 3 8B from S3 directly to GPU in **4.88 s @ concurrency 32** (8.45 s @16, 28.24 s @4); total vLLM ready **23.18 s**.
- **Azure AKS A100 vLLM v0.23.0 phi-4 ~29 GB via runai_streamer from Azure Blob:** **~10 s total zero disk staging** vs **~187 s** default HF→disk→GPU (113 + 74) → scales to **~5.2 s for 15 GB**.
- **Tensorizer:** GPT-J-6B (A40) **8.22 s median / 10.74 s avg** vs Safetensors **15.07 s avg** vs HF `.bin` **17.26 s avg**. 20 GB GPT-J wire speed ~5 GB/s (~4 s) over 40GbE HTTP/S3. AWS S3 Llama 3 8B 15 GB: **37.36 s GPU load with 16 readers** (65.18 s total vLLM), local IO2 SSD **10.36 s** (30.88 total), GP3 **16.11 s** (36.19 total).
- **Extrapolated to Colab L4** (overlap network chunk n+1 with PCIe DMA chunk n): wall time = network duration (PCIe 1.1 s for 15 GB hidden). Streaming 15 GB from co-located GCS via runai_streamer across 5.3–8.1 Gbps (665–1,013 MB/s) → **15.2–23.1 s** (or 7.5 s over a 2 GB/s cloud link). Pre-tensorized 15 GB from HF Hub HTTPS via Tensorizer @125–200 MB/s → **76.8–122.9 s**.

**Scaling to 75 GB:**
- Host memory decoupled: bounded buffers (`runai_streamer memory_limit=2GB` or Tensorizer Plaid Mode single-tensor) → constant **~1–2 GB DRAM** regardless of checkpoint size (vs Route 1 crash at 26 GB shm, Route 2 thrash at 52 GB page cache).
- Network time linear: CoreWeave OPT-30B ~60 GB on A100 — Tensorizer **23.23 s median** (22.8 avg) vs Safetensors **36.75 s** (39.3 avg) vs HF **35.18 s** (32.1 avg). Extrapolated 75 GB raw network→GPU **~24.4 s** at 32-stream S3 rates (~3 GB/s) or **~76–115 s** across Colab 5.3–8.1 Gbps ingress from GCS.
- **But** the 23 GB L4 VRAM is a hard barrier: streaming 75 GB into a single L4 → CUDA OOM after ~22.5 GB unless sharded across GPUs (`--tp 4`, each rank streams 18.75 GB) or quantized layer-by-layer during ingestion.

**Limiting factors:** 15 GB — object-store/Hub egress BW (plus Linux TCP stack at 40GbE); disk I/O and PCIe hidden by async pipelining. 75 GB — 23 GB L4 VRAM on one accelerator; with online quantization or TP, network egress remains the sole limiter.

## 5. Does download speed change with destination, or only the load step?
Both change, via three mechanisms:

1. **Destination write throttling.** When network ingress > physical disk write BW, download speed is constrained. Low-throughput clients (35 MB/s Python HTTP, 55 MB/s Git LFS) are below PD write ceiling (180 MiB/s standard, 675 SSD) → identical speed to shm or disk. High-throughput hf_transfer (125–250+ MB/s) or regional object storage (665–1,013 MB/s NIC) → `/dev/shm` or a pinned RAM ring buffer runs at DRAM bus >20 GB/s, full wire speed. Writing to local PD initially bursts into page cache until dirty pages hit `vm.dirty_ratio` (~10 GB on 52 GB VM), then the kernel blocks `write()` and throttles the network receive loop to 180 MiB/s standard PD. For 75 GB over a high-speed link, downloading to standard PD is **3.5–5.5× slower** than RAM/GPU streaming during the download itself.
2. **Route 3 protocol change via byte-range concurrency.** Standard downloads are sequential/low-thread. runai_streamer opens 16–32 concurrent HTTP/S3 byte-range streams inside a single `.safetensors` file. Measured S3: 4→32 readers compresses 15 GB ingestion from **28.24 s (~531 MB/s)** to **4.88 s (~3,073 MB/s)** = **5.8× acceleration**.
3. **GPU load step transformation.** Route 1 (shm) → Route 2 (cold disk) degrades GPU load from **~1.1 s** RAM→VRAM PCIe to **47–74 s** single-threaded mmap disk read. Route 3 eliminates the standalone load step by fusing PCIe DMA concurrently into the network receive pipeline → post-download load latency **~0 s**.

## 6. Comprehensive comparison
| Route & scenario | 15 GB download | 15 GB GPU load | 15 GB total | 75 GB total & behavior | Primary bottleneck | Basis |
| :--- | ---: | ---: | ---: | :--- | :--- | :--- |
| **Route 1: shm + std HTTP (35 MB/s)** | ~438.9 s | ~1.1–3.0 s | ~440–442 s | Fails at 26 GB (ENOSPC/OOM) | Network (single-stream HTTP) | Inferred 35 MB/s |
| **Route 1: shm + hf_transfer (125–200 MB/s)** | 76.8–122.9 s | ~1.1–3.0 s | 77.9–125.9 s | Fails at 26 GB (unless shard-cycled: ~390–640 s + quant) | Network → PCIe 4.0 | Measured 125–200 MB/s |
| **Route 2: disk (Colab warm cache, 180 MB/s)** | ~85.3 s | ~1.1–3.0 s | ~86.4–88.3 s | Cache thrashes → cold disk: ~660–1040 s | Disk write → page cache | fio & VFS |
| **Route 2: disk (Colab cold PD, 179–250 MB/s)** | ~85.3 s | 61.4–85.8 s | 146.7–171.1 s | ~660–1040+ s (write clamp + cold read + offload/quant) | Disk write & cold read | fio |
| **Route 2: disk (AWS GP3/IO2, default HF loader)** | N/A (pre-staged) | 47.00–47.99 s (62.7–66.1 s vLLM) | 47–66.1 s load only | ~235–330 s load only (OOMs on 23 GB L4 without quant) | Single-threaded mmap page faults | Measured Llama3 8B 15 GB |
| **Route 2: disk (Azure A100, default vLLM)** | ~58.4 s (113 s @29 GB) | ~38.3 s (74 s @29 GB) | ~96.7 s (187 s @29 GB) | ~483 s (OOMs on 23 GB L4 without quant) | Serial network + disk read | Measured 29 GB phi-4 |
| **Route 3: runai_streamer (S3/Azure Blob → GPU)** | 0 s (pipelined) | 4.88 s (S3 15 GB) / ~5.2 s (Azure 15 GB) | 4.88 s (23.18 s vLLM ready) | ~24.4 s stream (constant 2 GB RAM; needs TP/quant for VRAM) | Object-store egress BW | Measured 15 GB S3 & 29 GB Azure |
| **Route 3: tensorizer (40GbE HTTP/S3/CoreWeave)** | 0 s (pipelined) | ~3–8.22 s (CW) / 37.36 s (AWS S3) | 8.22–37.36 s | 23.23 s median measured on OPT-30B ~60 GB; ~29 s @75 GB concurrency | Network/libcurl stream | Measured GPT-J, Llama3 8B, OPT-30B |

## 7. Precision formats on the L4
**Hardware evaluation:** L4 Ada SM89, 24 GB GDDR6 (~23 usable), ~300 GB/s, 4th-gen Tensor Cores ~121 TFLOPS FP16/BF16, ~242 TFLOPS FP8/INT8. Bandwidth is ~3× lower than L40S (864 GB/s) and 6.7× lower than A100 (2,000 GB/s). Batch-1 decode is strictly memory-bandwidth bound. Precision governs three metrics: **cold-start I/O volume** (16→8/4-bit cuts 50–75%), **decode throughput** (halving/quartering bytes across 300 GB/s → 1.8–3.5×), and **KV-cache headroom** (8B BF16 15 GB leaves ~5–6 GB for KV+workspace; 8-bit 8.46 GB or 4-bit ~5 GB liberates 12–16 GB for 32K–128K context).

**Ranking on L4:**
- **FP8 (W8A8 E4M3/E5M2 and W8A16 Marlin) — Tier 1 optimal for 7B–16B.** Native SM89 FP8 Tensor Cores (Ampere A100/A10G/RTX3090 lack FP8 and fall back to W8A16 dequant). Pre-quantized FP8 (`quant_method: "fp8"`) compresses 15 GB 8B → **8.46 GB**, doubles Tensor Core throughput vs BF16, halves GDDR6 BW, near-lossless. Pair with FP8 KV (`--kv-cache-dtype fp8`) to halve KV memory with minimal TTFT overhead.
- **4-bit AWQ & 4-bit GPTQ (W4A16 Marlin) — Tier 1 optimal for 14B–32B.** ~0.6 GB VRAM per B params including scales/zero-points. 7–8B → ~4.5–5.5 GB; 24–32B → ~15–19 GB fits 23 GB. Legacy autoawq kernels are under-optimized, but modern vLLM/SGLang auto-dispatch to Marlin kernels (AWQ-Marlin, GPTQ-Marlin) on Ada SM89, fusing async global→shared prefetch + mma. AWQ is slightly stronger on instruction-tuned/multimodal by scaling salient channels via activation magnitudes; GPTQ and AWQ are identical speed once repacked to Marlin.
- **BF16 — Tier 2 high.** Native SM89, ideal for unquantized ~15 GB. Strictly superior to FP16: 8-bit exponent matches FP32 dynamic range, prevents overflow/underflow NaN/Inf in modern models (Gemma3/4, Qwen3, Llama3) trained natively in BF16. Old T4 Turing SM75 lacked BF16.
- **FP16 — Tier 2 moderate.** Native SM89 but narrower dynamic range. Useful only when a checkpoint is natively FP16 (older GPTQ/AWQ scale tensors).
- **INT8 (W8A8 SmoothQuant/llm-compressor or W8A16 BitsAndBytes) — Tier 3.** Supported via llm-compressor W8A8, but superseded by FP8 W8A8: same 1-byte footprint, superior non-uniform precision across outliers without fragile SmoothQuant calibration. Dynamic bitsandbytes INT8/NF4 works for memory reduction but has heavy runtime dequant overhead vs native FP8/Marlin.

**Direct-load runtime support:**

| Format | L4 suitability | vLLM/SGLang + runai_streamer | vLLM + CoreWeave tensorizer | Constraints |
| :--- | :--- | :--- | :--- | :--- |
| FP8 W8A8/W8A16 Marlin | Optimal Tier 1: native SM89 FP8, 50% VRAM/I/O reduction | Full support (pre-quant `.safetensors` F8_E4M3 + F32 scales) | Conditional/restricted: historical `ValueError Unknown quant type got fp8` & dtype bugs (Tensorizer v2.x #171); requires pre-serializing via vLLM compressed-tensors path | Avoid online dynamic FP8 quant (`--quantization fp8` on BF16 repo) for models >23 GB: vLLM loads original precision before quantizing → OOM. Always stream pre-quant FP8. |
| 4-bit AWQ | Optimal Tier 1: 75% VRAM reduction; fits 27–32B in 23 GB | Full support (streams packed INT32 weight/zero/scale, repacked to Marlin on GPU) | Supported but requires pre-serialization: load AWQ in vLLM, run `tensorize_vllm_model.py` to serialize post-repacked GPU tensors | Ensure checkpoint uses version `"GEMM"` (or Marlin-compatible group size 128) so execution upgrades to AWQ-Marlin. |
| 4-bit GPTQ | Optimal Tier 1: matches AWQ-Marlin speed; 75% reduction | Full support (streams pre-quant `.safetensors`, repacked to Marlin on GPU) | Supported but requires prior conversion to `.tensors` via vLLM serializer | GPTQModel/Marlin auto-repacks weights in GPU memory after runai_streamer finishes each tensor. |
| BF16 | High Tier 2: native SM89, ideal for unquant ~15 GB | Full native | Full native (explicitly supported in Tensorizer binary schema, modulo Issue #206 on complex64) | 2 bytes/param; limits 8B KV to ~5–6 GB and caps decode ~18–20 tok/s due to 300 GB/s bus. |
| FP16 | Moderate Tier 2: native SM89, narrower range than BF16 | Full native | Full native | Use only when weights are natively FP16 (older GPTQ/AWQ scales). |
| INT8 | Moderate Tier 3: supported via llm-compressor W8A8, inferior to FP8 | Full support for compressed-tensors/SmoothQuant `.safetensors` | Supported (`bitsandbytes_8bit` and compressed-tensors recognized) | Prefer FP8 W8A8 over INT8 W8A8 on L4 for accuracy and simpler scaling. |

## 8. GGUF: can it stream directly to GPU without local storage?
**No.** No production runtime can stream GGUF directly from a remote HTTP/HTTPS URL or HF Hub into GPU without first writing the file to local storage. Once `.gguf` is local (PD or `/dev/shm`), the runtime does **not** need to load the entire file into host DRAM if configured with the right mmap flags.

Why remote-to-GPU GGUF streaming does not exist:
- llama.cpp (`llama-cli`, `llama-server`) flags `-hf org/repo:Q4_K_M` or `-mu/--model-url https://.../model.gguf` are strictly **two-phase download-then-load wrappers** (`common/download.cpp`, `hf-cache.cpp`). libcurl downloads the complete `.gguf` into a local cache (`~/.cache/huggingface/hub` or llama.cpp cache), checks `Accept-Ranges` for resumability, writes temp files, atomically renames. Only after the entire file is committed does `common_init_from_params()` invoke the GGUF parser.
- Piping via stdin (`curl -sL <url> | llama-server -m /dev/stdin`) fails immediately: `gguf_init_from_file` requires a seekable fd (`fseek`/`mmap`) to jump across header KV metadata, alignment padding and tensor offset tables. An unseekable pipe (`/dev/stdin`, FIFO) → `ESPIPE Illegal seek`. llama.cpp does support piping text prompts and a seekable in-memory wrapper (`cache:pipe:0`) for small multimodal audio/image inputs in `mtmd-helper.cpp`, but the core model loader needs seekable storage.
- vLLM (`--load-format gguf`) and SGLang (`--load-format gguf`) can execute a local `.gguf`, but their remote object-storage streaming integration (`--load-format runai_streamer`) is restricted to `.safetensors` and rejects `.gguf`.

Local storage→GPU mechanics in llama.cpp:
- **Default (mmap enabled):** maps the entire `.gguf` into process virtual address space. As the loader iterates layers for GPU offload (`-ngl 99`), it copies tensor weights from mmap virtual pointers into CUDA memory, forcing the kernel to fault every GPU-bound page from disk into the OS page cache before the PCIe copy. If a 40 GB GGUF is partially offloaded (20 GB GPU + 20 GB CPU), mmap backs the entire 40 GB file in page cache.
- **`--no-mmap` + `--no-host`** (bypasses extra host buffers) or **`-lm dio`** (DirectIO): does **not** map the file; sequentially `read()`s each offloaded layer from a local fd through a small pinned staging buffer directly into VRAM via `cudaMemcpyAsync`. GGUF must still be staged to local storage first, but `--no-mmap` ensures GPU-offloaded layers stream from disk to VRAM without requiring the full checkpoint in host DRAM simultaneously.

## 9. Actionable recommendations
### Strategy 1 — eliminate the >26 GB transfer via offline FP8 or 4-bit Marlin (preferred for single-GPU L4)
Most >26 GB cases are 16-bit models (14–16B ~28–32 GB BF16; 27–32B ~54–64 GB BF16) that the practitioner intends to quantize at load time. But L4 has only 23 GB usable VRAM, so unquantized cannot fit anyway. vLLM warns that online dynamic quantization loads the original precision before quantizing → immediate OOM on 23 GB.

**Fix:** replace the BF16 checkpoint with pre-quantized `.safetensors` in **FP8 W8A8** (shrinks 30 GB 15B → ~15.5 GB) or **4-bit AWQ/GPTQ-Marlin** (shrinks 60 GB 30B → ~16–18 GB). Once <20 GB, it fits both 23 GB VRAM and 26 GB `/dev/shm`.

**Fastest cold start:**
- Absolute fastest (~15–23 s total weight load): stage the pre-quant 15–18 GB `.safetensors` in a regional GCS `gs://` bucket co-located with the Colab VM, launch vLLM/SGLang with `--load-format runai_streamer --model-loader-extra-config '{"concurrency": 16, "memory_limit": 2147483648}'`, streaming straight across network+PCIe into L4 with zero disk writes and only 2 GB host RAM.
- Without a GCS bucket, pulling directly from public HF Hub: `export HF_XET_HIGH_PERFORMANCE=1`, download the pre-quant checkpoint into `/dev/shm` (Route 1), full GPU readiness **~78–110 s**.

### Strategy 2 — fastest when the downloaded artifact must exceed 26 GB (30–75 GB)
When the artifact genuinely >26 GB (e.g., 30–75 GB BF16 into SGLang using layer-by-layer online quant `--load-format layered` so peak VRAM <23 GB, or 30–45 GB GGUF MoE/dense split across L4 23 GB VRAM + ~52 GB host DRAM), `/dev/shm` cannot hold it and standard disk loading thrashes page cache.

**For `.safetensors` served via SGLang/vLLM:**
- Fastest (~25–60 s): host weights in regional GCS `gs://` or S3, launch `--load-format runai_streamer --model-loader-extra-config '{"concurrency": 32, "memory_limit": 2147483648}'`. The 2 GB ring buffer streams 30–75 GB directly from object storage without touching `/dev/shm` or 180 MiB/s PD.
- If weights must be pulled from HF Hub to local Colab disk (`/content`): `export HF_XET_HIGH_PERFORMANCE=1` and launch SGLang with `--load-format layered --weight-loader-drop-cache-after-load`. Layered loading quantizes each transformer block on the fly before loading the next (prevents GPU OOM on 23 GB L4), while `--weight-loader-drop-cache-after-load` issues `posix_fadvise(DONTNEED)` immediately after each `.safetensors` shard is processed, purging the Linux page cache so the checkpoint never thrashes 52 GB host RAM.

**For hybrid GPU+CPU GGUF via llama.cpp:**
- Download `.gguf` directly to `/content` (never `/dev/shm` — crash at 26 GB; never `/content/drive` — FUSE throttles small block reads) using `huggingface_hub.snapshot_download(..., local_dir="/content/model")` with `HF_XET_HIGH_PERFORMANCE=1`, so files write directly to `/content/model` without duplicating blobs in `~/.cache/huggingface`.
- Launch `llama-server --no-mmap --no-host -ngl <N> --n-cpu-moe <M>` (or `--n-cpu-ffn <M>` for dense) `-ctk q8_0 -ctv q8_0` for quantized KV caching. `--no-mmap` + `--no-host` streams GPU-assigned layers directly from `/content` into 23 GB VRAM through a compact staging buffer, leaving ~52 GB host RAM free to hold CPU-resident expert/FFN layers without page-cache eviction stalls during inference.

## 10. Source note
The 23-page PDF cites 68 references, including fio/iperf3 measurements, NVIDIA Run:ai, Azure AKS, CoreWeave Tensorizer, and Hugging Face Hub benchmarks. `Summary.md` was a faithful 165-line condensation; this master document restores the route-level detail, tables and precision/GGUF sections. Some numbers are explicitly inferred or extrapolated; see 11 for the measured-vs-estimated split.
