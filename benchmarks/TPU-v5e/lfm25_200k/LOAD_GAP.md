# TPU v5e Cold-Start Gap: Qwen3-4B (80.7s) vs LFM2.5-24B-A2B (278-389s)

## 1. Why Qwen3-4B is Fast (80.7s Global / 72.5s In-VM)
Qwen3-4B achieves an 80.70s global cold-start floor (72.54s in-VM) via complete phase pipelining and aggressive runtime pruning on TPU v5e-1. First, venv preparation uses the NIC upfront (2.54s download, 6.51s zstd unpack to /dev/shm). Next, subrunner Python imports and libtpu initialization take 19.51s, running in parallel with background weight download. Weight download pulls 8.06 GB across 3 safetensors shards (two ~4 GB shards and one ~100 MB shard) into /dev/shm at 1337 MB/s in 6.03s; because the 19.51s import window completely dominates the 6.03s download, net wait stall is 0.00s. Network stalls during initialization are eliminated via /etc/hosts and a monkeypatch on urllib.request.urlopen that immediately aborts metadata.google.internal polling with URLError. Ingestion transfers dense BF16 weights to HBM in 11.85-11.98s; monkeypatching jax.clear_caches to a no-op preserves JAX dispatch caches (feasible on Qwen because dense 4B weights fit comfortably in 15.748 GiB HBM without risking OOM). Finally, compilation overhead is eliminated via a persistent XLA cache hit (58-109 entries restored from tar in ~2.0s) combined with a token padding ladder pruned to 2 boundary buckets ([16, 8192]) and structured-decoding precompilation bypass, resulting in 0 new compilations and 12.19-12.39s compile/warmup.

## 2. Side-by-Side Step Comparison

| Startup Step / Attribute | Qwen3-4B (Fast Path) | LFM2.5-24B-A2B (Current) | Difference & Architectural Impact | Likely Cost |
| :--- | :--- | :--- | :--- | ---: |
| VM Provisioning | 8.16s [MEASURED] | 5.23s [MEASURED] | Standard Colab TPU v5e allocation; identical flow | -2.93s [MEASURED] |
| In-VM Bootstrap & SHM Remount | 0.01s [MEASURED] | 0.04-0.06s [MEASURED] | Mounts /dev/shm exec + /etc/hosts metadata bypass | +0.04s [MEASURED] |
| GCE Metadata Bypass | /etc/hosts + urlopen patch [MEASURED] | /etc/hosts only [MEASURED] | Qwen monkeypatches urllib.request.urlopen to raise URLError instantly | +0.50s [INFERRED] |
| Python Imports & TPU Init | 19.39-19.51s [MEASURED] | 20.20-21.19s [MEASURED] | JAX/vLLM import + libtpu device initialization; similar import floor | +1.68s [MEASURED] |
| Weight DL Volume & Shards | 8.06 GB, 3 shards [MEASURED] | 13.15 GB, 9 shards [MEASURED] | LFM payload is 1.63x larger (9 shards vs Qwen 3 shards: two ~4 GB + 100 MB). Recommend 16-shard resharding (L4 recipe) | +4.78s [MEASURED] |
| Weight DL Concurrency & Overlap | Hidden behind imports [MEASURED] | Concurrent with venv dl [MEASURED] | Qwen runs venv then hides weights behind 19.5s imports (0.00s net wait). LFM weights dl (10.81s) shares NIC with venv dl (9.52s), leaving 0.0-1.3s net wait | +1.29s [MEASURED] |
| Download Flags & Env Order | Post-import HF flags [MEASURED] | Post-import HF flags [MEASURED] | Both set HF_TRANSFER/HF_XET after import; neither disables progress bars. Recommend setting env before import as LFM fix | +1.00s [INFERRED] |
| Weight Ingestion & H2D | 11.98s (stacked BF16) [MEASURED] | 119.32s (unstacked int8/int4) [MEASURED] | MoE experts unstacked; hundreds of separate H2D transfers and casts | +107.34s [MEASURED] |
| Cache Clearing (jax.clear_caches) | Monkeypatched to no-op [MEASURED] | Live (frees weight-load JIT) [MEASURED] | Qwen no-op avoids dispatch clears. LFM leaves live on purpose to free weight JIT before KV alloc (2.04 GiB KV in 15.748 GiB) | HIGH HBM RISK |
| Structured Decoding Precompile | Monkeypatched to no-op [MEASURED] | Monkeypatched to no-op [MEASURED] | Parity: both bypass guided decoding precompilation | 0.00s [MEASURED] |
| Max Context (max_model_len) | 32,768 (32K) [MEASURED] | 200,000 (200K) vs 131,072 [MEASURED] | Repacked config uses 200,000 vs official 131,072 limit, inflating KV metadata and graph memory | +5.00s [INFERRED] |
| XLA Compilation & Warmup | 12.19-12.39s (cached) [MEASURED] | 228.24s (cold) [MEASURED] | Nested costs (+216s total [MEASURED]): cache miss forces full JIT compile; warmup is 101.86s (bucket 4096 adds 38.57s over 2-bucket baseline) | +216.05s [MEASURED] |
| Time to First Token (TTFT) | 0.021s (16-tok bucket) [MEASURED] | 0.160s (raw) / 0.592s (Run B) [MEASURED] | Prompt evaluation and first token generation | +0.14s [MEASURED] |

## 3. Ranked Fix List (Seconds Saved per Unit of Risk)

1. **Restore Pre-compiled LFM XLA Cache (avgrocketman/tpu-compile-cache)**
   - *Files & Functions*: `lfm25_coldstart_driver.py` (`dl_venv`, `unpack_venv`), `lfm25_cache.py` (`restore_cache`).
   - *Golden Verification*: Compare generated token IDs against Run B golden (`[597, 4695, 20589, 34, 496, 89509, 278, 9761, 9896, 302, 5030, 37933, 1003]`).
   - *Expected Phase Time*: `compile_warmup` drops from 228.24s to ~16.00s [INFERRED]. (~212s saved, Zero risk).

2. **Stack MoE Expert Tensors for Host-to-Device Transfer**
   - *Files & Functions*: Offline repacking script + `tpu_v5e_patches.py` (`_patch_loaded_module`), checkpoint safetensors. Stack expert weights along a leading expert dimension to eliminate hundreds of individual small H2D transfers and casts.
   - *Golden Verification*: Assert bit-exact logits match and exact token IDs on Run B golden prompt.
   - *Expected Phase Time*: `weight_load` drops from 119.32s to ~20.00s [INFERRED]. (~99s saved, Low risk).

3. **Prune Padding Ladder to 2 Buckets ([16, 8192])**
   - *Files & Functions*: `tpu_v5e_patches.py` (`_wrap_paddings`), `lfm25_engine_subrunner.py` (`CACHE_KEY_ARGS`). Prune `[16, 4096, 8192]` down to `[16, 8192]`, dropping the 4096 bucket.
   - *Golden Verification*: Run B golden prompt (length 22) pads to 8192 instead of 4096 and generates bit-exact tokens.
   - *Expected Phase Time*: Warmup execution drops by 38.57s [MEASURED]. (~38s saved, Zero risk).

4. **16-Shard Checkpoint Plan & HF Transfer Optimization**
   - *Files & Functions*: Offline repacking tool, `lfm25_coldstart_driver.py` (`dl_weights`), `lfm25_engine_subrunner.py`. Reshard 13.15 GB checkpoint from 9 shards into 16 shards using max_workers=12 /dev/shm (adapting the fast L4 recipe). Export `HF_HUB_ENABLE_HF_TRANSFER=1`, `HF_XET_HIGH_PERFORMANCE=1`, `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY=1`, and `HF_HUB_DISABLE_PROGRESS_BARS=1` before importing `huggingface_hub` to eliminate GIL progress bar overhead. Sequence venv first, then overlap weight download with subrunner imports.
   - *Golden Verification*: Check `wait_for_weights_s == 0.00s` and token output parity against Run B golden.
   - *Expected Phase Time*: Net weight download wait drops from ~1.29s to 0.00s (~1s saved, Low risk).

5. **Right-size Context Length from 200,000 to Official 131,072 (or 32,768)**
   - *Files & Functions*: `lfm25_engine_subrunner.py` (`max_model_len=131072` or `32768`). Replace repacked 200K limit with model's native 131,072 limit to reduce KV block metadata allocations and graph memory.
   - *Golden Verification*: Golden Run B chat prompt produces bit-exact tokens.
   - *Expected Phase Time*: Engine init and warmup drop by ~5.00s [INFERRED]. (~5s saved, Low risk).

6. **Fast GCE Metadata Bypass via urlopen Monkeypatch**
   - *Files & Functions*: `tpu_v5e_patches.py` (`urllib.request.urlopen`). Replicate Qwen's monkeypatch on `urllib.request.urlopen` intercepting `metadata.google.internal` requests to raise `URLError` immediately, preventing socket poll timeouts during JAX/libtpu cloud init.
   - *Golden Verification*: Subrunner starts without metadata warnings; Run B golden tokens match bit-exact.
   - *Expected Phase Time*: Imports and initialization save ~0.50-1.00s [INFERRED]. (~1s saved, Low risk).

- **High-Risk Note on `jax.clear_caches`**: Do NOT monkeypatch `jax.clear_caches` to a no-op as a low-risk optimization. Unlike Qwen, LFM intentionally leaves it live to free weight-loading JIT memory before allocating 2.04 GiB KV cache in 15.748 GiB HBM limit. Neutralizing it presents severe TPU HBM OOM risk unless MoE tensor stacking is proven to reduce peak memory below 13.5 GiB.

## 4. Projected LFM Startup & <100s Feasibility

| Pipeline Phase | Current Cold (Nov 1) | No Cache (Stacked) | Cache Alone (Unstacked) | With Cache & Stacking | Notes on Projection |
| :--- | ---:| ---:| ---:| ---:| :--- |
| VM Provisioning | 5.23s [MEASURED] | 5.23s [MEASURED] | 5.23s [MEASURED] | 5.23s [MEASURED] | Standard Colab TPU v5e allocation |
| In-VM Prep & SHM Remount | 0.06s [MEASURED] | 0.06s [MEASURED] | 0.06s [MEASURED] | 0.06s [MEASURED] | tmpfs remount + hosts entry |
| Venv & Cache Download/Unpack | 9.52s [MEASURED] | 9.50s [INFERRED] | 11.50s [INFERRED] | 11.50s [INFERRED] | Includes ~2.0s cache tarball restore |
| Weight Download (Net Wait) | 1.29s [MEASURED] | 0.00s [INFERRED] | 0.00s [INFERRED] | 0.00s [INFERRED] | Hidden behind ~20s imports window |
| Imports & TPU Device Init | 21.00s [MEASURED] | 20.00s [INFERRED] | 20.00s [INFERRED] | 20.00s [INFERRED] | JAX/vLLM imports + libtpu device init |
| Weight Ingestion (H2D Transfer) | 119.32s [MEASURED] | 20.00s [INFERRED] | 119.32s [MEASURED] | 20.00s [INFERRED] | Stacked MoE tensors vs unstacked baseline |
| XLA Compile & Warmup | 228.24s [MEASURED] | 120.00s [INFERRED] | 16.00s [INFERRED] | 16.00s [INFERRED] | Cold 2-bucket JIT vs cached execution |
| Warmup TTFT & First Token | 0.16s [MEASURED] | 0.16s [MEASURED] | 0.16s [MEASURED] | 0.16s [MEASURED] | First token generation on pre-warmed bucket |
| **Total In-VM Startup** | **379.59s [MEASURED]** | **169.72s [INFERRED]** | **167.04s [INFERRED]** | **67.72s [INFERRED]** | In-VM time from boot to first token |
| **Total Global Startup** | **384.82s [MEASURED]** | **174.95s [INFERRED]** | **172.27s [INFERRED]** | **72.95s [INFERRED]** | Host start to first token |

### Feasibility Conclusion
- **<100s Target Requires BOTH Cache AND H2D Stacking**: Reaching sub-100s cold start cannot be achieved by any single optimization. It strictly requires both pre-compiled persistent XLA cache hit AND stacked MoE expert tensor ingestion.
- **Cache Alone (Without Stacking)**: If persistent cache is restored but weight loading remains unstacked (~119s), the in-VM floor is **~167s** (global **~172s**).
- **No-Cache (Cold Compile)**: Even with MoE tensor stacking (~20s) and 2-bucket pruning, cold JIT tracing of ShortConv and attention graphs takes ~120s, keeping the in-VM floor at **~170s** (global **~175s**), well above 100s.
- **Combined With-Cache & Stacking**: Projected in-VM cold-start drops to **~69s** (global **~74s**) only if weight_load falls from ~119s to ~20s, safely breaking the 100s target and approaching Qwen's 73.08s in-VM / 80.70s global benchmark.

### Summary of Golden Verification Protocol
All optimizations must preserve numerical and generative parity against Run B golden outputs:
- **Prompt**: Golden 22-token chat template prompt with BOS token id 124894.
- **Golden Token IDs**: `[597, 4695, 20589, 34, 496, 89509, 278, 9761, 9896, 302, 5030, 37933, 1003]`.
- **Integrity Criteria**: Bit-exact token match, zero NaN/Inf activations, and maximum TPU HBM peak under 15.748 GiB limit.
