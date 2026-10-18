# Runbook: Qwen 3.8 Flash-Next on Colab G4 via SGLang

Operational procedure for deploying and serving `Qwen3.8-Flash-Next-NVFP4` on a single Google Colab NVIDIA G4 instance (RTX PRO 6000 Blackwell Server Edition, 96 GB VRAM, ~177 GB host RAM) using SGLang.

Primary references:
- Official SGLang Cookbook: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next
- NVIDIA Official NVFP4 Checkpoint: https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4
- RadixArk Alternative Checkpoint: https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4
- Upstream SGLang PR #38121: https://github.com/sgl-project/sglang/pull/38121
- Google Colab FAQ: https://research.google.com/colaboratory/faq.html
- NVIDIA Blackwell Architecture: https://www.nvidia.com/en-us/data-center/blackwell-architecture/

---

## 1. Overview & Architecture

Qwen 3.8 Flash-Next is an extreme-throughput 125B Mixture-of-Experts (MoE) hybrid architecture:
- **Base Architecture**: 48 hidden layers (36 recurrent Gated DeltaNet linear attention + 12 full attention with QSA sequence pruning).
- **MoE Routing**: 512 routed fine-grained experts with top-10 routing + 1 shared expert (~6.0B active parameters per token).
- **Lexical Embedding (PLE)**: 51.2B parameter piece-level embedding table (~47.7 GiB in FP8) offloaded to host system RAM via `--ple-offload-embedding` with asynchronous PCIe prefetch.
- **Speculative Head**: Integrated 1-layer MoE MTP head (~2.76B physical parameters) quantized in FP8 (`FP8_PB_WO`).
- **Memory Footprint on 96 GB G4**: Checkpoint weights stay resident in ~78 GiB VRAM (74.7 GiB without MTP / 81.8 GiB with MTP resident), leaving 12.3–19.4 GiB free for KV cache. Because 12 full-attention layers with 2 KV heads require only **12 KiB/token (FP8)**, 12 GiB of KV cache comfortably accommodates >800,000 tokens (well beyond the 262K native context limit).

---

## 2. Checkpoint Staging and Drive Caching

The NVIDIA checkpoint `nvidia/Qwen3.8-Flash-Next-NVFP4` consists of **132.68 GB (132.7 GB)** across 11 safetensors shards:
- 78.96 GB for the base MoE and hybrid attention backbone
- 53.72 GB for the auxiliary PLE and MTP shards

To prevent downloading 132.7 GB across ephemeral VM reboots (which burns 10–20 minutes and ~1.5–3.0 CU at ~8.71 CU/hr), stage and cache the safetensors on Google Drive:

Canonical Path: `/content/drive/MyDrive/colab_llm/checkpoints/nvidia/Qwen3.8-Flash-Next-NVFP4/`

### Download and Caching Script (In-Guest):

```bash
# Ensure Drive is mounted at /content/drive
mkdir -p /content/drive/MyDrive/colab_llm/checkpoints/nvidia/Qwen3.8-Flash-Next-NVFP4

# Use huggingface_hub with fast transfer enabled
python3 -c '
from huggingface_hub import snapshot_download
import os

target_dir = "/content/drive/MyDrive/colab_llm/checkpoints/nvidia/Qwen3.8-Flash-Next-NVFP4"
print(f"Downloading nvidia/Qwen3.8-Flash-Next-NVFP4 (132.7 GB) to {target_dir}...")
snapshot_download(
    repo_id="nvidia/Qwen3.8-Flash-Next-NVFP4",
    local_dir=target_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=8
)
print("Download and caching complete.")
'
```

---

## 3. SGLang Launch Configuration: NVIDIA vs. RadixArk Flags

The official SGLang cookbook specifies distinct launch requirements depending on whether you deploy the official NVIDIA checkpoint or the RadixArk quant:

| Setting / Flag | NVIDIA (`nvidia/Qwen3.8-Flash-Next-NVFP4`) | RadixArk (`RadixArk/Qwen3.8-Flash-Next-NVFP4`) |
| :--- | :--- | :--- |
| **Quantization Format** | `modelopt_mixed` (mixed precision: linear attn/embed in FP8/BF16, MoE in NVFP4) | ModelOpt NVFP4 (auto-detected) |
| **`--quantization` flag** | **DO NOT PASS** (Quantization is auto-detected from checkpoint config; passing `--quantization` causes loader failure) | **DO NOT PASS** (Cookbook passes no `--quantization` flag for either export) |
| **Docker Container** | `lmsysorg/sglang:dev-qwen38-next-local` (requires PR #38121) | `lmsysorg/sglang:dev-qwen38-next-local` (or standard release with FP4 support) |
| **Tensor Parallelism** | `--tp 1` (verified single RTX PRO 6000 execution) | `--tp 1` |
| **Host PLE Offload** | `--ple-offload-embedding` | `--ple-offload-embedding` |
| **Host Memory Requirement** | `>= 64 GB free host RAM` (Colab G4 has ~177 GB host RAM) | `>= 64 GB free host RAM` |
| **Memory Pinning** | Docker `--ulimit memlock=-1` (Colab Docker availability and native mlock are UNVERIFIED) | Docker `--ulimit memlock=-1` |

### Option A: Docker Launch (Cookbook Verified Method)

> [!WARNING]
> **Colab Docker Availability is UNVERIFIED**: Standard Colab VMs often do not provide a running Docker daemon or have Docker installed. The official SGLang cookbook verified path is Docker with `--ulimit memlock=-1`. Both Docker availability and native host memory pinning (`mlock`) must be probed via [`g4-probe.md`](g4-probe.md). G4 unavailable or mlock fails → Flash-Next on Colab is blocked; stop and report.

```bash
# Launch inside Docker with unlimited memlock for host PLE pinning (if Docker is present)
docker run --gpus all \
  --shm-size 32g \
  --ulimit memlock=-1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p 30000:30000 \
  -v /content/drive/MyDrive/colab_llm:/colab_llm \
  lmsysorg/sglang:dev-qwen38-next-local \
  python3 -m sglang.launch_server \
    --model-path /colab_llm/checkpoints/nvidia/Qwen3.8-Flash-Next-NVFP4 \
    --tp 1 \
    --ple-offload-embedding \
    --host 0.0.0.0 \
    --port 30000 \
    --api-key "${LLAMA_API_KEY}" \
    --mem-fraction-static 0.96 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-num-draft-tokens 4 \
    --speculative-eagle-topk 1 \
    --trust-remote-code
```

> [!NOTE]
> - **Throughput Expectations**: Expect **~148 tok/s** with MTP (`--mem-fraction-static 0.96`) and **~83 tok/s** without MTP (`--mem-fraction-static 0.93`).
> - **Cookbook Quantization Matching**: The cookbook passes no `--quantization` flag for either export (NVIDIA or RadixArk). For RadixArk, simply point `--model-path` to the RadixArk directory.

### Option B: Native Host Launch (Subject to Native `mlock` Probe)

If the probe in [`g4-probe.md`](g4-probe.md) confirms native host `mlock` permission without Docker:

#### 1. Install SGLang from Source (UNVERIFIED on Colab CUDA 13 / sm_120)
Native host execution requires building SGLang from source to include the Flash-Next mixed precision loader from PR #38121 (or the commit in `lmsysorg/sglang:dev-qwen38-next-local`):

```bash
# Install SGLang from source with PR #38121 (UNVERIFIED on Colab CUDA 13 / sm_120)
git clone https://github.com/sgl-project/sglang.git
cd sglang
git fetch origin pull/38121/head:pr-38121 && git checkout pr-38121
pip install -e "python[all]"
```

> [!WARNING]
> **Drive Mount I/O Bottleneck:** Loading the 133 GB checkpoint directly from the Google Drive FUSE mount (`/content/drive`) is slow and can take 20–40+ minutes due to FUSE overhead. If local disk has sufficient space (`df -h /content` shows >= 150 GB free), copy the checkpoint to local disk before launching:
> ```bash
> df -h /content
> cp -r /content/drive/MyDrive/colab_llm/checkpoints/nvidia/Qwen3.8-Flash-Next-NVFP4 /content/Qwen3.8-Flash-Next-NVFP4
> ```

#### 2. Native Launch Command

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# If local copy ran, use /content/Qwen3.8-Flash-Next-NVFP4; otherwise keep the Drive path (/content/drive/MyDrive/colab_llm/checkpoints/nvidia/Qwen3.8-Flash-Next-NVFP4).
python3 -m sglang.launch_server \
  --model-path /content/Qwen3.8-Flash-Next-NVFP4 \
  --tp 1 \
  --ple-offload-embedding \
  --host 0.0.0.0 \
  --port 30000 \
  --api-key "${LLAMA_API_KEY}" \
  --mem-fraction-static 0.96 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-num-draft-tokens 4 \
  --speculative-eagle-topk 1 \
  --trust-remote-code > /tmp/sglang.log 2>&1 &
```

> [!NOTE]
> Expect **~148 tok/s** with MTP (`--mem-fraction-static 0.96`) and **~83 tok/s** without MTP (`--mem-fraction-static 0.93`). Cookbook passes no `--quantization` flag for either export.

---

## 4. Exposing the OpenAI-Compatible `/v1` Endpoint

Expose port 30000 over a public trycloudflare tunnel:

```bash
# Spawn cloudflared to tunnel SGLang
cloudflared tunnel --url http://127.0.0.1:30000 > /tmp/cloudflared.log 2>&1 &

# Extract tunnel URL from log file using Python (GNU grep -E lacks negative lookahead support)
TUNNEL_URL=$(python3 -c '
import re, time, sys, os
regex = re.compile(r"https://(?!api\.)[a-z0-9]+(?:-[a-z0-9]+)+\.trycloudflare\.com")
for _ in range(60):
    time.sleep(1)
    if os.path.exists("/tmp/cloudflared.log"):
        with open("/tmp/cloudflared.log", "r") as f:
            m = regex.search(f.read())
            if m:
                print(m.group(0))
                sys.exit(0)
sys.exit(1)
')

echo "TUNNEL_READY_URL=${TUNNEL_URL}"
echo "${TUNNEL_URL}" > /tmp/tunnel_url.txt

# Sleep 5-8 seconds to avoid NXDOMAIN DNS caching
sleep 8
```

---

## 5. Smoke & Functional Verification

Set your local or guest environment variables:
```bash
SGLANG_URL="http://127.0.0.1:30000"  # or "${TUNNEL_URL}"
```

### Smoke 1: Health & Model Listing

```bash
# Health check (open, no auth)
curl -sf "${SGLANG_URL}/health"

# Model discovery (authenticated)
curl -sf "${SGLANG_URL}/v1/models" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" | jq .
```

### Smoke 2: Autonomous Coding Completion

```bash
curl -sf "${SGLANG_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "temperature": 0.0,
    "max_tokens": 256,
    "messages": [
      {"role": "user", "content": "Write a self-contained Python module implementing an iterative AVL tree with insert, delete, and rebalance. Code only, no prose."}
    ]
  }' | jq '{
    content: .choices[0].message.content,
    usage: .usage
  }'
```

### Smoke 3: Function & Tool Calling

```bash
curl -sf "${SGLANG_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${LLAMA_API_KEY}" \
  -d '{
    "model": "nvidia/Qwen3.8-Flash-Next-NVFP4",
    "temperature": 0.0,
    "max_tokens": 128,
    "messages": [
      {"role": "user", "content": "Check the available disk space on /content in gigabytes."}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "check_disk_space",
        "description": "Inspect disk usage on target path",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {"type": "string"},
            "unit": {"type": "string", "enum": ["GB", "MB"]}
          },
          "required": ["path", "unit"]
        }
      }
    }],
    "tool_choice": "required"
  }' | jq '{
    finish_reason: .choices[0].finish_reason,
    tool_calls: .choices[0].message.tool_calls
  }'
```

---

## 6. Active UNVERIFIED Probes to Track

During execution of this runbook, the operator must treat the following five items as strictly **UNVERIFIED** until empirical measurements are logged:

1. **AI Pro G4 Entitlement**:
   - *Status*: **UNVERIFIED**.
   - *Observation*: Whether `colab new --gpu G4` assigns a VM under the Google AI Pro subscription pool without rejection or fallback to A100/L4.
2. **Colab Docker Availability & Native mlock**:
   - *Status*: **UNVERIFIED**.
   - *Observation*: Standard Colab VMs frequently do not provide a running Docker daemon or have Docker installed. Whether Docker is available in the provisioned Colab environment is UNVERIFIED. Furthermore, whether the standard Colab kernel user environment allows pinning >= 64 GB host memory for the 51.2B PLE table outside of Docker is UNVERIFIED. The official cookbook verified path requires Docker with `--ulimit memlock=-1`. Both Docker and native `mlock` must be tested via [`g4-probe.md`](g4-probe.md).
3. **Real Cold-Boot Latency and CU Overhead**:
   - *Status*: **UNVERIFIED**.
   - *Observation*: The true wall-clock time and CU burn required to download 132.7 GB of safetensors and JIT-compile FlashInfer CUTLASS kernels on ephemeral Colab VM disks.
4. **Abliterated NVFP4 Checkpoint Load & Perplexity Viability**:
   - *Status*: **UNVERIFIED**.
   - *Observation*: Whether community abliterated NVFP4 quants (`edp1096/Huihui-Qwen3.8-Flash-Next-abliterated-NVFP4-QAD` or `claidler/Qwen3.8-Flash-Next-Ablit-NVFP4-QAD-LMHead-NVFP4`) load cleanly without tensor shape mismatch errors in SGLang or perplexity degradation.
5. **Real-World G4 Server Edition Decode Speed**:
   - *Status*: **UNVERIFIED**.
   - *Observation*: The cookbook benchmark of ~148 tok/s (with MTP) was recorded on an active-cooled RTX PRO 6000 Workstation Edition (1,792 GB/s bandwidth). Colab's passively cooled Server Edition runs at 1,597 GB/s (10.9% lower bandwidth), with an expected decode speed of ~130–135 tok/s. Real speed must be empirically verified on live G4 silicon.
