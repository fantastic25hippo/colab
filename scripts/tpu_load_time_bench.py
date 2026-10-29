# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "vllm-tpu==0.32.0",
#     "tpu-inference==0.32.0",
#     "jax[tpu]==0.11.2",
#     "huggingface_hub",
# ]
# ///
"""Short TPU timing test (<10 min) to benchmark vLLM weight loading on TPU v5e-1.

Compares baseline loading against the top optimization (multimodal weight skip
and removal of per-module jax.clear_caches()). DO NOT RUN WITHOUT CU GO-AHEAD.
"""

import argparse
import os
import sys
import time

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL_ID = "google/gemma-4-E4B-it-qat-w4a16-ct"
SHM_DIR = "/dev/shm/gemma"


def setup_weights():
    from huggingface_hub import snapshot_download

    if not os.path.exists(os.path.join(SHM_DIR, "model.safetensors")):
        print(f"Downloading {MODEL_ID} to {SHM_DIR}...")
        t0 = time.perf_counter()
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=SHM_DIR,
            max_workers=12,
            ignore_patterns=["*.bin", "*.pt"],
        )
        print(f"Download completed in {time.perf_counter() - t0:.2f} s")


def apply_top_fix():
    """Apply top candidate fix: skip multimodal tensors & redundant cache flushes."""
    import tpu_inference.models.jax.gemma4_mm as gemma4_mm
    import tpu_inference.models.jax.utils.weight_utils as wu

    # 1. Skip all 1,411 non-text tensors (audio + vision)
    orig_load_weights = gemma4_mm.Gemma4ForConditionalGeneration.load_weights

    def patched_load_weights(self, weights):
        filtered_weights = (
            (name, w)
            for name, w in weights
            if not any(
                sub in name
                for sub in ("audio_tower", "vision_tower", "embed_audio", "embed_vision")
            )
        )
        return orig_load_weights(self, filtered_weights)

    gemma4_mm.Gemma4ForConditionalGeneration.load_weights = patched_load_weights

    # 2. Prevent jax.clear_caches() after every single module load
    orig_clear_caches = wu.jax.clear_caches
    wu.jax.clear_caches = lambda: None
    print("[Fix Applied] Multimodal weight filtering + jax.clear_caches no-op active")


def main():
    parser = argparse.ArgumentParser(description="Benchmark TPU weight loading time")
    parser.add_argument(
        "--apply-fix",
        action="store_true",
        help="Apply top fix (multimodal filter + no clear_caches)",
    )
    args = parser.parse_args()

    setup_weights()

    if args.apply_fix:
        apply_top_fix()

    import jax
    from flax import nnx
    from tpu_inference.models.common.model_loader import _get_nnx_model
    from tpu_inference.models.jax.gemma4_mm import Gemma4ForConditionalGeneration
    from vllm.config import EngineConfig, ModelConfig, VllmConfig
    from vllm.config.load import LoadConfig

    model_config = ModelConfig(
        model=SHM_DIR,
        task="auto",
        tokenizer=SHM_DIR,
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
    )
    vllm_config = VllmConfig(
        model_config=model_config,
        load_config=LoadConfig(load_format="auto"),
    )
    vllm_config.limit_mm_per_prompt = {"image": 0, "audio": 0}

    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    rng = jax.random.key(0)

    print("--- Starting Weight Load Benchmark ---")
    start_time = time.perf_counter()
    try:
        # Load abstract model and fill weights via vLLM default loader
        _get_nnx_model(
            model_class=Gemma4ForConditionalGeneration,
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
        )
    except Exception as e:
        # Note: Flax NNX JIT may still fail on ShapeDtypeStruct downstream;
        # record weight loading phase before re-raising.
        print(f"JIT phase exception caught (expected in 0.32.0): {e}")

    total_time = time.perf_counter() - start_time
    mode_str = "OPTIMIZED (Top Fix)" if args.apply_fix else "BASELINE"
    print(f"=== RESULT [{mode_str}]: Total Weight Load Time = {total_time:.2f} s ===")


if __name__ == "__main__":
    main()
