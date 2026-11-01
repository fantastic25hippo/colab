"""Canonical naming helper for TPU XLA compilation caches.

Format:
  tpu_compile_cache__<model-slug>__vllm<ver>__len<max_model_len>__b<buckets>__<8-char hash of patches+LLM args>.tar
"""

import hashlib
import json
import os
import re
from typing import Any, Sequence, Union


def build_cache_name(
    model_slug: str,
    vllm_version: str,
    max_model_len: int,
    buckets: Union[str, Sequence[Union[int, str]]],
    llm_args: Union[dict[str, Any], str],
    patch_content: Union[bytes, str],
) -> str:
    """Build canonical archive name from model, version, context, buckets, args, and patches."""
    # 1. Model slug
    slug = os.path.basename(str(model_slug).rstrip("/")).lower()
    slug = re.sub(r"[^a-z0-9.-]+", "-", slug).strip("-")
    if not slug:
        slug = "model"

    # 2. vLLM version
    ver = str(vllm_version).strip()
    if ver.startswith("vllm"):
        ver = ver[4:]

    # 3. Buckets
    if isinstance(buckets, (list, tuple)):
        b_str = "-".join(str(x) for x in buckets)
    else:
        b_str = str(buckets).strip()
        if b_str.startswith("b"):
            b_str = b_str[1:]

    # 4. Hash of LLM args + patch file contents
    if isinstance(llm_args, dict):
        args_repr = json.dumps(llm_args, sort_keys=True)
    else:
        args_repr = str(llm_args).strip()

    if isinstance(patch_content, bytes):
        p_bytes = patch_content
    elif isinstance(patch_content, str):
        if os.path.isfile(patch_content):
            with open(patch_content, "rb") as f:
                p_bytes = f.read()
        else:
            p_bytes = patch_content.encode("utf-8")
    else:
        p_bytes = str(patch_content).encode("utf-8")

    payload = f"{args_repr}\n".encode("utf-8") + p_bytes
    h8 = hashlib.sha256(payload).hexdigest()[:8]

    return f"tpu_compile_cache__{slug}__vllm{ver}__len{int(max_model_len)}__b{b_str}__{h8}.tar"
