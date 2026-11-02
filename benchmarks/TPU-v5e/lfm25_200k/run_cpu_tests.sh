#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

export PATH="/home/sahil/.unsloth/studio/unsloth_studio/bin:${PATH}"
export PYTHONPATH="${DIR}:/home/sahil/Desktop/sahil/src-0.32.0/tpu-inference:/home/sahil/.cache/uv/archive-v0/s66A7_ti0x9rZScx/lib/python3.13/site-packages"

python -m pytest -q --collect-only tests/test_int8_kv_cpu.py
python -m pytest -q tests/test_int8_kv_cpu.py
