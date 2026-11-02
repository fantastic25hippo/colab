#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

VENV="/home/sahil/Desktop/sahil/colab/.venv-cpu-tests"
PYTHON="${VENV}/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: venv python not found: ${PYTHON} (VENV=${VENV})" >&2
  exit 1
fi

export PYTHONPATH="${DIR}:/home/sahil/Desktop/sahil/src-0.32.0/tpu-inference"

"${PYTHON}" -m pytest -q --collect-only tests/test_int8_kv_cpu.py
"${PYTHON}" -m pytest -q tests/test_int8_kv_cpu.py
