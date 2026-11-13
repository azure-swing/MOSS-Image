#!/usr/bin/env bash
set -euo pipefail

# Setup script for MOSS-Image on Google Cloud TPU (Colab TPU v6e-1 / TPU VM) with TorchTPU.
# Requires Python 3.11+ per upstream torch_tpu requirements.

PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 11), "torch_tpu requires Python 3.11+"'

"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install torch_tpu --extra-index-url https://download.pytorch.org/whl/cpu
"${PYTHON_BIN}" -m pip install -r requirements.txt

cat <<'EOF'
TorchTPU environment setup complete.

Recommended persistent XLA compilation cache environment variables:
  export TORCH_TPU_TIER2_COMPILATION_CACHE="${HOME}/.cache/torch_tpu/tier2"
  export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${HOME}/.cache/torch_tpu/tier3"

Quick probe:
  python3 scripts/run_tpu.py --mode probe
EOF
