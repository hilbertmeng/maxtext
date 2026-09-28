#!/usr/bin/env bash
set -euo pipefail

REPO=/data0/xd/rmt-static-k64-truepile
PYTHON=/data0/xd/conda/envs/maxtext-cpu/bin/python
cd "$REPO"
exec env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText \
  "$PYTHON" MaxText/tests/rmt_mediumprop_test.py
