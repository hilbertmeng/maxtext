#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 \
  PYTHONPATH=MaxText /data0/xd/conda/envs/maxtext-cpu/bin/python \
  MaxText/tests/rmt_mediumprop_test.py
