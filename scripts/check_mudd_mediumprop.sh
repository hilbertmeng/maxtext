#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
exec /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/mudd_mediumprop_test.py
