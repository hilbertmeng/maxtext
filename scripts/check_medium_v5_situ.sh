#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONPATH=MaxText:MaxText/tests
PYTHON=/data0/xd/conda/envs/maxtext-cpu/bin/python
"$PYTHON" MaxText/tests/situ_mlp_test.py
"$PYTHON" MaxText/tests/summary_writer_test.py
