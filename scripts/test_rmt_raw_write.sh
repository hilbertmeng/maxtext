#!/usr/bin/env bash
set -euo pipefail
cd /data0/xd/rmt-xlprop-noo
exec env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/rmt_raw_write_test.py
