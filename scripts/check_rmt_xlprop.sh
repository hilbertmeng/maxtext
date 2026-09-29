#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText:MaxText/tests OMP_NUM_THREADS=8
exec /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest rmt_xlprop_test rmt_mediumprop_test.RMTMergedRuntimeTest.test_no_o_removes_only_o_gates_and_health
