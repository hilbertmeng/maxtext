#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
PYTHON=/data0/xd/conda/envs/maxtext-cpu/bin/python
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH=MaxText:MaxText/tests
"$PYTHON" MaxText/tests/rmt_truepile_noo_llf_test.py
"$PYTHON" -m unittest \
  rmt_mediumprop_test.RMTMergedRuntimeTest.test_no_o_removes_only_o_gates_and_health \
  rmt_depth_test.RMTDepthTest.test_block_scan_matches_direct_scan_with_mapped_parameters
