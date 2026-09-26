#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText
CPU_PYTHON=/data0/xd/conda/envs/maxtext-cpu/bin/python
"$CPU_PYTHON" experiments/bam_llama2_medium/audit_rmt_mha_budget.py
exec "$CPU_PYTHON" MaxText/tests/rmt_mediumprop_test.py \
  RMTMediumPropTest.test_dynamic_embedding_matches_original_bam_write \
  RMTMediumPropTest.test_dynamic_boundaries_health_initialization_and_gradients \
  RMTMediumPropTest.test_mha_budget_block_scan_health
