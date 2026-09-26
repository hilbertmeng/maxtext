#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText
CPU_PYTHON=/data0/xd/conda/envs/maxtext-cpu/bin/python
"$CPU_PYTHON" experiments/bam_llama2_medium/audit_rmt_mha_budget.py \
  --exp-class RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32
exec python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$PWD" \
  --test-file MaxText/tests/rmt_mediumprop_test.py --jobs 3 --cores-per-job 8 \
  --test RMTMediumPropTest.test_combined_boundaries_health_initialization_and_gradients \
  --test RMTMediumPropTest.test_dynamic_embedding_matches_original_bam_write \
  --test RMTMediumPropTest.test_direct32_unembedding_matches_full_tail_read_and_gradients
