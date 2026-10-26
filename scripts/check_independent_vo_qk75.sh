#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_independent_vo_qk75_test.py \
  --test IndependentVOQK75Test.test_equal_full_parameter_trees_and_runtime \
  --test IndependentVOQK75Test.test_model_native_pallas_all_gradients \
  --test IndependentVOQK75Test.test_kernel_qk93_v75_all_gradients_and_tile_boundaries --jobs 3
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/summary_writer_test.py --jobs 1
