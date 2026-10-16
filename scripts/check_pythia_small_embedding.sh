#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
 --test-file MaxText/tests/pythia_small_embedding_test.py \
 --test SmallEmbeddingTest.test_only_embedding_changes_and_full_budgets \
 --test SmallEmbeddingTest.test_health_forward_gradient_parity_and_actual_updates \
 --test SmallEmbeddingTest.test_writer_uses_buffered_step_for_health_cadence --jobs 2
env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=MaxText OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
 taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
