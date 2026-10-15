#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/pythia_embedding_write_scale_test.py --test EmbeddingWriteScaleTest.test_only_embedding_parameters_change --test EmbeddingWriteScaleTest.test_full_budget_and_flags --test EmbeddingWriteScaleTest.test_seed_compensation_and_default_identity --test EmbeddingWriteScaleTest.test_new_forward_gradients_causality --jobs 2
env JAX_PLATFORMS=cpu PYTHONPATH=MaxText OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
