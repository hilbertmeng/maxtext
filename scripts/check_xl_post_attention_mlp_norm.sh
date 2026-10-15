#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$TASK_ROOT/MaxText:$TASK_ROOT/MaxText/tests"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=8
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/bam_xl_post_attention_mlp_norm_test.py --test XLPostAttentionMlpNormTest.test_xl_parent_contract_and_full_parameter_budget --jobs 1
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/bam_post_attention_mlp_norm_test.py --test PostAttentionMlpNormTest.test_once_per_norm_actual_controls_and_identical_initialization --test PostAttentionMlpNormTest.test_scanned_gradients_and_causality --jobs 2
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
