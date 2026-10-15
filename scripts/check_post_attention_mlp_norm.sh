#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$TASK_ROOT/MaxText:$TASK_ROOT/MaxText/tests"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=8
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/bam_post_attention_mlp_norm_test.py --jobs 3
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/bam_shared_write_activation_test.py --jobs 3
bash scripts/check_post_residual_write.sh
