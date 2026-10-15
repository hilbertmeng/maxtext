#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$TASK_ROOT/MaxText:$TASK_ROOT/MaxText/tests"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=8
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_mediumprop_t2048_sota_test.py T2048SotaTest.test_transfer_budget_and_runtime_settings
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_projection_sharing_test.py ProjectionSharingTest.test_attention_computes_down_once_and_preserves_raw_carry ProjectionSharingTest.test_direct_and_deferred_write_and_shared_kernel_gradient ProjectionSharingTest.test_shift_boundary_and_initial_opening
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
