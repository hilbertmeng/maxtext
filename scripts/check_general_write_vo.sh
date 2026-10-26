#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_general_write_vo_test.py --jobs 3
# Batch-sharded write kernel (replicated static addresses): needs 8 host devices.
(cd "$TASK_ROOT" && JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
  PYTHONPATH=MaxText:MaxText/tests python3 -m unittest \
  bam_general_write_vo_test.GeneralWriteVOTest.test_v8t_write_kernel_batch_sharded_static_gradients)
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/summary_writer_test.py --jobs 1
