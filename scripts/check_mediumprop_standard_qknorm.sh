#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_standard_qknorm_test.py --jobs 3 --cores-per-job 8
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_full_qknorm_test.py --jobs 2 --cores-per-job 8 \
  --test FullQKNormTest.test_complete_head_before_rope \
  --test FullQKNormTest.test_learned_scales_precede_rope_and_zero_safe
exec python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/summary_writer_test.py --jobs 2 --cores-per-job 8
