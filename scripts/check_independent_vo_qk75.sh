#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_independent_vo_qk75_test.py --jobs 4
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/summary_writer_test.py --jobs 1
