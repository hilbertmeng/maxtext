#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$ROOT" \
  --test-file MaxText/tests/bam_xlprop_v5_test.py --jobs 3 --cores-per-job 8
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$ROOT" \
  --test-file MaxText/tests/summary_writer_test.py --jobs 1 --cores-per-job 8
