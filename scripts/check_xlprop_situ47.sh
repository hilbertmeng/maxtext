#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$repo" \
  --test-file MaxText/tests/situ_mlp_test.py \
  --test SiTUMLPTest.test_full_shape_budget_and_formal_contract \
  --test SiTUMLPTest.test_scanned_ordinary_and_terminal_only_private_writer_unchanged \
  --jobs 2 --cores-per-job 8
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$repo" \
  --test-file MaxText/tests/summary_writer_test.py --jobs 1 --cores-per-job 4
