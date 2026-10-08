#!/usr/bin/env bash
set -euo pipefail
TASK_WORKTREE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_WORKTREE" --test-file MaxText/tests/bam_attention_write_content_bias_test.py --jobs 2 --cores-per-job 8
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_WORKTREE" --test-file MaxText/tests/bam_mlp_write_test.py --test MLPWriteTest.test_original_write_and_combined_equivalence --jobs 1 --cores-per-job 8
