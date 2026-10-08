#!/usr/bin/env bash
set -euo pipefail
TASK_WORKTREE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_WORKTREE" --test-file MaxText/tests/bam_mlp_write_projection_test.py --jobs 2 --cores-per-job 8
