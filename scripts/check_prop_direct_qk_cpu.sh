#!/usr/bin/env bash
set -euo pipefail
TASK_WORKTREE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PROP_DIRECT_QK_SCALE=xl
exec python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_WORKTREE" --test-file MaxText/tests/bam_prop_direct_qk_test.py --jobs 3 --cores-per-job 8
