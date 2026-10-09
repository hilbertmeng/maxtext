#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$ROOT" --test-file MaxText/tests/bam_prop_separate_qkv_test.py --jobs 2
