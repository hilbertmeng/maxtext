#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
TASK_COMMIT=$(git rev-parse HEAD)
TASK_GATE="/data0/xd/bam_diagnostics/mediumprop-hybrid-v-heads/separate-vo/cpu-launch/$TASK_COMMIT"
mkdir -p "$TASK_GATE"
exec 9>"$TASK_GATE/lock"
flock 9
if [[ -f "$TASK_GATE/PASS" ]] && [[ $(cat "$TASK_GATE/PASS") == "$TASK_COMMIT" ]]; then
  printf 'EXTRA_CPU_GATE_OK commit=%s proof=%s/PASS\n' "$TASK_COMMIT" "$TASK_GATE"
  exit 0
fi
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_independent_vo_test.py --jobs 4
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_hybrid_value_test.py \
  --test HybridValueTest.test_pallas_keep_all_gradients \
  --test HybridValueTest.test_pallas_drop_all_gradients --jobs 2
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_pallas_mshift_test.py \
  --test ShiftedReadTest.test_forward_and_gradients --jobs 1
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/summary_writer_test.py --jobs 1
printf '%s\n' "$TASK_COMMIT" > "$TASK_GATE/PASS.tmp"
mv "$TASK_GATE/PASS.tmp" "$TASK_GATE/PASS"
printf 'EXTRA_CPU_GATE_OK commit=%s proof=%s/PASS\n' "$TASK_COMMIT" "$TASK_GATE"
