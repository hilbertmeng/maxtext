#!/usr/bin/env bash
# Both head-hybrid arms use the same sealed source and shared-path regression gate.
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
TASK_COMMIT=$(git rev-parse HEAD)
TASK_GATE="/data0/xd/bam_diagnostics/mediumprop-hybrid-v-heads/cpu-launch/$TASK_COMMIT"
mkdir -p "$TASK_GATE"
exec 9>"$TASK_GATE/lock"
flock 9
if [[ -f "$TASK_GATE/PASS" ]] && [[ $(cat "$TASK_GATE/PASS") == "$TASK_COMMIT" ]]; then
  printf 'SHARED_CPU_GATE_OK commit=%s proof=%s/PASS\n' "$TASK_COMMIT" "$TASK_GATE"
  exit 0
fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$TASK_ROOT/MaxText:$TASK_ROOT/MaxText/tests"
bash .agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh "$TASK_ROOT"
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_standard_value_test.py --jobs 4
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_hybrid_value_test.py --jobs 4
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/summary_writer_test.py --jobs 1
printf '%s\n' "$TASK_COMMIT" > "$TASK_GATE/PASS.tmp"
mv "$TASK_GATE/PASS.tmp" "$TASK_GATE/PASS"
printf 'SHARED_CPU_GATE_OK commit=%s proof=%s/PASS\n' "$TASK_COMMIT" "$TASK_GATE"
