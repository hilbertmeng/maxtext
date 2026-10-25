#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
TASK_COMMIT=$(git rev-parse HEAD)
TASK_GATE="/data0/xd/bam_diagnostics/mediumprop-hybrid-v-heads/hybrid-separate-vo/cpu-launch/$TASK_COMMIT"
mkdir -p "$TASK_GATE"
exec 9>"$TASK_GATE/lock"
flock 9
if [[ -f "$TASK_GATE/PASS" ]] && [[ $(cat "$TASK_GATE/PASS") == "$TASK_COMMIT" ]]; then
  printf 'SHARED_CPU_GATE_OK commit=%s proof=%s/PASS\n' "$TASK_COMMIT" "$TASK_GATE"
  exit 0
fi
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/bam_independent_vo_test.py \
  --test HybridIndependentVOTest.test_formal_hybrid_trees_and_budgets \
  --test HybridIndependentVOTest.test_hybrid_cloned_keys_masks_and_independence \
  --test HybridIndependentVOTest.test_hybrid_keep_pallas_all_gradients \
  --test HybridIndependentVOTest.test_hybrid_drop_pallas_all_gradients \
  --test IndependentVOTest.test_cloned_initial_key_and_independent_learning_paths --jobs 4
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" \
  --test-file MaxText/tests/summary_writer_test.py --jobs 1
printf '%s\n' "$TASK_COMMIT" > "$TASK_GATE/PASS.tmp"
mv "$TASK_GATE/PASS.tmp" "$TASK_GATE/PASS"
printf 'SHARED_CPU_GATE_OK commit=%s proof=%s/PASS\n' "$TASK_COMMIT" "$TASK_GATE"
