#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
git diff --quiet
TASK_HASH=$(git rev-parse HEAD)
TASK_CHECK=/data0/xd/bam_diagnostics/cross-layer-race-20261014/cpu/$TASK_HASH
mkdir -p "$TASK_CHECK"
exec 9>"$TASK_CHECK/check.lock"
flock 9
if [[ -f "$TASK_CHECK/ok" ]]; then exit 0; fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$TASK_ROOT/MaxText"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=8
# All17 formal parameter trees must satisfy the unchanged eight-chip sharding bound.
taskset -c 40-47 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_cross_layer_sharing_test.py CrossLayerSharingTest.test_formal_budgets
# The repaired path includes all conservative configurations and the new square bridge.
CROSS_LAYER_REPAIR_ONLY=1 taskset -c 40-47 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_cross_layer_sharing_test.py CrossLayerSharingTest.test_scan_gradients_and_causality
# Previous source's full47 scan/attention regressions and all8 MLP toggle tests passed.
test -f /data0/xd/bam_diagnostics/cross-layer-race-20261014/cpu/f8755258389ebff63d1aa7a23a79b700a34b1adf/ok
test -f /data0/xd/bam_diagnostics/cross-layer-race-20261014/cpu/c7a93f6a6c4d099c6537f40d8f0f9872399bc1ee/ok
touch "$TASK_CHECK/ok"
echo "CROSS_LAYER_SHARDING_REPAIR_OK $TASK_HASH"
