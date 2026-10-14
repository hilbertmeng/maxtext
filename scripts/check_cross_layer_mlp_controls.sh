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
if [[ -f "$TASK_CHECK/ok" ]]; then
  echo "CROSS_LAYER_CONTROLS_REUSED $TASK_HASH"
  exit 0
fi
# Configuration-only extension. Model files are identical to the fully tested parent.
git diff --exit-code f8755258 HEAD -- MaxText/layers MaxText/train.py MaxText/train_compile.py MaxText/optimizers.py
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$TASK_ROOT/MaxText"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=8
export CROSS_LAYER_NEW_CONTROLS_ONLY=1
taskset -c 32-39 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_cross_layer_sharing_test.py
# Full scan/attention regression and writer checks must also pass on original source.
TASK_PARENT=$(git rev-parse f8755258)
while [[ ! -f /data0/xd/bam_diagnostics/cross-layer-race-20261014/cpu/$TASK_PARENT/ok ]]; do
  sleep 5
done
touch "$TASK_CHECK/ok"
echo "CROSS_LAYER_CONTROLS_OK $TASK_HASH"
