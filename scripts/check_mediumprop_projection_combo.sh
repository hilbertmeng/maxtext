#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
git diff --quiet
TASK_HASH=$(git rev-parse HEAD)
TASK_CHECK=/data0/xd/bam_diagnostics/mediumprop-projection-sharing/combo-cpu/$TASK_HASH
mkdir -p "$TASK_CHECK"
exec 9>"$TASK_CHECK/check.lock"
flock 9
if [[ -f "$TASK_CHECK/ok" ]]; then exit 0; fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$TASK_ROOT/MaxText"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=8
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_projection_sharing_test.py
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
touch "$TASK_CHECK/ok"
