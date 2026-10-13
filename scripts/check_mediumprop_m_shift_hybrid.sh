#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
git diff --quiet
TASK_COMMIT=$(git rev-parse HEAD)
TASK_STATE=/data0/xd/bam_diagnostics/mediumprop-m-shift/cpu-hybrid/$TASK_COMMIT
mkdir -p "$TASK_STATE"
exec 9>"$TASK_STATE/check.lock"
flock 9
if [[ -f "$TASK_STATE/ok" ]]; then
  echo "CPU_HYBRID_GATE_REUSED commit=$TASK_COMMIT"
  exit 0
fi
trap 'TASK_EXIT=$?; if (( TASK_EXIT != 0 )); then echo "exit=$TASK_EXIT" > "$TASK_STATE/failed"; fi' EXIT
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$TASK_ROOT/MaxText"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export XLA_FLAGS=--xla_force_host_platform_device_count=8
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_m_shift_test.py
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
touch "$TASK_STATE/ok"
echo "CPU_HYBRID_GATE_OK commit=$TASK_COMMIT (legacy shifts, hybrid scan/grad/causality/health/budget, writer)"
