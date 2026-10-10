#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TASK_COMMIT=$(git rev-parse HEAD)
TASK_STATE=/data0/xd/bam_diagnostics/mediumprop-direct-address-checks/$TASK_COMMIT
mkdir -p "$TASK_STATE"
exec 9>"$TASK_STATE/check.lock"
flock 9
if [[ -f "$TASK_STATE/ok" ]]; then
  echo "CPU_GATE_REUSED commit=$TASK_COMMIT"
  exit 0
fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$ROOT/MaxText"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/data0/xd/conda/envs/maxtext-cpu/bin/python scripts/check_mediumprop_direct_address.py
touch "$TASK_STATE/ok"
echo "CPU_GATE_OK commit=$TASK_COMMIT"
