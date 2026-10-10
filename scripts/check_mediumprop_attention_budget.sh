#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
COMMIT=$(git rev-parse HEAD)
STATE=/data0/xd/bam_diagnostics/mediumprop-attention-budget-checks/$COMMIT
mkdir -p "$STATE"
exec 9>"$STATE/check.lock"
flock 9
if [[ -f "$STATE/ok" ]]; then
  echo "CPU_GATE_REUSED commit=$COMMIT"
  exit 0
fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$ROOT/MaxText"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/data0/xd/conda/envs/maxtext-cpu/bin/python scripts/check_mediumprop_attention_budget.py
bash /home/xd/projects/maxtext/.agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh "$ROOT"
touch "$STATE/ok"
echo "CPU_GATE_OK commit=$COMMIT"
