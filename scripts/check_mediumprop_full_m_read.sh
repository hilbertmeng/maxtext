#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
TASK_COMMIT=$(git rev-parse HEAD)
TASK_STATE=/data0/xd/bam_diagnostics/mediumprop-full-m-read-checks/$TASK_COMMIT
mkdir -p "$TASK_STATE"
exec 9>"$TASK_STATE/check.lock"
flock 9
if [[ -f "$TASK_STATE/ok" ]]; then
  echo "CPU_GATE_REUSED commit=$TASK_COMMIT"
  exit 0
fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$ROOT/MaxText"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
env XLA_FLAGS=--xla_force_host_platform_device_count=8 /data0/xd/conda/envs/maxtext-cpu/bin/python scripts/check_mediumprop_full_m_sharding.py
/data0/xd/conda/envs/maxtext-cpu/bin/python scripts/check_mediumprop_full_m_read.py
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$ROOT" --jobs 2 --test BamReadKeyTransformTest.test_bam_read_implementations_match_values_and_gradients --test BamReadKeyTransformTest.test_combined_shared_read_matches_separate_value_and_gradients
touch "$TASK_STATE/ok"
echo "CPU_GATE_OK commit=$TASK_COMMIT"
