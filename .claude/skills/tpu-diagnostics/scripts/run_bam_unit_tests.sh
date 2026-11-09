#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/home/xd/projects/maxtext}"
PYTHON="${MAXTEXT_CPU_PYTHON:-/data0/xd/conda/envs/maxtext-cpu/bin/python}"
TEST_FILE="${MAXTEXT_BAM_TEST_FILE:-MaxText/tests/bam_v5_test.py}"

env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 "$PYTHON" -c \
  'from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel'
cd "$REPO"
if [[ "${MAXTEXT_CPU_TEST_JOBS:-4}" != 1 ]]; then
  exec python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$REPO" \
    --test-file "$TEST_FILE" \
    --python "$PYTHON" --jobs "${MAXTEXT_CPU_TEST_JOBS:-4}" \
    --cores-per-job "${MAXTEXT_CPU_TEST_CORES:-8}"
fi
exec env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText \
  "$PYTHON" "$TEST_FILE"
