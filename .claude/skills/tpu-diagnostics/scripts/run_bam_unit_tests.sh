#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-/home/xd/projects/maxtext}"
PYTHON="${MAXTEXT_CPU_PYTHON:-/data0/xd/conda/envs/maxtext-cpu/bin/python}"

env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 "$PYTHON" -c \
  'from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel'
cd "$REPO"
exec env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText \
  "$PYTHON" MaxText/tests/bam_attention_test.py
