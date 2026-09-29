#!/usr/bin/env bash
set -euo pipefail
cd /data0/xd/rmt-xlprop-noo
REV=$(git rev-parse HEAD)
ROOT=/data0/xd/bam_diagnostics/rmt-readnorm-launch/cpu
mkdir -p "$ROOT"
exec 9>"$ROOT/$REV.lock"
flock 9
if [[ -f "$ROOT/$REV.ok" ]]; then cat "$ROOT/$REV.ok"; exit 0; fi
env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText OMP_NUM_THREADS=8 \
 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/rmt_read_norm_test.py
printf 'CPU_PASS %s\n' "$REV" > "$ROOT/$REV.ok"
