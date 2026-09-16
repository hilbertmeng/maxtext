#!/usr/bin/env bash
# One worker, sequential TPU stages, atomically resumable per-sequence artifacts.
set -euo pipefail
cd "${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}"
export VROW_STOP=${VROW_STOP:-64}
for stage in dose qk route focus; do
  VROW_STAGE=$stage bash experiments/bam_llama2_medium/run_localv_row_causal.sh
done
