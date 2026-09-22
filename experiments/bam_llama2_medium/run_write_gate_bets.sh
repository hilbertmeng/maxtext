#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BET_OUTPUT=${BET_OUTPUT:-/tmp/alllocal-write-gate-bets-0922}
export BET_PROTOCOL=$BET_OUTPUT/protocol.json
GCS=${BET_GCS:-gs://newproject-1-llm_projects_europe-west4/log/diagnostics/alllocal-write-gate-bets-0922}
mkdir -p "$BET_OUTPUT/maxtext-output/write-bets"
gsutil cp "$GCS/protocol.json" "$BET_PROTOCOL"
if [[ ! -f /tmp/pile_eval_cohort.npz ]]; then
 gsutil cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz /tmp/pile_eval_cohort.npz
fi
env HARDWARE=tpu /home/lishengping/miniconda3/bin/python "${BET_RUNNER:-experiments/bam_llama2_medium/write_gate_bets.py}" MaxText/configs/base.yml exp_class=GeometryProbe run_name=write-bets only_eval=True enable_checkpointing=True async_checkpointing=False base_output_directory="$BET_OUTPUT/maxtext-output/" tensorboard_dir="$BET_OUTPUT/tb" "$@" > "$BET_OUTPUT/probe.log" 2>&1 &
job=$!
while kill -0 "$job" 2>/dev/null; do
 sleep 30
 gsutil -m rsync -r -x '(^|/)(tb/.*|maxtext-output/.*|.*launcher\.log|.*\.tmp)' "$BET_OUTPUT" "$GCS/worker" || true
done
result=0
wait "$job" || result=$?
gsutil -m rsync -r -x '(^|/)(tb/.*|maxtext-output/.*|.*launcher\.log|.*\.tmp)' "$BET_OUTPUT" "$GCS/worker"
exit "$result"
