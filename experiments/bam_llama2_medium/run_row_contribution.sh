#!/usr/bin/env bash
set -euo pipefail
REPO=${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}
PYTHON=${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}
OUTPUT=${ROW_OUTPUT:-/tmp/llf-row-contribution}
GCS=${ROW_GCS:-gs://newproject-1-llm_projects_europe-west4/log/diagnostics/llf-row-contribution-13500-0914}
mkdir -p "$OUTPUT/maxtext-output/row-contribution"
gsutil cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz /tmp/pile_eval_cohort.npz
cd "$REPO"
env HARDWARE=tpu ROW_OUTPUT="$OUTPUT" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 "$PYTHON" experiments/bam_llama2_medium/row_contribution.py MaxText/configs/base.yml \
 exp_class=RowContributionProbe run_name=row-contribution only_eval=True \
 enable_checkpointing=True async_checkpointing=False \
 base_output_directory="$OUTPUT/maxtext-output/" tensorboard_dir="$OUTPUT/tb" \
 > "$OUTPUT/${ROW_STAGE:-all}_${ROW_START:-0}_${ROW_STOP:-32}.log" 2>&1 &
probe_pid=$!
while kill -0 "$probe_pid" 2>/dev/null; do
 sleep 30
 gsutil -m rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS" || true
done
result=0
wait "$probe_pid" || result=$?
gsutil -m rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS"
exit "$result"
