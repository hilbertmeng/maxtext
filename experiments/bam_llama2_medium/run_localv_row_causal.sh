#!/usr/bin/env bash
set -euo pipefail
REPO=${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}
PYTHON=${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}
OUTPUT=${VROW_OUTPUT:-/tmp/xl-localv-row-causal}
GCS=${VROW_GCS:-gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl-localv-row-causal-20260916/results}
mkdir -p "$OUTPUT/maxtext-output/vrow-probe"
gsutil -q cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz /tmp/pile_eval_cohort.npz
cd "$REPO"
# Stage-specific logs and disjoint output prefixes permit sequential safe resume.
STAGE=${VROW_STAGE:-dose}
env HARDWARE=tpu VROW_OUTPUT="$OUTPUT" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON" experiments/bam_llama2_medium/localv_row_causal.py MaxText/configs/base.yml \
  exp_class=LocalVRowCausalProbe run_name=vrow-probe only_eval=True \
  enable_checkpointing=True async_checkpointing=False \
  base_output_directory="$OUTPUT/maxtext-output/" tensorboard_dir="$OUTPUT/tb" \
  > "$OUTPUT/$STAGE.log" 2>&1 &
probe_pid=$!
while kill -0 "$probe_pid" 2>/dev/null; do
  sleep 30
  gsutil -m -q rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS" || true
done
result=0
wait "$probe_pid" || result=$?
gsutil -m -q rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS"
exit "$result"
