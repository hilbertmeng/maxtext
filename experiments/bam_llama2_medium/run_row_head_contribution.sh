#!/usr/bin/env bash
set -euo pipefail
REPO=${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}
PYTHON=${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}
OUTPUT=${HEAD_OUTPUT:?}
GCS=${HEAD_GCS:?}
REFERENCE_GCS=${HEAD_REFERENCE_GCS:?}
mkdir -p "$OUTPUT/maxtext-output/row-head-contribution"
gsutil cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz /tmp/pile_eval_cohort.npz
gsutil cp "$REFERENCE_GCS" "$OUTPUT/reference.npz"
if [[ -n ${HEAD_SCENARIOS_GCS:-} ]]; then
 gsutil cp "$HEAD_SCENARIOS_GCS" "$OUTPUT/input_scenarios.json"
 export HEAD_SCENARIOS="$OUTPUT/input_scenarios.json"
fi
cd "$REPO"
env HARDWARE=tpu HEAD_REFERENCE="$OUTPUT/reference.npz" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 "$PYTHON" experiments/bam_llama2_medium/row_head_contribution.py MaxText/configs/base.yml \
 exp_class=RowContributionProbe run_name=row-head-contribution only_eval=True \
 enable_checkpointing=True async_checkpointing=False \
 base_output_directory="$OUTPUT/maxtext-output/" tensorboard_dir="$OUTPUT/tb" \
 > "$OUTPUT/probe.log" 2>&1 &
probe_pid=$!
while kill -0 "$probe_pid" 2>/dev/null; do
 sleep 30
 gsutil -m rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS" || true
done
result=0
wait "$probe_pid" || result=$?
gsutil -m rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS"
exit "$result"
