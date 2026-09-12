#!/usr/bin/env bash
set -euo pipefail
REPO=${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}
PYTHON=${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}
OUTPUT=${ORANK_OUTPUT:-/tmp/local-o-row-rank}
GCS=${ORANK_GCS:-gs://newproject-1-llm_projects_europe-west4/log/diagnostics/llf-o-row-rank-13500}
STAGE=${ORANK_STAGE:-spectra}
case "$STAGE" in
  spectra) SCRIPT=local_o_row_rank_probe.py ;;
  ablation) SCRIPT=local_o_row_rank_ablation.py ;;
  *) echo "Unknown ORANK_STAGE=$STAGE" >&2; exit 2 ;;
esac
mkdir -p "$OUTPUT/maxtext-output/orank-probe"
gsutil cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz /tmp/pile_eval_cohort.npz
cd "$REPO"
env HARDWARE=tpu ORANK_OUTPUT="$OUTPUT" OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "$PYTHON" "experiments/bam_llama2_medium/$SCRIPT" MaxText/configs/base.yml \
  exp_class=LocalORowRankProbe run_name=orank-probe only_eval=True \
  enable_checkpointing=True async_checkpointing=False \
  base_output_directory="$OUTPUT/maxtext-output/" tensorboard_dir="$OUTPUT/tb" \
  > "$OUTPUT/${STAGE}.log" 2>&1 &
probe_pid=$!
while kill -0 "$probe_pid" 2>/dev/null; do
  sleep 30
  gsutil -m rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS" || true
done
result=0
wait "$probe_pid" || result=$?
gsutil -m rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS"
exit "$result"
