#!/usr/bin/env bash
set -euo pipefail
REPO="${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}"
PYTHON="${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}"
OUTPUT="${QKV_OUTPUT:-/tmp/xl-qkr4-basis-16000}"
GCS="${QKV_GCS:-gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl-qkr4-basis-16000/results}"
mkdir -p "$OUTPUT/maxtext-output/qkv-probe"
if gsutil -q stat "$GCS/metadata.json"; then
  gsutil -m rsync -r -x '(^|/)(probe.log|tb/.*|maxtext-output/.*)' "$GCS" "$OUTPUT"
fi
gsutil cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz /tmp/pile_eval_cohort.npz
cd "$REPO"
env HARDWARE=tpu QKV_OUTPUT="$OUTPUT" "$PYTHON" \
  experiments/bam_llama2_medium/xl_qkr4_basis_probe.py MaxText/configs/base.yml \
  exp_class=LocalQKVKeyProbe run_name=qkv-probe only_eval=True \
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
