#!/usr/bin/env bash
set -euo pipefail
REPO="${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}"
PYTHON="${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}"
OUTPUT="${OV_OUTPUT:-/tmp/local-ov-gate-final128}"
GCS="${OV_GCS:-gs://newproject-1-llm_base_models_us-central1/log/diagnostics/local-ov-gate-final128}"
mkdir -p "$OUTPUT/maxtext-output/ov-probe"
if gsutil -q stat "$GCS/metadata.json"; then
  gsutil -m rsync -r -x '(^|/)(probe.log|tb/.*|maxtext-output/.*)' "$GCS" "$OUTPUT"
fi
gsutil cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz /tmp/pile_eval_cohort.npz
cd "$REPO"
env HARDWARE=tpu OV_OUTPUT="$OUTPUT" "$PYTHON" \
  experiments/bam_llama2_medium/local_ov_gate_probe.py MaxText/configs/base.yml \
  exp_class=LocalOVGateProbe run_name=ov-probe only_eval=True \
  enable_checkpointing=True async_checkpointing=False \
  base_output_directory="$OUTPUT/maxtext-output/" tensorboard_dir="$OUTPUT/tb" \
  > "$OUTPUT/probe.log" 2>&1 &
probe_pid=$!
# Upload only committed batch files; the hub never handles diagnostic bytes.
while kill -0 "$probe_pid" 2>/dev/null; do
  sleep 30
  gsutil -m rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS" || true
done
wait "$probe_pid"
"$PYTHON" experiments/bam_llama2_medium/analyze_local_ov_gates.py "$OUTPUT"
gsutil -m rsync -r -x '(^|/)(\.pending_.*|tb/.*|maxtext-output/.*)' "$OUTPUT" "$GCS"
echo "OV_DIAG_UPLOADED $GCS"
