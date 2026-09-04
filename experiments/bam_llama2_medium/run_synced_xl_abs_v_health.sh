#!/usr/bin/env bash
set -euo pipefail

if (( $# < 6 )); then
  echo "usage: $0 CONFIG TRAINER_COMMIT RUN_PREFIX CHECKPOINT_BASE OUTPUT_BASE STEP..." >&2
  exit 2
fi

CONFIG="$1"
TRAINER_COMMIT="$2"
RUN_PREFIX="$3"
CHECKPOINT_BASE="${4%/}"
OUTPUT_BASE="${5%/}"
shift 5

: "${BAM_RESIDUAL_ATTR_COHORT_GCS:?set BAM_RESIDUAL_ATTR_COHORT_GCS}"
: "${DATASET_PATH:?set DATASET_PATH}"

PYTHON="${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}"
CODE_COMMIT="${BAM_ABSV_DIAG_CODE_COMMIT:-$(git rev-parse HEAD)}"
COHORT_PATH="${BAM_ABSV_DIAG_COHORT_PATH:-/tmp/pile-eval-t2048-seed9876-n128-v1.npz}"
gsutil -q stat "$COHORT_PATH" 2>/dev/null || \
  gsutil cp "$BAM_RESIDUAL_ATTR_COHORT_GCS" "$COHORT_PATH"

for STEP in "$@"; do
  CHECKPOINT="$CHECKPOINT_BASE/$STEP/items"
  TAG="${RUN_PREFIX}-s${STEP}"
  OUTPUT="/tmp/$TAG"
  echo "$(date -u +%FT%TZ) waiting checkpoint=$CHECKPOINT"
  until gsutil -q stat "$CHECKPOINT/commit_success.txt"; do
    sleep 30
  done
  mkdir -p "$OUTPUT/output/$TAG" "$OUTPUT/tensorboard"
  echo "$(date -u +%FT%TZ) running tag=$TAG"
  env \
    HARDWARE=tpu \
    JAX_TRACEBACK_FILTERING=off \
    BAM_ABSV_DIAG_BATCHES=8 \
    BAM_ABSV_DIAG_CAPTURE_BATCHES=8 \
    BAM_ABSV_DIAG_SCALES=1 \
    BAM_ABSV_DIAG_RANKS= \
    BAM_ABSV_DIAG_LAYERWISE_RANK=0 \
    BAM_ABSV_DIAG_CODE_COMMIT="$CODE_COMMIT" \
    BAM_ABSV_DIAG_TRAINER_COMMIT="$TRAINER_COMMIT" \
    BAM_ABSV_DIAG_COHORT_PATH="$COHORT_PATH" \
    BAM_ABSV_DIAG_OUTPUT="$OUTPUT/report.json" \
    "$PYTHON" experiments/bam_llama2_medium/xl_abs_v_width_diagnostics.py \
      MaxText/configs/base.yml \
      "exp_class=$CONFIG" \
      "run_name=$TAG" \
      only_eval=True steps=1 \
      "load_parameters_path=$CHECKPOINT" \
      "dataset_path=$DATASET_PATH" \
      "base_output_directory=$OUTPUT/output" \
      "tensorboard_dir=$OUTPUT/tensorboard" \
      per_device_batch_size=1 eval_per_device_batch_size=16 \
      enable_checkpointing=True async_checkpointing=False \
      >"$OUTPUT/run.log" 2>&1
  gsutil cp "$OUTPUT/report.json" "$OUTPUT_BASE/$TAG/report.json"
  gsutil cp "$OUTPUT/run.log" "$OUTPUT_BASE/$TAG/run.log"
  echo "$(date -u +%FT%TZ) done tag=$TAG"
done
