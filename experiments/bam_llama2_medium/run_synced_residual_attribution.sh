#!/usr/bin/env bash
set -euo pipefail

if (( $# < 7 )); then
  echo "usage: $0 CONFIG TRAINER_COMMIT RUN_PREFIX CHECKPOINT_BASE OUTPUT_BASE SEQUENCES STEP..." >&2
  exit 2
fi

CONFIG="$1"
TRAINER_COMMIT="$2"
RUN_PREFIX="$3"
CHECKPOINT_BASE="${4%/}"
OUTPUT_BASE="${5%/}"
SEQUENCES="$6"
shift 6

: "${BAM_RESIDUAL_ATTR_COHORT_GCS:?set BAM_RESIDUAL_ATTR_COHORT_GCS}"
: "${DATASET_PATH:?set DATASET_PATH}"

for STEP in "$@"; do
  CHECKPOINT="$CHECKPOINT_BASE/$STEP/items"
  TAG="${RUN_PREFIX}-s${STEP}"
  echo "$(date -u +%FT%TZ) waiting checkpoint=$CHECKPOINT"
  until gsutil -q stat "$CHECKPOINT/commit_success.txt"; do
    sleep 30
  done
  echo "$(date -u +%FT%TZ) running tag=$TAG"
  (
    flock 9
    env \
      BAM_RESIDUAL_ATTR_BASE_CONFIG="$CONFIG" \
      BAM_RESIDUAL_ATTR_TRAINER_COMMIT="$TRAINER_COMMIT" \
      BAM_RESIDUAL_ATTR_CHECKPOINT="$CHECKPOINT" \
      BAM_RESIDUAL_ATTR_COHORT_GCS="$BAM_RESIDUAL_ATTR_COHORT_GCS" \
      DATASET_PATH="$DATASET_PATH" \
      BAM_RESIDUAL_ATTR_BATCH_SIZE="${BAM_RESIDUAL_ATTR_BATCH_SIZE:-2}" \
      BAM_RESIDUAL_ATTR_IG_NODES="${BAM_RESIDUAL_ATTR_IG_NODES:-10}" \
      bash experiments/bam_llama2_medium/run_residual_attribution.sh \
        "$TAG" "$SEQUENCES" 0 "$OUTPUT_BASE/$TAG"
  ) 9>"${BAM_DIAGNOSTIC_TPU_LOCK:-/tmp/bam-diagnostic-tpu.lock}"
done
