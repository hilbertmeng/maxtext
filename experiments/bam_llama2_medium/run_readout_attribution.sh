#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "usage: $0 OUTPUT_TAG SEQUENCES SEQUENCE_OFFSET GCS_PREFIX" >&2
  exit 2
fi

OUTPUT_TAG="$1"
SEQUENCES="$2"
SEQUENCE_OFFSET="$3"
GCS_PREFIX="${4%/}"
REPO="${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}"
PYTHON="${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}"
DATASET="${DATASET_PATH:-gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord}"
CHECKPOINT="${BAM_ATTR_CHECKPOINT:-gs://newproject-1-llm_base_models_us-central1/log/BamLlama2MediumV2/checkpoints/13250/items}"
OUTPUT_DIR="/tmp/$OUTPUT_TAG"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to mix attribution shards in existing $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"

cd "$REPO"
env \
  HARDWARE=tpu \
  JAX_TRACEBACK_FILTERING=off \
  BAM_ATTR_OUTPUT_DIR="$OUTPUT_DIR" \
  BAM_ATTR_SEQUENCES="$SEQUENCES" \
  BAM_ATTR_SEQUENCE_OFFSET="$SEQUENCE_OFFSET" \
  BAM_ATTR_DIAGNOSTIC_COMMIT="$(git rev-parse HEAD)" \
  "$PYTHON" experiments/bam_llama2_medium/readout_attribution.py \
    MaxText/configs/base.yml \
    exp_class=BamLlama2MediumV2ReadoutAttribution \
    "run_name=$OUTPUT_TAG" \
    "dataset_path=$DATASET" \
    "load_parameters_path=$CHECKPOINT" \
    "base_output_directory=$OUTPUT_DIR/maxtext-output" \
    "tensorboard_dir=$OUTPUT_DIR/tensorboard" \
    only_eval=True enable_checkpointing=False async_checkpointing=False

gsutil -m rsync -r "$OUTPUT_DIR" "$GCS_PREFIX"
echo "READOUT_ATTRIBUTION_UPLOADED local=$OUTPUT_DIR gcs=$GCS_PREFIX"
