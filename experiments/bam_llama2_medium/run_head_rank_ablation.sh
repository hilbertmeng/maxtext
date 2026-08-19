#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  echo "usage: $0 OUTPUT_TAG BASIS_GCS_URI OUTPUT_GCS_PREFIX" >&2
  exit 2
fi

OUTPUT_TAG="$1"
BASIS_GCS_URI="$2"
OUTPUT_GCS_PREFIX="${3%/}"
REPO="${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}"
PYTHON="${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}"
DATASET="${DATASET_PATH:-gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord}"
CHECKPOINT="${BAM_HEAD_RANK_CHECKPOINT:-gs://newproject-1-llm_base_models_us-central1/log/BamLlama2MediumV2/checkpoints/13250/items}"
OUTPUT_DIR="/tmp/$OUTPUT_TAG"
BASIS_PATH="/tmp/${OUTPUT_TAG}_bases.npz"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to mix ablation artifacts in existing $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR/maxtext-output/$OUTPUT_TAG"
gsutil cp "$BASIS_GCS_URI" "$BASIS_PATH"

cd "$REPO"
env \
  HARDWARE=tpu \
  JAX_TRACEBACK_FILTERING=off \
  BAM_HEAD_RANK_ABLATION_OUTPUT_DIR="$OUTPUT_DIR" \
  BAM_HEAD_RANK_BASIS_PATH="$BASIS_PATH" \
  BAM_HEAD_RANK_COMMIT="$(git rev-parse HEAD)" \
  "$PYTHON" experiments/bam_llama2_medium/head_rank_ablation.py \
    MaxText/configs/base.yml \
    exp_class=BamLlama2MediumV2HeadRankAblation \
    "run_name=$OUTPUT_TAG" \
    "dataset_path=$DATASET" \
    "load_parameters_path=$CHECKPOINT" \
    "base_output_directory=$OUTPUT_DIR/maxtext-output" \
    "tensorboard_dir=$OUTPUT_DIR/tensorboard" \
    only_eval=True enable_checkpointing=True async_checkpointing=False

gsutil -m rsync -r "$OUTPUT_DIR" "$OUTPUT_GCS_PREFIX"
echo "HEAD_RANK_ABLATION_UPLOADED local=$OUTPUT_DIR gcs=$OUTPUT_GCS_PREFIX"
