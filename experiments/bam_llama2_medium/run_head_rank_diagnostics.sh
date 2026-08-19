#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "usage: $0 bam|mha OUTPUT_TAG SEQUENCES GCS_PREFIX" >&2
  exit 2
fi

MODEL_KIND="$1"
OUTPUT_TAG="$2"
SEQUENCES="$3"
GCS_PREFIX="${4%/}"
REPO="${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}"
PYTHON="${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}"
DATASET="${DATASET_PATH:-gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord}"
OUTPUT_DIR="/tmp/$OUTPUT_TAG"

case "$MODEL_KIND" in
  bam)
    EXP_CLASS=BamLlama2MediumV2HeadRankDiagnostics
    CHECKPOINT="${BAM_HEAD_RANK_CHECKPOINT:-gs://newproject-1-llm_base_models_us-central1/log/BamLlama2MediumV2/checkpoints/13250/items}"
    ;;
  mha)
    EXP_CLASS=Llama2MediumHeadRankDiagnostics
    CHECKPOINT="${BAM_HEAD_RANK_CHECKPOINT:-gs://newproject-1-llm_base_models_us-central1/log/Llama2Medium/checkpoints/13500/items}"
    ;;
  *)
    echo "MODEL_KIND must be bam or mha, got $MODEL_KIND" >&2
    exit 2
    ;;
esac

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to mix head-rank artifacts in existing $OUTPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR/maxtext-output/$OUTPUT_TAG"

cd "$REPO"
env \
  HARDWARE=tpu \
  JAX_TRACEBACK_FILTERING=off \
  BAM_HEAD_RANK_MODEL="$MODEL_KIND" \
  BAM_HEAD_RANK_OUTPUT_DIR="$OUTPUT_DIR" \
  BAM_HEAD_RANK_SEQUENCES="$SEQUENCES" \
  BAM_HEAD_RANK_COMMIT="$(git rev-parse HEAD)" \
  "$PYTHON" experiments/bam_llama2_medium/head_rank_diagnostics.py \
    MaxText/configs/base.yml \
    "exp_class=$EXP_CLASS" \
    "run_name=$OUTPUT_TAG" \
    "dataset_path=$DATASET" \
    "load_parameters_path=$CHECKPOINT" \
    "base_output_directory=$OUTPUT_DIR/maxtext-output" \
    "tensorboard_dir=$OUTPUT_DIR/tensorboard" \
    only_eval=True enable_checkpointing=True async_checkpointing=False &
DIAGNOSTIC_PID=$!

while kill -0 "$DIAGNOSTIC_PID" 2>/dev/null; do
  sleep 120
  gsutil -m rsync -r "$OUTPUT_DIR" "$GCS_PREFIX" || true
done
wait "$DIAGNOSTIC_PID"

gsutil -m rsync -r "$OUTPUT_DIR" "$GCS_PREFIX"
echo "HEAD_RANK_UPLOADED model=$MODEL_KIND local=$OUTPUT_DIR gcs=$GCS_PREFIX"
