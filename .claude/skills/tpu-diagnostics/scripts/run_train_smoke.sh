#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "usage: $0 EXP RUN [STEPS]" >&2
  exit 2
fi

EXP="$1"
RUN="$2"
STEPS="${3:-15}"
REPO="${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}"
PYTHON="${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}"
DATASET="${DATASET_PATH:-gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord}"
OUTPUT="${SMOKE_OUTPUT:-gs://newproject-1-llm_base_models_us-central1/log/diagnostics/smoke}"

cd "$REPO"
exec env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off "$PYTHON" \
  MaxText/train.py MaxText/configs/base.yml \
  "exp_class=$EXP" "run_name=$RUN" "steps=$STEPS" \
  "dataset_path=$DATASET" "base_output_directory=$OUTPUT" \
  "tensorboard_dir=$OUTPUT/tensorboard" \
  enable_checkpointing=False async_checkpointing=False \
  upload_all_profiler_results=False
