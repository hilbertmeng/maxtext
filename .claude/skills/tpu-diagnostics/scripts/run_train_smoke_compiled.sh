#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 EXP RUN COMPILED [STEPS]" >&2
  exit 2
fi

EXP=$1
RUN=$2
COMPILED=$3
STEPS=${4:-15}
REPO=${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}
PYTHON=${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}
DATASET=${DATASET_PATH:-gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord}
OUTPUT=${SMOKE_OUTPUT:-gs://newproject-1-llm_base_models_us-central1/log/diagnostics/smoke}
PROFILE_SKIP=${PROFILE_SKIP:-}
PROFILE_PERIOD=${PROFILE_PERIOD:-}
PROFILE_DURATION=${PROFILE_DURATION:-}

profile_args=()
if [[ -n "$PROFILE_SKIP" ]]; then
  profile_args+=("skip_first_n_steps_for_profiler=$PROFILE_SKIP")
fi
if [[ -n "$PROFILE_PERIOD" ]]; then
  profile_args+=("profile_periodically_period=$PROFILE_PERIOD")
fi
if [[ -n "$PROFILE_DURATION" ]]; then
  profile_args+=("profiler_steps=$PROFILE_DURATION")
fi

if [[ "$OUTPUT" != gs://* ]]; then
  mkdir -p "$OUTPUT/$RUN" "$OUTPUT/tensorboard"
fi

cd "$REPO"
exec env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off "$PYTHON" \
  MaxText/train.py MaxText/configs/base.yml \
  "exp_class=$EXP" "run_name=$RUN" "steps=$STEPS" \
  "compiled_trainstep_file=$COMPILED" \
  "dataset_path=$DATASET" "base_output_directory=$OUTPUT" \
  "tensorboard_dir=$OUTPUT/tensorboard" \
  enable_checkpointing=False async_checkpointing=False \
  upload_all_profiler_results=False "${profile_args[@]}"
