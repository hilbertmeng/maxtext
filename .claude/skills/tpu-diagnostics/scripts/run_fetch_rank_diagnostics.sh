#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 EXP CHECKPOINT TAG BATCHES" >&2
  exit 2
fi

EXP=$1
CHECKPOINT=$2
TAG=$3
BATCHES=$4
ROOT=/home/lishengping/xd/projects/maxtext
PYTHON=/home/lishengping/miniconda3/bin/python
LOG=/tmp/fetch_rank_${TAG}.log
EXIT_FILE=/tmp/fetch_rank_${TAG}.exit

export JAX_TRACEBACK_FILTERING=off
export BAM_FETCH_RANK_BATCHES="$BATCHES"
export BAM_FETCH_RANK_OUTPUT="/tmp/fetch_rank_${TAG}.json"
rm -f "$EXIT_FILE"
mkdir -p "/tmp/fetch_rank_${TAG}_output/fetch_rank_${TAG}"

cd "$ROOT" || exit 2
"$PYTHON" MaxText/bam_fetch_rank_diagnostics.py MaxText/configs/base.yml \
  base_output_directory="/tmp/fetch_rank_${TAG}_output" \
  run_name="fetch_rank_${TAG}" \
  exp_class="$EXP" \
  dataset_path=gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord \
  load_parameters_path="$CHECKPOINT" \
  only_eval=True \
  enable_checkpointing=True \
  async_checkpointing=False \
  per_device_batch_size=8 \
  eval_per_device_batch_size=8 \
  eval_shuffle_buffer_size=32768 \
  >"$LOG" 2>&1
status=$?
echo "$status" >"$EXIT_FILE"
exit "$status"
