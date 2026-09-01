#!/usr/bin/env bash
set -euo pipefail

EXP_CLASS=${1:?EXP_CLASS is required}
CHECKPOINT=${2:?checkpoint /items path is required}
OUTPUT_URI=${3:?output gs:// URI is required}
PYTHON=${PYTHON:-/home/lishengping/miniconda3/bin/python}
DATASET_PATH=${DATASET_PATH:-gs://newproject-1-common_datasets_europe-west4/pythia_pile_idxmaps_tfrecord}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
OUT=/tmp/fetch_gate_bins_${EXP_CLASS}

mkdir -p "$OUT/tensorboard"
env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
  BAM_FETCHAMP_DIAG_OUTPUT="$OUT/report.json" \
  "$PYTHON" experiments/bam_llama2_medium/fetch_amplitude_diagnostics.py \
  MaxText/configs/base.yml exp_class="$EXP_CLASS" \
  run_name="${EXP_CLASS}GateBinDiag" only_eval=True steps=1 \
  load_parameters_path="$CHECKPOINT" dataset_path="$DATASET_PATH" \
  base_output_directory="$OUT/output" tensorboard_dir="$OUT/tensorboard" \
  per_device_batch_size=1 eval_per_device_batch_size="$EVAL_BATCH_SIZE" \
  enable_checkpointing=True async_checkpointing=False \
  >"$OUT/run.log" 2>&1

gcloud storage cp "$OUT/report.json" "$OUTPUT_URI"
gcloud storage cp "$OUT/run.log" "${OUTPUT_URI%.json}.log"
echo "FETCH_GATE_BIN_DIAG_DONE exp=$EXP_CLASS output=$OUTPUT_URI"
