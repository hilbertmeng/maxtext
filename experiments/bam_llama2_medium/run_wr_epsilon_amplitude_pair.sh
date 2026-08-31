#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/lishengping/miniconda3/bin/python}
DATASET=${DATASET_PATH:-gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord}
OUTPUT_DIR=${BAM_FETCHAMP_DIAG_OUTPUT_DIR:-/tmp/wr_epsilon_amplitude_pair}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-gs://newproject-1-llm_projects_us-east5/log}

mkdir -p "$OUTPUT_DIR"
for exp_class in \
  BamLlama2MediumV2NonScanJitRepro \
  BamLlama2MediumV2NonScanJitWRReadEps1e3; do
  run_name="${exp_class}FetchAmpDiag"
  checkpoint="${CHECKPOINT_ROOT}/${exp_class}/checkpoints/200/items"
  mkdir -p "$OUTPUT_DIR/$exp_class/$run_name"
  env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
    BAM_FETCHAMP_DIAG_OUTPUT="$OUTPUT_DIR/${exp_class}.json" \
    "$PYTHON" experiments/bam_llama2_medium/fetch_amplitude_diagnostics.py \
    MaxText/configs/base.yml exp_class="$exp_class" run_name="$run_name" \
    only_eval=True steps=1 load_parameters_path="$checkpoint" \
    dataset_path="$DATASET" base_output_directory="$OUTPUT_DIR/$exp_class" \
    tensorboard_dir="$OUTPUT_DIR/$exp_class/tensorboard" \
    per_device_batch_size=1 eval_per_device_batch_size=1 \
    enable_checkpointing=False async_checkpointing=False \
    >"$OUTPUT_DIR/${exp_class}.log" 2>&1
done

echo "WR_EPSILON_AMPLITUDE_PAIR_DONE output=$OUTPUT_DIR"
