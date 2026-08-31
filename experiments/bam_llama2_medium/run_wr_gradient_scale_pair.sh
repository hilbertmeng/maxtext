#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/lishengping/miniconda3/bin/python}
DATASET=${DATASET_PATH:-gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord}
OUTPUT_DIR=${BAM_GRAD_OUTPUT_DIR:-/tmp/wr_gradient_scale_pair}

mkdir -p "$OUTPUT_DIR"
for exp_class in \
  BamLlama2MediumV2NonScanJitRepro \
  BamLlama2MediumV2NonScanJitWRGradScale01; do
  output="$OUTPUT_DIR/${exp_class}.json"
  log="$OUTPUT_DIR/${exp_class}.log"
  mkdir -p "$OUTPUT_DIR/$exp_class/${exp_class}GradientProfile"
  env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
    BAM_GRAD_STEPS=1 BAM_GRAD_VARIANTS=baseline BAM_GRAD_OUTPUT="$output" \
    "$PYTHON" experiments/bam_llama2_medium/medium_v1_v2_gradient_profile.py \
    MaxText/configs/base.yml exp_class="$exp_class" \
    run_name="${exp_class}GradientProfile" steps=1 dataset_path="$DATASET" \
    base_output_directory="$OUTPUT_DIR/$exp_class" \
    tensorboard_dir="$OUTPUT_DIR/$exp_class/tensorboard" \
    per_device_batch_size=1 eval_per_device_batch_size=1 \
    enable_checkpointing=False async_checkpointing=False >"$log" 2>&1
done

echo "WR_GRAD_SCALE_PAIR_DONE output=$OUTPUT_DIR"
