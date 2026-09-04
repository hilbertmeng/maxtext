#!/usr/bin/env bash
set -euo pipefail

PYTHON="${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}"
COHORT_GCS="${BAM_RESIDUAL_ATTR_COHORT_GCS:-gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz}"
DATASET="${DATASET_PATH:-gs://newproject-1-common_datasets_europe-west4/pythia_pile_idxmaps_tfrecord}"
OUTPUT_BASE="${OUTPUT_BASE:-gs://newproject-1-llm_projects_europe-west4/log/diagnostics}"
CODE_COMMIT="${BAM_ABSV_DIAG_CODE_COMMIT:-c09b010}"
COHORT_PATH="${BAM_ABSV_DIAG_COHORT_PATH:-/tmp/pile-eval-t2048-seed9876-n128-v1.npz}"

gsutil cp "$COHORT_GCS" "$COHORT_PATH"

run_one() {
  local config="$1" trainer_commit="$2" step="$3" short_name="$4"
  local checkpoint="gs://newproject-1-llm_projects_europe-west4/log/$config/checkpoints/$step/items"
  local run="xl-rank2-${short_name}-s${step}-health-pile128-${CODE_COMMIT:0:7}"
  local output="/tmp/$run"
  mkdir -p "$output/output/$run" "$output/tensorboard"
  env \
    HARDWARE=tpu \
    JAX_TRACEBACK_FILTERING=off \
    BAM_ABSV_DIAG_BATCHES=8 \
    BAM_ABSV_DIAG_CAPTURE_BATCHES=8 \
    BAM_ABSV_DIAG_SCALES=1 \
    BAM_ABSV_DIAG_RANKS= \
    BAM_ABSV_DIAG_LAYERWISE_RANK=0 \
    BAM_ABSV_DIAG_CODE_COMMIT="$CODE_COMMIT" \
    BAM_ABSV_DIAG_TRAINER_COMMIT="$trainer_commit" \
    BAM_ABSV_DIAG_COHORT_PATH="$COHORT_PATH" \
    BAM_ABSV_DIAG_OUTPUT="$output/report.json" \
    "$PYTHON" experiments/bam_llama2_medium/xl_abs_v_width_diagnostics.py \
      MaxText/configs/base.yml \
      "exp_class=$config" \
      "run_name=$run" \
      only_eval=True steps=1 \
      "load_parameters_path=$checkpoint" \
      "dataset_path=$DATASET" \
      "base_output_directory=$output/output" \
      "tensorboard_dir=$output/tensorboard" \
      per_device_batch_size=1 eval_per_device_batch_size=16 \
      enable_checkpointing=True async_checkpointing=False \
      >"$output/run.log" 2>&1
  gsutil cp "$output/report.json" "$OUTPUT_BASE/$run/report.json"
  gsutil cp "$output/run.log" "$OUTPUT_BASE/$run/run.log"
  echo "XL_ABSV_HEALTH_DONE run=$run"
}

run_one BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV16 \
  fbde4efd3336ab65221a7887b9c3548232d8c10f 4250 c16
run_one BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV32Projected \
  c930d04a1302d045ef52d1cf38c6ce7768e221c5 6000 c32p
run_one BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AbsV32Native \
  c930d04a1302d045ef52d1cf38c6ce7768e221c5 8750 c32n
