#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-/home/lishengping/miniconda3/bin/python}
CHECKPOINT=${CHECKPOINT:-gs://newproject-1-llm_projects_europe-west4/log/BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2/checkpoints/49720/items}
DATASET_PATH=${DATASET_PATH:-gs://newproject-1-common_datasets_europe-west4/pythia_pile_idxmaps_tfrecord}
OUTPUT_URI=${OUTPUT_URI:-gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl_rank2_c8_redundancy/report.json}
CODE_COMMIT=${CODE_COMMIT:-$(git rev-parse HEAD)}
OUT=${OUT:-/tmp/xl_rank2_c8_redundancy}

mkdir -p "$OUT/output" "$OUT/tensorboard"
env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
  BAM_ABSV_DIAG_BATCHES=8 BAM_ABSV_DIAG_CAPTURE_BATCHES=8 \
  BAM_ABSV_DIAG_SCALES=1 BAM_ABSV_DIAG_RANKS=2,4,6 \
  BAM_ABSV_DIAG_CODE_COMMIT="$CODE_COMMIT" \
  BAM_ABSV_DIAG_OUTPUT="$OUT/report.json" \
  "$PYTHON" experiments/bam_llama2_medium/xl_abs_v_width_diagnostics.py \
  MaxText/configs/base.yml \
  exp_class=BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2 \
  run_name=BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2C8RedundancyDiag \
  only_eval=True steps=1 load_parameters_path="$CHECKPOINT" \
  dataset_path="$DATASET_PATH" base_output_directory="$OUT/output" \
  tensorboard_dir="$OUT/tensorboard" per_device_batch_size=1 \
  eval_per_device_batch_size=16 enable_checkpointing=True \
  async_checkpointing=False >"$OUT/run.log" 2>&1

gcloud storage cp "$OUT/report.json" "$OUTPUT_URI"
gcloud storage cp "$OUT/run.log" "${OUTPUT_URI%.json}.log"
echo "XL_C8_REDUNDANCY_DONE output=$OUTPUT_URI"
