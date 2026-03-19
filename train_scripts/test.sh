#!/bin/bash
set -e

export BUCKET_ZONE=us-east5
export BASE_OUTPUT_DIR=gs://newproject-1-llm_base_models_\$BUCKET_ZONE/test/
export RUN_NAME=V4p5LongTest
export DATASET_PATH=gs://newproject-1-llm_base_models_\$BUCKET_ZONE/data/xiaomeng/v3.5/tfids1210
export QCHUNK=1024

cd /lustre-data/maxtext

python3 MaxText/train.py MaxText/configs/base.yml \
  base_output_directory=\$BASE_OUTPUT_DIR \
  run_name=\$RUN_NAME \
  dataset_path=\$DATASET_PATH \
  exp_class='V4p5LongTest' \
  enable_checkpointing=False \
  dataset_type='pretrain_4k' \
  bucket_logging_enabled=False \
  max_target_length=16384 \
  sharding_tolerance=100 \
  query_chunk_size=\$QCHUNK \
  query_chunk_method='ddd' \
  attention='flash' \
  debug=True \
  record_internal_nn_metrics=0 \
  per_device_batch_size=1.0 \
  eval_per_device_batch_size=1.0 \
  scan_use_mudd=False \
  eval_split=valid \
  train_shuffle_buffer_size=1000 \
  eval_steps=100 \
  num_vocab_tiling=4 \
  base_num_decoder_layers=4 \
  2>&1 | tee /lustre-data/\$RUN_NAME.log
