#!/bin/bash
set -e

BASE_DIR=/lustre-data
ENV_DIR=$BASE_DIR/conda_env
CODE_DIR=$BASE_DIR/maxtext

export PYTHONPATH=$ENV_DIR/lib/python3.12/site-packages:$PYTHONPATH
export PATH=$ENV_DIR/bin:$PATH

cd $CODE_DIR

$ENV_DIR/bin/python3 MaxText/train.py \
  MaxText/configs/base.yml \
  base_output_directory=gs://newproject-1-llm_base_models_us-east5/test/ \
  run_name=V4p5LongTest \
  dataset_path=gs://newproject-1-llm_base_models_us-east5/data/xiaomeng/v3.5/tfids1210 \
  exp_class='V4p5LongTest' \
  enable_checkpointing=False \
  dataset_type='pretrain_4k' \
  bucket_logging_enabled=False \
  max_target_length=16384 \
  sharding_tolerance=100 \
  query_chunk_size=1024 \
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
  2>&1 | tee /lustre-data/logs/V4p5LongTest.log