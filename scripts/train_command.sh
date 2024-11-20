#!/bin/bash


if [ $# -eq 0 ]; then
  echo "请提供一个参数。"
  exit 1
fi


if [ "$1" == "instruct" ]; then
echo "Instruct train strat."
BUCKET_ZONE=europe-west4
ZONE=$BUCKET_ZONE-b
TPU_TYPE=v5p-128
TPU_NAME=llm-jax-$TPU_TYPE-11
export HARDWARE='tpu'
CONFIG=dc_7b_instruct.yml
WORK_DIR=/home/lishengping/projects/maxtext
RUN_NAME=gs://llm_base_models_$BUCKET_ZONE/v5p_256/7B/xm3.5-7b-chat-v7
DATASET_PATH=gs://jax_llm_data_$BUCKET_ZONE/instruct_datasets/instruct_role_play_translation/role_play_v3@gs://jax_llm_data_$BUCKET_ZONE/instruct_datasets/instruct_role_play_translation/role_play_new1@gs://jax_llm_data_$BUCKET_ZONE/instruct_datasets/instruct_role_play_translation/translation@gs://jax_llm_data_$BUCKET_ZONE/xiaomeng/en_data_Qwen-14B_1014@gs://jax_llm_data_$BUCKET_ZONE/xiaomeng/zh_data_Qwen-14B_1014@gs://jax_llm_data_$BUCKET_ZONE/instruct_datasets/processed_general_1016_v2@gs://jax_llm_data_us-east5/instruct_datasets/whoareyou
# COMPILED_FILE=dc_7b_instruct.$TPU_TYPE.compiled
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd $WORK_DIR; gsutil cp $RUN_NAME/$COMPILED_FILE ."
COMPILED_FILE=''
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall train.py;sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/;export HARDWARE=tpu; cd $WORK_DIR; nohup /home/lishengping/miniconda3/bin/python MaxText/train.py MaxText/configs/$CONFIG  run_name=$RUN_NAME  dataset_path=$DATASET_PATH compiled_trainstep_file=$COMPILED_FILE enable_checkpointing=True per_device_batch_size=16 eval_per_device_batch_size=16  load_ocdbt=True save_ocdbt=True train_shuffle_buffer_size=100000 keep_period=400 checkpoint_period=200 dataset_path=$DATASET_PATH dataset_type=instruct iter_file_nums=223 epoch=3 eval_start_step=False eval_loop_num_batches=60 rope_max_timescale=500000 2>&1 | tee chat.train.log &" --project=ntpu-413714


elif [ "$1" == "moe" ]; then
echo "MOE train strat."
export HARDWARE='tpu'
BUCKET_ZONE=europe-west4
ZONE=$BUCKET_ZONE-b
TPU_TYPE=v5p-256
TPU_NAME=llm-jax-$TPU_TYPE-10
RUN_NAME=gs://llm_base_models_$BUCKET_ZONE/v5p_256/7B/xm_45x7B_moe_1113
DATASET_PATH=gs://jax_llm_data_$BUCKET_ZONE/xiaomeng/v3.5/tfids0527
WORK_DIR=/home/lishengping/projects/maxtext/
CONFIG=dc_8x7b_moe.yml
COMPILED_FILE=''
# COMPILED_FILE='v5p-256-compiled.pkl'
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd $WORK_DIR; gsutil cp $RUN_NAME/$COMPILED_FILE ."
# start from 0 step
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall train.py;sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/;export HARDWARE=tpu; cd $WORK_DIR; nohup /home/lishengping/miniconda3/bin/python MaxText/train.py MaxText/configs/$CONFIG  run_name=$RUN_NAME  dataset_path=$DATASET_PATH compiled_trainstep_file=$COMPILED_FILE enable_checkpointing=True per_device_batch_size=1 eval_per_device_batch_size=1 load_parameters_path=$RUN_NAME/checkpoints/0 load_ocdbt=False save_ocdbt=True  mgate=True keep_period=1000 checkpoint_period=125 2>&1 | tee moe_train.log &" --project=ntpu-413714
# continue train from break step
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall train.py;sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/;export HARDWARE=tpu; cd $WORK_DIR; nohup /home/lishengping/miniconda3/bin/python MaxText/train.py MaxText/configs/$CONFIG  run_name=$RUN_NAME  dataset_path=$DATASET_PATH compiled_trainstep_file=$COMPILED_FILE enable_checkpointing=True per_device_batch_size=8 eval_per_device_batch_size=8 load_ocdbt=False save_ocdbt=True mgate=True keep_period=1000 checkpoint_period=200 2>&1 num_groups=1024 load_parameters_path=$RUN_NAME/checkpoints/0 | tee moe_train.log &" --project=ntpu-413714


elif [ "$1" == "compile" ]; then
# v5p-8 complie
export HARDWARE='tpu'
BUCKET_ZONE=us-east5
ZONE=$BUCKET_ZONE-a
TPU_TYPE=v5p-8
TPU_NAME=llm-jax-$TPU_TYPE-10
# 之后训练的tpu type
CoTPU=v5p-128
CONFIG=dc_7b_instruct.yml
COMPILED_FILE=dc_7b_instruct.$CoTPU.compiled
RUN_NAME=gs://llm_base_models_$BUCKET_ZONE/v5p_256/7B/instruct_role_trans_base_xm32k_more_general_novel_v5
WORK_DIR=/home/lishengping/projects/maxtext
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="export HARDWARE=tpu;cd $WORK_DIR;/home/lishengping/miniconda3/bin/python MaxText/train_compile.py MaxText/configs/$CONFIG compile_topology=$CoTPU compile_topology_num_slices=1 compiled_trainstep_file=$COMPILED_FILE per_device_batch_size=16 eval_per_device_batch_size=16 run_name=$RUN_NAME" --project=ntpu-413714
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd $WORK_DIR; gsutil cp $COMPILED_FILE $RUN_NAME/$COMPILED_FILE"


# v5p-8 complie
export HARDWARE='tpu'
BUCKET_ZONE=europe-west4
ZONE=$BUCKET_ZONE-b
TPU_TYPE=v5p-8
TPU_NAME=llm-jax-$TPU_TYPE-10
# 之后训练的tpu type
CoTPU=v5p-256
CONFIG=dc_8x7b_moe.yml
COMPILED_FILE=dc_8x7b_moe.$CoTPU.compiled
RUN_NAME=gs://llm_base_models_$BUCKET_ZONE/v5p_256/7B/xm_45x7B_moe_base500k_1022
WORK_DIR=/home/lishengping/projects/maxtext
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="export HARDWARE=tpu;cd $WORK_DIR;/home/lishengping/miniconda3/bin/python MaxText/train_compile.py MaxText/configs/$CONFIG compile_topology=$CoTPU compile_topology_num_slices=1 compiled_trainstep_file=$COMPILED_FILE per_device_batch_size=8 eval_per_device_batch_size=8 run_name=$RUN_NAME mgate=False" --project=ntpu-413714
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd $WORK_DIR; gsutil cp $COMPILED_FILE $RUN_NAME/$COMPILED_FILE"

else
  echo "Unknow argv."
  exit 1

fi

# 在远程vm上启动tensorboard
# python miniconda3/lib/python3.10/site-packages/tensorboard/main.py  --logdir gs://llm_base_models_us-east5/maxtext_tensorboard_dir/xm_45x7B_moe_0922 --bind_all --port 60000