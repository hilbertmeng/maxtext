#!/bin/bash

NODE_COUNT=2
TPU_TYPE=v5p-16
BUCKET_ZONE=europe-west4
ZONE=$BUCKET_ZONE-b
TPU_NAME=llm-jax-$TPU_TYPE-multislice

RUN_NAME='multislice-test'
# INSTALL_FILE=/Users/lishengping/temp/tpu/install_maxtext.sh
INSTALL_FILE=/home/lishengping/lsp/install_maxtext.sh

PROJECT=ntpu-413714

# 删除之前的 TPU 资源
echo 'Y' | gcloud alpha compute tpus queued-resources delete $TPU_NAME --zone=$ZONE --project=$PROJECT

# 创建新的 TPU 资源
gcloud alpha compute tpus queued-resources create $TPU_NAME \
  --accelerator-type=$TPU_TYPE \
  --runtime-version=tpu-ubuntu2204-base \
  --node-count=$NODE_COUNT \
  --node-prefix=$TPU_NAME \
  --best-effort \
  --zone=$ZONE \
  --project=$PROJECT

# 定义并行任务函数
run_parallel_tasks() {
  NODE_INDEX=$1

  # 传输安装文件到对应节点
gcloud compute tpus tpu-vm scp $INSTALL_FILE $TPU_NAME-$NODE_INDEX:~/ \
    --zone=$ZONE \
    --worker=all \
    --project=$PROJECT

  # 在对应节点上执行安装脚本
gcloud compute tpus tpu-vm ssh $TPU_NAME-$NODE_INDEX \
    --zone=$ZONE \
    --worker=all \
    --command="bash install_maxtext.sh $BUCKET_ZONE" \
    --project=$PROJECT
}

# 并行运行两个节点的任务
run_parallel_tasks 0 &
PID1=$!
run_parallel_tasks 1 &
PID2=$!

# 等待所有任务完成
wait $PID1 $PID2

echo "所有节点已完成安装任务。"