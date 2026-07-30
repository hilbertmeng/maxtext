#!/bin/bash

start_time=$(date +%s)

ZONE=$1
TPU_TYPE=$2
BUCKET_PREFIX=$3

if [ -z "$ZONE" ]; then
    echo "Error argv zone is not detected"
    exit 1
fi

# 定义允许的ZONE值列表
allowed_zones=("us-east1" "us-west4" "us-central2" "us-central1" "europe-west4" "us-east5")

# 检查ZONE的值是否在允许的列表中
if [[ " ${allowed_zones[@]} " =~ " $ZONE " ]]; then
    echo "ZONE value: $ZONE"
else
    echo "ZONE value not in correct choices: ‘us-east1、us-west4、us-central2、us-central1’, please check zone again..."
    exit 1
fi


python==3.12
PACKAGE_NAME=maxtext_py312_packages.tar.gz
CONDA_NAME=Miniconda3-py312-Linux-x86_64.sh
PY_VERSION=3.12


CONDA_BUCKET=gs://newproject-1-conda_script_$ZONE
gsutil cp $CONDA_BUCKET'/'$PACKAGE_NAME  ./

HOME="/home/lishengping"
#git clone -b lsp/dev https://github.com/hilbertmeng/maxtext.git  /home/lishengping/projects/maxtext/
#git clone -b xd/dev https://github.com/hilbertmeng/maxtext.git  /home/lishengping/projects/maxtext/
git clone -b refactor https://github.com/hilbertmeng/maxtext.git  /home/lishengping/projects/maxtext/
#gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="killall train.py;sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/;export HARDWARE=tpu; sudo rm -r /home/lishengping/projects/maxtext; wget https://github.com/hilbertmeng/maxtext/archive/refs/heads/lsp/dev.zip; unzip dev.zip; mv maxtext-lsp-dev $WORK_DIR;" --project=ntpu-413714

# 安装conda环境
 if [ ! -d "$HOME/miniconda3" ]; then
  echo 'Conda is not existed, now start to install...'
  gsutil cp -r $CONDA_BUCKET/$CONDA_NAME  .
  bash $CONDA_NAME
  echo export PATH=$HOME"/miniconda3/bin:\$PATH" >> ~/.bashrc
  echo 'conda activate base' >> ~/.bashrc
  echo "Conda install finished..."
fi

pigz -dc $PACKAGE_NAME | tar xv -C  /home/lishengping/miniconda3/lib/python$PY_VERSION/

/home/lishengping/miniconda3/bin/pip install -f https://storage.googleapis.com/jax-releases/libtpu_releases.html jax[tpu]==0.8.1
/home/lishengping/miniconda3/bin/pip install -U flax==0.12.1  optax==0.2.6

