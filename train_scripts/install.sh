#!/bin/bash
set -e

ZONE=$1
BASE_DIR=/lustre-data
ENV_DIR=$BASE_DIR/conda_env
CODE_DIR=$BASE_DIR/maxtext

CONDA_BUCKET=gs://newproject-1-conda_script_$ZONE
PACKAGE_NAME=maxtext_py312_packages.tar.gz

echo "===== SETUP START ====="

# ===== 1. 下载环境（只执行一次）=====
if [ ! -d "$ENV_DIR" ]; then
    echo "Downloading prebuilt env..."

    gsutil cp $CONDA_BUCKET/$PACKAGE_NAME $BASE_DIR/

    mkdir -p $ENV_DIR
    tar -xzf $BASE_DIR/$PACKAGE_NAME -C $ENV_DIR

    echo "Env extracted to $ENV_DIR"
else
    echo "Env already exists"
fi

# ===== 2. clone 代码 =====
if [ ! -d "$CODE_DIR" ]; then
    git clone -b lsp/refactor20260130 https://github.com/hilbertmeng/maxtext.git $CODE_DIR
fi

echo "===== SETUP DONE ====="