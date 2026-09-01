#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo "usage: $0 TPU ZONE SCALE ARTIFACT_ROOT" >&2
  exit 2
fi

TPU=$1
ZONE=$2
SCALE=$3
ARTIFACT_ROOT=$4
PROJECT=newproject-1-451205
HISTORICAL_COMMIT=f079da1
REMOTE_REPO=/home/lishengping/xd/projects/maxtext
PYTHON=/home/lishengping/miniconda3/bin/python
DATASET=gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RUN=WRMNormSharedScale0411624Short30

gcloud compute tpus tpu-vm scp --internal-ip \
  "$SCRIPT_DIR/mnorm_rescale_runner.py" "$TPU:/tmp/mnorm_rescale_runner.py" \
  --zone="$ZONE" --project="$PROJECT" --worker=all

gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" \
  --zone="$ZONE" --project="$PROJECT" --worker=all --command="
    set -euo pipefail
    cd '$REMOTE_REPO'
    git cat-file -e '$HISTORICAL_COMMIT^{commit}'
    git checkout --detach '$HISTORICAL_COMMIT'
    cp /tmp/mnorm_rescale_runner.py experiments/bam_llama2_medium/
    out=/tmp/mnorm_shared_short
    mkdir -p \"\$out/base/$RUN\" \"\$out/tb\"
    env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
      BAM_MNORM_MODE=shared_rescale BAM_MNORM_SCALE='$SCALE' \
      BAM_MNORM_TARGET=MaxText/train.py '$PYTHON' \
      experiments/bam_llama2_medium/mnorm_rescale_runner.py \
      MaxText/configs/base.yml \
      exp_class=BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK \
      run_name='$RUN' steps=31 learning_rate_schedule_steps=13500 \
      dataset_path='$DATASET' base_output_directory=\"\$out/base\" \
      tensorboard_dir=\"\$out/tb\" enable_checkpointing=False \
      async_checkpointing=False >\"\$out/train.log\" 2>&1
  "

# Multi-host process 0 is not guaranteed to be gcloud worker 0. Preserve both
# workers so the primary-host TensorBoard event file is always collected.
for worker in 0 1; do
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" \
    --zone="$ZONE" --project="$PROJECT" --worker="$worker" --command="
      set -euo pipefail
      gcloud storage cp --recursive /tmp/mnorm_shared_short \
        '$ARTIFACT_ROOT/worker-$worker/' \
        > /tmp/mnorm_shared_short_upload.log
    "
done
echo "MNORM_SHARED_SHORT_DONE artifact=$ARTIFACT_ROOT"
