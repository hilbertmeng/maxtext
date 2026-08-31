#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 TPU ZONE" >&2
  exit 2
fi

TPU=$1
ZONE=$2
PROJECT=newproject-1-451205
REMOTE_REPO=/home/lishengping/xd/projects/maxtext
PYTHON=/home/lishengping/miniconda3/bin/python
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SCRIPT=$SCRIPT_DIR/medium_v1_v2_gradient_profile.py
REMOTE_SCRIPT=/tmp/medium_v1_v2_gradient_profile.py
OUTPUT_DIR=${BAM_GRAD_OUTPUT_DIR:-/home/lishengping/xd/projects/gradient_profiles}
DATASET=${DATASET_PATH:-gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord}
TRACE_STEPS=${BAM_GRAD_STEPS:-3}
VARIANTS=${BAM_GRAD_VARIANTS:-baseline}

mkdir -p "$OUTPUT_DIR"
gcloud compute tpus tpu-vm scp --internal-ip "$SCRIPT" \
  "$TPU:$REMOTE_SCRIPT" --zone="$ZONE" --project="$PROJECT" --worker=0

profiles=(
  "v1 03367ac BamLlama2MediumV1"
  "v2 1afd942 BamLlama2MediumV2"
)

for profile in "${profiles[@]}"; do
  read -r label commit exp_class <<<"$profile"
  remote_output="/tmp/medium_${label}_gradient_profile.json"
  remote_log="/tmp/medium_${label}_gradient_profile.log"
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" \
    --zone="$ZONE" --project="$PROJECT" --worker=0 --command="
      set -euo pipefail
      cd '$REMOTE_REPO'
      git reset --hard >/dev/null
      git clean -ffd >/dev/null
      git cat-file -e '$commit^{commit}' 2>/dev/null || git fetch --quiet origin '$commit'
      git checkout --quiet --detach '$commit'
      mkdir -p experiments/bam_llama2_medium \
        /tmp/medium_${label}_gradient_profile/tensorboard \
        /tmp/medium_${label}_gradient_profile/MediumGradientProfile_${label}
      cp '$REMOTE_SCRIPT' experiments/bam_llama2_medium/medium_v1_v2_gradient_profile.py
      env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
        BAM_GRAD_GIT_COMMIT='$commit' BAM_GRAD_STEPS='$TRACE_STEPS' \
        BAM_GRAD_VARIANTS='$VARIANTS' \
        BAM_GRAD_OUTPUT='$remote_output' '$PYTHON' \
        experiments/bam_llama2_medium/medium_v1_v2_gradient_profile.py \
        MaxText/configs/base.yml exp_class='$exp_class' \
        run_name='MediumGradientProfile_${label}' steps='$TRACE_STEPS' \
        dataset_path='$DATASET' \
        base_output_directory='/tmp/medium_${label}_gradient_profile' \
        tensorboard_dir='/tmp/medium_${label}_gradient_profile/tensorboard' \
        per_device_batch_size=1 eval_per_device_batch_size=1 \
        enable_checkpointing=False async_checkpointing=False \
        >'$remote_log' 2>&1
    "
  gcloud compute tpus tpu-vm scp --internal-ip \
    "$TPU:$remote_output" "$OUTPUT_DIR/" \
    --zone="$ZONE" --project="$PROJECT" --worker=0
  gcloud compute tpus tpu-vm scp --internal-ip \
    "$TPU:$remote_log" "$OUTPUT_DIR/" \
    --zone="$ZONE" --project="$PROJECT" --worker=0
  echo "$label commit=$commit output=$OUTPUT_DIR/$(basename "$remote_output")"
done
