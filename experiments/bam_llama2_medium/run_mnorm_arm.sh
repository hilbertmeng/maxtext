#!/usr/bin/env bash
set -euo pipefail

if (( $# != 5 )); then
  echo "usage: $0 TPU ZONE MODE SCALE ARTIFACT_ROOT" >&2
  exit 2
fi

TPU=$1
ZONE=$2
MODE=$3
SCALE=$4
ARTIFACT_ROOT=$5
PROJECT=newproject-1-451205
REMOTE_REPO=/home/lishengping/xd/projects/maxtext
HISTORICAL_COMMIT=f079da1
PYTHON=/home/lishengping/miniconda3/bin/python
DATASET=gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BAM_DIAGNOSTICS_SCRIPT=${BAM_DIAGNOSTICS_SCRIPT:-$SCRIPT_DIR/bam_diagnostics.py}
ARM=${MODE}_s${SCALE//./p}

for script in \
  "$SCRIPT_DIR/mnorm_rescale_runner.py" \
  "$SCRIPT_DIR/medium_v1_v2_gradient_profile.py" \
  "$SCRIPT_DIR/mnorm_training_dynamics.py" \
  "$SCRIPT_DIR/mnorm_historical_exp.patch" \
  "$BAM_DIAGNOSTICS_SCRIPT"; do
  gcloud compute tpus tpu-vm scp --internal-ip "$script" \
    "$TPU:/tmp/$(basename "$script")" \
    --zone="$ZONE" --project="$PROJECT" --worker=0
done

gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 --command="
    set -euo pipefail
    cd '$REMOTE_REPO'
    git cat-file -e '$HISTORICAL_COMMIT^{commit}'
    git checkout --detach '$HISTORICAL_COMMIT'
    if git apply --reverse --check /tmp/mnorm_historical_exp.patch >/dev/null 2>&1; then
      :
    else
      git apply /tmp/mnorm_historical_exp.patch
    fi
    cp /tmp/mnorm_rescale_runner.py experiments/bam_llama2_medium/
    cp /tmp/medium_v1_v2_gradient_profile.py experiments/bam_llama2_medium/
    cp /tmp/mnorm_training_dynamics.py experiments/bam_llama2_medium/
    cp /tmp/bam_diagnostics.py MaxText/
    out=/tmp/mnorm_arm_$ARM
    rm -rf \"\$out\"
    mkdir -p \
      \"\$out/tb\" \
      \"\$out/base/WRMNorm_$ARM\" \
      \"\$out/grad_base/WRMNormGrad_$ARM\" \
      \"\$out/dynamics_base/WRMNormDynamics_$ARM\"

    env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
      BAM_MNORM_MODE='$MODE' BAM_MNORM_SCALE='$SCALE' \
      BAM_MNORM_TARGET=MaxText/bam_diagnostics.py \
      BAM_DIAG_BATCHES=1 BAM_DIAG_TOKEN_STRIDE=32 BAM_DIAG_SAVE_RAW=0 \
      BAM_DIAG_OUTPUT_DIR=\"\$out\" '$PYTHON' \
      experiments/bam_llama2_medium/mnorm_rescale_runner.py \
      MaxText/configs/base.yml \
      exp_class=BamLlama2MediumFactorizedLocalQKMNormDiagnostics \
      run_name=WRMNorm_$ARM only_eval=True steps=1 \
      dataset_path='$DATASET' base_output_directory=\"\$out/base\" \
      tensorboard_dir=\"\$out/tb\" per_device_batch_size=1 \
      eval_per_device_batch_size=1 enable_checkpointing=False \
      async_checkpointing=False >\"\$out/forward.log\" 2>&1
    gcloud storage cp --recursive \"\$out\" '$ARTIFACT_ROOT/' \
      >\"/tmp/mnorm_upload_${ARM}_forward.log\"

    env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
      BAM_MNORM_MODE='$MODE' BAM_MNORM_SCALE='$SCALE' \
      BAM_MNORM_TARGET=experiments/bam_llama2_medium/medium_v1_v2_gradient_profile.py \
      BAM_GRAD_STEPS=1 BAM_GRAD_OUTPUT=\"\$out/gradient.json\" '$PYTHON' \
      experiments/bam_llama2_medium/mnorm_rescale_runner.py \
      MaxText/configs/base.yml \
      exp_class=BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK \
      run_name=WRMNormGrad_$ARM steps=13500 \
      dataset_path='$DATASET' base_output_directory=\"\$out/grad_base\" \
      tensorboard_dir=\"\$out/tb\" per_device_batch_size=1 \
      eval_per_device_batch_size=1 enable_checkpointing=False \
      async_checkpointing=False >\"\$out/gradient.log\" 2>&1
    gcloud storage cp --recursive \"\$out\" '$ARTIFACT_ROOT/' \
      >\"/tmp/mnorm_upload_${ARM}_gradient.log\"

    env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
      BAM_MNORM_MODE='$MODE' BAM_MNORM_SCALE='$SCALE' \
      BAM_MNORM_TARGET=experiments/bam_llama2_medium/mnorm_training_dynamics.py \
      BAM_DYNAMICS_STEPS=21 BAM_DYNAMICS_OUTPUT=\"\$out/dynamics.json\" '$PYTHON' \
      experiments/bam_llama2_medium/mnorm_rescale_runner.py \
      MaxText/configs/base.yml \
      exp_class=BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK \
      run_name=WRMNormDynamics_$ARM steps=13500 \
      dataset_path='$DATASET' base_output_directory=\"\$out/dynamics_base\" \
      tensorboard_dir=\"\$out/tb\" per_device_batch_size=1 \
      eval_per_device_batch_size=1 enable_checkpointing=False \
      async_checkpointing=False >\"\$out/dynamics.log\" 2>&1
    gcloud storage cp --recursive \"\$out\" '$ARTIFACT_ROOT/' \
      >\"/tmp/mnorm_upload_${ARM}_dynamics.log\"
    echo MNORM_ARM_DONE arm='$ARM' artifact='$ARTIFACT_ROOT/$ARM'
  "
