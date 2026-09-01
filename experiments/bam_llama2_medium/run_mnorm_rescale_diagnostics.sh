#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 TPU ZONE" >&2
  exit 2
fi

TPU=$1
ZONE=$2
PROJECT=newproject-1-451205
HISTORICAL_COMMIT=f079da1
REMOTE_REPO=/home/lishengping/xd/projects/maxtext
PYTHON=/home/lishengping/miniconda3/bin/python
DATASET=gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord
ARTIFACT_ROOT=gs://newproject-1-llm_base_models_us-central1/log/diagnostics/mnorm_rescale
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

for script in \
  "$SCRIPT_DIR/mnorm_rescale_runner.py" \
  "$SCRIPT_DIR/medium_v1_v2_gradient_profile.py" \
  "$SCRIPT_DIR/mnorm_historical_exp.patch" \
  "${BAM_DIAGNOSTICS_SCRIPT:-$SCRIPT_DIR/../../MaxText/bam_diagnostics.py}"; do
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
    cp /tmp/bam_diagnostics.py MaxText/
    root=/tmp/mnorm_rescale
    rm -rf \"\$root\"
    mkdir -p \"\$root\"
    artifact='$ARTIFACT_ROOT/'\$(date -u +%Y%m%dT%H%M%SZ)

    upload_arm() {
      local arm=\$1
      gcloud storage cp --recursive \"\$root/\$arm\" \"\$artifact/\" \
        >\"/tmp/mnorm_upload_\$arm.log\"
    }

    run_forward() {
      local mode=\$1
      local arm_scale=\$2
      local out=\"\$root/\$mode\"
      mkdir -p \"\$out/tb\" \"\$out/base/WRMNorm_\$mode\"
      env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
        BAM_MNORM_MODE=\"\$mode\" BAM_MNORM_SCALE=\"\$arm_scale\" \
        BAM_MNORM_TARGET=MaxText/bam_diagnostics.py \
        BAM_DIAG_BATCHES=1 BAM_DIAG_TOKEN_STRIDE=32 BAM_DIAG_SAVE_RAW=0 \
        BAM_DIAG_OUTPUT_DIR=\"\$out\" '$PYTHON' \
        experiments/bam_llama2_medium/mnorm_rescale_runner.py \
        MaxText/configs/base.yml \
        exp_class=BamLlama2MediumFactorizedLocalQKMNormDiagnostics \
        run_name=WRMNorm_\"\$mode\" only_eval=True steps=1 \
        dataset_path='$DATASET' \
        base_output_directory=\"\$out/base\" tensorboard_dir=\"\$out/tb\" \
        per_device_batch_size=1 eval_per_device_batch_size=1 \
        enable_checkpointing=False async_checkpointing=False \
        >\"\$out/forward.log\" 2>&1
    }

    run_gradient() {
      local mode=\$1
      local arm_scale=\$2
      local out=\"\$root/\$mode\"
      mkdir -p \"\$out/grad_base/WRMNormGrad_\$mode\"
      env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
        BAM_MNORM_MODE=\"\$mode\" BAM_MNORM_SCALE=\"\$arm_scale\" \
        BAM_MNORM_TARGET=experiments/bam_llama2_medium/medium_v1_v2_gradient_profile.py \
        BAM_GRAD_STEPS=1 BAM_GRAD_OUTPUT=\"\$out/gradient.json\" '$PYTHON' \
        experiments/bam_llama2_medium/mnorm_rescale_runner.py \
        MaxText/configs/base.yml \
        exp_class=BamLlama2MediumRmsGateOnlyDynamicRmsMixFull1CombinedReadFactorizedLocalQK \
        run_name=WRMNormGrad_\"\$mode\" steps=1 \
        dataset_path='$DATASET' \
        base_output_directory=\"\$out/grad_base\" tensorboard_dir=\"\$out/tb\" \
        per_device_batch_size=1 eval_per_device_batch_size=1 \
        enable_checkpointing=False async_checkpointing=False \
        >\"\$out/gradient.log\" 2>&1
    }

    run_forward none 1
    upload_arm none
    calibrated_scale=\$('$PYTHON' - <<'PY'
import json
import math
from pathlib import Path

report = json.loads(Path('/tmp/mnorm_rescale/none/bam_diagnostics.json').read_text())
layers = report['batches'][0]['layers']
for index in range(len(layers)):
  fro_rms = layers[f'layer_{index:02d}']['matrix']['M_in_fro']['rms']
  matrix_rms = fro_rms / math.sqrt(32 * 32)
  if matrix_rms > 1e-8:
    print(f'{matrix_rms:.12g}')
    break
else:
  raise RuntimeError('No nonzero M input found')
PY
    )
    echo \"CALIBRATED_SHARED_SCALE=\$calibrated_scale\"
    run_forward unit 1
    upload_arm unit
    run_forward shared_rescale \"\$calibrated_scale\"
    upload_arm shared_rescale
    run_gradient none 1
    upload_arm none
    run_gradient unit 1
    upload_arm unit
    run_gradient shared_rescale \"\$calibrated_scale\"
    upload_arm shared_rescale
    echo \"MNORM_RESCALE_DONE scale=\$calibrated_scale artifact=\$artifact\"
  "
