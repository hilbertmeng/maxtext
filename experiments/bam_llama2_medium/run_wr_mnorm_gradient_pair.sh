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
GRAD_SCRIPT=$SCRIPT_DIR/medium_v1_v2_gradient_profile.py
CAPTURE_SCRIPT=$SCRIPT_DIR/fetch_amplitude_diagnostics.py
OUTPUT_DIR=${BAM_MNORM_OUTPUT_DIR:-/home/lishengping/xd/projects/gradient_profiles/wr_mnorm}
DATASET=${DATASET_PATH:-gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord}

mkdir -p "$OUTPUT_DIR"
for script in "$GRAD_SCRIPT" "$CAPTURE_SCRIPT"; do
  gcloud compute tpus tpu-vm scp --internal-ip "$script" \
    "$TPU:/tmp/$(basename "$script")" \
    --zone="$ZONE" --project="$PROJECT" --worker=0
done

for norm in none rms; do
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" \
    --zone="$ZONE" --project="$PROJECT" --worker=0 --command="
      set -euo pipefail
      cd '$REMOTE_REPO'
      cp /tmp/medium_v1_v2_gradient_profile.py \
        experiments/bam_llama2_medium/medium_v1_v2_gradient_profile.py
      cp /tmp/fetch_amplitude_diagnostics.py \
        experiments/bam_llama2_medium/fetch_amplitude_diagnostics.py
      out=/tmp/wr_mnorm_${norm}
      mkdir -p \"\$out/tensorboard\" \"\$out/grad\" \"\$out/capture\"
      env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off BAM_GRAD_STEPS=1 \
        BAM_GRAD_OUTPUT=\"\$out/grad.json\" '$PYTHON' \
        experiments/bam_llama2_medium/medium_v1_v2_gradient_profile.py \
        MaxText/configs/base.yml \
        exp_class=BamLlama2MediumV2NonScanJitRepro \
        run_name=WRMNormGrad_${norm} bam_m_read_norm=${norm} steps=1 \
        dataset_path='$DATASET' base_output_directory=\"\$out/grad\" \
        tensorboard_dir=\"\$out/tensorboard\" \
        per_device_batch_size=1 eval_per_device_batch_size=1 \
        enable_checkpointing=False async_checkpointing=False \
        >\"\$out/grad.log\" 2>&1
      env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
        BAM_FETCHAMP_DIAG_OUTPUT=\"\$out/capture.json\" '$PYTHON' \
        experiments/bam_llama2_medium/fetch_amplitude_diagnostics.py \
        MaxText/configs/base.yml \
        exp_class=BamLlama2MediumV2NonScanJitRepro \
        run_name=WRMNormCapture_${norm} bam_m_read_norm=${norm} \
        only_eval=True steps=1 dataset_path='$DATASET' \
        base_output_directory=\"\$out/capture\" \
        tensorboard_dir=\"\$out/tensorboard\" \
        per_device_batch_size=1 eval_per_device_batch_size=1 \
        enable_checkpointing=False async_checkpointing=False \
        >\"\$out/capture.log\" 2>&1
    "
  for artifact in grad.json grad.log capture.json capture.log; do
    gcloud compute tpus tpu-vm scp --internal-ip \
      "$TPU:/tmp/wr_mnorm_${norm}/$artifact" \
      "$OUTPUT_DIR/${norm}_${artifact}" \
      --zone="$ZONE" --project="$PROJECT" --worker=0
  done
done

echo "WR_MNORM_GRADIENT_PAIR_DONE output=$OUTPUT_DIR"
