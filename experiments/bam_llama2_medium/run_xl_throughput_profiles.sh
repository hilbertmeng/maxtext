#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  echo "usage: $0 TPU ZONE COMMIT LABEL [EXP_CLASS...]" >&2
  exit 2
fi

TPU=$1
ZONE=$2
COMMIT=$3
LABEL=$4
shift 4

PROJECT=newproject-1-451205
REMOTE_REPO=/home/lishengping/xd/projects/maxtext
REMOTE_ROOT=/home/lishengping/xd/profile_outputs/xl_throughput
LOG_ROOT=/home/lishengping/xd/projects/logs
DATASET=gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord
BASE_OUTPUT=gs://newproject-1-llm_base_models_us-central1/log/
GCS_ROOT=gs://newproject-1-llm_base_models_us-central1/log/diagnostics/xl_throughput

if (( $# == 0 )); then
  set -- \
    BamMHALlama2XLHead16x128C256T2048Profile \
    BamLlama2XLHead16x128V2C256T2048Profile \
    BamMHALlama2XLHead16x128C256T4096Profile \
    BamLlama2XLHead16x128V2C256T4096Profile \
    BamMHALlama2XLHead32x64C256T2048Profile \
    BamLlama2XLHead32x64V2C256T2048Profile \
    BamMHALlama2XLHead32x64C256T4096Profile \
    BamLlama2XLHead32x64V2C256T4096Profile
fi

gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
  --project="$PROJECT" --worker=all \
  --command="cd '$REMOTE_REPO' && git fetch origin refactor-bam && \
  git reset --hard && git clean -ffd && git checkout --detach '$COMMIT' && \
  test \"\$(git rev-parse HEAD)\" = '$COMMIT'"

gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
  --project="$PROJECT" --worker=0 \
  --command="cd '$REMOTE_REPO' && PYTHONPATH=MaxText \
  /home/lishengping/miniconda3/bin/python -m unittest \
  MaxText.tests.bam_attention_test.BamReadKeyTransformTest.test_bam_read_head_mapping_pads_or_adapts_only_v_side \
  MaxText.tests.bam_attention_test.BamReadKeyTransformTest.test_bam_read_head_mapping_rejects_wide_v_without_adapter"

mkdir -p "$LOG_ROOT"
for exp_class in "$@"; do
  run="XL${COMMIT:0:7}_${LABEL}_${exp_class}"
  remote_run="$REMOTE_ROOT/$run"
  gcs_run="$GCS_ROOT/${COMMIT:0:7}/$LABEL/$run"
  collector_log="$LOG_ROOT/${run}-collector.log"

  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker=all \
    --command="pkill -KILL -f '[M]axText/train.py.*run_name=$run' \
    2>/dev/null || true; rm -rf '$remote_run'; mkdir -p '$remote_run'; \
    sudo rm -f /tmp/libtpu_lockfile"

  /home/lishengping/xd/projects/collect_xplane.sh \
    "$TPU" "$ZONE" "$remote_run/tensorboard" "$gcs_run" "$PROJECT" 2 0 \
    >"$collector_log" 2>&1 &
  collector_pid=$!

  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker=all \
    --command="export HARDWARE=tpu JAX_TRACEBACK_FILTERING=off; \
    cd '$REMOTE_REPO'; nohup /home/lishengping/miniconda3/bin/python \
    MaxText/train.py MaxText/configs/base.yml base_output_directory='$BASE_OUTPUT' \
    tensorboard_dir='$remote_run/tensorboard' run_name='$run' exp_class='$exp_class' \
    dataset_path='$DATASET' > /home/lishengping/train_${run}.log 2>&1 </dev/null &"

  deadline=$((SECONDS + 3600))
  reached_trace=false
  while (( SECONDS < deadline )); do
    status=$(gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
      --project="$PROJECT" --worker=0 \
      --command="if grep -q 'completed step: 15,' \
      /home/lishengping/train_${run}.log 2>/dev/null; then echo DONE; \
      elif grep -Eq '(^| )Traceback \(most recent call last\):|ValueError:|AssertionError:|TypeError:|RESOURCE_EXHAUSTED' \
      /home/lishengping/train_${run}.log 2>/dev/null; then echo ERROR; \
      elif pgrep -f '[M]axText/train.py.*run_name=$run' >/dev/null; then echo RUNNING; \
      else echo EXITED; fi" 2>/dev/null | tail -1)
    case "$status" in
      DONE) reached_trace=true; break ;;
      ERROR|EXITED)
        echo "$run $status" >&2
        gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
          --project="$PROJECT" --worker=0 \
          --command="tail -100 /home/lishengping/train_${run}.log" >&2 || true
        break ;;
    esac
    sleep 10
  done

  if $reached_trace; then
    wait "$collector_pid"
  else
    kill "$collector_pid" 2>/dev/null || true
    wait "$collector_pid" 2>/dev/null || true
  fi
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker=all \
    --command="pkill -KILL -f '[M]axText/train.py.*run_name=$run' \
    2>/dev/null || true; sudo rm -f /tmp/libtpu_lockfile"
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker=0 \
    --command="grep 'completed step:' /home/lishengping/train_${run}.log | tail -6" || true

  $reached_trace || exit 1
done
