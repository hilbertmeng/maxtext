#!/usr/bin/env bash
set -euo pipefail

if (( $# < 5 )); then
  echo "usage: $0 TPU ZONE COMMIT LABEL EXP_CLASS..." >&2
  exit 2
fi

TPU=$1
ZONE=$2
COMMIT=$3
LABEL=$4
shift 4

PROJECT=newproject-1-451205
REMOTE_REPO=/home/lishengping/xd/projects/maxtext
REMOTE_ROOT=/home/lishengping/xd/profile_outputs
ARTIFACT_ROOT=/home/lishengping/xd/projects/bam_diagnostics/mix_alpha/${COMMIT:0:7}/${LABEL}
LOG_ROOT=/home/lishengping/xd/projects/logs
DATASET=gs://newproject-1-llm_base_models_us-central1/data/pythia_pile_idxmaps_tfrecord
BASE_OUTPUT=gs://newproject-1-llm_base_models_us-central1/log/
WORKER_SCOPE=${PROFILE_WORKER_SCOPE:-0}

gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
  --project="$PROJECT" --worker="$WORKER_SCOPE" \
  --command="cd '$REMOTE_REPO' && git fetch origin refactor-bam && \
  git reset --hard && git clean -ffd && git checkout --detach '$COMMIT' && \
  test \"\$(git rev-parse HEAD)\" = '$COMMIT'"

mkdir -p "$ARTIFACT_ROOT" "$LOG_ROOT"
index=0
for exp_class in "$@"; do
  run="Mix${COMMIT:0:7}_${LABEL}_${index}_${exp_class#BamV2GScanLayerMix}"
  remote_run="$REMOTE_ROOT/$run"
  artifact_dir="$ARTIFACT_ROOT/$run"
  collector_log="$LOG_ROOT/${run}-collector.log"
  mkdir -p "$artifact_dir"

  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker="$WORKER_SCOPE" \
    --command="pkill -KILL -f '[M]axText/train.py.*run_name=$run' \
    2>/dev/null || true; rm -rf '$remote_run'; mkdir -p '$remote_run'; \
    sudo rm -f /tmp/libtpu_lockfile"

  /home/lishengping/xd/projects/collect_xplane.sh \
    "$TPU" "$ZONE" "$remote_run/tensorboard" "$artifact_dir" "$PROJECT" 2 \
    >"$collector_log" 2>&1 &
  collector_pid=$!

  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker="$WORKER_SCOPE" \
    --command="export HARDWARE=tpu JAX_TRACEBACK_FILTERING=off; \
    cd '$REMOTE_REPO'; nohup /home/lishengping/miniconda3/bin/python \
    MaxText/train.py MaxText/configs/base.yml base_output_directory='$BASE_OUTPUT' \
    tensorboard_dir='$remote_run/tensorboard' run_name='$run' exp_class='$exp_class' \
    dataset_path='$DATASET' steps=100 \
    > /home/lishengping/train_${run}.log 2>&1 </dev/null &"

  deadline=$((SECONDS + 1200))
  reached_trace=false
  while (( SECONDS < deadline )); do
    status=$(gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
      --project="$PROJECT" --worker=0 \
      --command="if grep -q 'completed step: 15,' \
      /home/lishengping/train_${run}.log 2>/dev/null; then echo DONE; \
      elif grep -Eq '(^| )Traceback \(most recent call last\):|ValueError:|AssertionError:|TypeError:' \
      /home/lishengping/train_${run}.log 2>/dev/null; then echo ERROR; \
      elif pgrep -f '[M]axText/train.py.*run_name=$run' >/dev/null; then echo RUNNING; \
      else echo EXITED; fi" 2>/dev/null | tail -1)
    case "$status" in
      DONE) reached_trace=true; break ;;
      ERROR|EXITED)
        echo "$run $status" >&2
        gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
          --project="$PROJECT" --worker=0 \
          --command="tail -80 /home/lishengping/train_${run}.log" >&2 || true
        break ;;
    esac
    sleep 5
  done

  if $reached_trace; then
    wait "$collector_pid"
  else
    kill "$collector_pid" 2>/dev/null || true
    wait "$collector_pid" 2>/dev/null || true
  fi
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker="$WORKER_SCOPE" \
    --command="pkill -KILL -f '[M]axText/train.py.*run_name=$run' \
    2>/dev/null || true; sudo rm -f /tmp/libtpu_lockfile; \
    grep 'completed step:' /home/lishengping/train_${run}.log | tail -6" || true

  if ! $reached_trace; then
    echo "$run failed before primary trace" >&2
    exit 1
  fi
  index=$((index + 1))
done
