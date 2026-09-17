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
REMOTE_ROOT=/home/lishengping/xd/profile_outputs/local_qk_rank
LOG_ROOT=/home/lishengping/xd/projects/logs
GCS_ROOT=gs://newproject-1-llm_base_models_us-central1/log/diagnostics/local_qk_rank
COLLECTOR=/home/lishengping/xd/projects/collect_xplane.sh
SMOKE=.claude/skills/tpu-diagnostics/scripts/run_train_smoke.sh
PROFILE_STEPS=${PROFILE_STEPS:-20}
PROFILE_DONE_STEP=${PROFILE_DONE_STEP:-15}
PROFILE_TRACE_COUNT=${PROFILE_TRACE_COUNT:-2}
AOT_GCS_ROOT=${AOT_GCS_ROOT:-}
AOT_PREPARED=${AOT_PREPARED:-false}
AOT_ALREADY_RUNNING=${AOT_ALREADY_RUNNING:-false}
AOT_SMOKE_LOCAL=${AOT_SMOKE_LOCAL:-/home/lishengping/xd/projects/run_train_smoke_compiled.sh}
AOT_SMOKE_REMOTE=/tmp/run_train_smoke_compiled.sh

if [[ "$AOT_PREPARED" != true ]]; then
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker=all \
    --command="cd '$REMOTE_REPO' && git fetch origin refactor-bam && \
    git reset --hard && git clean -ffd && git checkout --detach '$COMMIT' && \
    test \"\$(git rev-parse HEAD)\" = '$COMMIT'" </dev/null
fi

if [[ -n "$AOT_GCS_ROOT" && "$AOT_PREPARED" != true ]]; then
  gcloud compute tpus tpu-vm scp --internal-ip "$AOT_SMOKE_LOCAL" \
    "$TPU:$AOT_SMOKE_REMOTE" --zone="$ZONE" --project="$PROJECT" \
    --worker=all </dev/null
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker=all \
    --command="chmod 755 '$AOT_SMOKE_REMOTE'" </dev/null
fi

mkdir -p "$LOG_ROOT"
index=0
for exp_class in "$@"; do
  run="LQRank${COMMIT:0:7}_${LABEL}_${index}_${exp_class}"
  remote_run="$REMOTE_ROOT/$run"
  gcs_run="$GCS_ROOT/${COMMIT:0:7}/$LABEL/$run"
  collector_log="$LOG_ROOT/${run}-collector.log"
  train_log="/home/lishengping/train_${run}.log"
  smoke="$SMOKE"
  smoke_args="'$exp_class' '$run' '$PROFILE_STEPS'"

  if [[ -n "$AOT_GCS_ROOT" ]]; then
    compiled="/tmp/${exp_class}.pickle"
    artifact="${AOT_GCS_ROOT%/}/${exp_class}.pickle"
    if [[ "$AOT_PREPARED" != true ]]; then
      gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
        --project="$PROJECT" --worker=all \
        --command="gsutil -q cp '$artifact' '$compiled'" </dev/null
    fi
    smoke="$AOT_SMOKE_REMOTE"
    smoke_args="'$exp_class' '$run' '$compiled' '$PROFILE_STEPS'"
  fi

  if [[ "$AOT_ALREADY_RUNNING" != true ]]; then
    gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
      --project="$PROJECT" --worker=all \
      --command="pkill -KILL -f '[M]axText/train.py.*run_name=$run' \
      2>/dev/null || true; rm -rf '$remote_run'; mkdir -p '$remote_run'; \
      sudo rm -f /tmp/libtpu_lockfile" </dev/null
  fi

  "$COLLECTOR" "$TPU" "$ZONE" "$remote_run/tensorboard" \
    "$gcs_run" "$PROJECT" "$PROFILE_TRACE_COUNT" auto \
    </dev/null >"$collector_log" 2>&1 &
  collector_pid=$!

  if [[ "$AOT_ALREADY_RUNNING" != true ]]; then
    gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
      --project="$PROJECT" --worker=all \
      --command="cd '$REMOTE_REPO'; nohup env SMOKE_OUTPUT='$remote_run' \
      '$smoke' $smoke_args \
      >'$train_log' 2>&1 </dev/null &" </dev/null
  fi
  echo "$run launch_utc=$(date -u +%FT%TZ)"

  deadline=$((SECONDS + 1800))
  reached_trace=false
  saw_first_step=false
  while (( SECONDS < deadline )); do
    status=$(gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
      --project="$PROJECT" --worker=0 \
      --command="if grep -q 'completed step: $PROFILE_DONE_STEP,' '$train_log' 2>/dev/null; \
      then echo DONE; elif grep -q 'completed step: 0,' '$train_log' 2>/dev/null; \
      then echo FIRST; elif pgrep -f '[M]axText/train.py.*run_name=$run' \
      >/dev/null; then echo RUNNING; elif grep -Eq \
      'ValueError:|AssertionError:|TypeError:|RESOURCE_EXHAUSTED' \
      '$train_log' 2>/dev/null; then echo ERROR; else echo EXITED; fi" \
      </dev/null 2>/dev/null | tail -1)
    case "$status" in
      DONE) reached_trace=true; break ;;
      FIRST)
        if ! $saw_first_step; then
          saw_first_step=true
          echo "$run first_step_utc=$(date -u +%FT%TZ)"
        fi
        ;;
      ERROR|EXITED)
        echo "$run $status" >&2
        gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
          --project="$PROJECT" --worker=0 \
          --command="tail -100 '$train_log'" </dev/null >&2 || true
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
    --project="$PROJECT" --worker=all \
    --command="pkill -KILL -f '[M]axText/train.py.*run_name=$run' \
    2>/dev/null || true; sudo rm -f /tmp/libtpu_lockfile" </dev/null
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker=0 \
    --command="grep 'completed step:' '$train_log' | tail -6" </dev/null || true

  $reached_trace || exit 1
  index=$((index + 1))
done
