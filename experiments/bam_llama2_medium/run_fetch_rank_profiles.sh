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
REPO=/home/lishengping/xd/projects/maxtext
LOG_ROOT=/home/lishengping/xd/projects/logs
REMOTE_ROOT=/home/lishengping/xd/profile_outputs/fetch_rank
GCS_ROOT=gs://newproject-1-llm_base_models_us-central1/log/diagnostics/fetch_rank
COLLECTOR=/home/lishengping/xd/projects/collect_xplane.sh
SMOKE=.claude/skills/tpu-diagnostics/scripts/run_train_smoke.sh
COMPILED_SMOKE=~/run_train_smoke_compiled.sh
AOT_ROOT=${AOT_ROOT:-}
AOT_PROFILE_SKIP=${AOT_PROFILE_SKIP:-10}
AOT_PROFILE_PERIOD=${AOT_PROFILE_PERIOD:-1000}
AOT_PROFILE_DURATION=${AOT_PROFILE_DURATION:-5}
PROFILE_STEPS=${PROFILE_STEPS:-20}
DONE_STEP=${PROFILE_DONE_STEP:-15}
MIN_XPLANES=${MIN_XPLANES:-}
if [[ -z "$MIN_XPLANES" ]]; then
  MIN_XPLANES=$([[ -n "$AOT_ROOT" ]] && echo 1 || echo 2)
fi

gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
  --project="$PROJECT" --worker=all --command="cd '$REPO' && \
  git fetch origin refactor-bam && git reset --hard && git clean -ffd && \
  git checkout --detach '$COMMIT' && \
  test \"\$(git rev-parse HEAD)\" = \"\$(git rev-parse '$COMMIT^{commit}')\"" \
  </dev/null

mkdir -p "$LOG_ROOT"
index=0
for exp_class in "$@"; do
  run="FetchRank${COMMIT:0:7}_${LABEL}_${index}_${exp_class}"
  remote_run="$REMOTE_ROOT/$run"
  train_log="/home/lishengping/train_${run}.log"
  collector_log="$LOG_ROOT/${run}-collector.log"
  gcs_run="$GCS_ROOT/${COMMIT:0:7}/$LABEL/$run"

  if [[ -n "$AOT_ROOT" ]]; then
    compiled="/tmp/${exp_class}.pickle"
    gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
      --project="$PROJECT" --worker=all --command="gsutil -q cp \
      '$AOT_ROOT/$exp_class.pickle' '$compiled'" </dev/null
    smoke_command="env PROFILE_SKIP='$AOT_PROFILE_SKIP' \
      PROFILE_PERIOD='$AOT_PROFILE_PERIOD' PROFILE_DURATION='$AOT_PROFILE_DURATION' \
      '$COMPILED_SMOKE' '$exp_class' '$run' '$compiled' '$PROFILE_STEPS'"
  else
    smoke_command="'$SMOKE' '$exp_class' '$run' '$PROFILE_STEPS'"
  fi

  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker=all --command="rm -rf '$remote_run'; \
    mkdir -p '$remote_run'; sudo rm -f /tmp/libtpu_lockfile; cd '$REPO'; \
    nohup env SMOKE_OUTPUT='$remote_run' $smoke_command \
    >'$train_log' 2>&1 </dev/null &" </dev/null

  "$COLLECTOR" "$TPU" "$ZONE" "$remote_run/tensorboard" "$gcs_run" \
    "$PROJECT" "$MIN_XPLANES" auto </dev/null >"$collector_log" 2>&1 &
  collector_pid=$!
  echo "$run launch_utc=$(date -u +%FT%TZ)"

  deadline=$((SECONDS + 2400))
  first=false
  done=false
  while (( SECONDS < deadline )); do
    state=$(gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
      --project="$PROJECT" --worker=0 --command="if grep -q \
      'completed step: $DONE_STEP,' '$train_log' 2>/dev/null; then echo DONE; \
      elif grep -q 'completed step: 0,' '$train_log' 2>/dev/null; then echo FIRST; \
      elif pgrep -f '[M]axText/train.py.*run_name=$run' >/dev/null; then echo RUNNING; \
      elif grep -Eq 'ValueError:|AssertionError:|TypeError:|Traceback' \
      '$train_log' 2>/dev/null; then echo ERROR; else echo EXITED; fi" \
      </dev/null 2>/dev/null | tail -1)
    case "$state" in
      DONE) done=true; break ;;
      FIRST)
        if ! $first; then
          first=true
          echo "$run first_step_utc=$(date -u +%FT%TZ)"
        fi
        ;;
      ERROR|EXITED)
        echo "$run $state" >&2
        gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
          --project="$PROJECT" --worker=0 --command="tail -100 '$train_log'" \
          </dev/null >&2 || true
        break
        ;;
    esac
    sleep 5
  done

  if $done; then
    wait "$collector_pid"
  else
    kill "$collector_pid" 2>/dev/null || true
    wait "$collector_pid" 2>/dev/null || true
  fi
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker=all --command="pkill -KILL -f \
    '[M]axText/train.py.*run_name=$run' 2>/dev/null || true; \
    sudo rm -f /tmp/libtpu_lockfile" </dev/null
  gcloud compute tpus tpu-vm ssh --internal-ip "$TPU" --zone="$ZONE" \
    --project="$PROJECT" --worker=0 --command="grep 'completed step:' \
    '$train_log' | tail -6" </dev/null || true
  $done || exit 1
  index=$((index + 1))
done
