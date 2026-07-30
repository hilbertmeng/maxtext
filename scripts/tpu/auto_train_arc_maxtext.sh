#!/usr/bin/env bash
set -u

if (($# < 14)); then
  echo "Expected 14 positional arguments; got $#" >&2
  exit 2
fi

INSTALL=$1
TRAIN=$2
TPU_TYPE=$3
ZONE=$4
TPU_NAME=$5
PROJECT_ID=$6
EXP=$7
DEBUG=$8
NotINSTALLED=$9
NotTraining=${10}
SUFFIX=${11}
CREATE_ARGS="--runtime-version ${12}"
BRANCH=${13}
TrainCompile=${14}

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
INSTALL_SCRIPT="${MAXTEXT_INSTALL_SCRIPT:-${SCRIPT_DIR}/install_0812_v5p_mqy_maxtext_jax081.sh}"
CONTROLLER_LOG_DIR="${CONTROLLER_LOG_DIR:-/home/lishengping/mengqy/projects/logs}"
TPU_WORK_DIR="${TPU_WORK_DIR:-/home/lishengping/projects/maxtext}"
MAXTEXT_SYNC_REF="${MAXTEXT_SYNC_REF:-refactor-arc}"
MAXTEXT_EXPECTED_COMMIT="${MAXTEXT_EXPECTED_COMMIT:-}"

BUCKET_PREFIX="${BUCKET_PREFIX:-newproject-1-}"
BUCKET_ZONE="${ZONE%-?}"
if [[ "$BUCKET_ZONE" == "us-central1" ]]; then
  BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-gs://newproject-1-llm_base_models_us-central1/log/}"
else
  BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-gs://${BUCKET_PREFIX}llm_projects_${BUCKET_ZONE}/log/}"
fi

RUN_NAME="${RUN_NAME:-$EXP}"
GCLOUD_SHORT_TIMEOUT="${GCLOUD_SHORT_TIMEOUT:-300s}"
GCLOUD_TRAIN_TIMEOUT="${GCLOUD_TRAIN_TIMEOUT:-300s}"
GCLOUD_INSTALL_TIMEOUT="${GCLOUD_INSTALL_TIMEOUT:-1800s}"
TPU_SSH_FLAGS="${TPU_SSH_FLAGS:---internal-ip}"

if [[ ! -f "$INSTALL_SCRIPT" ]]; then
  echo "Install script not found: $INSTALL_SCRIPT" >&2
  exit 2
fi
if [[ ! "$MAXTEXT_SYNC_REF" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "Unsafe MAXTEXT_SYNC_REF: $MAXTEXT_SYNC_REF" >&2
  exit 2
fi
if [[ -n "$MAXTEXT_EXPECTED_COMMIT" ]] &&
   [[ ! "$MAXTEXT_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "MAXTEXT_EXPECTED_COMMIT must be a full lowercase SHA-1" >&2
  exit 2
fi

mkdir -p "$CONTROLLER_LOG_DIR"
echo "$$" >"${CONTROLLER_LOG_DIR}/${EXP}.pid"

cat <<EOF
######################
TPU_TYPE: $TPU_TYPE
CREATE_ARGS: $CREATE_ARGS
ZONE: $ZONE
PROJECT_ID: $PROJECT_ID
INSTALL: $INSTALL
TRAIN: $TRAIN
SUFFIX: $SUFFIX
DEBUG: $DEBUG
NotINSTALLED: $NotINSTALLED
NotTraining: $NotTraining
TPU_NAME: $TPU_NAME
EXP: $EXP
RUN_NAME: $RUN_NAME
BRANCH: $BRANCH
TrainCompile: $TrainCompile
MAXTEXT_SYNC_REF: $MAXTEXT_SYNC_REF
MAXTEXT_EXPECTED_COMMIT: ${MAXTEXT_EXPECTED_COMMIT:-<floating-ref>}
TPU_WORK_DIR: $TPU_WORK_DIR
######################
EOF

read -r -p "Are you sure to run the exp? [y/N] " response
case "$response" in
  [yY][eE][sS]|[yY]) echo "Start running" ;;
  *) echo "aborted"; exit 0 ;;
esac

FLAG=0
COUNT=0

while true; do
  tpu_status="$(
    gcloud alpha compute tpus describe "$TPU_NAME" \
      --zone="$ZONE" --project="$PROJECT_ID" \
      --format="value[terminator=''](state)" 2>/dev/null || true
  )"
  echo "TPU status is ${tpu_status}"

  if [[ "$tpu_status" == "READY" ]]; then
    FLAG=0
    echo "$(date) Start training......"
    if ! "$NotTraining"; then
      echo "Training already launched; sleeping before the next TPU health check."
      sleep 60
      continue
    fi

    if "$INSTALL"; then
      # shellcheck disable=SC2086
      ENV_OK="$(
        timeout "$GCLOUD_SHORT_TIMEOUT" \
          gcloud compute tpus tpu-vm ssh $TPU_SSH_FLAGS "$TPU_NAME" \
          --zone="$ZONE" --project="$PROJECT_ID" --worker=0 \
          --command="/home/lishengping/miniconda3/bin/python -c 'import jax; print(\"ok\")' 2>/dev/null" \
          2>/dev/null || true
      )"
      if [[ "$ENV_OK" != "ok" ]]; then
        echo "Environment not found; installing from versioned script."
        # shellcheck disable=SC2086
        if ! timeout "$GCLOUD_SHORT_TIMEOUT" \
          gcloud compute tpus tpu-vm scp $TPU_SSH_FLAGS "$INSTALL_SCRIPT" \
          "${TPU_NAME}:~/install_0812_v5p_mqy_maxtext_jax081.sh" \
          --zone="$ZONE" --project="$PROJECT_ID" --worker=all; then
          echo "Failed to copy install script; retrying after TPU state check."
          sleep 30
          continue
        fi
        # shellcheck disable=SC2086
        if ! timeout "$GCLOUD_INSTALL_TIMEOUT" \
          gcloud compute tpus tpu-vm ssh $TPU_SSH_FLAGS "$TPU_NAME" \
          --zone="$ZONE" --project="$PROJECT_ID" --worker=all \
          --command="bash ~/install_0812_v5p_mqy_maxtext_jax081.sh '${ZONE%-?}' '$TPU_TYPE' '$BUCKET_PREFIX' 2>&1 | tee ~/install.log; /home/lishengping/miniconda3/bin/pip install google-cloud-storage"; then
          echo "Install command failed or timed out; retrying after TPU state check."
          sleep 30
          continue
        fi
      else
        echo "Environment already installed; skipping."
      fi
    fi

    if "$TRAIN" && "$NotTraining"; then
      if [[ "$BRANCH" == "arc" ]]; then
        if [[ -n "$MAXTEXT_EXPECTED_COMMIT" ]]; then
          echo "Syncing pinned MaxText commit: $MAXTEXT_EXPECTED_COMMIT"
          sync_command="cd '$TPU_WORK_DIR' && git fetch origin '$MAXTEXT_SYNC_REF' && git checkout --detach '$MAXTEXT_EXPECTED_COMMIT' && actual_commit=\$(git rev-parse HEAD) && echo EXPECTED_COMMIT='$MAXTEXT_EXPECTED_COMMIT' ACTUAL_COMMIT=\$actual_commit && test \"\$actual_commit\" = '$MAXTEXT_EXPECTED_COMMIT'"
        else
          echo "Syncing floating MaxText ref with fast-forward-only pull: $MAXTEXT_SYNC_REF"
          sync_command="cd '$TPU_WORK_DIR' && git fetch origin '$MAXTEXT_SYNC_REF' && (git checkout '$MAXTEXT_SYNC_REF' || git checkout -b '$MAXTEXT_SYNC_REF' --track 'origin/$MAXTEXT_SYNC_REF') && git pull --ff-only origin '$MAXTEXT_SYNC_REF'"
        fi
        # shellcheck disable=SC2086
        if ! timeout "$GCLOUD_SHORT_TIMEOUT" \
          gcloud compute tpus tpu-vm ssh $TPU_SSH_FLAGS "$TPU_NAME" \
          --project="$PROJECT_ID" --zone="$ZONE" --worker=all \
          --command="$sync_command"; then
          echo "Git sync failed or timed out; retrying after TPU state check."
          sleep 30
          continue
        fi
      fi

      echo "Code sync complete."
      # shellcheck disable=SC2086
      if ! timeout "$GCLOUD_SHORT_TIMEOUT" \
        gcloud compute tpus tpu-vm ssh $TPU_SSH_FLAGS "$TPU_NAME" \
        --zone="$ZONE" --project="$PROJECT_ID" --worker=all \
        --command='killall train.py 2>/dev/null || true; pids=$(sudo lsof -t /dev/vfio/0 2>/dev/null || true); if [ -n "$pids" ]; then sudo kill -9 $pids || true; fi; sudo rm -f /tmp/libtpu_lockfile; sudo mkdir -p /tmp/tpu_logs; sudo chmod +777 -R /tmp/tpu_logs/'; then
        echo "TPU cleanup failed or timed out; retrying after TPU state check."
        sleep 30
        continue
      fi
      sleep 2

      if [[ -z "$TrainCompile" ]]; then
        echo "Attempting to restore compiled model cache."
        # shellcheck disable=SC2086
        timeout "$GCLOUD_SHORT_TIMEOUT" \
          gcloud compute tpus tpu-vm ssh $TPU_SSH_FLAGS "$TPU_NAME" \
          --zone="$ZONE" --project="$PROJECT_ID" --worker=all \
          --command="gsutil cp '${BASE_OUTPUT_DIR}${EXP}/${EXP}.pkl' '$TPU_WORK_DIR/'" || true
      fi

      train_log="/home/lishengping/train_${EXP}.log"
      train_command="export HARDWARE=tpu; export JAX_TRACEBACK_FILTERING=off; cd '$TPU_WORK_DIR'; /home/lishengping/miniconda3/bin/python 'MaxText/train${TrainCompile}.py' MaxText/configs/base.yml base_output_directory='$BASE_OUTPUT_DIR' run_name='$RUN_NAME' exp_class='$EXP' >'$train_log' 2>&1 &"
      # shellcheck disable=SC2086
      if ! timeout "$GCLOUD_TRAIN_TIMEOUT" \
        gcloud compute tpus tpu-vm ssh $TPU_SSH_FLAGS "$TPU_NAME" \
        --project="$PROJECT_ID" --zone="$ZONE" --worker=all \
        --command="$train_command"; then
        echo "Training launch failed or timed out; retrying after TPU state check."
        sleep 30
        continue
      fi
      NotTraining=false
      COUNT=0
    fi

  elif [[ "$tpu_status" == "CREATING" ]]; then
    FLAG=1
    COUNT=0
    echo "TPU is creating......"
    sleep 30

  elif [[ "$FLAG" == 0 || "$COUNT" -ge 4 ]]; then
    echo "TPU does not exist; recreating queued resource."
    gcloud compute tpus tpu-vm delete "$TPU_NAME" \
      --zone="$ZONE" --project="$PROJECT_ID" --quiet || true
    gcloud alpha compute tpus queued-resources delete "$TPU_NAME" \
      --zone="$ZONE" --project="$PROJECT_ID" --quiet || true
    gcloud alpha compute tpus queued-resources create "$TPU_NAME" \
      --node-id="$TPU_NAME" --project="$PROJECT_ID" --zone="$ZONE" \
      --accelerator-type="$TPU_TYPE" $CREATE_ARGS \
      --service-account=626151558586-compute@developer.gserviceaccount.com \
      --best-effort
    NotTraining=true
    NotINSTALLED=true
    sleep 60
    FLAG=1

  elif [[ -z "$tpu_status" && "$COUNT" -lt 4 ]]; then
    COUNT=$((COUNT + 1))
    echo "Empty status retry count: $COUNT"
    sleep 15

  else
    FLAG=1
    echo "TPU is not ready......"
    sleep 30
  fi
done
