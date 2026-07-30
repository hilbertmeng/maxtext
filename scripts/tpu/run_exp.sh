#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
AUTO_TRAIN_SCRIPT="${SCRIPT_DIR}/auto_train_arc_maxtext.sh"

usage() {
  cat <<'EOF'
Usage:
  run_exp.sh <plan|install+train|train|status|stop-controller|delete-tpu> --config FILE

The config is a trusted shell env file; see profiles/template.env.
The default action is plan. No TPU is created unless install+train or train is explicit.
EOF
}

MODE="plan"
CONFIG=""
if (($# > 0)) && [[ "$1" != --* ]]; then
  MODE="$1"
  shift
fi
while (($# > 0)); do
  case "$1" in
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CONFIG" ]]; then
  usage >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 2
fi

# The profile is an operator-controlled shell env file.
# shellcheck disable=SC1090
source "$CONFIG"

: "${ID:?ID must be set in config}"
: "${EXP:?EXP must be set in config}"
: "${PROJECT_ID:?PROJECT_ID must be set in config}"
: "${TPU_TYPE:?TPU_TYPE must be set in config}"
: "${ZONE:?ZONE must be set in config}"

RUN_NAME="${RUN_NAME:-$EXP}"
RUNTIME_VERSION="${RUNTIME_VERSION:-v2-alpha-tpuv5}"
SUFFIX="${SUFFIX:-v5p}"
DEBUG="${DEBUG:-false}"
TRAIN_COMPILE_SUFFIX="${TRAIN_COMPILE_SUFFIX:-}"
MAXTEXT_SYNC_REF="${MAXTEXT_SYNC_REF:-refactor-arc}"
MAXTEXT_EXPECTED_COMMIT="${MAXTEXT_EXPECTED_COMMIT:-}"
TPU_WORK_DIR="${TPU_WORK_DIR:-/home/lishengping/projects/maxtext}"
CONTROLLER_LOG_DIR="${CONTROLLER_LOG_DIR:-/home/lishengping/mengqy/projects/logs}"
TPU_NAME="${TPU_NAME:-llada-${TPU_TYPE}-${ID}-maxtext}"
BRANCH="arc"

if [[ ! "$EXP" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "EXP contains unsafe characters: $EXP" >&2
  exit 2
fi
if [[ ! "$MAXTEXT_SYNC_REF" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "MAXTEXT_SYNC_REF contains unsafe characters: $MAXTEXT_SYNC_REF" >&2
  exit 2
fi
if [[ -n "$MAXTEXT_EXPECTED_COMMIT" ]] &&
   [[ ! "$MAXTEXT_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "MAXTEXT_EXPECTED_COMMIT must be a full 40-character lowercase SHA-1" >&2
  exit 2
fi

PID_FILE="${CONTROLLER_LOG_DIR}/${EXP}.pid"
LOG_FILE="${CONTROLLER_LOG_DIR}/${EXP}.log"

print_plan() {
  cat <<EOF
MODE=$MODE
EXP=$EXP
RUN_NAME=$RUN_NAME
TPU_NAME=$TPU_NAME
TPU_TYPE=$TPU_TYPE
ZONE=$ZONE
PROJECT_ID=$PROJECT_ID
RUNTIME_VERSION=$RUNTIME_VERSION
MAXTEXT_SYNC_REF=$MAXTEXT_SYNC_REF
MAXTEXT_EXPECTED_COMMIT=${MAXTEXT_EXPECTED_COMMIT:-<floating-ref>}
TPU_WORK_DIR=$TPU_WORK_DIR
CONTROLLER_LOG_DIR=$CONTROLLER_LOG_DIR
AUTO_TRAIN_SCRIPT=$AUTO_TRAIN_SCRIPT
EOF
}

case "$MODE" in
  plan)
    print_plan
    ;;
  install+train|train)
    mkdir -p "$CONTROLLER_LOG_DIR"
    if [[ -f "$PID_FILE" ]]; then
      old_pid="$(<"$PID_FILE")"
      if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "Controller already running: PID $old_pid ($PID_FILE)" >&2
        exit 1
      fi
    fi
    # Historical install+train and train modes both kept INSTALL=true. The
    # controller performs a cheap environment probe and skips installation
    # when the TPU worker is already prepared; keeping this true also makes a
    # later TPU recreation recoverable.
    install="true"
    export RUN_NAME MAXTEXT_SYNC_REF MAXTEXT_EXPECTED_COMMIT TPU_WORK_DIR
    export CONTROLLER_LOG_DIR
    printf 'Y\n' |
      bash "$AUTO_TRAIN_SCRIPT" \
        "$install" true "$TPU_TYPE" "$ZONE" "$TPU_NAME" "$PROJECT_ID" "$EXP" \
        "$DEBUG" true true "$SUFFIX" "$RUNTIME_VERSION" "$BRANCH" \
        "$TRAIN_COMPILE_SUFFIX" >"$LOG_FILE" 2>&1 &
    launcher_pid=$!
    echo "Controller launcher PID: $launcher_pid"
    echo "Controller PID file: $PID_FILE"
    echo "Log: $LOG_FILE"
    ;;
  status)
    if [[ -f "$PID_FILE" ]]; then
      controller_pid="$(<"$PID_FILE")"
      if [[ "$controller_pid" =~ ^[0-9]+$ ]] && kill -0 "$controller_pid" 2>/dev/null; then
        echo "Controller: running (PID $controller_pid)"
      else
        echo "Controller: stale PID file ($controller_pid)"
      fi
    else
      echo "Controller: no PID file"
    fi
    gcloud alpha compute tpus describe "$TPU_NAME" \
      --zone="$ZONE" --project="$PROJECT_ID" \
      --format='value(state)' 2>/dev/null || true
    ;;
  stop-controller)
    if [[ ! -f "$PID_FILE" ]]; then
      echo "No PID file: $PID_FILE"
      exit 0
    fi
    controller_pid="$(<"$PID_FILE")"
    if [[ ! "$controller_pid" =~ ^[0-9]+$ ]]; then
      echo "Invalid PID file: $PID_FILE" >&2
      exit 1
    fi
    if kill -0 "$controller_pid" 2>/dev/null; then
      kill "$controller_pid"
      echo "Stopped controller PID $controller_pid"
    else
      echo "Controller PID $controller_pid is not running"
    fi
    ;;
  delete-tpu)
    if [[ "${CONFIRM_DELETE_TPU:-}" != "yes-really-delete" ]]; then
      echo "Refusing deletion. Set CONFIRM_DELETE_TPU=yes-really-delete." >&2
      exit 1
    fi
    gcloud compute tpus tpu-vm delete "$TPU_NAME" \
      --zone="$ZONE" --project="$PROJECT_ID" --quiet
    gcloud alpha compute tpus queued-resources delete "$TPU_NAME" \
      --zone="$ZONE" --project="$PROJECT_ID" --quiet
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
