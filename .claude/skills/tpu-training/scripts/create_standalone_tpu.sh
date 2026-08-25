#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 TPU_NAME ACCELERATOR_TYPE ZONE [INSTALL_SCRIPT]" >&2
  exit 2
fi

tpu=$1
accelerator=$2
zone=$3
install_script=${4:-}
project=${TPU_PROJECT:-newproject-1-451205}
account=${TPU_ACCOUNT:-626151558586-compute@developer.gserviceaccount.com}
configuration=${TPU_GCLOUD_CONFIGURATION:-xd-tpu}
poll_seconds=${TPU_CREATE_POLL_SECONDS:-20}
heartbeat_seconds=${TPU_CREATE_HEARTBEAT_SECONDS:-300}
root=${TPU_AG_ROOT:-/home/lishengping/xd/projects}
preflight=${TPU_AG_PREFLIGHT:-$root/tpu_ag_preflight.sh}
pid_file=${TPU_CREATE_PID_FILE:-$root/logs/${tpu}-create.pid}

case "$accelerator" in
  v5p-*) runtime=v2-alpha-tpuv5 ;;
  v6e-*) runtime=v2-alpha-tpuv6e ;;
  *) echo "ERROR: standalone queued-resource supports v5p-* or v6e-*: $accelerator" >&2; exit 2 ;;
esac

[[ "$tpu" =~ ^[a-z0-9-]+$ ]] || { echo "ERROR: invalid TPU name: $tpu" >&2; exit 2; }
[[ "$zone" =~ ^[a-z0-9-]+$ ]] || { echo "ERROR: invalid zone: $zone" >&2; exit 2; }
[[ -x "$preflight" ]] || { echo "ERROR: missing preflight: $preflight" >&2; exit 1; }
"$preflight"
export CLOUDSDK_ACTIVE_CONFIG_NAME="$configuration"

mkdir -p "$(dirname "$pid_file")"
if [[ -s "$pid_file" ]]; then
  old_pid=$(<"$pid_file")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "ERROR: creator already active for $tpu: pid=$old_pid" >&2
    exit 1
  fi
fi
printf '%s\n' "$$" >"$pid_file"
cleanup_pid() {
  if [[ -f "$pid_file" ]] && [[ "$(<"$pid_file")" == "$$" ]]; then
    rm -f "$pid_file"
  fi
}
trap cleanup_pid EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

gcloud_base=(gcloud --configuration="$configuration" --account="$account")

node_state() {
  "${gcloud_base[@]}" compute tpus tpu-vm describe "$tpu" --zone="$zone" \
    --project="$project" --format='value(state)' 2>/dev/null || true
}

queue_state() {
  "${gcloud_base[@]}" alpha compute tpus queued-resources describe "$tpu" --zone="$zone" \
    --project="$project" --format='value(state.state)' 2>/dev/null || true
}

node=$(node_state)
queue=$(queue_state)
if [[ -z "$node" && -z "$queue" ]]; then
  echo "SUBMIT TPU=$tpu TYPE=$accelerator ZONE=$zone"
  set +e
  create_output=$("${gcloud_base[@]}" alpha compute tpus queued-resources create "$tpu" \
    --node-id="$tpu" --project="$project" --zone="$zone" \
    --accelerator-type="$accelerator" --runtime-version="$runtime" \
    --service-account="$account" --best-effort 2>&1)
  create_status=$?
  set -e
  if (( create_status != 0 )); then
    node=$(node_state)
    queue=$(queue_state)
    if [[ -z "$node" && -z "$queue" ]]; then
      printf 'ERROR: queued-resource submission failed:\n%s\n' "${create_output: -4096}" >&2
      exit "$create_status"
    fi
    echo "ADOPT existing TPU/queue after concurrent submission"
  fi
else
  echo "ADOPT TPU=$tpu node=${node:-absent} queue=${queue:-absent}"
fi

last_state=
last_heartbeat=0
while true; do
  node=$(node_state)
  queue=$(queue_state)
  combined="node=${node:-absent} queue=${queue:-absent}"
  now=$(date +%s)
  if [[ "$combined" != "$last_state" ]] || (( now - last_heartbeat >= heartbeat_seconds )); then
    printf '%s %s\n' "$(date -u +%FT%TZ)" "$combined"
    last_state=$combined
    last_heartbeat=$now
  fi

  if [[ "$node" == READY ]]; then
    break
  fi
  case "$queue" in
    FAILED|SUSPENDED|CANCELLED)
      echo "ERROR: terminal queued-resource: $combined" >&2
      exit 1
      ;;
  esac
  if [[ -z "$node" && -z "$queue" ]]; then
    echo "ERROR: TPU and queued-resource disappeared after submission" >&2
    exit 1
  fi
  sleep "$poll_seconds"
done

echo "READY TPU=$tpu"
if [[ -n "$install_script" ]]; then
  [[ -f "$install_script" ]] || { echo "ERROR: install script not found: $install_script" >&2; exit 1; }
  install_name=$(basename "$install_script")
  parent_zone=${zone%-*}
  "${gcloud_base[@]}" compute tpus tpu-vm scp "$install_script" "$tpu:~/$install_name" \
    --zone="$zone" --worker=all --project="$project"
  "${gcloud_base[@]}" compute tpus tpu-vm ssh "$tpu" --zone="$zone" --worker=all \
    --project="$project" --command="bash ~/$install_name '$parent_zone'"
  echo "INSTALL_OK TPU=$tpu"
fi
