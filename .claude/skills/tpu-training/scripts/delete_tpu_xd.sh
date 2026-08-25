#!/usr/bin/env bash
set -u

tpu=${1:?usage: delete_tpu_xd.sh TPU_NAME ZONE PROJECT_ID}
zone=${2:?usage: delete_tpu_xd.sh TPU_NAME ZONE PROJECT_ID}
project=${3:?usage: delete_tpu_xd.sh TPU_NAME ZONE PROJECT_ID}
account=${TPU_ACCOUNT:-626151558586-compute@developer.gserviceaccount.com}
attempts=${RELEASE_ATTEMPTS:-30}
wait_seconds=${RELEASE_WAIT_SECONDS:-10}
root=${TPU_AG_ROOT:-/home/lishengping/xd/projects}
pid_file=${TPU_CREATE_PID_FILE:-$root/logs/${tpu}-create.pid}
gcloud_base=(gcloud --account="$account")

stop_creator() {
  [[ -s "$pid_file" ]] || return 0
  local pid cmdline
  pid=$(<"$pid_file")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    if [[ "$cmdline" == *"create_standalone_tpu.sh"* && "$cmdline" == *"$tpu"* ]]; then
      echo "Stopping creator pid=$pid for TPU=$tpu"
      kill "$pid" 2>/dev/null || true
      for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
      done
      kill -KILL "$pid" 2>/dev/null || true
    else
      echo "ERROR: refusing to stop pid=$pid with unexpected command: $cmdline" >&2
      return 1
    fi
  fi
  rm -f "$pid_file"
}

describe_resource() {
  local kind=$1 output rc
  if [[ "$kind" == node ]]; then
    output=$("${gcloud_base[@]}" compute tpus tpu-vm describe "$tpu" \
      --zone="$zone" --project="$project" --format='value(state)' 2>&1)
  else
    output=$("${gcloud_base[@]}" alpha compute tpus queued-resources describe "$tpu" \
      --zone="$zone" --project="$project" --format='value(state.state)' 2>&1)
  fi
  rc=$?
  if (( rc == 0 )); then
    printf '%s' "$output"
    return 0
  fi
  [[ "$output" == *NOT_FOUND* ]] && return 0
  echo "ERROR: failed to inspect $kind for $tpu: $output" >&2
  return 1
}

stop_creator || exit 1
for ((attempt=1; attempt<=attempts; attempt++)); do
  node_state=$(describe_resource node) || { sleep "$wait_seconds"; continue; }
  queued_state=$(describe_resource queue) || { sleep "$wait_seconds"; continue; }
  if [[ -z "$node_state" && -z "$queued_state" ]]; then
    echo "Verified absent: TPU=$tpu queued-resource=$tpu creator=absent"
    exit 0
  fi

  echo "Release attempt $attempt/$attempts: node=${node_state:-absent} queue=${queued_state:-absent}"
  if [[ -n "$node_state" ]]; then
    "${gcloud_base[@]}" compute tpus tpu-vm delete "$tpu" \
      --zone="$zone" --project="$project" --quiet || true
  elif [[ -n "$queued_state" ]]; then
    "${gcloud_base[@]}" alpha compute tpus queued-resources delete "$tpu" \
      --zone="$zone" --project="$project" --quiet || true
  fi
  sleep "$wait_seconds"
done

node_state=$(describe_resource node 2>/dev/null || echo unknown)
queued_state=$(describe_resource queue 2>/dev/null || echo unknown)
echo "ERROR: release unverified: node=${node_state:-absent} queue=${queued_state:-absent}" >&2
exit 1
