#!/bin/bash
set -euo pipefail

if (( $# < 4 || $# > 7 )); then
  echo "usage: $0 TPU ZONE REMOTE_PROFILE_DIR GCS_PREFIX [PROJECT] [MIN_XPLANES] [WORKER]" >&2
  exit 2
fi

tpu=$1
zone=$2
remote_dir=$3
gcs_prefix=${4%/}
project=${5:-newproject-1-451205}
min_xplanes=${6:-1}
worker=${7:-auto}

if [[ "$worker" == auto ]]; then
  worker_count=$(gcloud compute tpus tpu-vm describe "$tpu" --zone="$zone" \
    --project="$project" --format=json | python3 -c \
    'import json, sys; print(max(1, len(json.load(sys.stdin).get("networkEndpoints", []))))')
  mapfile -t workers < <(seq 0 $((worker_count - 1)))
else
  workers=("$worker")
fi

list_remote_profiles() {
  gcloud compute tpus tpu-vm ssh "$tpu" --zone="$zone" --project="$project" \
    --worker="$1" \
    --command="find '$remote_dir' -type f \\( -name '*.xplane.pb' -o -name '*.trace.json.gz' \\) -size +0c -printf '%s %p\\n' 2>/dev/null | sort" \
    2>/dev/null || true
}

while true; do
  state=$(gcloud compute tpus tpu-vm describe "$tpu" --zone="$zone" \
    --project="$project" --format='value(state)' 2>/dev/null || true)
  case "$state" in
    READY|ACTIVE|CREATING) ;;
    *) echo "$tpu disappeared before XPlane completed (state=${state:-missing})" >&2; exit 1 ;;
  esac

  selected_worker=
  remote_xplanes=
  for candidate in "${workers[@]}"; do
    candidate_profiles=$(list_remote_profiles "$candidate")
    remote_count=$(grep -c '\.xplane\.pb$' <<<"$candidate_profiles" || true)
    remote_trace_count=$(grep -c '\.trace\.json\.gz$' <<<"$candidate_profiles" || true)
    if (( remote_count >= min_xplanes && remote_trace_count >= min_xplanes )); then
      selected_worker=$candidate
      remote_xplanes=$candidate_profiles
      break
    fi
  done
  if [[ -n "$selected_worker" ]]; then
    sleep 2
    stable_xplanes=$(list_remote_profiles "$selected_worker")
    [[ "$remote_xplanes" == "$stable_xplanes" ]] || continue
    remote_host=$(gcloud compute tpus tpu-vm ssh "$tpu" --zone="$zone" --project="$project" \
      --worker="$selected_worker" --command='hostname' 2>/dev/null | tail -n 1)
    [[ -n "$remote_host" ]] || continue
    worker_prefix="$gcs_prefix/$remote_host"
    gcloud compute tpus tpu-vm ssh "$tpu" --zone="$zone" --project="$project" \
      --worker="$selected_worker" \
      --command="gsutil -m rsync -r '$remote_dir' '$worker_prefix'" >/dev/null
    uploaded=$(gsutil ls -l "$worker_prefix/**" 2>/dev/null || true)
    uploaded_xplanes=$(grep -c '\.xplane\.pb$' <<<"$uploaded" || true)
    uploaded_traces=$(grep -c '\.trace\.json\.gz$' <<<"$uploaded" || true)
    (( uploaded_xplanes >= min_xplanes && uploaded_traces >= min_xplanes )) || continue
    echo "selected_worker=$selected_worker"
    grep -E '\.(xplane\.pb|trace\.json\.gz)$' <<<"$uploaded"
    exit 0
  fi
  sleep 2
done
