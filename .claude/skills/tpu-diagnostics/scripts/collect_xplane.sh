#!/bin/bash
set -euo pipefail

if (( $# < 4 || $# > 5 )); then
  echo "usage: $0 TPU ZONE REMOTE_PROFILE_DIR DEST_DIR [PROJECT]" >&2
  exit 2
fi

tpu=$1
zone=$2
remote_dir=$3
dest_dir=$4
project=${5:-newproject-1-451205}
mkdir -p "$dest_dir"

while true; do
  state=$(gcloud compute tpus tpu-vm describe "$tpu" --zone="$zone" \
    --project="$project" --format='value(state)' 2>/dev/null || true)
  case "$state" in
    READY|ACTIVE|CREATING) ;;
    *) echo "$tpu disappeared before XPlane completed (state=${state:-missing})" >&2; exit 1 ;;
  esac

  if gcloud compute tpus tpu-vm ssh "$tpu" --zone="$zone" --project="$project" \
      --command="find '$remote_dir' -type f -name '*.xplane.pb' -print -quit 2>/dev/null" \
      2>/dev/null | grep -q '\.xplane\.pb$'; then
    gcloud compute tpus tpu-vm scp --recurse "$tpu:$remote_dir" "$dest_dir" \
      --zone="$zone" --project="$project" >/dev/null
    find "$dest_dir" -type f -name '*.xplane.pb' -print -quit
    exit 0
  fi
  sleep 2
done
