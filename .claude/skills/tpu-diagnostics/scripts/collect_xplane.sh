#!/bin/bash
set -euo pipefail

if (( $# < 4 || $# > 7 )); then
  echo "usage: $0 TPU ZONE REMOTE_PROFILE_DIR DEST_DIR [PROJECT] [MIN_XPLANES] [WORKER]" >&2
  exit 2
fi

tpu=$1
zone=$2
remote_dir=$3
dest_dir=$4
project=${5:-newproject-1-451205}
min_xplanes=${6:-1}
worker=${7:-0}
mkdir -p "$dest_dir"

while true; do
  state=$(gcloud compute tpus tpu-vm describe "$tpu" --zone="$zone" \
    --project="$project" --format='value(state)' 2>/dev/null || true)
  case "$state" in
    READY|ACTIVE|CREATING) ;;
    *) echo "$tpu disappeared before XPlane completed (state=${state:-missing})" >&2; exit 1 ;;
  esac

  remote_xplanes=$(gcloud compute tpus tpu-vm ssh "$tpu" --zone="$zone" --project="$project" \
      --worker="$worker" \
      --command="find '$remote_dir' -type f -name '*.xplane.pb' -size +0c -printf '%s %p\\n' 2>/dev/null | sort" \
      2>/dev/null || true)
  remote_count=$(grep -c '\.xplane\.pb$' <<<"$remote_xplanes" || true)
  if (( remote_count >= min_xplanes )); then
    sleep 2
    stable_xplanes=$(gcloud compute tpus tpu-vm ssh "$tpu" --zone="$zone" --project="$project" \
        --worker="$worker" \
        --command="find '$remote_dir' -type f -name '*.xplane.pb' -size +0c -printf '%s %p\\n' 2>/dev/null | sort" \
        2>/dev/null || true)
    [[ "$remote_xplanes" == "$stable_xplanes" ]] || continue
    gcloud compute tpus tpu-vm scp --recurse "$tpu:$remote_dir" "$dest_dir" \
      --zone="$zone" --project="$project" --worker="$worker" >/dev/null
    local_count=$(find "$dest_dir" -type f -name '*.xplane.pb' -size +0c | wc -l)
    (( local_count >= min_xplanes )) || continue
    find "$dest_dir" -type f -name '*.xplane.pb' -size +0c -print
    exit 0
  fi
  sleep 2
done
