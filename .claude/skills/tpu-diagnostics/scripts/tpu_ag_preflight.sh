#!/usr/bin/env bash
set -euo pipefail

project=${TPU_PROJECT:-newproject-1-451205}
account=${TPU_ACCOUNT:-626151558586-compute@developer.gserviceaccount.com}
configuration=${TPU_GCLOUD_CONFIGURATION:-xd-tpu}
min_free_gb=${TPU_AG_MIN_FREE_GB:-10}
max_log_mb=${TPU_AG_MAX_LOG_MB:-512}
root=${TPU_AG_ROOT:-/home/lishengping/xd/projects}
lock=/tmp/xd-tpu-gcloud-config.lock

exec 9>"$lock"
flock 9
if ! gcloud config configurations describe "$configuration" >/dev/null 2>&1; then
  gcloud config configurations create "$configuration" --no-activate >/dev/null
fi
configured_account=$(gcloud config get-value account --configuration="$configuration" 2>/dev/null || true)
configured_project=$(gcloud config get-value project --configuration="$configuration" 2>/dev/null || true)
if [[ "$configured_account" != "$account" ]]; then
  gcloud config set account "$account" --configuration="$configuration" >/dev/null
fi
if [[ "$configured_project" != "$project" ]]; then
  gcloud config set project "$project" --configuration="$configuration" >/dev/null
fi
flock -u 9

export CLOUDSDK_ACTIVE_CONFIG_NAME="$configuration"
gcloud auth print-access-token --account="$account" >/dev/null
gcloud projects describe "$project" --account="$account" --format='value(projectId)' \
  | grep -Fx "$project" >/dev/null

available_kb=$(df -Pk / | awk 'NR == 2 {print $4}')
required_kb=$((min_free_gb * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "ERROR: tpu-ag has only $((available_kb / 1024 / 1024)) GiB free; require ${min_free_gb} GiB" >&2
  exit 1
fi

mkdir -p "$root/logs"
mapfile -t oversized_logs < <(
  find "$root/logs" -maxdepth 1 -type f -size "+${max_log_mb}M" -print 2>/dev/null | sort
)
if (( ${#oversized_logs[@]} > 0 )); then
  printf 'ERROR: oversized tpu-ag control logs (limit %s MiB):\n' "$max_log_mb" >&2
  printf '  %s\n' "${oversized_logs[@]}" >&2
  exit 1
fi

# gcloud command logs are disposable; bounded retention prevents silent accumulation.
find "$HOME/.config/gcloud/logs" -type f -mtime +14 -delete 2>/dev/null || true

printf 'PREFLIGHT_OK project=%s account=%s config=%s free_gb=%s\n' \
  "$project" "$account" "$configuration" "$((available_kb / 1024 / 1024))"
