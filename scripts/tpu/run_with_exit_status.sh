#!/usr/bin/env bash

set -u

if [[ $# -lt 4 || "$3" != "--" ]]; then
  echo "Usage: $0 STATUS_FILE LOG_FILE -- COMMAND [ARG ...]" >&2
  exit 64
fi

status_file=$1
log_file=$2
shift 3

mkdir -p "$(dirname "$status_file")" "$(dirname "$log_file")"

write_status() {
  local state=$1
  local rc=${2:-}
  local ended_at=${3:-}
  local tmp="${status_file}.tmp.$$"

  {
    printf 'state=%s\n' "$state"
    printf 'pid=%s\n' "$$"
    printf 'started_at_utc=%s\n' "$started_at"
    printf 'git_sha=%s\n' "$git_sha"
    if [[ -n "$ended_at" ]]; then
      printf 'ended_at_utc=%s\n' "$ended_at"
    fi
    if [[ -n "$rc" ]]; then
      printf 'rc=%s\n' "$rc"
    fi
  } >"$tmp"
  mv "$tmp" "$status_file"
}

started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
git_sha=$(git rev-parse HEAD 2>/dev/null || printf 'unknown')
write_status running

set +e
"$@" >"$log_file" 2>&1
rc=$?
set -e

ended_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
write_status exited "$rc" "$ended_at"
exit "$rc"
