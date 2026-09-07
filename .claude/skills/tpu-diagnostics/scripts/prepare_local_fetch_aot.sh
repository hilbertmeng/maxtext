#!/usr/bin/env bash
# Compatibility entry; the coordinator maintains separate short/long compilation lanes.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: $0 COMMIT" >&2; exit 2; }
root=${TPU_AG_ROOT:-/home/lishengping/xd/projects}
exec /usr/bin/python3 "$(dirname "$0")/prepare_local_fetch_aot.py" "$1"
