#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || ($# - 1) % 2 != 0 )); then
  echo "usage: $0 MATRIX_MANIFEST TPU ZONE [TPU ZONE...]" >&2
  exit 2
fi

manifest=$1
shift
project=${TPU_PROJECT:-newproject-1-451205}
root=${TPU_AG_ROOT:-/home/lishengping/xd/projects}
delete_helper=${TPU_DELETE_HELPER:-$root/delete_tpu_xd.sh}

[[ -s "$manifest" ]] || { echo "ERROR: missing matrix manifest: $manifest" >&2; exit 1; }
awk -F '\t' '$3 == "TRACE_VERIFIED" {found=1} END {exit !found}' "$manifest" || {
  echo "ERROR: target profile has no verified trace; retain backup resources" >&2
  exit 1
}

while (( $# > 0 )); do
  tpu=$1
  zone=$2
  shift 2
  "$delete_helper" "$tpu" "$zone" "$project"
done
