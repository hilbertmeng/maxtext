#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "usage: $0 GCS_PREFIX LOCAL_DIR" >&2
  exit 2
fi

prefix=${1%/}
local_dir=$2
[[ "$prefix" == gs://* ]] || { echo "ERROR: expected gs:// prefix" >&2; exit 2; }
mkdir -p "$local_dir"

mapfile -t objects < <(
  gcloud storage ls --recursive "$prefix/**" 2>/dev/null \
    | grep -E '/step_10/.+[.](xplane[.]pb|trace[.]json[.]gz)$' \
    | sort
)
if (( ${#objects[@]} == 0 )); then
  echo "ERROR: no primary step_10 XPlane objects under $prefix" >&2
  exit 1
fi

for object in "${objects[@]}"; do
  relative=${object#"$prefix"/}
  target="$local_dir/$relative"
  mkdir -p "$(dirname "$target")"
  gcloud storage cp "$object" "$target" >/dev/null
done

xplanes=$(find "$local_dir" -type f -path '*/step_10/*' -name '*.xplane.pb' | wc -l)
traces=$(find "$local_dir" -type f -path '*/step_10/*' -name '*.trace.json.gz' | wc -l)
if (( xplanes == 0 || xplanes != traces )); then
  echo "ERROR: incomplete primary artifacts: xplanes=$xplanes traces=$traces" >&2
  exit 1
fi
echo "PRIMARY_XPLANES_OK xplanes=$xplanes local=$local_dir"
