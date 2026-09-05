#!/usr/bin/env bash
# Local workstation: complete bounded parallel transfers before parsing anything.
set -euo pipefail
label=${1:?analysis label}
shift
(( $# > 0 )) || exit 2
root=${BAM_DIAGNOSTIC_ROOT:-/data0/xd/bam_diagnostics}
gcs=${BAM_DIAGNOSTIC_GCS:-gs://newproject-1-llm_base_models_us-central1/log/diagnostics}
gsutil=${GSUTIL_BIN:-/home/xd/google-cloud-sdk/bin/gsutil}
python=${MAXTEXT_CPU_PYTHON:-/data0/xd/conda/envs/maxtext-cpu/bin/python}
script_dir=$(cd "$(dirname "$0")" && pwd)
[[ "$label" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
for prefix in "$@"; do [[ "$prefix" =~ ^bam-row-mediation-[A-Za-z0-9_-]+$ ]] || exit 2; done
paths=(); pids=(); failed=0
reap() {
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  pids=()
}
for prefix in "$@"; do
  mkdir -p "$root/$prefix"
  paths+=("$root/$prefix")
  "$gsutil" -m rsync -r "$gcs/$prefix" "$root/$prefix" \
    >"$root/$prefix.sync.log" 2>&1 &
  pids+=("$!")
  if (( ${#pids[@]} == 2 )); then reap; fi
done
reap
if (( failed )); then echo "TRANSFER_FAILED: inspect $root/*.sync.log" >&2; exit 1; fi
"$python" "$script_dir/analyze_row_mediation.py" "${paths[@]}" \
  --output "$root/$label.json" >"$root/$label.txt"
echo "ANALYSIS_READY $root/$label.json"
