#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-/data0/xd/bam_diagnostics/xl-localv-row-causal-5250}
GCS=${VROW_GCS:-gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl-localv-row-causal-20260916/results}
mkdir -p "$OUT"
# Serialize this destination; mutable progress logs are not scientific artifacts.
exec 9>"$OUT/.collect.lock"
flock 9
gsutil -m -q rsync -r -x '.*\.log$' "$GCS" "$OUT"
python "$ROOT/validate_localv_row_causal.py" "$OUT"
python "$ROOT/summarize_localv_row_causal.py" "$OUT" --limit 64 > "$OUT/summary64.md"
