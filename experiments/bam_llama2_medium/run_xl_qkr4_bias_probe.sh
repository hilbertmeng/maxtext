#!/usr/bin/env bash
set -euo pipefail
export QKV_DECOMPOSE_BIAS=1
export QKV_OUTPUT="${QKV_OUTPUT:-/tmp/xl-qkr4-bias-16000}"
export QKV_GCS="${QKV_GCS:-gs://newproject-1-llm_projects_europe-west4/log/diagnostics/xl-qkr4-basis-16000/bias-results}"
exec bash "$(dirname "$0")/run_xl_qkr4_basis_probe.sh"
