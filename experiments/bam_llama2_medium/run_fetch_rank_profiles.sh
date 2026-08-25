#!/usr/bin/env bash
set -euo pipefail

controller_root=${TPU_AG_ROOT:-/home/lishengping/xd/projects}
export PROFILE_REMOTE_ROOT=${PROFILE_REMOTE_ROOT:-/home/lishengping/xd/profile_outputs/fetch_rank}
export PROFILE_GCS_ROOT=${PROFILE_GCS_ROOT:-gs://newproject-1-llm_base_models_us-central1/log/diagnostics/fetch_rank}
exec "$controller_root/run_profile_matrix.sh" "$@"
