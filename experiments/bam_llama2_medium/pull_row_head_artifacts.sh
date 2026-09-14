#!/usr/bin/env bash
# Usage: pull_row_head_artifacts.sh GCS_HEADS_PREFIX LOCAL_HEADS_DIR [--complete]
# Active uploads replace log object generations; defer logs until all uploaders finish.
set -euo pipefail
source_prefix=${1:?GCS heads prefix required}
local_target=${2:?Local heads directory required}
mkdir -p "$local_target"
if [[ ${3:-} == --complete ]]; then
 gsutil -m rsync -r "$source_prefix" "$local_target"
else
 gsutil -m rsync -r -x '.*\.log$' "$source_prefix" "$local_target"
fi
