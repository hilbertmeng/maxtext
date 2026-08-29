#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 || $# > 4 )); then
  echo "usage: $0 EXP GCS_OUTPUT [STEPS] [TARGET_TOPOLOGY]" >&2
  exit 2
fi

EXP=$1
GCS_OUTPUT=$2
STEPS=${3:-20}
TARGET_TOPOLOGY=${4:-v5p-32}
REPO=${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}
PYTHON=${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}
TMP_DIR=$(mktemp -d /tmp/maxtext-aot.XXXXXX)
COMPILED="$TMP_DIR/$EXP.pickle"
trap 'rm -rf "$TMP_DIR"' EXIT

[[ -e /dev/vfio/0 ]] || {
  echo "ERROR: AOT compilation requires an installed TPU VM" >&2
  exit 1
}
[[ -x "$PYTHON" ]] || {
  echo "ERROR: TPU-VM MaxText Python is unavailable: $PYTHON" >&2
  exit 1
}
mkdir -p "$TMP_DIR/output/AOT_$EXP"

cd "$REPO"
set +e
env HARDWARE=tpu JAX_PLATFORMS=cpu JAX_TRACEBACK_FILTERING=off \
  MAXTEXT_SKIP_AOT_ANALYSIS=1 "$PYTHON" \
  MaxText/train_compile.py MaxText/configs/base.yml \
  "exp_class=$EXP" "run_name=AOT_$EXP" "steps=$STEPS" \
  "base_output_directory=$TMP_DIR/output" \
  "compile_topology=$TARGET_TOPOLOGY" compile_topology_num_slices=1 \
  "compiled_trainstep_file=$COMPILED" \
  enable_checkpointing=False async_checkpointing=False \
  upload_all_profiler_results=False
compile_status=$?
set -e
if [[ ! -s "$COMPILED" ]]; then
  exit "$compile_status"
fi

gsutil -q cp "$COMPILED" "$GCS_OUTPUT"
gsutil stat "$GCS_OUTPUT"
