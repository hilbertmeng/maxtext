#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  echo "usage: $0 GCS_ROOT TARGET_TOPOLOGY STEPS EXP_CLASS..." >&2
  exit 2
fi

gcs_root=${1%/}
topology=$2
steps=$3
shift 3
# libtpu owns a per-VM lock even for cross-topology compilation. Parallelize
# across TPU VMs, not concurrent compiler processes on one VM.
jobs=${AOT_COMPILE_JOBS:-1}
repo=${MAXTEXT_REPO:-/home/lishengping/xd/projects/maxtext}
compiler=${AOT_COMPILER:-$repo/.claude/skills/tpu-diagnostics/scripts/compile_trainstep_aot.sh}
commit=$(git -C "$repo" rev-parse HEAD)

for exp in "$@"; do
  [[ "$exp" =~ ^[A-Za-z0-9_]+$ ]] || { echo "ERROR: invalid exp class: $exp" >&2; exit 2; }
done
[[ -x "$compiler" ]] || { echo "ERROR: missing compiler: $compiler" >&2; exit 1; }

export compiler gcs_root topology steps
printf '%s\0' "$@" | xargs -0 -n1 -P "$jobs" bash -c '
  set -euo pipefail
  exp=$1
  "$compiler" "$exp" "$gcs_root/$exp.pickle" "$steps" "$topology"
' _

for exp in "$@"; do
  gsutil stat "$gcs_root/$exp.pickle" >/dev/null
done
printf 'AOT_MATRIX_OK commit=%s topology=%s count=%s root=%s\n' \
  "$commit" "$topology" "$#" "$gcs_root"
