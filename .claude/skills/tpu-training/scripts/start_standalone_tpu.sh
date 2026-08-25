#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 TPU_NAME ACCELERATOR ZONE [INSTALL_SCRIPT]" >&2
  exit 2
fi

tpu=$1
accelerator=$2
zone=$3
install_script=${4:-install_xd_maxtext_jax081.sh}
root=${TPU_AG_PROJECTS_DIR:-/home/lishengping/xd/projects}
creator=$root/create_standalone_tpu.sh
log_dir=$root/logs
log=$log_dir/$tpu-create.log

if [[ "$install_script" != /* ]]; then
  install_script=$root/$install_script
fi

[[ -x "$creator" ]] || { echo "ERROR: missing creator: $creator" >&2; exit 1; }
[[ -f "$install_script" ]] || { echo "ERROR: missing installer: $install_script" >&2; exit 1; }
mkdir -p "$log_dir"
nohup "$creator" "$tpu" "$accelerator" "$zone" "$install_script" \
  >"$log" 2>&1 < /dev/null &
pid=$!
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  tail -40 "$log" >&2 || true
  exit 1
fi
printf 'CREATOR_STARTED tpu=%s zone=%s pid=%s log=%s\n' "$tpu" "$zone" "$pid" "$log"
