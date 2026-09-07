#!/usr/bin/env bash
# Run on tpu-ag. Each bounded worker delegates resource ownership to prepare_train_aot.py.
set -euo pipefail
[[ $# == 1 ]] || { echo "usage: $0 COMMIT" >&2; exit 2; }
root=${TPU_AG_ROOT:-/home/lishengping/xd/projects}
commit=$("$root/prepare_train_aot.py" verify-commit "$1")
output="gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/local-fetch/$commit"
logs="$root/logs/local-fetch-aot-$commit"
mkdir -p "$logs"
export root commit output logs
experiments=()
for variant in Control C8 C8LocalV Full FullLocalV C8SharedRead; do
  for layout in Scan NonScan; do
    experiments+=("BamLlama2MediumV2C256LocalFetch${variant}${layout}")
  done
done
printf '%s\n' "${experiments[@]}" | xargs -P "${AOT_PARALLELISM:-4}" -n 1 bash -c '
  exp=$1
  echo "AOT_START exp=$exp utc=$(date -u +%FT%TZ) log=$logs/$exp.log"
  if "$root/prepare_train_aot.py" "$exp" "$commit" v5p-16 13500 \
      --output "$output/$exp.pickle" >"$logs/$exp.log" 2>&1; then
    echo "AOT_DONE exp=$exp utc=$(date -u +%FT%TZ)"
  else
    echo "AOT_FAILED exp=$exp log=$logs/$exp.log" >&2
    exit 1
  fi
' _
echo "LOCAL_FETCH_AOT_MATRIX_READY root=$output logs=$logs"
