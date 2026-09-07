#!/usr/bin/env bash
# Run one ready six-arm group on the standalone target; preserve its training AOT contract.
set -euo pipefail
[[ $# == 4 ]] || { echo "usage: $0 TPU ZONE COMMIT Scan|NonScan" >&2; exit 2; }
tpu=$1 zone=$2 commit=$3 layout=$4
[[ "$layout" == Scan || "$layout" == NonScan ]] || exit 2
root=${TPU_AG_ROOT:-/home/lishengping/xd/projects}
commit=$("$root/prepare_train_aot.py" verify-commit "$commit")
export AOT_ROOT="gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps/local-fetch/$commit"
export PROFILE_STEPS=13500 PROFILE_SKIP=10 PROFILE_DURATION=5 PROFILE_DONE_STEP=15
case "${zone%-*}" in
  us-central1) bucket=newproject-1-llm_base_models_us-central1 ;;
  europe-west4|us-east5) bucket="newproject-1-llm_projects_${zone%-*}" ;;
  *) echo "Unsupported profile zone: $zone" >&2; exit 2 ;;
esac
export PROFILE_GCS_ROOT="gs://$bucket/log/diagnostics/local-fetch"
experiments=()
for variant in Control C8 C8LocalV Full FullLocalV C8SharedRead; do
  experiments+=("BamLlama2MediumV2C256LocalFetch${variant}${layout}")
done
exec "$root/run_profile_matrix.sh" "$tpu" "$zone" "$commit" "local-fetch-$layout" "${experiments[@]}"
