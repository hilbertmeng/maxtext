#!/usr/bin/env bash
set -euo pipefail
model=${1:?medium or xl}
repo=$(cd "$(dirname "$0")/../.." && pwd)
python=${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}
commit=${DIAGNOSTIC_COMMIT:-$(git -C "$repo" rev-parse HEAD 2>/dev/null || cat "$repo/.source_commit")}
case "$model" in
 xl)
  base=BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
  checkpoint=gs://newproject-1-llm_projects_europe-west4/log/$base/checkpoints/49720/items
  trainer=aef0d97411a1725386ebba1aeae1bf4acb1bb79e; batch=1 ;;
 medium)
  base=BamLlama2MediumV2
  checkpoint=gs://newproject-1-llm_base_models_us-central1/log/$base/checkpoints/13250/items
  trainer=1afd942; batch=2 ;;
 *) exit 2 ;;
esac
source_layer=${BAM_MEDIATION_SOURCE:-11}
phase=${BAM_MEDIATION_PHASE:-coarse}
tag="bam-row-mediation-$model-L$source_layer-$phase-${BAM_MEDIATION_LABEL:-all}-${commit:0:7}"
if [[ ${BAM_MEDIATION_REFERENCE:-opposite} == self ]]; then tag="$tag-selfref"; fi
output="/tmp/$tag"
gcs="gs://newproject-1-llm_base_models_us-central1/log/diagnostics/$tag"
mkdir -p "$output/maxtext-output/$tag"
gsutil cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz "$output/cohort.npz"
cd "$repo"
env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off DIAGNOSTIC_COMMIT="$commit" \
 BAM_MEDIATION_OUTPUT="$output" BAM_RESIDUAL_ATTR_BASE_CONFIG="$base" \
 BAM_RESIDUAL_ATTR_TRAINER_COMMIT="$trainer" BAM_RESIDUAL_ATTR_BATCH_SIZE="$batch" \
 BAM_RESIDUAL_ATTR_COHORT_PATH="$output/cohort.npz" \
 "$python" experiments/bam_llama2_medium/row_mediation.py MaxText/configs/base.yml \
 exp_class=BamRowMediation run_name="$tag" load_parameters_path="$checkpoint" \
 base_output_directory="$output/maxtext-output" tensorboard_dir="$output/tensorboard" \
 only_eval=True dataset_path=gs://newproject-1-common_datasets_europe-west4/pythia_pile_idxmaps_tfrecord \
 enable_checkpointing=True async_checkpointing=False &
pid=$!
upload() { gsutil -m rsync -r -x 'maxtext-output/.*|tensorboard/.*' "$output" "$gcs"; }
while kill -0 "$pid" 2>/dev/null; do
 sleep 60
 upload || true
done
rc=0
wait "$pid" || rc=$?
upload
echo "MEDIATION_UPLOADED model=$model gcs=$gcs exit=$rc"
exit "$rc"
