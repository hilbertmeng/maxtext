#!/usr/bin/env bash
set -euo pipefail
model=${1:?medium or xl}
repo=$(cd "$(dirname "$0")/../.." && pwd)
python=${MAXTEXT_PYTHON:-/home/lishengping/miniconda3/bin/python}
commit=${DIAGNOSTIC_COMMIT:-$(git -C "$repo" rev-parse HEAD 2>/dev/null || cat "$repo/.source_commit")}
case "$model" in
 medium)
  base=BamLlama2MediumV2
  checkpoint=gs://newproject-1-llm_base_models_us-central1/log/$base/checkpoints/13250/items
  trainer=1afd942; batch=2; layers=8,9,11,6 ;;
 xl)
  base=BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2
  checkpoint=gs://newproject-1-llm_projects_europe-west4/log/$base/checkpoints/49720/items
  trainer=aef0d97411a1725386ebba1aeae1bf4acb1bb79e; batch=1; layers=11,6,10 ;;
 *) exit 2 ;;
esac
tag="bam-row-cross-sign-$model-${commit:0:7}"
output="/tmp/$tag"
gcs="gs://newproject-1-llm_base_models_us-central1/log/diagnostics/$tag"
mkdir -p "$output/maxtext-output/$tag"
gsutil cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz "$output/cohort.npz"
cd "$repo"
env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off \
 DIAGNOSTIC_COMMIT="$commit" BAM_ROW_SIGN_OUTPUT="$output" BAM_ROW_SIGN_LAYERS="$layers" \
 BAM_RESIDUAL_ATTR_BASE_CONFIG="$base" BAM_RESIDUAL_ATTR_TRAINER_COMMIT="$trainer" \
 BAM_RESIDUAL_ATTR_BATCH_SIZE="$batch" BAM_RESIDUAL_ATTR_COHORT_PATH="$output/cohort.npz" \
 "$python" experiments/bam_llama2_medium/row_cross_sign.py MaxText/configs/base.yml \
 exp_class=BamRowCrossSign run_name="$tag" load_parameters_path="$checkpoint" \
 base_output_directory="$output/maxtext-output" tensorboard_dir="$output/tensorboard" \
 only_eval=True dataset_path=gs://newproject-1-common_datasets_europe-west4/pythia_pile_idxmaps_tfrecord \
 enable_checkpointing=True async_checkpointing=False &
pid=$!
upload() { gsutil -m rsync -r -x 'maxtext-output/.*|tensorboard/.*' "$output" "$gcs"; }
while kill -0 "$pid" 2>/dev/null; do
 sleep 60
 upload || true
done
wait "$pid"
upload
echo "ROW_SIGN_UPLOADED model=$model gcs=$gcs"
