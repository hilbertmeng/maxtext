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
program=row_mediation.py
if [[ $phase == consumers ]]; then program=row_consumer_positions.py; fi
if [[ $phase == neighbors ]]; then program=row_neighbors.py; fi
if [[ $phase == mlp_export ]]; then program=row_mlp_export.py; fi
if [[ $phase == v_export ]]; then program=row_v_export.py; fi
if [[ $phase == delivery ]]; then program=row_delivery.py; fi
if [[ $phase == token_worlds ]]; then program=row_token_worlds.py; fi
tag="bam-row-mediation-$model-L$source_layer-$phase-${BAM_MEDIATION_LABEL:-all}-${commit:0:7}"
if [[ ${BAM_MEDIATION_COMPONENT:-cross} == self ]]; then tag="$tag-rowself"; fi
if [[ ${BAM_MEDIATION_COMPONENT:-cross} == both ]]; then tag="$tag-rowboth"; fi
if [[ ${BAM_MEDIATION_REFERENCE:-opposite} == self ]]; then tag="$tag-selfref"; fi
output="/tmp/$tag"
gcs="gs://newproject-1-llm_base_models_us-central1/log/diagnostics/$tag"
mkdir -p "$output/maxtext-output/$tag"
if [[ -n ${BAM_MEDIATION_RESUME_GCS:-} ]]; then
  gsutil -m rsync -r -x 'maxtext-output/.*|tensorboard/.*|.*\.pending' \
    "$BAM_MEDIATION_RESUME_GCS" "$output"
fi
gsutil cp gs://newproject-1-llm_base_models_us-central1/log/diagnostics/cohorts/pile-eval-t2048-seed9876-n128-v1/pile_eval_cohort.npz "$output/cohort.npz"
cd "$repo"
env HARDWARE=tpu JAX_TRACEBACK_FILTERING=off DIAGNOSTIC_COMMIT="$commit" \
 BAM_MEDIATION_OUTPUT="$output" BAM_RESIDUAL_ATTR_BASE_CONFIG="$base" \
 BAM_RESIDUAL_ATTR_TRAINER_COMMIT="$trainer" BAM_RESIDUAL_ATTR_BATCH_SIZE="$batch" \
 BAM_RESIDUAL_ATTR_COHORT_PATH="$output/cohort.npz" \
 "$python" "experiments/bam_llama2_medium/$program" MaxText/configs/base.yml \
 exp_class=BamRowMediation run_name="$tag" load_parameters_path="$checkpoint" \
 base_output_directory="$output/maxtext-output" tensorboard_dir="$output/tensorboard" \
 only_eval=True dataset_path=gs://newproject-1-common_datasets_europe-west4/pythia_pile_idxmaps_tfrecord \
 enable_checkpointing=True async_checkpointing=False &
pid=$!
upload() { gsutil -m rsync -r -x 'maxtext-output/.*|tensorboard/.*|.*\.pending' "$output" "$gcs"; }
periodic_upload() { while sleep 60; do upload || true; done; }
periodic_upload &
uploader=$!
rc=0
wait "$pid" || rc=$?
# Reap the computation immediately; periodic artifact upload must not insert
# an extra polling interval between paired jobs on the same spot worker.
kill "$uploader" 2>/dev/null || true
wait "$uploader" 2>/dev/null || true
upload
echo "MEDIATION_UPLOADED model=$model gcs=$gcs exit=$rc"
exit "$rc"
