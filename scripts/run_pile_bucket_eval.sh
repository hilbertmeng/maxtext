#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 4 ]]; then
  echo 'usage: run_pile_bucket_eval.sh EXP CHECKPOINT TAG GCS_OUTPUT' >&2
  exit 2
fi
exp=$1
checkpoint=$2
tag=$3
gcs=$4
root=$(cd "$(dirname "$0")/.." && pwd)
python=${PILE_BUCKET_PYTHON:-/home/lishengping/miniconda3/bin/python}
output=/tmp/pile-bucket-t2048/$tag
cache=/tmp/pile-bucket-t2048/source
mkdir -p "$output" "$cache" "/tmp/pile-bucket-t2048/config/$tag"
export CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1
export PILE_BUCKET_SOURCE=$cache/val_with_eos.tfrecord
export PILE_BUCKET_OUTPUT=$output PILE_BUCKET_GCS=$gcs
export PILE_BUCKET_SAVE_EVERY=${PILE_BUCKET_SAVE_EVERY:-128}
export LIBTPU_INIT_ARGS=${LIBTPU_INIT_ARGS:-}
on_exit() {
  status=$?
  printf '%s\n' "$status" >"$output/exit.txt"
  [[ ! -f "$output/eval.log" ]] || gsutil -q cp "$output/eval.log" "$gcs/eval.log" || true
  gsutil -q cp "$output/exit.txt" "$gcs/exit.txt" || true
  exit "$status"
}
trap on_exit EXIT
cd "$root"
if [[ ! -f "$PILE_BUCKET_SOURCE" ]]; then
  gsutil -q cp 'gs://newproject-1-common_datasets_us-east5/pythia_model_test/pile_test/val_with_eos.tfrecord#1742902686628652' "$PILE_BUCKET_SOURCE.partial"
  mv "$PILE_BUCKET_SOURCE.partial" "$PILE_BUCKET_SOURCE"
fi
# Missing progress is normal on the first launch; access failures remain fatal.
"$python" - "$gcs" "$output/progress.json" <<'PY'
import sys
from pathlib import Path
from google.api_core.exceptions import NotFound
from google.cloud import storage
bucket,name=sys.argv[1][5:].split('/',1)
b=storage.Client(project='newproject-1-451205').bucket(bucket).blob(name+'/progress.json')
try:Path(sys.argv[2]).write_bytes(b.download_as_bytes())
except NotFound:pass
try:storage.Client(project='newproject-1-451205').bucket(bucket).blob(name+'/exit.txt').delete()
except NotFound:pass
PY
"$python" MaxText/pile_bucket_eval.py MaxText/configs/base.yml \
  exp_class="$exp" run_name="$tag" \
  base_output_directory=/tmp/pile-bucket-t2048/config \
  tensorboard_dir=/tmp/pile-bucket-t2048/tensorboard \
  load_parameters_path="$checkpoint" load_full_state_path= \
  only_eval=True enable_checkpointing=True \
  per_device_batch_size=1.0 eval_per_device_batch_size=16.0 \
  max_target_length=2048 jax_cache_dir=/tmp/pile-bucket-t2048/jax-cache \
  log_config=False \
  >"$output/eval.log" 2>&1
