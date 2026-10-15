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
cd "$root"
entrypoint=MaxText/pile_bucket_eval.py
if [[ ${PILE_BUCKET_BENCHMARK:-0} == 1 ]]; then
  entrypoint=MaxText/pile_bucket_benchmark.py
fi
config_args=(
  MaxText/configs/base.yml
  exp_class="$exp" run_name="$tag"
  base_output_directory=/tmp/pile-bucket-t2048/config
  tensorboard_dir=/tmp/pile-bucket-t2048/tensorboard
  load_parameters_path="$checkpoint"
  only_eval=True enable_checkpointing=True async_checkpointing=False
  skip_jax_distributed_system=True
  per_device_batch_size=1.0 eval_per_device_batch_size=16.0
  max_target_length=2048 jax_cache_dir=/tmp/pile-bucket-t2048/jax-cache
  log_config=False
)
# These historical MHA/Mudd checkpoints used the C256 BF16-logits core.
# Later base.yml defaults enable Splash; preserve the training implementation.
case "$exp" in
  BamMHAMediumPropC256TruePile|MuddLlama2MediumPropTruePile)
    config_args+=(bam_splash_attention=False)
    ;;
esac
# Exercise the exact shell/CLI/config contract without contacting GCS or a checkpoint.
if [[ ${PILE_BUCKET_CONFIG_ONLY:-0} == 1 ]]; then
  "$python" "$entrypoint" "${config_args[@]}"
  exit
fi
on_exit() {
  status=$?
  printf '%s\n' "$status" >"$output/exit.txt"
  [[ ! -f "$output/eval.log" ]] || gsutil -q cp "$output/eval.log" "$gcs/eval.log" || true
  gsutil -q cp "$output/exit.txt" "$gcs/exit.txt" || true
  exit "$status"
}
trap on_exit EXIT
if [[ ! -f "$PILE_BUCKET_SOURCE" ]]; then
  gsutil -q cp 'gs://newproject-1-common_datasets_us-east5/pythia_model_test/pile_test/val_with_eos.tfrecord#1742902686628652' "$PILE_BUCKET_SOURCE.partial"
  mv "$PILE_BUCKET_SOURCE.partial" "$PILE_BUCKET_SOURCE"
fi
"$python" - "$gcs" "$output/progress.json" <<'PY'
import sys
from pathlib import Path
from google.api_core.exceptions import NotFound
from google.cloud import storage
bucket,name=sys.argv[1][5:].split('/',1)
client=storage.Client(project='newproject-1-451205')
b=client.bucket(bucket).blob(name+'/progress.json')
try:Path(sys.argv[2]).write_bytes(b.download_as_bytes())
except NotFound:pass
try:client.bucket(bucket).blob(name+'/exit.txt').delete()
except NotFound:pass
PY
"$python" "$entrypoint" "${config_args[@]}" >"$output/eval.log" 2>&1
