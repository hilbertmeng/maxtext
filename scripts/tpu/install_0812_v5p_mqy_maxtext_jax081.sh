#!/usr/bin/env bash
set -euo pipefail

ZONE="${1:-}"
TPU_TYPE="${2:-}"
BUCKET_PREFIX="${3:-newproject-1-}"

if [[ -z "$ZONE" || -z "$TPU_TYPE" ]]; then
  echo "Usage: $0 ZONE_WITHOUT_SUFFIX TPU_TYPE [BUCKET_PREFIX]" >&2
  exit 2
fi

allowed_zones=(us-east1 us-west4 us-central2 us-central1 europe-west4 us-east5)
valid_zone=false
for allowed in "${allowed_zones[@]}"; do
  if [[ "$ZONE" == "$allowed" ]]; then
    valid_zone=true
    break
  fi
done
if ! "$valid_zone"; then
  echo "Unsupported TPU region: $ZONE" >&2
  exit 2
fi

HOME="/home/lishengping"
PACKAGE_NAME="maxtext_py312_packages.tar.gz"
CONDA_NAME="Miniconda3-py312-Linux-x86_64.sh"
PY_VERSION="3.12"
CONDA_BUCKET="gs://newproject-1-conda_script_${ZONE}"
MAXTEXT_INITIAL_BRANCH="${MAXTEXT_INITIAL_BRANCH:-refactor}"
MAXTEXT_WORK_DIR="${MAXTEXT_WORK_DIR:-${HOME}/projects/maxtext}"

echo "Installing MaxText TPU environment: region=$ZONE type=$TPU_TYPE"
gsutil cp "${CONDA_BUCKET}/${PACKAGE_NAME}" ./

if [[ ! -d "${MAXTEXT_WORK_DIR}/.git" ]]; then
  mkdir -p "$(dirname -- "$MAXTEXT_WORK_DIR")"
  git clone -b "$MAXTEXT_INITIAL_BRANCH" \
    https://github.com/hilbertmeng/maxtext.git "$MAXTEXT_WORK_DIR"
else
  echo "MaxText checkout already exists; controller will synchronize the requested ref."
fi

if [[ ! -d "${HOME}/miniconda3" ]]; then
  gsutil cp "${CONDA_BUCKET}/${CONDA_NAME}" ./
  bash "$CONDA_NAME"
  printf '%s\n' 'export PATH=/home/lishengping/miniconda3/bin:$PATH' >>"${HOME}/.bashrc"
fi

pigz -dc "$PACKAGE_NAME" |
  tar xv -C "${HOME}/miniconda3/lib/python${PY_VERSION}/"

"${HOME}/miniconda3/bin/pip" install \
  -f https://storage.googleapis.com/jax-releases/libtpu_releases.html \
  'jax[tpu]==0.8.1'
"${HOME}/miniconda3/bin/pip" install -U flax==0.12.1 optax==0.2.6
