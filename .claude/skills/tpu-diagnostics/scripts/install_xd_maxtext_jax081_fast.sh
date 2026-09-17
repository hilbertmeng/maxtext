#!/usr/bin/env bash
set -euo pipefail

REGION=${1:?region is required}
PACKAGE=maxtext_py312_packages_jax081.tar.gz
CONDA_INSTALLER=Miniconda3-py312-Linux-x86_64.sh
BUCKET=gs://newproject-1-conda_script_$REGION
USER_ROOT=/home/lishengping
PROJECTS=$USER_ROOT/xd/projects

if ! gsutil -q stat "$BUCKET/$PACKAGE"; then
  BUCKET=gs://newproject-1-conda_script_europe-west4
fi
gsutil -q cp "$BUCKET/$PACKAGE" /tmp/$PACKAGE
mkdir -p "$PROJECTS"
if [[ ! -d "$PROJECTS/maxtext" ]]; then
  clone_tmp="$PROJECTS/maxtext.clone.$$"
  for delay in 0 2 4 8 16; do
    sleep "$delay"
    rm -rf "$clone_tmp"
    if git clone -q -b refactor-bam https://github.com/hilbertmeng/maxtext.git "$clone_tmp"; then
      mv "$clone_tmp" "$PROJECTS/maxtext"
      break
    fi
  done
  [[ -d "$PROJECTS/maxtext/.git" ]] || exit 1
fi

if [[ ! -d "$USER_ROOT/miniconda3" ]]; then
  gsutil -q cp "$BUCKET/$CONDA_INSTALLER" /tmp/$CONDA_INSTALLER
  bash /tmp/$CONDA_INSTALLER -b -p "$USER_ROOT/miniconda3"
fi

pigz -dc /tmp/$PACKAGE | tar -xf - -C "$USER_ROOT/miniconda3/lib/python3.12"
"$USER_ROOT/miniconda3/bin/python" - <<'PY'
import flax
import jax
import jaxlib
import numpy
import optax
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel  # noqa: F401

assert jax.__version__ == "0.8.1"
assert jaxlib.__version__ == "0.8.1"
assert flax.__version__ == "0.12.1"
assert optax.__version__ == "0.2.6"
assert numpy.__version__ == "2.1.3"
PY
