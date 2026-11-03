#!/usr/bin/env bash
set -euo pipefail

MODE=install
if [[ ${1:-} == --source-only || ${1:-} == --verify-source ]]; then
  MODE=${1#--}
  shift
fi
REGION=${1:?region is required}
SOURCE_COMMIT=${2:-}
# Existing controller processes can still pass REGION ARCHIVE COMMIT on recovery.
if [[ ${3:-} =~ ^[0-9a-f]{40}$ ]]; then SOURCE_COMMIT=$3; fi
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]] || {
  echo "ERROR: a full source commit is required" >&2; exit 2;
}
SOURCE_REPO=${MAXTEXT_REPO_URL:-https://github.com/hilbertmeng/maxtext.git}
PACKAGE=maxtext_py312_packages_jax081.tar.gz
CONDA_INSTALLER=Miniconda3-py312-Linux-x86_64.sh
BUCKET=gs://newproject-1-conda_script_us-east5
USER_ROOT=/home/lishengping/mengqy/hunit_prenorm_23k_diag
PROJECTS=$USER_ROOT/xd/projects

mkdir -p "$PROJECTS"
export GIT_TERMINAL_PROMPT=0 GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_SYSTEM=/dev/null
source_git() { git -c credential.helper= -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 "$@"; }
source_clean() {
  [[ -d "$1/.git" ]] &&
    source_git -C "$1" diff --quiet && source_git -C "$1" diff --cached --quiet
}
source_matches() {
  source_clean "$1" && [[ $(source_git -C "$1" rev-parse HEAD) == "$SOURCE_COMMIT" ]]
}
if [[ "$MODE" == verify-source ]]; then
  source_matches "$PROJECTS/maxtext"
  exit $?
fi
if ! source_matches "$PROJECTS/maxtext"; then
  checkout=$PROJECTS/maxtext
  if [[ -d "$checkout/.git" ]]; then
    source_clean "$checkout" || { echo "ERROR: tracked source modifications in $checkout" >&2; exit 1; }
  else
    checkout=$(mktemp -d "$PROJECTS/.maxtext-git.XXXXXX")
    source_git init -q "$checkout"
    source_git -C "$checkout" remote add origin "$SOURCE_REPO"
  fi
  if ! source_git -C "$checkout" cat-file -e "$SOURCE_COMMIT^{commit}" 2>/dev/null; then
    fetched=false
    for attempt in 1 2 3; do
      echo "SOURCE_FETCH attempt=$attempt commit=$SOURCE_COMMIT repo=$SOURCE_REPO"
      if timeout 180 git -c credential.helper= -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 \
          -C "$checkout" fetch --depth=1 --no-tags "$SOURCE_REPO" "$SOURCE_COMMIT"; then
        fetched=true; break
      fi
      (( attempt == 3 )) || sleep $((attempt * 2))
    done
    $fetched || { echo "ERROR: exact source fetch failed; retained $checkout" >&2; exit 1; }
  fi
  source_git -C "$checkout" checkout -q --detach "$SOURCE_COMMIT"
  source_matches "$checkout" || { echo "ERROR: source verification failed" >&2; exit 1; }
  if [[ "$checkout" != "$PROJECTS/maxtext" ]]; then
    if [[ -e "$PROJECTS/maxtext" || -L "$PROJECTS/maxtext" ]]; then
      backup="$PROJECTS/maxtext.pre-git.$(date -u +%s).$$"
      mv "$PROJECTS/maxtext" "$backup"
      echo "SOURCE_PREVIOUS retained=$backup"
    fi
    mv "$checkout" "$PROJECTS/maxtext"
  fi
fi
printf '%s\n' "$SOURCE_COMMIT" >"$PROJECTS/maxtext/.source_commit"
echo "SOURCE_READY transport=git commit=$SOURCE_COMMIT tree=$(source_git -C "$PROJECTS/maxtext" rev-parse HEAD^{tree})"
[[ "$MODE" != source-only ]] || exit 0

if ! gsutil -q stat "$BUCKET/$PACKAGE"; then
  BUCKET=gs://newproject-1-conda_script_europe-west4
fi

if ! "$USER_ROOT/miniconda3/bin/python" -c 'import jax' >/dev/null 2>&1; then
  gsutil -q cp "$BUCKET/$PACKAGE" /tmp/$PACKAGE
  if [[ ! -d "$USER_ROOT/miniconda3" ]]; then
  gsutil -q cp "$BUCKET/$CONDA_INSTALLER" /tmp/$CONDA_INSTALLER
  bash /tmp/$CONDA_INSTALLER -b -p "$USER_ROOT/miniconda3"
  fi
  if command -v pigz >/dev/null 2>&1; then pigz -dc /tmp/$PACKAGE; else gzip -dc /tmp/$PACKAGE; fi | tar -xf - -C "$USER_ROOT/miniconda3/lib/python3.12"
fi
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

NATIVE_SOURCE=$PROJECTS/praxis
if [[ ! -d "$NATIVE_SOURCE/.git" ]]; then
 source_git init -q "$NATIVE_SOURCE"
 source_git -C "$NATIVE_SOURCE" remote add origin https://github.com/xiaoda99/praxis.git
fi
PRAXIS_COMMIT=5059b47c18e7fd2fc951c6454e8b72e386c53b0a
source_git -C "$NATIVE_SOURCE" fetch --depth=1 --no-tags origin "$PRAXIS_COMMIT"
source_git -C "$NATIVE_SOURCE" checkout -q --detach "$PRAXIS_COMMIT"
[[ $(source_git -C "$NATIVE_SOURCE" rev-parse HEAD) == "$PRAXIS_COMMIT" ]]
gsutil -q cp gs://newproject-1-llm_projects_us-east5/log/diagnostics/mengqy/mudd_spectral_20261102/praxis-native-deps.tar.gz /tmp/praxis-native-deps.tar.gz
tar -xzf /tmp/praxis-native-deps.tar.gz -C "$USER_ROOT"
echo PRAXIS_NATIVE_READY
