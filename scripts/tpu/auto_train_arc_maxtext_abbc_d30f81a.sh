#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f -- "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd)"
export MAXTEXT_SYNC_REF="recurrent-mudd-abbc-validation-d30f81a"
export MAXTEXT_EXPECTED_COMMIT="4f0bd7d891dae790b0cb6947a2632e4691f7547c"
exec bash "${SCRIPT_DIR}/auto_train_arc_maxtext.sh" "$@"
