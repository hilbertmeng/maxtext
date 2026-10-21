#!/usr/bin/env bash
# The matrix runner supplies dataset, output and sealed source; only common timing overrides here.
set -euo pipefail
export M_PROFILER=xplane
export M_SKIP_FIRST_N_STEPS_FOR_PROFILER=10
export M_PROFILER_STEPS=5
export M_PROFILE_PERIODICALLY_PERIOD=1000
export M_PROFILE_CLEANLY=true
export M_SA_BLOCK_Q=1024 M_SA_BLOCK_KV=1024 M_SA_BLOCK_KV_COMPUTE=512
export M_SA_BLOCK_Q_DKV=1024 M_SA_BLOCK_KV_DKV=1024 M_SA_BLOCK_KV_DKV_COMPUTE=1024
export M_SA_USE_FUSED_BWD_KERNEL=true
exec bash /home/lishengping/run_train_smoke.sh "$@"
