#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
TASK_FINGERPRINT=$(python3 - <<'PY'
from pathlib import Path
import hashlib
files=['MaxText/exp.py','MaxText/layers/attentions.py','MaxText/layers/fusion.py','MaxText/layers/linears.py','MaxText/layers/bam_mlp.py','MaxText/tests/bam_mlp_read_test.py','MaxText/tests/summary_writer_test.py','MaxText/train.py','MaxText/train_compile.py','MaxText/configs/base.yml','MaxText/pyconfig.py']
print(hashlib.sha256(b''.join(f.encode()+Path(f).read_bytes() for f in files)).hexdigest())
PY
)
TASK_CACHE=/data0/xd/bam_diagnostics/mediumprop-mlp-bam-read/cpu/$TASK_FINGERPRINT
mkdir -p "$TASK_CACHE"
exec 9>"$TASK_CACHE/check.lock"
flock 9
if [[ -f "$TASK_CACHE/ok" ]]; then echo "MLP_BAM_READ_CPU_REUSED fingerprint=$TASK_FINGERPRINT"; exit 0; fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$TASK_ROOT/MaxText:$TASK_ROOT/MaxText/tests"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=8
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/bam_mlp_read_test.py --jobs 3
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
touch "$TASK_CACHE/ok"
