#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
TASK_HASH=$(git rev-parse HEAD)
TASK_CHECK=/data0/xd/bam_diagnostics/pythia-swiglu/cpu/$TASK_HASH
mkdir -p "$TASK_CHECK"
exec 9>"$TASK_CHECK/check.lock"
flock 9
if [[ -f "$TASK_CHECK/ok" ]]; then echo "CPU_SWIGLU_REUSED commit=$TASK_HASH"; exit 0; fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$TASK_ROOT/MaxText:$TASK_ROOT/MaxText/tests"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=8
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/pythia_swiglu_test.py --test PythiaSwiGLUTest.test_full_size_budgets_and_non_mlp_shapes --test PythiaSwiGLUTest.test_all_arm_gradients_and_causality --test PythiaSwiGLUTest.test_reject_unrequested_variant_and_invalid_widths --jobs 2
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/pythia_test.py --test PythiaTest.test_config_and_init_roles --test PythiaTest.test_torch_neox_reference --jobs 2
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
bash /home/xd/projects/maxtext/.agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh "$TASK_ROOT"
touch "$TASK_CHECK/ok"
echo "CPU_SWIGLU_ALL_OK commit=$TASK_HASH"
