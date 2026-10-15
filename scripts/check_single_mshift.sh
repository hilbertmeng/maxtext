#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3
export PYTHONPATH="$TASK_ROOT/MaxText:$TASK_ROOT/MaxText/tests"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=8
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/bam_m_shift_test.py --test MShiftTest.test_boundaries_column_broadcast_and_nonrecursive_source --test MShiftTest.test_health_masks_and_known_mix --test MShiftTest.test_read_and_persist_same_reads_but_distinct_write_bases --test MShiftTest.test_scan_forward_gradients_and_future_causality --test MShiftTest.test_full_parameter_budget_and_gate_sharding --jobs 2
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/pythia_test.py --test PythiaTest.test_config_and_init_roles --test PythiaTest.test_torch_neox_reference --jobs 2
bash /home/xd/projects/maxtext/.agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh "$TASK_ROOT"
