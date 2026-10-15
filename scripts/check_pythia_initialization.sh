#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/pythia_initialization_test.py --test PythiaInitializationTest.test_parameter_roles_and_unchanged_semantic_initializers --test PythiaInitializationTest.test_full_size_parameter_budget --test PythiaInitializationTest.test_first_matrix_reproduces_original_scale --test PythiaInitializationTest.test_forward_gradients_and_causality --test PythiaInitializationTest.test_health_option_is_wired --jobs 2
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/pythia_test.py --test PythiaTest.test_config_and_init_roles --jobs 1
env JAX_PLATFORMS=cpu PYTHONPATH=MaxText OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
