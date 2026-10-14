#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$TASK_ROOT"
git diff --quiet
TASK_HASH=$(git rev-parse HEAD)
TASK_CHECK=/data0/xd/bam_diagnostics/pythia-family/cpu/$TASK_HASH
mkdir -p "$TASK_CHECK"
exec 9>"$TASK_CHECK/check.lock"
flock 9
if [[ -f "$TASK_CHECK/ok" ]]; then echo "CPU_PYTHIA_FAMILY_REUSED commit=$TASK_HASH"; exit 0; fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH="$TASK_ROOT/MaxText:$TASK_ROOT/MaxText/tests"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 XLA_FLAGS=--xla_force_host_platform_device_count=8
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/pythia_family_test.py --test PythiaFamilyTest.test_full_size_shapes --test PythiaFamilyTest.test_mudd_bam_gradients --test PythiaFamilyTest.test_fusion_matches_pythia_parallel_circuit --test PythiaFamilyTest.test_boundaries_column_broadcast_and_nonrecursive_source --test PythiaFamilyTest.test_health_masks_and_known_mix --jobs 2
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$TASK_ROOT" --test-file MaxText/tests/pythia_test.py --test PythiaTest.test_config_and_init_roles --test PythiaTest.test_scan_gradients_causality_and_checkpoint --test PythiaTest.test_bfloat16_remat_training --test PythiaTest.test_fused_projection_parity_and_cached_decode --test PythiaTest.test_torch_neox_reference --jobs 2
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python -m unittest discover -s MaxText/tests -p summary_writer_test.py
bash /home/xd/projects/maxtext/.agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh "$TASK_ROOT"
touch "$TASK_CHECK/ok"
echo "CPU_PYTHIA_FAMILY_OK commit=$TASK_HASH"
