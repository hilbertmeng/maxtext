#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision="$(git rev-parse HEAD)"
cache_root=/data0/xd/bam_diagnostics/mediumprop-h-final-local-o/cpu-gates
mkdir -p "$cache_root"
exec 9>"$cache_root/$revision.lock"
flock -x 9
if [[ -f "$cache_root/$revision.ok" ]]; then
  cat "$cache_root/$revision.ok"
  exit 0
fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText:MaxText/tests
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$PWD" \
 --test-file MaxText/tests/bam_final_local_o_test.py --jobs 3 --cores-per-job 8 \
 --log-dir "$cache_root/$revision-final-local-o"
taskset -c 24-31 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/summary_writer_test.py
bash .agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh "$PWD"
printf 'FINAL_LOCAL_O_CPU_GATE_OK commit=%s four-exit-tests+five-writer-tests+full-BAM-regression\n' "$revision" | tee "$cache_root/$revision.ok"
