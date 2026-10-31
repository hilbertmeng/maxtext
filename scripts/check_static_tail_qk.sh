#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision="$(git rev-parse HEAD)"
root=/data0/xd/bam_diagnostics/mediumprop-h-static-tail-qk/cpu-gates
mkdir -p "$root"
exec 9>"$root/$revision.lock"
flock -x 9
if [[ -f "$root/$revision.ok" ]]; then cat "$root/$revision.ok"; exit 0; fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText:MaxText/tests
python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$PWD" \
 --test-file MaxText/tests/bam_static_tail_qk_test.py --jobs 3 --cores-per-job 8 \
 --log-dir "$root/$revision-static-tail"
taskset -c 24-31 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/summary_writer_test.py
bash .agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh "$PWD"
printf 'STATIC_TAIL_CPU_GATE_OK commit=%s four-focused+five-summary-writer+47-BAM-tests\n' "$revision" | tee "$root/$revision.ok"
