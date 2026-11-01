#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision="$(git rev-parse HEAD)"
root=/data0/xd/bam_diagnostics/mediumprop-h-relocated-local-o
mkdir -p "$root/cpu-gates"
exec 9>"$root/cpu-gates/$revision.lock"
flock -x 9
if [[ -f "$root/cpu-gates/$revision.ok" ]]; then
  cat "$root/cpu-gates/$revision.ok"
  exit 0
fi
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= TF_CPP_MIN_LOG_LEVEL=3 PYTHONPATH=MaxText:MaxText/tests
# Development checks can be reused only when every tested source byte is identical.
# A fresh checkout or a changed model/test source always executes the focused suite.
if ! python3 - "$root" <<'PY'
import hashlib, json, sys, time, os
from pathlib import Path
root = Path(sys.argv[1])
try:
    signature = json.loads((root/'focused-second-source.json').read_text())
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
               for name, digest in signature.items())
    log = root/'focused-second.log'
    pid = int((root/'focused-second.pid').read_text())
    announced = False
    while 'CPU_ALL_OK' not in log.read_text():
        cmd = Path(f'/proc/{pid}/cmdline')
        if not cmd.exists() or b'run_cpu_tests_parallel.py' not in cmd.read_bytes():
            raise AssertionError('Earlier focused check did not succeed')
        if not announced:
            print('WAIT_EXISTING_FOCUSED_CHECK_EXACT_SOURCE', flush=True)
            announced = True
        time.sleep(2)
    assert 'CPU_ALL_OK' in log.read_text()
except (OSError, ValueError, AssertionError):
    raise SystemExit(1)
print('FOCUSED_REUSED_EXACT_SOURCE: eight tests, including five full-model Pallas gradient checks')
PY
then
  python3 /home/xd/projects/xd_tpu_scripts/run_cpu_tests_parallel.py "$PWD" \
    --test-file MaxText/tests/bam_relocated_local_o_test.py --jobs 4 --cores-per-job 8 \
    --log-dir "$root/cpu-gates/$revision-focused"
fi
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/bam_final_local_o_test.py \
  FinalLocalOTest.test_explicit_generalized_read_and_all_gradients \
  FinalLocalOTest.test_formal_budget_and_scope
taskset -c 0-7 /data0/xd/conda/envs/maxtext-cpu/bin/python MaxText/tests/summary_writer_test.py
bash .agents/skills/tpu-diagnostics/scripts/run_bam_unit_tests.sh "$PWD"
printf 'RELOCATED_O_CPU_GATE_OK commit=%s eight-focused+two-parent+five-summary+47-BAM\n' "$revision" \
  | tee "$root/cpu-gates/$revision.ok"
