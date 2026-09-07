#!/usr/bin/env bash
# Start target measurements as soon as scan AOTs verify; non-scan compiles independently.
set -euo pipefail
[[ $# == 3 ]] || { echo "usage: $0 COMMIT TPU ZONE" >&2; exit 2; }
commit=$1 tpu=$2 zone=$3
root=${TPU_AG_ROOT:-/home/lishengping/xd/projects}
commit=$("$root/prepare_train_aot.py" verify-commit "$commit")
exec 9>"$root/logs/local-fetch-pipeline-${commit:0:7}.lock"
flock -n 9 || { echo 'Pipeline already running' >&2; exit 1; }
wait_aot_group() {
  /usr/bin/python3 - "$root" "$commit" "$1" <<'PY'
import json
from pathlib import Path
import sys
import time
root, commit, layout = sys.argv[1:]
expected = {f'BamLlama2MediumV2C256LocalFetch{variant}{layout}'
            for variant in ('Control', 'C8', 'C8LocalV', 'Full', 'FullLocalV', 'C8SharedRead')}
deadline = time.monotonic() + 14400
while time.monotonic() < deadline:
    states = {}
    for path in (Path(root) / 'aot_runs').glob(f'{commit[:7]}-*.json'):
        state = json.loads(path.read_text())
        if state.get('commit') == commit and state.get('exp') in expected:
            states[state['exp']] = state['status']
    failed = {exp: status for exp, status in states.items()
              if status in ('failed', 'source_failed', 'cleanup_failed', 'interrupted')}
    if failed:
        raise SystemExit(f'AOT_GROUP_FAILED {failed}')
    if len(states) == 6 and all(status in ('artifact_ready', 'ready') for status in states.values()):
        print(f'AOT_GROUP_VERIFIED layout={layout}', flush=True)
        break
    time.sleep(20)
else:
    raise SystemExit(f'AOT_GROUP_TIMEOUT layout={layout} states={states}')
PY
}
wait_aot_group Scan
"$root/start_standalone_tpu.sh" "$tpu" v5p-16 "$zone" install_xd_maxtext_jax081.sh "$commit"
log="$root/logs/$tpu-create.log"
deadline=$((SECONDS + 14400))
while ! grep -q "INSTALL_OK TPU=$tpu" "$log"; do
  if grep -q '^ERROR:' "$log" || (( SECONDS > deadline )); then
    tail -40 "$log"
    echo "TARGET_INSTALL_FAILED tpu=$tpu zone=$zone" >&2
    exit 1
  fi
  sleep 20
done
echo "TARGET_INSTALLED utc=$(date -u +%FT%TZ) tpu=$tpu zone=$zone"
bash "$root/profile_local_fetch_matrix.sh" "$tpu" "$zone" "$commit" Scan
wait_aot_group NonScan
bash "$root/profile_local_fetch_matrix.sh" "$tpu" "$zone" "$commit" NonScan
echo "LOCAL_FETCH_PROFILE_COMPLETE utc=$(date -u +%FT%TZ) tpu=$tpu zone=$zone retained=true"
