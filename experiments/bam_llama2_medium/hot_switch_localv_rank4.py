#!/usr/bin/env python3
"""Run on tpu-ag after LocalVRank4 AOT is ready; retain Legacy's physical TPU."""
import json
import os
from pathlib import Path
import shlex
import sys
import time

ROOT = Path('/home/lishengping/xd/projects')
sys.path.insert(0, str(ROOT))
import closeout_runs as closeout

OLD = 'BamMediumIndependentLLFRoutingLegacy'
NEW = OLD + 'LocalVRank4'
COMMIT = '6f831293f6ce912cf14364d99fb19b86bd1b3874'
STATE = ROOT / 'aot_runs/6f83129-a89d740b.json'

def main():
    state = json.loads(STATE.read_text())
    assert state['status'] == 'ready', state['status']
    assert state['commit'] == COMMIT and state['exp'] == NEW
    closeout.run_command(['gsutil', '-q', 'stat', state['artifact']])
    item = closeout.resolve_run(ROOT, OLD, 'stopped', 'Resumable pause for user-requested LocalVRank4 hot switch')
    assert not item.already_closed
    assert item.tpu == 'xd-v5p16-gram-routing-0' and item.zone == 'us-east5-a'
    closeout.stop_controller(item)
    closeout.signal_train(item)
    closeout.wait_for_checkpoint(item, 600, 10)
    assert item.final_checkpoint is not None
    assert item.final_checkpoint >= (item.last_progress_step or 0)
    pattern = f'[M]axText/train.py.*run_name={OLD}'
    for _ in range(60):
        result = closeout.run_command([
            'gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', '--internal-ip', item.tpu,
            f'--zone={item.zone}', f'--project={item.project}', '--worker=all',
            f'--command=if pgrep -f {shlex.quote(pattern)} >/dev/null; then echo OLD_ALIVE; else echo OLD_GONE; fi',
        ], check=True, timeout=120)
        if 'OLD_ALIVE' not in result.stdout and result.stdout.count('OLD_GONE') >= 2:
            break
        time.sleep(5)
    else:
        raise RuntimeError('Old worker process still present; new RUN not launched')
    env = dict(os.environ, EXP=NEW, ID='gram-localv-r4', MODE='train',
               TPU_NAME_OVERRIDE=item.tpu, TPU_TYPE='v5p-16', PRIMARY_ZONE=item.zone,
               ZONE=item.zone, BACKUP_ZONES='', BRANCH='codex/local-read-gram',
               CODE_COMMIT=COMMIT, COMPARE_RUNS=OLD, PLANNED_STEPS='13500',
               COMPILED_TRAINSTEP_GCS=state['artifact'])
    import subprocess
    subprocess.run(['bash', str(ROOT / 'run_exp_xd.sh')], env=env, check=True)
    print(f'NEW_LAUNCH_SUBMITTED old_checkpoint={item.final_checkpoint}', flush=True)
    closeout.run_command([str(ROOT / 'run_registry.py'), 'collect-loss', OLD], timeout=180)
    closeout.register_closeout(item)
    print(json.dumps(item.as_dict(), indent=2), flush=True)

if __name__ == '__main__':
    main()
