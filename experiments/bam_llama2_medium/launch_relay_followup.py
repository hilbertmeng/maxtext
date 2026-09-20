"""Prepare each exact AOT, then launch its formal run without waiting for its peers."""
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path('/home/lishengping/xd/projects')
RUNS = [
    ('BamMediumColOnlyK32PartialMRelayM3', 'relay-k32-partial-m3',
     'BamMediumIndependentLLFMLPPerLayerColOnlyK32NoPE48PartialRoPE,BamMediumColOnlyK32MRelayM3'),
    ('BamMediumColOnlyK64MRelayM3QKOnly', 'relay-k64-m3-qk',
     'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE,BamMediumColOnlyK64TruncateMRelayM3'),
    ('BamMediumColOnlyK64MRelayM3VOnly', 'relay-k64-m3-v',
     'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE,BamMediumColOnlyK64TruncateMRelayM3'),
    ('BamMediumColOnlyK64MRelayM3OOnly', 'relay-k64-m3-o',
     'BamMediumIndependentLLFMLPPerLayerColOnlyK64QK48TruncatePartialRoPE,BamMediumColOnlyK64TruncateMRelayM3'),
]


def launch(item):
    run, ident, bases = item
    artifact = None
    command = [str(ROOT / 'prepare_train_aot.py'), run, sys.argv[1], 'v5p-16', '13500',
               '--primary-zone', 'europe-west4-a', '--backup-zones', 'us-east5-a', 'us-central1-a']
    with (ROOT / 'logs' / f'{ident}-aot.log').open('a', buffering=1) as log:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            if line.startswith('AOT_READY ') and artifact is None:
                artifact = next(x.split('=', 1)[1] for x in line.split() if x.startswith('artifact='))
                env = dict(EXP=run, ID=ident, MODE='install+train',
                           PRIMARY_ZONE='us-east5-a', BACKUP_ZONES='europe-west4-b',
                           BRANCH='codex/llf-m-anchor-relay', CODE_COMMIT=sys.argv[1],
                           COMPARE_RUNS=bases, COMPILED_TRAINSTEP_GCS=artifact,
                           LOSS_REPORT_INTERVAL='200', PLANNED_STEPS='13500')
                command = shlex.join(['env', *[f'{k}={v}' for k, v in env.items()],
                                      'bash', str(ROOT / 'run_exp_xd.sh')])
                command += ' >> ' + shlex.quote(str(ROOT / 'logs' / f'{ident}-launch.log')) + ' 2>&1'
                subprocess.run(['tmux', 'new-session', '-d', '-s', f'{run}-TPU{ident}-xd', command],
                               check=True)
                print(f'TRAIN_SUBMITTED {run} {artifact}', flush=True)
        code = proc.wait()
        if code or artifact is None:
            raise RuntimeError(f'{run}: AOT preparation exit={code}, ready={artifact}')


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(launch, RUNS))
