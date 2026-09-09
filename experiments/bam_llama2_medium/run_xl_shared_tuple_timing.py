#!/usr/bin/env python3
"""Paired old/new AOT timing; run on tpu-ag, never starts auto-train."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--tpu', required=True)
    parser.add_argument('--zone', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--old-aot', required=True)
    parser.add_argument('--new-aot', required=True)
    args = parser.parse_args()
    assert args.tpu.startswith('xd-') and re.fullmatch('[0-9a-f]{40}', args.commit)
    root = Path('/home/lishengping/xd/projects')
    repo = str(root / 'maxtext')
    result = {'tpu': args.tpu, 'zone': args.zone, 'commit': args.commit, 'arms': []}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')

    def ssh(command, workers='0'):
        process = subprocess.run([
            'gcloud', '--configuration=xd-tpu', 'compute', 'tpus', 'tpu-vm', 'ssh',
            args.tpu, '--project=newproject-1-451205', '--zone=' + args.zone,
            '--worker=' + workers, '--command=' + command,
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        if process.returncode:
            raise RuntimeError(process.stdout[-6000:])
        return process.stdout

    install_log = root / 'logs' / (args.tpu + '-create.log')
    deadline = time.monotonic() + 7200
    while 'INSTALL_OK TPU=' + args.tpu not in install_log.read_text():
        if time.monotonic() > deadline:
            raise TimeoutError('standalone installation did not complete')
        time.sleep(15)
    print('INSTALL_VERIFIED', flush=True)
    ssh('cd ' + shlex.quote(repo) + ' && test "$(git rev-parse HEAD)" = ' + args.commit,
        'all')
    dataset = 'gs://newproject-1-common_datasets_us-east5/pythia_pile_idxmaps_tfrecord'
    if args.zone != 'us-east5-a':
        raise ValueError('This sealed pair uses the UE5a dataset and historical timing control')
    prefix = 'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8'
    for suffix, artifact in (('old', args.old_aot), ('tuple', args.new_aot)):
        exp = prefix + 'SharedReadLLF'
        compiled = '/tmp/shared-llf-' + suffix + '.pickle'
        ssh('gsutil -q cp ' + shlex.quote(artifact) + ' ' + shlex.quote(compiled), 'all')
        run = 'TimingTupleLLF_' + suffix + '_' + str(int(time.time()))
        log = '/home/lishengping/train_' + run + '.log'
        pattern = '[M]axText/train.py.*run_name=' + run
        arm = {'exp': exp, 'run': run, 'log': log, 'state': 'launching', 'artifact': artifact}
        result['arms'].append(arm)
        save()
        # Both AOT arms preserve the same 50k schedule and no-health configuration.
        smoke = repo + '/.claude/skills/tpu-diagnostics/scripts/run_train_smoke_compiled.sh'
        command = ('cd ' + shlex.quote(repo) + '; nohup env DATASET_PATH=' + shlex.quote(dataset)
                   + ' SMOKE_OUTPUT=' + shlex.quote('/tmp/' + run)
                   + ' bash ' + shlex.quote(smoke) + ' ' + exp + ' ' + run
                   + ' ' + shlex.quote(compiled) + ' 50000 >' + shlex.quote(log) + ' 2>&1 </dev/null &')
        ssh(command, 'all')
        print('LAUNCHED ' + run, flush=True)
        try:
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                text = ssh('tail -80 ' + shlex.quote(log)
                           + '; if pgrep -f ' + shlex.quote(pattern)
                           + ' >/dev/null; then echo PROCESS_ALIVE; fi')
                if 'completed step: 15,' in text:
                    break
                if 'PROCESS_ALIVE' not in text:
                    raise RuntimeError('train process exited: ' + text[-4000:])
                time.sleep(10)
            else:
                raise TimeoutError('no step15 within one hour')
            ssh('grep -q "Loaded compiled function!" ' + shlex.quote(log))
            text = ssh('grep "completed step:" ' + shlex.quote(log))
            records = {int(s): float(v) for s, v in re.findall(
                r'completed step: (\d+), steps/s:\s*([\d.]+)', text)}
            rates = [records[s] for s in range(10, 15)]
            arm.update(state='complete', speeds_10_14=rates,
                       mean_steps_per_second=sum(rates)/len(rates))
            output.with_name(run + '.log').write_text(text)
            print(json.dumps(arm), flush=True)
        finally:
            ssh('pkill -KILL -f ' + shlex.quote(pattern) + ' || true', 'all')
            ssh('! pgrep -f ' + shlex.quote(pattern), 'all')
            save()
    result['state'] = 'complete'
    save()
    print('TIMING_COMPLETE ' + str(output), flush=True)


if __name__ == '__main__':
    main()
