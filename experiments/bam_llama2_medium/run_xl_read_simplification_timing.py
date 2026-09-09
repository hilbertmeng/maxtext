#!/usr/bin/env python3
"""Sealed read-simplification AOT matrix; combined arm also verifies step0..100."""
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
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--exp', default='BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2LocalFetchC8SharedReadLLF')
    parser.add_argument('--steps', type=int, default=50000)

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
    arms = json.loads(Path(args.manifest).read_text())['arms']
    for spec in arms:
        suffix = spec['label']
        artifact = spec.get('artifact')
        deadline = time.monotonic() + 10800
        while not artifact:
            compiler_log = Path(spec['aot_log'])
            matches = re.findall(r'AOT_READY artifact=(gs://\\S+)',
                                 compiler_log.read_text() if compiler_log.exists() else '')
            if matches:
                artifact = matches[-1]
            elif time.monotonic() > deadline:
                raise TimeoutError('AOT not ready: ' + suffix)
            else:
                time.sleep(20)
        final_step = spec.get('final_step', 15)
        exp = args.exp
        compiled = '/tmp/shared-llf-' + suffix + '.pickle'
        ssh('gsutil -q cp ' + shlex.quote(artifact) + ' ' + shlex.quote(compiled), 'all')
        run = 'TimingReadSimplify_' + suffix + '_' + str(int(time.time()))
        log = '/home/lishengping/train_' + run + '.log'
        pattern = '[M]axText/train.py.*run_name=' + run
        arm = {'runtime_commit': spec['commit'], 'final_step': final_step, 'exp': exp, 'run': run, 'log': log, 'state': 'launching', 'artifact': artifact}
        result['arms'].append(arm)
        save()
        # Both arms preserve the sealed configuration's LR schedule and health flags.
        smoke = repo + '/.claude/skills/tpu-diagnostics/scripts/run_train_smoke_compiled.sh'
        command = ('cd ' + shlex.quote(repo) + '; nohup env DATASET_PATH=' + shlex.quote(dataset)
                   + ' SMOKE_OUTPUT=' + shlex.quote('/tmp/' + run)
                   + ' bash ' + shlex.quote(smoke) + ' ' + exp + ' ' + run
                   + ' ' + shlex.quote(compiled) + ' ' + str(args.steps) + ' >' + shlex.quote(log) + ' 2>&1 </dev/null &')
        ssh(command, 'all')
        print('LAUNCHED ' + run, flush=True)
        try:
            deadline = time.monotonic() + 3600
            while time.monotonic() < deadline:
                text = ssh('tail -80 ' + shlex.quote(log)
                           + '; if pgrep -f ' + shlex.quote(pattern)
                           + ' >/dev/null; then echo PROCESS_ALIVE; fi')
                if re.search(r'completed step: (\\d+),', text) and max(map(int, re.findall(r'completed step: (\\d+),', text))) >= final_step:
                    break
                if 'PROCESS_ALIVE' not in text:
                    raise RuntimeError('train process exited: ' + text[-4000:])
                time.sleep(10)
            else:
                raise TimeoutError('target step not reached within one hour')
            ssh('grep -q "Loaded compiled function!" ' + shlex.quote(log))
            text = ssh('grep "completed step:" ' + shlex.quote(log))
            records = {int(s): float(v) for s, v in re.findall(
                r'completed step: (\d+), steps/s:\s*([\d.]+)', text)}
            rates = [records[s] for s in range(10, 15)]
            losses = {int(step): float(loss) for step, loss in re.findall(
                r'completed step: (\d+),[^\n]*? loss: ([\d.eE+-]+)', text)}
            arm.update(state='complete', losses=losses, speeds_10_14=rates,
                       mean_steps_per_second=sum(rates)/len(rates))
            if suffix == 'all':
                comparisons = {}
                for base in (
                    'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2',
                    'BamLlama2XLHead16x128V2C256PartialRoPELocalQKRank2AllDecayRepro200',
                ):
                    cache = root / 'run_registry' / 'loss_cache' / (base + '.json')
                    points = json.loads(cache.read_text())['points']
                    common = [s for s in range(0, 101, 10)
                              if s in losses and str(s) in points]
                    gaps = {s: losses[s] - float(points[str(s)]) for s in common}
                    comparisons[base] = {
                        'gaps_10_step': gaps,
                        'max_abs_gap': max(map(abs, gaps.values())) if gaps else None,
                        'first_gap_above_1e6': next((s for s in common if abs(gaps[s]) > 1e-6), None),
                        'missing_steps': [s for s in range(0, 101, 10) if s not in common],
                    }
                arm['loss_comparisons'] = comparisons
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
