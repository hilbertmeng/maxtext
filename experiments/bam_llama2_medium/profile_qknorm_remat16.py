#!/usr/bin/env python3
"""tpu-ag: diagnose the topology-specific full/minimal inversion using existing AOTs.

Both arms run on one owned v5p-16, exact formal per-device batch16/global128,
13500-step schedule, profiler10–14 and stable20–99. The formal RUN is untouched.
On preemption, repeat the whole pair on a backup. Delete all owned candidates.
"""
import json
import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, '/home/lishengping/xd/projects')
import prepare_train_aot as aot

EXP = 'BamMediumPropMShiftReadG050StandardQKNormTruePile'
FULL = '2f6466ba18472c4ea3886534cf44bfc65e1ac394'
MINIMAL = 'ea0c153bd9e09c9cc86655b506b3009757c685b1'
AOT = 'gs://newproject-1-llm_base_models_us-central1/log/compiled_trainsteps'
STAMP = '20261021-remat16'
ROOT = aot.ROOT


def run_pair(candidate, state, arms):
    for policy, commit, exp, artifact in arms:
        name, zone = candidate['name'], candidate['zone']
        dataset = aot.run(['python3', str(ROOT/'dataset_paths.py'), '--zone', zone,
                           '--variant=truepile4096', '--exp', exp, '--commit', commit]).stdout.strip()
        command = ['env', f'AOT_ROOT={artifact}', 'PROFILE_STEPS=13500',
                   'PROFILE_DONE_STEP=99', 'PROFILE_SKIP=10', 'PROFILE_PERIOD=1000',
                   'PROFILE_DURATION=5', 'PROFILE_TRACE_COUNT=1', f'DATASET_PATH={dataset}',
                   str(ROOT/'run_profile_matrix.sh'), name, zone, commit,
                   f'qknorm-remat16-{policy}', exp]
        log = ROOT/'logs'/f'{STAMP}-{name}-{policy}.log'
        state['active'] = dict(candidate=candidate, policy=policy, commit=commit, log=str(log))
        save(state)
        print('PROFILE_ARM_START', state['active'], flush=True)
        with log.open('w') as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-4000:], flush=True)
            raise RuntimeError(f'{policy} profile failed: {log}')
        state.setdefault('completed_arms', []).append(state['active'])
        save(state)
        print('PROFILE_ARM_OK', policy, name, flush=True)


def save(state):
    path = ROOT/'logs'/f'{STAMP}-state.json'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2)+'\n');tmp.replace(path)


def main():
    global STAMP
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--aligned-commit')
    parser.add_argument('--aligned-aot-root')
    args=parser.parse_args()
    arms=[(policy,commit,EXP,f'{AOT}/{commit[:7]}/jax081-i0ae3f58-c17f538a/v5p-16/s13500')
          for policy,commit in [('full',FULL),('minimal',MINIMAL)]]
    label='rematdiag'
    if args.aligned_commit:
        if not args.aligned_aot_root:parser.error('--aligned-aot-root required')
        commit=aot.resolve_remote_commit(args.aligned_commit)
        arms += [(f'align128-{policy}',commit,
                  f'BamMediumPropMShiftReadG050StandardQKNormAligned128{policy.title()}Profile',
                  args.aligned_aot_root.rstrip('/')) for policy in ('full','minimal')]
        label='align128';STAMP+='-aligned128'
    # AOT-first: do not allocate target hardware while a required executable is missing.
    for _,_,exp,artifact in arms:aot.run(['gsutil','stat',f'{artifact}/{exp}.pickle'])
    candidates = [dict(name=f'xd-v5p-16-qknorm-{label}-20261021-{tag}', zone=zone)
                  for tag,zone in [('ue5a','us-east5-a'), ('uc1a','us-central1-a'),
                                   ('ew4b','europe-west4-b')]]
    state = dict(candidates=candidates, status='acquiring', arms=arms)
    save(state)
    try:
        for c in candidates:
            result = aot.run([str(ROOT/'start_standalone_tpu.sh'), c['name'], 'v5p-16',
                              c['zone'], 'install_xd_maxtext_jax081.sh', FULL], check=False)
            print(result.stdout, flush=True)
        tried = set()
        deadline = time.monotonic()+7200
        while time.monotonic()<deadline:
            for c in candidates:
                if c['name'] in tried:
                    continue
                log = ROOT/'logs'/f'{c["name"]}-create.log'
                if not log.exists() or f'INSTALL_OK TPU={c["name"]}' not in log.read_text(errors='replace'):
                    continue
                tried.add(c['name'])
                try:
                    state['status']='profiling';save(state)
                    run_pair(c,state,arms)
                    state.update(status='complete', winner=c);save(state)
                    print('PROFILE_PAIR_OK', c, flush=True)
                    return
                except Exception as exc:
                    print('PROFILE_PAIR_RETRY',repr(exc),flush=True)
                    state.setdefault('errors',[]).append(str(exc));save(state)
            if len(tried)==len(candidates):
                raise RuntimeError('All owned profile candidates failed')
            time.sleep(20)
        raise TimeoutError('Profile candidate timeout')
    finally:
        failures=[]
        for c in candidates:
            result=aot.run([str(ROOT/'delete_tpu_xd.sh'),c['name'],c['zone'],aot.PROJECT],check=False,timeout=900)
            if result.returncode:failures.append(dict(candidate=c,output=result.stdout))
        state['cleanup_failures']=failures;save(state)
        if failures:raise RuntimeError(f'Owned profile cleanup failed: {failures}')


if __name__=='__main__':main()
