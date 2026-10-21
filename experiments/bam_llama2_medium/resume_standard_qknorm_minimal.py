#!/usr/bin/env python3
"""Experiment-specific, checkpoint-preserving same-RUN remat handoff on tpu-ag."""
import argparse
import fcntl
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

ROOT = Path('/home/lishengping/xd/projects')
sys.path.insert(0, str(ROOT))
import closeout_runs as closeout
import run_registry as registry

RUN = 'BamMediumPropMShiftReadG050StandardQKNormTruePile'
OLD = '2f6466ba18472c4ea3886534cf44bfc65e1ac394'
NEW = 'ea0c153bd9e09c9cc86655b506b3009757c685b1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('aot_state', type=Path)
    parser.add_argument('--execute', action='store_true', help='Default only previews the validated handoff')
    parser.add_argument('--restore-full', action='store_true', help='Return from the tested minimal runtime to the verified full runtime')
    args = parser.parse_args()
    old_commit, new_commit = (NEW, OLD) if args.restore_full else (OLD, NEW)
    old_mode, new_mode = ('minimal', 'full') if args.restore_full else ('full', 'minimal')
    aot = json.loads(args.aot_state.read_text())
    assert aot['status'] in ('ready', 'artifact_ready'), aot['status']
    expected = dict(exp=RUN, commit=new_commit, target_topology='v5p-16', steps=13500)
    assert all(aot.get(k) == v for k, v in expected.items())
    closeout.run_command(['gsutil', '-q', 'stat', aot['artifact']])
    manifest = json.loads(closeout.run_command(['gsutil', 'cat', aot['manifest']]).stdout)
    assert all(manifest.get(k) == v for k, v in expected.items())
    item = closeout.resolve_run(ROOT, RUN, 'running', 'User-requested compatible minimal-remat continuation')
    assert item.registry['declared_status'] == 'running'
    if item.registry['code_commit'] == new_commit:
        print('ALREADY_SUBMITTED: verify newer FIRST_STEP and checkpoint; trainer not touched')
        return
    assert item.registry['code_commit'] == old_commit
    assert item.tpu == 'xd-v5p-16-310260-maxtext' and item.zone == 'us-east5-a'
    closeout.assert_resource_ownership(item)
    node = json.loads(closeout.run_command(['gcloud', 'compute', 'tpus', 'tpu-vm', 'describe',
        item.tpu, f'--zone={item.zone}', f'--project={item.project}', '--format=json']).stdout)
    assert node['state'] == 'READY'
    workers = len(node['networkEndpoints'])
    env = dict(EXP=RUN, RUN_NAME=RUN, ID='310260', MODE='train',
               TPU_NAME_OVERRIDE=item.tpu, TPU_TYPE='v5p-16', PRIMARY_ZONE=item.zone, ZONE=item.zone,
               BACKUP_ZONES=','.join(item.registry.get('backup_zones', [])) if isinstance(item.registry.get('backup_zones'), list) else item.registry.get('backup_zones', ''),
               BRANCH='codex/mediumprop-qknorm', CODE_COMMIT=new_commit, PLANNED_STEPS='13500',
               COMPARE_RUNS=','.join(item.registry['compare_runs']), LOSS_REPORT_INTERVAL='200', LOSS_WINDOW='25',
               DATASET_PATH=item.registry['dataset_path'], DATASET_VARIANT='truepile4096',
               BASE_OUTPUT_DIR_OVERRIDE=item.registry['base_output_directory'],
               COMPILED_TRAINSTEP_GCS=aot['artifact'])
    launch = ['tmux', 'new-session', '-d', '-s', item.registry['tmux_session'],
              shlex.join(['env', *(f'{k}={v}' for k, v in env.items()), 'bash', str(ROOT/'run_exp_xd.sh')])]
    state = dict(run=RUN, old_commit=old_commit, new_commit=new_commit, tpu=item.tpu, zone=item.zone,
                 observed_step=item.last_progress_step, artifact=aot['artifact'], launch=launch)
    if not args.execute:
        print(json.dumps(dict(dry_run=True, **state), indent=2))
        return
    folder = ROOT/'logs'/'runtime-restarts'
    folder.mkdir(parents=True, exist_ok=True)
    journal = folder/f'{RUN}-{new_mode}-{new_commit[:7]}.json'
    def save(phase):
        state.update(phase=phase, updated_utc=registry.now_utc())
        registry.atomic_json(journal, state)
        print('RUNTIME_HANDOFF', phase, flush=True)
    lock_path = ROOT/'logs'/'hot-switch'/f'{item.tpu}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry.collect_live_loss(item.registry)
        item.last_progress_step = max(registry.load_loss_points(RUN))
        state['observed_step'] = item.last_progress_step
        save('stopping_old_trainer')
        closeout.stop_controller(item)
        closeout.signal_train(item)
        closeout.wait_for_checkpoint(item, 600, 10)
        pattern = f'[M]axText/train.py.*run_name={RUN}([[:space:]]|$)'
        probe = f'if pgrep -f {shlex.quote(pattern)} >/dev/null; then echo OLD_ALIVE; else echo OLD_GONE; fi'
        deadline = time.monotonic()+600
        while True:
            out = closeout.run_command(['gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', '--internal-ip',
                item.tpu, f'--zone={item.zone}', f'--project={item.project}', '--worker=all', f'--command={probe}'], timeout=120).stdout
            if 'OLD_ALIVE' not in out and out.count('OLD_GONE') == workers:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Old worker remains; TPU retained, new trainer not launched')
            time.sleep(10)
        registry.collect_live_loss(item.registry)
        final = closeout.latest_committed_checkpoint(item)
        state['old_last_logged_step'] = max(registry.load_loss_points(RUN))
        assert final is not None and final >= state['old_last_logged_step'], (final, state['old_last_logged_step'])
        cursor = json.loads(closeout.run_command(['gsutil', 'cat', f'{item.checkpoint_root}/{final}/skip_file_and_step.json']).stdout)
        newest = json.loads(closeout.run_command(['gsutil', 'cat', f'{item.checkpoint_root}/skip_file_and_step.json']).stdout)
        assert cursor == newest and cursor['checkpoint_step'] == final
        assert cursor['global_batch_size'] == 128 and cursor['num_infeed_hosts'] == workers
        state.update(checkpoint=final, data_cursor=cursor)
        gcs_log = f'gs://newproject-1-llm_projects_us-east5/log/diagnostics/qknorm_minimal_resume/{new_commit[:7]}/{old_mode}-before-{final}.log'
        copy_log = shlex.join(['gsutil', '-q', 'cp', item.registry['train_log'], gcs_log])
        closeout.run_command(['gcloud', 'compute', 'tpus', 'tpu-vm', 'ssh', '--internal-ip', item.tpu,
            f'--zone={item.zone}', f'--project={item.project}', '--worker=0', f'--command={copy_log}'], timeout=120)
        state['old_worker_log_gcs'] = gcs_log
        shutil.copy2(item.registry['launcher_log'], folder/f'{RUN}-{old_mode}-before-{final}.launcher.log')
        pid_file = ROOT/'logs'/f'{RUN}.pid'
        if pid_file.exists() and pid_file.read_text().strip() == str(item.registry.get('auto_pid')):
            pid_file.unlink()
        save('checkpoint_committed_all_workers_quiescent')
        closeout.run_command(launch)
        save('new_controller_submitted')
        print(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
