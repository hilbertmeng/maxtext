"""Run production rescue code with real Orbax faults and OS SIGTERM on two CPU ranks.

No TPU backend, training process, formal checkpoint path or cloud lifecycle is used.
Each arm owns exact child PIDs and a temporary local checkpoint directory.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time

os.environ['JAX_PLATFORMS'] = 'cpu'
# Also supports staging this file beside checkpoint_recovery.py on a worker.
source_dir = Path(__file__).resolve().parents[2] / 'MaxText'
if source_dir.is_dir():
  sys.path.insert(0, str(source_dir))


def child(args):
  cpus = sorted(os.sched_getaffinity(0))
  os.sched_setaffinity(0, cpus[args.rank*2:args.rank*2+2] or cpus[:2])
  import importlib.metadata
  import jax
  import numpy as np
  import orbax.checkpoint as ocp
  from orbax.checkpoint._src.path import atomicity
  from jax.experimental import multihost_utils
  from checkpoint_recovery import CoordinatedCheckpointSaver

  root = Path(args.root)
  from etils import epath
  checkpoint_root = epath.Path(args.checkpoint_root or str(root / 'checkpoints'))
  def event(name, **details):
    row = dict(event=name, rank=args.rank, **details)
    with (root / f'events-{args.rank}.jsonl').open('a') as f:
      f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)
  jax.distributed.initialize(coordinator_address=f'127.0.0.1:{args.port}', num_processes=2,
      process_id=args.rank, initialization_timeout=30, heartbeat_timeout_seconds=20,
      shutdown_timeout_seconds=5)
  event('initialized', jax=jax.__version__, orbax=importlib.metadata.version('orbax-checkpoint'))
  manager = ocp.CheckpointManager(str(checkpoint_root), item_names=('items',),
      item_handlers={'items': ocp.PyTreeCheckpointHandler()},
      options=ocp.CheckpointManagerOptions(save_interval_steps=1000, max_to_keep=2,
          temporary_path_class=atomicity.CommitFileTemporaryPath,
          async_options=ocp.options.AsyncOptions(timeout_secs=args.save_timeout, create_directories_asynchronously=False)))
  current = 2000
  def save(target, force):
    return manager.save(target, args=ocp.args.Composite(items=ocp.args.PyTreeSave({
        'step': np.asarray(current+1), 'params': np.full(4, current, np.float32),
        'optimizer': np.full(4, current*2, np.float32)})), force=force)
  cursor_failed = False
  def cursor(target):
    nonlocal cursor_failed
    if args.rank == 0:
      if args.arm == 'cursor' and target > 3000 and not cursor_failed:
        cursor_failed = True
        raise OSError('INJECTED_RANK0_CURSOR_FAILURE')
      value = dict(checkpoint_step=target, consumed=current+1)
      (checkpoint_root / str(target) / 'skip_file_and_step.json').write_text(json.dumps(value))
  def verify(target):
    restored = manager.restore(target)
    assert int(restored['items']['step']) == current+1
    np.testing.assert_array_equal(restored['items']['params'], np.full(4, current, np.float32))
    np.testing.assert_array_equal(restored['items']['optimizer'], np.full(4, current*2, np.float32))
    value = json.loads((checkpoint_root / str(target) / 'skip_file_and_step.json').read_text())
    assert value == dict(checkpoint_step=target, consumed=current+1)
    assert (checkpoint_root / str(target) / 'commit_success.txt').exists()
    event('restored_and_verified', step=target, state_step=current+1, cursor=value)
  save(2000, True)
  cursor(2000)
  manager.wait_until_finished()
  if args.arm not in ('healthy', 'cursor'):
    am = manager._checkpointer._async_manager
    original = am.start_async_commit
    injected = False
    class TimeoutFuture:
      def result(self, timeout=None):
        assert timeout is not None
        time.sleep(max(0, timeout))
        raise TimeoutError('INJECTED_SAVE_TIMEOUT')
      def cancel(self):
        return False
    def start(directory, commit_futures, on_commit_callback):
      nonlocal injected
      if args.rank == 0 and (not injected or args.arm == 'persistent'):
        commit_futures = [TimeoutFuture()] + list(commit_futures)
        injected = True
      return original(directory, commit_futures, on_commit_callback)
    am.start_async_commit = start
  current = 3000
  save(3000, True)
  cursor(3000)
  if args.arm in ('healthy', 'cursor'):
    manager.wait_until_finished()
  elif args.arm != 'early':
    time.sleep(args.save_timeout + .5)
  event('ready', old_async_alive=bool(manager._checkpointer._async_manager._thread and
        manager._checkpointer._async_manager._thread.is_alive()))
  (root / f'ready-{args.rank}').touch()
  saver = CoordinatedCheckpointSaver(manager, log=lambda line: event('protocol', message=line), verify=verify)
  for current in range(3100, 3300):
    multihost_utils.sync_global_devices(f'train_{current}')
    if saver.save(current, save, cursor):
      event('healthy_exit', step=current)
      manager.close()
      jax.distributed.shutdown()
      return
    time.sleep(.03)
  raise AssertionError('SIGTERM did not request a checkpoint')


def run_arm(arm, budget, save_timeout=1, gcs_prefix=None):
  with tempfile.TemporaryDirectory(prefix='xd-production-rescue-') as temp:
    root = Path(temp)
    with socket.socket() as sock:
      sock.bind(('127.0.0.1', 0))
      port = sock.getsockname()[1]
    processes, outputs = [], []
    checkpoint_root = gcs_prefix.rstrip('/') + '/' + root.name if gcs_prefix else None
    result = dict(arm=arm, checkpoint_root=checkpoint_root, save_timeout=save_timeout)
    environment = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1')
    try:
      for rank in range(2):
        output = (root / f'worker-{rank}.log').open('w')
        outputs.append(output)
        processes.append(subprocess.Popen([sys.executable, __file__, '--rank', str(rank),
            '--root', temp, '--port', str(port), '--arm', arm, '--save-timeout', str(save_timeout),
            *(['--checkpoint-root', checkpoint_root] if checkpoint_root else [])], env=environment,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True))
      deadline = time.monotonic()+budget
      while not all((root / f'ready-{rank}').exists() for rank in range(2)):
        if time.monotonic() >= deadline or any(p.poll() is not None for p in processes):
          raise RuntimeError('Children failed before signal')
        time.sleep(.02)
      # Real signal to rank0 only; JAX propagates the distributed preemption.
      os.kill(processes[0].pid, signal.SIGTERM)
      while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(.05)
      result['timed_out'] = any(p.poll() is None for p in processes)
    except Exception as error:
      result['parent_error'] = repr(error)
      result['timed_out'] = True
    finally:
      for process in processes:
        if process.poll() is None:
          os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)
      for output in outputs:
        output.close()
      result['exit_codes'] = [p.returncode for p in processes]
      if checkpoint_root:
        listing = subprocess.run(['gsutil', 'ls', checkpoint_root + '/*/commit_success.txt'], capture_output=True, text=True, timeout=60)
        result['committed_steps'] = sorted(int(line.split('/')[-2]) for line in listing.stdout.splitlines() if line.endswith('/commit_success.txt'))
      else:
        result['committed_steps'] = sorted(int(p.parent.name) for p in (root / 'checkpoints').glob('*/commit_success.txt'))
      result['ranks'] = [dict(rank=r, log=(root / f'worker-{r}.log').read_text(),
          events=[json.loads(l) for l in (root / f'events-{r}.jsonl').read_text().splitlines()]
          if (root / f'events-{r}.jsonl').exists() else []) for r in range(2)]
    if checkpoint_root:
      cleanup = subprocess.run(['gsutil', '-m', 'rm', '-r', checkpoint_root], capture_output=True, text=True, timeout=120)
      result['cleanup_exit'] = cleanup.returncode
      if cleanup.returncode:
        raise RuntimeError('Could not remove private GCS probe prefix: ' + checkpoint_root)
    return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--arm', choices=['healthy', 'expired', 'early', 'persistent', 'cursor'], action='append')
  parser.add_argument('--rank', type=int)
  parser.add_argument('--root')
  parser.add_argument('--checkpoint-root')
  parser.add_argument('--gcs-prefix')
  parser.add_argument('--save-timeout', type=int, default=1)
  parser.add_argument('--port', type=int)
  parser.add_argument('--budget', type=int, default=60)
  args = parser.parse_args()
  if args.rank is not None:
    args.arm = args.arm[0]
    child(args)
    return
  failed = False
  for arm in args.arm or ['healthy', 'expired', 'early', 'persistent', 'cursor']:
    result = run_arm(arm, args.budget, args.save_timeout, args.gcs_prefix)
    print(json.dumps(result), flush=True)
    expected = [0, 0] if arm == 'healthy' else ([1, 1] if arm == 'persistent' else [75, 75])
    failed |= result['exit_codes'] != expected or result['timed_out']
    if arm != 'persistent':
      failed |= not all(any(e['event'] == 'restored_and_verified' for e in rank['events']) for rank in result['ranks'])
    else:
      failed |= result['committed_steps'] != [2000]
  if failed:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
