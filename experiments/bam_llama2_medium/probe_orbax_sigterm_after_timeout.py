"""Isolated two-process CPU test of real SIGTERM after an Orbax save timeout."""
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
import argparse
import json
import pathlib
import signal
import socket
import subprocess
import sys
import tempfile
import time


def worker(args):
    available = sorted(os.sched_getaffinity(0))
    chosen = available[args.rank * 2:args.rank * 2 + 2] or available[:2]
    os.sched_setaffinity(0, chosen)
    import importlib.metadata
    import numpy as np
    import jax
    from jax.experimental import multihost_utils
    import orbax.checkpoint as ocp
    root = pathlib.Path(args.root)
    events = root / f'events-{args.rank}.jsonl'
    start = time.monotonic()
    def event(name, **details):
        row = dict(event=name, rank=args.rank, seconds=round(time.monotonic()-start, 3), **details)
        with events.open('a') as f:
            f.write(json.dumps(row)+'\n')
        print('EVENT', json.dumps(row), flush=True)
    jax.distributed.initialize(coordinator_address=f'127.0.0.1:{args.port}',
                               num_processes=2, process_id=args.rank,
                               initialization_timeout=30, heartbeat_timeout_seconds=20,
                               shutdown_timeout_seconds=5)
    event('distributed_initialized', jax=jax.__version__, orbax=importlib.metadata.version('orbax-checkpoint'))
    manager = ocp.CheckpointManager(str(root/'checkpoints'), item_names=('items',),
        item_handlers={'items':ocp.PyTreeCheckpointHandler()},
        options=ocp.CheckpointManagerOptions(save_interval_steps=1000,
            enable_async_checkpointing=True,
            async_options=ocp.options.AsyncOptions(timeout_secs=1,
                create_directories_asynchronously=False)))
    def state(n):
        return ocp.args.Composite(items=ocp.args.PyTreeSave({
            'step':np.asarray(n), 'params':np.full(4,n,np.float32),
            'optimizer':np.zeros(4,np.float32)}))
    manager.save(2000,args=state(2000))
    manager.wait_until_finished()
    event('prior_checkpoint_committed',step=2000)
    if args.arm == 'expired':
        am = manager._checkpointer._async_manager
        original = am.start_async_commit
        injected = False
        class TimeoutFuture:
            def result(self,timeout=None):
                if timeout is None:
                    raise AssertionError('Expected a bounded future wait')
                time.sleep(max(0.,timeout))
                raise TimeoutError('INJECTED_3000_COMMIT_TIMEOUT')
            def cancel(self):
                return False
        def start_async_commit(directory,commit_futures,on_commit_callback):
            nonlocal injected
            if not injected and args.rank == 0:
                commit_futures = [TimeoutFuture()]+list(commit_futures)
                injected = True
            return original(directory,commit_futures,on_commit_callback)
        am.start_async_commit = start_async_commit
    manager.save(3000,args=state(3000))
    if args.arm == 'healthy':
        manager.wait_until_finished()
    else:
        time.sleep(1.5)
        if args.settled:
            deadline=time.monotonic()+12
            while manager._finalize_thread.get().is_alive() and time.monotonic()<deadline:
                time.sleep(.05)
    am = manager._checkpointer._async_manager
    ft = manager._finalize_thread.get()
    event('ready_for_sigterm', async_alive=bool(am._thread and am._thread.is_alive()),
          finalizer_alive=bool(ft and ft.is_alive()),
          finalizer_error=type(ft.exception).__name__ if ft and ft.exception else None)
    (root/f'ready-{args.rank}').write_text('ready')
    # This is the same save -> reached_preemption -> wait -> exit ordering
    # as MaxText. The only signal handler is the one JAX installed.
    for step in range(3100, 3300):
        multihost_utils.sync_global_devices(f'probe_train_step_{step}')
        try:
            accepted = manager.save(step,args=state(step))
            reached = manager.reached_preemption(step)
            if accepted or reached:
                event('preemption_save',step=step,accepted=accepted,reached=reached)
            if reached:
                manager.wait_until_finished()
                event('clean_preemption_exit',step=step)
                manager.close()
                jax.distributed.shutdown()
                return
        except BaseException as error:
            event('main_thread_exception',step=step,error=type(error).__name__,message=str(error))
            if args.retry and isinstance(error,TimeoutError):
                # A single coordinated retry only, after every rank has surfaced
                # its old failed-finalizer exception. No error suppression loop.
                multihost_utils.sync_global_devices(f'probe_recovery_{step}')
                accepted=manager.save(step,args=state(step),force=True)
                manager.wait_until_finished()
                restored=manager.restore(step)
                event('coordinated_retry_committed',step=step,accepted=accepted,
                      restored_step=int(restored['items']['step']))
                manager.close()
                jax.distributed.shutdown()
                return
            raise
        time.sleep(.05)
    raise RuntimeError('Real SIGTERM did not produce a preemption exit')


def run_arm(arm, budget, settled=False, retry=False, signal_rank0_only=False):
    with tempfile.TemporaryDirectory(prefix=f'xd-orbax-term-{arm}-') as tmp:
        root = pathlib.Path(tmp)
        with socket.socket() as s:
            s.bind(('127.0.0.1',0))
            port = s.getsockname()[1]
        env = dict(os.environ,JAX_PLATFORMS='cpu',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                   TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
        processes=[]
        files=[]
        result=dict(arm=arm,signal_sent=False,wait_for_finalizer=settled,coordinated_retry=retry,signaled_ranks=[0] if signal_rank0_only else [0,1])
        try:
            for rank in range(2):
                f=(root/f'worker-{rank}.log').open('w');files.append(f)
                cmd=[sys.executable,__file__,'--rank',str(rank),'--arm',arm,'--root',str(root),'--port',str(port)]
                if settled:cmd.append('--settled')
                if retry:cmd.append('--retry')
                processes.append(subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True))
            deadline=time.monotonic()+budget
            while not all((root/f'ready-{rank}').exists() for rank in range(2)):
                if any(p.poll() is not None for p in processes):
                    result['pre_signal_failure']=True
                    break
                if time.monotonic()>deadline:
                    result['ready_timeout']=True
                    break
                time.sleep(.1)
            else:
                for p in (processes[:1] if signal_rank0_only else processes):
                    os.kill(p.pid,signal.SIGTERM)
                result['signal_sent']=True
            while any(p.poll() is None for p in processes) and time.monotonic()<deadline:
                time.sleep(.1)
            if any(p.poll() is None for p in processes):
                result['exit_timeout']=True
        finally:
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGKILL)
                p.wait(timeout=10)
            for f in files:f.close()
            result['exit_codes']=[p.returncode for p in processes]
            result['committed_steps']=sorted(int(p.parent.name) for p in (root/'checkpoints').glob('*/commit_success.txt'))
            result['ranks']=[]
            for rank in range(2):
                p=root/f'events-{rank}.jsonl';log=root/f'worker-{rank}.log'
                result['ranks'].append(dict(rank=rank,events=[json.loads(l) for l in p.read_text().splitlines()] if p.exists() else [],log=log.read_text() if log.exists() else ''))
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int)
    p.add_argument('--arm',choices=['healthy','expired'])
    p.add_argument('--root')
    p.add_argument('--port',type=int)
    p.add_argument('--budget',type=float,default=50)
    p.add_argument('--settled',action='store_true')
    p.add_argument('--retry',action='store_true')
    p.add_argument('--signal-rank0-only',action='store_true')
    a=p.parse_args()
    if a.rank is not None:
        worker(a)
    else:
        for arm in ([a.arm] if a.arm else ['healthy','expired']):
            print('ARM_RESULT',json.dumps(run_arm(arm,a.budget,a.settled,a.retry,a.signal_rank0_only)),flush=True)

if __name__=='__main__':
    main()
