"""CPU-only fault injection for an isolated Orbax CheckpointManager.

Run as a standalone process with the exact worker environment. It creates only
small NumPy checkpoints under a temporary local directory. It never restores a
training checkpoint or joins an existing distributed JAX runtime. Private APIs
are deliberately inspected here to reproduce exception propagation, not used as
a production recovery policy.
"""
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import time
import tempfile
import json
import pathlib
import importlib.metadata
import inspect
import numpy as np
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint._src.checkpointers import async_checkpointer


def main():
 assert jax.process_count() == 1, "Use a separate, single-process CPU invocation"
 if "timeout_secs" not in inspect.signature(async_checkpointer._background_wait_for_commit_futures).parameters:
  raise RuntimeError("This probe requires the cumulative async timeout implementation (Orbax PR2966)")
 rows=[];start=time.monotonic()
 def event(name,**data):
  v=dict(event=name,elapsed=round(time.monotonic()-start,3),**data);rows.append(v);print('PROBE_EVENT',json.dumps(v),flush=True)
 class TimeoutFuture:
  def result(self,timeout=None):
   if timeout is None:raise AssertionError('Expected bounded wait')
   time.sleep(max(0.,timeout))
   raise TimeoutError('INJECTED_COMMIT_TIMEOUT')
  def cancel(self):return False

 event('environment',orbax=importlib.metadata.version('orbax-checkpoint'))
 with tempfile.TemporaryDirectory(prefix='xd-orbax-timeout-') as root:
  manager=ocp.CheckpointManager(root,item_names=('items',),item_handlers={'items':ocp.PyTreeCheckpointHandler()},options=ocp.CheckpointManagerOptions(save_interval_steps=1000,enable_async_checkpointing=True,async_options=ocp.options.AsyncOptions(timeout_secs=1,create_directories_asynchronously=False)))
  am=manager._checkpointer._async_manager
  original=am.start_async_commit;injected=False
  def start_async_commit(directory,commit_futures,on_commit_callback):
   nonlocal injected
   if not injected:
    commit_futures=[TimeoutFuture()]+list(commit_futures);injected=True
   return original(directory,commit_futures,on_commit_callback)
  am.start_async_commit=start_async_commit
  def args(n):return ocp.args.Composite(items=ocp.args.PyTreeSave({'step':np.asarray(n),'params':np.ones(4,np.float32)*n}))
  event('initial_save_return',accepted=manager.save(0,args=args(0)))
  time.sleep(1.5)
  finalizer=manager._finalize_thread.get()
  event('after_deadline',async_thread_alive=am._thread is not None and am._thread.is_alive(),finalizer_alive=finalizer.is_alive(),finalizer_error=type(finalizer.exception).__name__ if finalizer.exception else None,committed=pathlib.Path(root,'0','commit_success.txt').exists())
  out=[]
  for n in [1,2,3]:out.append({'step':n,'accepted':manager.save(n,args=args(n))})
  event('non_checkpoint_steps',outcomes=out)
  try:
   manager.check_for_errors();event('public_check_for_errors',result='returned_without_error')
  except BaseException as e:event('public_check_for_errors',result=type(e).__name__,error=str(e))
  try:
   manager.save(1000,args=args(1000));event('next_scheduled_save',result='returned_without_error')
  except BaseException as e:event('next_scheduled_save',result=type(e).__name__,error=str(e))
  try:
   accepted=manager.save(1001,args=args(1001),force=True);manager.wait_until_finished();restored=manager.restore(1001);event('explicit_retry_after_catch',accepted=accepted,result='success',restored_step=int(restored['items']['step']))
  except BaseException as e:event('explicit_retry_after_catch',result=type(e).__name__,error=str(e))
  try:manager.close()
  except BaseException as e:event('manager_close_error',error=str(e))
 # Probe the boundary of the advertised timeout: callbacks are invoked directly.
 a=async_checkpointer._AsyncManager(barrier_sync_fn=lambda **kw:None,timeout_secs=1,primary_host=0)
 t=time.monotonic();a.start_async_commit(pathlib.Path('/tmp/xd-not-a-real-checkpoint'),[],lambda:time.sleep(2.2));time.sleep(1.3)
 event('callback_after_deadline',thread_alive=a._thread.is_alive(),error=type(a._exception).__name__ if a._exception else None)
 try:a.wait_until_finished();event('callback_completed',duration=round(time.monotonic()-t,3),result='no_timeout')
 except BaseException as e:event('callback_completed',duration=round(time.monotonic()-t,3),result=type(e).__name__)
 print('PROBE_RESULTS',json.dumps(rows),flush=True)


if __name__ == "__main__":
 main()
