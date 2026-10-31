"""Bounded, collective checkpoint rescue using the live training state.

Only the training thread calls this code. An external SIGTERM requests the existing
JAX preemption save; it does not run Python checkpoint code in a signal handler.
The external watchdog remains the hard wall-clock bound if a rank cannot return.
"""
import logging


# Distinguish a resumable checkpoint rescue from a successful end of training.
RECOVERY_EXIT_CODE = 75
RECOVERY_FAILED_EXIT_CODE = 76


class CheckpointRecoveryError(RuntimeError):
  """Checkpoint rescue failed; restart from verified persistent state."""


class _RetryableSaveError(Exception):
  pass


def gather_checkpoint_status(step, status):
  """All ranks participate, including ranks without a local exception."""
  import jax
  import numpy as np
  from jax.experimental import multihost_utils

  local = np.asarray([step, status], dtype=np.int32)
  if jax.process_count() == 1:
    return local.reshape(1, 2)
  return np.asarray(multihost_utils.process_allgather(local, tiled=False)).reshape(-1, 2)


class CoordinatedCheckpointSaver:
  """Keep normal saves asynchronous; make one rescue attempt, then exit.

  A previous save is drained collectively BEFORE any rank starts the next one.
  Otherwise a healthy rank could start writing while a failed rank retries.
  Retry uses step+1 as a fresh directory, with exactly the same in-memory state
  and cursor; it never performs another training step or reuses a failed prefix.
  """

  def __init__(self, manager, *, gather=gather_checkpoint_status, log=logging.warning, verify=None):
    self.manager = manager
    self.gather = gather
    self.log = log
    self.verify = verify or (lambda step: None)

  def _phase(self, step, name, action):
    error = None
    result = None
    try:
      result = action()
    except Exception as caught:  # SystemExit/KeyboardInterrupt must propagate.
      error = caught
      self.log(f"CHECKPOINT_PHASE_ERROR step={step} phase={name} error={caught!r}")
    status = 0 if error is None else (1 if isinstance(error, (TimeoutError, OSError)) else 2)
    votes = self.gather(step, status)
    if any(int(row[0]) != step for row in votes):
      raise CheckpointRecoveryError(f"Checkpoint ranks disagree on step during {name}")
    worst = max(int(row[1]) for row in votes)
    if worst == 2:
      raise CheckpointRecoveryError(f"Non-retryable checkpoint failure at {step}, {name}") from error
    if worst == 1:
      raise _RetryableSaveError(f"Checkpoint failure at {step}, {name}") from error
    return result

  def _submit(self, step, save, cursor, *, force):
    def submit():
      if not save(step, force):
        raise CheckpointRecoveryError(f"Expected checkpoint {step} to be accepted")
    self._phase(step, 'submit', submit)
    self._phase(step, 'cursor', lambda: cursor(step))

  def _complete(self, step):
    self._phase(step, 'wait', self.manager.wait_until_finished)
    self._phase(step, 'verify', lambda: self.verify(step))

  def _rescue(self, step, save, cursor):
    self.log(f"CHECKPOINT_RECOVERY_BEGIN step={step}")
    # A cursor/submit error may have occurred while a new async save was active.
    # Drain it too. A completed failed finalizer may surface its old error here;
    # consuming that error is not itself another retry.
    try:
      self._phase(step, 'rescue_drain', self.manager.wait_until_finished)
    except _RetryableSaveError:
      pass
    rescue_step = step + 1
    try:
      self._submit(rescue_step, save, cursor, force=True)
      self._complete(rescue_step)
    except _RetryableSaveError as error:
      raise CheckpointRecoveryError('One checkpoint rescue attempt failed') from error
    self.log(f"CHECKPOINT_RECOVERY_COMMITTED step={rescue_step}")
    raise SystemExit(RECOVERY_EXIT_CODE)

  def save(self, step, save, cursor, *, force=False, finish=False):
    """Return True when a healthy preemption save finished and trainer may exit."""
    if not force and not self.manager.should_save(step):
      return False
    try:
      self._phase(step, 'previous_save', self.manager.wait_until_finished)
      self._submit(step, save, cursor, force=force)
      reached = self.manager.reached_preemption(step)
      if reached or finish:
        self._complete(step)
      return reached
    except _RetryableSaveError:
      self._rescue(step, save, cursor)

  def finish(self, step, save, cursor):
    """Drain the last async save on normal training completion."""
    try:
      self._phase(step, 'final_wait', self.manager.wait_until_finished)
    except _RetryableSaveError:
      self._rescue(step, save, cursor)
