"""Collective rescue protocol, including asymmetric failures and cursor safety."""
import unittest
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import Mock

from checkpoint_recovery import CoordinatedCheckpointSaver, CheckpointRecoveryError


class CheckpointRecoveryTest(unittest.TestCase):
  def make(self, *, preempt=False):
    self.manager = Mock()
    self.manager.should_save.return_value = True
    self.manager.reached_preemption.return_value = preempt
    self.save = Mock(return_value=True)
    self.cursor = Mock()
    self.verify = Mock()
    self.log = []
    return CoordinatedCheckpointSaver(self.manager, gather=lambda step, status: [(step, status)],
                                      log=self.log.append, verify=self.verify)

  def test_normal_async_save_does_not_wait_for_new_save(self):
    saver = self.make()
    self.assertFalse(saver.save(3000, self.save, self.cursor))
    self.assertEqual(self.manager.wait_until_finished.call_count, 1)
    self.save.assert_called_once_with(3000, False)
    self.cursor.assert_called_once_with(3000)
    self.verify.assert_not_called()

  def test_non_checkpoint_step_has_no_extra_collective_or_wait(self):
    saver = self.make()
    self.manager.should_save.return_value = False
    saver.save(3001, self.save, self.cursor)
    self.manager.wait_until_finished.assert_not_called()
    self.save.assert_not_called()

  def test_healthy_preemption_waits_and_verifies(self):
    saver = self.make(preempt=True)
    self.assertTrue(saver.save(3118, self.save, self.cursor))
    self.assertEqual(self.manager.wait_until_finished.call_count, 2)
    self.verify.assert_called_once_with(3118)

  def test_failed_previous_save_rescues_fresh_step_and_exits_resumably(self):
    saver = self.make()
    self.manager.wait_until_finished.side_effect = [TimeoutError('old'), None, None]
    with self.assertRaises(SystemExit) as error:
      saver.save(3100, self.save, self.cursor)
    self.assertEqual(error.exception.code, 75)
    self.save.assert_called_once_with(3101, True)
    self.cursor.assert_called_once_with(3101)
    self.verify.assert_called_once_with(3101)

  def test_rank_without_error_also_enters_rescue_before_new_submit(self):
    saver = self.make()
    count = 0
    def gather(step, status):
      nonlocal count
      count += 1
      return [(step, status), (step, 1 if count == 1 else 0)]
    saver.gather = gather
    with self.assertRaises(SystemExit):
      saver.save(3100, self.save, self.cursor)
    self.save.assert_called_once_with(3101, True)

  def test_cursor_write_failure_drains_current_save_and_uses_fresh_directory(self):
    saver = self.make()
    self.cursor.side_effect = [OSError('cursor storage'), None]
    with self.assertRaises(SystemExit):
      saver.save(3100, self.save, self.cursor)
    self.assertEqual(self.save.call_args_list[0].args, (3100, False))
    self.assertEqual(self.save.call_args_list[1].args, (3101, True))
    self.assertEqual(self.manager.wait_until_finished.call_count, 3)

  def test_failure_during_preemption_wait_also_rescues(self):
    saver = self.make(preempt=True)
    self.manager.wait_until_finished.side_effect = [None, TimeoutError('new'), None, None]
    with self.assertRaises(SystemExit):
      saver.save(3100, self.save, self.cursor)
    self.assertEqual(self.save.call_count, 2)
    self.verify.assert_called_once_with(3101)

  def test_persistent_fault_gets_only_one_rescue_attempt(self):
    saver = self.make()
    self.manager.wait_until_finished.side_effect = [TimeoutError('old'), None, TimeoutError('again')]
    with self.assertRaisesRegex(CheckpointRecoveryError, 'One checkpoint rescue'):
      saver.save(3100, self.save, self.cursor)
    self.save.assert_called_once_with(3101, True)
    self.assertFalse(any('RECOVERY_COMMITTED' in line for line in self.log))

  def test_programming_errors_are_not_silently_retried(self):
    saver = self.make()
    self.save.side_effect = ValueError('wrong tree')
    with self.assertRaisesRegex(CheckpointRecoveryError, 'Non-retryable'):
      saver.save(3100, self.save, self.cursor)
    self.assertEqual(self.save.call_count, 1)

  def test_mismatched_steps_fail_before_new_submission(self):
    saver = self.make()
    saver.gather = lambda step, status: [(step, status), (step + 1, 0)]
    with self.assertRaisesRegex(CheckpointRecoveryError, 'disagree'):
      saver.save(3100, self.save, self.cursor)
    self.save.assert_not_called()

  def test_final_wait_failure_preserves_current_state_via_same_rescue(self):
    saver = self.make()
    self.manager.wait_until_finished.side_effect = [TimeoutError('old'), None, None]
    with self.assertRaises(SystemExit):
      saver.finish(28000, self.save, self.cursor)
    self.save.assert_called_once_with(28001, True)

  def test_cursor_verification_failure_never_reports_success(self):
    saver = self.make(preempt=True)
    self.verify.side_effect = ValueError('mismatched cursor')
    with self.assertRaises(CheckpointRecoveryError):
      saver.save(3100, self.save, self.cursor)
    self.assertFalse(any('RECOVERY_COMMITTED' in line for line in self.log))


class CompletedRescueResumeTest(unittest.TestCase):
  def test_completed_rescue_exits_without_another_training_update(self):
    source = Path(__file__).resolve().parents[1] / 'train.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'train_loop')
    for argument in node.args.args:
      argument.annotation = None
    node.returns = None
    manager, writer = Mock(), Mock()
    state = dict(step=28000)
    close = Mock()
    setup = Mock(return_value=(None, writer, manager, None, None, None, None, None, None, state))
    namespace = dict(setup_train_loop=setup, get_first_step=lambda s: s['step'],
        max_logging=Mock(), max_utils=SimpleNamespace(close_summary_writer=close))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    result = namespace['train_loop'](SimpleNamespace(only_eval=False, steps=28000))
    self.assertIs(result, state)
    manager.wait_until_finished.assert_called_once()
    close.assert_called_once_with(writer)


class CursorPublicationTest(unittest.TestCase):
  def writer(self):
    # Exercise the real function without loading TensorFlow or a dataset.
    source = Path(__file__).resolve().parents[1] / 'input_pipeline/_pile_data_processing.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'record_file_and_step')
    namespace = dict(epath=SimpleNamespace(Path=Path), json=json,
                     jax=SimpleNamespace(process_index=lambda: 0), SKIP_STEP_NAME='skip_file_and_step.json')
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['record_file_and_step']

  def test_step_local_cursor_and_global_cursor_match_snapshot(self):
    with tempfile.TemporaryDirectory() as temp:
      (Path(temp) / '3101').mkdir()
      snapshot = SimpleNamespace(meta_dict=dict(file_in_data=2, step_in_file=17))
      self.writer()(3101, SimpleNamespace(checkpoint_dir=temp), snapshot)
      expected = dict(file_in_data=2, step_in_file=17, checkpoint_step=3101)
      for path in (Path(temp), Path(temp) / '3101'):
        self.assertEqual(json.loads((path / 'skip_file_and_step.json').read_text()), expected)

  def test_cursor_write_error_propagates_without_advancing_global_pointer(self):
    with tempfile.TemporaryDirectory() as temp:
      pointer = Path(temp) / 'skip_file_and_step.json'
      pointer.write_text('{"checkpoint_step":2000}')
      with self.assertRaises(OSError):
        self.writer()(3101, SimpleNamespace(checkpoint_dir=temp),
                      SimpleNamespace(meta_dict=dict(file_in_data=2, step_in_file=17)))
      self.assertEqual(json.loads(pointer.read_text()), dict(checkpoint_step=2000))


if __name__ == '__main__':
  unittest.main()
