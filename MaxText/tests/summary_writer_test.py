"""Host-only regression tests for bounded GCS event uploads."""

import tempfile
import unittest
from pathlib import Path

from summary_writer import create_summary_writer


class FakeWriter:
  def __init__(self, logdir, filename_suffix=""):
    self.logdir = logdir
    self.suffix = filename_suffix
    self.events = []
    self.uploads = []
    self.closed = False

  def add_scalar(self, tag, value, step):
    self.events.append((tag, value, step))

  def flush(self):
    self.uploads.append(len(self.events))

  def close(self):
    self.flush()
    self.closed = True


class SummaryWriterTest(unittest.TestCase):
  def test_rotated_uploads_keep_all_steps_and_bound_work(self):
    writers = []

    def factory(*args, **kwargs):
      writers.append(FakeWriter(*args, **kwargs))
      return writers[-1]

    wrapped = create_summary_writer("gs://bucket/run", 10, factory)
    expected = []
    for step in range(35):
      wrapped.begin_step(step)
      wrapped.begin_step(step)  # Multiple metric groups at the same step.
      for tag in ("loss", "health"):
        event = (tag, step / 100, step)
        expected.append(event)
        wrapped.add_scalar(*event)
      wrapped.flush()
    wrapped.close()
    wrapped.close()
    self.assertEqual([event for w in writers for event in w.events], expected)
    self.assertEqual([len(w.events) for w in writers], [20, 20, 20, 10])
    self.assertEqual(len({w.suffix for w in writers}), 4)
    self.assertTrue(all(w.closed for w in writers))
    self.assertLessEqual(max(size for w in writers for size in w.uploads), 20)

  def test_resume_skipped_windows_and_eval_do_not_duplicate(self):
    writers = []

    def factory(*args, **kwargs):
      writers.append(FakeWriter(*args, **kwargs))
      return writers[-1]

    wrapped = create_summary_writer("gs://bucket/run", 500, factory)
    for step in (21710, 21990, 22000, 22000, 21999, 23500):
      wrapped.begin_step(step)
      wrapped.add_scalar("loss", 1, step)
    wrapped.close()
    self.assertEqual(len(writers), 3)
    self.assertEqual(sum(len(w.events) for w in writers), 6)

  def test_local_and_disabled_writers_remain_native(self):
    for path, period in (("/tmp/run", 500), ("gs://bucket/run", 0)):
      self.assertIsInstance(create_summary_writer(path, period, FakeWriter), FakeWriter)
    with self.assertRaises(ValueError):
      create_summary_writer("gs://bucket/run", -1, FakeWriter)

  def test_failed_finalization_does_not_silently_advance(self):
    wrapped = create_summary_writer("gs://bucket/run", 10, FakeWriter)
    wrapped.begin_step(9)
    old_writer = wrapped._writer

    def fail():
      raise OSError("upload failed")

    old_writer.close = fail
    with self.assertRaises(OSError):
      wrapped.begin_step(10)
    self.assertIs(wrapped._writer, old_writer)
    self.assertEqual(wrapped._segment, 0)

  def test_real_event_segments_load_as_one_continuous_run(self):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboardX import SummaryWriter

    with tempfile.TemporaryDirectory() as directory:
      def factory(_logdir, **kwargs):
        return SummaryWriter(directory, **kwargs)

      wrapped = create_summary_writer("gs://test/run", 10, factory)
      wrapped.add_text("config", "test")
      for step in range(25):
        wrapped.begin_step(step)
        wrapped.add_scalar("learning/loss", 1 / (step + 1), step)
        wrapped.add_scalar("health/gate", .5, step)
        wrapped.flush()
      wrapped.close()
      self.assertEqual(len(list(Path(directory).glob("events.out.tfevents*"))), 3)
      events = EventAccumulator(directory, size_guidance={"scalars": 0}).Reload()
      for tag in ("learning/loss", "health/gate"):
        self.assertEqual([event.step for event in events.Scalars(tag)], list(range(25)))
      self.assertAlmostEqual(events.Scalars("learning/loss")[-1].value, .04)


if __name__ == "__main__":
  unittest.main()
