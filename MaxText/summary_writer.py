"""Bound TensorBoardX's whole-file GCS uploads by rotating event files."""

import uuid


class RotatingSummaryWriter:
  """Keep all metrics for a step together in bounded-length event segments.

  TensorBoardX's GCS backend reuploads the entire active event on every flush.
  Closing a segment finalizes it; TensorBoard reads the segments as one run.
  """

  def __init__(self, logdir, period, writer_factory):
    self._logdir = logdir
    self._period = period
    self._factory = writer_factory
    self._session = uuid.uuid4().hex
    self._file_index = 0
    self._segment = None
    self._closed = False
    self._writer = self._open_writer()

  def _open_writer(self):
    # The suffix prevents collisions even if two segments open in one second.
    suffix = f".part-{self._session}-{self._file_index:05d}"
    return self._factory(self._logdir, filename_suffix=suffix)

  def begin_step(self, step):
    """Rotate before any scalar for this step, including evaluation metrics."""
    if self._closed:
      raise RuntimeError("Cannot write to a closed summary writer")
    segment = int(step) // self._period
    if self._segment is None:
      self._segment = segment
    elif segment > self._segment:
      # A failed final upload must propagate; never silently discard a segment.
      self._writer.close()
      self._file_index += 1
      self._writer = self._open_writer()
      self._segment = segment

  def close(self):
    if not self._closed:
      self._writer.close()
      self._closed = True

  def __getattr__(self, name):
    return getattr(self._writer, name)


def create_summary_writer(logdir, rotation_period, writer_factory):
  """Use rotation for GCS only; zero retains the original writer."""
  if rotation_period < 0:
    raise ValueError("tensorboard_gcs_rotate_period must be non-negative")
  if str(logdir).startswith("gs://") and rotation_period > 0:
    return RotatingSummaryWriter(logdir, rotation_period, writer_factory)
  return writer_factory(logdir)
