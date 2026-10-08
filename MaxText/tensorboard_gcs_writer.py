"""Keep transient GCS failures from killing TensorBoardX's event thread.

Each event file belongs to one writer. Re-uploading its complete byte prefix is
idempotent; no model, optimizer, data iterator or JAX computation is involved.
"""
import hashlib
import logging
import os
from pathlib import Path
import tempfile
import time

from google.api_core import exceptions, retry
from requests.exceptions import RequestException
from tensorboardX import record_writer


class ResilientGCSRecordWriter(record_writer.GCSRecordWriter):
  def __init__(self, path):
    super().__init__(path)
    self._uploaded_size = -1
    self._retry_after = 0.0
    key = hashlib.sha256(path.encode()).hexdigest()[:20]
    self._spool = Path(tempfile.gettempdir()) / 'maxtext-tb-spool' / key / Path(path).name

  def _save_spool(self, data):
    self._spool.parent.mkdir(parents=True, exist_ok=True)
    temporary = self._spool.with_suffix(self._spool.suffix + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, self._spool)

  def flush(self):
    size = self.buffer.tell()
    if size == self._uploaded_size:
      return
    data = self.buffer.getvalue()
    if time.monotonic() < self._retry_after:
      self._save_spool(data)
      return
    try:
      # The enclosing EventsWriter lock serializes flush/write. This unique
      # event object has a single owner; unconditional complete-prefix PUT is
      # safe to retry even if a preceding PUT committed but its reply was lost.
      self.blob.upload_from_string(data=data, timeout=10,
          retry=retry.Retry(initial=1, maximum=4, multiplier=2, deadline=15))
    except (exceptions.GoogleAPICallError, RequestException) as exc:
      self._save_spool(data)
      self._retry_after = time.monotonic() + 30
      logging.warning('TB_GCS_RETRY path=%s spool=%s error=%s', self.path, self._spool, exc)
      return  # Keep the event thread alive and the complete buffer intact.
    self._uploaded_size = size
    self._retry_after = 0.0
    self._spool.unlink(missing_ok=True)

  def close(self):
    if hasattr(self, '_retry_after'):
      self._retry_after = 0.0
      self.flush()


class _Factory:
  @staticmethod
  def directory_check(path):
    pass

  @staticmethod
  def open(path):
    return ResilientGCSRecordWriter(path)


def install():
  record_writer.register_writer_factory('gs', _Factory())
