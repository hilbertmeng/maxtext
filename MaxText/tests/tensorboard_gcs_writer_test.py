import io, tempfile, unittest, time
from pathlib import Path
from unittest import mock
from google.api_core.exceptions import ServiceUnavailable
from tensorboardX import record_writer, event_file_writer
from tensorboardX.proto.event_pb2 import Event
import tensorboard_gcs_writer as tested

class GCSWriterTest(unittest.TestCase):
  def setUp(self):
    self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
    self.blob=mock.Mock();blob=self.blob
    def init(w,path):w.path=path;w.buffer=io.BytesIO();w.blob=blob
    self.patch=mock.patch.object(record_writer.GCSRecordWriter,'__init__',init);self.patch.start();self.addCleanup(self.patch.stop)
    self.tmp=mock.patch.object(tested.tempfile,'gettempdir',return_value=self.temp.name);self.tmp.start();self.addCleanup(self.tmp.stop)
  def test_failure_retains_complete_prefix_and_recovery(self):
    w=tested.ResilientGCSRecordWriter('gs://test/run/events.foo')
    self.blob.upload_from_string.side_effect=[ServiceUnavailable('503'),None]
    w.write(b'first');w.flush();self.assertEqual(w._spool.read_bytes(),b'first')
    w.write(b'second');w.flush();self.assertEqual(self.blob.upload_from_string.call_count,1)
    self.assertEqual(w._spool.read_bytes(),b'firstsecond')
    w._retry_after=0;w.flush();self.assertEqual(self.blob.upload_from_string.call_args.kwargs['data'],b'firstsecond')
    self.assertNotIn('if_generation_match',self.blob.upload_from_string.call_args.kwargs)
    self.assertFalse(w._spool.exists());w.flush();self.assertEqual(self.blob.upload_from_string.call_count,2)
  def test_event_thread_survives_upload_failure(self):
    self.blob.upload_from_string.side_effect=ServiceUnavailable('503')
    with mock.patch.dict(record_writer.REGISTERED_FACTORIES):
      tested.install();w=event_file_writer.EventFileWriter('gs://test/async',max_queue_size=4,flush_secs=.01)
      try:
        # Would block on the full queue if the upload exception killed its consumer.
        import threading
        done=threading.Event()
        def send():
          for i in range(50):w.add_event(Event(step=i))
          done.set()
        producer=threading.Thread(target=send,daemon=True);producer.start()
        self.assertTrue(done.wait(5));self.assertTrue(w._worker.is_alive())
        self.blob.upload_from_string.side_effect=None
        w._ev_writer._py_recordio_writer._writer._retry_after=0
        w.flush()
      finally:w.close()

if __name__=='__main__':unittest.main()
