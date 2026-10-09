"""Pinned-CPU configuration helpers for BAM Prop merge regressions."""
import tempfile
from pathlib import Path
import pyconfig

class MLPWriteTest:
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    Path(self.tmp.name, 'audit').mkdir()

  def tearDown(self):
    self.tmp.cleanup()

  def config(self, exp, **kwargs):
    return pyconfig.initialize(
        [None, 'MaxText/configs/base.yml'], exp_class=exp, run_name='audit',
        enable_checkpointing=False, base_output_directory=self.tmp.name + '/',
        jax_cache_dir='', log_config=False, dataset_type='synthetic',
        max_target_length=4, max_prefill_predict_length=4, query_chunk_size=2,
        per_device_batch_size=1., **kwargs)
