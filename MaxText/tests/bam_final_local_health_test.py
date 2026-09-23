"""An unscanned final L must export its own gates, not a scanned block's gates."""
import unittest
from types import SimpleNamespace
import numpy as np
from train import record_bam_concat_health_metrics


class BamFinalLocalHealthTest(unittest.TestCase):
  def test_final_local_is_exported_with_absolute_layer_index(self):
    def attention(value):
      return {'block': {'self_attention': {'concat_local_q_gate': (value,)}}}
    scanned = np.arange(45, dtype=np.float32).reshape(9, 5)
    decoder = {'layers': {name: attention(scanned + offset * 100)
                          for offset, name in enumerate(('local_0', 'local_1', 'fetch_2'))},
               'final_local_layer': attention(np.arange(5, dtype=np.float32) + 1000)}
    intermediates = {'intermediates': {'decoder': decoder}}
    for tail in (False, True):
      with self.subTest(tail=tail):
        config = SimpleNamespace(bam_local_fetch_block_size=3,
                                 num_decoder_layers=27 + int(tail),
                                 bam_extra_final_local_layer=tail)
        output = {'scalar': {}}
        record_bam_concat_health_metrics(output, intermediates, config)
        metrics = output['scalar']
        self.assertEqual(len(metrics), 5 * config.num_decoder_layers)
        self.assertEqual(metrics['bam/concat/local_q_gate/layer_026/mean'], 240)
        if tail:
          self.assertEqual(metrics['bam/concat/local_q_gate/layer_027/mean'], 1000)
          self.assertEqual(metrics['bam/concat/local_q_gate/layer_027/frac_gt_095'], 1004)


if __name__ == '__main__':
  unittest.main()
