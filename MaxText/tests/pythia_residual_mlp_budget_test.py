"""Full-tree budgets and live paths for the residual-MLP mechanism controls."""
import math
import unittest

from flax import linen as nn
from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np

from bam_prop_test_utils import MLPWriteTest
from bam_pythia_transfer_test import PythiaTransferTest
from layers.models import Transformer
import max_utils
import train_compile

MHA = 'Pythia2p8BShortTrainResidualMLP7552'
PARENT = 'BamPythia2p8BSotaIndependentR768PallasShortTrainMemoryMLPRatio2to1'
WIDE = PARENT + 'ResidualMLP10240'
R640 = PARENT + 'AllR640'


class ResidualBudgetTest(MLPWriteTest, unittest.TestCase):
  def test_full_parameter_budgets_and_projection_widths(self):
    for exp, expected in ((MHA, 2332810890), (WIDE, 3217324032),
                          (R640, 2774883968)):
      cfg = self.config(exp, target_train_batch_size=None)
      cfg.get_keys().update(max_target_length=256, max_prefill_predict_length=256,
                            query_chunk_size=256)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      shaped, *_ = train_compile.get_shaped_inputs(mesh, cfg)
      flat = flatten_dict(nn.unbox(shaped[0].params))
      total = sum(math.prod(v.shape) for v in flat.values())
      self.assertEqual(total, expected, exp)
      ordinary = sum(math.prod(v.shape) for p, v in flat.items() if 'mlp' in p)
      self.assertEqual(ordinary, 1678131200 if exp == WIDE else
                       1309091456 if exp == R640 else 1235728010)
      self.assertTrue(cfg.record_training_health_metrics)
      self.assertFalse(cfg.record_internal_nn_metrics)
      if exp == MHA:
        self.assertEqual(cfg.num_decoder_layers, 32)
        tail = ('params', 'decoder', 'final_pythia_layer', 'mlp')
        self.assertEqual(flat[tail + ('wi', 'kernel')].shape, (2560, 7178))
        self.assertEqual(flat[tail + ('wo', 'kernel')].shape, (7178, 2560))
        self.assertFalse(any('memory_mlp' in p for p in flat))
        self.assertTrue(cfg.pythia_parallel_residual)
      else:
        self.assertEqual(cfg.bam_memory_mlp_dim, 3072)
        rank = 640 if exp == R640 else 768
        self.assertEqual(cfg.bam_local_full_m_read_bottleneck_dim, rank)
        self.assertEqual(cfg.emb_bam_v_bottleneck_dim, rank)
        self.assertEqual(cfg.bam_write_v_bottleneck_dim, rank)
        self.assertEqual(cfg.bam_mlp_write_address_rank, rank)
        for p, value in flat.items():
          if p[-2:] == ('P_loc_down', 'kernel') or p[-2:] == ('mlp_address_down', 'kernel'):
            self.assertEqual(value.shape[-1], rank, p)
        if exp == R640:
          self.assertEqual(cfg.bam_final_local_layer_count, 2)
          for p, value in flat.items():
            if 'mlp' in p and p[-2:] == ('wi', 'kernel'):
              self.assertEqual(value.shape[-1] % 128, 0, p)
          self.assertLess(abs(total - 2774920842), 5121 * 128 / 2)

  def test_narrow_mha_tail_forward_causality_and_gradients(self):
    cfg = self.config(MHA, target_train_batch_size=None, base_emb_dim=64,
        base_num_decoder_layers=3, base_num_query_heads=2, base_num_kv_heads=2,
        head_dim=32, base_mlp_dim=32, pythia_last_mlp_dim=24,
        pythia_rotary_dim=8, vocab_size=32, dtype='float32',
        weight_dtype='float32', attention='dot_product')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
    positions = jnp.arange(4)[None, :]
    segments = jnp.ones_like(tokens)
    rngs = {'params': jax.random.key(7), 'dropout': jax.random.key(8)}
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params = nn.unbox(model.init(rngs, tokens, positions, tokens, segments,
          decoder_segment_ids=segments, enable_dropout=False)['params'])

      def forward(p, t=tokens):
        return model.apply({'params': p}, t, positions, t, segments,
            decoder_segment_ids=segments, enable_dropout=False, rngs=rngs)[0]

      np.testing.assert_allclose(forward(params, tokens.at[:, 2:].set(9))[:, :2],
                                 forward(params)[:, :2], rtol=3e-5, atol=3e-5)
      loss, grad = jax.jit(jax.value_and_grad(lambda p: jnp.mean(forward(p)**2)))(params)
      self.assertTrue(np.isfinite(float(loss)))
      flat = flatten_dict(grad)
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
      for part in ('layers', 'final_pythia_layer'):
        leaves = [v for p, v in flat.items() if part in p and 'mlp' in p]
        self.assertTrue(leaves, part)
        self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves), 0.)

  tiny = PythiaTransferTest.tiny
  check_forward_backward = PythiaTransferTest.check_forward_backward

  def test_all_r640_causality_and_writer_gradients(self):
    self.check_forward_backward(R640)


if __name__ == '__main__':
  unittest.main()
