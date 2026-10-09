"""Keep the successful Medium V48/C12 configuration runnable on the shared core."""
import functools
import unittest
import jax
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from bam_prop_test_utils import MLPWriteTest

EXP = 'BamMediumPropK75V48C12EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'

class PropAddressWidthTest(MLPWriteTest, unittest.TestCase):
  def test_v48_budget_and_training_health(self):
    config = self.config(EXP)
    self.assertEqual((config.bam_k, config.bam_v, config.bam_abs_v_compression_dim), (75, 48, 12))
    self.assertEqual(config.bam_mlp_write_address_rank, 384)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(config), config.mesh_axes)
    args, kwargs, sharding, model = train_compile.get_shaped_inputs(mesh, config)
    params = flatten_dict(args[0].params)
    self.assertEqual(sum(int(np.prod(p.shape)) for p in params.values()), 432115072)
    for path, value in params.items():
      if path[-1] in ('static_v_key', 'static_o_key'):
        self.assertEqual((value.shape[0], value.shape[-1]), (48, 16))
    with mesh, nn.partitioning.axis_rules(config.logical_axis_rules):
      metrics = jax.eval_shape(functools.partial(train.train_step, model, config, sharding),
                              *args, **kwargs)[1]['scalar']
    for layer in range(18):
      self.assertEqual(f'bam/concat/mlp_write_gate/layer_{layer:03d}/mean' in metrics,
                       layer % 3 == 1)

if __name__ == '__main__':
  unittest.main()
