"""bam_replicate_rank_up changes only the sharding of rank-input up projections."""
import unittest

import jax
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import train_compile
from bam_prop_test_utils import MLPWriteTest

REPLICATED = ('W_lq_c8_up', 'W_lk_c8_up', 'W_lv_c8_up', 'W_R_up', 'P_loc_up', 'mlp_address_up')


class ReplicateRankUpTest(MLPWriteTest, unittest.TestCase):
  def shaped(self, run):
    cfg = self.config(run)
    cfg.get_keys().update(max_target_length=128, max_prefill_predict_length=128, query_chunk_size=64)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    shaped, *_ = train_compile.get_shaped_inputs(mesh, cfg)
    return {p: v for p, v in flatten_dict(shaped[0].params).items()}

  def test_same_tree_and_only_rank_up_kernels_replicated(self):
    for base in ('BamXLPropV5SiTU47SpeedProfile', 'BamXLPropV5SiTU47M60R480SpeedProfile'):
      native = self.shaped(base)
      replicated = self.shaped(base.replace('SpeedProfile', 'ReplicateRankUpSpeedProfile'))
      self.assertEqual(native.keys(), replicated.keys())
      changed = 0
      for path, leaf in native.items():
        other = replicated[path]
        self.assertEqual((leaf.shape, leaf.dtype), (other.shape, other.dtype), path)
        spec, new_spec = tuple(leaf.sharding.spec), tuple(other.sharding.spec)
        if path[-1] == 'kernel' and path[-2] in REPLICATED:
          self.assertIsNotNone(spec[0], path)   # rank axis; scanned kernels put layers at axis 1
          self.assertIsNone(new_spec[0], path)
          self.assertEqual(new_spec[1:], spec[1:], path)
          changed += 1
        else:
          self.assertEqual(spec, new_spec, path)
      self.assertGreaterEqual(changed, len(REPLICATED), base)


if __name__ == '__main__':
  unittest.main()
