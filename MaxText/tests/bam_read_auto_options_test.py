"""'auto' resolution of the v8t read options and of rank-input up-projection replication."""
import math
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import maxtext_utils
from layers import attentions
from layers import bam_pallas_v8 as v8


def _mesh(fsdp):
  return SimpleNamespace(shape={'data': 1, 'fsdp': fsdp, 'fsdp_transpose': 1, 'sequence': 1,
                                'tensor_transpose': 1, 'expert': 1})


def _cfg(mode='auto', layers=28, heads=20, v=40, rank=400):
  return SimpleNamespace(bam_replicate_rank_up=mode, num_decoder_layers=layers, num_query_heads=heads, bam_v=v,
                         bam_local_full_m_read_bottleneck_dim=rank, bam_write_v_bottleneck_dim=rank,
                         bam_mlp_write_address_rank=rank)


class RankUpTest(unittest.TestCase):
  def test_auto_skips_replication_that_would_cost_too_much_hbm(self):
    self.assertTrue(attentions.replicate_rank_up(_cfg(), _mesh(16), 400))                 # XL: 0.8 GiB
    pythia = _cfg(layers=32, heads=32, v=64, rank=640)                                    # 2.8B: 3.75 GiB
    self.assertGreater(attentions.rank_up_replicated_gib(pythia), attentions.RANK_UP_MAX_REPLICATED_GIB)
    self.assertFalse(attentions.replicate_rank_up(pythia, _mesh(64), 640))
    self.assertTrue(attentions.replicate_rank_up(_cfg(True, layers=32, heads=32, v=64, rank=640), _mesh(64), 640))

  def test_auto_threshold_and_forcing(self):
    auto = _cfg()
    self.assertFalse(attentions.replicate_rank_up(auto, _mesh(4), 400))     # 100 rows per shard
    self.assertTrue(attentions.replicate_rank_up(auto, _mesh(16), 480))     # 30 rows per shard
    self.assertFalse(attentions.replicate_rank_up(auto, _mesh(1), 32))      # nothing to shard
    self.assertTrue(attentions.replicate_rank_up(SimpleNamespace(bam_replicate_rank_up=True), _mesh(4), 400))
    self.assertFalse(attentions.replicate_rank_up(SimpleNamespace(bam_replicate_rank_up=False), _mesh(16), 480))

  def test_sharding_check_counts_designed_replication_as_sharded(self):
    if jax.device_count() != 4:
      self.skipTest('Requires --xla_force_host_platform_device_count=4')
    mesh = Mesh(np.array(jax.devices()).reshape(4, 1, 1, 1, 1, 1, 1, 1),
                ('fsdp', 'fsdp_transpose', 'sequence', 'tensor', 'tensor_transpose', 'tensor_sequence',
                 'stage', 'expert'))
    sharded = jax.device_put(jnp.zeros((64, 8)), NamedSharding(mesh, P('fsdp')))
    replicated = jax.device_put(jnp.zeros((64, 8)), NamedSharding(mesh, P()))
    params = {'mlp': {'kernel': sharded}, 'W_lq_c8_up': {'kernel': replicated}}
    with self.assertRaises(AssertionError):
      maxtext_utils.assert_params_sufficiently_sharded(params, mesh, .02)
    maxtext_utils.assert_params_sufficiently_sharded(
        params, mesh, .02, replicated_by_design=lambda p: p[-2].key in attentions.RANK_UP_KERNELS)


class StrideTest(unittest.TestCase):
  def test_auto_skew_only_off_multiples_of_32_within_vmem(self):
    opts = lambda mode, vmem=63: {'stride_skew': mode, 'vmem_mib': vmem}
    small, big = (lambda cp: 30 * 2 ** 20), (lambda cp: 45 * 2 ** 20)
    self.assertEqual(v8._stride(60, opts('auto'), small), 72)
    self.assertEqual(v8._stride(64, opts('auto'), big), 64)       # 45 MiB x overhead exceeds 63
    self.assertEqual(v8._stride(64, opts(True), big), 72)         # forced
    self.assertEqual(v8._stride(40, opts('auto'), small), 40)     # 40 is not a multiple of 32
    self.assertEqual(v8._stride(60, opts(None), small), 64)


class CarryAutoTest(unittest.TestCase):
  def _grads(self, b, mode):
    t, n, v, k, rs = 128, 4, 16, 128, 8
    keys = iter(jax.random.split(jax.random.key(3), 16))
    rand = lambda shape: jax.random.normal(next(keys), shape)
    m = rand((b, t, v, k))
    xs = (rand((b, t, n, v)), rand((b, t, n)), rand((b, t, n, v)), rand((b, t, n)), rand((b, t, 2 * n, v)),
          rand((b, t, n)), rand((b, t, n)), rand((b, t, n, rs)), rand((b, t, n, rs)))
    shift = jax.nn.sigmoid(rand((b, t, v)))
    static = rand((4 * n, v)) * .3

    def loss(m):
      outs = v8.read_tm(m, static, *xs, qk_cols=k - rs, read_epsilon=1e-4, key_scale=.1, forward_tile=64,
                        reverse_tile=64, interpret=True, shift=shift, chunk=8, shift_carry=mode)
      return sum(jnp.sum(o * (i + 1)) for i, o in enumerate(outs))
    return jax.jit(jax.grad(loss))(m)

  def test_auto_matches_explicit_choice_per_batch(self):
    np.testing.assert_array_equal(self._grads(1, 'auto'), self._grads(1, False))
    np.testing.assert_array_equal(self._grads(2, 'auto'), self._grads(2, True))


if __name__ == '__main__':
  unittest.main()
