"""O-only regularization contracts, independent of the future Pallas adapter."""
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict

import max_utils
import pyconfig
import train_compile
from bam_config import validate_bam_config
from bam_prop_test_utils import MLPWriteTest
from layers.attentions import BamAttention, _split_read_keys
from layers.models import Transformer
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash

PARENT = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile'
HEAD = PARENT.replace('TruePile', 'LocalODropHeadP020TruePile')
HEAD4 = PARENT.replace('TruePile', 'LocalODropHeadP040TruePile')
WEIGHT = PARENT.replace('TruePile', 'LocalOReadWeightDropoutP020TruePile')


class LocalORegularizationTest(MLPWriteTest, unittest.TestCase):
  def fixture(self, run=HEAD, dtype='float32'):
    cfg = self.config(run, base_emb_dim=150, base_num_query_heads=2, base_num_kv_heads=2,
                      dtype=dtype, weight_dtype='float32')
    cfg.get_keys().update(bam_pallas_core=False, bam_splash_attention=False,
                         bam_local_full_m_read_bottleneck_dim=8, bam_write_v_bottleneck_dim=8)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    module = BamAttention(config=cfg, num_query_heads=2, num_kv_heads=2, head_dim=75,
                          bam_k=75, bam_v=32, read_side='col',
                          layer_mode='local_qk+local_v+local_o', layer_inx=0,
                          max_target_length=4, max_prefill_predict_length=4, mesh=mesh,
                          attention_kernel='dot_product_chunk', dtype=cfg.dtype)
    x = jax.random.normal(jax.random.key(1), (4, 4, 150), dtype=cfg.dtype)
    matrix = jax.random.normal(jax.random.key(2), (4, 4, 75, 32), dtype=cfg.dtype)
    args = (x, x, jnp.broadcast_to(jnp.arange(4), (4, 4)), jnp.ones((4, 4), jnp.int32))
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      variables = nn.unbox(module.init(jax.random.key(3), *args, M_in=matrix))
    return cfg, mesh, module, variables, args, matrix

  def test_configs_parameter_shapes_and_budget(self):
    trees = []
    for run in (PARENT, HEAD, WEIGHT, HEAD4):
      cfg = self.config(run)
      cfg.get_keys().update(max_target_length=256, max_prefill_predict_length=256)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, cfg)
      flat = flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(x.shape) for x in flat.values()), 432112192)
      trees.append({key: (x.shape, str(x.dtype)) for key, x in flat.items()})
      self.assertEqual((cfg.mlp_dim, cfg.bam_memory_mlp_dim, cfg.bam_last_mlp_dim), (2510, 1128, 3904))
      self.assertTrue(cfg.enable_dropout and cfg.sa_use_fused_bwd_kernel and cfg.record_training_health_metrics)
      self.assertFalse(cfg.record_internal_nn_metrics)
      if run != PARENT:
        self.assertEqual(cfg.bam_pallas_core, run != WEIGHT)
        self.assertEqual(cfg.compare_runs, [PARENT])
    self.assertEqual(trees[0], trees[1])
    self.assertEqual(trees[0], trees[2])
    self.assertEqual(trees[0], trees[3])
    print('LOCAL_O_REGULARIZATION_IDENTICAL_BUDGET 432112192', flush=True)

  def test_p040_resolved_runtime_and_mask(self):
    cfg, mesh, module, variables, args, matrix = self.fixture(HEAD4)
    self.assertEqual(cfg.bam_local_o_drop_head_rate, .4)
    self.assertEqual(cfg.train_no_dropout_loss_interval, 200)
    self.assertEqual(cfg.train_no_dropout_loss_window, 25)
    self.assertEqual(cfg.train_no_dropout_loss_sample_period, 10)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      x = jnp.ones((128, 4, 2, 75))
      y = module.apply(variables, x, False, method=module._regularize_local_o_heads,
                       rngs={'dropout': jax.random.key(44)})
      self.assertEqual(y.shape, x.shape)
      np.testing.assert_array_equal(y[..., 0], y[..., -1])
      self.assertLess(abs(float(jnp.mean(y == 0)) - .4), .05)
      np.testing.assert_allclose(np.unique(y), [0., 1/.6], rtol=1e-6)
      self.assertFalse(np.array_equal(y[:, 0], y[:, 1]))

  def test_validation(self):
    base = dict(bam_enabled=True, bam_layer_modes=['local_qk+local_v+local_o'])
    for flag in ('bam_local_o_drop_head_rate', 'bam_local_o_read_weight_dropout_rate'):
      for value in (-.1, 1., float('nan'), True):
        with self.assertRaises(ValueError):
          validate_bam_config(dict(base, **{flag: value}))
    cfg = self.config(WEIGHT)
    cfg.get_keys().update(bam_pallas_core=True,bam_pallas_body='v8t')
    validate_bam_config(cfg)
    cfg.get_keys()['bam_pallas_body'] = 'v7'
    with self.assertRaisesRegex(ValueError, 'mask support'):
      validate_bam_config(cfg)
    cfg.get_keys().update(bam_pallas_core=False, bam_local_vo_independent_keys=False)
    with self.assertRaisesRegex(ValueError, 'independent VO keys'):
      validate_bam_config(cfg)

  def test_masks_eval_and_exact_initialization(self):
    cfg, mesh, module, variables, args, matrix = self.fixture()
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def sample(m, x, width, deterministic=False):
        return m._o_dropout_multiplier(x, .2, width, deterministic)
      x = jnp.ones((1024, 4, 150))
      for width in (1, 32):
        mask = module.apply(variables, x, width, method=sample, rngs={'dropout': jax.random.key(4)})
        self.assertEqual(mask.shape, (1024, 4, 2, width))
        self.assertFalse(np.array_equal(mask[:, 0], mask[:, 1]))
        self.assertTrue(set(np.unique(mask)).issubset({0., 1.25}))
        self.assertLess(abs(float(mask.mean()) - 1), .04)
        self.assertIsNone(module.apply(variables, x, width, True, method=sample))
        other = module.apply(variables, x, width, method=sample, rngs={'dropout': jax.random.key(5)})
        self.assertFalse(np.array_equal(mask, other))
      for flag in ('bam_local_o_drop_head_rate', 'bam_local_o_read_weight_dropout_rate'):
        keys = dict(cfg.get_keys(), bam_local_o_drop_head_rate=0., bam_local_o_read_weight_dropout_rate=0.)
        parent = module.clone(config=pyconfig.HyperParameters(SimpleNamespace(keys=keys)))
        p = nn.unbox(parent.init(jax.random.key(3), *args, M_in=matrix))
        for a, b in zip(jax.tree.leaves(variables), jax.tree.leaves(p)):
          np.testing.assert_array_equal(a, b)
        keys[flag] = .2
        child = module.clone(config=pyconfig.HyperParameters(SimpleNamespace(keys=keys)))
        child_p = nn.unbox(child.init(jax.random.key(3), *args, M_in=matrix))
        for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(child_p)):
          np.testing.assert_array_equal(a, b)
        expected = parent.apply(p, *args, M_in=matrix, deterministic=True)
        actual = child.apply(p, *args, M_in=matrix, deterministic=True)
        for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
          np.testing.assert_array_equal(a, b)

  def test_effective_address_reference_all_gradients_and_unmodified_v(self):
    cfg, mesh, module, variables, args, matrix = self.fixture(WEIGHT)
    mask = (jax.random.bernoulli(jax.random.key(8), .8, (4, 4, 2, 32)).astype(jnp.float32) / .8)
    target = jax.random.normal(jax.random.key(9), (4, 4, 2, 75))
    def actual(m, matrix, x, mask):
      hidden = nn.gelu(m.full_m_read_down(x))
      v, o = m._independent_local_vo(matrix, x, shared_hidden=hidden, o_column_weight_multiplier=mask)
      o = o + m._static_column(matrix, 'o', x, column_weight_multiplier=mask)
      return v, o
    def reference(m, matrix, x, mask):
      hidden = nn.gelu(m.full_m_read_down(x))
      v, _ = m._independent_local_vo(matrix, x, shared_hidden=hidden)
      raw = jnp.squeeze(m._project_column_read_key('W_R', x, hidden), axis=-2)
      _, dynamic = _split_read_keys(raw, m._fetched_arm_ungated)
      logits = m._project_read_gate_logits('W_R_gate', x, squeeze_fetch_axis=True)
      dynamic = dynamic * m._read_key_scale * m._read_gate_activation(logits)
      static = m.static_o_key.T * (m._static_gate('general_o_static_gate', x)
                                  / cfg.bam_static_read_gate_init)[..., None]
      effective_address = (dynamic + static) * mask
      return v, jnp.einsum('btkv,btnv->btnk', matrix, effective_address)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      av, ao = module.apply(variables, matrix, args[0], mask, method=actual)
      bv, bo = module.apply(variables, matrix, args[0], mask, method=reference)
      np.testing.assert_array_equal(av, bv)
      np.testing.assert_allclose(ao, bo, rtol=3e-6, atol=3e-7)
      def loss(p, matrix, x, method):
        v, o = module.apply(p, matrix, x, mask, method=method)
        return jnp.sum((o + .3 * v) * target)
      ga = jax.grad(lambda p, m, x: loss(p, m, x, actual), argnums=(0, 1, 2))(variables, matrix, args[0])
      gb = jax.grad(lambda p, m, x: loss(p, m, x, reference), argnums=(0, 1, 2))(variables, matrix, args[0])
      for a, b in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
        np.testing.assert_allclose(a, b, rtol=3e-4, atol=2e-5)
      # Mask all O columns: static and dynamic O disappear; V remains nonzero.
      v, o = module.apply(variables, matrix, args[0], jnp.zeros_like(mask), method=actual)
      np.testing.assert_array_equal(o, 0)
      self.assertGreater(float(jnp.linalg.norm(v)), 0)

  def test_drop_head_vector_and_write_share_masked_o(self):
    cfg, mesh, module, variables, args, matrix = self.fixture()
    captured = []
    mask = jnp.tile(jnp.array([0., 1.25, 1.25, 0.]).reshape(1, 2, 2, 1), (4, 2, 1, 1))
    def capture_identity(m, o, deterministic):
      captured.append(o)
      return o
    def capture_drop(m, o, deterministic):
      captured.append(o)
      return o * mask
    def raw_write(m, o, x, shared_hidden=None):
      return o, jnp.zeros_like(o), jnp.zeros(o.shape[:-1])
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      with mock.patch.object(BamAttention, '_deferred_write_factors', raw_write):
        with mock.patch.object(BamAttention, '_regularize_local_o_heads', capture_identity):
          _, _, base = module.apply(variables, *args, M_in=matrix, defer_write=True, deterministic=True)
        with mock.patch.object(BamAttention, '_regularize_local_o_heads', capture_drop):
          y, _, dropped = module.apply(variables, *args, M_in=matrix, defer_write=True, deterministic=False)
      np.testing.assert_array_equal(captured[0], captured[1])
      # AV is unchanged; only O gets the mask, and write content sees the same sum.
      expected = base[0] + (mask - 1) * captured[0]
      np.testing.assert_allclose(dropped[0], expected, rtol=3e-5, atol=3e-7)
      y_expected = jnp.einsum('btnd,ndh->bth', expected, variables['params']['out']['kernel'])
      np.testing.assert_allclose(y, y_expected, rtol=3e-5, atol=3e-7)

  def test_training_rng_and_remat_gradients(self):
    for run in (HEAD, WEIGHT):
      cfg, mesh, module, variables, args, matrix = self.fixture(run)
      def loss(p, key):
        y, updated = module.apply(p, *args, M_in=matrix, deterministic=False, rngs={'dropout': key})
        return jnp.mean(y ** 2) + jnp.mean(updated ** 2)
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        key = jax.random.key(12)
        a, ga = jax.value_and_grad(loss)(variables, key)
        b, gb = jax.value_and_grad(jax.checkpoint(loss))(variables, key)
        np.testing.assert_allclose(a, b, rtol=2e-6, atol=2e-7)
        for x, y in zip(jax.tree.leaves(ga), jax.tree.leaves(gb)):
          np.testing.assert_allclose(x, y, rtol=3e-5, atol=2e-6)
        self.assertNotEqual(float(a), float(loss(variables, jax.random.key(13))))
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(ga)))

  def test_bf16_training_finite_and_preserves_dtype(self):
    for run in (HEAD, WEIGHT):
      cfg, mesh, module, variables, args, matrix = self.fixture(run, dtype='bfloat16')
      def forward(p):
        return module.apply(p, *args, M_in=matrix, deterministic=False,
                            rngs={'dropout': jax.random.key(15)})
      with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
        y, updated = forward(variables)
        self.assertEqual(y.dtype, jnp.bfloat16)
        self.assertEqual(updated.dtype, jnp.bfloat16)
        def loss(p):
          y, updated = forward(p)
          return jnp.mean(y.astype(jnp.float32)**2) + jnp.mean(updated.astype(jnp.float32)**2)
        grads = jax.grad(loss)(variables)
        self.assertTrue(all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads)))

  def test_drop_head_scan_pallas_logits_and_all_gradients(self):
    cfg = self.config(HEAD, base_emb_dim=150, base_num_query_heads=2, base_num_kv_heads=2,
                      base_num_decoder_layers=3, base_mlp_dim=32, vocab_size=32,
                      dtype='float32', weight_dtype='float32')
    cfg.get_keys().update(
        bam_pallas_core=False, bam_splash_attention=False, bam_memory_mlp_dim=16,
        bam_last_mlp_dim=32, bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0, 1], bam_mlp_write_layers=[0, 1], emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8, emb_bam_v_bottleneck_dim=8, bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8, max_target_length=128, max_prefill_predict_length=128,
        query_chunk_size=64)
    pcfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(
        cfg.get_keys(), bam_pallas_core=True, bam_pallas_interpret=True, bam_pallas_fused_inputs=True,
        bam_splash_attention=True)))
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    native, fused = Transformer(cfg, mesh, quant=None), Transformer(pcfg, mesh, quant=None)
    tok = jax.random.randint(jax.random.key(20), (1, 128), 0, 32)
    seg = jnp.where(jnp.arange(128) < 64, 1, 2)[None]
    pos = (jnp.arange(128) % 64)[None]
    original = splash.make_splash_mha
    def interpret(*args, **kwargs):
      return original(*args, **dict(kwargs, interpret=True))
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules), mock.patch.object(
        splash, 'make_splash_mha', side_effect=interpret):
      p, pp = (nn.unbox(m.init(jax.random.key(21), tok, pos, seg, tok)) for m in (native, fused))
      self.assertEqual(jax.tree.structure(p), jax.tree.structure(pp))
      for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(pp)):
        np.testing.assert_array_equal(a, b)
      def call(m, p):
        return m.apply(p, tok, pos, seg, tok, enable_dropout=True,
                       rngs={'dropout': jax.random.key(22)})[0]
      target = jax.random.normal(jax.random.key(23), call(native, p).shape)
      def loss(m, p):
        y = call(m, p)
        return jnp.mean(y * target), y
      (_, a), ga = jax.value_and_grad(lambda p: loss(native, p), has_aux=True)(p)
      (_, b), gb = jax.value_and_grad(lambda p: loss(fused, p), has_aux=True)(p)
    np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-6)
    errors = [float(jnp.max(jnp.abs(a-b)) / (jnp.max(jnp.abs(a)) + 1e-12))
              for a, b in zip(jax.tree.leaves(ga), jax.tree.leaves(gb))]
    self.assertLess(max(errors), 1e-3)
    print('O_DROP_HEAD_SCAN_NATIVE_PALLAS_ALL_GRADIENTS_OK', max(errors), flush=True)


if __name__ == '__main__':
  unittest.main()
