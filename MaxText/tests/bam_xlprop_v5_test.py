"""XLProp V5 budgets and native/v8t parity at K96/V40/NoPE96+RoPE24."""
import math
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash

import max_utils
import pyconfig
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import bam_pallas_v8 as v8
from layers.models import Transformer


class XLPropV5Test(MLPWriteTest, unittest.TestCase):
  def test_full_parameter_budget_and_final_write_omission(self):
    counts = {}
    for run in ('Llama2XLPropTruePileMHA', 'BamXLPropV5'):
      cfg = self.config(run)
      cfg.get_keys().update(max_target_length=128, max_prefill_predict_length=128,
                            query_chunk_size=64)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
      shaped, *_ = train_compile.get_shaped_inputs(mesh, cfg)
      flat = flatten_dict(nn.unbox(shaped[0].params))
      counts[run] = sum(math.prod(v.shape) for v in flat.values())
      if run != 'BamXLPropV5':
        continue
      self.assertEqual((cfg.emb_dim, cfg.num_decoder_layers, cfg.num_query_heads,
                        cfg.head_dim, cfg.bam_k, cfg.bam_v), (1920, 28, 20, 96, 96, 40))
      self.assertEqual((cfg.mlp_dim, cfg.bam_memory_mlp_dim, cfg.bam_last_mlp_dim),
                       (4048, 1827, 6277))
      self.assertEqual((cfg.steps, cfg.learning_rate_schedule_steps, cfg.learning_rate),
                       (50000, 50000, 2e-4))
      for key in ('bam_local_full_m_read_bottleneck_dim', 'bam_write_v_bottleneck_dim',
                  'emb_bam_v_bottleneck_dim', 'bam_mlp_write_address_rank'):
        self.assertEqual(getattr(cfg, key), 400)
      self.assertEqual(cfg.bam_attention_write_layers, list(range(27)))
      self.assertEqual(cfg.bam_mlp_write_layers, list(range(27)))
      self.assertTrue(cfg.scan_layers and not cfg.bam_pair_scan)
      self.assertTrue(cfg.bam_pallas_core and cfg.bam_pallas_write
                      and cfg.bam_pallas_packed_read_keys and cfg.bam_pallas_fused_inputs)
      self.assertEqual((cfg.bam_pallas_body, cfg.bam_pallas_vmem_mib), ('v8t', 63))
      self.assertEqual(cfg.bam_m_read_norm, 'rms')
      self.assertTrue(cfg.bam_common_qk_read_scale and cfg.bam_common_vo_read_scale)
      self.assertEqual(cfg.bam_common_vo_read_scale_init, 1.)
      self.assertFalse(cfg.qk_norm)
      self.assertTrue(cfg.sa_use_fused_bwd_kernel and cfg.record_training_health_metrics)
      self.assertFalse(cfg.record_internal_nn_metrics or cfg.bam_record_m_shift_health)
      self.assertAlmostEqual(cfg.bam_embedding_write_content_divisor,
                             .0016052833 * math.sqrt(1920 / 1200))
      for name in ('common_qk_read_scale', 'common_vo_read_scale'):
        self.assertEqual(sum(math.prod(v.shape) for p, v in flat.items() if p[-1] == name), 28)
      final = {p: v for p, v in flat.items() if 'final_local_layer' in p}
      self.assertFalse(any('memory_mlp' in p or any(x in p for x in
          ('mlp_address_down', 'mlp_address_up', 'mlp_write_gate', 'gw_b0')) for p in final))
      ordinary = sum(math.prod(v.shape) for p, v in flat.items() if 'mlp' in p)
      memory = sum(math.prod(v.shape) for p, v in flat.items() if 'memory_mlp' in p)
      controls = sum(math.prod(v.shape) for p, v in flat.items() if set(p) & {
          'mlp_address_down', 'mlp_address_up', 'mlp_address_bias',
          'mlp_write_gate', 'mlp_write_gate_bias'})
      body_ordinary = (ordinary - cfg.bam_last_mlp_dim * 3 * cfg.emb_dim) // 27
      self.assertAlmostEqual(body_ordinary / ((memory + controls) / 27), 2.00128747186874)
    self.assertEqual(counts, {'Llama2XLPropTruePileMHA': 1432398720,
                              'BamXLPropV5': 1432396996})

  def test_xl_read_kernel_against_independent_reference_all_gradients(self):
    b, t, n, v, k, rs = 1, 128, 20, 40, 96, 24
    keys = iter(jax.random.split(jax.random.key(70), 24))
    rand = lambda shape: jax.random.normal(next(keys), shape)
    matrix = rand((b, t, v, k))
    static = rand((4 * n, v)) * .3
    xs = (rand((b, t, n, v)), rand((b, t, n)), rand((b, t, n, v)),
          rand((b, t, n)), rand((b, t, 2 * n, v)), rand((b, t, n)),
          rand((b, t, n)), rand((b, t, n, rs)), rand((b, t, n, rs)))
    gates = tuple(jax.nn.sigmoid(rand((b, t, n))) for _ in range(4))
    shift = jax.nn.sigmoid(rand((b, t, v))).at[:, 0].set(0.).at[:, 64].set(0.)
    args = (matrix, static, xs, gates, shift, jnp.asarray(-.7), jnp.asarray(.8))
    eps, read_eps, scale = 1e-6, 1e-4, .1

    def kernel(m, sw, x, sg, sh, qk, vo):
      rows = jnp.repeat(jnp.stack((qk, qk, vo, vo)), n)[:, None]
      head = lambda s: jnp.full((n,), s)
      return v8.read_tm(m, sw * rows, *x, qk_cols=k, read_epsilon=read_eps,
          key_scale=scale, forward_tile=64, reverse_tile=64, interpret=True,
          q_scale=1 / math.sqrt(k), static_gates=sg, shift=sh, chunk=8, m_norm=eps,
          q_dynamic_scale=head(qk), k_dynamic_scale=head(qk),
          v_dynamic_scale=head(vo), o_dynamic_scale=head(vo))

    def reference(m, sw, x, sg, sh, qk, vo):
      previous = jnp.concatenate((m[:, :1], m[:, :-1]), axis=1)
      view = m + sh[..., None] * (previous - m)
      view *= jax.lax.rsqrt(jnp.mean(view * view, axis=(-2, -1), keepdims=True) + eps)
      rq, lq, rk, lk, rvo, lo, lv, sq, sk = x
      stat = sw.reshape(4, n, v)
      norm = lambda a: a * jax.lax.rsqrt(jnp.mean(a * a, axis=-1, keepdims=True) + read_eps)
      arm = lambda i, raw, logit, s: s * (
          scale * jax.nn.sigmoid(logit)[..., None] * norm(raw) + stat[i] * sg[i][..., None])
      reads = [jnp.einsum('btvk,btnv->btnk', view, a) for a in (
          arm(0, rq, lq, qk), arm(1, rk, lk, qk),
          arm(2, rvo[:, :, :n], lv, vo), arm(3, rvo[:, :, n:], lo, vo))]
      return (jnp.concatenate((reads[0], sq), -1) / math.sqrt(k),
              jnp.concatenate((reads[1], sk), -1), reads[2], reads[3], m)

    actual, expected = jax.jit(kernel)(*args), jax.jit(reference)(*args)
    for a, e in zip(actual, expected):
      np.testing.assert_allclose(a, e, rtol=3e-4, atol=2e-6)
    cotangents = tuple(jax.random.normal(jax.random.key(90 + i), a.shape)
                       for i, a in enumerate(expected))
    def loss(fn):
      return lambda *a: sum(jnp.sum(o * cot) for o, cot in zip(fn(*a), cotangents))
    ga, ge = (jax.jit(jax.grad(loss(fn), argnums=tuple(range(7))))(*args)
              for fn in (kernel, reference))
    for a, e in zip(jax.tree.leaves(ga), jax.tree.leaves(ge)):
      self.assertTrue(np.isfinite(np.asarray(a)).all())
      error = np.max(np.abs(np.asarray(a) - np.asarray(e))) / (np.max(np.abs(e)) + 1e-12)
      self.assertLess(error, 3e-4)
    self.assertGreater(abs(float(ga[-1])), 0.)
    self.assertGreater(abs(float(ga[-2])), 0.)

  def test_native_pallas_model_loss_and_all_gradients(self):
    cfg = self.config('BamXLPropV5', dtype='float32', weight_dtype='float32')
    cfg.get_keys().update(emb_dim=192, num_query_heads=2, num_kv_heads=2,
        num_decoder_layers=2, mlp_dim=24, vocab_size=32, emb_bam_num_head=2,
        bam_layer_modes=['local_qk+local_v+local_o'] * 2,
        bam_attention_write_layers=[0], bam_mlp_write_layers=[0],
        bam_memory_mlp_dim=12, bam_last_mlp_dim=24,
        bam_write_v_bottleneck_dim=8, emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8, bam_local_full_m_read_bottleneck_dim=8,
        bam_pallas_core=False, bam_pallas_write=False, max_target_length=128,
        max_prefill_predict_length=128, query_chunk_size=64)
    pcfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(),
        bam_pallas_core=True, bam_pallas_write=True, bam_pallas_interpret=True)))
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    native, fused = (Transformer(c, mesh, quant=None) for c in (cfg, pcfg))
    tokens = jax.random.randint(jax.random.key(31), (1, 128), 0, 32)
    pos = (jnp.arange(128) % 64)[None]
    seg = jnp.where(jnp.arange(128) < 64, 1, 2)[None]
    target = (tokens + 1) % cfg.vocab_size
    mask = jnp.ones_like(tokens)
    make = splash.make_splash_mha
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules), mock.patch.object(
        splash, 'make_splash_mha', side_effect=lambda *a, **kw: make(*a, **dict(kw, interpret=True))):
      params = nn.unbox(native.init(jax.random.key(32), tokens, pos, target, mask,
                                   seg, enable_dropout=False))
      def perturb(path, value):
        if path[-1].key == 'common_qk_read_scale': return value * -.65
        if path[-1].key == 'common_vo_read_scale': return value * .8
        return value
      params = jax.tree.map_with_path(perturb, params)
      def forward(model, p):
        return model.apply(p, tokens, pos, target, mask, seg, enable_dropout=False)[0]
      ya, ye = (jax.jit(lambda p: forward(m, p))(params) for m in (fused, native))
      np.testing.assert_allclose(ya, ye, rtol=3e-4, atol=2e-6)
      def loss(model):
        return lambda p: jnp.mean(forward(model, p).astype(jnp.float32))
      ga, ge = (jax.jit(jax.grad(loss(m)))(params) for m in (fused, native))
      fa, fe = flatten_dict(ga), flatten_dict(ge)
      self.assertEqual(fa.keys(), fe.keys())
      relative_errors = []
      for path in fa:
        self.assertTrue(np.isfinite(np.asarray(fa[path])).all(), path)
        a, e = np.asarray(fa[path]), np.asarray(fe[path])
        error = np.max(np.abs(a - e)) / (np.max(np.abs(e)) + 1e-12)
        relative_errors.append(float(error))
        self.assertLess(error, 5e-4, str(path))
      for name in ('common_vo_read_scale', 'common_qk_read_scale'):
        self.assertGreater(sum(float(jnp.sum(v * v)) for p, v in fa.items() if p[-1] == name), 0.)
      print('XLPROP_V5_MODEL_MAX_GRADIENT_RELATIVE_ERROR', max(relative_errors), flush=True)


if __name__ == '__main__':
  unittest.main()
