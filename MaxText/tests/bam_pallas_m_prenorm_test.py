"""Whole-M read pre-norm inside the v8t read: kernel and scanned-model checks against the native view.

Kernel reference: shift and RMS-normalize M in XLA (one FP32 statistic per token over V x K), then the
plain v8t read (whose equivalence is covered by bam_pallas_mshift_test / bam_pallas_model_test).
Model reference: bam_pallas_core=False (BamAttention._matrix_for_read).
"""
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash

sys.path.insert(0, 'MaxText')
sys.path.insert(0, 'MaxText/tests')
import max_utils  # pylint: disable=wrong-import-position
import pyconfig  # pylint: disable=wrong-import-position
from bam_prop_test_utils import MLPWriteTest  # pylint: disable=wrong-import-position
from layers import bam_pallas_v8 as v8  # pylint: disable=wrong-import-position
from layers.models import Transformer  # pylint: disable=wrong-import-position
import bam_pallas_mshift_test as shift_test  # pylint: disable=wrong-import-position

B, T, V, K, N = shift_test.B, shift_test.T, shift_test.V, shift_test.K, shift_test.N
M_EPS = 1e-6
PARENT = ('BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeys'
          'LogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixedReadScaleInitOneTruePile')
KEEP = PARENT.removesuffix('TruePile') + 'MPreNormTruePile'
DROP = PARENT.removesuffix('TruePile') + 'MPreNormNoReadScaleTruePile'


def normalized(m):
  x = m.astype(jnp.float32)
  return (x * jax.lax.rsqrt(jnp.mean(x * x, axis=(-2, -1), keepdims=True) + M_EPS)).astype(m.dtype)


def read(m, sw, xs, sgs, shift, tile, chunk=8, m_norm=None):
  return v8.read_tm(m, sw, *xs, qk_cols=shift_test.QKC, read_epsilon=shift_test.EPS, key_scale=shift_test.SCALE,
                    forward_tile=tile, reverse_tile=tile, interpret=True, q_scale=0.125, static_gates=sgs,
                    shift=shift, chunk=chunk, m_norm=m_norm)


class KernelPreNormTest(unittest.TestCase):

  def assert_close(self, a, b, tol, name):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    err = np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12)
    self.assertLess(err, tol, f'{name}: relative max error {err:.3e}')

  def test_forward_and_gradients_with_and_without_shift(self):
    m, sw, xs, sgs, shift = shift_test.inputs(5)
    # Large per-token magnitude spread, zero matrices (alone, and as the previous token of a mix),
    # and zero tokens on both sides of a tile boundary.
    m = m * jnp.exp(2 * jax.random.normal(jax.random.PRNGKey(50), (B, T, 1, 1)))
    m = m.at[0, 5].set(0.).at[0, 63].set(0.).at[0, 64].set(0.).at[1, 200].set(0.).at[1, 201].set(0.)
    ks = iter(jax.random.split(jax.random.PRNGKey(51), 8))
    cts = tuple(jax.random.normal(next(ks), (B, T, N, K)) for _ in range(4))
    ct_m = jax.random.normal(next(ks), m.shape)
    xs_ind = xs[:4] + (jax.random.normal(next(ks), (B, T, 2 * N, V)),) + xs[5:]

    def loss(fn):
      def f(m, sw, xs, sgs, shift):
        outs = fn(m, sw, xs, sgs, shift)
        extra = jnp.sum(outs[4] * ct_m) if len(outs) > 4 else jnp.sum(m * ct_m)
        return sum(jnp.sum(o * c) for o, c in zip(outs[:4], cts)) + extra
      return f

    for name, args in (('shared VO', xs), ('independent VO', xs_ind)):
      for shifted in (True, False):
        for tile, chunk in ((64, 8), (256, 8), (64, 4)):
          tag = f'{name} shift={shifted} tile {tile} chunk {chunk}'
          sh = shift if shifted else None
          kernel = lambda m, sw, xs, sgs, sh, tile=tile, chunk=chunk: read(m, sw, xs, sgs, sh, tile, chunk, M_EPS)
          def ref(m, sw, xs, sgs, sh, tile=tile):
            view = normalized(shift_test.mixed(m, sh) if sh is not None else m)
            return read(view, sw, xs, sgs, None, tile)[:4] + (m,)
          outs = kernel(m, sw, args, sgs, sh)
          for i, (a, b) in enumerate(zip(outs, ref(m, sw, args, sgs, sh))):
            self.assert_close(a, b, 1e-5, f'{tag} output {i}')
          np.testing.assert_array_equal(outs[4], m)       # raw passthrough
          plain = read(m, sw, args, sgs, sh, tile)[2]
          self.assertGreater(float(jnp.max(jnp.abs(plain - outs[2])) / jnp.max(jnp.abs(plain))), .05)
          ga = jax.grad(loss(kernel), argnums=(0, 1, 2, 3) + ((4,) if shifted else ()))(m, sw, args, sgs, sh)
          gb = jax.grad(loss(ref), argnums=(0, 1, 2, 3) + ((4,) if shifted else ()))(m, sw, args, sgs, sh)
          for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(ga)[0], jax.tree.leaves(gb)):
            self.assertTrue(np.isfinite(np.asarray(a)).all(), f'{tag} {jax.tree_util.keystr(path)}')
            self.assert_close(a, b, 2e-5, f'{tag} grad {jax.tree_util.keystr(path)}')
    print('V8T_M_PRENORM_KERNEL_ALL_GRADIENTS_OK', flush=True)

  def test_bfloat16_matches_float32_truth(self):
    m, sw, xs, sgs, shift = shift_test.inputs(6)
    m = m * jnp.exp(2 * jax.random.normal(jax.random.PRNGKey(60), (B, T, 1, 1)))
    cts = tuple(jax.random.normal(jax.random.PRNGKey(61 + i), (B, T, N, K)) for i in range(4))
    bf = lambda tree: jax.tree.map(lambda x: x.astype(jnp.bfloat16), tree)

    def loss(m, sw, xs, sgs, shift):
      outs = read(m, sw, xs, sgs, shift, 64, 8, M_EPS)
      return sum(jnp.sum(o.astype(jnp.float32) * c) for o, c in zip(outs[:4], cts))

    ga = jax.grad(loss, argnums=(0, 2, 4))(m, sw, xs, sgs, shift)
    gb = jax.grad(loss, argnums=(0, 2, 4))(*bf((m, sw, xs, sgs, shift)))
    for (path, a), b in zip(jax.tree_util.tree_flatten_with_path(ga)[0], jax.tree.leaves(gb)):
      self.assert_close(b, a, 3e-2, f'bf16 grad {jax.tree_util.keystr(path)}')
    print('V8T_M_PRENORM_BF16_OK', flush=True)


class ModelPreNormTest(MLPWriteTest, unittest.TestCase):
  """Complete scanned model, native _matrix_for_read vs the v8t core: logits and every gradient."""

  def configs(self, run, dtype='float32'):
    cfg = self.config(run, base_emb_dim=150, base_num_query_heads=2, base_num_kv_heads=2,
                      base_num_decoder_layers=3, base_mlp_dim=32, vocab_size=32,
                      dtype=dtype, weight_dtype='float32')
    cfg.get_keys().update(
        bam_pallas_core=False, bam_pallas_write=False, bam_splash_attention=False, bam_memory_mlp_dim=16,
        bam_last_mlp_dim=32, bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0, 1], bam_mlp_write_layers=[0, 1], emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8, emb_bam_v_bottleneck_dim=8, bam_mlp_write_address_rank=8,
        bam_local_full_m_read_bottleneck_dim=8, max_target_length=128, max_prefill_predict_length=128,
        query_chunk_size=64)
    pcfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(
        cfg.get_keys(), bam_pallas_core=True, bam_pallas_write=True, bam_pallas_interpret=True,
        bam_pallas_fused_inputs=True, bam_pallas_packed_read_keys=True, bam_splash_attention=True)))
    return cfg, pcfg

  def grads(self, run, dtype, pallas, perturbed=True):
    """Output and all gradients of one model variant at parameters shared by every variant."""
    cfg, pcfg = self.configs(run, dtype)
    cfg = pcfg if pallas else cfg
    self.assertEqual(cfg.bam_m_read_norm, 'rms')
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = Transformer(cfg, mesh, quant=None)
    tok = jax.random.randint(jax.random.key(20), (1, 128), 0, 32)
    seg = jnp.where(jnp.arange(128) < 64, 1, 2)[None]
    pos = (jnp.arange(128) % 64)[None]
    original = splash.make_splash_mha
    def interpret(*args, **kwargs):
      return original(*args, **dict(kwargs, interpret=True))
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules), mock.patch.object(
        splash, 'make_splash_mha', side_effect=interpret):
      flat = flatten_dict(nn.unbox(model.init(jax.random.key(21), tok, pos, seg, tok)))
      scales = [k for k in flat if k[-1] == 'common_vo_read_scale']
      self.assertEqual(bool(scales), run == KEEP)
      for i, k in enumerate(scales if perturbed else ()):    # away from the unit initialization, signed
        flat[k] = flat[k] * (1 + .5 * jax.random.normal(jax.random.key(30 + i), flat[k].shape))
      p = unflatten_dict(flat)
      shape = jax.eval_shape(lambda p: model.apply(p, tok, pos, seg, tok)[0], p).shape
      target = jax.random.normal(jax.random.key(23), shape)
      def loss(p):
        y = model.apply(p, tok, pos, seg, tok)[0].astype(jnp.float32)
        return jnp.mean(y * target), y
      (_, y), g = jax.value_and_grad(loss, has_aux=True)(p)
    return y, flatten_dict(g), scales

  @staticmethod
  def errors(fa, fb):
    return {k: float(jnp.max(jnp.abs(fa[k] - fb[k])) / (jnp.max(jnp.abs(fa[k])) + 1e-12)) for k in fa}

  def check_native_vs_pallas(self, run):
    a, fa, scales = self.grads(run, 'float32', False)
    b, fb, _ = self.grads(run, 'float32', True)
    np.testing.assert_allclose(a, b, rtol=2e-4, atol=2e-6)
    self.assertEqual(fa.keys(), fb.keys())
    for k in scales:
      self.assertGreater(float(jnp.max(jnp.abs(fa[k]))), 0)
    self.assertTrue([k for k in fa if 'm_shift' in '/'.join(map(str, k))])
    errors = self.errors(fa, fb)
    worst = max(errors, key=errors.get)
    self.assertLess(errors[worst], 1e-3, worst)
    print('M_PRENORM_SCAN_NATIVE_PALLAS_ALL_GRADIENTS_OK', run.removeprefix(PARENT.removesuffix('TruePile')),
          errors[worst], worst, flush=True)

  def test_keep_scan_native_vs_pallas_all_gradients(self):
    self.check_native_vs_pallas(KEEP)

  def test_drop_scan_native_vs_pallas_all_gradients(self):
    self.check_native_vs_pallas(DROP)

  def test_bf16_pallas_vs_float32_truth_against_native_bf16(self):
    """BF16 v8t vs FP32 truth, judged against the native BF16 model's own error."""
    truth, ft, _ = self.grads(KEEP, 'float32', False)
    out = {}
    for pallas in (False, True):
      y, fg, _ = self.grads(KEEP, 'bfloat16', pallas)
      for k in fg:
        self.assertTrue(np.isfinite(np.asarray(fg[k])).all(), k)
      errors = self.errors(ft, fg)
      out[pallas] = (float(jnp.max(jnp.abs(y - truth)) / jnp.max(jnp.abs(truth))),
                     float(np.median(list(errors.values()))), max(errors.values()))
    print('M_PRENORM_BF16 native(out,median,max)=%s pallas=%s' % (out[False], out[True]), flush=True)
    self.assertLess(out[True][0], 2 * out[False][0] + 1e-3)
    self.assertLess(out[True][1], 2 * out[False][1] + 1e-3)


if __name__ == '__main__':
  unittest.main()
