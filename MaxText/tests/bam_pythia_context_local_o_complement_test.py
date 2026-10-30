"""Pythia H competition: fixed parameter budget, scan health and both gate gradients."""
import contextlib
import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

import max_utils
import train
import train_compile
from bam_config import validate_bam_config
from bam_prop_test_utils import MLPWriteTest
from bam_pythia_common_read_raw_write_test import H, PythiaCommonReadRawWriteTest
from layers.models import Transformer

RUN = H + 'ContextLocalOComplement'


class PythiaContextLocalOComplementTest(MLPWriteTest, unittest.TestCase):
  tiny = PythiaCommonReadRawWriteTest.tiny

  def test_formal_parameter_tree_and_configuration_scope(self):
    parent, cfg = self.config(H), self.config(RUN)
    changed = {k for k, v in cfg.get_keys().items() if v != parent.get_keys()[k]}
    self.assertEqual(changed, {'exp_class', 'model_name', 'compare_runs', 'bam_context_local_o_complement'})
    self.assertEqual(cfg.compare_runs, [H])
    self.assertTrue(cfg.bam_context_local_o_complement and cfg.bam_attention_negative_diagonal)
    self.assertFalse(cfg.bam_local_o_dynamic_norm_scale)
    self.assertTrue(cfg.qk_norm and cfg.bam_qk_norm_after_concat)
    self.assertEqual((cfg.mlp_dim, cfg.bam_memory_mlp_dim, cfg.bam_last_mlp_dim), (7808, 2944, 9984))
    self.assertEqual((cfg.num_decoder_layers, cfg.num_query_heads, cfg.bam_k, cfg.bam_v), (32, 32, 80, 64))
    self.assertEqual((cfg.bam_pallas_body, cfg.bam_pallas_vmem_mib, cfg.remat_policy), ('v8t', 60, 'minimal'))
    self.assertTrue(cfg.bam_pallas_core and cfg.bam_pallas_write and cfg.bam_pallas_fused_inputs)
    self.assertTrue(cfg.scan_layers and not cfg.bam_pair_scan and cfg.bam_skip_final_mlp_write)
    self.assertEqual(cfg.bam_attention_write_layers, list(range(31)))
    self.assertTrue(cfg.record_training_health_metrics and not cfg.record_internal_nn_metrics)
    trees = []
    for c in [parent, cfg]:
      c.get_keys().update(max_target_length=128, max_prefill_predict_length=128)
      mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
      args, *_ = train_compile.get_shaped_inputs(mesh, c)
      flat = flatten_dict(nn.unbox(args[0].params))
      trees.append({path: value.shape for path, value in flat.items()})
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), 2775190912)
    self.assertEqual(trees[0], trees[1])
    print('PYTHIA_COMPLEMENT_BUDGET_OK', 2775190912, flush=True)

  def test_native_reference_gate_affects_residual_and_subsequent_memory_reads(self):
    cfg, pcfg = self.tiny(RUN, length=4), self.tiny(H, length=4)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model, parent = Transformer(cfg, mesh, quant=None), Transformer(pcfg, mesh, quant=None)
    tok = jnp.array([[1, 2, 3, 4]], jnp.int32)
    pos, seg = jnp.arange(4)[None], jnp.ones_like(tok)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params, original = (nn.unbox(m.init(jax.random.key(63), tok, pos, seg, tok)) for m in (model, parent))
      for path, value in flatten_dict(params).items():
        np.testing.assert_array_equal(value, flatten_dict(original)[path])
      flat = flatten_dict(params)
      for path, value in flat.items():
        if path[-1] == 'static_o_key':
          flat[path] = jax.random.normal(jax.random.key(66), value.shape)*.1
        elif path[-1] == 'W_R_gate_b0':
          flat[path] = jnp.linspace(-3., 3., value.size).reshape(value.shape)
      params = unflatten_dict(flat)
      target = jax.random.normal(jax.random.key(64), model.apply(params, tok, pos, seg, tok)[0].shape)
      def loss(p, reference):
        captured = {}
        def intercept(next_fun, args, kwargs, ctx):
          if ctx.method_name == '_project_read_gate_logits':
            result = next_fun(*args, **kwargs)
            if args[0] == 'W_R_gate':
              captured['gate'] = result
            return result
          if ctx.method_name == '_context_with_local_o_complement':
            # Independent intervention on H; retain its ordinary static O,
            # residual projection, attention write and downstream M consumers.
            gate = jax.nn.sigmoid(captured['gate'].reshape(args[0].shape[:-1]))
            return args[0]*(1.-gate)[..., None]
          return next_fun(*args, **kwargs)
        with nn.intercept_methods(intercept) if reference else contextlib.nullcontext():
          logits = (parent if reference else model).apply(p, tok, pos, seg, tok)[0]
        return jnp.mean(logits*target), logits
      (_, a), ga = jax.value_and_grad(lambda p: loss(p, False), has_aux=True)(params)
      (_, b), gb = jax.value_and_grad(lambda p: loss(p, True), has_aux=True)(params)
    np.testing.assert_allclose(a, b, rtol=2e-5, atol=3e-6)
    for path, value in flatten_dict(ga).items():
      np.testing.assert_allclose(value, flatten_dict(gb)[path], rtol=1e-3, atol=3e-6)
      if path[-2:] == ('W_R_gate', 'kernel'):
        self.assertGreater(float(jnp.max(jnp.abs(value))), 0.)

  def test_scan_health_includes_separate_tail_and_all32_layers(self):
    c = self.tiny(RUN, length=4)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(c), c.mesh_axes)
    model = Transformer(c, mesh, quant=None)
    tok = jnp.array([[1, 2, 3, 4]], jnp.int32)
    with mesh, nn.partitioning.axis_rules(c.logical_axis_rules):
      p = model.init(jax.random.key(1), tok, jnp.arange(4)[None], jnp.ones_like(tok), tok)
      _, inter = model.apply(p, tok, jnp.arange(4)[None], jnp.ones_like(tok), tok,
                             enable_dropout=False, mutable=['intermediates'])
    metrics = {'scalar': {}}
    train.record_bam_concat_health_metrics(metrics, inter, c)
    self.assertEqual(len(metrics['scalar']), 6)
    self.assertTrue(all(math.isfinite(float(v)) for v in metrics['scalar'].values()))
    c = self.config(RUN)
    branch = lambda value: {'block': {'self_attention': {'context_local_o_competition': (value,)}}}
    inter = {'intermediates': {'decoder': {'layers': branch(jnp.ones((31, 2))*.25),
                                         'final_local_layer': branch(jnp.ones(2)*.75)}}}
    metrics = {'scalar': {}}
    train.record_bam_concat_health_metrics(metrics, inter, c)
    self.assertEqual(len(metrics['scalar']), 64)
    for layer in range(32):
      self.assertEqual(float(metrics['scalar'][f'bam/context_local_o_competition/layer_{layer:03d}/gate_mean']),
                       .25 if layer < 31 else .75)

  def test_model_f32_optimized_pallas_logits_and_all_gradients(self):
    PythiaCommonReadRawWriteTest.check_model(self, RUN)

  def test_model_bf16_optimized_pallas_all_gradients_against_f32(self):
    PythiaCommonReadRawWriteTest.check_model(self, RUN, 'bfloat16')

  def test_reject_incompatible_gate_modes(self):
    c = dict(self.config(RUN).get_keys())
    for key, value in [('bam_read_gate_activation', 'softplus'),
                       ('bam_local_vo_independent_gates', False), ('bam_prune_all_row_reads', False)]:
      with self.assertRaisesRegex(ValueError, 'Context/localO complement'):
        validate_bam_config(dict(c, **{key: value}))


if __name__ == '__main__':
  unittest.main()
