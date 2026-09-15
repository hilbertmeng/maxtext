"""L1 anchor: global indices, scan carry, mapped unroll, gradients and TB export."""
from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import max_utils
import bam_local_fetch_test
from layers.attentions import _mix_local_v_row_anchor
from layers.fusion import BamLayerPair
from layers.bam_row_anchor_init import map_row_anchor_params
import train
import train_compile
from flax.traverse_util import flatten_dict
from types import SimpleNamespace
from layers.models import Transformer
import pyconfig

EXP = 'BamMediumIndependentLLFLocalVRank4BLocalORowDecodeL1Anchor'


class RowAnchorTest(absltest.TestCase):
  config = bam_local_fetch_test.LocalFetchTest.config

  def test_real_model_mapped_initialization_and_initial_logits(self):
    cfg = self.config(EXP.replace('L1Anchor', 'L1DirectAnchor'))
    cfg.get_keys().update({
        'num_decoder_layers': 6, 'base_num_decoder_layers': 6, 'vocab_size': 128,
        'bam_layer_modes': ['local_qk+local_o', 'local_qk+local_o', 'local_qk+full'] * 2,
        'bam_local_o_v_mode': ['rank2', 'rank2', 'none'] * 2,
        'dtype': jnp.float32,
    })
    parent_cfg = pyconfig.HyperParameters(SimpleNamespace(keys=dict(
        cfg.get_keys(), bam_l1_direct_local_v_row=False)))
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model, parent = Transformer(cfg, mesh, quant=None), Transformer(parent_cfg, mesh, quant=None)
    key = jax.random.key(17)
    with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
      state = jax.jit(lambda: max_utils.init_initial_state(model, None, cfg, False, key))()
      old = jax.jit(lambda: max_utils.init_initial_state(parent, None, parent_cfg, False, key))()
      expected = map_row_anchor_params(state.params['params'], old.params['params'], cfg.param_scan_axis)
      for got, want in zip(jax.tree.leaves(state.params['params']), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(got, want)
      tokens = jnp.arange(8)[None]
      inputs = (tokens, tokens, jnp.ones_like(tokens), tokens)
      y = jax.jit(lambda: model.apply(state.params, *inputs))()
      y0 = jax.jit(lambda: parent.apply(old.params, *inputs))()
      for got, want in zip(jax.tree.leaves(y), jax.tree.leaves(y0)):
        np.testing.assert_allclose(got, want, atol=2e-5, rtol=2e-5)

  def test_complete_training_signature_scan_and_non_scan(self):
    for exp, scanned in ((EXP, True), (EXP, False),
                         (EXP.replace('L1Anchor', 'L1DirectAnchor'), True)):
      with self.subTest(exp=exp, scan=scanned):
        cfg = self.config(exp)
        cfg.get_keys().update({
            'scan_layers': scanned, 'vocab_size': 128,
            'num_decoder_layers': 6, 'base_num_decoder_layers': 6,
            'bam_layer_modes': ['local_qk+local_o', 'local_qk+local_o', 'local_qk+full'] * 2,
            'bam_local_o_v_mode': ['rank2', 'rank2', 'none'] * 2,
        })
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
        args, _, shardings, model = train_compile.get_shaped_inputs(mesh, cfg)
        with mesh, nn_partitioning.axis_rules(cfg.logical_axis_rules):
          _, metrics = jax.eval_shape(
              lambda state, data, rng: train.train_step(model, cfg, shardings, state, data, rng), *args)
        self.assertIn('learning/raw_grad_norm', metrics['scalar'])
        self.assertEqual(sum(k.startswith('bam/row_anchor/') for k in metrics['scalar']), 48)
        self.assertEqual(cfg.checkpoint_period, 200)
        self.assertEqual(cfg.steps, 13500)

  def test_first_block_direct_row_and_mapped_common_parameters(self):
    cfg = self.config(EXP.replace('L1Anchor', 'L1DirectAnchor'))
    cfg.get_keys().update({
        'num_decoder_layers': 6, 'base_num_decoder_layers': 6,
        'bam_layer_modes': ['local_qk+local_o', 'local_qk+local_o', 'local_qk+full'] * 2,
        'bam_local_o_v_mode': ['rank2', 'rank2', 'none'] * 2,
        'dtype': jnp.float32,
    })
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    block = BamLayerPair(cfg, mesh, 8, first_block=True)
    native = BamLayerPair(cfg, mesh, 8)
    h = jax.random.normal(jax.random.key(1), (1, 8, 128))
    carry = (h, jnp.zeros((1, 8, 32, 32)), jnp.zeros((1, 8, 2, 32)))
    args = (jnp.ones((1, 8), jnp.int32), jnp.arange(8)[None],
            jnp.ones((1, 8), jnp.int32), None, True, 'train', None, None, None, None, jnp.array(0))
    keys = {'params': jax.random.key(2), 'aqt': jax.random.key(3)}
    variables = block.init(keys, carry, *args)
    old = native.init(keys, carry, *args)
    params = nn.unbox(variables['params'])
    p = params['local_1']['block']['self_attention']
    self.assertEqual(p['W_lv_direct_row']['kernel'].shape, (128, 2, 32))
    self.assertNotIn('W_lv_direct_row', params['local_0']['block']['self_attention'])
    self.assertNotIn('W_lv_direct_row', params['fetch_2']['block']['self_attention'])
    p['lv_direct_row_bias'] = jax.random.normal(jax.random.key(4), (2, 32)) * .02
    (result, _), metrics = block.apply({'params': params}, carry, *args, mutable='intermediates')
    self.assertGreater(float(jnp.linalg.norm(result[2])), 0.)
    later = BamLayerPair(cfg, mesh, 8)
    (after, _), _ = later.apply(old, result, *args[:-1], jnp.array(1), mutable='intermediates')
    np.testing.assert_array_equal(after[2], result[2])
    grad = jax.jit(jax.grad(lambda pp: block.apply({'params': pp}, carry, *args)[0][0].sum()))(params)
    self.assertGreater(float(jnp.linalg.norm(grad['local_1']['block']['self_attention']['W_lv_direct_row']['kernel'])), 0.)

    # Structural mapping uses identical per-layer arrays even though scan axes change.
    source = {'decoder': {'layers': jax.tree.map(lambda x: jnp.stack([x, x + 1], axis=1), nn.unbox(old['params']))}}
    target = {'decoder': {'first_block': params,
                          'layers': jax.tree.map(lambda x: x[:, 1:], source['decoder']['layers'])}}
    mapped = map_row_anchor_params(target, source, 1)
    for path, value in flatten_dict(nn.unbox(old['params'])).items():
      np.testing.assert_array_equal(flatten_dict(mapped['decoder']['first_block'])[path], value)

  def test_global_l1_write_and_consumer_gradient(self):
    rows = jnp.arange(6., dtype=jnp.float32).reshape(6, 1, 1, 1, 1)
    def run(rows):
      anchor = jnp.zeros_like(rows[0])
      outputs = []
      for layer in range(6):
        if layer % 3 != 2:
          out, anchor = _mix_local_v_row_anchor(rows[layer], anchor, .1, jnp.array(layer))
          outputs.append(out)
      return jnp.stack(outputs), anchor
    out, anchor = jax.jit(run)(rows)
    np.testing.assert_allclose(out.ravel(), [0, 1, 2.8, 3.7], rtol=1e-6)
    np.testing.assert_array_equal(anchor, rows[1])
    grads = jax.grad(lambda rr: run(rr)[0][2:].sum())(rows)
    np.testing.assert_allclose(grads.ravel(), [0, .2, 0, .9, .9, 0], rtol=1e-6)

  def test_full_blocks_scan_vs_mapped_unroll(self):
    cfg = self.config(EXP)
    for key, value in {
        'num_decoder_layers': 6, 'base_num_decoder_layers': 6,
        'bam_layer_modes': ['local_qk+local_o', 'local_qk+local_o', 'local_qk+full'] * 2,
        'bam_local_o_v_mode': ['rank2', 'rank2', 'none'] * 2,
        'dtype': jnp.float32,
    }.items():
      cfg.get_keys()[key] = value
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    scanned = nn.scan(
        BamLayerPair, variable_axes={'params': cfg.param_scan_axis, 'intermediates': 0},
        split_rngs={'params': True, 'dropout': False},
        in_axes=(nn.broadcast,) * 10 + (0,), length=2,
        metadata_params={nn.PARTITION_NAME: 'layers'})(cfg, mesh, 8, all_global_attention=True)
    block = BamLayerPair(cfg, mesh, 8, all_global_attention=True)
    h = jax.random.normal(jax.random.key(1), (1, 8, 128))
    m = jnp.zeros((1, 8, 32, 32), jnp.float32)
    anchor = jnp.zeros((1, 8, 2, 32), jnp.float32)
    carry = (h, m, anchor)
    args = (jnp.ones((1, 8), jnp.int32), jnp.arange(8)[None],
            jnp.ones((1, 8), jnp.int32), None, True, 'train', None, None, None, None)
    variables = scanned.init({'params': jax.random.key(2), 'aqt': jax.random.key(3)},
                             carry, *args, jnp.arange(2))
    params = nn.unbox(variables['params'])
    self.assertEqual(set(params), {'local_0', 'local_1', 'fetch_2'})
    for offset in (0, 1):
      p = params[f'local_{offset}']['block']['self_attention']
      self.assertIn('W_row_anchor_gate', p)
      np.testing.assert_array_equal(p['W_row_anchor_gate']['kernel'], 0)
      np.testing.assert_allclose(jax.nn.sigmoid(p['row_anchor_gate_bias']), .1, rtol=1e-6)
      # Step-zero LocalV bases are zero: use nonzero read biases to exercise
      # actual anchor values and gradients, not a vacuous all-zero equality.
      p['W_lv_bias'] = jax.random.normal(jax.random.key(10 + offset),
                                       p['W_lv_bias'].shape) * .02
    self.assertNotIn('W_row_anchor_gate', params['fetch_2']['block']['self_attention'])
    wd = train.get_wd_tree(cfg, params)
    self.assertEqual(wd['local_1']['block']['self_attention']['row_anchor_gate_bias'], 0.)
    def scan(p):
      return scanned.apply({'params': p}, carry, *args, jnp.arange(2), mutable='intermediates')
    def unroll(p):
      state = carry
      states = []
      for index in range(2):
        pp = jax.tree.map(lambda x: jnp.take(x, index, axis=cfg.param_scan_axis), p)
        (state, _), _ = block.apply({'params': pp}, state, *args, jnp.array(index),
                                     mutable='intermediates')
        states.append(state)
      return state, states[0]
    (got, _), metrics = jax.jit(scan)(params)
    expected, first_block = jax.jit(unroll)(params)
    for a, b in zip(got, expected):
      np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(first_block[2], expected[2])
    np.testing.assert_allclose(first_block[2], got[2], atol=2e-8, rtol=2e-5)
    self.assertGreater(float(jnp.linalg.norm(got[2])), 0.)
    def score(state):
      return sum(jnp.mean(x*x) for x in state)
    gs = jax.jit(jax.grad(lambda p: score(scan(p)[0][0])))(params)
    gu = jax.jit(jax.grad(lambda p: score(unroll(p)[0])))(params)
    for a, b in zip(jax.tree.leaves(gs), jax.tree.leaves(gu)):
      np.testing.assert_allclose(a, b, atol=5e-5, rtol=5e-4)
    graph = str(jax.make_jaxpr(scan)(params))
    self.assertIn('scan[', graph)
    self.assertNotIn('cond[', graph)
    output = {'scalar': {}}
    train.record_bam_row_anchor_metrics(output, {'intermediates': {'decoder': {
        'layers': metrics['intermediates']}}}, cfg)
    self.assertEqual(len(output['scalar']), 48)
    for layer in (0, 1, 3, 4):
      active = output['scalar'][f'bam/row_anchor/layer_{layer:03d}/active']
      self.assertEqual(float(active), float(layer > 1))
    anchor_rms = output['scalar']['bam/row_anchor/layer_001/anchor_rms']
    for layer in (3, 4):
      np.testing.assert_array_equal(
          output['scalar'][f'bam/row_anchor/layer_{layer:03d}/anchor_rms'], anchor_rms)


if __name__ == '__main__':
  absltest.main()
