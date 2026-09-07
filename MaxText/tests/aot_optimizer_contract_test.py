"""CPU regression for optimizer setup and serialized executable updates.

Uses a small parameter tree, the real rule builder/optimizer, and the actual
AOT setup entrypoint. This isolates the optimizer contract; it is not a claim
of full-Transformer cross-topology numerical equivalence.
"""
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import unittest
from jax.experimental.serialize_executable import serialize, deserialize_and_load

import exp
import train
import train_compile


def assert_tree_equal(a, b):
  assert jax.tree.structure(a) == jax.tree.structure(b)
  for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
    np.testing.assert_array_equal(x, y)


def check_optimizer_serialization_contract(with_rules, config_class=None):
  cfg = SimpleNamespace(
      wd_mults=(config_class.wd_mults if config_class else
                exp.BamLlama2MediumV2C256ScanAotCleanControl.wd_mults if with_rules else None),
      adam_weight_decay=.1, adam_b1=.9, adam_b2=.95, adam_eps=1e-8,
      adam_eps_root=0., opt_type='adam_pax', init_weights_seed=0, debug=False)
  leaves = {'fetch_mix_scale': jnp.full((1, 24), .25),
            'W_R': {'kernel': jnp.full((4, 8), .006)},
            'W_lq_bias': jnp.full((2, 4), .1),
            'gw_b0': jnp.full((16,), -2.197),
            'W_R_gate_b0': jnp.full((16, 2), -5.293),
            'norm': {'scale': jnp.ones((32,))}}
  params = {'params': {'decoder': {'layers': {'block': {'self_attention': leaves}}}}}
  schedule = lambda s: 3e-4 * jnp.minimum(s / 200., 1.) * jnp.maximum(1. - s / 13500., 0.)
  captured = []

  def abstract(model, tx, config, rng, mesh):
    captured.append(tx)
    return (), (), ()

  with mock.patch.object(train, 'model_init', return_value=params), \
       mock.patch.object(train_compile, 'Transformer', return_value=object()), \
       mock.patch.object(train_compile.quantizations, 'configure_quantization', return_value=None), \
       mock.patch.object(train_compile.max_utils, 'create_learning_rate_schedule', return_value=schedule), \
       mock.patch.object(train_compile.max_utils, 'get_abstract_state', side_effect=abstract), \
       mock.patch.object(train_compile.input_pipeline_interface, 'get_shaped_batch', return_value=()):
    jit_tx = train.create_model_optimizer(cfg, object(), schedule, jax.random.PRNGKey(0))
    train_compile.get_shaped_inputs(None, cfg)
  aot_tx, = captured

  def update(tx):
    def step(p, state, grads):
      delta, state = tx.update(grads, state, p)
      return optax.apply_updates(p, delta), state
    return step

  zeros = jax.tree.map(jnp.zeros_like, params)
  state = jit_tx.init(params)
  assert_tree_equal(state, aot_tx.init(params))
  jit_step = jax.jit(update(jit_tx))
  compiled = jax.jit(update(aot_tx)).lower(params, state, zeros).compile()
  payload, in_tree, out_tree = serialize(compiled)
  restored = deserialize_and_load(payload, in_tree, out_tree)
  # Counters and moments are dynamic: exercise warmup, resumed and final steps
  # through one saved executable, with independent optimizer states.
  p_aot, s_aot = params, state
  p_jit, s_jit = params, state
  for step in (0, 1, 199, 200, 201, 2800, 13499, 13500):
    def set_count(path, value):
      return jnp.asarray(step, value.dtype) if getattr(path[-1], 'name', None) == 'count' else value
    s_aot = jax.tree_util.tree_map_with_path(set_count, s_aot)
    s_jit = jax.tree_util.tree_map_with_path(set_count, s_jit)
    grads = zeros if step < 200 else jax.tree.map(lambda p: jnp.full_like(p, .031), params)
    p_aot, s_aot = restored(p_aot, s_aot, grads)
    p_jit, s_jit = jit_step(p_jit, s_jit, grads)
    assert_tree_equal((p_aot, s_aot), (p_jit, s_jit))
    if step == 199:
      wd = train.get_wd_tree(cfg, params)
      wd = jax.tree.map(lambda _: cfg.adam_weight_decay, params) if wd is None else wd
      for actual, initial, decay in zip(
          jax.tree.leaves(p_aot), jax.tree.leaves(params), jax.tree.leaves(wd)):
        if decay == 0:
          np.testing.assert_array_equal(actual, initial)
        else:
          assert bool(jnp.all(jnp.abs(actual) < jnp.abs(initial)))


class AotOptimizerContractTest(unittest.TestCase):
  def test_only_new_mix_scale_skips_decay(self):
    cls = exp.BamLlama2MediumV2C256ScanAotOldMixScaleOnly
    self.assertEqual(cls.wd_mults, exp.BamLlama2MediumV2C256ScanAotOldGeluMixScaleNoWD.wd_mults)
    params = {'params': {'decoder': {'layers': {
        'self_attention': {'fetch_mix_scale': 1., 'W_R_gate_b0': 1.,
                           'gw_b0': 1., 'fetch_head_mix': {'bias': 1.},
                           'P_loc_up': {'bias': 1., 'kernel': 1.}},
        'pre_self_attention_layer_norm': {'scale': 1.},
        'mlp': {'bias': 1., 'kernel': 1.}}}}}
    cfg = SimpleNamespace(wd_mults=cls.wd_mults, adam_weight_decay=.1)
    expected = jax.tree.map(lambda _: .1, params)
    expected['params']['decoder']['layers']['self_attention']['fetch_mix_scale'] = 0.
    self.assertEqual(train.get_wd_tree(cfg, params), expected)
    check_optimizer_serialization_contract(True, cls)

  def test_old_gate050_keeps_all_decay(self):
    cls = exp.BamLlama2MediumV2C256ScanAotOldGate050FixedAmplitude
    self.assertEqual(cls.wd_mults, [])
    check_optimizer_serialization_contract(False, cls)

  def test_clean_scale_only_preserves_clean_rules(self):
    self.assertEqual(
        exp.BamLlama2MediumV2C256ScanAotCleanMixScaleOnly.wd_mults,
        exp.BamLlama2MediumV2C256ScanAotCleanControl.wd_mults)

  def test_bam_only_exclusions(self):
    cfg = SimpleNamespace(
        wd_mults=exp.BamLlama2MediumV2C256ScanAotBamOnlyWDControl.wd_mults,
        adam_weight_decay=.1)
    bam = {name: 1. for name in (
        'gw_b0', 'W_lq_bias', 'W_lk_bias', 'W_lq_gate_b0',
        'W_lk_gate_b0', 'W_R_gate_b0')}
    bam['P_loc_up'] = {'kernel': 1., 'bias': 1.}
    for name in ('W_R', 'W_R_gate', 'W_gw', 'W_local_qk_packed', 'P_loc_down',
                 'query', 'key', 'value', 'out'):
      bam[name] = {'kernel': 1.}
    block = {'self_attention': bam,
             'pre_self_attention_layer_norm': {'scale': 1.},
             'post_self_attention_layer_norm': {'scale': 1.},
             'mlp': {'wi_0': {'kernel': 1., 'bias': 1.}}}
    params = {'params': {'decoder': {'layers': {'block': block},
                                    'decoder_norm': {'scale': 1.}}}}
    actual = train.get_wd_tree(cfg, params)
    expected = jax.tree.map(lambda _: .1, params)
    expected_bam = expected['params']['decoder']['layers']['block']['self_attention']
    for name in ('gw_b0', 'W_lq_bias', 'W_lk_bias', 'W_lq_gate_b0',
                 'W_lk_gate_b0', 'W_R_gate_b0'):
      expected_bam[name] = 0.
    expected_bam['P_loc_up']['bias'] = 0.
    self.assertEqual(actual, expected)

  def test_with_rules(self):
    check_optimizer_serialization_contract(True)

  def test_without_rules(self):
    check_optimizer_serialization_contract(False)


if __name__ == '__main__':
  unittest.main()
