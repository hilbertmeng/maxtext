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


def check_optimizer_serialization_contract(with_rules):
  cfg = SimpleNamespace(
      wd_mults=exp.BamLlama2MediumV2C256ScanAotCleanControl.wd_mults if with_rules else None,
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
      actual = p_aot['params']['decoder']['layers']['block']['self_attention']
      if with_rules:
        for name in ('fetch_mix_scale', 'W_lq_bias', 'gw_b0', 'W_R_gate_b0', 'norm'):
          assert_tree_equal(actual[name], leaves[name])
      else:
        assert float(actual['fetch_mix_scale'][0, 0]) < .25
      assert float(actual['W_R']['kernel'][0, 0]) < .006


class AotOptimizerContractTest(unittest.TestCase):
  def test_with_rules(self):
    check_optimizer_serialization_contract(True)

  def test_without_rules(self):
    check_optimizer_serialization_contract(False)


if __name__ == '__main__':
  unittest.main()

