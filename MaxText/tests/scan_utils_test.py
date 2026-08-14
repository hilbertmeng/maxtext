"""Tests for explicit/scanned decoder tree conversion."""

import jax.numpy as jnp

from layers import scan_utils


def test_stack_unstack_named_layers_round_trip():
  decoder = {
      'embed': {'kernel': jnp.ones((3, 4))},
      'layers_0': {'a': jnp.arange(6).reshape(2, 3), 'b': jnp.ones((3,))},
      'layers_1': {'a': jnp.arange(6, 12).reshape(2, 3), 'b': 2 * jnp.ones((3,))},
  }
  scanned = scan_utils.stack_named_layers(decoder, 2, axis=1)
  assert scanned['layers']['a'].shape == (2, 2, 3)
  assert scan_utils.parameter_count(scanned) == scan_utils.parameter_count(decoder)
  restored = scan_utils.unstack_named_layers(scanned, 2, axis=1)
  scan_utils.assert_same_unscanned_shapes(decoder, restored)
  for i in range(2):
    for name in ('a', 'b'):
      assert jnp.array_equal(
          restored[f'layers_{i}'][name], decoder[f'layers_{i}'][name])


def test_stack_rejects_heterogeneous_layer_trees():
  decoder = {'layers_0': {'a': jnp.ones(1)}, 'layers_1': {'b': jnp.ones(1)}}
  try:
    scan_utils.stack_named_layers(decoder, 2)
  except ValueError as error:
    assert 'same parameter tree' in str(error)
  else:
    raise AssertionError('heterogeneous layer trees were accepted')
