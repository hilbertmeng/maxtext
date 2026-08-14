"""Lossless transforms between explicit and Linen-scanned decoder layer trees."""

import jax
import jax.numpy as jnp
from flax import traverse_util
from flax.core import freeze, unfreeze


def _as_mutable(tree):
  return unfreeze(tree) if not isinstance(tree, dict) else dict(tree)


def stack_named_layers(decoder_tree, num_layers, axis=1):
  """Replace ``layers_0..layers_N`` with one ``layers`` tree stacked on axis."""
  decoder = _as_mutable(decoder_tree)
  layers = [decoder.pop(f'layers_{i}') for i in range(num_layers)]
  reference = jax.tree_util.tree_structure(layers[0])
  if any(jax.tree_util.tree_structure(layer) != reference for layer in layers[1:]):
    raise ValueError('all scanned decoder layers must have the same parameter tree')
  decoder['layers'] = jax.tree_util.tree_map(
      lambda *values: jnp.stack(values, axis=axis), *layers)
  return freeze(decoder) if not isinstance(decoder_tree, dict) else decoder


def unstack_named_layers(decoder_tree, num_layers, axis=1):
  """Replace a scanned ``layers`` tree with ``layers_0..layers_N`` trees."""
  decoder = _as_mutable(decoder_tree)
  scanned = decoder.pop('layers')
  for i in range(num_layers):
    decoder[f'layers_{i}'] = jax.tree_util.tree_map(
        lambda value, index=i: jnp.take(value, index, axis=axis), scanned)
  return freeze(decoder) if not isinstance(decoder_tree, dict) else decoder


def parameter_count(tree):
  """Return the exact scalar count for a parameter or optimizer-state pytree."""
  return sum(value.size for value in jax.tree_util.tree_leaves(tree))


def assert_same_unscanned_shapes(left, right):
  """Raise with the first path whose shape differs after conversion."""
  left_flat = traverse_util.flatten_dict(unfreeze(left) if not isinstance(left, dict) else left)
  right_flat = traverse_util.flatten_dict(unfreeze(right) if not isinstance(right, dict) else right)
  if left_flat.keys() != right_flat.keys():
    missing = sorted(set(left_flat) ^ set(right_flat))
    raise ValueError(f'parameter paths differ: {missing[:8]}')
  for path in left_flat:
    if left_flat[path].shape != right_flat[path].shape:
      raise ValueError(
          f'parameter shape differs at {"/".join(path)}: '
          f'{left_flat[path].shape} != {right_flat[path].shape}')
