"""Preserve parent step-zero parameters when separating the first LLF block."""
from types import SimpleNamespace
from flax.traverse_util import flatten_dict, unflatten_dict
import jax.numpy as jnp
import pyconfig


def map_m_relay_params(target, source, scan_axis):
  flat_source = flatten_dict(source)
  result = {}
  for path, target_value in flatten_dict(target).items():
    source_path, selection = path, None
    if path[:2] == ('decoder', 'first_block'):
      source_path = ('decoder', 'layers') + path[2:]
      selection = 0
    elif path[:2] == ('decoder', 'layers'):
      selection = slice(1, None)
    if source_path not in flat_source:
      assert any(name in path for name in ('m_relay_scale', 'm_relay_gate_b0', 'm_relay_amplitude_scale')), path
      result[path] = target_value
      continue
    value = flat_source[source_path]
    value = value.value if hasattr(value, 'unbox') else value
    if selection is not None:
      indices = [slice(None)] * value.ndim
      indices[scan_axis] = selection
      value = value[tuple(indices)]
    old = target_value.value if hasattr(target_value, 'unbox') else target_value
    assert value.shape == old.shape, (path, value.shape, old.shape)
    result[path] = target_value.replace(value=value) if hasattr(target_value, 'unbox') else value
  return unflatten_dict(result)


def initialize_from_m_relay_parent(model, variables, config, key, input_shape):
  parent_cfg = pyconfig.HyperParameters(SimpleNamespace(
      keys=dict(config.get_keys(), bam_m_relay_anchor=0)))
  parent = model.clone(config=parent_cfg)
  inputs = jnp.ones(input_shape, jnp.int32)
  source = parent.init({'params': key, 'dropout': key, 'aqt': key}, inputs, inputs, inputs, inputs)
  variables['params'] = map_m_relay_params(variables['params'], source['params'], config.param_scan_axis)
  return variables
