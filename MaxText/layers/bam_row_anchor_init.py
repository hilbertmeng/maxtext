"""Match shared initialization when extracting the first LLF block from scan."""
from types import SimpleNamespace
from flax.traverse_util import flatten_dict, unflatten_dict
import jax.numpy as jnp
import pyconfig


def map_row_anchor_params(target, source, scan_axis):
  """Copy common parameters; preserve target partition metadata and new zero keys."""
  flat_source = flatten_dict(source)
  result = {}
  for path, target_value in flatten_dict(target).items():
    source_path = path
    selection = None
    if path[:2] == ('decoder', 'first_block'):
      source_path = ('decoder', 'layers') + path[2:]
      selection = 0
    elif path[:2] == ('decoder', 'layers'):
      selection = slice(1, None)
    if source_path not in flat_source:
      assert path[-1] == 'lv_direct_row_bias' or 'W_lv_direct_row' in path, path
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


def initialize_from_row_anchor_parent(model, variables, config, key, input_shape):
  """Initialization only; resumes still load the target RUN's own checkpoints."""
  parent_keys = dict(config.get_keys(), bam_l1_direct_local_v_row=False)
  parent_cfg = pyconfig.HyperParameters(SimpleNamespace(keys=parent_keys))
  parent = model.clone(config=parent_cfg)
  inputs = jnp.ones(input_shape, jnp.int32)
  source = parent.init({'params': key, 'dropout': key, 'aqt': key}, inputs, inputs, inputs, inputs)
  variables['params'] = map_row_anchor_params(
      variables['params'], source['params'], config.param_scan_axis)
  return variables
