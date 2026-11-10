"""Frozen B1 parameter-family transplants; no optimizer updates or probes."""
import hashlib
import json


def run(train, model, config, mesh, new, batch, output, old_checkpoint, *, fine_attention=False, plans_uri=None):
  import jax
  import jax.numpy as jnp
  import numpy as np
  import orbax.checkpoint as ocp
  from flax import linen as nn
  from flax.core import unfreeze
  from xlprop_parameter_family_probe import parameter_family

  assert '/diagnostics/' in old_checkpoint and old_checkpoint.endswith('/items')
  abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), new)
  cp = ocp.PyTreeCheckpointer()
  restored = cp.restore(old_checkpoint, item={'params': abstract}, transforms={},
      restore_args=ocp.checkpoint_utils.construct_restore_args({'params': abstract}))
  cp.close()
  old = unfreeze(nn.unbox(restored['params']))
  del restored, abstract
  assert jax.tree.structure(old) == jax.tree.structure(new)
  families = jax.tree.map_with_path(lambda path, _: parameter_family(
      path, fine_attention=fine_attention, split_vo=fine_attention), new)
  names = sorted(set(jax.tree.leaves(families)))
  data = {k:jnp.asarray(v[:1]) for k,v in batch.items()}

  @jax.jit
  def measure(params, data):
    def loss(params):
      return train.loss_fn(model, config, dict(data), jax.random.PRNGKey(0), params, is_train=True)[0]
    value, grads = jax.value_and_grad(loss)(params)
    return value, train.max_utils.l2norm_pytree(grads)

  plans = [tuple(names), (), ('ordinary_mlp',), tuple(n for n in names if n != 'ordinary_mlp')]
  eligible = [n for n in names if n.startswith('attention_') or n == 'read_scales'] if fine_attention else names
  plans += [(n,) for n in eligible]
  plans += [tuple(n for n in names if n != removed) for removed in eligible]
  plans = list(dict.fromkeys(plans))
  labels = {}
  if plans_uri:
    import tensorflow as tf
    with tf.io.gfile.GFile(plans_uri) as f:
      requested = json.load(f)
    assert isinstance(requested, list) and 0 < len(requested) <= 64
    plans = []
    for row in requested:
      selected = tuple(sorted(row['new_families']))
      assert len(set(selected)) == len(selected) and set(selected) <= set(names), row
      assert selected not in labels, 'Duplicate parameter intervention'
      plans.append(selected); labels[selected] = row['label']
    plans = [tuple(names)] + [selected for selected in plans if set(selected) != set(names)]
  result = dict(scope='Frozen adjacent-state B1 function; change parameter values only, no training updates',
      old_checkpoint=old_checkpoint,
      fine_attention=fine_attention,
      plans_uri=plans_uri,
      input_sha256=hashlib.sha256(np.asarray(batch['inputs'][:1]).tobytes()).hexdigest(),
      families={n:sum(x.size for x,f in zip(jax.tree.leaves(new),jax.tree.leaves(families)) if f==n) for n in names},
      cases=[])
  output.mkdir(exist_ok=True, parents=True)
  with mesh, nn.partitioning.axis_rules(config.logical_axis_rules):
    for selected in plans + [tuple(names)]:
      params = jax.tree.map(lambda a,b,f:b if f in selected else a, old, new, families)
      ce, raw = map(float, jax.device_get(measure(params, data)))
      row = dict(new_families=selected, mean_ce=ce, raw_grad_norm=raw,
                 label=labels.get(tuple(sorted(selected))))
      result['cases'].append(row)
      (output/'single-device-families.json').write_text(json.dumps(result, indent=2)+'\n')
      print('SINGLE_DEVICE_FAMILY', json.dumps(row), flush=True)
  first = next(r for r in result['cases'] if set(r['new_families']) == set(names))
  last = result['cases'][-1]
  assert (first['mean_ce'], first['raw_grad_norm']) == (last['mean_ce'],last['raw_grad_norm']), 'Frozen repeat control moved'
  print('SINGLE_DEVICE_FAMILY_REPEAT_EXACT', flush=True)
