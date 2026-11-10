"""Layer-local parameter transplants; the differentiated model graph is unchanged."""
import hashlib
import json


def select_leaf(path, old, new, selected, num_layers):
  """Select physical layers for the immutable single-layer-scan XL runtime."""
  import jax.numpy as jnp
  from xlprop_parameter_family_probe import parameter_family

  family = parameter_family(path, fine_attention=True, split_vo=True)
  layers = selected.get(family, ())
  if not layers:
    return old
  names = tuple(getattr(key, 'key', None) for key in path)
  if 'layers' in names and 'block' in names:
    assert old.ndim >= 2 and old.shape[1] == num_layers - 1, (names, old.shape)
    mask = jnp.asarray([i in layers for i in range(num_layers - 1)])
    shape = [1] * old.ndim
    shape[1] = num_layers - 1
    return jnp.where(mask.reshape(shape), new, old)
  assert 'final_local_layer' in names, ('Unexpected selected layer layout', names)
  return new if num_layers - 1 in layers else old


def run(train, model, config, mesh, old, new, data, output, old_checkpoint, *, plans_uri=None):
  import jax
  import numpy as np
  from flax import linen as nn

  assert config.num_decoder_layers == 28 and not config.bam_pair_scan
  v, o, wo = 'attention_local_v', 'attention_local_o', 'attention_output_projection'
  whole = list(range(28))
  plans = [('old', {}), ('pair_all', {v:whole, wo:whole}),
           ('vo_output_all', {v:whole, o:whole, wo:whole})]
  for lo, hi in ((0,7),(7,14),(14,21),(21,28),(0,14),(14,28)):
    region = list(range(lo,hi))
    plans.append((f'pair_only_{lo}_{hi}', {v:region, wo:region}))
    plans.append((f'pair_except_{lo}_{hi}',
                  {v:[i for i in whole if i not in region], wo:[i for i in whole if i not in region]}))
  plans += [('v_early_wo_late', {v:list(range(14)),wo:list(range(14,28))}),
            ('v_late_wo_early', {v:list(range(14,28)),wo:list(range(14))})]
  if plans_uri:
    import tensorflow as tf
    with tf.io.gfile.GFile(plans_uri) as handle:
      requested = json.load(handle)
    assert isinstance(requested,list) and 0 < len(requested) <= 60
    plans = plans[:3]
    for row in requested:
      assert row['label'] not in {label for label,_ in plans}
      selected = row['selected_layers']
      assert isinstance(selected,dict) and set(selected) <= {v,o,wo}
      for layers in selected.values():
        assert isinstance(layers,list) and len(layers)==len(set(layers))
        assert all(type(i) is int and 0 <= i < 28 for i in layers)
      plans.append((row['label'],selected))

  @jax.jit
  def measure(params, batch):
    def loss(params):
      return train.loss_fn(model, config, dict(batch), jax.random.PRNGKey(0), params, is_train=True)[0]
    value, grads = jax.value_and_grad(loss)(params)
    return value, train.max_utils.l2norm_pytree(grads)

  result = dict(scope='Frozen post-update5000 B1; parameter-value transplants only; no optimizer updates',
                old_checkpoint=old_checkpoint, plans_uri=plans_uri, cases=[],
                input_sha256=hashlib.sha256(np.asarray(data['inputs']).tobytes()).hexdigest())
  with mesh, nn.partitioning.axis_rules(config.logical_axis_rules):
    for label, plan in [('all_new_control',None)] + plans + [('all_new_repeat',None)]:
      params = new if plan is None else jax.tree.map_with_path(
          lambda path,a,b:select_leaf(path,a,b,plan,config.num_decoder_layers), old,new)
      ce, raw = map(float,jax.device_get(measure(params,data)))
      row = dict(label=label, selected_layers=plan, mean_ce=ce, raw_grad_norm=raw)
      result['cases'].append(row)
      (output/'vo-layer-screen.json').write_text(json.dumps(result,indent=2)+'\n')
      print('VO_LAYER_SCREEN',json.dumps(row),flush=True)
  first,last = result['cases'][0],result['cases'][-1]
  assert (first['mean_ce'],first['raw_grad_norm']) == (last['mean_ce'],last['raw_grad_norm'])
  controls = {row['label']:row['raw_grad_norm'] for row in result['cases']}
  for label,expected in [('all_new_control',4751.140137),('old',1.451448),
                         ('pair_all',67.953827),('vo_output_all',3588.412109)]:
    assert abs(controls[label]-expected) <= 1e-5*max(1.,expected), (label,controls[label],expected)
  print('VO_LAYER_SCREEN_CONTROLS_EXACT',flush=True)
