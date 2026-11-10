"""Read-only, identity-VJP boundary capture with a plain-gradient fidelity gate."""
from functools import partial
import json


def make_identity_tap(record):
  import jax
  @partial(jax.custom_vjp, nondiff_argnums=(2,))
  def tap(x, layer, site):
    jax.debug.callback(partial(record, 'forward', site=site), layer, value=x, ordered=True)
    return x
  def forward(x, layer, site):
    jax.debug.callback(partial(record, 'forward', site=site), layer, value=x, ordered=True)
    return x, layer
  def backward(site, layer, cotangent):
    jax.debug.callback(partial(record, 'backward', site=site), layer, value=cotangent, ordered=True)
    return cotangent, None
  tap.defvjp(forward, backward)
  return tap


def run(train, model, config, mesh, params, batch, output, target_layer):
  import jax
  import jax.numpy as jnp
  import numpy as np
  from flax import linen as nn
  from layers import fusion, linears, normalizations

  stats, arrays, layer_stack = {}, {}, []

  def record(direction, layer, site, value):
    layer = int(np.asarray(layer))
    a = np.asarray(value, np.float32)
    l2 = float(np.sqrt(np.einsum('i,i->', a.reshape(-1), a.reshape(-1), dtype=np.float64)))
    stats.setdefault(str(layer), {}).setdefault(site, {})[direction] = dict(
        l2=l2, rms=l2 / np.sqrt(a.size), absmax=float(np.max(np.abs(a))),
        finite=bool(np.isfinite(a).all()), shape=list(a.shape))
    if layer == target_layer:
      arrays[(site, direction)] = a.copy()

  tap = make_identity_tap(record)

  def intercept(next_fun, args, kwargs, context):
    module = context.module
    if context.method_name != '__call__':
      return next_fun(*args, **kwargs)
    if isinstance(module, fusion.SubDecoderLayer):
      layer = kwargs.get('layer_index', module.layer_inx)
      layer_stack.append(module.layer_inx if layer is None else layer)
      try:
        return next_fun(*args, **kwargs)
      finally:
        layer_stack.pop()
    name = None
    if isinstance(module, linears.MlpBlock) and module.name in ('mlp', 'memory_mlp'):
      name = module.name
    elif isinstance(module, normalizations.RMSNorm) and module.name == 'post_self_attention_layer_norm':
      name = 'mlp_norm'
    if name is None:
      return next_fun(*args, **kwargs)
    args = list(args)
    args[0] = tap(args[0], layer_stack[-1], name + '_input')
    return tap(next_fun(*args, **kwargs), layer_stack[-1], name + '_output')

  def measure(p, data, *, capture):
    def loss(p):
      if capture:
        with nn.intercept_methods(intercept):
          return train.loss_fn(model, config, dict(data), jax.random.PRNGKey(0), p, is_train=True)[0]
      return train.loss_fn(model, config, dict(data), jax.random.PRNGKey(0), p, is_train=True)[0]
    value, grads = jax.value_and_grad(loss)(p)
    return value, train.max_utils.l2norm_pytree(grads)

  output.mkdir(exist_ok=True, parents=True)
  data = {k: jnp.asarray(v[:1]) for k, v in batch.items()}
  with mesh, nn.partitioning.axis_rules(config.logical_axis_rules):
    compiled = jax.jit(measure, static_argnames=('capture',))
    plain = tuple(map(float, jax.device_get(compiled(params, data, capture=False))))
    print('IDENTITY_TAP_PLAIN_CONTROL', plain, flush=True)
    captured = tuple(map(float, jax.device_get(compiled(params, data, capture=True))))
    jax.effects_barrier()
  ce_error = abs(captured[0] - plain[0])
  norm_error = abs(captured[1] - plain[1]) / max(plain[1], 1e-30)
  fidelity = dict(passed=ce_error <= .001 and norm_error <= .01,
      plain_ce=plain[0], plain_raw_grad_norm=plain[1], captured_ce=captured[0],
      captured_raw_grad_norm=captured[1], ce_error=ce_error, raw_grad_relative_error=norm_error,
      scope='Frozen B1 adjacent-state function; identity primal and identity VJP, no zero additions',
      target_layer=target_layer)
  for layer, sites in stats.items():
    for name in ('mlp', 'memory_mlp', 'mlp_norm'):
      a = sites.get(name + '_input', {}).get('backward')
      b = sites.get(name + '_output', {}).get('backward')
      if a and b:
        sites[name + '_directional_gain'] = a['rms'] / max(b['rms'], 1e-30)
  (output / 'identity-tap-fidelity.json').write_text(json.dumps(fidelity, indent=2) + '\n')
  (output / 'layer-boundaries.json').write_text(json.dumps(stats, indent=2) + '\n')
  for (site, direction), array in arrays.items():
    np.save(output / f'{site}-{direction}.npy', array)
  print('IDENTITY_TAP_FIDELITY', json.dumps(fidelity), flush=True)
  if not fidelity['passed']:
    raise RuntimeError('Identity taps failed the plain-gradient gate; do not attribute the original point')
