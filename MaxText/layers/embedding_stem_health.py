"""Nine scalar diagnostics for embedding scale and the first residual block."""
import jax
import jax.numpy as jnp
from flax import linen as nn

FORWARD_NAMES = ('embedding_rms', 'attention_rms', 'mlp_rms',
                 'attention_over_embedding', 'mlp_over_embedding', 'mlp_norm_rms')
UPDATE_NAMES = ('embedding_parameter_rms', 'embedding_update_rms',
                'embedding_update_over_parameter_rms')
PREFIX = 'embedding_stem/'


def record(module, cfg, inputs, attention, mlp, mlp_input, index):
  if not getattr(cfg, 'record_embedding_stem_health_metrics', False) or module.is_initializing():
    return
  if isinstance(index, int) and index != 0:
    return
  index = 0 if index is None else index
  def values(_):
    rms = lambda x: jnp.sqrt(jnp.mean(jnp.asarray(x, jnp.float32)**2))
    e, a, f, n = (rms(x) for x in (inputs, attention, mlp, mlp_input))
    return jnp.stack((e, a, f, a/jnp.maximum(e, 1e-12), f/jnp.maximum(e, 1e-12), n))
  stats = jax.lax.cond(jnp.asarray(index) == 0, values,
                       lambda _: jnp.zeros((len(FORWARD_NAMES),), jnp.float32), None)
  module.sow('intermediates', 'embedding_stem_health', jax.lax.stop_gradient(stats))


def metrics(intermediate_outputs, old_params, new_params, step, period):
  """Only the large embedding-table reductions are sampled inside the JIT."""
  if period <= 0:
    raise ValueError('embedding_stem_health_period must be positive')
  decoder = intermediate_outputs['intermediates']['decoder']
  if 'layers' in decoder:  # Flat scanned Pythia MHA.
    layer = decoder['layers']
    forward = (layer['embedding_stem_health'][0][0]
               if 'embedding_stem_health' in layer else
               layer['local_0']['block']['embedding_stem_health'][0][0])
  else:  # Mudd keeps its unscanned full-history layout.
    forward = decoder['layers_0']['block']['embedding_stem_health'][0]
  a = nn.unbox(old_params)['params']['token_embedder']['embedding']
  b = nn.unbox(new_params)['params']['token_embedder']['embedding']
  def updates(_):
    r = jnp.sqrt(jnp.mean(a.astype(jnp.float32)**2))
    d = jnp.sqrt(jnp.mean((b.astype(jnp.float32)-a.astype(jnp.float32))**2))
    return jnp.stack((r, d, d/jnp.maximum(r, 1e-12)))
  update = jax.lax.cond(step % period == 0, updates,
                        lambda _: jnp.zeros((len(UPDATE_NAMES),), jnp.float32), None)
  return {PREFIX+k:v for k,v in zip(FORWARD_NAMES+UPDATE_NAMES,
                                    tuple(forward)+tuple(update))}
