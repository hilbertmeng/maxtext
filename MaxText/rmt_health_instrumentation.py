"""Device-side, scalar-only diagnostics; never changes returned model tensors."""
from functools import partial
import json
import os
import threading
import jax
import jax.numpy as jnp
import numpy as np

_LOCK = threading.Lock()
TEMPERATURE_FACTORS = (.5, .75, .9, 1., 1.1, 1.25, 1.5)

def emit(tag, names, layer, values):
  record = {'tag': tag, 'layer': int(layer), **dict(zip(names, np.asarray(values).tolist()))}
  with _LOCK, open(os.environ['RMT_HEALTH_FILE'], 'a') as f:
    f.write(json.dumps(record) + '\n')

def record(tag, layer, names, values):
  jax.debug.callback(partial(emit, tag, names), layer, values)

def readout(hidden, logits, targets, mask):
  """Logit offset is softmax-invariant: use vocabulary-centered scale."""
  z = logits.astype(jnp.float32)
  valid = (mask > 0).astype(jnp.float32)
  count = jnp.maximum(valid.sum(), 1.)
  weighted = lambda x: jnp.sum(x * valid) / count
  mean = z.mean(-1, keepdims=True)
  lp = jax.nn.log_softmax(z, axis=-1)
  p = jnp.exp(lp)
  target_z = jnp.take_along_axis(z, targets[..., None], axis=-1)[..., 0]
  factors = jnp.asarray(TEMPERATURE_FACTORS, jnp.float32)
  def ce(factor):
    return weighted(jax.nn.logsumexp(z * factor, axis=-1) - target_z * factor)
  values = jnp.stack((
      jnp.sqrt(weighted(jnp.mean(hidden.astype(jnp.float32)**2, -1))),
      jnp.sqrt(weighted(jnp.mean((z - mean)**2, -1))),
      weighted(mean[..., 0]), weighted(-jnp.sum(p * lp, -1)),
      weighted(jnp.max(p, -1)), weighted((jnp.argmax(z, -1) == targets).astype(jnp.float32)),
      weighted(jnp.sum(p * z, -1) - target_z), count,
      *[ce(f) for f in TEMPERATURE_FACTORS]))
  names = ('hidden_rms', 'centered_logits_rms', 'logits_mean', 'entropy',
           'top1_probability', 'accuracy', 'ce_logscale_derivative', 'tokens',
           *[f'ce_alpha_{f:g}' for f in TEMPERATURE_FACTORS])
  record('readout', -1, names, values)

def attention(alpha, query, key, layer, q0, valid, selected_layers):
  pred = jnp.any(layer == jnp.asarray(selected_layers))
  def measure(_):
    p = alpha.astype(jnp.float32)
    entropy = -jnp.sum(jnp.where(p > 0, p * jnp.log(jnp.maximum(p, 1e-30)), 0), -1)
    eligible = jnp.any(valid, -1)[:, None, :]
    denom = jnp.maximum(eligible.sum() * p.shape[1], 1)
    avg = lambda x: jnp.sum(jnp.where(eligible, x, 0)) / denom
    values = jnp.stack((avg(entropy), avg(jnp.max(p, -1)),
                        avg(jnp.exp(entropy)), jnp.asarray(q0, jnp.float32)))
    record('attention', layer, ('entropy', 'max_probability', 'effective_sources', 'q0'), values)
    return jnp.asarray(0)
  jax.lax.cond(pred, measure, lambda _: jnp.asarray(0), operand=None)

def matrix_structure(matrix, layer, tag, selected_layers):
  """Participation rank and dominant-energy proxy on fixed token positions."""
  pred = jnp.any(layer == jnp.asarray(selected_layers))
  def measure(_):
    positions = sorted(set(min(i, matrix.shape[1]-1) for i in (0,256,1024,2048,4095)))
    x = matrix[:, jnp.asarray(positions)].astype(jnp.float32)
    gram = jnp.einsum('btkv,btjv->btkj',x,x)
    tr = jnp.trace(gram,axis1=-2,axis2=-1)
    rank = tr**2 / jnp.maximum(jnp.sum(gram**2,axis=(-2,-1)),1e-30)
    v = jnp.ones(gram.shape[:-1],jnp.float32)
    for _ in range(10):
      v = jnp.einsum('btkj,btj->btk',gram,v)
      v /= jnp.maximum(jnp.linalg.norm(v,axis=-1,keepdims=True),1e-30)
    energy = jnp.einsum('btk,btkj,btj->bt',v,gram,v) / jnp.maximum(tr,1e-30)
    # Top fraction is a power-iteration estimate, not an exact singular value.
    values = jnp.stack((jnp.mean(rank),jnp.min(rank),jnp.mean(energy),
                        jnp.sqrt(jnp.mean(x**2))))
    record(tag,layer,('participation_rank_mean','participation_rank_min',
                      'top_energy_power_estimate','rms'),values)
    return jnp.asarray(0)
  jax.lax.cond(pred,measure,lambda _:jnp.asarray(0),operand=None)
