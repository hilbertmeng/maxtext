"""Device-side, scalar-only diagnostics; never changes returned model tensors."""
from functools import partial
import json
import os
import threading
import jax
import jax.numpy as jnp
import numpy as np

_LOCK = threading.Lock()
TEMPERATURE_FACTORS = (.75, .9, 1., 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.4, 1.45, 1.5, 1.6, 1.75, 2.)

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


def numeric_readout(original, precise, targets, mask):
  """Paired output-quantization controls on exactly the same hidden and bf16 weights."""
  z = precise.astype(jnp.float32)
  old = original.astype(jnp.float32)
  centered = z-z.mean(-1,keepdims=True)
  # Explicit IEEE round-to-nearest avoids XLA eliding a bf16 round-trip cast.
  def round_bf16(v):
    bits=jax.lax.bitcast_convert_type(v,jnp.uint32)
    bits=(bits+jnp.uint32(0x7fff)+((bits>>16)&jnp.uint32(1))) & jnp.uint32(0xffff0000)
    return jax.lax.bitcast_convert_type(bits,jnp.float32)
  centered_bf16 = round_bf16(centered)
  rounded = round_bf16(z)
  m = (mask>0).astype(jnp.float32)
  count = jnp.maximum(m.sum(),1.)
  avg = lambda v: (v*m).sum()/count
  def ce(v):
    target = jnp.take_along_axis(v,targets[...,None],axis=-1)[...,0]
    return avg(jax.nn.logsumexp(v,axis=-1)-target)
  def error(v):
    d=v-z
    d=d-d.mean(-1,keepdims=True)
    return jnp.sqrt(avg(jnp.mean(d*d,-1)))
  base=ce(old)
  # softmax-gradient error is normalized relative to the original CE logits gradient.
  onehot=jax.nn.one_hot(targets,z.shape[-1])
  gp=jax.nn.softmax(z,-1)-onehot
  go=jax.nn.softmax(old,-1)-onehot
  gc=jax.nn.softmax(centered_bf16,-1)-onehot
  norm=lambda a: avg(jnp.sum(a*a,-1))
  # Match the actual custom-CE backward, including bf16 output rounding.
  import max_utils
  actual_grad = jax.grad(lambda v: jnp.sum(max_utils.cross_entropy_with_logits(v,onehot,0.)[0]))(original).astype(jnp.float32)
  grad_mass=actual_grad.sum(-1)
  pe=jax.nn.softmax(z,-1)
  diff=old-z
  diff=diff-jnp.sum(pe*diff,-1,keepdims=True)
  active_error=jnp.sqrt(avg(jnp.sum(pe*diff*diff,-1)))
  target_z=jnp.take_along_axis(z,targets[...,None],axis=-1)[...,0]
  values=jnp.stack((base,ce(z),ce(centered),ce(centered_bf16),ce(rounded),
                   error(old),error(rounded),
                   jnp.sqrt(avg(jnp.mean((centered_bf16-centered)**2,-1))),
                   jnp.sqrt(norm(go-gp)/jnp.maximum(norm(gp),1e-30)),
                   jnp.sqrt(norm(gc-gp)/jnp.maximum(norm(gp),1e-30)),
                   avg(grad_mass),jnp.sqrt(avg(grad_mass**2)),
                   avg((grad_mass>0).astype(jnp.float32)),
                   jnp.sqrt(norm(actual_grad-gp)/jnp.maximum(norm(gp),1e-30)),
                   avg(jnp.max(z,-1)),avg(target_z),active_error,count))
  record('numeric_readout',-1,
         ('ce_original','ce_fp32','ce_fp32_centered','ce_centered_bf16','ce_rounded_fp32',
          'original_centered_error_rms','rounded_centered_error_rms','centered_bf16_error_rms',
          'relative_logit_gradient_error','centered_relative_logit_gradient_error',
          'actual_ce_gradient_mass_mean','actual_ce_gradient_mass_rms',
          'actual_ce_gradient_mass_positive_fraction','actual_ce_relative_gradient_error',
          'max_logit_mean','target_logit_mean','probability_weighted_centered_error_rms','tokens'),values)


def write_update(before, static, dynamic, layer, tag):
  x=before.astype(jnp.float32)
  st=static.astype(jnp.float32)
  dy=dynamic.astype(jnp.float32)
  update=st+dy
  rms=lambda a:jnp.sqrt(jnp.mean(a*a))
  mr=jnp.maximum(rms(x),1e-30)
  cosine=lambda a,b:jnp.mean(a*b)/jnp.maximum(rms(a)*rms(b),1e-30)
  vals=jnp.stack((mr,rms(st),rms(dy),rms(update),rms(update)/mr,
                 cosine(update,x),cosine(st,dy)))
  record(tag,layer,('carry_rms','static_rms','dynamic_rms','update_rms',
                   'update_to_carry','update_carry_cosine','static_dynamic_cosine'),vals)


def activation_geometry(x, layer, tag):
  v=x.astype(jnp.float32)
  mean=v.mean((0,1))
  power=jnp.mean(v*v)
  mean_power=jnp.mean(mean*mean)
  values=jnp.stack((jnp.sqrt(power),jnp.sqrt(mean_power),
                    jnp.sqrt(jnp.maximum(power-mean_power,0)),
                    mean_power/jnp.maximum(power,1e-30),jnp.max(jnp.abs(v))))
  record(tag,layer,('rms','token_common_rms','token_variable_rms',
                    'token_common_energy_fraction','absmax'),values)


def emit_activation_tail(tag, layer, token_rms, ids, fractions):
  row={'tag':tag+'_tail','layer':int(layer),'token_rms':np.asarray(token_rms).tolist(),
       'top_neuron_ids':np.asarray(ids).tolist(),'top_neuron_energy_fractions':np.asarray(fractions).tolist()}
  with _LOCK,open(os.environ['RMT_HEALTH_FILE'],'a') as f:f.write(json.dumps(row)+'\n')


def activation_tail(x,layer,tag,num_layers):
  def measure(_):
    v=x.astype(jnp.float32)
    power=jnp.mean(v*v,(0,1))
    top,ids=jax.lax.top_k(power,4)
    token_rms=jnp.sqrt(jnp.mean(v*v,-1)).reshape(-1)
    jax.debug.callback(partial(emit_activation_tail,tag),layer,token_rms,ids,
                       top/jnp.maximum(power.sum(),1e-30))
    return jnp.asarray(0)
  jax.lax.cond(layer>=num_layers-2,measure,lambda _:jnp.asarray(0),None)


def emit_token_losses(q0, ce, mask):
  row={'tag':'token_losses','layer':-1,'q0':int(q0),'ce':np.asarray(ce).reshape(-1).tolist(),
       'mask':np.asarray(mask).reshape(-1).tolist()}
  with _LOCK,open(os.environ['RMT_HEALTH_FILE'],'a') as f:f.write(json.dumps(row)+'\n')


def token_losses(logits,targets,mask,q0):
  z=logits.astype(jnp.float32)
  ce=jax.nn.logsumexp(z,-1)-jnp.take_along_axis(z,targets[...,None],-1)[...,0]
  jax.debug.callback(emit_token_losses,q0,ce,mask)
