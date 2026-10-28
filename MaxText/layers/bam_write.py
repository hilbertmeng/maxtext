"""Native token-major write reference with post-RMS content multipliers."""
import math
from typing import NamedTuple
import jax
import jax.numpy as jnp
from layers import normalizations

class ScaledWrite(NamedTuple):
  """Raw content/logits/address plus independent scale[N,K] AFTER content RMSNorm."""
  content: jax.Array
  logits: jax.Array
  address: jax.Array
  scale: jax.Array

def scale_initializer(config):
  value=getattr(config,'bam_write_content_norm_scale_init',1.0)
  if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
    raise ValueError('bam_write_content_norm_scale_init must be a finite scalar')
  return float(value)

def native_write(m, groups, *, epsilon):
  """M[B,T,V,K] += sum_n norm(address) outer (sigmoid(logits)*scale*norm(content))."""
  dtype=m.dtype;acc=m.astype(jnp.float32)
  norm=lambda x:normalizations.rms_norm(x,dtype=x.dtype,epsilon=epsilon,statistics_dtype=jnp.float32)
  for group in groups:
    if isinstance(group,ScaledWrite):content,logits,address,scale=group
    else:content,logits,address=group;scale=None
    content=norm(content)
    if scale is not None:
      assert scale.shape==content.shape[-2:],(scale.shape,content.shape)
      content=content*jnp.asarray(scale,content.dtype)
    content=jax.nn.sigmoid(logits.astype(jnp.float32)).astype(content.dtype)[...,None]*content
    address=norm(address)
    acc=acc+jnp.einsum('btnv,btnk->btvk',address.astype(jnp.float32),content.astype(jnp.float32),precision=jax.lax.Precision.HIGHEST)
  return acc.astype(dtype)
