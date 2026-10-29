"""Native token-major write reference with explicit post-RMS factor multipliers."""
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

class AddressScaledWrite(NamedTuple):
  """Raw factors and independent post-RMS address scale [N,V]."""
  content: jax.Array
  logits: jax.Array
  address: jax.Array
  scale: jax.Array

def scale_shape(config, heads, width):
  return (heads,) if getattr(config, 'bam_write_address_norm_scale_per_head', False) else (heads, width)

def scale_axes(config):
  return ('q_heads',) if getattr(config, 'bam_write_address_norm_scale_per_head', False) else ('q_heads', 'kv')

def address_scale(config, value, width, dtype):
  value = jnp.asarray(value, dtype)
  return jnp.broadcast_to(value[:, None], (value.shape[0], width)) if getattr(config, 'bam_write_address_norm_scale_per_head', False) else value

def scale_initializer(config):
  value=getattr(config,'bam_write_address_norm_scale_init',1.0)
  if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
    raise ValueError('bam_write_address_norm_scale_init must be a finite scalar')
  return float(value)

def native_write(m, groups, *, epsilon):
  """Accumulate normalized outer products; factor type chooses the scaled side."""
  dtype=m.dtype;acc=m.astype(jnp.float32)
  norm=lambda x:normalizations.rms_norm(x,dtype=x.dtype,epsilon=epsilon,statistics_dtype=jnp.float32)
  for group in groups:
    if isinstance(group,(ScaledWrite,AddressScaledWrite)):content,logits,address,scale=group
    else:content,logits,address=group;scale=None
    content=norm(content)
    if isinstance(group,ScaledWrite):
      assert scale.shape==content.shape[-2:],(scale.shape,content.shape)
      content=content*jnp.asarray(scale,content.dtype)
    content=jax.nn.sigmoid(logits.astype(jnp.float32)).astype(content.dtype)[...,None]*content
    address=norm(address)
    if isinstance(group,AddressScaledWrite):
      assert scale.shape == address.shape[-2:]
      address = address * jnp.asarray(scale, address.dtype)
    acc=acc+jnp.einsum('btnv,btnk->btvk',address.astype(jnp.float32),content.astype(jnp.float32),precision=jax.lax.Precision.HIGHEST)
  return acc.astype(dtype)
