from typing import Any, Optional

import jax
from flax import linen as nn
import jax.numpy as jnp
from jax.sharding import Mesh

from layers import initializers
from layers import normalizations
from layers import linears
from layers import quantizations 

Quant = quantizations.AqtQuantization
NdInitializer = initializers.NdInitializer
nd_dense_init = initializers.nd_dense_init


def shift_1d(inputs, offset: int, axis: int):
  paddings = [
      ((max(offset, 0), -min(offset, 0)) if i == axis else (0, 0))
      for i in range(len(inputs.shape))
  ]
  input_length = jnp.shape(inputs)[axis]
  padded_inputs = jnp.pad(inputs, paddings, mode='edge')
  if offset > 0:
    output = jax.lax.slice_in_dim(
        padded_inputs, start_index=0, limit_index=input_length, axis=axis
    )
  else:
    output = jax.lax.slice_in_dim(
        padded_inputs,
        start_index=-offset,
        limit_index=input_length - offset,
        axis=axis,
    )
  return output


class KVshift(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  num_kv_heads: int = 16
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  num_kv_heads: int = None
  
  def setup(self):
    cfg = self.config
    norm_kwargs = {
                "dtype": cfg.dtype,
                "weight_dtype": cfg.weight_dtype,
                "epsilon": cfg.normalization_layer_epsilon,
                }
    if not cfg.kv_shift_skip_knorm:
      self.kv_shift_norm = normalizations.get_rmsnorm("kv_shift_knorm", cfg)
    self.kv_shift_prenorm = normalizations.get_rmsnorm("kv_shift_prenorm", cfg)
    
    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    self.q_shift = cfg.use_q_shift
    self.num_shifts = 2 if not self.q_shift else 3
    self.kv_shift_hidden_way = cfg.kv_shift_hidden_way
    
    if self.kv_shift_hidden_way in ['kv', 'qkv'] and cfg.kv_shift_flash: # kv
      for mode in self.kv_shift_hidden_way:
        setattr(self, f'dw_proj_{mode}', linears.DenseGeneral(
                                    (self.num_kv_heads, ),
                                    kernel_init=initializers.contant_dense_init(0.0),
                                    kernel_axes=('embed', "kv_heads"),
                                    use_bias=False,
                                    name=f'kv_shift_proj_{mode}',
                                    **kwargs))
    else:
      self.dw_proj = linears.DenseGeneral(
                                    (self.num_kv_heads * self.num_shifts, ),
                                      kernel_init=initializers.contant_dense_init(0.0),
                                      kernel_axes=('embed', "kv_heads"),
                                      use_bias=False,
                                      name='kv_shift_proj',
                                      **kwargs)
      
  @nn.compact
  def __call__(
      self,
      inputs_q, # BTD
      query, # BTND
      key, # BTND
      value, # BTND 
      inputs_k=None, # BTD
      inputs_v=None, # BTD
      inputs_m=None, # BTD
  ):
    inputs = inputs_q

    if self.config.kv_shift_flash:
      kg = jax.nn.sigmoid(self.dw_proj_k(inputs_k))[..., jnp.newaxis]
      vg = jax.nn.sigmoid(self.dw_proj_v(inputs_v))[..., jnp.newaxis]
      key = key * kg + (1-kg) * shift_1d(key, offset=1, axis=1)
      value = value * vg + (1-vg) * shift_1d(value, offset=1, axis=1)

    else:
      dw = jax.nn.sigmoid(self.dw_proj(inputs[:,1:]))
      dw = dw.reshape(*dw.shape[:-1], -1, self.num_shifts)
      kg, vg = dw[...,:1], dw[...,1:] # B(T-1)N1
      key = key.at[:, 1:].set( key[:,1:] * kg + (1-kg) * key[:,:-1]) 
      value = value.at[:, 1:].set( value[:,1:] * vg + (1-vg) * value[:,:-1])

    if not self.config.kv_shift_skip_knorm:
      key = self.kv_shift_norm(key)

    return query, key, value


class Oshift(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  num_query_heads: int = None
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")

  def setup(self):
    cfg = self.config
    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    self.dw_proj_o = linears.DenseGeneral(
        (self.num_query_heads,),
        kernel_init=initializers.contant_dense_init(0.0),
        kernel_axes=("embed", "heads"),
        use_bias=False,
        name="o_shift_proj",
        **kwargs,
    )

  @nn.compact
  def __call__(self, inputs_q, out):
    og = jax.nn.sigmoid(self.dw_proj_o(inputs_q))[..., jnp.newaxis]
    out = out * og + (1 - og) * shift_1d(out, offset=1, axis=1)
    return out