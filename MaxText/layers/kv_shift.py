from typing import Any, Tuple, Optional

import numpy as np
import jax
from flax import linen as nn
from jax import lax
import jax.numpy as jnp
from jax.sharding import Mesh

from layers import initializers
from layers import normalizations
from layers import linears
import max_logging
from layers import quantizations 
from einops import rearrange
import max_logging

Quant = quantizations.AqtQuantization

def shift_1d(inputs, offset: int, axis: int):
  """Shifts the input tensor by offset in the dimension axis.

  To shift right the offset is positive and the input is padded at the
  beginning, while to shift left the offset is negative and the input is
  padded at the end.

  Args:
    inputs: The input tensor to shift.
    offset: The number of positions to shift. If the offset is positive, pad at
      the beginning of the sequence, if the offset is negative, then pad at the
      end of the sequence.
    axis: The dimension in which to shift the input.

  Returns:
    The shifted input.
  """
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
  
  def setup(self):
    cfg = self.config
    norm_kwargs = {
                "dtype": cfg.dtype,
                "weight_dtype": cfg.weight_dtype,
                "name": "kv_shift_knorm",
                "epsilon": cfg.normalization_layer_epsilon,
                }
    self.kv_shift_norm = normalizations.get_rmsnorm(**norm_kwargs)

    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    self.dw_proj = linears.DenseGeneral(
                                    (cfg.num_kv_heads, 2),
                                    kernel_init=initializers.contant_dense_init(0.0),
                                    kernel_axes=('embed', "kv_heads", None),
                                    use_bias=False,
                                    name='kv_shift_proj',
                                    **kwargs)
    
  @nn.compact
  def __call__(
      self,
      inputs, # BTD
      key, # BTND
      value, # BTND 
  ):
    if self.config.kv_shift_flash:
      dw = jax.nn.sigmoid(self.dw_proj(inputs)) # B(T-1)D, DN2->B(T-1)N2
    else:
      dw = jax.nn.sigmoid(self.dw_proj(inputs[:,1:])) # B(T-1)D, DN2->B(T-1)N2
    kg, vg = dw[...,:1], dw[...,1:] # B(T-1)N1

    # kv shift
    if self.config.kv_shift_flash:
      key = key * kg + (1-kg) * shift_1d(key, offset=1, axis=1)
      value = value * vg + (1-vg) * shift_1d(value, offset=1, axis=1)
    else:
      key = key.at[:, 1:].set( key[:,1:] * kg + (1-kg) * key[:,:-1] ) 
      value = value.at[:, 1:].set( value[:,1:] * vg + (1-vg) * value[:,:-1] )

    # post_norm on key only 
    key = self.kv_shift_norm(key)

    return key, value    