from typing import Any, Tuple, Optional

import numpy as np
import jax
from flax import linen as nn
import jax.numpy as jnp
from jax.sharding import Mesh

from layers import initializers
from layers import normalizations
from layers import linears
from layers import quantizations 
import max_logging

Quant = quantizations.AqtQuantization
NdInitializer = initializers.NdInitializer
nd_dense_init = initializers.nd_dense_init


class DynamicTemperature(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  
  def setup(self):
    cfg = self.config
    norm_kwargs = {
                "dtype": cfg.dtype,
                "weight_dtype": cfg.weight_dtype,
                "epsilon": cfg.normalization_layer_epsilon,
                "scale_init": None,
                }
    self.dt_prenorm = normalizations.get_rmsnorm(name="dt_prenorm", **norm_kwargs)
    self.dt_postnorm = normalizations.get_rmsnorm(name="dt_postnorm", **norm_kwargs)

    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    hid_dim = 128
    self.dt_up_proj = linears.DenseGeneral(
                                      (hid_dim,),
                                      kernel_init=self.kernel_init,
                                      kernel_axes=('embed', None),
                                      use_bias=False,
                                      name='dt_proj_up',
                                      **kwargs)
    self.dt_down_proj = linears.DenseGeneral(
                                      (1,),
                                      kernel_init=initializers.contant_dense_init(0.0),
                                      kernel_axes=(None, None),
                                      use_bias=True,
                                      name='dt_proj_down',
                                      **kwargs)
    self.dt_tanh = self.config.dynamic_temp_tanh
    if self.dt_tanh: # Transformers without Normalization https://arxiv.org/pdf/2503.10622
      alpha_init, gamma_init = 0.001, 1
      self.alpha = self.param('alpha', nn.with_logical_partitioning(initializers.constant_init(alpha_init), (None,)), (1,), cfg.weight_dtype)
      self.gamma = self.param('gamma', nn.with_logical_partitioning(initializers.constant_init(gamma_init), (None,)), (1,), cfg.weight_dtype)


  @nn.compact
  def __call__(
      self,
      hid, # BTD before last norm
      normed_hid, # BTD after last norm
  ):
    dt = self.dt_down_proj(jax.nn.gelu(self.dt_up_proj(self.dt_prenorm(hid))))
    dt = dt + 1 
    if self.dt_tanh:
      out = normed_hid + jnp.tanh(normed_hid * dt * self.alpha) * self.gamma
    else:
      out = normed_hid + self.dt_postnorm(normed_hid * dt)
    return out