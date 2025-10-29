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
NdInitializer = initializers.NdInitializer
nd_dense_init = initializers.nd_dense_init

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

class StaticShiftPerChannel(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")

  def setup(self):
    cfg = self.config
    self.mu =  self.param('w1_bias',nn.with_logical_partitioning(initializers.constant_init(0.0), (None, None, None, 'kv_heads')), bias_shape, self.weight_dtype)

class KVshift(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  num_kv_heads: int | None = None
  
  def setup(self):
    cfg = self.config
    norm_kwargs = {
                "dtype": cfg.dtype,
                "weight_dtype": cfg.weight_dtype,
                "epsilon": cfg.normalization_layer_epsilon,
                }
    if not self.config.kv_shift_skip_knorm:
      self.kv_shift_norm = normalizations.get_rmsnorm(name="kv_shift_knorm", **norm_kwargs)
    self.kv_shift_prenorm = normalizations.get_rmsnorm(name="kv_shift_prenorm", **norm_kwargs)
    
    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    self.kv_shift_mlp = self.config.kv_shift_mlp
    self.kv_shift_hidden_way = self.config.kv_shift_hidden_way
    self.q_shift = self.config.use_q_shift
    num_shifts = 2 if not self.q_shift else 3 
    num_kv_heads = self.num_kv_heads if self.num_kv_heads is not None else cfg.num_kv_heads
    self.kv_shift_per_channel = self.config.kv_shift_per_channel
    if self.kv_shift_per_channel:
      assert self.kv_shift_mlp
      self.mu_k = self.param('mu_k',nn.with_logical_partitioning(initializers.constant_init(0.5), ('embed',)), (self.config.emb_dim,), cfg.weight_dtype)
      self.mu_v = self.param('mu_v',nn.with_logical_partitioning(initializers.constant_init(0.5), ('embed',)), (self.config.emb_dim,), cfg.weight_dtype)
    if self.kv_shift_mlp:
      self.kv_shift_lora_act = {'tanh': jax.nn.tanh, 'gelu': jax.nn.gelu, 'identity': jax.nn.identity}[self.config.kv_shift_lora_act]
    if self.kv_shift_mlp:
      if self.kv_shift_hidden_way in ['kv', 'qkv']:
        for mode in self.kv_shift_hidden_way:
          setattr(self, f'dw_up_proj_{mode}', linears.DenseGeneral(
                                      (num_kv_heads if not self.kv_shift_per_channel else self.config.kv_shift_lora_dim),
                                      kernel_init=self.kernel_init,
                                      # kernel_axes=('embed', "kv_heads"),  # by mqy
                                      kernel_axes=('embed', None), # fix by xd
                                      use_bias=False,
                                      name=f'kv_shift_proj_up_{mode}',
                                      **kwargs))
          setattr(self, f'dw_down_proj_{mode}', linears.DenseGeneral(
                                      (num_kv_heads, cfg.head_dim if self.kv_shift_per_channel else 1),
                                      kernel_init=initializers.contant_dense_init(0.0),
                                      # kernel_axes=('embed', None),  # by mqy
                                      kernel_axes=(None, None, None), # TODO xd: how to shard?
                                      use_bias=True,
                                      name=f'kv_shift_proj_down_{mode}',
                                      **kwargs))
      else:
        self.dw_up_proj = linears.DenseGeneral(
                                      (num_kv_heads * num_shifts),
                                      kernel_init=self.kernel_init,
                                      kernel_axes=('embed', "kv_heads"),
                                      use_bias=False,
                                      name='kv_shift_proj_up',
                                      **kwargs)
        self.dw_down_proj = linears.DenseGeneral(
                                      (num_kv_heads, num_shifts),
                                      kernel_init=initializers.contant_dense_init(0.0),
                                      kernel_axes=('embed', "kv_heads", None),
                                      use_bias=True,
                                      name='kv_shift_proj_down',
                                      **kwargs)
    else:
      if self.kv_shift_hidden_way in ['kv', 'qkv']:
        for mode in self.kv_shift_hidden_way:
          num_heads = num_kv_heads if mode in 'kv' else self.config.num_query_heads
          setattr(self, f'dw_proj_{mode}', linears.DenseGeneral(
                                      (num_heads, 1),
                                      kernel_init=initializers.contant_dense_init(0.0),
                                      kernel_axes=('embed', "kv_heads", None),
                                      use_bias=False,
                                      name=f'kv_shift_proj_{mode}',
                                      **kwargs))
      else:
        self.dw_proj = linears.DenseGeneral(
                                      (num_kv_heads, num_shifts),
                                      kernel_init=initializers.contant_dense_init(0.0),
                                      kernel_axes=('embed', "kv_heads", None),
                                      use_bias=False,
                                      name='kv_shift_proj',
                                      **kwargs)
      
  @nn.compact
  def __call__(
      self,
      inputs_q, # inputs_q BTD
      query, # BTND
      key, # BTND
      value, # BTND 
      inputs_k=None, # BTD
      inputs_v=None, # BTD
      inputs_m=None, # BTD
  ):
    assert self.config.kv_shift_flash
    inputs = inputs_q

    if self.kv_shift_hidden_way == 'm':
      inputs = self.kv_shift_prenorm(inputs_m)

    if self.config.kv_shift_per_channel:
      inputs_k = inputs_k * self.mu_k + (1 - self.mu_k) * shift_1d(inputs_k, offset=1, axis=1)
      inputs_v = inputs_v * self.mu_v + (1 - self.mu_v) * shift_1d(inputs_v, offset=1, axis=1)

    if self.config.kv_shift_flash:
      if self.kv_shift_hidden_way == 'kv':
        if self.kv_shift_mlp: # best branch
          kg = jax.nn.sigmoid(self.dw_down_proj_k(self.kv_shift_lora_act(self.dw_up_proj_k(inputs_k))))
          vg = jax.nn.sigmoid(self.dw_down_proj_v(self.kv_shift_lora_act(self.dw_up_proj_v(inputs_v))))
        else:
          kg = jax.nn.sigmoid(self.dw_proj_k(inputs_k))
          vg = jax.nn.sigmoid(self.dw_proj_v(inputs_v))
      elif self.kv_shift_hidden_way == 'qkv':
        if self.kv_shift_mlp:
          qg = jax.nn.sigmoid(self.dw_down_proj_q(jax.nn.gelu(self.dw_up_proj_q(inputs_q))))
          kg = jax.nn.sigmoid(self.dw_down_proj_k(jax.nn.gelu(self.dw_up_proj_k(inputs_k))))
          vg = jax.nn.sigmoid(self.dw_down_proj_v(jax.nn.gelu(self.dw_up_proj_v(inputs_v))))
        else:
          qg = jax.nn.sigmoid(self.dw_proj_q(inputs_q))
          kg = jax.nn.sigmoid(self.dw_proj_k(inputs_k))
          vg = jax.nn.sigmoid(self.dw_proj_v(inputs_v))
      else:
        if self.kv_shift_mlp:
          dw = jax.nn.sigmoid(self.dw_down_proj(jax.nn.gelu(self.dw_up_proj(inputs)))) # B(T-1)D, DN2->B(T-1)N2
        else:
          dw = jax.nn.sigmoid(self.dw_proj(inputs)) # B(T-1)D, DN2->B(T-1)N2
        if self.q_shift:
          kg, vg, qg = dw[...,:1], dw[...,1:2], dw[...,2:] 
        else: 
          kg, vg = dw[...,:1], dw[...,1:] # B(T-1)N1
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

    if self.q_shift:
      query = query * qg + (1-qg) * shift_1d(query, offset=1, axis=1)

    # post_norm on key only 
    if not self.config.kv_shift_skip_knorm:
      key = self.kv_shift_norm(key)

    return query, key, value, inputs_k, inputs_v    

class OShift(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  num_heads: int | None = None
  offset: int | None = None
  
  def setup(self):
    cfg = self.config
    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    if self.offset is None: self.offset = 1
    setattr(self, f'dw_proj_o', linears.DenseGeneral(
                                      (self.num_heads, self.offset),
                                      kernel_init=initializers.contant_dense_init(0.0),
                                      kernel_axes=('embed', "heads", None),
                                      use_bias=False,
                                      name=f'o_shift_proj',
                                      **kwargs))
  @nn.compact
  def __call__(self,
      out, # BTNd
      inputs_m=None, # BTD
  ):
    og = jax.nn.sigmoid(self.dw_proj_o(inputs_m))  # BTD,DNk->BTNk
    if self.offset == 1:
      out = out * og + (1-og) * shift_1d(out, offset=1, axis=1)  # BTNd, BTN1->BTNd
    else:
      out = sum(out * og[..., i:i+1] + (1-og[..., i:i+1]) * shift_1d(out, offset=i+1, axis=1) 
               for i in range(self.offset))
    return out

class Hiddenshift(nn.Module):
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
                }
    self.hidden_shift_norm = normalizations.get_rmsnorm(name="hidden_shift_knorm", **norm_kwargs)
    
    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant) 
    hid_dim = 128
    self.dw_up_proj = linears.DenseGeneral(
                                  (hid_dim,),
                                  kernel_init=self.kernel_init,
                                  kernel_axes=('embed', "kv_heads"),
                                  use_bias=False,
                                  name='hidden_shift_proj_up',
                                  **kwargs)
    self.dw_down_proj = linears.DenseGeneral(
                                  (1,),
                                  kernel_init=initializers.contant_dense_init(0.0),
                                  kernel_axes=(None, None),
                                  use_bias=True,
                                  name='hidden_shift_proj_down',
                                  **kwargs)
  @nn.compact
  def __call__(
      self,
      hid, # hidden states BTD
  ):
    # generate gates
    hid_normed = self.hidden_shift_norm(hid)
    hg = jax.nn.sigmoid(self.dw_down_proj(jax.nn.gelu(self.dw_up_proj(hid_normed))))

    # hidden states shift
    hid = hid * hg + (1-hg) * shift_1d(hid, offset=1, axis=1)
    return hid           
  

class ValueResidual(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  
  def setup(self):
    cfg = self.config
    # norm_kwargs = {
    #             "dtype": cfg.dtype,
    #             "weight_dtype": cfg.weight_dtype,
    #             "epsilon": cfg.normalization_layer_epsilon,
    #             }
    # self.hidden_shift_norm = normalizations.get_rmsnorm(name="hidden_shift_knorm", **norm_kwargs)
    
    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant) 
    self.dw_proj = linears.DenseGeneral(
                                  (cfg.num_kv_heads,), # DN
                                  kernel_init=initializers.contant_dense_init(0.0),
                                  kernel_axes=('embed', "kv_heads"),
                                  use_bias=True,
                                  name='value_residual_proj',
                                  **kwargs)

  @nn.compact
  def __call__(
      self,
      inputs_v,
      value, # hidden states BTD
      value_residual,
      inputs_m=None,
  ):
    vg = jax.nn.sigmoid(self.dw_proj(inputs_v))[...,None] # BTD, DN->BTN1      BTD, DD -> BTD -> BTNd
    value = (1-vg) * value + vg * value_residual # BTNd
    return value 


class KVshiftVR(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  
  def setup(self):
    self.kv_shift = KVshift(config=self.config,mesh=self.mesh, quant=self.quant, kernel_init=self.kernel_init)
    self.value_residual = ValueResidual(config=self.config,mesh=self.mesh, quant=self.quant, kernel_init=self.kernel_init)

  @nn.compact
  def __call__(
      self,
      key, # BTND
      value, # BTND 
      value_residual,
      inputs_k=None, # BTD
      inputs_v=None, # BTD
      inputs_m=None, # BTD
  ):
    assert self.config.kv_shift_flash and self.config.kv_shift_hidden_way == 'kv' and not self.config.kv_shift_mlp
    assert self.config.kv_shift_skip_knorm

    kg = jax.nn.sigmoid(self.kv_shift.dw_proj_k(inputs_k))
    vg = jax.nn.sigmoid(self.kv_shift.dw_proj_v(inputs_v))

    vrg = jax.nn.sigmoid(self.value_residual.dw_proj(inputs_v))[...,None]

    key = key * kg + (1-kg) * shift_1d(key, offset=1, axis=1)
    value = value * (1 - vg - vrg) + vg/2 * shift_1d(value, offset=1, axis=1) + value_residual * vrg/2

    return key, value    