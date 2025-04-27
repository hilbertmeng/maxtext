from typing import Any, Tuple, Optional

import math
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


def unbind(ary, n, axis=0):
  return [jnp.squeeze(a, axis=axis) for a in jnp.split(ary, n, axis=axis)]

class HeadPool(nn.Module):
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
    self.dw1_norm = jax.nn.identity if not cfg.hp_dw_norm else normalizations.get_rmsnorm(name="dw1_norm", scale_init=None, **norm_kwargs)
    
    self.hp_num_heads = cfg.hp_num_heads
    self.hp_head_gate = cfg.hp_head_gate
    self.activation = jax.nn.identity # jax.nn.gelu

    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    self.head_proj = linears.DenseGeneral(
                                      (self.hp_num_heads, cfg.head_dim),
                                      kernel_init=self.kernel_init,
                                      kernel_axes=('embed', "kv_heads", None),
                                      use_bias=False,
                                      name='head_proj',
                                      **kwargs)

    if self.hp_head_gate:
        self.head_gate_proj = linears.DenseGeneral(
                                      (self.hp_num_heads, cfg.head_dim),
                                      kernel_init=self.kernel_init,
                                      kernel_axes=('embed', "kv_heads", None),
                                      use_bias=False,
                                      name='head_gate_proj',
                                      **kwargs)

    rank = 4
    C = 4 # qkvo
    K = C * (self.hp_num_heads + cfg.num_kv_heads) * rank 
    std = math.sqrt(cfg.base_emb_dim / K / math.sqrt(rank * self.hp_num_heads)) * 0.01
    # std = math.sqrt(cfg.base_emb_dim / K / math.sqrt(rank * self.hp_num_heads)) * 0.001
    dw2_init = self.kernel_init if not cfg.hp_custom_dw2_init else initializers.nd_dense_init_normal(std)
    self.dw1_proj = linears.DenseGeneral(
                                      (K,),
                                      kernel_init=self.kernel_init,
                                      kernel_axes=('embed', None),
                                      use_bias=False,
                                      name='hp_dw1_proj',
                                      **kwargs)
    self.dw2_proj = linears.DenseGeneral(
                                      (C, self.hp_num_heads + cfg.num_kv_heads, rank,),
                                      kernel_init=dw2_init,
                                      kernel_axes=("embed", None, None, None),
                                      use_bias=True,
                                      name='hp_dw2_proj',
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
    headpool = self.head_proj(inputs_q) # BTD, DMd->BTMd

    if self.hp_head_gate:
        headpool = headpool * jax.nn.silu(self.head_gate_proj(inputs_q))

    dw = self.dw2_proj(jax.nn.gelu(self.dw1_proj(inputs_q))) # BTD, DC(M+N)I -> BTC(M+N)I
    qw, kw, vw, ow = unbind(dw, 4, axis=2) # BT(M+N)I

    # share inner; dw1 norm on N dim; I = N; open gate; rm w2 
    q_inner = self.activation(jnp.einsum("BTMD,BTMI->BTID", headpool, self.dw1_norm(qw[:, :, :self.hp_num_heads])))
    q_out = jnp.einsum("BTID,BTNI->BTND", q_inner, qw[:, :, self.hp_num_heads:])
    query = query + q_out

    k_inner = self.activation(jnp.einsum("BTMD,BTMI->BTID", headpool, self.dw1_norm(kw[:, :, :self.hp_num_heads])))
    k_out = jnp.einsum("BTID,BTNI->BTND", k_inner, kw[:, :, self.hp_num_heads:])
    key = key + k_out

    v_inner = self.activation(jnp.einsum("BTMD,BTMI->BTID", headpool, self.dw1_norm(vw[:, :, :self.hp_num_heads])))
    v_out = jnp.einsum("BTID,BTNI->BTND", v_inner, vw[:, :, self.hp_num_heads:])
    value = value + v_out

    o_inner = self.activation(jnp.einsum("BTMD,BTMI->BTID", headpool, self.dw1_norm(ow[:, :, :self.hp_num_heads])))
    o_out = jnp.einsum("BTID,BTNI->BTND", o_inner, ow[:, :, self.hp_num_heads:])

    return query, key, value, o_out

