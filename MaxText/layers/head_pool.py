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
NormalInitializer = initializers.nd_dense_init_normal


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
    
    if self.config.hp_norm:
      self.hp_norm = normalizations.get_rmsnorm(name="hp_norm", **norm_kwargs)

    self.hp_num_heads = cfg.hp_num_heads
    self.hp_head_gate = cfg.hp_head_gate
    self.activation = jax.nn.identity # jax.nn.gelu
    self.hp_ways = "qkvo" if self.config.hp_ways is None else self.config.hp_ways
    self.hp_static = False if self.config.hp_static is None else self.config.hp_static
    self.hp_dynamic = True if self.config.hp_dynamic is None else self.config.hp_dynamic
    self.C = len(self.hp_ways) # qkvo
    rank = 4 if self.config.hp_rank is None else self.config.hp_rank 
    K = self.C * (self.hp_num_heads + cfg.num_kv_heads) * rank 

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


    if self.config.hp_static:
        if self.config.hp_no_lora:
            kernel_init_shard = nn.with_logical_partitioning(NormalInitializer(0.006), (None, None, None,))
            self.sw = self.param('sw', kernel_init_shard, (self.C, cfg.hp_num_heads, cfg.num_kv_heads), cfg.weight_dtype) # CMN
        else:
            kernel_init_shard = nn.with_logical_partitioning(NormalInitializer(0.006), (None, None,))
            self.sw1 = self.param('sw1', kernel_init_shard, (self.hp_num_heads, cfg.num_kv_heads), cfg.weight_dtype) # MN
            kernel_init_shard = nn.with_logical_partitioning(NormalInitializer(0.006), (None, None, None,))
            self.sw2 = self.param('sw2', kernel_init_shard, (self.C, cfg.num_kv_heads, cfg.num_kv_heads), cfg.weight_dtype) # CNN
        if self.config.hp_use_sw_scale:
            kernel_init_shard = nn.with_logical_partitioning(initializers.constant_init(1), (None, 'kv_heads', None,))
            self.sw_scale = self.param('sw_scale', kernel_init_shard, (self.C, cfg.num_kv_heads, cfg.head_dim), cfg.weight_dtype) # CNd

    if self.config.hp_o_transform:
        kernel_init_shard = nn.with_logical_partitioning(NormalInitializer(0.006), (None, None,))
        self.sw_o = self.param('sw_o', kernel_init_shard, (cfg.num_kv_heads, self.hp_num_heads), cfg.weight_dtype) # NM

    if self.config.hp_dynamic:
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
                                        (self.C, self.hp_num_heads + cfg.num_kv_heads, rank,),
                                        kernel_init=dw2_init,
                                        kernel_axes=("embed", None, None, None),
                                        use_bias=True,
                                        name='hp_dw2_proj',
                                        **kwargs)
    if self.config.hp_dynamic_mixed_v:
        self.mixed_v_proj = linears.DenseGeneral(
                                        (self.hp_num_heads,),
                                        kernel_init=self.kernel_init,
                                        kernel_axes=(None, None),
                                        use_bias=False,
                                        name='hp_mixedv_proj',
                                        **kwargs)
    
    if self.config.hp_out_proj:
        self.out_proj = linears.DenseGeneral(
        features=self.config.base_emb_dim,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=("heads", "kv", "embed"),
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="hp_out",
        quant=self.quant,
        use_bias=self.config.use_bias,
        matmul_precision=self.config.matmul_precision,
        )
      
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
      ffn_act=None, # BTI 
  ):
    hidden_state = inputs_q if not self.config.hp_norm else self.hp_norm(inputs_m)

    o_out = None
    ow = None

    if self.config.hp_from_ffn: 
        B, T, I = ffn_act.shape
        headpool = ffn_act.reshape(B, T, self.hp_num_heads, -1)
    else:
        headpool = self.head_proj(hidden_state) # BTD, DMd->BTMd

        if self.hp_head_gate:
            headpool = headpool * jax.nn.silu(self.head_gate_proj(hidden_state))

    if self.hp_static:
        if self.config.hp_no_lora:
            out = jnp.einsum("BTMD, CMN->CBTND", headpool, self.sw)
        else:
            shared_inner = jnp.einsum("BTMD, MN->BTND", headpool, self.sw1)
            out = jnp.einsum("BTMD, CMN->CBTND", shared_inner, self.sw2)
        if self.config.hp_use_sw_scale: 
            out = out * self.sw_scale[:,None,None] # CBTND, C11ND
        if self.hp_ways =="qkv":
            q_out, k_out, v_out = unbind(out, self.C, axis=0)
            query = query + q_out
            key = key + k_out
            value = value + v_out

    if self.hp_dynamic:
        dw = self.dw2_proj(jax.nn.gelu(self.dw1_proj(inputs_q))) # BTD, DC(M+N)I -> BTC(M+N)I
        if self.hp_ways == "qkv":
          qw, kw, vw = unbind(dw, self.C, axis=2) # BT(M+N)I
        elif self.hp_ways == "qkvo":
          qw, kw, vw, ow = unbind(dw, self.C, axis=2) # BT(M+N)I
          if self.config.hp_dynamic_mixed_v:
            ow = jnp.einsum("BTMI,BTNI->BTNM", self.dw1_norm(ow[:, :, :self.hp_num_heads], axis=-2), ow[:, :, self.hp_num_heads:])

        if self.config.hp_share_inner:
          shared_inner = jnp.einsum("BTMD,BTMI->BTID", headpool, self.dw1_norm(qw[:, :, :self.hp_num_heads], axis=-2))
          q_out = jnp.einsum("BTID,BTNI->BTND", shared_inner, qw[:, :, self.hp_num_heads:])
          query = query + q_out
          k_out = jnp.einsum("BTID,BTNI->BTND", shared_inner, kw[:, :, self.hp_num_heads:])
          key = key + k_out
          v_out = jnp.einsum("BTID,BTNI->BTND", shared_inner, vw[:, :, self.hp_num_heads:])
          value = value + v_out
        else:
          # share inner; dw1 norm on N dim; I = N; open gate; rm w2 
          q_inner = self.activation(jnp.einsum("BTMD,BTMI->BTID", headpool, self.dw1_norm(qw[:, :, :self.hp_num_heads], axis=-2)))
          q_out = jnp.einsum("BTID,BTNI->BTND", q_inner, qw[:, :, self.hp_num_heads:])
          query = query + q_out

          k_inner = self.activation(jnp.einsum("BTMD,BTMI->BTID", headpool, self.dw1_norm(kw[:, :, :self.hp_num_heads], axis=-2)))
          k_out = jnp.einsum("BTID,BTNI->BTND", k_inner, kw[:, :, self.hp_num_heads:])
          key = key + k_out

          v_inner = self.activation(jnp.einsum("BTMD,BTMI->BTID", headpool, self.dw1_norm(vw[:, :, :self.hp_num_heads], axis=-2)))
          v_out = jnp.einsum("BTID,BTNI->BTND", v_inner, vw[:, :, self.hp_num_heads:])
          value = value + v_out

          o_inner = self.activation(jnp.einsum("BTMD,BTMI->BTID", headpool, self.dw1_norm(ow[:, :, :self.hp_num_heads], axis=-2)))
          o_out = jnp.einsum("BTID,BTNI->BTND", o_inner, ow[:, :, self.hp_num_heads:])
    
    if self.config.hp_o_shortcut:
       o_out = headpool

    return query, key, value, o_out, ow

