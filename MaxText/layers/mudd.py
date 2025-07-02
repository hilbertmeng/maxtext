from typing import Any, Tuple, Optional

import numpy as np
from flax import linen as nn
import jax.numpy as jnp
from jax.sharding import Mesh
from jax import lax
import jax

from layers import initializers
from layers import normalizations
from layers import linears
import max_logging
from layers import quantizations 
from einops import rearrange
import max_logging

Quant = quantizations.AqtQuantization


def l2norm(x):
  return jnp.sqrt(jnp.sum(jnp.square(x)))


def wsum(w: jnp.ndarray, # CBTL1
         hids: list[jnp.ndarray], # list of BTD
         seq_chunk_size: int = None
         ) -> jnp.ndarray:  # CBTD
  C, B, T, L, _ = w.shape
  D = hids[0].shape[-1]
  out = jnp.zeros((C, B, T, D), dtype=hids[0].dtype)
  seq_chunk_size = seq_chunk_size or T
  assert T % seq_chunk_size == 0
  for chunk_i in range(T // seq_chunk_size):
    sli = slice(chunk_i * seq_chunk_size, (chunk_i + 1) * seq_chunk_size)
    for l in range(L): # 每层
      out = out.at[:, :, sli, :].set(out[:, :, sli, :] + w[:, :, sli, l, :] * hids[l][:, sli, :])  # CBt1*BtD->CBtD)
  return out


def wsum_jit(w: jnp.ndarray, # CBTL1
         hids: list[jnp.ndarray], # list of BTD
         seq_chunk_size: int = None
         ) -> jnp.ndarray:  # CBTD
  
  @jax.jit
  def dot(w, h):
    return w * h

  C, B, T, L, _ = w.shape
  D = hids[0].shape[-1]
  carry = jnp.zeros((C, B, T, D), dtype=hids[0].dtype)
  seq_chunk_size = seq_chunk_size or T
  assert T % seq_chunk_size == 0
  for chunk_i in range(T // seq_chunk_size):
    for l in range(L):
      start = chunk_i * seq_chunk_size
      end = (chunk_i + 1) * seq_chunk_size
      _h = hids[l][:, start: end, :]
      _w = w[:, :, start: end, l, :]
      chunk_out = dot(_w, _h)
      carry = lax.dynamic_update_slice(carry, chunk_out, (0, 0, start, 0))
  return carry


def wsum_fori(w: jnp.ndarray, hids: list[jnp.ndarray]) -> jnp.ndarray:
    # w: CBTL1, hids: list of BTD
    C, B, T, L, _ = w.shape
    D = hids[0].shape[-1]
    hids_stacked = jnp.stack(hids, axis=0)  # → LBTD
    out_init = jnp.zeros((C, B, T, D), dtype=w.dtype)
    def body_fn(l, out):
        h = hids_stacked[l]  # BTD
        return out + w[:, :, :, l, :] * h
    out = lax.fori_loop(0, L, body_fn, out_init)
    del hids_stacked
    return out


def einsum(w: jnp.ndarray, # CBTL1
         hids: list[jnp.ndarray], # list of BTD
         ) -> jnp.ndarray:  # CBTD
  
  return jnp.einsum('btcl,lbtd->cbtd', w, hids)


class Norm(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    assert isinstance(inputs, (tuple, list, jnp.ndarray)) and len(inputs) == 3
    name = 'pre_self_attention_layer_norm'
    lnx_q, lnx_k, lnx_v = [nn.with_logical_constraint(
                          normalizations.get_rmsnorm(f'{name}_{suffix}', cfg)(inp), 
                          ("activation_batch", "activation_norm_length", "activation_embed")
                          )
                          for inp, suffix in zip(inputs, 'qkv')]
    return lnx_q, lnx_k, lnx_v


class Mlp(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  layer_inx: int = None
  use_bias: bool = True

  def setup(self):
    cfg = self.config
    if not cfg.dense_conn: return
    norm_kwargs = {
                "dtype": cfg.dtype,
                "weight_dtype": cfg.weight_dtype,
                "name": "pre_dense_proj1_norm",
                "epsilon": cfg.normalization_layer_epsilon,
                }
    if not getattr(cfg, 'mudd_prenorm', False):
        max_logging.log(f'mudd_prenorm is False', debug=self.config.debug)
        norm_kwargs['scale_init'] = None # it means use scale is false
    self.pre_dense_proj1_norm = normalizations.get_rmsnorm(**norm_kwargs)
    
    factor = 1
    layer_inx = self.layer_inx
    C = 1 if cfg.dynamic_dense_fix_last_layer and layer_inx == cfg.num_decoder_layers - 1 else len(cfg.dynamic_dense_type)
    dw_shape = (C, ((layer_inx + 1) * factor + 1))

    dynamic_dense_hidden_expand = len(cfg.dynamic_dense_type) if layer_inx == cfg.num_decoder_layers - 1 else 1
    max_logging.log(f'dynamic_dense_hidden_expand-{layer_inx}: {dynamic_dense_hidden_expand}', debug=self.config.debug)
    dynamic_dense_inter_dim = int(np.prod(dw_shape) * dynamic_dense_hidden_expand)

    if cfg.dynamic_dense_hidden_round:  # default: round to 64 or 128
      dynamic_dense_inter_dim = (dynamic_dense_inter_dim// 64 + 1) * 64
    self.dynamic_dense_inter_dim = dynamic_dense_inter_dim

    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    self.dense_proj1 = linears.DenseGeneral(
                                    dynamic_dense_inter_dim,
                                    kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
                                    kernel_axes=('embed', 'kv'),
                                    use_bias=False,
                                    name='dynamic_dense_conn1',
                                    **kwargs)
    self.dense_activation = linears._convert_to_activation_function(cfg.dynamic_dense_act_cls)
    
    self.dense_proj2 = linears.DenseGeneral(dw_shape, 
                                    kernel_init=initializers.contant_dense_init(0.0), 
                                    kernel_axes=('kv', None), 
                                    use_bias=False, 
                                    name='dynamic_dense_conn2', 
                                    **kwargs)
    if self.use_bias:
      self.dense2_bias_init_value = 0.0 if cfg.mudd_prenorm and cfg.mudd_postnorm else 1.0
      init_v = jnp.array([0] * ((layer_inx + 1) * factor) + [self.dense2_bias_init_value]).astype(cfg.weight_dtype)
      init_v = init_v[None].repeat(C, 0)
      self.dense_proj2_bias = self.param(f"dense_proj2.bias", init_fn=lambda rng: init_v)

  @nn.compact
  def __call__(
      self,
      layer_output,
  ):
    cfg = self.config
    mesh = self.mesh
    dyn_dense_w = None
    if cfg.dynamic_dense_type == 'qkvm' and cfg.dense_conn:
      x_out_normed = self.pre_dense_proj1_norm(layer_output)
      dense_w_inner = self.dense_activation(self.dense_proj1(x_out_normed))
      dyn_dense_kernel_out = self.dense_proj2(dense_w_inner)
      if cfg.dynamic_dense_scale_dw:
        max_logging.log(f'dynamic_dense_scale_dw: {cfg.dynamic_dense_scale_dw}', debug=self.config.debug)
        dyn_dense_kernel_out /= jnp.sqrt(self.dynamic_dense_inter_dim)
      if self.use_bias:
        dyn_dense_w = dyn_dense_kernel_out + self.dense_proj2_bias.astype(dyn_dense_kernel_out.dtype)
      else:
        dyn_dense_w = dyn_dense_kernel_out
    return dyn_dense_w


class Compose(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  layer_inx: int = None
  
  def setup(self):
    if self.config.mudd_in_layer:
        self.mudd_mlp = Mlp(self.config, self.mesh, self.quant, self.layer_inx)
          
  @nn.compact
  def __call__(
      self,
      layer_output,
      hids,
  ):

    if self.config.mudd_in_layer:
        y, dyn_dense_w = layer_output, self.mudd_mlp(layer_output)
    else:
        y, dyn_dense_w = layer_output
        if dyn_dense_w is None: 
          max_logging.log(f'Compose dyn_dense_w is None', debug=self.config.debug)
          return y, hids

    max_logging.log(f'Compose history hidden states.', debug=self.config.debug)
    layer_inx = self.layer_inx
    cfg = self.config

    if self.config.record_internal_nn_metrics:
        _dyn_dense_w = dyn_dense_w.astype(jnp.float32)
        self.sow('intermediates', f'dyn_dense_w/max/layer_{layer_inx}', jnp.max(_dyn_dense_w))
        self.sow('intermediates', f'dyn_dense_w/mean/layer_{layer_inx}', jnp.mean(_dyn_dense_w))
        self.sow('intermediates', f'dyn_dense_w/min/layer_{layer_inx}', jnp.min(_dyn_dense_w))
        self.sow('intermediates', f'dyn_dense_w/norm/layer_{layer_inx}', l2norm(_dyn_dense_w))
        self.sow('intermediates', f'dyn_dense_w/std/layer_{layer_inx}', jnp.std(_dyn_dense_w))
        self.sow('intermediates', f'layer_output/norm/layer_{layer_inx}', l2norm(y.astype(jnp.float32)))
        del _dyn_dense_w

    y_normed = normalizations.get_rmsnorm(name=f"mudd_prenorm_{layer_inx}", cfg=cfg)(y) if cfg.mudd_prenorm else y
    # hids = hids.at[self.layer_inx].set(y_normed)
    hids.append(y_normed)
    C = 1 if cfg.dynamic_dense_fix_last_layer and layer_inx == cfg.num_decoder_layers - 1 else len(cfg.dynamic_dense_type)
    max_logging.log(f'Compose dyn_dense_w: {dyn_dense_w.shape} layer_inx: {layer_inx}', debug=self.config.debug)

    def wsum_scan(y, w, hids, seq_chunk_size):
      # hids: LBTD
      B, T, C, L = w.shape
      D = hids[0].shape[-1]
      seq_chunk_size = seq_chunk_size or T
      assert T % seq_chunk_size == 0
      num_chunks = T // seq_chunk_size
      def chunk_step(carry, chunk_i):
          start = chunk_i * seq_chunk_size
          w_chunk = lax.dynamic_slice_in_dim(w, start, seq_chunk_size, axis=1) # btcl
          h_chunk = lax.dynamic_slice(hids, (0, 0, start, 0), (L, B, seq_chunk_size, D)) # lbtd
          chunk_out = jnp.einsum('btcl,lbtd->cbtd', w_chunk, h_chunk)
          carry = lax.dynamic_update_slice(carry, chunk_out, (0, 0, start, 0))
          return carry, None
      out_init = jnp.zeros((C, B, T, D), dtype=w.dtype)
      out_final, _ = lax.scan(chunk_step, out_init, jnp.arange(num_chunks))
      if cfg.mudd_postnorm:
        y = tuple([y + (post_norm(out_final[cidx]) if cidx == C - 1 else out_final[cidx]) for cidx in range(C)])
      else:
        y = out_final
      return y

    max_logging.log(f'ddw_gen_pattern: {cfg.ddw_gen_pattern} mudd_postnorm is {cfg.mudd_postnorm}....', debug=self.config.debug)
    if cfg.mudd_postnorm:
      post_norm = normalizations.get_rmsnorm(name=f"mudd_postnorm_{layer_inx}", cfg=cfg, scale_init=nn.initializers.constant(0.001))
      if cfg.mudd_compose_method == 'scan':
        dyn_dense_w = rearrange(dyn_dense_w, 'B T C L -> C B T L 1', C=C)
        y = wsum_scan(y, dyn_dense_w, hids, cfg.ddw_gen_chunk_size)
      elif cfg.mudd_compose_method == 'fori':
        dyn_dense_w = rearrange(dyn_dense_w, 'B T C L -> C B T L 1', C=C)
        y = tuple([y + (post_norm(
            wsum_fori(dyn_dense_w[cidx: cidx + 1], hids).squeeze(0)
                                  ) if cidx == C - 1 else 
            wsum_fori(dyn_dense_w[cidx: cidx + 1], hids).squeeze(0)
                        ) for cidx in range(C)])
      elif cfg.mudd_compose_method == 'einsum':
        assert isinstance(hids, jnp.Array)
        y = tuple([y + post_norm(r) if i == C - 1 else y + r for i, r in enumerate(einsum(dyn_dense_w, hids[:dyn_dense_w.shape[-1]]))])
      elif cfg.mudd_compose_method == 'jit':
        dyn_dense_w = rearrange(dyn_dense_w, 'B T C L -> C B T L 1', C=C)
        y = tuple([y + post_norm(r) if i == C - 1 else y + r for i, r in enumerate(wsum_jit(dyn_dense_w, hids))])
      else:
        dyn_dense_w = rearrange(dyn_dense_w, 'B T C L -> C B T L 1', C=C)
        y = tuple([y + (post_norm(
            wsum(dyn_dense_w[cidx: cidx + 1], hids, cfg.ddw_gen_chunk_size).squeeze(0)
                                  ) if cidx == C - 1 else 
            wsum(dyn_dense_w[cidx: cidx + 1], hids, cfg.ddw_gen_chunk_size).squeeze(0)
                        ) for cidx in range(C)])
    else:
      if cfg.mudd_compose_scan:
        y = wsum_scan(y, dyn_dense_w, hids, cfg.ddw_gen_chunk_size)
      else:
        # (btl, btl, btl, btl)
        y = tuple([wsum_scan(dyn_dense_w[cidx: cidx + 1], hids, cfg.ddw_gen_chunk_size).squeeze(0) for cidx in range(C)])
    if layer_inx == cfg.num_decoder_layers - 1:
      del hids
      return y[0], []
    return y, hids