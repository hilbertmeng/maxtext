from typing import Any, Optional

import numpy as np
from flax import linen as nn
import jax.numpy as jnp
from jax.sharding import Mesh
from einops import rearrange

from layers import initializers
from layers import normalizations
from layers import linears
from layers import quantizations
from layers import embeddings
import max_logging
import jax

# Type alias for quantization
Quant = quantizations.AqtQuantization

# Constants
def l2norm(x: jnp.ndarray) -> jnp.ndarray:
  """Compute L2 norm of input tensor."""
  return jnp.sqrt(jnp.sum(jnp.square(x)))


def wsum2_qy(w: jnp.ndarray, # CBTL1
         hids: list[jnp.ndarray], # list of BTD
         mask: Optional[list[bool]] = None,
         ) -> jnp.ndarray:  # CBTD
  C, B, T, L, _ = w.shape
  D = hids[0].shape[-1]
  out = jnp.zeros((C, B, T, D), dtype=hids[0].dtype)
  idx = 0 
  if mask is not None:
    assert len(mask) == len(hids)
  for hidx in range(len(hids)): # 每层
    if mask is not None and not mask[hidx]:
      continue
    out += w[..., idx, :] * hids[hidx]
    idx += 1
  return out


def wsum_for(w: jnp.ndarray, # CBTL1
         hids: list[jnp.ndarray], # list of BTD
         mask: int = None
         ) -> jnp.ndarray:  # CBTD
  C, B, T, L, _ = w.shape
  D = hids[0].shape[-1]
  out = jnp.zeros((C, B, T, D), dtype=hids[0].dtype)
  for l in range(L): # 每层
    out += w[..., l, :] * hids[l]
  return out


def wsum_muddde(w: jnp.ndarray, # CBTL1
         hids: list[jnp.ndarray], # list of BTD
         seq_chunk_size: int = None
         ) -> jnp.ndarray:  # CBTD
  L = w.shape[-2]
  de_layer_num = hids[0].shape[-1]
  out = jnp.einsum('cbtl,btdl->cbtd', w[..., :de_layer_num, 0], hids[0])
  for l in range(de_layer_num, L): # 每层
    out += w[..., l, :] * hids[l - de_layer_num + 1]
  return out


class Norm(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(self, inputs):
    cfg = self.config
    assert isinstance(inputs, (tuple, list, jnp.ndarray)) and len(inputs) == 3
    name = 'pre_self_attention_layer_norm'
    lnx_q, lnx_k, lnx_v = [normalizations.get_rmsnorm(f'{name}_{suffix}', cfg)(inp) for inp, suffix in zip(inputs, 'qkv')]
    return lnx_q, lnx_k, lnx_v


class Mlp(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  hids_length: int = None
  use_bias: bool = True
  C: int = 4

  def setup(self):
    cfg = self.config
    if not cfg.dense_conn: return

    if not getattr(cfg, 'mudd_prenorm', False):
        self.pre_dense_proj1_norm = normalizations.get_rmsnorm("pre_dense_proj1_norm", cfg, scale_init=None)
    else:
      self.pre_dense_proj1_norm = normalizations.get_rmsnorm("pre_dense_proj1_norm", cfg)
    
    hids_length = self.hids_length
    num_extra_emb = cfg.mudd_num_extra_emb + 1 if cfg.mudd_num_extra_emb is not None else 0
    num_extra_emb = num_extra_emb + cfg.mudd_num_extra_emb if cfg.mudd_cat_prefix_emb else num_extra_emb
    is_last_layer = hids_length == num_extra_emb + cfg.num_decoder_layers - 1 + cfg.mtp_num_layers

    C = self.C
    compose_length = hids_length - (cfg.mudd_num_extra_emb + 1)//cfg.mudd_emb_dilation * (cfg.mudd_emb_dilation -1) if cfg.mudd_emb_dilation is not None and not is_last_layer else hids_length
    dw_shape = (C, compose_length) # lsp
    self.dw_shape = dw_shape
    # lsp
    
    dynamic_dense_hidden_expand = len(cfg.dynamic_dense_type) if is_last_layer else 1
    dynamic_dense_inter_dim = int(np.prod(dw_shape) * dynamic_dense_hidden_expand)

    if cfg.dynamic_dense_hidden_round:  # default: round to 64 or 128
      dynamic_dense_inter_dim = (dynamic_dense_inter_dim// 64 + 1) * 64

    self.dynamic_dense_inter_dim = dynamic_dense_inter_dim
    max_logging.log(f'hids length: {hids_length} dw_shape: {dw_shape} dynamic_dense_inter_dim: {dynamic_dense_inter_dim}', debug=cfg.debug)
    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    # (model_dim, inter_dim), inter_dim << model_dim
    self.dense_proj1 = linears.DenseGeneral(
                                    dynamic_dense_inter_dim,
                                    kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
                                    kernel_axes=('embed', None),
                                    use_bias=False,
                                    name='dynamic_dense_conn1',
                                    **kwargs)
    self.dense_activation = linears._convert_to_activation_function(cfg.dynamic_dense_act_cls)
    
    self.dense_proj2 = linears.DenseGeneral(
                                    dw_shape if not cfg.mudd_use_muon else np.prod(dw_shape), 
                                    kernel_init=initializers.contant_dense_init(0.0), 
                                    kernel_axes=('kv', None, None) if not cfg.mudd_use_muon else ('kv', None), 
                                    use_bias=False, 
                                    name='dynamic_dense_conn2', 
                                    **kwargs)
    if self.use_bias:
      self.dense2_bias_init_value = 0.0 if cfg.mudd_prenorm and cfg.mudd_postnorm else 1.0
      init_v = jnp.array([0] * (dw_shape[1] - 1) + [self.dense2_bias_init_value]).astype(cfg.weight_dtype)
      init_v = init_v[None].repeat(C, 0)
      self.dense_proj2_bias = self.param(f"dense_proj2.bias", init_fn=lambda rng: init_v)

    
  @nn.compact
  def __call__(
      self,
      layer_output,
  ):
    cfg = self.config
    dyn_dense_w = None
    if cfg.dynamic_dense_type == 'qkvm' and cfg.dense_conn:
      x_out_normed = self.pre_dense_proj1_norm(layer_output)
      dense_w_inner = self.dense_activation(self.dense_proj1(x_out_normed))
      dyn_dense_kernel_out = self.dense_proj2(dense_w_inner)
      if cfg.mudd_use_muon:
        # bt(c*l) -> btcl
        dyn_dense_kernel_out = dyn_dense_kernel_out.reshape(*dyn_dense_kernel_out.shape[:-1], *self.dw_shape)

      if cfg.dynamic_dense_scale_dw:
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
  C: int = 4
  compose: bool = False

  def get_compose_mask(self, lidx, cfg, hids_length):
    if lidx is None or cfg.mudd_emb_dilation is None:
      return None
    mask = []
    for i in range(hids_length):
      if i < cfg.mudd_num_extra_emb + 1: # prefix embeddings
        group_idx = lidx % cfg.mudd_emb_dilation
        if i % cfg.mudd_emb_dilation == group_idx: 
          mask.append(True)
        else:
          mask.append(False)
      else: # layer outputs
        mask.append(True)
    return mask

  def gate_fn(self, inputs, embeds):
    # ---------------------------------------------------------
    # B, T, D, L = embeds.shape
    K = self.config.mudd_embed_topk
    kwargs = dict(dtype=self.config.dtype, weight_dtype=self.config.weight_dtype, quant=self.quant)
    logits = linears.DenseGeneral(
      self.config.mudd_embed_topk,
      kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
      kernel_axes=('embed', None),
      use_bias=False,
      name='gate',
      **kwargs)(inputs)
    scores = nn.sigmoid(logits)
    top_scores, indexes = jax.lax.top_k(scores, k=K)   # (B,T,K), (B,T,K)
    # jnp.take_along_axis requires matching dims, so expand indexes
    expanded_idx = indexes[..., None, :]    
    print(f'logits: {logits.shape} embeds: {embeds.shape} expanded_idx: {expanded_idx.shape}')
                    # (B, T, K, 1)
    topk_embeds = jnp.take_along_axis(embeds, expanded_idx, axis=-1)  # (B, T, D, K)
    topk_embeds = topk_embeds * top_scores[..., None, :]

    return topk_embeds
    
  @nn.compact
  def __call__(
      self,
      layer_output,
      hids,
      lidx=None, # if layer index is None, compose all layers
  ):
    cfg = self.config
    
    y = layer_output
    C = self.C
    # y_normed = normalizations.get_rmsnorm("mudd_prenorm", cfg)(y) if cfg.mudd_prenorm else y
    # hids.append(y_normed)
    if not self.compose:
      return y, hids

    # # lsp: mudd embed gate
    # topk_embeds = self.gate_fn(layer_output, hids[0])
    # print(f'topk_embeds: {topk_embeds.shape}'
    # )
    # hids = [topk_embeds] + hids[1:]

    # mask = self.get_compose_mask(lidx, cfg, len(hids))
    mask = None
    if (cfg.mudd_num_extra_emb and lidx < cfg.num_decoder_layers):
      mudd_emb_length = hids[0].shape[-1] - 1
    else:
      mudd_emb_length = 0
      hids = hids[1:] # mtp layer no use prefix embeddings

    dyn_dense_w = Mlp(self.config, self.mesh, self.quant, mudd_emb_length + len(hids), name='mlp', C=C)(layer_output)
    
    if self.config.record_internal_nn_metrics:
      for op in [jnp.max, jnp.mean, jnp.min, jnp.std, l2norm]:
        self.sow('intermediates', f'dyn_dense_w/{op.__name__}', op(dyn_dense_w.astype(jnp.float32)))
      self.sow('intermediates', f'layer_output/norm', l2norm(y.astype(jnp.float32)))

    max_logging.log(f'C: {C} hids length: {len(hids)}')
    wsum = wsum_muddde if cfg.mudd_num_extra_emb and lidx < cfg.num_decoder_layers else wsum_for

    if cfg.mudd_postnorm:
      post_norm = normalizations.get_rmsnorm(name=f"mudd_postnorm", cfg=cfg, scale_init=nn.initializers.constant(0.001), direct_scale=True)
      print(f'dyn_dense_w: {dyn_dense_w.shape} C: {C}')
      dyn_dense_w = rearrange(dyn_dense_w, 'B T C L -> C B T L 1', C=C)
      y = tuple([y + (post_norm(
          wsum(dyn_dense_w[cidx: cidx + 1], hids, cfg.ddw_gen_chunk_size).squeeze(0)
                                ) if cidx == C - 1 else 
          wsum(dyn_dense_w[cidx: cidx + 1], hids, cfg.ddw_gen_chunk_size).squeeze(0)
                      ) for cidx in range(C)])
    else:
        # (btl, btl, btl, btl)
        dyn_dense_w = rearrange(dyn_dense_w, 'B T C L -> C B T L 1', C=C)
        y = tuple([wsum(dyn_dense_w[cidx: cidx + 1], hids, cfg.ddw_gen_chunk_size).squeeze(0) for cidx in range(C)])
        
    return y, hids