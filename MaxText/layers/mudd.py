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

# Type alias for quantization
Quant = quantizations.AqtQuantization

# Constants
def l2norm(x: jnp.ndarray) -> jnp.ndarray:
  """Compute L2 norm of input tensor."""
  return jnp.sqrt(jnp.sum(jnp.square(x)))


def wsum(w: jnp.ndarray, # CBTL1
         hids: list[jnp.ndarray], # list of BTD
         seq_chunk_size: int = None
         ) -> jnp.ndarray:  # CBTD
  C, B, T, L, _ = w.shape
  D = hids[0].shape[-1]
  out = jnp.zeros((C, B, T, D), dtype=hids[0].dtype)
  for l in range(L): # 每层
    out += w[..., l, :] * hids[l]
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
  C: int = 4

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
        norm_kwargs['scale_init'] = None # it means use scale is false
    self.pre_dense_proj1_norm = normalizations.get_rmsnorm(**norm_kwargs)
    
    factor = 1
    layer_inx = self.layer_inx
    C = self.C
    dw_shape = (C, layer_inx * factor + 1) # lsp
    self.dw_shape = dw_shape
    # lsp
    dynamic_dense_hidden_expand = len(cfg.dynamic_dense_type) if layer_inx == cfg.num_decoder_layers - 1 + cfg.mtp_num_layers else 1
    dynamic_dense_inter_dim = int(np.prod(dw_shape) * dynamic_dense_hidden_expand)

    if cfg.dynamic_dense_hidden_round:  # default: round to 64 or 128
      dynamic_dense_inter_dim = (dynamic_dense_inter_dim// 64 + 1) * 128

    self.dynamic_dense_inter_dim = dynamic_dense_inter_dim
    print('\n==================================================')
    print(f'layer_inx: {layer_inx} dw_shape: {dw_shape} dynamic_dense_inter_dim: {dynamic_dense_inter_dim}')
    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    # (model_dim, inter_dim), inter_dim << model_dim
    self.dense_proj1 = linears.DenseGeneral(
                                    dynamic_dense_inter_dim,
                                    kernel_init=initializers.nd_dense_init(1.0, "fan_in", "normal"),
                                    kernel_axes=('embed', 'kv'),
                                    use_bias=False,
                                    name='dynamic_dense_conn1',
                                    **kwargs)
    self.dense_activation = linears._convert_to_activation_function(cfg.dynamic_dense_act_cls)
    
    self.dense_proj2 = linears.DenseGeneral(
                                    dw_shape if not cfg.mudd_use_muon else np.prod(dw_shape), 
                                    kernel_init=initializers.contant_dense_init(0.0), 
                                    kernel_axes=('kv', None, 'mlp') if not cfg.mudd_use_muon else ('kv', 'mlp'), 
                                    use_bias=False, 
                                    # lsp：这个参数相当于scale的作用，感觉不适合muon
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
      decoder_input_tokens,
  ):
    cfg = self.config
    dyn_dense_w = None
    if cfg.dynamic_dense_type == 'qkvm' and cfg.dense_conn:
      x_out_normed = self.pre_dense_proj1_norm(layer_output)
      dense_w_inner = self.dense_activation(self.dense_proj1(x_out_normed))

      # Apply 4x deep embedding if configured
      if cfg.mudd_deep_embed == '4x':
        dense_w_inner = self._apply_deep_embed_4x(cfg, layer_output, dense_w_inner, decoder_input_tokens)

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

      # Apply deep embedding for 1x and 1xScale modes
      if cfg.mudd_deep_embed == '1x':
        print(f'Mudd 1x deep embed')
        dyn_dense_w = self._apply_deep_embed_1x(cfg, layer_output, dyn_dense_w, decoder_input_tokens)
      elif cfg.mudd_deep_embed == '1xScale':
        print(f'Mudd 1xScale deep embed')
        dyn_dense_w = self._apply_deep_embed_1x(cfg, layer_output, dyn_dense_w, decoder_input_tokens, scale_mode=True)
    return dyn_dense_w

  def _create_deep_embedding(self, cfg, decoder_input_tokens, features, embed_init=None, shard_axes=None):
    """Create deep embedding with specified configuration."""
    embed_kwargs = {
      "num_embeddings": cfg.vocab_size,
      "features": features,
      "dtype": cfg.dtype,
      "embedding_init": embed_init or initializers.nd_dense_init_normal(0.006),
      "name": "deep_embed",
      "config": cfg,
    }
    if shard_axes:
      embed_kwargs["shard_axis_name"] = shard_axes
    
    return embeddings.Embed(**embed_kwargs)(decoder_input_tokens.astype("int32"))

  def _apply_deep_embed_4x(self, cfg, layer_output, dense_w_inner, decoder_input_tokens):
    """Apply 4x deep embedding logic."""
    deep_embedding = self._create_deep_embedding(
      cfg, decoder_input_tokens, self.dynamic_dense_inter_dim
    )
    
    return linears.DeepEmbedBlock(
      config=cfg,
      kernel_init=initializers.nd_dense_init_normal(0.006),
      weight_dtype=cfg.weight_dtype,
      dtype=cfg.dtype,
      intermediate_dim=self.dynamic_dense_inter_dim
    )(layer_output, dense_w_inner, deep_embedding)

  def _apply_deep_embed_1x(self, cfg, layer_output, dyn_dense_w, decoder_input_tokens, scale_mode=False):
    """Apply 1x deep embedding logic."""
    intermediate_dim = np.prod(self.dw_shape)
    
    if scale_mode:
      # 1xScale mode: use constant initialization for scaling
      embed_init = nn.initializers.constant(1.0)
      deep_embedding = self._create_deep_embedding(
        cfg, decoder_input_tokens, intermediate_dim, embed_init, ("embed", "vocab")
      )
      deep_embedding = deep_embedding.reshape(*decoder_input_tokens.shape[:2], *self.dw_shape)
      return deep_embedding * dyn_dense_w
    else:
      # Standard 1x mode
      deep_embedding = self._create_deep_embedding(
        cfg, decoder_input_tokens, intermediate_dim, shard_axes=("embed", "vocab")
      )
      
      dyn_dense_w_reshaped = dyn_dense_w.reshape(*decoder_input_tokens.shape[:2], -1)
      dyn_dense_w_processed = linears.DeepEmbedBlock(
        config=cfg,
        kernel_init=initializers.nd_dense_init_normal(0.006),
        weight_dtype=cfg.weight_dtype,
        dtype=cfg.dtype,
        intermediate_dim=intermediate_dim,
        default_d1=self.C
      )(layer_output, dyn_dense_w_reshaped, deep_embedding)
      
      return dyn_dense_w_processed.reshape(*dyn_dense_w_processed.shape[:2], *self.dw_shape)


class Compose(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  layer_inx: int = None
  C: int = 4
          
  @nn.compact
  def __call__(
      self,
      layer_output,
      hids,
      decoder_input_tokens,
  ):
    cfg = self.config
    if not cfg.dense_conn:
      return layer_output, hids
    
    y = layer_output
    layer_inx = self.layer_inx
    C = 1 if layer_inx == cfg.num_decoder_layers + cfg.mtp_num_layers - 1 else len(cfg.dynamic_dense_type)
    dyn_dense_w = Mlp(self.config, self.mesh, self.quant, self.layer_inx, name='mlp', C=C)(layer_output, decoder_input_tokens)

    if self.config.record_internal_nn_metrics:
      for op in [jnp.max, jnp.mean, jnp.min, jnp.std, l2norm]:
        self.sow('intermediates', f'dyn_dense_w/{op.__name__}/layer_{layer_inx}', op(dyn_dense_w.astype(jnp.float32)))
      self.sow('intermediates', f'layer_output/norm/layer_{layer_inx}', l2norm(y.astype(jnp.float32)))

    y_normed = normalizations.get_rmsnorm(name=f"mudd_prenorm", cfg=cfg)(y) if cfg.mudd_prenorm else y
    hids.append(y_normed)
    print(f'C: {C} hids length: {len(hids)}')
    if cfg.mudd_postnorm:
      post_norm = normalizations.get_rmsnorm(name=f"mudd_postnorm", cfg=cfg, scale_init=nn.initializers.constant(0.001))
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