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

# Type alias for quantization
Quant = quantizations.AqtQuantization

# Constants
def l2norm(x: jnp.ndarray) -> jnp.ndarray:
  """Compute L2 norm of input tensor."""
  return jnp.sqrt(jnp.sum(jnp.square(x)))


def wsum(w: jnp.ndarray, # CBTL1
         hids: list[jnp.ndarray], # list of BTD
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
    C = self.C
    dw_shape = (C, hids_length) # lsp
    self.dw_shape = dw_shape
    # lsp
    dynamic_dense_hidden_expand = len(cfg.dynamic_dense_type) if hids_length == cfg.num_decoder_layers - 1 + cfg.mtp_num_layers else 1
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
          
  @nn.compact
  def __call__(
      self,
      layer_output,
      hids,
  ):
    cfg = self.config
    
    y = layer_output
    C = self.C
    y_normed = normalizations.get_rmsnorm("mudd_prenorm", cfg)(y) if cfg.mudd_prenorm else y
    hids.append(y_normed)
    if not self.compose:
      return y, hids

    dyn_dense_w = Mlp(self.config, self.mesh, self.quant, len(hids), name='mlp', C=C)(layer_output)
    if self.config.record_internal_nn_metrics:
      for op in [jnp.max, jnp.mean, jnp.min, jnp.std, l2norm]:
        self.sow('intermediates', f'dyn_dense_w/{op.__name__}', op(dyn_dense_w.astype(jnp.float32)))
      self.sow('intermediates', f'layer_output/norm', l2norm(y.astype(jnp.float32)))

    if cfg.mudd_postnorm:
      post_norm = normalizations.get_rmsnorm("mudd_postnorm", cfg, scale_init=nn.initializers.constant(0.001), direct_scale=True)
      dyn_dense_w = rearrange(dyn_dense_w, 'B T C L -> C B T L 1', C=C)
      y = tuple(
        [y + (post_norm(wsum(dyn_dense_w[cidx: cidx + 1], hids).squeeze(0))
          if cidx == C - 1 else 
        wsum(dyn_dense_w[cidx: cidx + 1], hids).squeeze(0))
          for cidx in range(C)])
    else:
        # (btl, btl, btl, btl)
        dyn_dense_w = rearrange(dyn_dense_w, 'B T C L -> C B T L 1', C=C)
        y = tuple([wsum(dyn_dense_w[cidx: cidx + 1], hids).squeeze(0) for cidx in range(C)])
        
    return y, hids