import os
from typing import Optional
import max_logging

import jax
from flax import linen as nn
from jax.sharding import Mesh
import jax.numpy as jnp
import numpy as np

from layers import dc_attentions, attentions
from layers import embeddings
from layers import linears
from layers import normalizations
from layers import models
from layers import initializers
import tensorflow as tf


if os.environ["HARDWARE"] == "gpu":
    Quant = None
else:
    from layers import quantizations
    Quant = quantizations.AqtQuantization

import common_types


Embed = embeddings.Embed
Attention = attentions.Attention
Quant = quantizations.AqtQuantization
nd_dense_init = initializers.nd_dense_init  # XD
DenseGeneral = linears.DenseGeneral  # XD
contant_dense_init = initializers.contant_dense_init  # lsp
NormalInitializer = initializers.nd_dense_init_normal # lsp


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
# Attention = dc_attentions.Attention
RMSNorm = normalizations.RMSNorm
NormalInitializer = initializers.nd_dense_init_normal

#-----------------------------------------
# The Decoder Layer specific for Dcformer++
#-----------------------------------------


class FusionDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None

  def setup(self) -> None:  # XD
    cfg = self.config
    if not getattr(cfg, 'dense_conn', False): return
    norm_kwargs = {
                "dtype": cfg.dtype,
                "weight_dtype": cfg.weight_dtype,
                "name": "pre_dense_proj1_norm",
                "epsilon": cfg.normalization_layer_epsilon,
                }
    if not getattr(cfg, 'mudd_prenorm', False):
        max_logging.log(f'mudd_prenorm is False')
        norm_kwargs['scale_init'] = None # it means use scale is false
    self.pre_dense_proj1_norm = models.get_rmsnorm(**norm_kwargs)
    
    factor = 1
    layer_inx = int(self.name.split('_')[-1])  # name=f"layers_{i}"
    C = 1 if cfg.dynamic_dense_fix_last_layer and layer_inx == cfg.num_decoder_layers - 1 else len(cfg.dynamic_dense_type)
    dw_shape = (C, ((layer_inx + 1) * factor + 1))
    max_logging.log(f'dynamic_dense_hidden_expand-{layer_inx}: {cfg.dynamic_dense_hidden_expand[layer_inx]}')

    dynamic_dense_inter_dim = int(np.prod(dw_shape) * cfg.dynamic_dense_hidden_expand[layer_inx]) # lsp
    if cfg.dynamic_dense_hidden_round:  # default: round to 64 or 128
      dynamic_dense_inter_dim = (dynamic_dense_inter_dim// 64 + 1) * 64
    self.dynamic_dense_inter_dim = dynamic_dense_inter_dim

    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    self.dense_proj1 = DenseGeneral(
      dynamic_dense_inter_dim,
      kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
      kernel_axes=('embed', 'kv'),
      use_bias=False,
      name='dynamic_dense_conn1',
      **kwargs
    )
    self.dense_activation = linears._convert_to_activation_function(cfg.dynamic_dense_act_cls)
    
    self.dense_proj2 = DenseGeneral(dw_shape, 
                                    kernel_init=contant_dense_init(0.0), 
                                    kernel_axes=('kv', None), 
                                    use_bias=False, 
                                    name='dynamic_dense_conn2', 
                                    **kwargs)
    self.dense2_bias_init_value = 0.0 if cfg.mudd_prenorm and cfg.mudd_postnorm else 1.0
    init_v = jnp.array([0] * ((layer_inx + 1) * factor) + [self.dense2_bias_init_value]).astype(cfg.weight_dtype) # dense_bias_init_method == 'current_only'
    init_v = init_v[None].repeat(C, 0)
    self.dense_proj2_bias = self.param(f"dense_proj2.bias", init_fn=lambda rng: init_v)

    if cfg.dynamic_mlp_dim:
      self.updated_mlp_dim = round(cfg.mlp_dim * (layer_inx / (cfg.num_decoder_layers - 1) + 0.5) / 128) * 128 
    else:
      self.updated_mlp_dim = cfg.mlp_dim
    max_logging.log(f'updated_mlp_dim: {self.updated_mlp_dim}')

  @nn.compact
  def __call__(self,
               inputs,
               decoder_segment_ids,
               decoder_positions,
               deterministic,
               model_mode,
               num_layers_per_block=None,
               eos_sum=None,
               ):
    num_layers_per_block = 1 if num_layers_per_block is None else int(num_layers_per_block)
    window_size = self.config.window_size
    if window_size is None:
        window_size = [None]
    elif isinstance(window_size, list):
        for size in window_size:
            assert isinstance(size, int) or size is None, max_logging.log(f'window_size value error: {size}')
    else:
        raise ValueError(f'Window size: ‘{window_size}’ type is error.....')
    input_len = inputs[0].shape[1] if isinstance(inputs, (list, tuple)) else inputs.shape[1]
    for layer_inx in range(num_layers_per_block):
        ws = input_len if window_size[layer_inx] is None else window_size[layer_inx]
        max_logging.log(f'window_size-{layer_inx}: {ws}')
        layer_output = self.sub_block(inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, ws, layer_inx, eos_sum)
        inputs = layer_output[0] if self.config.scan_layers else layer_output

    return layer_output
        
  def sub_block(self,
               inputs,
               decoder_segment_ids,
               decoder_positions,
               deterministic,
               model_mode,
               window_size,
               block_index,
               eos_sum,
               ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(
        inputs, ('activation_batch', 'activation_length', 'activation_embed')) # fsdp, 1, mp

    def norm(inputs, name_suffix=''):  # XD
        name = 'pre_self_attention_layer_norm' + name_suffix
        lnx = models.get_rmsnorm(name=name, cfg=cfg)(inputs)
        return nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))
    max_logging.log(f'dynamic_dense_type: {cfg.dynamic_dense_type}')

    if cfg.dynamic_dense_type == 'qkvm': # XD
      assert isinstance(inputs, (tuple, list, Array)) and len(inputs) == 4 # lsp: Array also support, but C dimenson must be in 0.
      inputs = [nn.with_logical_constraint(i, ("activation_batch", "activation_length", "activation_embed")) for i in inputs]
      lnx_q, *lnx_kv = [norm(inp, f'_{name_suffix}') for inp, name_suffix in zip(inputs[:3], 'qkv')]
      inputs = inputs[-1] # m: bld
    else:
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
      lnx_q = lnx_kv = norm(inputs)

    if cfg.pre_compose or cfg.post_compose:
        assert cfg.attention == 'dot_product', max_logging.log(f'Now dcformer model only support ’dot_product‘ method to compute attention')
        # Self-attention block
        attention_layer = dc_attentions.Attention(
                        config = cfg,
                        num_query_heads=cfg.num_query_heads,
                        num_kv_heads=cfg.num_kv_heads,
                        head_dim=cfg.head_dim,
                        max_target_length=cfg.max_target_length,
                        max_prefill_predict_length=cfg.max_prefill_predict_length,
                        attention_kernel=cfg.attention,
                        mesh=mesh,
                        dtype=cfg.dtype,
                        dropout_rate=cfg.dropout_rate,
                        name=f'self_attention_{block_index}',
                        float32_qk_product = False,  # computes logits in float32 for stability.
                        float32_logits = True,
                        quant=self.quant,
                        window_size=window_size,
                        kernel_init=NormalInitializer(0.006),
                        )
    else:
        # Self-attention block
        attention_layer = attentions.Attention(
                        config=cfg,
                        num_query_heads=cfg.num_query_heads,
                        num_kv_heads=cfg.num_kv_heads,
                        head_dim=cfg.head_dim,
                        max_target_length=cfg.max_target_length,
                        max_prefill_predict_length=cfg.max_prefill_predict_length,
                        attention_kernel=cfg.attention,
                        mesh=mesh,
                        dtype=cfg.dtype,
                        weight_dtype=cfg.weight_dtype,
                        dropout_rate=cfg.dropout_rate,
                        name=f"self_attention_{block_index}",
                        quant=self.quant,
                        kernel_init=NormalInitializer(0.006), # lsp
                        float32_qk_product = False,  # computes logits in float32 for stability.
                        float32_logits = True,
                        quantize_kvcache=cfg.quantize_kvcache,
                        prefill_cache_axis_order=tuple([int(i) for i in cfg.prefill_cache_axis_order.split(",")]),
                        ar_cache_axis_order=tuple([int(i) for i in cfg.ar_cache_axis_order.split(",")]),
                        compute_axis_order=tuple([int(i) for i in cfg.compute_axis_order.split(",")]),
                        reshape_q=cfg.reshape_q,
                        kv_quant_axis=cfg.kv_quant_axis,
                    )
    attention_lnx = attention_layer(
        lnx_q,
        lnx_kv,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )

    attention_lnx = nn.with_logical_constraint(attention_lnx, ('activation_batch', 'activation_length', 'activation_embed'))
    intermediate_inputs = inputs + attention_lnx

    hidden_states = models.get_rmsnorm(name=f"post_self_attention_layer_norm_{block_index}", cfg=cfg)(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(hidden_states, ('activation_batch', 'activation_length', 'activation_embed'))

    shared_mlp_lnx, unshared_mlp_lnx, aux_loss = None, None, None
    max_logging.log(f'num_experts: {cfg.num_experts} n_shared_experts: {cfg.n_shared_experts}')
    if cfg.n_shared_experts:
        shared_mlp_lnx = linears.MlpBlock(
            intermediate_dim=self.updated_mlp_dim if cfg.dynamic_mlp_dim else cfg.mlp_dim,  # lsp
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.intermediate_dropout_rate,
            weight_dtype=cfg.weight_dtype,
            dtype=cfg.dtype,
            name=f'mlp_{block_index}',
            config=cfg,
            quant=self.quant,
            kernel_init=NormalInitializer(0.006),
        )(hidden_states, deterministic=deterministic)
        # shared_mlp_lnx = nn.Dropout(rate=cfg.mlp_residual_dropout_rate, broadcast_dims=(-2,))(shared_mlp_lnx, deterministic=deterministic)
        shared_mlp_lnx /= cfg.shared_mlp_scale

        if cfg.record_internal_nn_metrics:
            shared_mlp_l2norm = jnp.sqrt(jnp.sum(jnp.square(shared_mlp_lnx)))
            self.sow('intermediates', 'shared_mlp_l2norm', shared_mlp_l2norm)

    if cfg.num_experts >= 1 and block_index % cfg.insert_moe_divisor == 0:
        if cfg.moe_type == 'mistral':
            unshared_mlp_lnx, aux_loss = linears.MoeBlock(
                                    name=f'unshared_mlp_{block_index}',
                                    config=cfg,
                                    num_experts=cfg.num_experts,
                                    num_experts_per_tok=cfg.num_experts_per_tok,
                                    mesh=mesh,
                                    kernel_init=initializers.nd_dense_init(1.0, 'fan_in', 'truncated_normal'),
                                    kernel_axes=('embed', 'mlp'),
                                    weight_dtype=cfg.weight_dtype,
                                    dtype=cfg.dtype,
                                    )(hidden_states)
        else:
            max_logging.log(f'Add moe layer: {block_index}')
            unshared_mlp_lnx, aux_loss = linears.DcMoeBlock(
                name=f'unshared_mlp_{block_index}',
                config=cfg,
                mesh=mesh,
                kernel_init=initializers.nd_dense_init(1.0, 'fan_in', 'truncated_normal'),
                kernel_axes=('embed', 'mlp'),
                weight_dtype=cfg.weight_dtype,
                dtype=cfg.dtype,
                num_experts=cfg.num_experts,
                intermediate_dim=cfg.mlp_dim, # cfg.mlp_dim
                intermediate_dropout_rate = cfg.intermediate_dropout_rate,
            )(hidden_states, paddings=decoder_segment_ids, deterministic=deterministic)
        
        unshared_mlp_lnx = nn.Dropout(rate=cfg.unshared_mlp_dropout_rate, broadcast_dims=(-2,))(unshared_mlp_lnx, deterministic=deterministic)
        unshared_mlp_lnx /= cfg.unshared_mlp_scale

        if cfg.record_internal_nn_metrics:
            unshared_mlp_l2norm = jnp.sqrt(jnp.sum(jnp.square(unshared_mlp_lnx)))
            self.sow('intermediates', 'unshared_mlp/l2norm', unshared_mlp_l2norm)

    if shared_mlp_lnx is None:
        max_logging.log(f'shared_mlp_lnx is None')
        mlp_lnx = unshared_mlp_lnx
    elif unshared_mlp_lnx is None:
        max_logging.log(f'unshared_mlp_lnx is None')
        mlp_lnx = shared_mlp_lnx
    else:
        mlp_lnx = (shared_mlp_lnx + unshared_mlp_lnx) / 2
    max_logging.log(f'mlp_lnx: {mlp_lnx.dtype} shape: {mlp_lnx.shape}')

    mlp_lnx = nn.with_logical_constraint(
        mlp_lnx, ('activation_batch', 'activation_length', 'activation_embed')
    )
    layer_output = mlp_lnx + intermediate_inputs

    layer_output = nn.Dropout(
        rate=cfg.dropout_rate, broadcast_dims=(-2,))(
            layer_output, deterministic=deterministic)

    layer_output = nn.with_logical_constraint(
        layer_output,
        ('activation_batch', 'activation_length', 'activation_embed'),
    )

    if cfg.num_experts > 0 and aux_loss is not None:
      max_logging.log(f'return moe_lb_loss...')
      self.sow("intermediates", "moe_lb_loss", aux_loss)

    if cfg.dynamic_dense_type == 'qkvm':
      x_out_normed = self.pre_dense_proj1_norm(layer_output, x_dtype=jnp.bfloat16)
      dense_w_inner = self.dense_activation(self.dense_proj1(x_out_normed))
      dyn_dense_kernel_out = self.dense_proj2(dense_w_inner)
      if cfg.dynamic_dense_scale_dw:
        max_logging.log(f'dynamic_dense_scale_dw: {cfg.dynamic_dense_scale_dw}')
        dyn_dense_kernel_out /= jnp.sqrt(self.dynamic_dense_inter_dim)
      dyn_dense_w = dyn_dense_kernel_out + self.dense_proj2_bias.astype(dyn_dense_kernel_out.dtype)

    if cfg.scan_layers:
      assert cfg.dynamic_dense_type != 'qkvm'
      return layer_output, None
    elif cfg.dynamic_dense_type == 'qkvm':
      assert not cfg.scan_layers
      return  layer_output, dyn_dense_w
    else:
      return layer_output
