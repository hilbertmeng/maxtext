import os
from typing import Optional

import jax
from flax import linen as nn
from jax.sharding import Mesh
import jax.numpy as jnp

from layers import dc_attentions
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


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Attention = dc_attentions.Attention
RMSNorm = normalizations.RMSNorm
NormalInitializer = initializers.nd_dense_init_normal

#-----------------------------------------
# The Decoder Layer specific for Dcformer++
#-----------------------------------------


class DcformerDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""
  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None

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
    max_logging.log(f'num_layers_per_block: {num_layers_per_block}')
    max_logging.log(f'window_size: {self.config.window_size}')
    window_size = self.config.window_size
    if window_size is None:
        window_size = [None]
    elif isinstance(window_size, list):
        for size in window_size:
            assert isinstance(size, int) or size is None, max_logging.log(f'window_size value error: {size}')
    else:
        raise ValueError(f'Window size: ‘{window_size}’ type is error.....')

    for i in range(num_layers_per_block):
        ws = inputs.shape[1] if window_size[i] is None else window_size[i]
        max_logging.log(f'window_size-{i}: {ws}')
        layer_output = self.sub_block(inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, ws, i, eos_sum)
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

    lnx_rms = models.RMSNorm(
        weight_dtype=cfg.weight_dtype,
        dtype=cfg.dtype,
        name=f'pre_self_attention_layer_norm_{block_index}',
        kernel_axes=('embed',),
        epsilon=cfg.normalization_layer_epsilon,
        )
    lnx = lnx_rms(inputs)
    max_logging.log(f'lnx: {lnx.dtype} block_index: {block_index}')

    lnx = nn.with_logical_constraint(
        lnx, ('activation_batch', 'activation_length', 'activation_embed'))

    assert cfg.attention == 'dot_product', max_logging.log(f'Now dcformer model only support ’dot_product‘ method to compute attention')
    # Self-attention block
    attention_layer = Attention(
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

    attention_lnx = attention_layer(
            lnx,
            lnx,
            decoder_positions,
            decoder_segment_ids=decoder_segment_ids,
            eos_sum=eos_sum,
            deterministic=deterministic,
            model_mode=model_mode)

    max_logging.log(f'attention_lnx: {attention_lnx.dtype}')

    attention_lnx = nn.with_logical_constraint(
        attention_lnx,
        ('activation_batch', 'activation_length', 'activation_embed'))
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = models.RMSNorm(
        weight_dtype=cfg.weight_dtype,
        dtype=cfg.dtype, 
        name=f'post_self_attention_layer_norm_{block_index}', kernel_axes=('embed',),
        epsilon=cfg.normalization_layer_epsilon,
        )(intermediate_inputs)
    max_logging.log(f'hidden_states: {hidden_states.dtype}')
    hidden_states = nn.with_logical_constraint(hidden_states, ('activation_batch', 'activation_length', 'activation_embed'))

    shared_mlp_lnx, unshared_mlp_lnx, aux_loss = None, None, None
    max_logging.log(f'num_experts: {cfg.num_experts} n_shared_experts: {cfg.n_shared_experts}')
    if cfg.n_shared_experts:
        shared_mlp_lnx = linears.MlpBlock(
            intermediate_dim=cfg.mlp_dim,
            activations=cfg.mlp_activations,
            intermediate_dropout_rate=cfg.intermediate_dropout_rate,
            weight_dtype=cfg.weight_dtype,
            dtype=cfg.dtype,
            name=f'mlp_{block_index}',
            config=cfg,
            quant=self.quant,
            kernel_init=NormalInitializer(0.006),
        )(hidden_states, deterministic=deterministic)

        shared_mlp_lnx = nn.Dropout(rate=cfg.mlp_residual_dropout_rate, broadcast_dims=(-2,))(shared_mlp_lnx, deterministic=deterministic)
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
            self.sow('intermediates', 'unshared_mlp_l2norm', unshared_mlp_l2norm)

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

    if cfg.record_internal_nn_metrics:
      self.sow('intermediates', 'activation_mean', jnp.mean(layer_output))
      self.sow('intermediates', 'activation_stdev', jnp.std(layer_output))
    #   index = 4 if layer_output.shape[1] > 30000 else None  # size exceed int32 range, overflow
      index = 4  # 取4条数据，不然多了会溢出
      self.sow(
          'intermediates',
          'activation_fraction_zero',
          jnp.sum(layer_output[:index] == 0) / jnp.size(layer_output[:index]),
      )

    if cfg.scan_layers:
      return layer_output, None
    else:
      return layer_output
