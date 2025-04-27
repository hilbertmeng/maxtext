"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Transformer model definition."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from flax import linen as nn
from jax.sharding import Mesh
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
# from jax.experimental.pallas.ops.tpu import flash_attention
from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations
from layers import models
from layers import quantizations
from layers import mudd
from layers import initializers
import max_logging

from layers.gpt3 import Gpt3LayerNorm

import maxtext_utils
import common_types
from typing import Optional, Any

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

AxisNames = common_types.AxisNames
BATCH = common_types.BATCH
KV_BATCH = common_types.KV_BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
KV_HEAD = common_types.KV_HEAD
D_KV = common_types.D_KV
KV_HEAD_DIM = common_types.KV_HEAD_DIM


Embed = embeddings.Embed
Attention = attentions.Attention
RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization


# -----------------------------------------
# The Decoder Layer specific for llama2
# -----------------------------------------
class SubDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: models.Config
  mesh: Mesh
  quant: Optional[Quant] = None
  sliding_window_size: int|None = None
  layer_inx: int|None = None

  def setup(self):
    max_logging.log(f'SubDecoderLayer layer_inx: {self.layer_inx} sliding_window_size: {self.sliding_window_size}', debug=self.config.debug)
    self.mudd_mlp = mudd.Mlp(self.config, self.mesh, self.quant, self.layer_inx)
    self.mudd_qkvnorm = mudd.Norm(self.config, self.mesh, self.quant)

    if self.config.dynamic_mlp_dim:
      self.updated_mlp_dim = round(self.config.mlp_dim * (self.layer_inx / (self.config.num_decoder_layers - 1) + 0.5) / 128) * 128 
    else:
      self.updated_mlp_dim = self.config.mlp_dim
    max_logging.log(f'updated_mlp_dim: {self.updated_mlp_dim}', debug=self.config.debug)


  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      value_residual=None,
  ):
    cfg = self.config
    mesh = self.mesh

    norm_class = models.RMSNorm if self.config.norm_type == 'rmsnorm' else Gpt3LayerNorm
    raw_inputs = inputs

    if cfg.dense_conn and cfg.dynamic_dense_type == 'qkvm': # lsp
      lnx, *lnx_kv = self.mudd_qkvnorm(inputs[:3])
      inputs = inputs[3]
    else:
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
      inputs = checkpoint_name(inputs, "decoder_layer_input")
      lnx_rms = norm_class(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_layer_norm",
        # kernel_axes=("norm",),
        kernel_axes=("embed",),
        epsilon=cfg.normalization_layer_epsilon,
    )
      lnx = lnx_rms(inputs)

      lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    if cfg.inner_ffn_dim:
      lnx = linears.MlpBlock(
          intermediate_dim=cfg.inner_ffn_dim, # lsp
          activations=cfg.inner_ffn_activations or cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="inner_mlp",
          config=cfg,
          quant=self.quant,
          use_bias=cfg.use_bias,
          kernel_init=initializers.nd_dense_init_normal(0.006), # lsp
      )(lnx, deterministic=deterministic)
      lnx = lnx + inputs # inner ffn residual for attn
      lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))
      # attn norm (postnorm after inner ffn)
      lnx_rms = norm_class(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="post_inner_ffn_layer_norm",
        # kernel_axes=("norm",),
        kernel_axes=("embed",),
        epsilon=cfg.normalization_layer_epsilon,
      )
      lnx = lnx_rms(lnx)

    max_logging.log(f'Attention inputs: {inputs.shape}', debug=self.config.debug)
    # Self-attention block

    if cfg.attention_type == 'mla':
      attention_class = attentions.MLA
      mla_kwargs = dict(
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        max_seq_len=cfg.max_target_length,
        original_seq_len=cfg.original_seq_len,
        mscale=cfg.mscale,
        rope_factor=cfg.rope_factor,
        )
    else:
      attention_class = Attention
      mla_kwargs = {}

    attention_layer = attention_class(
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
      name="self_attention",
      float32_qk_product=cfg.float32_qk_product,
      float32_logits=cfg.float32_logits,
      quant=self.quant,
      kv_quant=quantizations.configure_kv_quant(cfg),
      prefill_cache_axis_order=tuple([int(i) for i in cfg.prefill_cache_axis_order.split(",")]),
      ar_cache_axis_order=tuple([int(i) for i in cfg.ar_cache_axis_order.split(",")]),
      compute_axis_order=tuple([int(i) for i in cfg.compute_axis_order.split(",")]),
      reshape_q=cfg.reshape_q,
      use_ragged_attention=cfg.use_ragged_attention,
      ragged_block_size=cfg.ragged_block_size,
      kernel_init=initializers.nd_dense_init_normal(0.006), # lsp
      sliding_window_size=self.sliding_window_size,
      layer_inx=self.layer_inx,
      use_kv_shift=cfg.use_kv_shift,
      use_alibi=cfg.use_alibi,
      use_postnorm=cfg.use_postnorm,
      **mla_kwargs,
    )

    attention_lnx, value_residual = attention_layer(
        lnx,
        lnx if not cfg.dense_conn else lnx_kv,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        hidden_states=inputs,
        value_residual=value_residual,
    )

    if self.config.record_internal_nn_metrics:
      self.sow('intermediates', 'attn_out', maxtext_utils.l2norm(attention_lnx))

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    intermediate_inputs = inputs + attention_lnx

    if self.config.attn_ffn_parallel:
      if cfg.dense_conn and cfg.dynamic_dense_type == 'qkvm':
        mlp_inputs = raw_inputs[3]
      else:
        mlp_inputs = raw_inputs
    else:
      mlp_inputs = intermediate_inputs
    # Fully Connected
    hidden_states = norm_class(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="post_self_attention_layer_norm",
        # kernel_axes=("norm",),
        kernel_axes=("embed",),
        epsilon=cfg.normalization_layer_epsilon,
    )(mlp_inputs)
    hidden_states = nn.with_logical_constraint(
        hidden_states, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    
    mlp_lnx = None
    if cfg.shared_experts == 1:
      # MLP block.
      mlp_lnx = linears.MlpBlock(
          intermediate_dim=self.updated_mlp_dim, # lsp
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="mlp",
          config=cfg,
          quant=self.quant,
          use_bias=cfg.use_bias,
          kernel_init=initializers.nd_dense_init_normal(0.006), # lsp
      )(hidden_states, deterministic=deterministic)
      mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    # lsp: moe
    moe_lnx = None
    load_balance_loss = None
    if cfg.num_experts > 1:
      if cfg.moe_type == 'openmoe':
        moe_layer = linears.OpenMoeBlock
      else:
        moe_layer = linears.MoeBlock
      moe_lnx, load_balance_loss = moe_layer(
        config=cfg,
        num_experts=cfg.num_experts,
        num_experts_per_tok=cfg.num_experts_per_tok,
        mesh=mesh,
        kernel_init=initializers.nd_dense_init_normal(0.006),
        kernel_axes=("embed", None),
        intermediate_dim=cfg.mlp_dim,
        weight_dtype=cfg.weight_dtype,
        dtype=cfg.dtype,
        quant=self.quant,
        name='moe'
        )(hidden_states, paddings=decoder_segment_ids)
      max_logging.log(f'moe_lnx: {moe_lnx.shape}', debug=cfg.debug)
        
      if load_balance_loss is not None:
        self.sow("intermediates", "moe_lb_loss", load_balance_loss)
      moe_lnx = nn.with_logical_constraint(moe_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    if mlp_lnx is not None and moe_lnx is not None:
      max_logging.log('mlp_lnx is not None and moe_lnx is not None.', debug=cfg.debug)
      layer_output = mlp_lnx + intermediate_inputs + moe_lnx
    elif mlp_lnx is not None and moe_lnx is None:
      max_logging.log('mlp_lnx is not None and moe_lnx is None.', debug=cfg.debug)
      layer_output = mlp_lnx + intermediate_inputs
    elif mlp_lnx is None and moe_lnx is not None:
      max_logging.log('mlp_lnx is None and moe_lnx is not None.', debug=cfg.debug)
      layer_output = intermediate_inputs + moe_lnx
    else:
      raise ValueError("Both mlp_lnx and moe_lnx is None, it's not allowed.")

    if self.config.record_internal_nn_metrics:
      self.sow('intermediates', 'mlp_out', maxtext_utils.l2norm(mlp_lnx))

    layer_output = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(layer_output, deterministic=deterministic)

    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )

    if 0 and cfg.record_internal_nn_metrics: # lsp: unused
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    dyn_dense_w = self.mudd_mlp(layer_output) if not self.config.mudd_in_layer else None# lsp
    if self.config.value_residual_learning:
      return layer_output, dyn_dense_w, value_residual
    else:
      return layer_output, dyn_dense_w


class FusionDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  sliding_window_size: list|int|None = -1 # lsp

  def setup(self):
    layer_inx = None if self.config.scan_layers else int(self.name.split('_')[-1])
    # When no sliding_window_size is passed in, the sliding_window_size in config is used, otherwise the passed in sliding_window_size is used.
    sliding_window_size = self.config.sliding_window_size if self.sliding_window_size == -1 else self.sliding_window_size
    max_logging.log(f'FusionDecoderLayer layer_inx: {layer_inx} sliding_window_size: {sliding_window_size}', debug=self.config.debug)
    if not isinstance(sliding_window_size, (list, tuple)):
        sliding_window_size = [sliding_window_size]

    if len(sliding_window_size) != 1:
        assert not self.config.dense_conn
    self.layer_inx = layer_inx
    self.subs = [SubDecoderLayer(self.config, self.mesh, self.quant, sws, layer_inx, name=f'sub_{i}') for i, sws in enumerate(sliding_window_size)]

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
      hids=None,
      value_residual=None,
  ):
    if self.config.mudd_in_layer:
        if self.layer_inx == 0: # first layer
            inputs = [inputs] * len(self.config.dynamic_dense_type)
        else:
            inputs, hids = mudd.Compose(self.config, self.mesh, self.quant, self.layer_inx-1, name=f'compose_{self.layer_inx-1}')(inputs, hids) # lsp

    for layer in self.subs:
        if self.config.value_residual_learning:
          inputs, dyn_dense_w, value_residual = layer(inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode, value_residual=value_residual)
        else:
          inputs, dyn_dense_w = layer(inputs, decoder_segment_ids, decoder_positions, deterministic, model_mode,)
    
    if self.config.mudd_in_layer:
        if self.layer_inx == self.config.base_num_decoder_layers-1: # last layer
            inputs, hids = mudd.Compose(self.config, self.mesh, self.quant, self.layer_inx, name=f'compose_{self.layer_inx}')(inputs, hids) # lsp
        return inputs, hids, value_residual
    return inputs, dyn_dense_w, value_residual
