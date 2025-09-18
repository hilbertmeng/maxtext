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
from errno import EILSEQ
import jax
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
    cfg = self.config
    self.mudd_qkvnorm = mudd.Norm(cfg, self.mesh, self.quant)

    if cfg.dynamic_mlp_dim:
      self.updated_mlp_dim = round(cfg.mlp_dim * (self.layer_inx / (cfg.num_decoder_layers - 1) + 0.5) / 128) * 128 
    else:
      self.updated_mlp_dim = cfg.mlp_dim
    max_logging.log(f'sliding_window_size: {self.sliding_window_size} updated_mlp_dim: {self.updated_mlp_dim}', debug=cfg.debug)


  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      decoder_input_tokens,
      deep_embedding,
      deterministic,
      model_mode,
      eos_sum,
      hids=None,
  ):
    cfg = self.config
    mesh = self.mesh
    if cfg.dense_conn and cfg.dynamic_dense_type == 'qkvm': # lsp
      lnx, *lnx_kv = self.mudd_qkvnorm(inputs[:3])
      inputs = inputs[3]
    else:
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
      inputs = checkpoint_name(inputs, "decoder_layer_input")
      lnx_rms = models.RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_layer_norm",
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
    )
      lnx = lnx_rms(inputs)

      lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    # Self-attention block
    attention_layer = Attention(
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
        kernel_init=initializers.get_init_method(cfg.init_method), # lsp
        sliding_window_size=self.sliding_window_size,
        rng=jax.random.PRNGKey(9),  # lsp
    )

    attention_lnx = attention_layer(
        lnx,
        lnx if not cfg.dense_conn else lnx_kv,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        eos_sum=eos_sum,
        decoder_input_tokens=decoder_input_tokens,
        deep_embedding=deep_embedding,
    )
    if 'attn' in cfg.deep_embed_type:
      attention_lnx = linears.DeepEmbedBlock(
        name='attnout_deep_embed',
        config=cfg, 
        kernel_init=initializers.get_init_method(cfg.init_method),
        weight_dtype=cfg.weight_dtype, 
        dtype=cfg.dtype, 
        input_dim=cfg.emb_dim,
        output_dim=cfg.emb_dim,
        de_d1_d2_dims=(32, cfg.emb_dim // 32)  # fix first dimension to 32, and don't need to follow mudd mlp dim.
        )(inputs, attention_lnx, decoder_input_tokens, deep_embedding)
      print(f'Outside DE is None, inside 1x Attn DE')

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = models.RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="post_self_attention_layer_norm",
        kernel_axes=("norm",),
        epsilon=cfg.normalization_layer_epsilon,
    )(intermediate_inputs)
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
          kernel_init=initializers.get_init_method(cfg.init_method), # lsp
          rng=jax.random.PRNGKey(10),  # lsp
      )(hidden_states, deep_embedding=deep_embedding, decoder_input_tokens=decoder_input_tokens, deterministic=deterministic)
      mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

      if cfg.record_internal_nn_metrics:
        mlp_l2norm = jnp.sqrt(jnp.sum(jnp.square(mlp_lnx)))
        self.sow('intermediates', 'mlp_lnx/l2norm', mlp_l2norm)

    # lsp: moe
    moe_lnx = None
    load_balance_loss = 0.0
    if cfg.num_experts > 1:
      kwargs = {
        'config': cfg,
        'mesh': mesh,
        'kernel_init': initializers.get_init_method(cfg.init_method), # lsp
        'kernel_axes': ("embed", None),
        'dtype': cfg.dtype,
        'weight_dtype': cfg.weight_dtype,
        'quant': self.quant,
        'name': 'moe'
      }
      extra_kwargs = {
        'num_experts': cfg.num_experts,
        'num_experts_per_tok': cfg.num_experts_per_tok,
        'intermediate_dim': self.updated_mlp_dim, # lsp
        }
      if cfg.moe_type == 'open': # with capacity and noise and balance loss
        moe_layer = linears.OpenMoeBlock
        kwargs.update(extra_kwargs)
        kwargs.update(extra_kwargs)
      elif cfg.moe_type == 'deepseek': # model performance bad
        moe_layer = linears.DeepSeekMoeBlock
      elif cfg.moe_type == 'dropless': # no capacity and nosie, maybe have balance loss, bug no imporve with balance loss 0.01
        kwargs.update(extra_kwargs)
        moe_layer = linears.MoeBlock
      else:
        raise ValueError(f'Unknow moe type: {cfg.moe_type}, it must be in [open, deepseek, ol, dropless]')
      if cfg.moe_type == 'dropless':
        moe_lnx, load_balance_loss = moe_layer(**kwargs)(
          hidden_states, 
          decoder_input_tokens=decoder_input_tokens, 
          paddings=decoder_segment_ids, 
          deterministic=deterministic
          )
      else:
        moe_lnx, load_balance_loss = moe_layer(**kwargs)(
          hidden_states, 
          paddings=decoder_segment_ids, 
          deterministic=deterministic)

      if cfg.record_internal_nn_metrics: # lsp
            moe_mlp_l2norm = jnp.sqrt(jnp.sum(jnp.square(moe_lnx)))
            self.sow('intermediates', 'moe_lnx/l2norm', moe_mlp_l2norm)
        
      if load_balance_loss is not None:
        self.sow("intermediates", "moe_lb_loss", load_balance_loss)
      moe_lnx = nn.with_logical_constraint(moe_lnx, ("activation_batch", "activation_norm_length", "activation_embed"))

    if mlp_lnx is not None and moe_lnx is not None:
      layer_output = mlp_lnx + intermediate_inputs + moe_lnx
    elif mlp_lnx is not None and moe_lnx is None:
      layer_output = mlp_lnx + intermediate_inputs
    elif mlp_lnx is None and moe_lnx is not None:
      layer_output = intermediate_inputs + moe_lnx
    else:
      raise ValueError("Both mlp_lnx and moe_lnx is None, it's not allowed.")

    layer_output = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(layer_output, deterministic=deterministic)

    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )
    if cfg.mudd_in_layer:
      # return's inputs length is 4
      layer_output, hids = mudd.Compose(
          cfg, self.mesh, self.quant, self.layer_inx, 
          name=f'compose'
          )(
            layer_output=layer_output, 
            hids=hids,
            decoder_input_tokens=decoder_input_tokens,
          )
    return layer_output, hids


class FusionDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  sliding_window_size: list|int|None = -1 # lsp

  def setup(self):
    cfg = self.config
    layer_inx = None if cfg.scan_layers else int(self.name.split('_')[-1])
    self.layer_inx = layer_inx
    # When no sliding_window_size is passed in, the sliding_window_size in config is used, otherwise the passed in sliding_window_size is used.
    sliding_window_size = cfg.sliding_window_size if self.sliding_window_size == -1 else self.sliding_window_size
    if not isinstance(sliding_window_size, (list, tuple)):
        sliding_window_size = [sliding_window_size]

    sliding_window_size = [s or cfg.max_target_length for s in sliding_window_size]
    # if cfg.num_layers_per_block > 1 or not cfg.mudd_in_layer: # mudd_in_layer is false, sub must use remat
    RematSubDecoderLayer = nn.remat(SubDecoderLayer,
                                      prevent_cse=True,
                                      policy=models.get_remat_policy(cfg),
                                      static_argnums=(6, 7),  # Deterministic and model mode are static arguments.
                                      )
    # RematSubDecoderLayer = SubDecoderLayer
    self.subs = [RematSubDecoderLayer(cfg, self.mesh, self.quant, sws, layer_inx, name=f'sub_{i}')
                                      for i, sws in enumerate(sliding_window_size)]
    
  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      decoder_input_tokens,
      deep_embedding,
      deterministic,
      model_mode,
      hids=None,
      eos_sum=None,
  ):
    cfg = self.config
    for layer in self.subs: # subs length must be 1 when train mudd.
        # return's inputs length is 4 if mudd_in_layer is True else inputs length is 1
        inputs, hids = layer(
            inputs,
            decoder_segment_ids,
            decoder_positions,
            decoder_input_tokens,
            deep_embedding,
            deterministic,
            model_mode,
            eos_sum,
            hids,
        )
        if not cfg.mudd_in_layer:
          inputs, hids = mudd.Compose(
              cfg, self.mesh, self.quant, self.layer_inx, 
              name=f'compose'
              )(
                layer_output=inputs, 
                hids=hids,
                decoder_input_tokens=decoder_input_tokens,
              )
    return inputs, hids