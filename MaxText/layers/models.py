#  Copyright 2023 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Transformer models."""
# pylint: disable=arguments-differ
# pylint: disable=no-name-in-module

from typing import Any, Callable, Optional

import math
from flax import linen as nn
import functools
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
import common_types
from layers import attentions
from layers import embeddings
from layers import linears
from layers import normalizations, quantizations
from layers import pipeline
from layers import mudd
from layers import initializers
from layers.dynamic_temperature import DynamicTemperature
from layers.kv_shift import Hiddenshift

import max_logging
import maxtext_utils

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Attention = attentions.Attention
RMSNorm = normalizations.RMSNorm
PositionalEmbedding = embeddings.PositionalEmbedding
Quant = quantizations.AqtQuantization

# ------------------------------------------------------------------------------
# The network: Decoder & Transformer Definitions
# ------------------------------------------------------------------------------


class DecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Config
  mesh: Mesh
  quant: Optional[Quant] = None

  @nn.compact
  def __call__(
      self,
      inputs,
      decoder_segment_ids,
      decoder_positions,
      deterministic,
      model_mode,
  ):
    cfg = self.config
    mesh = self.mesh

    inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_length", "activation_embed"))
    inputs = checkpoint_name(inputs, "decoder_layer_input")
    # inputs: embedded inputs to the decoder with shape [batch, length, emb_dim]
    lnx = RMSNorm(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="pre_self_attention_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(inputs)
    lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_length", "activation_embed"))

    attention_layer = Attention(
        config=self.config,
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
    )

    attention_lnx = attention_layer(
        lnx,
        lnx,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
    )

    attention_lnx = nn.with_logical_constraint(attention_lnx, ("activation_batch", "activation_length", "activation_embed"))

    # MLP block.
    mlp_lnx = linears.MlpBlock(
        intermediate_dim=cfg.mlp_dim,
        activations=cfg.mlp_activations,
        intermediate_dropout_rate=cfg.dropout_rate,
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="mlp",
        config=cfg,
        quant=self.quant,
    )(lnx, deterministic=deterministic)
    mlp_lnx = nn.with_logical_constraint(mlp_lnx, ("activation_batch", "activation_length", "activation_embed"))

    next_layer_addition = mlp_lnx + attention_lnx

    next_layer_addition_dropped_out = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(
        next_layer_addition, deterministic=deterministic
    )

    layer_output = next_layer_addition_dropped_out + inputs
    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_length", "activation_embed"),
    )

    if 0 and cfg.record_internal_nn_metrics: # lsp: unused
      self.sow("intermediates", "activation_mean", jnp.mean(layer_output))
      self.sow("intermediates", "activation_stdev", jnp.std(layer_output))
      self.sow(
          "intermediates",
          "activation_fraction_zero",
          jnp.sum(layer_output == 0) / jnp.size(layer_output),
      )

    return layer_output, None if cfg.scan_layers else layer_output


class SequentialBlockDecoderLayers(nn.Module):
  """Sequential unscanned series of decoder layers."""

  decoder_layer: Any
  num_decoder_layers: int
  config: Config
  mesh: Mesh
  quant: Quant

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, decoder_segment_ids, decoder_positions, deterministic, model_mode) -> jnp.ndarray:
    for lyr in range(self.num_decoder_layers):
      inputs = self.decoder_layer(config=self.config, mesh=self.mesh, name=f"layers_{lyr}", quant=self.quant)(
          inputs,
          decoder_segment_ids,
          decoder_positions,
          deterministic,
          model_mode,
      )
    return inputs


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""

  config: Config
  shared_embedding: nn.Module
  mesh: Mesh
  quant: Optional[Quant] = None

  def setup(self):
    """Initialize decoder layer."""
    self.decoder_layer = self.get_decoder_layers()
    # self.norm_layer = self.get_norm_layer()
    if self.config.using_pipeline_parallelism:
      pipeline_stage_module = self.get_pipeline_stage_module(self.decoder_layer[0])
      remat_policy = self.get_remat_policy()
      self.pipeline_module = pipeline.Pipeline(
          config=self.config, mesh=self.mesh, layers=pipeline_stage_module, remat_policy=remat_policy
      )
    
    cfg = self.config
    if self.config.channel_gating:
      channel_gating_init_scale = 1 if self.config.channel_gating_init_scale is None else self.config.channel_gating_init_scale
      self.channel_gating = Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=initializers.constant_init(channel_gating_init_scale), # lsp
        name="channel_gating",
        config=cfg,
      )

    if self.config.vocab_gating:
      vocab_gating_init_scale = 1 if self.config.vocab_gating_init_scale is None else self.config.vocab_gating_init_scale 
      self.vocab_gating = Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.vocab_size,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=initializers.constant_init(vocab_gating_init_scale), # lsp
        name="vocab_gating",
        config=cfg,
      )

    if self.config.use_rins_linear_adapters:
      norm_layer_class = normalizations.Gpt3LayerNorm if self.config.rins_layer_norm else self.get_norm_layer()
      self.rins_norms = [norm_layer_class(
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        name=f"rins_norm_{pat_idx}",
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
    ) for pat_idx in range(len(self.config.skip_layers))]
      self.rins_linear_adapters = [linears.DenseGeneral(features=(self.config.emb_dim,),axis=-1, kernel_init=initializers.nd_dense_init_normal(0.006),
        kernel_axes=("embed", None), dtype=self.config.dtype, weight_dtype=self.config.weight_dtype,name=f"rins_linear_adapter_{pat_idx}",
        quant=self.quant, use_bias=False, matmul_precision=self.config.matmul_precision) for pat_idx in range(len(self.config.skip_layers))]
      # rins_rank = 128
      # # lora linear adapter for stochastic rins
      # self.rins_linear_adapters = [linears.MlpBlock(intermediate_dim=rins_rank, activations=['linear'],dtype=self.config.dtype,weight_dtype=self.config.weight_dtype,
      #         name=f"rins_linear_adapter_{pat_idx}",config=self.config,quant=self.quant,use_bias=False,kernel_init=initializers.nd_dense_init_normal(0.006),) for pat_idx in range(len(self.config.skip_layers))]

    if self.config.shift_last_hidden:
      self.hidden_shift = Hiddenshift(config=self.config,mesh=self.mesh, quant=self.quant, kernel_init=initializers.nd_dense_init_normal(0.006))
  
    if self.config.use_dynamic_temp: 
       self.dynamic_temp = DynamicTemperature(config=self.config,mesh=self.mesh, quant=self.quant, kernel_init=initializers.nd_dense_init_normal(0.006))

  def initialize_rins_adapters(self, dummy_input):
    """Explicitly initialize all RINS adapters to ensure parameter creation.
    
    Args:
      dummy_input: A dummy tensor with shape compatible with adapter input
    """
    if self.config.use_rins_linear_adapters and hasattr(self, 'rins_linear_adapters'):
      for adapter, norm in zip(self.rins_linear_adapters, self.rins_norms):
        _ = adapter(norm(dummy_input * 0.0))  # Zero-weight call to force parameter creation

  def initialize_mudd_compose_layers(self, y, cfg, layers_dict, decoder_segment_ids, decoder_positions, deterministic, model_mode):
    for pat_idx in range(len(cfg.skip_layers)):
      layer_skips = 0
      hids = [y]
      for lyr in range(cfg.num_decoder_layers):
        layer = layers_dict[cfg.recursive_pattern[lyr]]
        if cfg.skip_layers[pat_idx] is not None and lyr in cfg.skip_layers[pat_idx]:
          layer_skips += 1
          continue 
        lyr = lyr - layer_skips
        _ = layer(
            y,
            decoder_segment_ids,
            decoder_positions,
            deterministic,
            model_mode,
            (lyr, pat_idx),
            hids=hids,
        )
        hids.append(y) # dummy call to force parameter creation

  def get_remat_policy(self):
    cfg = self.config
    if cfg.remat_policy != "none":
      if cfg.remat_policy == "minimal":
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      elif cfg.remat_policy == "save_nothing":
        policy = jax.checkpoint_policies.nothing_saveable
      elif cfg.remat_policy == "save_dot_with_context_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "context",
            "out_proj",
        )
      elif cfg.remat_policy == "save_dot_except_mlpwi":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "out_proj",
            "mlpwo",
        )
      elif cfg.remat_policy == "save_dot_except_mlp":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
            "out_proj",
        )
      elif cfg.remat_policy == "save_qkv_proj":
        policy = jax.checkpoint_policies.save_only_these_names(
            "query_proj",
            "value_proj",
            "key_proj",
            "qkv_proj",
        )
      elif cfg.remat_policy == "qkv_proj_offloaded":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=[],
            names_which_can_be_offloaded=["query_proj", "value_proj", "key_proj"],
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "minimal_offloaded":
        policy = jax.checkpoint_policies.offload_dot_with_no_batch_dims(offload_src="device", offload_dst="pinned_host")
      elif cfg.remat_policy == "custom":
        policy = jax.checkpoint_policies.save_and_offload_only_these_names(
            names_which_can_be_saved=cfg.tensors_on_device,
            names_which_can_be_offloaded=cfg.tensors_to_offload,
            offload_src="device",
            offload_dst="pinned_host",
        )
      elif cfg.remat_policy == "minimal_flash":
        policy = jax.checkpoint_policies.save_from_both_policies(
            jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims,
            jax.checkpoint_policies.save_only_these_names(
                "context",
            ),
        )
      elif cfg.remat_policy == "save_out_proj":
        policy = jax.checkpoint_policies.save_only_these_names(
            "out_proj",
        )
      else:
        assert cfg.remat_policy == "full", "Remat policy needs to be on list of remat policies"
        policy = None
      return policy

  def set_remat_policy(self, block_layers, policy):
    RemattedBlockLayers = []
    static_argnums = (4,5,6) if (self.config.mudd_in_layer or self.config.sep_dc or self.config.mod_sparse_gate or self.config.lora_rank) and self.config.recursive_pattern else (4,5)
    # static_argnums = (4,5)
    for block_layer in block_layers:
      layer = nn.remat(  # pylint: disable=invalid-name
          block_layer,
          prevent_cse=not self.config.scan_layers,
          policy=policy,
          static_argnums=static_argnums,  # Deterministic and model mode are static arguments.
      )
      RemattedBlockLayers.append(layer)
    return RemattedBlockLayers

  def get_decoder_layers(self):
    if self.config.decoder_block == "default":
      return [DecoderLayer]
    elif self.config.decoder_block == "llama2":
      from layers import llama2

      return [llama2.LlamaDecoderLayer]
    elif self.config.decoder_block == "mistral":
      # TODO(ranran): update to Mistral with sliding window attention
      from layers import mistral

      return [mistral.MistralDecoderLayer]
    elif self.config.decoder_block == "deepseek":
      from layers import deepseek

      return [deepseek.DeepSeekDenseLayer, deepseek.DeepSeekMoELayer]
    elif self.config.decoder_block == "gemma":
      from layers import gemma

      return [gemma.GemmaDecoderLayer]
    elif self.config.decoder_block == "gemma2":
      from layers import gemma2

      return [gemma2.Gemma2DecoderLayer]
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return [gpt3.Gpt3DecoderLayer]
    elif self.config.decoder_block == "simple":
      from layers import simple_layer

      return [simple_layer.SimpleDecoderLayer]
    elif self.config.decoder_block == "simple_mlp":
      from layers import simple_layer

      return [simple_layer.SimpleMlpDecoderLayer]

    elif self.config.decoder_block == "fusion": # lsp
      from layers import fusion
      return [fusion.FusionDecoderLayer]

    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def get_norm_layer(self): # lsp
    if self.config.norm_type == 'layernorm':
      from layers import gpt3
      return gpt3.Gpt3LayerNorm
    elif self.config.norm_type == 'rmsnorm':
      return RMSNorm

    if self.config.decoder_block in ("default", "llama2", "mistral", "deepseek", "gemma", "gemma2", "simple", "simple_mlp", "fusion"):
      return RMSNorm
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=True)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def scan_decoder_layers(self, cfg, decoder_layer, length, metdata_axis_name, mesh):
    initializing = self.is_mutable_collection("params")
    params_spec = cfg.param_scan_axis if initializing else ScanIn(cfg.param_scan_axis)
    cache_spec = 0
    scan_fn = nn.scan(
        decoder_layer,
        variable_axes={
            "params": params_spec,
            "cache": cache_spec,
            "intermediates": 0,
            "aqt": 0,
            "_overwrite_with_gradient": 0,
        },
        split_rngs={
            "params": True,
            "dropout": cfg.enable_dropout,
        },
        in_axes=(
            nn.broadcast,
            nn.broadcast,
            nn.broadcast,
            nn.broadcast,
        ),
        length=length,
        metadata_params={nn.PARTITION_NAME: metdata_axis_name},
    )
    return scan_fn(config=cfg, mesh=mesh, name=metdata_axis_name, quant=self.quant)

  def get_pipeline_stage_module(self, base_stage):
    cfg = self.config
    if cfg.set_remat_policy_on_layers_per_stage:
      policy = self.get_remat_policy()
      base_stage = self.set_remat_policy([base_stage], policy)[0]
    if cfg.num_layers_per_pipeline_stage == 1:
      stage_module = base_stage(config=cfg, mesh=self.mesh, quant=self.quant)
    elif cfg.scan_layers:
      stage_module = self.scan_decoder_layers(
          cfg, base_stage, cfg.num_layers_per_pipeline_stage, "layers_per_stage", self.mesh
      )
    else:
      stage_module = SequentialBlockDecoderLayers(
          decoder_layer=base_stage,
          num_decoder_layers=cfg.num_layers_per_pipeline_stage,
          config=cfg,
          mesh=self.mesh,
          quant=self.quant,
      )
    return stage_module

  @nn.compact
  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      skip_layers=None,
      decoder_segment_ids=None,
      deterministic=False,
      model_mode=common_types.MODEL_MODE_TRAIN,
  ):
    cfg = self.config
    mesh = self.mesh
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype("int32"))
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if cfg.use_untrainable_positional_embedding:
      y = PositionalEmbedding(cfg.base_emb_dim)(y, decoder_positions)

    if cfg.trainable_position_size > 0:
      y += Embed(
          num_embeddings=cfg.trainable_position_size,
          features=cfg.emb_dim,
          dtype=cfg.dtype,
          embedding_init=nn.initializers.normal(stddev=1.0),
          name="position_embedder",
          config=cfg,
      )(decoder_positions)

    if cfg.dense_conn: # lsp
      if cfg.mudd_prenorm:
        assert cfg.ddw_gen_pattern == 'q,k,v,m', max_logging.log(f'Error: ddw_gen_pattern must be ‘q,k,v,m’ when mudd_prenorm is true.')
        y_normed = normalizations.get_rmsnorm(name="mudd_prenorm", cfg=cfg)(y)
      else:
        y_normed = y
      if cfg.mudd_in_layer:
        y, hids = y, [y_normed]
      else:
        y, hids = [y] * len(cfg.dynamic_dense_type), [y_normed]
    else:
      hids = []

    # Force initialization of RINS adapters early in the forward pass
    if cfg.use_rins_linear_adapters and self.is_mutable_collection('params'):
      self.initialize_rins_adapters(y)

    policy = self.get_remat_policy()
    RemattedBlockLayers = self.set_remat_policy(self.decoder_layer, policy)

    if cfg.using_pipeline_parallelism:
      if cfg.pipeline_fsdp_ag_once:
        partition_spec = self.pipeline_module.get_weight_sharding(
            y, decoder_segment_ids, decoder_positions, deterministic, model_mode
        )
      else:
        partition_spec = None  # This partition spec is only used for the fsdp_ag_once feature.
      y = self.pipeline_module(
          y, decoder_segment_ids, decoder_positions, deterministic, model_mode, partition_spec=partition_spec
      )
    else:
      if cfg.scan_layers:
        if cfg.decoder_block == "deepseek":
          assert len(RemattedBlockLayers) == 2, f"Scanned layers must have a length of 2 using deepseek."
          dense_layer = RemattedBlockLayers[0]
          moe_layer = RemattedBlockLayers[1]
          y, _ = self.scan_decoder_layers(cfg, dense_layer, cfg.first_num_dense_layers, "dense_layers", mesh)(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
          )
          num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
          y, _ = self.scan_decoder_layers(cfg, moe_layer, num_moe_layers, "moe_layers", mesh)(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
          )
        else:
          RemattedBlockLayer = RemattedBlockLayers[0]
          y, _ = self.scan_decoder_layers(cfg, RemattedBlockLayer, cfg.num_decoder_layers // cfg.num_layers_per_block, "layers", mesh)(
              y,
              decoder_segment_ids,
              decoder_positions,
              deterministic,
              model_mode,
              None,
          )
      else:
        if cfg.decoder_block == "deepseek":
          assert len(RemattedBlockLayers) == 2, f"Unscanned layers must have a length of 2 using deepseek."
          dense_layer = RemattedBlockLayers[0]
          moe_layer = RemattedBlockLayers[1]
          num_moe_layers = cfg.num_decoder_layers - cfg.first_num_dense_layers
          layers = [dense_layer, moe_layer]
          layer_prefix = ["dense_layers", "moe_layers"]
          num_layers = [cfg.first_num_dense_layers, num_moe_layers]
          for index in range(len(layers)):
              for index_j in range(num_layers[index]):
                        y = layers[index](config=cfg, mesh=mesh, name=f"{layer_prefix[index]}_{index_j}", quant=self.quant)(
                            y,
                            decoder_segment_ids,
                            decoder_positions,
                            deterministic,
                            model_mode,
                        )
        elif cfg.decoder_block == "fusion" and cfg.recursive_pattern: #mqy
          RemattedBlockLayer = RemattedBlockLayers[0]
          pat = cfg.recursive_pattern
          assert len(pat) == cfg.num_decoder_layers # 'ABC' * 8, ''
          if cfg.lora_layers is not None: # lora_layers and sep_dc cannot be used together
            assert not cfg.sep_dc 
          layers_dict = dict([(layer_sym, RemattedBlockLayer(config=cfg, mesh=mesh, name=f"layers_{layer_sym}", quant=self.quant, 
               sliding_window_size=cfg.sliding_window_size[pat.index(layer_sym) % len(cfg.sliding_window_size)] if isinstance(cfg.sliding_window_size, list) else None)) for layer_sym in set(pat)])
          if cfg.mudd_in_layer and cfg.dense_conn and cfg.skip_layers is not None and self.is_mutable_collection('params'):
            self.initialize_mudd_compose_layers(y, cfg, layers_dict, decoder_segment_ids, decoder_positions, deterministic, model_mode)
          layer_skips = 0
          for lyr in range(cfg.num_decoder_layers):
            layer = layers_dict[pat[lyr]]
            pat_idx = cfg.skip_layers.index(skip_layers) if cfg.skip_layers is not None else 0
            if skip_layers is not None and lyr in skip_layers:
              max_logging.log(f'skipping layer {lyr}, skip_layers: {skip_layers}')
              layer_skips += 1
              continue 
            lyr = lyr - layer_skips
            y = layer(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
                (lyr, pat_idx),
                hids=hids,
            )
            if self.config.mudd_in_layer:
              y, hids = y[:2]
            else:
              y = y[0]
            
          # BUG
          if cfg.use_rins_linear_adapters:
            pat_idx = cfg.skip_layers.index(skip_layers)
            y = y + self.rins_linear_adapters[pat_idx](self.rins_norms[pat_idx](y))

        else:
          n = cfg.num_decoder_layers // cfg.num_layers_per_block
          sliding_window_sizes = n * cfg.sliding_window_size if isinstance(cfg.sliding_window_size, list) else n * [cfg.sliding_window_size]
          max_logging.log(f'sliding_window_sizes: {sliding_window_sizes}', debug=cfg.debug)
          value_residual = None
          for lyr in range(cfg.num_decoder_layers):
            RemattedBlockLayer = RemattedBlockLayers[0]
            y = RemattedBlockLayer(config=cfg, mesh=mesh, name=f"layers_{lyr}", quant=self.quant, sliding_window_size=sliding_window_sizes[lyr])(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
                None,
                hids=hids,
                value_residual=value_residual,
            )
            # if self.config.value_residual_learning:
            y, value_residual, attn_out = y[:-2], y[-2], y[-1]
            if self.config.mudd_in_layer:
                y, hids = y
            if self.config.record_internal_nn_metrics:
              self.sow('intermediates', f'hidden_states_layer_{lyr}', maxtext_utils.l2norm(y[0])) # layer_out, dyn_dense_w = y
            if not self.config.mudd_in_layer:
              y, hids = mudd.Compose(cfg, mesh, self.quant, lyr, name=f'compose_{lyr}')(y, hids, attn_out=attn_out) # lsp

    if self.config.shift_last_hidden:
      y = self.hidden_shift(y)

    last_hid = y         
    y = self.get_norm_layer()(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="decoder_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )(y)

    if self.config.use_dynamic_temp: 
      y = self.dynamic_temp(last_hid, y)

    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)

    if self.config.channel_gating:
      channel_gating_norm = self.get_norm_layer()(
        dtype=cfg.dtype,
        weight_dtype=cfg.weight_dtype,
        name="channel_gating_norm",
        epsilon=cfg.normalization_layer_epsilon,
        kernel_axes=("norm",),
      )
      if self.config.channel_gating_norm:
        y = y + channel_gating_norm(y * self.channel_gating(decoder_input_tokens.astype("int32")))
      else:
        y = y * self.channel_gating(decoder_input_tokens.astype("int32"))

    # [batch, length, emb_dim] -> [batch, length, vocab_size]
    if cfg.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = self.shared_embedding.attend(y)
      if self.config.normalize_embedding_logits:
        # Correctly normalize pre-softmax logits for this shared case.
        logits = logits / jnp.sqrt(y.shape[-1])
      if cfg.final_logits_soft_cap:
        logits = logits / cfg.final_logits_soft_cap
        logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
      logits = linears.DenseGeneral(
          cfg.vocab_size,
          weight_dtype=cfg.weight_dtype,
          dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
          kernel_axes=("embed", "vocab"),
          name="logits_dense",
          matmul_precision=self.config.matmul_precision,
          kernel_init=initializers.nd_dense_init_normal(0.006) if not cfg.neox_init else initializers.nd_dense_init_normal(math.sqrt(2/(5*cfg.emb_dim))), #lsp
      )(
          y
      )  # We do not quantize the logits matmul.
    max_logging.log(f'logits: {logits.shape}', debug=cfg.debug)
    logits = nn.with_logical_constraint(
        logits, ("activation_embed_and_logits_batch", "activation_length", "activation_vocab")
    )
    
    if self.config.vocab_gating:
      # logits = logits * self.vocab_gating(decoder_input_tokens.astype("int32"))
      logits = logits + self.vocab_gating(decoder_input_tokens.astype("int32"))


    if self.config.cast_logits_to_fp32:
      logits = logits.astype(jnp.float32)
    return logits


class Transformer(nn.Module):
  """An decoder-only Transformer model."""

  # Make new attributes required, so that all Transformer dependencies (train, decode, compile, etc) will error instead of silently use defaults.
  # pylint: disable=attribute-defined-outside-init
  config: Config
  mesh: Mesh
  quant: Quant

  def setup(self):
    """Initialize shared_embedding & decoder layers."""

    cfg = self.config
    mesh = self.mesh
    self.shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        features=cfg.emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=initializers.nd_dense_init_normal(0.006) if not cfg.neox_init else initializers.nd_dense_init_normal(math.sqrt(2/(5*cfg.emb_dim))), # lsp
        name="token_embedder",
        config=cfg,
    )

    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding, mesh=mesh, quant=self.quant)

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      skip_layers=None,
      decoder_segment_ids=None,
      enable_dropout=True,
      model_mode=common_types.MODEL_MODE_TRAIN,
  ):
    """Applies Transformer decoder-branch on encoded-input and target."""

    if decoder_segment_ids is not None and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      raise ValueError(
          f"During autoregressive decoding we assume the tokens are in the active sequence"
          f" which is always {common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR}."
      )

    logits = self.decoder(
        decoder_input_tokens=decoder_input_tokens,
        decoder_positions=decoder_positions,
        skip_layers=skip_layers,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=not enable_dropout,
        model_mode=model_mode,
    )
    return logits
