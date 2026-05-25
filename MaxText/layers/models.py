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
from layers import initializers
import max_logging
import re
import max_utils
from layers import mtp
from layers import mudd
from layers import engram
from layers.kv_shift import shift_1d
from collections import defaultdict
from functools import partial

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh
ScanIn = common_types.ScanIn

Embed = embeddings.Embed
Attention = attentions.Attention
PositionalEmbedding = embeddings.PositionalEmbedding
Quant = quantizations.AqtQuantization

# ------------------------------------------------------------------------------
# The network: Decoder & Transformer Definitions
# ------------------------------------------------------------------------------

def get_deep_embedding(cfg, deep_embedding):
  total_de_dim = deep_embedding.shape[-1]
  start_idx = 0
  deep_embeddings = []
  max_logging.log(f'Use outside DE, deep_embed_type: {cfg.deep_embed_type}', debug=cfg.debug)
  if '4xmlp' in cfg.deep_embed_type:
    assert not cfg.scan_layers, f'dynamic_mlp_dim is not supported with scan_layers'
    num_decoder_layers = cfg.num_decoder_layers if cfg.deep_embed_effective_layers is None else cfg.deep_embed_effective_layers
    for layer_inx in range(num_decoder_layers):
      # updated_mlp_dim = round(cfg.mlp_dim * (layer_inx / (cfg.num_decoder_layers - 1) + 0.5) / 128) * 128 if cfg.dynamic_mlp_dim else cfg.mlp_dim
      updated_mlp_dim = cfg.mlp_dim # dynamic_mlp_dim loss higher than static mlp_dim, about 0.003 gap
      d1 = 32 if updated_mlp_dim < 4096 else 64
      d2 = updated_mlp_dim // d1
      end_idx = start_idx + updated_mlp_dim
      assert end_idx <= total_de_dim, f'end_idx: {end_idx} > total_de_dim: {total_de_dim}'
      cur_layer_deep_embedding = deep_embedding[..., start_idx: end_idx]
      start_idx += updated_mlp_dim
      if cfg.deep_embed_effective_layers is None:
        cur_layer_deep_embedding = cur_layer_deep_embedding.reshape(*deep_embedding.shape[:2], d1, d2)
      deep_embeddings.append(cur_layer_deep_embedding)
    deep_embeddings = jnp.stack(deep_embeddings, axis=0)
    max_logging.log(f'4xmlp deep_embeddings: {deep_embeddings.shape}', debug=cfg.debug)
  elif 'devalue' in cfg.deep_embed_type.lower():
    for layer_inx in range(cfg.num_decoder_layers):
      num_kv_heads = cfg.num_kv_heads[layer_inx % len(cfg.num_kv_heads)] \
      if isinstance(cfg.num_kv_heads, list) else cfg.num_kv_heads
      de_dim = num_kv_heads * cfg.head_dim
      d1 = 32 if de_dim < 4096 else 64
      d2 = de_dim // d1
      end_idx = start_idx + de_dim
      assert end_idx <= total_de_dim, f'end_idx: {end_idx} > total_de_dim: {total_de_dim}'
      cur_layer_deep_embedding = deep_embedding[..., start_idx: end_idx]
      start_idx += de_dim
      cur_layer_deep_embedding = cur_layer_deep_embedding.reshape(*deep_embedding.shape[:2], d1, d2)
      deep_embeddings.append(cur_layer_deep_embedding)
  elif re.findall(r'1xmlp|1xattn', cfg.deep_embed_type):
    d1 = 32 if cfg.emb_dim < 4096 else 64
    d2 = cfg.emb_dim // d1
    deep_embeddings = deep_embedding.reshape(*deep_embedding.shape[:2], cfg.num_decoder_layers, d1, d2).transpose(2, 0, 1, 3, 4)
    max_logging.log(f'1xmlp|1xattn deep_embeddings: {deep_embeddings.shape}', debug=cfg.debug)
  else:
    raise ValueError(f'Invalid deep_embed_type: {cfg.deep_embed_type}')
  return deep_embeddings


def get_remat_policy(cfg):
  if cfg.remat_policy != "none":
    if cfg.remat_policy == "minimal":
      policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
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
    elif cfg.remat_policy == "save_all":
      policy = jax.checkpoint_policies.save_only_these_names(
          "query_proj",
          "value_proj",
          "key_proj",
          "qkv_proj",
          "out_proj",
          "mlpwo",
          "context",
          "mlpwi_0",
          "mlpwi_1",
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
  

def get_group_index(n, groups, i, include_first_L=True):
  '''
  仅适用LGLL的window:
  n: 总层数;
  groups: dilation;
  i: 当前层索引;
  include_first_L: 是否包含第一层L;
  '''
  cll = (n - 1) // 4
  if not include_first_L and i == 0:
      return -1
  if i >= 3 and i % 4 == 3 and (i - 3) // 4 < cll:
      return -1
  if i >= 4 and i % 4 == 0 and i // 4 - 1 < cll:
      return -1
  d3 = min(cll, (i - 4) // 4 + 1) if i > 3 else 0
  d4 = min(cll, (i - 5) // 4 + 1) if i > 4 else 0
  deleted = d3 + d4
  if not include_first_L:
      deleted += 1
  rem_idx = i - deleted
  total = n - 2 * cll - (0 if include_first_L else 1)
  base, r = divmod(total, groups)
  thresh = r * (base + 1)
  return rem_idx // (base + 1) if rem_idx < thresh else r + (rem_idx - thresh) // max(1, base)


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
    lnx = normalizations.get_rmsnorm("pre_self_attention_norm", cfg)(inputs)

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


class OutputHead(nn.Module):

  config: Config
  shared_embedding: nn.Module
  mesh: Mesh
  quant: Optional[Quant] = None

  def setup(self):
    cfg = self.config
    self.norm = normalizations.get_rmsnorm("decoder_norm", cfg)
    dense_kernel = self.param(
      "logits_dense",
      nn.with_logical_partitioning(initializers.get_init_method(cfg.init_method), ("embed", "vocab")),
      (cfg.emb_dim, cfg.vocab_size),
      cfg.weight_dtype,
      ) # int use 888 more quick，but more memory
    self.logits_dense = dense_kernel.astype(cfg.dtype)
    self.dropout = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))

  def project_logits(self, inputs: Array) -> Array:
    cfg = self.config
    dense_kernel = nn.with_logical_constraint(self.logits_dense, (None, "vocab"))
    if cfg.logits_via_embedding:
      logits = self.shared_embedding.attend(inputs)
      if cfg.normalize_embedding_logits:
        logits = logits / jnp.sqrt(inputs.shape[-1])
      if cfg.final_logits_soft_cap:
        logits = logits / cfg.final_logits_soft_cap
        logits = jnp.tanh(logits) * cfg.final_logits_soft_cap
    else:
      logits = jnp.tensordot(inputs, dense_kernel, axes=((-1,), (0,)))
    return logits

  def logits_from_hidden_states(
      self,
      hidden_states: Array,
      deterministic: bool = True,
      mtp_layer: bool = False,
  ) -> Array:
    inputs = hidden_states
    if not mtp_layer:
      inputs = self.norm(inputs)
      inputs = self.dropout(inputs, deterministic=deterministic)
    return self.project_logits(inputs)
  
  @nn.compact
  def __call__(
    self,
    inputs: Array,
    target_tokens: Array,
    target_mask: Array,
    chunk_size: int = 1024,
    deterministic: bool = True,
    mtp_layer: bool = False,
  ) -> tuple[Array, Array, Array]:

    cfg = self.config
    seq_len = inputs.shape[1]
    xents = []
    correct = 0
    preds = []
    logits = []
    # chunk_size = cfg.loss_chunk_size  # manual shard to speed up, about 1%
    if not mtp_layer:
      inputs = self.norm(inputs)
      inputs = self.dropout(inputs, deterministic=deterministic)

    for start_idx in range(0, seq_len, chunk_size):
      end_idx = min(start_idx + chunk_size, seq_len)
      print(f'lm head chunk start_idx: {start_idx} end_idx: {end_idx}')
      chunk_slice = slice(start_idx, end_idx)
      logits_chunk = self.project_logits(inputs[:, chunk_slice])
      if cfg.return_logits:
        logits.append(logits_chunk)
      preds_chunk = jnp.argmax(logits_chunk, axis=-1)
      mask_chunk = target_mask[:, chunk_slice]
      targets_chunk = target_tokens[:, chunk_slice]
      one_hot_targets_chunk = jax.nn.one_hot(targets_chunk, cfg.vocab_size) # must chunk
      predictions_match = jnp.argmax(logits_chunk, axis=-1) == targets_chunk
      correct_chunk = jnp.where(mask_chunk > 0.0, predictions_match, jnp.array(False))
      correct += jnp.sum(correct_chunk)
      xent, _ = max_utils.cross_entropy_with_logits(logits_chunk, one_hot_targets_chunk, 0.0)
      xent = nn.with_logical_constraint(
          xent, ("activation_embed_and_logits_batch", "activation_length")
      )
      xents.append(xent)
      preds.append(preds_chunk)
    xents = jnp.concatenate(xents, axis=1)
    preds = jnp.concatenate(preds, axis=1)
    if cfg.return_logits:
      return jnp.concatenate(logits, axis=1), correct, preds
    return xents, correct, preds


class Decoder(nn.Module):
  """A stack of decoder layers as a part of an encoder-decoder architecture."""

  config: Config
  shared_embedding: nn.Module
  deep_embedding: nn.Module
  mesh: Mesh
  quant: Optional[Quant] = None

  def setup(self):
    """Initialize decoder layer."""
    self.decoder_layer = self.get_decoder_layers()
    if self.config.using_pipeline_parallelism:
      pipeline_stage_module = self.get_pipeline_stage_module(self.decoder_layer[0])
      remat_policy = get_remat_policy(self.config)
      self.pipeline_module = pipeline.Pipeline(
          config=self.config, mesh=self.mesh, layers=pipeline_stage_module, remat_policy=remat_policy
      )

  def set_remat_policy(self, block_layers, policy):
    RemattedBlockLayers = []
    for block_layer in block_layers:
      layer = nn.remat(
          block_layer,
          prevent_cse=True, #lsp: scan_layers is True, 该参数务必注意，设置为False可能会卡住
          policy=policy,
          static_argnums=(6, 7),
          rngs={"params": True, "aqt": True, "dropout": True},
      )
      layer1 = nn.remat(
          block_layer,
          prevent_cse=True,#lsp: scan_layers is False，该参数务必注意，设置为False可能会卡住
          policy=policy,
          static_argnums=(6, 7),
          rngs={"params": True, "aqt": True, "dropout": True},
      )
      RemattedBlockLayers.append(layer1)
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

    elif self.config.decoder_block == "llada":
      from layers import llada
      return [llada.LLaDADecoderLayer]

    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def scan_decoder_layers(self, cfg, decoder_layer, length, metdata_axis_name, mesh, 
   sliding_window_size=None, scan_length=1):
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
            0 if cfg.deep_embed_effective_layers is None else nn.broadcast, # deep_embedding
            nn.broadcast,
            nn.broadcast, # hids
            nn.broadcast, # 关键字参数不在这个范围内
        ),
        length=length,
        metadata_params={nn.PARTITION_NAME: metdata_axis_name},
    )
    return scan_fn(config=cfg, mesh=mesh, name=metdata_axis_name, quant=self.quant, 
      sliding_window_size=sliding_window_size, scan_length=scan_length)

  def get_pipeline_stage_module(self, base_stage):
    cfg = self.config
    if cfg.set_remat_policy_on_layers_per_stage:
      policy = get_remat_policy(cfg)
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
  def logits_from_hidden_states(
      self,
      hidden_states,
      deterministic,
      mtp_layer: bool = False,
  ):
    output_head_layer = OutputHead(
        config=self.config,
        shared_embedding=self.shared_embedding,
        mesh=self.mesh,
        quant=self.quant,
        name="lm_head",
    )
    return output_head_layer.logits_from_hidden_states(
        hidden_states,
        deterministic=deterministic,
        mtp_layer=mtp_layer,
    )

  @nn.compact
  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions, # trained position or rotary position
      decoder_target_tokens,
      decoder_target_mask,
      decoder_segment_ids=None, # mask
      deterministic=False,
      model_mode=common_types.MODEL_MODE_TRAIN,
  ):
    cfg = self.config
    mesh = self.mesh
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    if cfg.mix_attn and decoder_segment_ids is not None:
      # ======================================32k long context max window size set==================================================
      eos_sum = (decoder_segment_ids == 0).sum(1)  # 3.5 mini train
      max_logging.log(f'[lsp]decoder_segment_ids: {decoder_segment_ids.shape}', debug=cfg.debug)
      # eos_sum = (decoder_input_tokens == 151643).sum(1) # v4 moe
      eos_sum = jnp.where(eos_sum > 0, 1, 0) # batch
      if cfg.record_internal_nn_metrics:
        self.sow("intermediates", "eos_sum_mean", eos_sum.mean(), ) # 每个batch带有eos数据的比例
        self.sow("intermediates", "eos_sum", eos_sum.sum(), ) # batch总的eos数量
    else:
      eos_sum = None

    # [batch, length] -> [batch, length, emb_dim]
    y = self.shared_embedding(decoder_input_tokens.astype("int32"))
    y, mtp_de = jnp.split(y, [cfg.emb_dim], axis=-1)
    max_logging.log(f'mtp_de: {mtp_de}', debug=cfg.debug)
    # mtp need to roll_input_ids embedding, it can reduce 12ms time to extract in here.
    rolled_y = jnp.pad(y[:, 1:], ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)
    y = y.astype(cfg.dtype)

    if cfg.deep_embed_init == 'outside' and re.findall(r'1xmlp|1xattn|4xmlp|devalue', cfg.deep_embed_type):
      deep_embeddings = self.deep_embedding(decoder_input_tokens.astype("int32"))
      deep_embeddings = get_deep_embedding(cfg, deep_embeddings)
      max_logging.log(f'deep_embeddings: {deep_embeddings[0].shape} length:{len(deep_embeddings)} y: {y.shape}', debug=cfg.debug)
    
    else:
      deep_embeddings = None if cfg.scan_layers or cfg.partial_scan_layers else [None] * cfg.num_decoder_layers

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

    hids = []
    if cfg.dense_conn and not cfg.mudd_in_layer:
      max_logging.log(f'Outside layers don\'t use remat', debug=cfg.debug)
      RemattedBlockLayers = self.decoder_layer * 2
      # assert not cfg.partial_scan_layers, f'partial_scan_layers is not supported with mudd_in_layer=False'
    else:
      max_logging.log(f'Outside layers use remat', debug=cfg.debug)
      RemattedBlockLayers = self.set_remat_policy(self.decoder_layer, get_remat_policy(cfg)) 

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
      if not isinstance(cfg.sliding_window_size, list):
        sws_list = [cfg.sliding_window_size]
      else:
        sws_list = cfg.sliding_window_size

      def format_swss(sws_list):
        sws = [cfg.max_target_length if s is None else s for s in sws_list]
        if len(sws) == cfg.num_decoder_layers + cfg.mtp_num_layers:
          return sws
        sws = sws * (cfg.num_decoder_layers // len(sws) + 1)
        sws = sws[:cfg.num_decoder_layers]
        if cfg.mtp_num_layers > 0:
          sws += sws[-1:]  # mtp layer's sws must be the same as the last layer
        return sws

      def split_layer_ranges_uniform(total_layers, num_groups):
        base = total_layers // num_groups
        remainder = total_layers % num_groups
        ranges = []
        start = 0
        for i in range(num_groups):
            size = base + 1 if i < remainder else base
            end = start + size
            ranges.append((start, end))
            start = end
        return ranges

      print(f'Using uniform split method')
      split_layer_ranges = split_layer_ranges_uniform

      if cfg.use_compressed_vocab:
        compressed_vocab_lookup = engram.CompressedVocabLookup(config=cfg, name="compressed_vocab_lookup")
        compressed_decoder_input_tokens, compressed_vocab_size = compressed_vocab_lookup(decoder_input_tokens.astype(jnp.int32))
        max_logging.log(f'Using compressed vocab for engram, compressed_vocab_size: {compressed_vocab_size}', debug=cfg.debug)
        vocab_size = compressed_vocab_size
      else:
        compressed_decoder_input_tokens = decoder_input_tokens
        vocab_size = cfg.vocab_size

      if cfg.engram_sizes_layers:
        engram_embeddings = defaultdict(list)
        engram_nums_per_group = defaultdict(int)
        for _, (ngram_size, ngram_layers) in enumerate(cfg.engram_sizes_layers):
          for engram_idx in range(ngram_layers):
              engram_layer = engram.EngramNGram(
                name=f"engram_{ngram_size}_{engram_idx}", 
                config=cfg, 
                engram_idx=engram_idx, 
                ngram_size=ngram_size,
                compressed_vocab_size=vocab_size,
                )
              engram_embed = engram_layer(compressed_decoder_input_tokens)
              engram_embeddings[ngram_size].append(engram_embed)
          ngram_nums = len(engram_embeddings[ngram_size])
          engram_nums_per_group[ngram_size] = split_layer_ranges(ngram_nums, cfg.me_dilation)
          max_logging.log(f'Engram embeddings-{ngram_size}: {ngram_nums} layers-0 shape: {engram_embeddings[ngram_size][0].shape}', debug=cfg.debug)
      else:
        engram_embeddings = None
      
      me = []
      if cfg.me_nums is not None:
        # Pad vocab_size for embedding table to be divisible by TP degree (vocab axis sharding).
        # Compressed vocab sizes (e.g. 107667) may not be divisible by ici_tensor_parallelism.
        # Extra padded rows are never accessed since token indices < original vocab_size.
        _tp_degree = (
            getattr(cfg, 'ici_tensor_parallelism', 1)
            * getattr(cfg, 'ici_tensor_transpose_parallelism', 1)
            * getattr(cfg, 'ici_tensor_sequence_parallelism', 1)
            * getattr(cfg, 'ici_autoregressive_parallelism', 1)
        )
        _tp_degree = max(_tp_degree, 1)
        padded_vocab_size = ((vocab_size + _tp_degree - 1) // _tp_degree) * _tp_degree
        mudd_emb = Embed(
            num_embeddings=padded_vocab_size,
            features=cfg.emb_dim * cfg.me_nums,
            dtype=cfg.dtype,
            attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,
            embedding_init=initializers.get_init_method(cfg.init_method), # lsp
            name="mudd_embedder",
            config=cfg,
        )
        me_rangs_per_group = split_layer_ranges(cfg.me_nums, cfg.me_dilation)
        me_group_cache = {}

        def load_me_group(group_idx):
          if group_idx < 0:
            return []
          if group_idx not in me_group_cache:
            start, end = me_rangs_per_group[group_idx]
            start_col = start * cfg.emb_dim
            end_col = end * cfg.emb_dim
            group_mes = jnp.asarray(mudd_emb.embedding, cfg.dtype)[:, start_col:end_col]
            group_mes = group_mes[compressed_decoder_input_tokens.astype("int32")]
            group_mes = nn.with_logical_constraint(
                group_mes, ("activation_embed_and_logits_batch", "activation_length", "activation_embed")
            )
            cached_group = []
            for i in range(start, end):
              me_item = group_mes[:, :, (i - start) * cfg.emb_dim:(i - start + 1) * cfg.emb_dim]
              if cfg.me_prenorm:
                max_logging.log(f'me_prenorm is true.', debug=cfg.debug)
                me_item = normalizations.get_rmsnorm(f"me_prenorm_{i}", cfg)(me_item)
              cached_group.append(me_item)
            me_group_cache[group_idx] = cached_group
          return me_group_cache[group_idx]

      if cfg.dense_conn:
        y = normalizations.get_rmsnorm("mudd_prenorm", cfg)(y) if cfg.mudd_prenorm or cfg.me_prenorm else y
        hids.append(y)

      if getattr(cfg, "recurrent_block_repeats", 1) > 1:
        assert not cfg.scan_layers, "Shared recurrent layers are only supported with scan_layers=False."
        assert not cfg.partial_scan_layers, "Shared recurrent layers are not supported with partial_scan_layers."
        assert not cfg.using_pipeline_parallelism, "Shared recurrent layers are not supported with pipeline parallelism."
        assert not cfg.dense_conn, "Shared recurrent layers are not supported with dense_conn."
        assert cfg.mtp_num_layers == 0, "Shared recurrent layers are not supported with MTP."

        physical_layers = cfg.recurrent_physical_num_layers or cfg.base_num_decoder_layers
        recurrent_start = cfg.recurrent_layer_start
        recurrent_end = cfg.recurrent_layer_end
        recurrent_repeats = cfg.recurrent_block_repeats
        assert 0 <= recurrent_start < recurrent_end <= physical_layers, (
            "Layer recurrence requires 0 <= recurrent_layer_start < recurrent_layer_end "
            "<= recurrent_physical_num_layers/base_num_decoder_layers."
        )

        layer_order = (
            list(range(recurrent_start))
            + list(range(recurrent_start, recurrent_end)) * recurrent_repeats
            + list(range(recurrent_end, physical_layers))
        )
        assert len(layer_order) == cfg.num_decoder_layers, (
            f"Recurrent layer order has {len(layer_order)} virtual layers, "
            f"but config.num_decoder_layers={cfg.num_decoder_layers}."
        )
        max_logging.log(f"recurrent layer order: {layer_order}", debug=cfg.debug)

        swss = format_swss(sws_list)
        layer_sws = {}
        for virtual_lyr, physical_lyr in enumerate(layer_order):
          if physical_lyr in layer_sws:
            assert layer_sws[physical_lyr] == swss[virtual_lyr], (
                "A shared recurrent layer cannot be called with different sliding_window_size values."
            )
          else:
            layer_sws[physical_lyr] = swss[virtual_lyr]

        RecurrentBlockLayer = RemattedBlockLayers[0]
        recurrent_layers = {
            physical_lyr: RecurrentBlockLayer(
                config=cfg,
                mesh=mesh,
                name=f"layers_{physical_lyr}",
                quant=self.quant,
                sliding_window_size=layer_sws[physical_lyr],
            )
            for physical_lyr in sorted(set(layer_order))
        }

        for virtual_lyr, physical_lyr in enumerate(layer_order):
          max_logging.log(
              f"\n=================decoder virtual layer: {virtual_lyr} "
              f"(params layers_{physical_lyr})=====================\n",
              debug=cfg.debug,
          )
          layer_output = recurrent_layers[physical_lyr](
              y,
              decoder_segment_ids,
              decoder_positions,
              decoder_input_tokens,
              deep_embeddings[virtual_lyr],
              deterministic,
              model_mode,
              hids=hids,
              eos_sum=eos_sum,
          )
          if isinstance(layer_output, tuple):
            y = layer_output[0]
            if len(layer_output) > 1 and layer_output[1] is not None:
              hids = layer_output[1]
          else:
            y = layer_output

      elif cfg.scan_layers:
        RemattedBlockLayer = RemattedBlockLayers[1]
        y, _ = self.scan_decoder_layers(cfg, RemattedBlockLayer, cfg.num_decoder_layers, "layers", mesh)(
            y,
            decoder_segment_ids,
            decoder_positions,
            decoder_input_tokens,
            deep_embeddings,
            deterministic,
            model_mode,
            None,
            eos_sum=eos_sum,
        )

      elif cfg.partial_scan_layers:

        swss = format_swss(sws_list)
        max_logging.log(f'[partial_scan_layers] roll_window_size: {cfg.roll_sws} swss: {swss}', debug=cfg.debug)
        lyr = 0
        while lyr < cfg.num_decoder_layers:
          current_sws = swss[lyr]
          max_logging.log(f'\n=================partial scan layers: {lyr}=====================\n', debug=cfg.debug)
          scan_start = lyr
          scan_length = 1
          if cfg.scan_use_mudd:
            # 连续的2个及以上相同sws的层组成一个scan层
            while (lyr + scan_length < cfg.num_decoder_layers and swss[lyr + scan_length] == current_sws):
              scan_length += 1
          else:
            # L, G, L, LL, G, L, LL
            if lyr > 0 and swss[lyr - 1] != current_sws:
              scan_length = 1
            else:
              while (lyr + scan_length < cfg.num_decoder_layers and swss[lyr + scan_length] == current_sws):
                scan_length += 1

          if cfg.mudd_local_window and current_sws != cfg.max_target_length:
            start_index = -cfg.mudd_local_window
          else:
            start_index = -100

          if scan_length == 1:
            max_logging.log(f'Processing layer {lyr} individually with sws={current_sws}', debug=cfg.debug)
            me_group_idx = -1
            me = []
            if cfg.me_nums and swss[1] == cfg.max_target_length: # LGLL
              me_group_idx = get_group_index(cfg.num_decoder_layers, cfg.me_dilation, lyr, include_first_L=False)
              if me_group_idx >= 0:
                me = list(load_me_group(me_group_idx))
                if engram_embeddings is not None:
                  for ngram_size, engram_embeds in engram_embeddings.items():
                    egram_rangs_per_group = engram_nums_per_group[ngram_size]
                    engram_embed = engram_embeds[egram_rangs_per_group[me_group_idx][0]: egram_rangs_per_group[me_group_idx][1]]
                    me.extend(engram_embed)
            max_logging.log(f'me_len: {len(me)} me_group_idx: {me_group_idx}', debug=cfg.debug)
            if deep_embeddings is None:
              de = None
            elif cfg.deep_embed_effective_layers is not None:
              de = deep_embeddings
            else:
              de = deep_embeddings[scan_start][None]
            de = jnp.stack(de, axis=0) if de is not None and isinstance(de, list) else de
            max_logging.log(f'de: {de}', debug=cfg.debug)
            c_hids = hids
            y, _ = self.scan_decoder_layers(
                cfg, 
                RemattedBlockLayers[1], 
                scan_length, 
                f"layers_{scan_start}", 
                mesh,
                sliding_window_size=current_sws,
                scan_length=scan_length,
            )(
                y,
                decoder_segment_ids,
                decoder_positions,
                decoder_input_tokens,
                de, # scan need to add a dimension
                deterministic,
                model_mode,
                me + c_hids[start_index:],
                eos_sum=eos_sum,
            )
            lyr += 1

          else:
            max_logging.log(f'Scanning layers {scan_start} to {scan_start + scan_length - 1} with sws={current_sws}', debug=cfg.debug)
            if deep_embeddings is None:
              de = None
            elif cfg.deep_embed_effective_layers is not None:
              de = deep_embeddings
            else:
              de = deep_embeddings[scan_start:scan_start + scan_length]    
            de = jnp.stack(de, axis=0) if de is not None and isinstance(de, list) else de
            max_logging.log(f'de: {de}', debug=cfg.debug)
            # outputs: [scan_length, batch, length, emb_dim]
            if cfg.scan_use_mudd:
              if len(hids) <= 4:
                c_hids = hids
              else:
                c_hids = hids[:1] + hids[-3:]
            else:
              c_hids = None
            y, outputs = self.scan_decoder_layers(
                cfg, 
                RemattedBlockLayers[1], 
                scan_length, 
                f"layers_{scan_start}", 
                mesh,
                sliding_window_size=current_sws,
                scan_length=scan_length,
            )(
                y,
                decoder_segment_ids,
                decoder_positions,
                decoder_input_tokens,
                de,
                deterministic,
                model_mode,
                c_hids, # me + local hids
                eos_sum=eos_sum,
            )
            if cfg.dense_conn and cfg.compose_all_layers:
              for output in outputs[:-1]:
                hids.append(output)
          
            lyr += scan_length

          if scan_length == 1: # scan output no compose
            y_normed = normalizations.get_rmsnorm("mudd_prenorm", cfg)(y) if cfg.mudd_prenorm else y
            hids.append(y_normed)
          else:
            if cfg.scan_use_mudd:
              y_normed = normalizations.get_rmsnorm("mudd_prenorm", cfg)(outputs[0]) if cfg.mudd_prenorm else outputs[0]
              hids.append(y_normed)

      else:
        swss = format_swss(sws_list)
        max_logging.log(f'swss: {len(swss)}-{swss}, num_decoder_layers: {cfg.num_decoder_layers}', debug=cfg.debug)
        for lyr in range(cfg.num_decoder_layers):
          max_logging.log(f'\n=================decoder layer: {lyr}=====================\n', debug=cfg.debug)
          RemattedBlockLayer = RemattedBlockLayers[0]
          y, hids = RemattedBlockLayer(
            config=cfg, 
            mesh=mesh, 
            name=f"layers_{lyr}", 
            quant=self.quant, 
            sliding_window_size=swss[lyr])(
              y,
              decoder_segment_ids,
              decoder_positions,
              decoder_input_tokens,
              deep_embeddings[lyr],
              deterministic,
              model_mode,
              hids=hids,
              eos_sum=eos_sum,
          )

    if cfg.dense_conn:
      print(f'me: {len(me)} hids: {len(hids)}')
      C = 2 if cfg.mtp_num_layers > 0 else 1
      y, hids = mudd.Compose(
        cfg, self.mesh, self.quant, 
        compose=True,
        name=f'compose_final',
        C=C,
        )(
          layer_output=y if isinstance(y, jnp.ndarray) else y[0], 
          hids=me + hids,
        )
            
    max_logging.log(f'y: {y.shape if isinstance(y, jnp.ndarray) else y[0].shape}', debug=cfg.debug)
    if cfg.dense_conn:
      mtp_head_inputs, main_head_inputs = y if cfg.mtp_num_layers > 0 else [None, y[0]]
    else:
      main_head_inputs, mtp_head_inputs = [y, y] if cfg.mtp_num_layers > 0 else [y, None]

    # mtp share llm head params
    OutputHeadLayer = OutputHead(config=cfg, 
                        shared_embedding=self.shared_embedding,
                        mesh=mesh,
                        quant=self.quant,
                        name='lm_head')
    max_logging.log(f'OutputHeadLayer input: {main_head_inputs.shape}', debug=cfg.debug)
    xent, correct, main_model_preds = OutputHeadLayer(
      main_head_inputs, 
      decoder_target_tokens, 
      decoder_target_mask, 
      cfg.loss_chunk_size, # set 1k better than 4k
      deterministic, 
      mtp_layer=False
      )
    if cfg.mtp_num_layers > 0:
      assert mtp_head_inputs is not None, 'mtp_head_inputs is None'
      # lsp: Don't to use remat in here where will lead to decrease performance and inscrease hbm significantly.
      mtp.MultiTokenPredictionBlock(
        config=cfg,
        mesh=mesh,
        quant=self.quant,
        name="mtp_block",
        transformer_layer_module=self.decoder_layer[0] \
          if cfg.mtp_use_remat or cfg.base_emb_dim <= 2048 else RemattedBlockLayers[0], # 模型小的时候，不使用remat
        shared_embedding=[rolled_y, mtp_de],
        sliding_window_size=swss[-1],
      )(
        OutputHeadLayer,
        main_hidden_state=mtp_head_inputs,
        input_ids=decoder_input_tokens,
        target_ids=decoder_target_tokens,
        target_mask=decoder_target_mask,
        position_ids=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        model_mode=model_mode,
        hids=hids[len(me): ] if me else hids, # mtp layer no use prefix embeddings
      )
    return xent, correct, main_model_preds


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
    de_dim = 0
    if cfg.deep_embed_init == 'outside':
      if '1xattn' in cfg.deep_embed_type:
        de_dim += cfg.num_decoder_layers * cfg.emb_dim
      elif 'devalue' in cfg.deep_embed_type.lower():
        if isinstance(cfg.num_kv_heads, list):
          head_nums_mean = sum(cfg.num_kv_heads) // len(cfg.num_kv_heads)
        else:
          head_nums_mean = cfg.num_kv_heads
        de_dim = head_nums_mean * cfg.num_decoder_layers * cfg.head_dim
      elif '1xmlp' in cfg.deep_embed_type.lower():
        de_dim  += cfg.num_decoder_layers * cfg.emb_dim
      elif '4xmlp' in cfg.deep_embed_type.lower():
        if cfg.deep_embed_effective_layers is not None:
          de_dim +=  cfg.deep_embed_effective_layers * cfg.mlp_dim
        else:
          de_dim +=  cfg.num_decoder_layers * cfg.mlp_dim
      else:
        # Multi deep embed don't support outside init
        raise ValueError(f'Invalid deep_embed_type: {cfg.deep_embed_type} for outside init')
     
    max_logging.log(f'deep_embed_init: {cfg.deep_embed_init} de_dim: {de_dim}', debug=cfg.debug)
    if cfg.mtp_num_layers > 0: # mtp de 和 word emb 一起是最快的，mtp de 和普通层放一起第二，word emb和mtp和普通层放一起最慢
      emb_dim = cfg.emb_dim + de_dim // cfg.num_decoder_layers
    else:
      emb_dim = cfg.emb_dim

    self.shared_embedding = Embed(
        num_embeddings=cfg.vocab_size,
        features=emb_dim,
        dtype=cfg.dtype,
        attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,  # for logit training stability
        embedding_init=initializers.get_init_method(cfg.init_method), # lsp
        name="token_embedder",
        config=cfg,
    )
    if de_dim > 0:
      self.deep_embedding = Embed(
          num_embeddings=cfg.vocab_size,
          features=de_dim,
          dtype=cfg.dtype,
          attend_dtype=jnp.float32 if cfg.logits_dot_in_fp32 else cfg.dtype,
          embedding_init=initializers.get_init_method(cfg.init_method), # lsp
          name="de_token_embedder",
          config=cfg,
      )
    else:
      self.deep_embedding = None
      
    self.decoder = Decoder(
        config=cfg, 
        shared_embedding=self.shared_embedding,
        deep_embedding=self.deep_embedding,
        mesh=mesh, 
        quant=self.quant, 
        name='decoder'
    )

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
      decoder_target_tokens,
      decoder_target_mask,
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
        decoder_segment_ids=decoder_segment_ids,
        decoder_target_tokens=decoder_target_tokens,
        decoder_target_mask=decoder_target_mask,
        deterministic=not enable_dropout,
        model_mode=model_mode,
    )
    return logits

  def logits_from_hidden_states(
      self,
      hidden_states,
      deterministic: bool = True,
      mtp_layer: bool = False,
  ):
    return self.decoder.logits_from_hidden_states(
        hidden_states,
        deterministic=deterministic,
        mtp_layer=mtp_layer,
    )
