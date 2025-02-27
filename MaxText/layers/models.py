
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
    lnx = get_rmsnorm(name="pre_self_attention_norm", cfg=cfg)(inputs)
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

    if cfg.record_internal_nn_metrics:
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
    self.norm_layer = self.get_norm_layer()
    if self.config.using_pipeline_parallelism:
      pipeline_stage_module = self.get_pipeline_stage_module(self.decoder_layer[0])
      remat_policy = self.get_remat_policy()
      self.pipeline_module = pipeline.Pipeline(
          config=self.config, mesh=self.mesh, layers=pipeline_stage_module, remat_policy=remat_policy
      )

  def get_remat_policy(self):
    cfg = self.config
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
    for block_layer in block_layers:
      layer = nn.remat(  # pylint: disable=invalid-name
          block_layer,
          prevent_cse=not self.config.scan_layers,
          policy=policy,
          static_argnums=(4, 5),  # Deterministic and model mode are static arguments.
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
    elif self.config.decoder_block == "dcformer":
      from layers import dcformer
      return dcformer.DcformerDecoderLayer

    elif self.config.decoder_block == "fusion":
      from layers import fusion_decoder
      return fusion_decoder.FusionDecoderLayer

    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def get_norm_layer(self, name, cfg=None, **kwargs):
    if self.config.decoder_block in ("default", "llama2", "mistral", "deepseek", "gemma", "gemma2", "simple", "simple_mlp"):
      return RMSNorm
    elif self.config.decoder_block == "gpt3":
      from layers import gpt3

      return functools.partial(gpt3.Gpt3LayerNorm, reductions_in_fp32=False, use_bias=True)
    elif self.config.decoder_block in ['dcformer', 'fusion']:
      return get_rmsnorm(name=name, cfg=cfg, **kwargs)
    else:
      raise ValueError(f"Incorrect decoder_block name {self.config.decoder_block=}")

  def scan_decoder_layers(self, cfg, decoder_layer, length, metdata_axis_name, mesh, args_length):
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
        in_axes=(nn.broadcast, ) * args_length,
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
      decoder_segment_ids=None,
      deterministic=False,
      model_mode=common_types.MODEL_MODE_TRAIN,
  ):
    cfg = self.config
    mesh = self.mesh
    assert decoder_input_tokens.ndim == 2  # [batch, len]

    if cfg.set_mask_by_eos:
      # ======================================32k long context max window size set==================================================
      eos_sum = (decoder_input_tokens == 151643).sum(1) 
      eos_sum = jnp.where(eos_sum > 0, 1, 0) # batch
      if cfg.record_internal_nn_metrics:
        self.sow("intermediates", "eos_sum_mean", eos_sum.mean(), ) # 每个batch带有eos数据的比例
        self.sow("intermediates", "eos_sum", eos_sum.sum(), ) # batch总的eos数量
    else:
      eos_sum = None

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

    if cfg.dense_conn: 
      if cfg.mudd_prenorm:
        assert cfg.ddw_gen_pattern == 'q,k,v,m', max_logging.log(f'Error: ddw_gen_pattern must be ‘q,k,v,m’ when mudd_prenorm is true.')
        y_normed = self.get_norm_layer(name="mudd_prenorm", cfg=cfg)(y)
      else:
        y_normed = y
      y, hids = [y] * len(cfg.dynamic_dense_type), [y_normed]  # XD

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
      num_layers_per_block = cfg.num_layers_per_block
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
          assert not cfg.dense_conn
          args = (y, decoder_segment_ids, decoder_positions, deterministic, model_mode, )
          if cfg.decoder_block == 'fusion':
            args += (num_layers_per_block, eos_sum, cfg.window_size, )
          args_length = len(args) - 1
          y, _ = self.scan_decoder_layers(cfg, RemattedBlockLayer, cfg.num_decoder_layers // num_layers_per_block, "layers", mesh, args_length)(*args)

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
            y = layers[index](config=cfg, mesh=mesh, name=f"{layer_prefix[index]}_{index}", quant=self.quant)(
                y,
                decoder_segment_ids,
                decoder_positions,
                deterministic,
                model_mode,
            )
        else:
          window_size = cfg.window_size
          n_layers = cfg.num_decoder_layers // num_layers_per_block
          if isinstance(window_size, list):
            if len(window_size) == num_layers_per_block:
              ws = [window_size] * n_layers
            else:
              ws = n_layers * [window_size * num_layers_per_block]
          else:
            ws = [[window_size] * num_layers_per_block] * n_layers
          max_logging.log(f'scan is False, all layers window size: {ws}')

          for lyr in range(n_layers):
            RemattedBlockLayer = RemattedBlockLayers[0]
            args = (y, decoder_segment_ids, decoder_positions, deterministic, model_mode, )
            if cfg.decoder_block == 'fusion':
              args += (num_layers_per_block, eos_sum, ws[lyr], )
            y = RemattedBlockLayer(config=cfg, mesh=mesh, name=f"layers_{lyr}", quant=self.quant)(*args)
            if getattr(cfg, 'dense_conn', False):  # XD
              max_logging.log(f'dense_conn is true')
              i = lyr  # to be compatible with pax code
              y, dyn_dense_w = y  # unpack tuple
              max_logging.log(f'dyn_dense_w dtype: {dyn_dense_w.dtype}')
              
              if self.config.record_internal_nn_metrics:
                self.sow('intermediates', f'dyn_dense_w/max/layer_{lyr}', jnp.max(dyn_dense_w))
                self.sow('intermediates', f'dyn_dense_w/mean/layer_{lyr}', jnp.mean(dyn_dense_w))
                self.sow('intermediates', f'dyn_dense_w/min/layer_{lyr}', jnp.min(dyn_dense_w))
                self.sow('intermediates', f'dyn_dense_w/norm/layer_{lyr}', l2norm(dyn_dense_w))
                self.sow('intermediates', f'dyn_dense_w/std/layer_{lyr}', jnp.std(dyn_dense_w))
                self.sow('intermediates', f'layer_output/norm/layer_{lyr}', l2norm(y))

              if cfg.mudd_prenorm:
                y_normed = self.get_norm_layer(name=f"mudd_prenorm_{lyr}", cfg=cfg)(y)
              else:
                y_normed = y
              hids.append(y_normed)

              C = 1 if cfg.dynamic_dense_fix_last_layer and i == cfg.num_decoder_layers - 1 else len(cfg.dynamic_dense_type)
              dyn_dense_w = rearrange(dyn_dense_w, 'B T C L -> C B T L 1', C=C)
              factor = 1
              hid_idxs = list(range((i+1) * factor + 1)) # L+1
              if cfg.ddw_gen_pattern == 'q,k,v,m':
                max_logging.log(f'ddw_gen_pattern: {cfg.ddw_gen_pattern} mudd_postnorm is {cfg.mudd_postnorm}....')
                if cfg.mudd_postnorm:
                  post_norm = self.get_norm_layer(name=f"mudd_postnorm_{lyr}", cfg=cfg, scale_init=jax.nn.initializers.constant(0.001))
                  y = tuple([y + (post_norm(
                      wsum(dyn_dense_w[cidx: cidx + 1], hids, cfg.ddw_gen_chunk_size).squeeze(0)
                                            ) if cidx == C - 1 else 
                      wsum(dyn_dense_w[cidx: cidx + 1], hids, cfg.ddw_gen_chunk_size).squeeze(0)
                                  ) for cidx in range(C)])
                else:
                  y = tuple([wsum(dyn_dense_w[cidx: cidx + 1], hids, cfg.ddw_gen_chunk_size).squeeze(0) for cidx in range(C)]) # (btl, btl, btl, btl)
              elif cfg.ddw_gen_pattern == 'qkvm':
                y = wsum(dyn_dense_w, hids, cfg.ddw_gen_chunk_size) # cbtl
              elif cfg.ddw_gen_pattern == 'qk,vm':
                yqk = wsum(dyn_dense_w[ :2], hids, cfg.ddw_gen_chunk_size) # cbtl
                yvm = wsum(dyn_dense_w[2: ], hids, cfg.ddw_gen_chunk_size) # cbtl
                y = jnp.concatenate([yqk, yvm], axis=0)
              
          if getattr(cfg, 'dense_conn', False):  # XD
            y = y[0] # if cfg.dynamic_dense_fix_last_layer else x_out[1]

    y = self.get_norm_layer(name="decoder_norm", cfg=cfg)(y)
    y = nn.Dropout(rate=cfg.dropout_rate, broadcast_dims=(-2,))(y, deterministic=deterministic)

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
      )(
          y
      )  # We do not quantize the logits matmul.
    logits = nn.with_logical_constraint(
        logits, ("activation_embed_and_logits_batch", "activation_length", "activation_vocab")
    )
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
        embedding_init=nn.initializers.normal(stddev=1.0),
        name="token_embedder",
        config=cfg,
    )

    self.decoder = Decoder(config=cfg, shared_embedding=self.shared_embedding, mesh=mesh, quant=self.quant)

  def __call__(
      self,
      decoder_input_tokens,
      decoder_positions,
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
        deterministic=not enable_dropout,
        model_mode=model_mode,
    )
    return logits
