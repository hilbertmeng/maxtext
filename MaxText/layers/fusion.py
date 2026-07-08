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
import jax
from flax import linen as nn
from jax.sharding import Mesh
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
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
Quant = quantizations.AqtQuantization


def _parse_layer_list(layers):
  if layers in (None, "", [], ()):
    return None
  if isinstance(layers, str):
    return {int(x.strip()) for x in layers.split(",") if x.strip()}
  return {int(x) for x in layers}


def _uses_kv_shift_on_layer(cfg, layer_inx):
  explicit_layers = _parse_layer_list(getattr(cfg, "kv_shift_layers", None))
  if explicit_layers is not None:
    return layer_inx in explicit_layers
  period = int(getattr(cfg, "kv_shift_layer_period", 1) or 1)
  offset = int(getattr(cfg, "kv_shift_layer_offset", 0) or 0)
  if period <= 1:
    return True
  return (layer_inx - offset) % period == 0


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
      # consider mtp layer, mtp layer's mlp_dim is the same as the last layer
      layer_inx = self.layer_inx if self.layer_inx <= cfg.num_decoder_layers else self.layer_inx - 1
      self.updated_mlp_dim = round(cfg.mlp_dim * (layer_inx / (cfg.num_decoder_layers - 1) + 0.5) / 128) * 128 
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
      kv_shift_plan=None,
  ):
    cfg = self.config
    mesh = self.mesh
    if cfg.dense_conn and cfg.dynamic_dense_type == 'qkvm' and isinstance(inputs, tuple|list): # lsp
      if cfg.scan_use_mudd and len(inputs) == 2: # scan use mudd
        lnx = normalizations.get_rmsnorm("muddnorm", cfg)(inputs[0])
        lnx_kv = [lnx, lnx]
        inputs = inputs[1]
      else:
        lnx, *lnx_kv = self.mudd_qkvnorm(inputs[:3])
        inputs = inputs[3]
    else:
      inputs = nn.with_logical_constraint(inputs, ("activation_batch", "activation_norm_length", "activation_embed"))
      inputs = checkpoint_name(inputs, "decoder_layer_input")
      lnx = normalizations.get_rmsnorm("pre_self_attention_layer_norm", cfg)(inputs)
      lnx = nn.with_logical_constraint(lnx, ("activation_batch", "activation_norm_length", "activation_embed"))
      lnx_kv = [lnx, lnx]

    num_kv_heads = cfg.num_kv_heads[self.layer_inx % len(cfg.num_kv_heads)] \
      if isinstance(cfg.num_kv_heads, list) else cfg.num_kv_heads

    if cfg.global_attn_head_dim \
      and cfg.global_attn_head_dim > 0 \
      and self.sliding_window_size == cfg.max_target_length:
      head_dim = cfg.global_attn_head_dim
      n = cfg.global_attn_head_dim / cfg.head_dim
      num_kv_heads = int(num_kv_heads // n)
      num_query_heads = int(cfg.num_query_heads // n)
    else:
      head_dim = cfg.head_dim
      num_query_heads = cfg.num_query_heads
      n = 1

    max_logging.log(f'layer_inx: {self.layer_inx} sliding window size: {self.sliding_window_size} n: {n}', debug=cfg.debug)
    max_logging.log(f'query heads: {num_query_heads} kv heads: {num_kv_heads} head_dim: {head_dim}', debug=cfg.debug)
    apply_kv_shift = cfg.use_kv_shift and _uses_kv_shift_on_layer(cfg, self.layer_inx)
    instantiate_kv_shift = apply_kv_shift or (
        cfg.use_kv_shift and getattr(cfg, "kv_shift_keep_params_on_skipped_layers", False)
    )
    # splash/flash and dot_product only apply the local window when attention_type is
    # LOCAL_SLIDING. Derive it per layer so sliding-window layers (e.g. LGLL) take effect
    # under splash; global layers (sws == max_target_length) stay GLOBAL.
    attention_type = (
        attentions.AttentionType.LOCAL_SLIDING
        if self.sliding_window_size is not None and self.sliding_window_size < cfg.max_target_length
        else attentions.AttentionType.GLOBAL
    )
    # Self-attention block
    attention_layer = Attention(
        config=cfg,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_target_length=cfg.max_target_length,
        max_prefill_predict_length=cfg.max_prefill_predict_length,
        attention_kernel=cfg.attention,
        attention_type=attention_type,
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
        use_kv_shift=instantiate_kv_shift,
        apply_kv_shift=apply_kv_shift,
    )

    attention_lnx = attention_layer(
        lnx,
        lnx if not cfg.dense_conn else lnx_kv,
        decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        decoder_input_tokens=decoder_input_tokens,
        model_mode=model_mode,
        eos_sum=eos_sum,
        deep_embedding=deep_embedding,
        kv_shift_plan=kv_shift_plan,
    )
   
    if cfg.record_internal_nn_metrics:
        attention_lnx_l2norm = jnp.sqrt(jnp.sum(jnp.square(attention_lnx)))
        self.sow('intermediates', 'attn_lnx/l2norm', attention_lnx_l2norm)

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = normalizations.get_rmsnorm("post_self_attention_layer_norm", cfg)(intermediate_inputs)
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
    return layer_output


class FusionDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Any
  mesh: Mesh
  sliding_window_size: int # lsp
  quant: Optional[Quant] = None
  scan_length: int = 1

  def setup(self):
    cfg = self.config
    self.layer_inx = 0 if cfg.scan_layers else int(self.name.split('_')[-1])
    sws = self.sliding_window_size
    max_logging.log(f'fusion layer sws: {sws}', debug=cfg.debug)
    if sws is None:
      sws = cfg.max_target_length
    self.sws = sws
    RematSubDecoderLayer = SubDecoderLayer
    if cfg.dense_conn and not cfg.mudd_in_layer:
      RematSubDecoderLayer = nn.remat(
        SubDecoderLayer,
        prevent_cse=cfg.remat_prevent_cse,
        policy=models.get_remat_policy(cfg),
        static_argnums=(6, 7),  # Deterministic and model mode are static arguments.
        )

    self.layer = RematSubDecoderLayer(cfg, self.mesh, self.quant, sws, self.layer_inx, name=f'block')
    self.break_layers = list(range(cfg.num_decoder_layers - 1, cfg.num_decoder_layers + cfg.mtp_num_layers))

  def get_C(self, cfg):
    if self.layer_inx == cfg.num_decoder_layers - 1:
      C = 2 if cfg.mtp_num_layers > 0 else 1 # if use mtp, return 2 tensors, otherwise return 1 tensor
    elif self.layer_inx == cfg.num_decoder_layers + cfg.mtp_num_layers - 1:
      C = 1 # last layer return 1 tensor
    else:
      C = 4 # other layer return 4 tensors
    return C
    
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
      kv_shift_plan=None,
  ):
    cfg = self.config
    if cfg.partial_scan_layers:
      return self.partial_scan_call(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        decoder_input_tokens,
        deep_embedding,
        deterministic,
        model_mode,
        hids,
        eos_sum,
        kv_shift_plan,
      )

    mudd_hidden_norm = normalizations.get_rmsnorm("mudd_prenorm", cfg) if cfg.mudd_prenorm else None

    def append_mudd_hidden(hids, hidden):
      hidden = mudd_hidden_norm(hidden) if mudd_hidden_norm is not None else hidden
      hids.append(hidden)
      return hids

    if cfg.dense_conn:
      if hids is None:
        hids = []
      if self.layer_inx == 0:
        if not hids:
          # Layer 0 has no previous block output yet. The outer decoder usually
          # seeds hids with the token embedding; keep this guard for direct layer
          # calls and avoid duplicating that seed in the non-scan path.
          hids = append_mudd_hidden(hids, inputs)
      elif self.layer_inx == cfg.num_decoder_layers and not cfg.mtp_use_compose:
        inputs = [inputs] * len(cfg.dynamic_dense_type) # mtp
      else:
        # return's inputs length is 4
        inputs, hids = mudd.Compose(
          cfg, self.mesh, self.quant, 
          name='compose',
          C=4,
          compose=True if self.layer_inx in cfg.compose_layers else False,
          )(
            layer_output=inputs, 
            hids=hids,
            lidx=self.layer_inx,
          )
    # return's inputs length is 1
    inputs = self.layer(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        decoder_input_tokens,
        deep_embedding,
        deterministic,
        model_mode,
        eos_sum,
        kv_shift_plan=kv_shift_plan,
    )
    max_logging.log(f'layer_inx: {self.layer_inx} break_layers: {self.break_layers}', debug=cfg.debug)
    if cfg.dense_conn and self.layer_inx in self.break_layers:
      C = self.get_C(cfg)
      # compose_break is after the block, so the current block output must be
      # available as a source. This matters when mudd_postnorm=False, where
      # MUDD initializes by selecting the last hidden source.
      hids = append_mudd_hidden(hids, inputs)
      inputs, hids = mudd.Compose(
        cfg, self.mesh, self.quant, 
        compose=True,
        name=f'compose_break',
        C=C,
        )(
          layer_output=inputs, 
          hids=hids,  
          lidx=self.layer_inx,
        )
      hids = append_mudd_hidden(hids, inputs[0] if isinstance(inputs, (tuple, list)) else inputs)
    elif cfg.dense_conn:
      hids = append_mudd_hidden(hids, inputs)
   
    return inputs, hids

  def partial_scan_call(
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
      kv_shift_plan=None,
  ):
    cfg = self.config
    if cfg.dense_conn and self.layer_inx > 0:  # compose for the first layer when mudd has extra embeddings
      if self.scan_length == 1 or cfg.scan_use_mudd:
        C = 4
        # inputs length: 2 or 4
        inputs, hids = mudd.Compose(
          cfg, self.mesh, self.quant, 
          name=f'compose_start',
          C=C,
          compose=True,
          )(
            layer_output=inputs, 
            hids=hids,
            lidx=self.layer_inx,
          )
            
    # return's inputs length is 1
    output = self.layer(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        decoder_input_tokens,
        deep_embedding,
        deterministic,
        model_mode,
        eos_sum,
        kv_shift_plan=kv_shift_plan,
    )
    if cfg.record_internal_nn_metrics:
      layer_output_l2norm = jnp.sqrt(jnp.sum(jnp.square(output)))
      self.sow('intermediates', 'layer_output/l2norm', layer_output_l2norm)
      
    return output, hids if self.scan_length == 1 else output
