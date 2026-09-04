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
import math

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


def _bam_mlp_write_delta(
    data, address, gate, *, epsilon, statistics_dtype):
  """Form one gated MLP-to-M write from per-slot data/address factors."""
  data = normalizations.rms_norm(
      data, dtype=data.dtype, epsilon=epsilon,
      statistics_dtype=statistics_dtype)
  address = normalizations.rms_norm(
      address, dtype=address.dtype, epsilon=epsilon,
      statistics_dtype=statistics_dtype)
  with jax.named_scope("bam/mlp_write_outer"):
    return jnp.sum(
        gate[..., None, None] * data[..., :, None]
        * address[..., None, :], axis=-3)


class MlpBamWrite(nn.Module):
  """Write selected MLP hidden channels into the cross-layer BAM stream."""

  config: models.Config
  num_write_heads: int
  dtype: DType
  weight_dtype: DType
  quant: Optional[Quant]
  kernel_init: initializers.Initializer

  @nn.compact
  def __call__(self, inputs, mlp_hidden):
    cfg = self.config
    heads = self.num_write_heads
    bam_k = int(cfg.bam_k)
    bam_v = int(cfg.bam_v)
    bottleneck = int(cfg.bam_mlp_write_v_bottleneck_dim)
    data_width = heads * bam_k
    if mlp_hidden.shape[-1] < data_width:
      raise ValueError(
          f"MLP hidden width {mlp_hidden.shape[-1]} is smaller than the "
          f"BAM write-data width {data_width}")

    data = mlp_hidden[..., :data_width].reshape(
        mlp_hidden.shape[:-1] + (heads, bam_k))
    with jax.named_scope("bam/mlp_write_address"):
      address = linears.DenseGeneral(
          features=bottleneck, axis=-1, kernel_init=self.kernel_init,
          kernel_axes=("embed", None), dtype=self.dtype,
          weight_dtype=self.weight_dtype, name="P_loc_down", quant=self.quant,
          matmul_precision=cfg.matmul_precision, use_bias=False)(inputs)
      address = nn.gelu(address)
      address = linears.DenseGeneral(
          features=(heads, bam_v), axis=-1, kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_heads", "v_factor"), dtype=self.dtype,
          weight_dtype=self.weight_dtype, name="P_loc_up", quant=self.quant,
          matmul_precision=cfg.matmul_precision, use_bias=True)(address)

    with jax.named_scope("bam/mlp_write_gate"):
      gate_logits = linears.DenseGeneral(
          features=heads, axis=-1, kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_heads"), dtype=self.dtype,
          weight_dtype=self.weight_dtype, name="W_gw", quant=self.quant,
          matmul_precision=cfg.matmul_precision, use_bias=False)(inputs)
      eps = float(cfg.bam_write_eps)
      if not 0.0 < eps < 1.0:
        raise ValueError(f"bam_write_eps must be in (0, 1), got {eps}")
      gate_bias = self.param(
          "gw_b0",
          nn.with_logical_partitioning(
              lambda key, shape, dtype: jnp.full(
                  shape, math.log(eps / (1.0 - eps)), dtype),
              ("q_heads",)),
          (heads,), self.weight_dtype)
      gate = jax.nn.sigmoid(
          gate_logits + jnp.asarray(gate_bias, self.dtype))
      if cfg.bam_sqrt_n_scale:
        gate = gate / jnp.sqrt(jnp.asarray(heads, gate.dtype))

    statistics_dtype = (
        jnp.float32
        if cfg.bam_write_rms_statistics_dtype == "float32"
        else self.dtype)
    return _bam_mlp_write_delta(
        data, address, gate, epsilon=cfg.normalization_layer_epsilon,
        statistics_dtype=statistics_dtype)


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
      M_in=None,
      is_global=None,
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
    # Self-attention block
    attn_kwargs = dict(
        config=cfg,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
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
        use_kv_shift=cfg.use_kv_shift,
    )
    if cfg.bam_enabled:
        AttnCls = attentions.BamAttention
        modes = cfg.bam_layer_modes
        layer_mode = modes[self.layer_inx] if isinstance(modes, list) else modes
        read_sides = cfg.bam_read_sides
        read_side = read_sides[self.layer_inx] if isinstance(read_sides, list) else read_sides
        attn_kwargs.update(
            layer_mode=layer_mode, read_side=read_side, bam_k=cfg.bam_k, bam_v=cfg.bam_v)
    else:
        AttnCls = Attention
    attention_layer = AttnCls(**attn_kwargs)

    call_kwargs = dict(
        inputs_q=lnx,
        inputs_kv=lnx if not cfg.dense_conn else lnx_kv,
        inputs_positions=decoder_positions,
        decoder_segment_ids=decoder_segment_ids,
        deterministic=deterministic,
        decoder_input_tokens=decoder_input_tokens,
        model_mode=model_mode,
        eos_sum=eos_sum,
        deep_embedding=deep_embedding,
    )
    if cfg.bam_enabled:
        attention_lnx, M_out = attention_layer(
            **call_kwargs, M_in=M_in, is_global=is_global)
    else:
        attention_lnx = attention_layer(**call_kwargs)
        M_out = M_in

    if cfg.record_internal_nn_metrics:
        attention_lnx_l2norm = jnp.sqrt(jnp.sum(jnp.square(attention_lnx)))
        self.sow('intermediates', 'attn_lnx/l2norm', attention_lnx_l2norm)

    attention_lnx = nn.with_logical_constraint(
        attention_lnx, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    if getattr(cfg, 'bam_residual_attribution', False) and not self.is_initializing():
      self.sow('residual_attribution', 'attention_total', attention_lnx)
    intermediate_inputs = inputs + attention_lnx

    # Fully Connected
    hidden_states = normalizations.get_rmsnorm("post_self_attention_layer_norm", cfg)(intermediate_inputs)
    hidden_states = nn.with_logical_constraint(
        hidden_states, ("activation_batch", "activation_norm_length", "activation_embed")
    )
    
    mlp_lnx = None
    if cfg.shared_experts == 1:
      # MLP block.
      mlp_result = linears.MlpBlock(
          intermediate_dim=self.updated_mlp_dim, # lsp
          activations=cfg.mlp_activations,
          intermediate_dropout_rate=cfg.dropout_rate,
          dtype=cfg.dtype,
          weight_dtype=cfg.weight_dtype,
          name="mlp",
          config=cfg,
          quant=self.quant,
          kernel_init=initializers.get_init_method(cfg.init_method), # lsp
      )(
          hidden_states, deep_embedding=deep_embedding,
          decoder_input_tokens=decoder_input_tokens,
          deterministic=deterministic,
          return_hidden=bool(getattr(cfg, "bam_mlp_write", False)))
      if getattr(cfg, "bam_mlp_write", False):
        if not cfg.bam_enabled or M_out is None:
          raise ValueError("MLP BAM write requires an active BAM matrix stream")
        mlp_lnx, mlp_hidden = mlp_result
        configured_heads = getattr(cfg, "mlp_num_bam_head", None)
        mlp_num_bam_head = (
            num_query_heads // 2
            if configured_heads is None else int(configured_heads))
        if not 0 < mlp_num_bam_head <= num_query_heads:
          raise ValueError(
              "mlp_num_bam_head must be in [1, num_query_heads], got "
              f"{mlp_num_bam_head} for {num_query_heads} query heads")
        dM_mlp = MlpBamWrite(
            config=cfg, num_write_heads=mlp_num_bam_head,
            dtype=cfg.dtype, weight_dtype=cfg.weight_dtype,
            quant=self.quant, kernel_init=initializers.get_init_method(cfg.init_method),
            name="mlp_bam_write")(
                hidden_states, mlp_hidden)
        M_out = M_out + dM_mlp
      else:
        mlp_lnx = mlp_result
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

    if getattr(cfg, 'bam_residual_attribution', False) and not self.is_initializing():
      self.sow('residual_attribution', 'layer_input', inputs)
      self.sow('residual_attribution', 'mlp_residual', mlp_lnx)
      self.sow('residual_attribution', 'layer_output', layer_output)
      self.sow(
          'residual_attribution', 'layer_delta',
          layer_output.astype(jnp.float32) - inputs.astype(jnp.float32))

    layer_output = nn.with_logical_constraint(
        layer_output,
        ("activation_batch", "activation_norm_length", "activation_embed"),
    )
    return layer_output, M_out


class FusionDecoderLayer(nn.Module):
  """Transformer decoder layer that attends to the encoder."""

  config: Any
  mesh: Mesh
  sliding_window_size: int # lsp
  quant: Optional[Quant] = None
  scan_length: int = 1
  all_global_attention: bool = False

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
      eos_sum=None,
      is_global=None,
      hids=None,
      M_in=None,
  ):
    cfg = self.config
    scan_full_bam = (
        cfg.scan_layers and cfg.bam_enabled
        and not getattr(cfg, 'bam_mha_control', False))
    if cfg.scan_layers:
      assert not cfg.dense_conn, 'flat layer scan currently requires dense_conn=False'
      if scan_full_bam:
        inputs, M_in = inputs
      else:
        M_in = None
      if self.all_global_attention:
        is_global = None
    if cfg.partial_scan_layers:
      assert not cfg.bam_enabled, "BAM v0.1 does not support partial_scan_layers"
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
      )

    if cfg.dense_conn:
      if self.layer_inx == 0:
        y_normed = normalizations.get_rmsnorm("mudd_prenorm", cfg)(inputs) \
          if cfg.mudd_prenorm else inputs
        # inputs = [inputs] * len(cfg.dynamic_dense_type) # 0层要不要分4路？
        hids.append(y_normed)
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
    inputs, M_out = self.layer(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        decoder_input_tokens,
        deep_embedding,
        deterministic,
        model_mode,
        eos_sum,
        M_in=M_in,
        is_global=is_global,
    )
    max_logging.log(f'layer_inx: {self.layer_inx} break_layers: {self.break_layers}', debug=cfg.debug)
    if cfg.dense_conn and self.layer_inx in self.break_layers:
      C = self.get_C(cfg)
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

    if cfg.scan_layers:
      carry = (inputs, M_out) if scan_full_bam else inputs
      return carry, ()
    if cfg.bam_enabled:
      return inputs, hids, M_out
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
            
    # return's inputs length is 1 (BAM is blocked on the partial_scan path, so M_out is None here)
    output, _ = self.layer(
        inputs,
        decoder_segment_ids,
        decoder_positions,
        decoder_input_tokens,
        deep_embedding,
        deterministic,
        model_mode,
        eos_sum,
    )
    if cfg.record_internal_nn_metrics:
      layer_output_l2norm = jnp.sqrt(jnp.sum(jnp.square(output)))
      self.sow('intermediates', 'layer_output/l2norm', layer_output_l2norm)
      
    return output, hids if self.scan_length == 1 else output
