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

"""Attentions Layers."""

import dataclasses
import enum
import functools
import math
from typing import Any, Optional, Tuple

from flax import linen as nn
from einops import rearrange
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
import jax.numpy as jnp
import common_types
from bam_config import validate_bam_config
from kernels.ragged_attention import ragged_gqa
from kernels.ragged_attention import ragged_mha
from layers import embeddings
from layers import initializers
from layers import linears
from layers import quantizations
from layers import dc
from layers import accelerator
from layers import normalizations
from layers import kv_shift

import max_logging

# pylint: disable=line-too-long, g-doc-args, g-doc-return-or-yield, bad-continuation, g-inconsistent-quotes
# pytype: disable=attribute-error


class AttentionType(enum.Enum):
  GLOBAL = "global"
  LOCAL_SLIDING = "local_sliding"
  MLA = "mla"


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh

DenseGeneral = linears.DenseGeneral
RotaryEmbedding = embeddings.RotaryEmbedding
YarnRotaryEmbedding = embeddings.YarnRotaryEmbedding
NdInitializer = initializers.NdInitializer
Quant = quantizations.AqtQuantization
KVQuant = quantizations.KVQuant
KVTensor = quantizations.KVTensor

AxisNames = common_types.AxisNames
AxisIdxes = common_types.AxisIdxes
BATCH = common_types.BATCH
PREFILL_KV_BATCH = common_types.PREFILL_KV_BATCH
KV_BATCH = common_types.KV_BATCH
LENGTH = common_types.LENGTH
HEAD = common_types.HEAD
EMBED = common_types.EMBED
KV_HEAD = common_types.KV_HEAD
D_KV = common_types.D_KV
KV_HEAD_DIM = common_types.KV_HEAD_DIM
CACHE_BATCH_PREFILL = common_types.CACHE_BATCH_PREFILL
CACHE_BATCH = common_types.CACHE_BATCH
CACHE_SEQUENCE = common_types.CACHE_SEQUENCE
CACHE_HEADS = common_types.CACHE_HEADS
CACHE_KV = common_types.CACHE_KV
CACHE_SCALE_BATCH = common_types.CACHE_SCALE_BATCH
CACHE_SCALE_SEQUENCE = common_types.CACHE_SCALE_SEQUENCE
CACHE_SCALE_HEADS = common_types.CACHE_SCALE_HEADS
CACHE_SCALE_KV = common_types.CACHE_SCALE_KV
DEFAULT_MASK_VALUE = common_types.DEFAULT_MASK_VALUE

nd_dense_init = initializers.nd_dense_init
shard_map = shard_map.shard_map


def validate_compute_axis_order(s: AxisIdxes) -> None:
  valid_compute_axis_order = ((0, 1, 2, 3), (0, 2, 1, 3))
  if s not in valid_compute_axis_order:  # currently supported compute_axis_order
    raise ValueError("Invalid compute_axis_order was passed. Valid options ", valid_compute_axis_order)


def apply_mask_to_logits(logits: Array, mask: Array):
  """Applies a floating-point mask to a set of logits.

  The mask is represented as a tensor with some dtype where 0 represents true and values
  below a large negative number (here set to
  get_large_negative_number(logits.dtype) / 2) represent false. Applying the mask
  leaves the logits alone in the true case and replaces them by
  get_large_negative_number(logits.dtype) in the false case. Previously, this was
  done by adding the logits to the mask; however, this leads to a bad fusion
  decision in the compiler that saves the values in memory rather than
  just the predicate. This implementation avoids that problem.

  from https://github.com/google/praxis/blob/4712a6b9ee13e224b86e235ff55f7c6bab9fbab3/praxis/py_utils.py#L706

  Args:
    logits: A JTensor of logit values.
    mask: A JTensor of mask values with the encoding described in the
      function documentation.

  Returns:
    Masked logits.
  """
  return jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), logits, DEFAULT_MASK_VALUE)


class AttentionOp(nn.Module):
  config: Config
  mesh: Mesh
  attention_kernel: str
  max_target_length: int
  num_query_heads: int
  num_kv_heads: int
  float32_qk_product: bool = False
  max_prefill_predict_length: int = -1
  float32_logits: bool = False
  flash_axis_names: AxisNames = (BATCH, HEAD, LENGTH, D_KV)
  prefill_cache_logical_axis_names: AxisNames = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)
  cache_logical_axis_names: AxisNames = (CACHE_BATCH, CACHE_SEQUENCE, CACHE_HEADS, CACHE_KV)
  cache_scale_logical_axis_names: AxisNames = (CACHE_SCALE_BATCH, CACHE_SCALE_SEQUENCE, CACHE_SCALE_HEADS, CACHE_SCALE_KV)
  ragged_qkv_axis_names: AxisNames = (CACHE_BATCH, CACHE_HEADS, CACHE_SEQUENCE, CACHE_KV)
  ragged_lengths_names: AxisNames = (CACHE_BATCH,)
  prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  compute_axis_order: AxisIdxes = (0, 1, 2, 3)
  reshape_q: bool = False
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32
  quant: Optional[Quant] = None
  kv_quant: Optional[KVQuant] = None
  attention_type: AttentionType = AttentionType.GLOBAL  # Default to global attention
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  use_ragged_attention: bool = False
  ragged_block_size: int = 256

  def check_attention_inputs(self, query: Array, key: Array | KVTensor, value: Array | KVTensor) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, "k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

  # Following Pallas MHA Flash Attention Reference.
  # https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/flash_attention.py
  # This mask models (1) separate sequences (decoder_segment_ids) and (2) causality
  def generate_attention_mask(self, query, key, decoder_segment_ids: Array | None, model_mode: str) -> Array | None:
    mask = None
    if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      mask = decoder_segment_ids[:, None, None, None, :] == common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    elif decoder_segment_ids is not None:
      mask = decoder_segment_ids[:, :, None] == decoder_segment_ids[:, None, :]
      mask = mask[:, None, None, :, :]

    causal_mask = None
    # We enforce causality except for AUTOREGRESSION
    if model_mode != common_types.MODEL_MODE_AUTOREGRESSIVE:
      _, q_seq_len, _, _ = query.shape
      _, kv_seq_len, _, _ = key.shape
      mask_shape = (q_seq_len, kv_seq_len)
      row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
      col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
      causal_mask = (col_ids <= row_ids)[None, None, None, :, :]

    output_mask = None

    if (mask is not None) and (causal_mask is not None):
      output_mask = jnp.logical_and(mask, causal_mask)
    elif mask is not None:
      output_mask = mask
    elif causal_mask is not None:
      output_mask = causal_mask

    if self.attention_type == AttentionType.LOCAL_SLIDING and output_mask is not None:
      if self.sliding_window_size is None:
        raise ValueError("Sliding_window_size must be set if Local Sliding attention type")

      all_ones = jnp.ones_like(output_mask)
      sliding_mask = jnp.triu(all_ones, -1 * self.sliding_window_size + 1) * jnp.tril(all_ones, self.sliding_window_size - 1)
      output_mask = sliding_mask * output_mask

    return jnp.where(output_mask, 0.0, DEFAULT_MASK_VALUE) if output_mask is not None else None

  def apply_attention(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      decoder_segment_ids: Array | None,
      lengths: Array | None,
      model_mode: str,
      use_ragged_attention: bool = False,
      eos_sum: Array | None = None,
  ):
    self.check_attention_inputs(query, key, value)
    length = query.shape[-3]
    if use_ragged_attention and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      if lengths is None:
        lengths = jnp.sum(decoder_segment_ids, axis=-1)

      return self.ragged_attention(query, key, value, lengths, self.ragged_block_size)
    elif (
        self.attention_kernel == "dot_product"
        or (self.attention_kernel == "autoselected" and model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE)
        or (self.attention_kernel == "autoselected" and length < 128)
    ):
      return self.apply_attention_dot(query, key, value, decoder_segment_ids, model_mode)
    elif self.attention_kernel == "dot_product_chunk": # lsp: dc, llama, mudd etc. expecially when head_dim < 128, can't use flash to accelerate
      return accelerator.QChunk(config=self.config,
                                sliding_window_size=self.sliding_window_size,
                                kv_quant=self.kv_quant)(
                                query, key, value, decoder_segment_ids, model_mode, eos_sum
                                )

    elif self.attention_kernel == "flash" or self.attention_kernel == "autoselected":
      if isinstance(key, KVTensor):
        key = key.dequant()
      if isinstance(value, KVTensor):
        value = value.dequant()

      if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError(
            """Decode not supported with flash attention.
                            Use `dot_product` instead."""
        )
      return self.tpu_flash_attention(query, key, value, decoder_segment_ids, self.attn_logits_soft_cap), None, None
    elif self.attention_kernel == "cudnn_flash_te":
      if isinstance(key, KVTensor):
        key = key.dequant()
      if isinstance(value, KVTensor):
        value = value.dequant()
      if model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
        raise ValueError(
            """Decode not supported with flash attention.
                           Use `dot_product` instead."""
        )
      return self.cudnn_flash_attention(query, key, value, decoder_segment_ids, model_mode), None, None
    else:
      raise ValueError(f"Unexpected attention kernel {self.attention_kernel=}.")

  def ragged_attention(
      self, query: Array, key: Array | KVTensor, value: Array | KVTensor, lengths: Array, block_size: int
  ) -> tuple[Array, Array, Array]:
    """Ragged Attention."""
    if isinstance(query, KVTensor) or isinstance(query, KVTensor):
      raise TypeError("Ragged attention does not currently support quantized tensors.")
    b = nn.logical_to_mesh_axes(self.ragged_lengths_names)
    bsnd = nn.logical_to_mesh_axes(self.cache_logical_axis_names)

    @functools.partial(
        shard_map,
        mesh=self.mesh,
        in_specs=(
            bsnd,
            bsnd,
            bsnd,
            b,
            None,
        ),
        out_specs=bsnd,
        check_rep=False,
    )
    def wrap_ragged_attention(query, key, value, lengths, block_size):
      if query.shape[-2] == key.shape[-2]:
        return ragged_mha(query, key, value, lengths, block_size=block_size)
      else:
        return ragged_gqa(query, key, value, lengths, block_size=block_size)

    return wrap_ragged_attention(query, key, value, lengths, block_size)

  def tpu_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      attn_logits_soft_cap: float | None = None,
  ) -> Array:
    """TPU Flash Attention."""
    # Transpose to ('batch', 'heads', 'length', 'kv')
    query = jnp.transpose(query, axes=(0, 2, 1, 3))
    key = jnp.transpose(key, axes=(0, 2, 1, 3))
    value = jnp.transpose(value, axes=(0, 2, 1, 3))

    if decoder_segment_ids is not None:
      decoder_segment_ids = splash_attention_kernel.SegmentIds(decoder_segment_ids, decoder_segment_ids)
    axis_names = nn.logical_to_mesh_axes(self.flash_axis_names)
    segment_axis_names = nn.logical_to_mesh_axes((BATCH, "activation_length_no_heads"))

    global_block_q = self.config.sa_block_q
    global_block_kv = self.config.sa_block_kv
    global_block_kv_compute = self.config.sa_block_kv_compute
    global_block_q_dkv = self.config.sa_block_q_dkv
    global_block_kv_dkv = self.config.sa_block_kv_dkv
    global_block_kv_dkv_compute = self.config.sa_block_kv_dkv_compute
    global_block_q_dq = self.config.sa_block_q_dq
    global_block_kv_dq = self.config.sa_block_kv_dq
    global_use_fused_bwd_kernel = self.config.sa_use_fused_bwd_kernel
    global_q_layout = self.config.sa_q_layout
    global_k_layout = self.config.sa_k_layout
    global_v_layout = self.config.sa_v_layout

    @functools.partial(
        shard_map,
        mesh=self.mesh,
        in_specs=(
            axis_names,
            axis_names,
            axis_names,
            segment_axis_names,
        ),
        out_specs=axis_names,
        check_rep=False,
    )
    def wrap_flash_attention(query, key, value, decoder_segment_ids):
      if decoder_segment_ids is not None:
        assert (
            query.shape[2] == decoder_segment_ids.q.shape[1]
        ), "Sharding along sequence dimension not allowed in tpu kernel attention"
      block_sizes = splash_attention_kernel.BlockSizes(
          block_q=min(global_block_q, query.shape[2]),
          block_kv=min(global_block_kv, key.shape[2]),
          block_kv_compute=min(global_block_kv_compute, key.shape[2]),
          block_q_dkv=min(global_block_q_dkv, query.shape[2]),
          block_kv_dkv=min(global_block_kv_dkv, key.shape[2]),
          block_kv_dkv_compute=min(global_block_kv_dkv_compute, query.shape[2]),
          block_q_dq=None if global_use_fused_bwd_kernel else min(global_block_q_dq, query.shape[2]),
          block_kv_dq=None if global_use_fused_bwd_kernel else min(global_block_kv_dq, query.shape[2]),
          use_fused_bwd_kernel=global_use_fused_bwd_kernel,
          q_layout=splash_attention_kernel.QKVLayout[global_q_layout],
          k_layout=splash_attention_kernel.QKVLayout[global_k_layout],
          v_layout=splash_attention_kernel.QKVLayout[global_v_layout],
      )

      mask = splash_attention_mask.CausalMask(shape=(query.shape[2], query.shape[2]))

      # Apply local masking if local sliding attention is enabled.
      if self.attention_type == AttentionType.LOCAL_SLIDING:
        if self.sliding_window_size is None:
          raise ValueError("Sliding_window_size must be set if Local Sliding attention type")
        mask &= splash_attention_mask.LocalMask(
            shape=(query.shape[2], query.shape[2]),
            window_size=(self.sliding_window_size, self.sliding_window_size),
            offset=0,
        )

      # Create multi-head mask
      multi_head_mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * query.shape[1])
      splash_kernel = splash_attention_kernel.make_splash_mha(
          mask=multi_head_mask,
          head_shards=1,
          q_seq_shards=1,
          block_sizes=block_sizes,
          attn_logits_soft_cap=attn_logits_soft_cap,
      )

      return jax.vmap(splash_kernel)(query, key, value, segment_ids=decoder_segment_ids)

    devices_in_data_fsdp = self.mesh.shape["data"] * self.mesh.shape["fsdp"]
    assert (query.shape[0] / devices_in_data_fsdp).is_integer(), (
        "Batch dimension should be shardable among the devices in data and fsdp" " axis"
    )
    x = wrap_flash_attention(query, key, value, decoder_segment_ids)
    x = jnp.transpose(x, axes=(0, 2, 1, 3))
    return x

  def cudnn_flash_attention(
      self,
      query: Array,
      key: Array,
      value: Array,
      decoder_segment_ids: Array | None,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
  ) -> Array:
    """CUDNN Flash Attention with Transformer Engine.
    1. Stable API, supports GQA, SWA (only with causal masking)
    2. Head_dim = 256 is also supported from TE-1.12 stable release with CUDNN 12.6
    """
    # These imports are only meant to work in a GPU build.
    from transformer_engine.jax.flax.transformer import DotProductAttention  # pytype: disable=import-error

    _, _, _, head_dim = query.shape  # pylint: disable=unused-variable

    sliding_window_size = self.sliding_window_size
    if self.attention_type == AttentionType.LOCAL_SLIDING:
      sliding_window_size = [self.sliding_window_size, 0]
      mask_type = "causal"  # SWA only works with causal masking
      attn_mask = None
    else:
      # generate attn_mask
      mask_type = "padding_causal"  # only padding_causal mask type can take a created mask
      attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)

    dpa_layer = DotProductAttention(
        head_dim=head_dim,
        num_attention_heads=self.num_query_heads,
        num_gqa_groups=self.num_kv_heads,
        attn_mask_type=mask_type,  # 'no_mask', 'padding', 'causal', or 'padding_causal'
        attn_bias_type="no_bias",  # 'no_bias', 'pre_scale_bias' or 'post_scale_bias'
        attention_dropout=self.dropout_rate,
        dropout_rng_name="aqt",
        dtype=self.dtype,
        float32_logits=self.float32_logits,
        qkv_layout="BSHD_BSHD_BSHD",  # 'BS3HD', 'BSHD_BS2HD' or 'BSHD_BSHD_BSHD'
        scale_factor=1.0 / math.sqrt(head_dim),
        transpose_batch_sequence=False,
        window_size=sliding_window_size,
    )
    return dpa_layer(query, key, value, mask=attn_mask)

  def compute_local_attention(
      self, attn_weights: Array, value: Array | KVTensor, q_seq_len: int, model_mode: str
  ) -> tuple[Array, Array, Array]:
    """Computes the attention of a local subset of the kv cache.
    Local attention results will need to be combined with any other local attentions and normalized
    Based on https://github.com/google-research/google-research/blob/master/scaling_transformer_inference_efficiency/attention.py

    Args:
        attn_weights (Array): Product of query and key
        value (Array): Current value
        aqt_rng (PRNGKey | None): Optional rng

    Returns:
        (local_out, local_max,): where
          local_out is local unnormalized output
          local_max is the local max of exponentials
          local_sum is the sum of exponentials for this chunk, divided by exp(local_max).
    """
    with jax.named_scope("attention/softmax"):
      local_max = jnp.max(attn_weights, axis=-1, keepdims=True)
      local_exps = jnp.exp(attn_weights - local_max)
      local_sum = jnp.sum(local_exps, axis=-1, keepdims=True)

      local_sum = jnp.moveaxis(local_sum, -2, 1)
      local_max = jnp.moveaxis(local_max, -2, 1)

      local_max = jnp.reshape(local_max, (local_max.shape[0], local_max.shape[1], local_max.shape[2] * local_max.shape[3], 1))
      local_sum = jnp.reshape(local_sum, (local_sum.shape[0], local_sum.shape[1], local_sum.shape[2] * local_sum.shape[3], 1))

    with jax.named_scope("attention/av"):
      local_out = self.wv_product(local_exps, value, model_mode)

    if self.reshape_q and q_seq_len == 1:
      local_max = local_max[:, 0:1, :, :]
      local_sum = local_sum[:, 0:1, :, :]
      local_out = local_out[:, 0:1, :, :]

    return local_out, local_max, local_sum

  def apply_attention_dot(
      self,
      query: Array,
      key: Array | KVTensor,
      value: Array | KVTensor,
      decoder_segment_ids: Array | None,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
  ):
    """Apply Attention."""
    validate_compute_axis_order(self.compute_axis_order)
    # Casting qk_product and softmaxt computation for float32 for model stability.
    if self.float32_qk_product:
      if isinstance(key, KVTensor):
        key = key.dequant()
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)

    q_seq_len = query.shape[1]
    with jax.named_scope("attention/qk_logits"):
      attn_weights = self.qk_product(query, key, q_seq_len, model_mode)
    max_logging.log(f'attn_weights: {attn_weights.shape}', debug=self.config.debug)

    if self.attn_logits_soft_cap:
      attn_weights = jnp.tanh(attn_weights / self.attn_logits_soft_cap)
      attn_weights = attn_weights * self.attn_logits_soft_cap

    # Casting softmaxt computation for float32 for model stability.
    if self.float32_logits:
      attn_weights = attn_weights.astype(jnp.float32)
    attn_mask = self.generate_attention_mask(query, key, decoder_segment_ids, model_mode)
    if attn_mask is not None:
      attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
    return self.compute_local_attention(attn_weights, value, q_seq_len, model_mode)

  def qk_product(self, query: Array, key: Array | KVTensor, q_seq_len: int, model_mode: str) -> Array:
    """Query-Key product.

    Args:
      query: Query projection, in shape of [b, t, n, d]
      key: Key projection in shape of [b, s, n_kv, d]

    Returns:
      results in shape [b, n_kv, n // n_kv, t, s].

    Annotations:
      b: batch size
      t: query length
      s: key / value length
      d: head / kv dimension
      n: number of query heads
      n_kv: number of kv heads, sometimes annotated as k
      n // n_kv: number of group for query, sometimes annotated with g
    """
    einsum = jnp.einsum
    if self.kv_quant:
      einsum = self.kv_quant.einsum_fn_with_rhs_qtensor(key)
    b, t, n, d = query.shape
    n_kv = key.shape[-2]
    assert n_kv == self.num_kv_heads
    if model_mode == common_types.MODEL_MODE_TRAIN or self.compute_axis_order == (0, 1, 2, 3):
      query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
      if self.reshape_q and q_seq_len == 1:
        query = jnp.broadcast_to(query, (b, 2, n_kv, n // n_kv, d))
      result = einsum("btkgd,bskd->bkgts", query, key)
    elif self.compute_axis_order == (0, 2, 1, 3):
      query = jnp.transpose(query, axes=self.compute_axis_order)
      key = jax.tree.map(lambda x: jnp.transpose(x, axes=self.compute_axis_order), key)
      query = jnp.reshape(query, (b, n_kv, n // n_kv, t, d))
      if self.reshape_q and q_seq_len == 1:
        query = jnp.broadcast_to(query, (b, n_kv, n // n_kv, 2, d))
      result = einsum("bkgtd,bksd->bkgts", query, key)
    return result

  def wv_product(self, attn_weights: Array, value: Array | KVTensor, model_mode: str) -> Array:
    """weighted value product.

    Args:
      attn_weights: Computed results of qk_einsum, in shape [b, n_kv, n // n_kv, t, s]
      value: Value projection, in shape of [b, s, n_kv, d]

    Returns:
      result in shape [b, t, n, d]

    Annotations:
      b: batch size
      t: query length
      s: key / value length
      d: head / kv dimension
      n: number of query heads
      n_kv: number of kv heads, sometimes annotated as k
      n // n_kv: number of group for query, sometimes annotated with g
    """

    einsum = jnp.einsum
    if self.kv_quant:
      einsum = self.kv_quant.einsum_fn_with_rhs_qtensor_and_dequant(value)
    if model_mode == common_types.MODEL_MODE_TRAIN or self.compute_axis_order == (0, 1, 2, 3):
      out = einsum("bkgts,bskd->btkgd", attn_weights, value)
      b, t, n_kv, g, d = out.shape
      result = jnp.reshape(out, (b, t, n_kv * g, d))
    elif self.compute_axis_order == (0, 2, 1, 3):
      value = jax.tree.map(lambda x: jnp.transpose(x, axes=self.compute_axis_order), value)
      out = einsum("bkgts,bksd->bkgtd", attn_weights, value)
      b, n_kv, g, t, d = out.shape
      result = jnp.reshape(out, (b, n_kv * g, t, d))
      result = self.reverse_transepose(result, self.compute_axis_order)
    return result

  def reverse_transepose(self, transposed_array, transpose_axis_order):
    return jax.numpy.moveaxis(transposed_array, (0, 1, 2, 3), transpose_axis_order)

  def transpose_tuple(self, items: tuple[Any, Any, Any, Any], axis_order: AxisIdxes) -> tuple[Any, Any, Any, Any]:
    return tuple([items[i] for i in axis_order])

  def _get_cached_kv_dtype(self, dtype):
    return self.kv_quant.dtype if self.kv_quant else dtype

  def _get_cache_scale_logical_shape(self, batch, heads, cache_length):
    assert self.kv_quant
    if self.kv_quant.axis_cfg == "dkv":
      return (batch, cache_length, heads, 1)
    if self.kv_quant.axis_cfg == "heads_and_dkv":
      return (batch, cache_length, 1, 1)
    raise f"Invalid config for kv_quant_axis:{self.kv_quant.axis_cfg}"

  def _get_prefill_cache_vars(self, batch, heads, kv_head_size, model_mode):

    cache_length = self.max_prefill_predict_length
    dtype = self._get_cached_kv_dtype(self.dtype)
    cache_logical_shape = (batch, cache_length, heads, kv_head_size)

    if model_mode == common_types.MODEL_MODE_PREFILL:
      cache_logical_axis_names = self.prefill_cache_logical_axis_names
    else:
      cache_logical_axis_names = self.cache_logical_axis_names

    cache_axis_names = self.transpose_tuple(cache_logical_axis_names, self.prefill_cache_axis_order)
    cache_shape = self.transpose_tuple(cache_logical_shape, self.prefill_cache_axis_order)

    cached_key_var = self.variable(
        "cache",
        "cached_prefill_key",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape,
        dtype,
    )
    cached_value_var = self.variable(
        "cache",
        "cached_prefill_value",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape,
        dtype,
    )
    if model_mode == common_types.MODEL_MODE_PREFILL:
      segment_id_axis_names = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE)
    else:
      segment_id_axis_names = (CACHE_BATCH, CACHE_SEQUENCE)

    cached_segment_id_var = self.variable(
        "cache",
        "cache_prefill_segment_id",
        nn.with_logical_partitioning(jnp.zeros, segment_id_axis_names),
        (cache_logical_shape[0], cache_length),
        jnp.int32,
    )

    if self.kv_quant:
      cache_scale_logical_shape = self._get_cache_scale_logical_shape(batch, heads, cache_length)
      cache_scale_axis_names = self.transpose_tuple(self.cache_scale_logical_axis_names, self.prefill_cache_axis_order)
      cache_scale_shape = self.transpose_tuple(cache_scale_logical_shape, self.prefill_cache_axis_order)

      cached_key_scale_var = self.variable(
          "cache",
          "cached_prefill_key_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
      cached_value_scale_var = self.variable(
          "cache",
          "cached_prefill_value_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
    else:
      cached_key_scale_var = None
      cached_value_scale_var = None

    key_vars = (cached_key_var, cached_key_scale_var)
    value_vars = (cached_value_var, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id_var

  def _get_ar_cache_vars(self, batch, heads, kv_head_size, model_mode):

    dtype = self._get_cached_kv_dtype(self.dtype)
    cache_length = self.max_target_length - self.max_prefill_predict_length
    cache_logical_shape = (batch, cache_length, heads, kv_head_size)

    if model_mode == common_types.MODEL_MODE_PREFILL:
      cache_logical_axis_names = self.prefill_cache_logical_axis_names
    else:
      cache_logical_axis_names = self.cache_logical_axis_names

    cache_axis_names = self.transpose_tuple(cache_logical_axis_names, self.ar_cache_axis_order)
    cache_shape = self.transpose_tuple(cache_logical_shape, self.ar_cache_axis_order)

    # TODO(b/339703100): investigate the issue why with_logical_partitioning doesn't enforce sharding
    cached_key_var = self.variable(
        "cache",
        "cached_ar_key",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape,
        dtype,
    )
    cached_key_var.value = nn.with_logical_constraint(
        cached_key_var.value,
        cache_axis_names,
    )

    cached_value_var = self.variable(
        "cache",
        "cached_ar_value",
        nn.with_logical_partitioning(jnp.zeros, cache_axis_names),
        cache_shape,
        dtype,
    )
    cached_value_var.value = nn.with_logical_constraint(
        cached_value_var.value,
        cache_axis_names,
    )

    if model_mode == common_types.MODEL_MODE_PREFILL:
      segment_id_axis_names = (CACHE_BATCH_PREFILL, CACHE_SEQUENCE)
    else:
      segment_id_axis_names = (CACHE_BATCH, CACHE_SEQUENCE)
    cached_segment_id_var = self.variable(
        "cache",
        "cache_ar_segment_id",
        nn.with_logical_partitioning(jnp.zeros, segment_id_axis_names),
        (cache_logical_shape[0], cache_length),
        jnp.int32,
    )

    cached_lengths_var = self.variable(
        "cache",
        "cached_ar_lengths",
        nn.with_logical_partitioning(jnp.zeros, (CACHE_BATCH,)),
        (cache_logical_shape[0],),
        jnp.int32,
    )

    if self.kv_quant:
      cache_scale_logical_shape = self._get_cache_scale_logical_shape(batch, heads, cache_length)
      cache_scale_axis_names = self.transpose_tuple(self.cache_scale_logical_axis_names, self.ar_cache_axis_order)
      cache_scale_shape = self.transpose_tuple(cache_scale_logical_shape, self.ar_cache_axis_order)

      cached_key_scale_var = self.variable(
          "cache",
          "cached_ar_key_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
      cached_value_scale_var = self.variable(
          "cache",
          "cached_ar_value_scale",
          nn.with_logical_partitioning(jnp.zeros, cache_scale_axis_names),
          cache_scale_shape,
          jnp.bfloat16,
      )
    else:
      cached_key_scale_var = None
      cached_value_scale_var = None

    cache_index_var = self.variable("cache", "cache_ar_index", nn.with_logical_partitioning(jnp.zeros, ()), (1,), jnp.int32)
    key_vars = (cached_key_var, cached_key_scale_var)
    value_vars = (cached_value_var, cached_value_scale_var)
    return key_vars, value_vars, cached_segment_id_var, cache_index_var, cached_lengths_var

  def kv_cache_prefill(
      self,
      key: Array,
      value: Array,
      decoder_segment_ids: Array,
  ):
    """In prefill mode, we zero out the existing cache, run the computation and
    prepare the cache as necessary.

    Args:
      key: in shape [b, s, n, d].
      value: in shape [b, s, n, d].
      decoder_segment_ids: [b, s] -- marking segment ids for tokens

    Returns:
      key, value, decoder_segment_id.

    """
    batch, _, heads, kv_head_size = key.shape
    assert key.dtype == value.dtype, "Key and Value Dtypes should match."

    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(
        batch, heads, kv_head_size, common_types.MODEL_MODE_PREFILL
    )
    # TODO: Find a way to not enable the ar cache for prefill mode.
    _ = self._get_ar_cache_vars(batch, heads, kv_head_size, common_types.MODEL_MODE_PREFILL)  # initialize it now

    key_shaped_for_cache = jnp.transpose(key, self.prefill_cache_axis_order)
    value_shaped_for_cache = jnp.transpose(value, self.prefill_cache_axis_order)

    if self.kv_quant:
      prefill_key_axis_names = self.transpose_tuple(self.cache_logical_axis_names, self.prefill_cache_axis_order)
      key_shaped_for_cache, key_scale_shaped_for_cache = self.kv_quant.quantize(key_shaped_for_cache, prefill_key_axis_names)
      value_shaped_for_cache, value_scale_shaped_for_cache = self.kv_quant.quantize(
          value_shaped_for_cache, prefill_key_axis_names
      )
      cached_prefill_key_vars[1].value = key_scale_shaped_for_cache
      cached_prefill_value_vars[1].value = value_scale_shaped_for_cache

    cached_prefill_key_vars[0].value = key_shaped_for_cache
    cached_prefill_value_vars[0].value = value_shaped_for_cache

    if decoder_segment_ids is not None:
      cached_prefill_segment_id_var.value = decoder_segment_ids

    return key, value, decoder_segment_ids

  def update_ar_key_value(
      self,
      one_token_key: Array,
      one_token_value: Array,
      cached_key_vars: tuple[nn.Variable, nn.Variable | None],
      cached_value_vars: tuple[nn.Variable, nn.Variable | None],
      one_hot_indices: Array,
      lengths: Array,
      use_ragged_attention: bool,
  ) -> None:
    """Adds a single token's results to the ar kv cache

    Args:
        one_token_key (Array): Key of one token to add to the cache
        one_token_value (Array): Value of one token to add to the cache
        cached_ar_key (tuple[nn.Variable, nn.Variable|None],): Cached keys to add new token key to, possibly with scale
        cached_ar_value (tuple[nn.Variable, nn.Variable|None],: Cached values to add new token value to, possible with scale
        one_hot_indices (Array): Location of the new token within the cache

    Returns:
        tuple[Array, Array]: Updated caches for key and value with new token info added
    """

    cached_key_var, cached_key_scale_var = cached_key_vars
    cached_value_var, cached_value_scale_var = cached_value_vars

    # In order to update the key, value caches with the current key and
    # value, we reshape the one_token_key and one_token_value
    one_token_key_shaped_for_cache = jnp.transpose(one_token_key, self.ar_cache_axis_order)
    one_token_value_shaped_for_cache = jnp.transpose(one_token_value, self.ar_cache_axis_order)

    ar_cache_axis_names = self.transpose_tuple(self.cache_logical_axis_names, self.ar_cache_axis_order)
    if self.kv_quant:
      one_token_key_shaped_for_cache, one_token_key_scale_shaped_for_cache = self.kv_quant.quantize(
          one_token_key_shaped_for_cache, ar_cache_axis_names
      )
      one_token_value_shaped_for_cache, one_token_value_scale_shaped_for_cache = self.kv_quant.quantize(
          one_token_value_shaped_for_cache, ar_cache_axis_names
      )

    ar_cache_update_idx = jnp.squeeze(one_hot_indices)
    ar_cache_sequence_axis = ar_cache_update_axis = ar_cache_axis_names.index(CACHE_SEQUENCE)
    ar_cache_batch_axis = ar_cache_axis_names.index(CACHE_BATCH)

    if use_ragged_attention:
      cache_locations = [slice(None)] * 4
      new_token_locations = [slice(None)] * 4
      new_token_locations[ar_cache_sequence_axis] = 0

      def key_body(i, val):
        cache_locations[ar_cache_batch_axis] = i
        cache_locations[ar_cache_sequence_axis] = lengths[i]
        new_token_locations[ar_cache_batch_axis] = i
        return val.at[tuple(cache_locations)].set(one_token_key_shaped_for_cache[tuple(new_token_locations)])

      def value_body(i, val):
        cache_locations[ar_cache_batch_axis] = i
        cache_locations[ar_cache_sequence_axis] = lengths[i]
        new_token_locations[ar_cache_batch_axis] = i
        return val.at[tuple(cache_locations)].set(one_token_value_shaped_for_cache[tuple(new_token_locations)])

      cached_key_var.value = jax.lax.fori_loop(
          0, one_token_key_shaped_for_cache.shape[0], key_body, cached_key_var.value, unroll=8
      )
      cached_value_var.value = jax.lax.fori_loop(
          0, one_token_value_shaped_for_cache.shape[0], value_body, cached_value_var.value, unroll=8
      )

    else:
      one_hot_indices = one_hot_indices.astype(int)
      cached_key_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_key_var.value, one_token_key_shaped_for_cache, ar_cache_update_idx, ar_cache_update_axis
      )
      cached_value_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_value_var.value, one_token_value_shaped_for_cache, ar_cache_update_idx, ar_cache_update_axis
      )

    cached_key_var.value = nn.with_logical_constraint(cached_key_var.value, ar_cache_axis_names)
    cached_value_var.value = nn.with_logical_constraint(cached_value_var.value, ar_cache_axis_names)

    if self.kv_quant:
      ar_cache_scale_axis_names = self.transpose_tuple(self.cache_scale_logical_axis_names, self.ar_cache_axis_order)
      ar_cache_scale_update_axis = ar_cache_scale_axis_names.index(CACHE_SCALE_SEQUENCE)
      cached_key_scale_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_key_scale_var.value, one_token_key_scale_shaped_for_cache, ar_cache_update_idx, ar_cache_scale_update_axis
      )
      cached_value_scale_var.value = jax.lax.dynamic_update_index_in_dim(
          cached_value_scale_var.value,
          one_token_value_scale_shaped_for_cache,
          ar_cache_update_idx,
          ar_cache_scale_update_axis,
      )

    return

  def get_cached_values(self, cache_vars, target_dtype, cache_axis_order) -> jax.Array | KVTensor:
    cache_var, cache_scale_var = cache_vars
    cache_value = cache_var.value
    if cache_scale_var is not None:
      scale_value = cache_scale_var.value
      dtype = cache_value.dtype
      if dtype == jnp.int8:
        scale_value /= quantizations.MAX_INT8
      elif dtype == jnp.int4:
        scale_value /= quantizations.MAX_INT4

      cache_value = KVTensor(qvalue=cache_value, scale=[scale_value], scale_t=None, dequant_dtype=target_dtype, bias=[])
    cache_value_in_logical_shape = jax.tree.map(lambda x: self.reverse_transepose(x, cache_axis_order), cache_value)
    return cache_value_in_logical_shape

  def kv_cache_autoregressive(
      self,
      key: Array,
      value: Array,
      use_ragged_attention: bool = False,
  ):
    """In autoregressive mode, we update the cache for this entry and
       then return the full cache.

    Args:
      key: in shape [b, 1, n, d].
      value: in shape [b, 1, n, d].
      decoder_segment_ids: [b, 1] -- marking segment ids for tokens

    Returns:
      tuple of (key, value, segment_id) for both prefill and ar cache,
    Raises:
      ValueError: when key/value shape is not [batch, 1, num_heads, heads_dim].
    """
    batch, sequence, heads, kv_head_size = key.shape
    if sequence != 1:
      raise ValueError(f"Sequence length should be 1 during autoregression, got {sequence=}")

    cached_ar_key_vars, cached_ar_value_vars, cached_ar_segment_id_var, cache_ar_index_var, cache_ar_lengths_var = (
        self._get_ar_cache_vars(batch, heads, kv_head_size, common_types.MODEL_MODE_AUTOREGRESSIVE)
    )

    self.update_ar_key_value(
        key,
        value,
        cached_ar_key_vars,
        cached_ar_value_vars,
        cache_ar_index_var.value,
        cache_ar_lengths_var.value,
        use_ragged_attention,
    )
    active_indicator = jnp.zeros((batch, 1), dtype=jnp.int32) + common_types.DECODING_ACTIVE_SEQUENCE_INDICATOR
    cached_ar_segment_id_var.value = jax.lax.dynamic_update_index_in_dim(
        cached_ar_segment_id_var.value, active_indicator, jnp.squeeze(cache_ar_index_var.value), 1
    )
    cache_ar_index_var.value = jnp.mod(
        cache_ar_index_var.value + 1, self.max_target_length - self.max_prefill_predict_length
    )
    cache_ar_lengths_var.value = cache_ar_lengths_var.value.at[:].add(1)

    # The below retrieves the existing prefill cache variables, not creating new ones
    cached_prefill_key_vars, cached_prefill_value_vars, cached_prefill_segment_id_var = self._get_prefill_cache_vars(
        batch, heads, kv_head_size, common_types.MODEL_MODE_AUTOREGRESSIVE
    )

    cached_prefill = (
        self.get_cached_values(cached_prefill_key_vars, key.dtype, self.prefill_cache_axis_order),
        self.get_cached_values(cached_prefill_value_vars, value.dtype, self.prefill_cache_axis_order),
        cached_prefill_segment_id_var.value,
    )

    cached_ar = (
        self.get_cached_values(cached_ar_key_vars, key.dtype, self.ar_cache_axis_order),
        self.get_cached_values(cached_ar_value_vars, value.dtype, self.ar_cache_axis_order),
        cached_ar_segment_id_var.value,
        cache_ar_lengths_var.value,
    )
    return cached_prefill, cached_ar

  def kv_cache(
      self, key: Array, value: Array, decoder_segment_ids: Array, model_mode: str, use_ragged_attention: bool = False
  ) -> tuple:
    """KV cache takes the current state and updates the state accordingly.

    The key and value have dimension [b, s, n_kv, d],
    but we cache them with a reshape as defined in *_axis_order config as a TPU
    fusion optimization. This also enables the "scatter via one-hot
    broadcast" trick, which means we do a one-hot broadcast instead of a
    scatter/gather operations, resulting in a 3-4x speedup in practice.

    Args:
      key: in shape [b, s, n_kv, d].
      value: in shape [b, s, n_kv, d].
      model_mode: model mode controlling model

    Returns:
      two tuples of (k, v, decoder_segments) -- either can be Nones

    """
    if key.shape != value.shape and self.config.attention_type != AttentionType.MLA.value:
      raise ValueError(f"Can't KV cache with mismatched shapes {key.shape=}, {value.shape=}")

    if model_mode == common_types.MODEL_MODE_TRAIN:
      return (key, value, decoder_segment_ids), None
    elif model_mode == common_types.MODEL_MODE_PREFILL:
      return self.kv_cache_prefill(key, value, decoder_segment_ids), None
    elif model_mode == common_types.MODEL_MODE_AUTOREGRESSIVE:
      return self.kv_cache_autoregressive(key, value, use_ragged_attention)
    else:
      raise ValueError(f"Model Mode isn't supported! {model_mode=}")

  def normalize_attention(self, local_outs, local_maxes, local_sums):
    """Normalize across multiple localized attentions

    Args:
        local_outs (list): List of unnormalized outputs entries for each local attention
        local_maxes (list): List of max exponentials entries for each local attention
        local_sums (list): List of exponential sum entries for each local attention

    Returns:
        Array: Combined attention that has been normalized
    """
    # Based on https://github.com/google-research/google-research/blob/master/scaling_transformer_inference_efficiency/attention.py
    global_max = functools.reduce(jnp.maximum, local_maxes)
    global_sum = sum(
        [jnp.exp(local_max - global_max) * local_sum for (local_sum, local_max) in zip(local_sums, local_maxes)]
    )

    attn_out = 0
    for local_max, local_out in zip(local_maxes, local_outs):
      local_normalizer = jnp.exp(local_max - global_max) / global_sum
      attn_out += local_normalizer * local_out
    return attn_out

  @nn.compact
  def __call__(self, query, key, value, decoder_segment_ids, model_mode, *args, eos_sum=None): # lsp
    prefill_kv_cache, ar_kv_cache = self.kv_cache(
        key, value, decoder_segment_ids, model_mode, use_ragged_attention=self.use_ragged_attention
    )

    prefill_unnormalized_output, prefill_exponentials_max, prefill_exponentials_sum = self.apply_attention(
        query=query,
        key=prefill_kv_cache[0],
        value=prefill_kv_cache[1],
        decoder_segment_ids=prefill_kv_cache[2],
        lengths=None,
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
    )

    # Return the "prefill" cache if it actually the combined prefill+ar kv cache
    if ar_kv_cache is None:
      if prefill_exponentials_sum is not None:
        return prefill_unnormalized_output / prefill_exponentials_sum
      return prefill_unnormalized_output

    ar_unnormalized_output, ar_exponentials_max, ar_exponentials_sum = self.apply_attention(
        query=query,
        key=ar_kv_cache[0],
        value=ar_kv_cache[1],
        decoder_segment_ids=ar_kv_cache[2],
        lengths=ar_kv_cache[3],
        model_mode=model_mode,
        use_ragged_attention=self.use_ragged_attention,
    )

    if ar_unnormalized_output is not None:
      unnormalized_outputs = [prefill_unnormalized_output, ar_unnormalized_output]
      exponentials_maxes = [prefill_exponentials_max, ar_exponentials_max]
      exponentials_sums = [prefill_exponentials_sum, ar_exponentials_sum]
      return self.normalize_attention(unnormalized_outputs, exponentials_maxes, exponentials_sums)
    else:
      return prefill_unnormalized_output / prefill_exponentials_sum


class Attention(nn.Module):
  """Generic Attention.

  Attributes:
    num_query_heads: number of query attention heads. Features (i.e. inputs_q.shape[-1])
      should be divisible by the number of heads.
    num_kv_heads: number of kv attention heads.
    head_dim: dimension of each head.
    mesh: Mesh, device mesh
    attention_kernel: str, guidance on if we should use an attention kernel
    dtype: the dtype of the computation.
    weight_dtype: the dtype of the weights.
    max_target_length: maximum target length
    max_prefill_predict_length: size of the maximum prefill
    dropout_rate: dropout rate
    kernel_init: initializer for the kernel of the Dense layers.
    float32_qk_product: bool, if True then compute logits via float32 qk_product to avoid
      numerical issues with bfloat16.
    float32_logits: bool, if True then cast logits to float32 before softmax to avoid
      numerical issues with bfloat16.
    quant: Quant, stores quantization parameters, defaults to None implying no quantization.
    kv_quant: KVQuant, stores KV cache quantization parameters, defaults to None
  """

  config: Config
  num_query_heads: int
  num_kv_heads: int
  head_dim: int
  max_target_length: int
  mesh: Mesh
  attention_kernel: str
  dtype: DType = jnp.float32
  weight_dtype: DType = jnp.float32
  max_prefill_predict_length: int = -1
  dropout_rate: float = 0.0
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  output_kernel_init: NdInitializer | None = None
  output_bias: bool = False
  rotary_dim: int | None = None  # Prefix width; None preserves the legacy full-head path.
  float32_qk_product: bool = False  # computes logits in float32 for stability.
  float32_logits: bool = False  # cast logits in float32 for stability.
  quant: Optional[Quant] = None
  kv_quant: Optional[KVQuant] = None

  attention_type: AttentionType = AttentionType.GLOBAL  # Default to global attention
  attn_logits_soft_cap: float | None = None
  sliding_window_size: int | None = None
  use_ragged_attention: bool = False
  ragged_block_size: int = 256

  # Shard the query activation as the same as the key and value.
  # TODO: Find a better sharding axis name.
  # TODO: Further break down the Training and Inference axes for the q, k, v.
  prefill_query_axis_names: AxisNames = (PREFILL_KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  prefill_key_axis_names: AxisNames = (PREFILL_KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  prefill_value_axis_names: AxisNames = (PREFILL_KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  query_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  input_axis_names: AxisNames = (BATCH, LENGTH, EMBED)
  key_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  value_axis_names: AxisNames = (KV_BATCH, LENGTH, KV_HEAD, KV_HEAD_DIM)
  out_axis_names: AxisNames = (BATCH, LENGTH, HEAD, D_KV)

  prefill_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  ar_cache_axis_order: AxisIdxes = (1, 2, 0, 3)
  compute_axis_order: AxisIdxes = (0, 1, 2, 3)
  reshape_q: bool = False
  use_kv_shift: bool = False

  def setup(self):
    if (self.config.pre_compose or self.config.post_compose) \
      and (self.sliding_window_size < self.config.max_target_length or self.attention_kernel == "dot_product_chunk"):
      max_logging.log(f'sws: {self.sliding_window_size} use dc chunk-{self.config.query_chunk_size} attn.', debug=self.config.debug)
      self.attention_op = dc.AttentionOp(self.config, self.quant, self.sliding_window_size)
    else:
      max_logging.log(f'sws: {self.sliding_window_size} use {self.attention_kernel} attn.', debug=self.config.debug)
      self.attention_op = AttentionOp(
        config=self.config,
        mesh=self.mesh,
        attention_kernel=self.attention_kernel,
        max_target_length=self.max_target_length,
        max_prefill_predict_length=self.max_prefill_predict_length,
        float32_qk_product=self.float32_qk_product,
        float32_logits=self.float32_logits,
        quant=self.quant,
        kv_quant=self.kv_quant,
        num_query_heads=self.num_query_heads,
        num_kv_heads=self.num_kv_heads,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        prefill_cache_axis_order=self.prefill_cache_axis_order,
        ar_cache_axis_order=self.ar_cache_axis_order,
        compute_axis_order=self.compute_axis_order,
        reshape_q=self.reshape_q,
        attention_type=self.attention_type,
        attn_logits_soft_cap=self.attn_logits_soft_cap,
        sliding_window_size=self.sliding_window_size,
        use_ragged_attention=self.use_ragged_attention,
        ragged_block_size=self.ragged_block_size,
    )
    if self.use_kv_shift:
      self.kv_shift = kv_shift.KVshift(config=self.config,mesh=self.mesh, quant=self.quant, kernel_init=self.kernel_init, num_kv_heads=self.num_kv_heads)


  def query_projection(self, inputs_q: Array, projection_dim=None) -> Array:
    """Query projection."""

    projection_dim = self.head_dim if projection_dim is None else projection_dim
    if self.config.opt_type == 'muon':
      kernel_axes=("embed", "mlp")
      features = (self.num_query_heads * projection_dim, )
    else:
      kernel_axes=("embed", "q_heads", "kv")
      features = (self.num_query_heads, projection_dim)

    b, t, h = inputs_q.shape

    query_proj = DenseGeneral(
        features=features,
        axis=-1,
        kernel_init=self.kernel_init, # lsp
        kernel_axes=kernel_axes,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="query",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.config.qkv_bias,
    )
    output = query_proj(inputs_q)
    max_logging.log(f'output: {output.shape}', debug=self.config.debug)
    if self.config.opt_type == 'muon':
      output = output.reshape(b, t, self.num_query_heads, projection_dim)
    return output

  def kv_projection(self, inputs_kv: Array, proj_name: str, projection_dim=None, num_heads=None) -> Array:
    """Projection for Key and Value.

    Args:
      inputs_kv: inputs_kv: key/values of shape `[batch, kv_length,
        num_kv_heads, kv_dim]`.
      proj_name: name of projection, `key` or `value`.

    Returns:
      Projection of key or value, in shape of `[batch, kv_length, head_dim]`.
    """
    projection_dim = self.head_dim if projection_dim is None else projection_dim
    b, t, h = inputs_kv.shape

    if self.num_kv_heads == -1:
      raise ValueError("num_kv_heads is not defined.")

    if self.num_query_heads % self.num_kv_heads != 0:
      raise ValueError("Invalid num_kv_heads for GQA.")

    num_kv_heads = self.num_kv_heads if num_heads is None else num_heads
    if self.config.opt_type == 'muon':
      kernel_axes = ("embed", "mlp")
      features=(num_kv_heads * projection_dim, )
    else:
      kernel_axes = ("embed", "kv_heads", "kv_head_dim")
      features=(num_kv_heads, projection_dim)
    kv_proj = DenseGeneral(
        features=features,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=kernel_axes,
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name=proj_name,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.config.qkv_bias,
    )(inputs_kv)
    if self.config.opt_type == 'muon':
      kv_proj = kv_proj.reshape(b, t, num_kv_heads, projection_dim)
    return kv_proj

  def qkv_projection(self, inputs: Array, proj_name: str):
    """Fused QKV projection"""

    qkv_proj = DenseGeneral(
        features=(3, self.num_query_heads, self.head_dim),
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "qkv", "heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name=proj_name,
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
        use_bias=self.config.qkv_bias,
    )(inputs)
    qkv_proj = checkpoint_name(qkv_proj, "qkv_proj")
    query, key, value = qkv_proj[:, :, 0, ...], qkv_proj[:, :, 1, ...], qkv_proj[:, :, 2, ...]
    return query, key, value

  def out_projection(self, output_dim: int, out: Array) -> Array:
    output_init = self.output_kernel_init or self.kernel_init
    if self.config.opt_type == 'muon':
      kernel_shape = (self.head_dim * self.num_query_heads, output_dim)
      kernel_in_axis = (0, )
      kernel_out_axis = (1, )
      kernel = self.param(
          "out",
          nn.with_logical_partitioning(output_init, ("mlp", "embed")),
          kernel_shape,
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
      kernel = kernel.reshape(self.num_query_heads, self.head_dim, output_dim)
      kernel = jnp.asarray(kernel, self.dtype)
      out_proj = jnp.einsum("btnh,nhd->btd", out, kernel)
      if self.output_bias:
        out_proj += self.param("out_bias", nn.initializers.zeros,
                               (output_dim,), self.weight_dtype).astype(self.dtype)
    else:
      out_proj = DenseGeneral(
          features=output_dim,
          axis=(-2, -1),
          kernel_init=output_init,
          kernel_axes=("heads", "kv", "embed"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="out",
          use_bias=self.output_bias,
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
      )(out)
    return out_proj

  def apply_rotary_embedding(
      self,
      inputs: Array,
      inputs_positions: Array,
      name: str,
      embedding_dims: int | None = None,
  ):
    """Applies rotary embeddings, handling different model types.

    Args:
      inputs: The input tensor to apply rotary embeddings to.
      inputs_positions: The positions of the inputs.
      name: A name for the embedding layer.
      embedding_dims: Optional rotary width when rotating a sliced subspace.

    Returns:
      The input tensor with rotary embeddings applied.
    """
    if embedding_dims is not None:
      rope_embedding_dims = embedding_dims
    elif self.config.attention_type == AttentionType.MLA.value:
      # For MLA attention RoPE is applied to only `self.qk_rope_head_dim` portion the heads.
      rope_embedding_dims = self.qk_rope_head_dim
    else:
      rope_embedding_dims = self.head_dim

    rope_type = self.config.rope_type.lower()
    if self.config.model_name.startswith("llama3.1") or rope_type.startswith("llama3.1"):
      rotary_embedding = embeddings.LLaMARotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.config.rope_max_timescale,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          name=name,
      )
    elif rope_type.startswith("yarn"):
      rotary_embedding = YarnRotaryEmbedding(
          max_seq_len=self.config.max_target_length,
          original_seq_len=self.config.original_seq_len,
          beta_fast=self.config.beta_fast,
          beta_slow=self.config.beta_slow,
          rope_theta=self.config.rope_theta,
          rope_factor=self.config.rope_factor,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          name=name,
      )
    else:
      rotary_embedding = RotaryEmbedding(
          min_timescale=self.config.rope_min_timescale,
          max_timescale=self.config.rope_max_timescale,
          embedding_dims=rope_embedding_dims,
          fprop_dtype=self.dtype,
          rope_half=self.config.rope_half,
          name=name,
      )
    inputs = rotary_embedding(inputs, inputs_positions)
    return inputs

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array,
      decoder_segment_ids: Array | None = None,
      decoder_input_tokens: Array | None = None,
      *,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
      deterministic: bool = False,
      eos_sum: Array | None = None,
      deep_embedding: Array | None = None,
  ):
    """Applies Attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    There are three modes: training, prefill and autoregression. During training, the KV cache
    is ignored. During prefill, the cache is filled. During autoregression the cache is used.

    In the cache initialization call, `inputs_q` has a shape [batch, length,
    q_features] and `inputs_kv`: [batch, length, kv_features]. During the
    incremental decoding stage, query, key and value all have the shape [batch,
    1, qkv_features] corresponding to a single step.

    Args:
      inputs_q: input queries of shape `[batch, q_length, q_features]`.
      inputs_kv: key/values of shape `[batch, kv_length, kv_features]`.
      model_mode: corresponding to train, prefill and decode.
      deterministic: Disables dropout if set to True.

    Returns:
      output of shape `[batch, length, q_features]`.
    """
    cfg = self.config
    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)

    # apply projection.
    if cfg.fused_qkv:
      query, key, value = self.qkv_projection(inputs_q, proj_name="qkv_proj")
    elif cfg.dense_conn and cfg.dynamic_dense_type == 'qkvm':
        assert isinstance(inputs_kv, (tuple, list)) and len(inputs_kv) == 2
        inputs_k, inputs_v = inputs_kv
        max_logging.log(f'inputs_q: {inputs_q.shape} inputs_k: {inputs_k.shape} inputs_v: {inputs_v.shape}', debug=cfg.debug)
        query = self.query_projection(inputs_q)
        key = self.kv_projection(inputs_k, proj_name="key")
        value = self.kv_projection(inputs_v, proj_name="value")
    else:
      query = self.query_projection(inputs_q)
      key = self.kv_projection(inputs_kv, proj_name="key")
      value = self.kv_projection(inputs_kv, proj_name="value")

    if self.use_kv_shift:
      inputs_k, inputs_v = inputs_kv if isinstance(inputs_kv, (tuple, list)) and len(inputs_kv) == 2 else (inputs_kv, inputs_kv)
      query, key, value = self.kv_shift(inputs_q, query, key, value, inputs_k=inputs_k, inputs_v=inputs_v)

    query, key = dc.QKNorm(cfg, name='qk_norm')(query, key) # lsp

    # A prefix-only RoPE for NeoX/Pythia; leave legacy full-head calls unchanged.
    if self.rotary_dim is None:
      query = self.apply_rotary_embedding(query, inputs_positions, name="query_rotary")
      key = self.apply_rotary_embedding(key, inputs_positions, name="key_rotary")
    else:
      width = self.rotary_dim
      if not isinstance(width, int) or width < 0 or width > self.head_dim or width % 2:
        raise ValueError("rotary_dim must be an even integer in [0, head_dim]")
      if width:
        query = jnp.concatenate((self.apply_rotary_embedding(
            query[..., :width], inputs_positions, name="query_rotary", embedding_dims=width),
            query[..., width:]), axis=-1)
        key = jnp.concatenate((self.apply_rotary_embedding(
            key[..., :width], inputs_positions, name="key_rotary", embedding_dims=width),
            key[..., width:]), axis=-1)

    if model_mode == common_types.MODEL_MODE_PREFILL:
      query = nn.with_logical_constraint(query, self.prefill_query_axis_names)
      key = nn.with_logical_constraint(key, self.prefill_key_axis_names)
      value = nn.with_logical_constraint(value, self.prefill_value_axis_names)
    else:
      query = nn.with_logical_constraint(query, self.query_axis_names)
      key = nn.with_logical_constraint(key, self.key_axis_names)
      value = nn.with_logical_constraint(value, self.value_axis_names)
    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    # DE value
    if 'devalue' in cfg.deep_embed_type.lower():
      b, t, n, d = value.shape
      de_value = value.reshape(b, t, n * d)
      de_inputs = inputs_kv if isinstance(inputs_kv, jnp.ndarray) else inputs_kv[1]
      de_inputs = de_inputs.reshape(*de_inputs.shape[:2], -1)
      value = linears.DeepEmbedBlock(
        name='value_deep_embed',
        config=cfg,
        kernel_init=initializers.get_init_method(cfg.init_method),
        weight_dtype=cfg.weight_dtype,
        dtype=cfg.dtype,
        input_dim=cfg.emb_dim,
        output_dim=n * d,
        )(de_inputs,
          de_value,
          decoder_input_tokens,
          deep_embedding=deep_embedding
          )
      value = value.reshape(b, t, n, d)
      max_logging.log(f'Outside DE is None, inside value DE')

    assert not cfg.quantize_kvcache or self.kv_quant

     # lsp
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query /= depth_scaling

    if self.num_query_heads > self.num_kv_heads and self.sliding_window_size < cfg.max_target_length: # local laeyr GQA
      assert self.num_query_heads % self.num_kv_heads == 0
      n_expands = self.num_query_heads // self.num_kv_heads
      if key.shape[-2] < self.num_query_heads: # for K lora
        key = jnp.repeat(key, n_expands, axis=-2) # BSNd
      if value.shape[-2] < self.num_query_heads:  # for V lora
        value = jnp.repeat(value, n_expands, axis=-2)

    if self.config.use_v_gate:
      assert value.shape[-2] == self.num_query_heads
      v_gate = DenseGeneral((self.num_query_heads,),dtype=self.dtype,weight_dtype=self.weight_dtype,quant=self.quant,
            kernel_init=self.kernel_init,kernel_axes=('embed', None),name="v_gate", use_bias=False,
        )(inputs_q)
      v_gate = jax.nn.tanh(v_gate) + 1
      value = value * v_gate[...,None] # BSND, BSN1->BSND

    print(f'query: {query.shape} key: {key.shape} value: {value.shape}')
    out = self.attention_op(query, key, value, decoder_segment_ids, model_mode, inputs_q, inputs_kv, eos_sum=eos_sum)

    out = nn.with_logical_constraint(out, self.out_axis_names)

    # apply output projection,  output dim is set to the input dim.
    out = self.out_projection(inputs_q.shape[-1], out)
    out = checkpoint_name(out, "out_proj")
    return out


class MLA(Attention):
  """Multi-Head Latent Attention (MLA) layer."""

  q_lora_rank: int = 0
  kv_lora_rank: int = 512
  qk_nope_head_dim: int = 128
  qk_rope_head_dim: int = 64
  v_head_dim: int = 128
  max_seq_len: int = 4096 * 4
  original_seq_len: int = 4096
  mscale: float = 1.0  # scaling factor for softmax
  rope_factor: float = 40.0  # rotary embedding factor

  @property
  def qk_head_dim(self) -> int:
    return self.qk_nope_head_dim + self.qk_rope_head_dim

  def setup(self):
    """Initialize MLA-specific parameters."""
    super().setup()

    # Assert required configuration parameters for MLA attention.
    assert (
        self.config.attention_type == AttentionType.MLA.value
    ), f"MLA requires MLA attention type {AttentionType.MLA.value}"
    assert self.kv_lora_rank > 0, "KV LoRA rank must be > 0"
    assert self.qk_nope_head_dim > 0, "QK NoPe head dim must be > 0"
    assert self.qk_rope_head_dim > 0, "QK RoPE head dim must be > 0"
    assert self.v_head_dim > 0, "V head dim must be > 0"
    assert self.num_query_heads == self.num_kv_heads, "MLA requires equal number of query and kv heads"
    assert not self.config.fused_qkv, "Fused QKV is not supported for MLA"

    if self.q_lora_rank == 0:
      # Standard Q projection (without LoRA).
      self.query_proj = DenseGeneral(
          features=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_heads", "kv"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="query",
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
      )
    else:
      # LoRA path for Q.
      self.wq_a = DenseGeneral(
          features=self.q_lora_rank,
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "q_lora"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="wq_a",
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
      )
      self.q_norm = normalizations.get_rmsnorm("q_norm", self.config)
      self.wq_b = DenseGeneral(
          features=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          # kernel_axes=("q_lora", "q_heads", "kv"),
          kernel_axes=("q_lora", "q_heads", "embed"), # lsp
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="wq_b",
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
      )

    # KV LoRA path.
    self.wkv_a = DenseGeneral(
        features=self.kv_lora_rank + self.qk_rope_head_dim,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv_lora"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="wkv_a",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )
    self.kv_norm = normalizations.get_rmsnorm("kv_norm", self.config)
    self.wkv_b = DenseGeneral(
        features=(self.num_query_heads, (self.qk_nope_head_dim + self.v_head_dim)),
        axis=-1,
        kernel_init=self.kernel_init,
        # kernel_axes=("kv_lora", "kv_heads", "kv_head_dim"),
        kernel_axes=("kv_lora", "kv_heads", "embed"), # lsp
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="wkv_b",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )

    # Set softmax scaling.
    self.softmax_scale = self.qk_head_dim**0.5 # lsp, -0.5 -> 0.5
    if self.max_seq_len > self.original_seq_len:
      mscale = 0.1 * self.mscale * jnp.log(self.rope_factor) + 1.0
      self.softmax_scale = self.softmax_scale * mscale * mscale

  def mla_query_projection(self, inputs_q: Array, inputs_positions: Array) -> Array:
    """Query projection for MLA, e.g. includes LoRA if q_lora_rank > 0."""
    if self.q_lora_rank == 0:
      q = self.query_proj(inputs_q)
    else:
      # LoRA path
      low_rank_q = self.wq_a(inputs_q)  # [B, L, q_lora_rank]
      low_rank_q = self.q_norm(low_rank_q)  # RMSNorm on low rank
      q = self.wq_b(low_rank_q)  # [B, L, n_heads * qk_head_dim]

    # Split into non-positional and rotary parts.
    q_nope, q_pe = jnp.split(q, [self.qk_nope_head_dim], axis=-1)
    q_pe = self.apply_rotary_embedding(q_pe, inputs_positions, name="query_rope")
    # Query projection is scaled by  1 / self.softmax_scale to be consistent MaxText implementation.
    # DeepSeek v3 was doing it in attention score computation.
    return jnp.concatenate([q_nope, q_pe], axis=-1) / self.softmax_scale

  def mla_kv_projection(self, inputs: Array, inputs_positions: Array) -> Tuple[Array, Array]:
    """MLA key/value projection with integrated rotary embedding."""
    low_rank = self.wkv_a(inputs)
    low_rank_main, low_rank_rope = jnp.split(low_rank, [self.kv_lora_rank], axis=-1)
    low_rank_main = self.kv_norm(low_rank_main)
    # Note: cache `low_rank_main` and `low_rank_rope` for inference.
    kv_out = self.wkv_b(low_rank_main)

    # Split kv_out into key_nope and value parts.
    key_nope, value = jnp.split(kv_out, [self.qk_nope_head_dim], axis=-1)

    # Apply rotary embedding to key_rope.
    key_rope = jnp.expand_dims(low_rank_rope, axis=2)
    key_rope = self.apply_rotary_embedding(key_rope, inputs_positions, name="key_rope")
    key_rope = jnp.broadcast_to(key_rope, (key_nope.shape[0], key_nope.shape[1], self.num_query_heads, key_rope.shape[3]))

    key = jnp.concatenate([key_nope, key_rope], axis=-1)
    return key, value

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array,
      decoder_segment_ids: Array | None = None,
      *,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
      deterministic: bool = False,
  ) -> Array:
    """Forward pass for MLA, reusing `AttentionOp` for the actual attention.

    Args:
      inputs_q: Query input [batch, q_length, embed_dim].
      inputs_kv: KV input   [batch, kv_length, embed_dim].
      inputs_positions: Positions for rotary embeddings or similar.
      decoder_segment_ids: Segment IDs for masking, if any.
      model_mode: "train", "prefill", or "autoregressive".
      deterministic: Disables dropout if set to True.

    Returns:
      A tensor of shape [batch, length, embed_dim] containing the
      MLA-attended outputs.
    """
    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)

    query = self.mla_query_projection(inputs_q, inputs_positions)
    key, value = self.mla_kv_projection(inputs_kv, inputs_positions)

    if model_mode == common_types.MODEL_MODE_PREFILL:
      query = nn.with_logical_constraint(query, self.prefill_query_axis_names)
      key = nn.with_logical_constraint(key, self.prefill_key_axis_names)
      value = nn.with_logical_constraint(value, self.prefill_value_axis_names)
    else:
      query = nn.with_logical_constraint(query, self.query_axis_names)
      key = nn.with_logical_constraint(key, self.key_axis_names)
      value = nn.with_logical_constraint(value, self.value_axis_names)

    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    out = self.attention_op(query, key, value, decoder_segment_ids, model_mode)
    out = nn.with_logical_constraint(out, self.out_axis_names)
    out = self.out_projection(inputs_q.shape[-1], out)
    return out


# ============================================================================
# BAM Attention (Bilinear Associative Memory)
# Design ref: bam_attention/DESIGN.md §4 / §4.6.5 / §7.4
# BAM train path: matrix write, factorized LocalQK, and dynamic full read; n == n_kv.
# ============================================================================

class GroupedRMSNorm(nn.Module):
  """RMSNorm with independent learned scales over explicit leading groups.

  Statistics are computed over the final feature axis only.  Unlike the usual
  RMSNorm, ``scale_shape`` may also include immediately preceding group axes;
  for example, ``(num_heads, head_dim)`` gives every head its own scale vector.
  The module is intentionally dormant until selected by a BAM experiment.
  """

  scale_shape: Tuple[int, ...]
  epsilon: float
  dtype: DType = jnp.float32
  statistics_dtype: DType = jnp.float32
  weight_dtype: DType = jnp.float32
  kernel_axes: Tuple[Optional[str], ...] = ()
  scale_init: Any = nn.initializers.zeros
  direct_scale: bool = False
  use_bias: bool = False
  bias_init: Any = nn.initializers.zeros

  @nn.compact
  def __call__(self, x: Array) -> Array:
    if tuple(x.shape[-len(self.scale_shape):]) != self.scale_shape:
      raise ValueError(
          f"GroupedRMSNorm expected trailing shape {self.scale_shape}, got {x.shape}")
    y = normalizations.rms_norm(
        x, dtype=self.dtype, epsilon=self.epsilon,
        statistics_dtype=self.statistics_dtype)
    if self.scale_init is not None:
      scale = self.param(
          "scale",
          nn.with_logical_partitioning(self.scale_init, self.kernel_axes),
          self.scale_shape,
          self.weight_dtype,
      )
      scale = jnp.asarray(scale, self.dtype)
      y = y * scale if self.direct_scale else y * (scale + 1.0)
    if self.use_bias:
      bias = self.param(
          "bias",
          nn.with_logical_partitioning(self.bias_init, self.kernel_axes),
          self.scale_shape,
          self.weight_dtype,
      )
      y = y + jnp.asarray(bias, self.dtype)
    return y


def _attention_op(
    query, key, value, valid, *, attn_logits_soft_cap=0.0,
    float32_logits=False, additive_bias=None, negative_diagonal_mask=None):
  """Apply masked QK/softmax/AV to one dense or query-chunk block."""
  with jax.named_scope("attention/qk_logits"):
    logits = jnp.einsum('bqnd,bsnd->bnqs', query, key)
  if attn_logits_soft_cap:
    logits = (
        jnp.tanh(logits / attn_logits_soft_cap) * attn_logits_soft_cap)
  if valid is not None:
    logits = jnp.where(valid[:, None], logits, DEFAULT_MASK_VALUE)
  if float32_logits:
    logits = logits.astype(jnp.float32)
  if additive_bias is not None:
    logits = logits + jnp.asarray(additive_bias, logits.dtype)
  if negative_diagonal_mask is not None:
    from layers.bam_attention_diagonal_zero import NEGATIVE_DIAGONAL_LOGIT
    logits = jnp.where(negative_diagonal_mask, jnp.asarray(NEGATIVE_DIAGONAL_LOGIT, logits.dtype), logits)
  with jax.named_scope("attention/softmax"):
    alpha = jax.nn.softmax(logits, axis=-1)
    if negative_diagonal_mask is not None:
      off_valid = (valid[:, None] if valid is not None else jnp.ones_like(alpha, dtype=jnp.bool_)) & ~negative_diagonal_mask
      alpha = jnp.where(jnp.any(off_valid, axis=-1, keepdims=True), alpha, jnp.asarray(0, alpha.dtype))
  with jax.named_scope("attention/av"):
    y_std = jnp.einsum('bnqs,bsnd->bqnd', alpha, value)
  return y_std, alpha


def _bam_splash_enabled(cfg, t):
  """Select Splash by configuration; short/non-aligned sequences use einsums."""
  # TPU AOT is traced on CPU: the host backend must not select the attention core.
  # CPU tests of the einsum path explicitly set bam_splash_attention=False.
  return bool(getattr(cfg, 'bam_splash_attention', True)) and t % 128 == 0


def _bam_splash_seq_minor(cfg, default):
  """SEQ_MINOR q/k/v (+6.9% pure JAX, -0.7% behind the fused read kernel on v5p-8); None = per-path default."""
  value = getattr(cfg, 'bam_splash_seq_minor', None)
  return default if value is None else bool(value)


def _bam_splash_attention(query, key, value, decoder_segment_ids, cfg, mesh, window=None, seq_minor=False):
  """Full causal BAM attention through the Splash kernel (MHA's flash path block sizes).

  query/key/value [B,T,N,D] (query pre-scaled); returns [B,T,N,Dv]. Logits/softmax accumulate in
  FP32 inside the kernel (the C256 einsum path keeps BF16 logits when float32_logits=False)."""
  # SEQ_MINOR: the Splash wrapper swaps q/k/v to [N,D,T] itself, which composes with the fused
  # read kernel's [B,N,K,T] outputs to a no-op instead of a relayout.
  q, k, v = (jnp.transpose(x, (0, 2, 1, 3)) for x in (query, key, value))
  t = q.shape[2]
  axis_names = out_names = nn.logical_to_mesh_axes((BATCH, HEAD, LENGTH, D_KV))
  segment_names = nn.logical_to_mesh_axes((BATCH, "activation_length_no_heads"))
  # bam_splash_ignore_segments: Pile inputs_segmentation is all ones without mix_attn, so the segment
  # mask is a no-op; the unsegmented kernel skips its per-block segment comparisons.
  ignore = bool(getattr(cfg, 'bam_splash_ignore_segments', False))
  segments = (None if decoder_segment_ids is None or ignore
              else splash_attention_kernel.SegmentIds(decoder_segment_ids, decoder_segment_ids))

  def body(q, k, v, segments=None):
    fused = cfg.sa_use_fused_bwd_kernel
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=min(cfg.sa_block_q, t), block_kv=min(cfg.sa_block_kv, t),
        block_kv_compute=min(cfg.sa_block_kv_compute, t),
        block_q_dkv=min(cfg.sa_block_q_dkv, t), block_kv_dkv=min(cfg.sa_block_kv_dkv, t),
        block_kv_dkv_compute=min(cfg.sa_block_kv_dkv_compute, t),
        block_q_dq=None if fused else min(cfg.sa_block_q_dq, t),
        block_kv_dq=None if fused else min(cfg.sa_block_kv_dq, t),
        use_fused_bwd_kernel=fused,
        q_layout=splash_attention_kernel.QKVLayout['SEQ_MINOR' if seq_minor else cfg.sa_q_layout],
        k_layout=splash_attention_kernel.QKVLayout['SEQ_MINOR' if seq_minor else cfg.sa_k_layout],
        v_layout=splash_attention_kernel.QKVLayout['SEQ_MINOR' if seq_minor else cfg.sa_v_layout])
    negative_diagonal = bool(getattr(cfg, 'bam_attention_negative_diagonal', False))
    mask = splash_attention_mask.CausalMask(shape=(t, t), offset=-1 if negative_diagonal else 0)
    if window is not None and window < t:     # C256: source > target - window
      mask &= splash_attention_mask.LocalMask(shape=(t, t), window_size=(window - 1, 0), offset=0)
    mask = splash_attention_mask.MultiHeadMask(masks=(mask,) * q.shape[1])
    # Optional name on Splash's own residuals (output and logsumexp) so a remat policy can keep
    # them instead of re-running the forward kernel inside the backward.
    residual_name = getattr(cfg, 'bam_splash_residual_name', None) or None
    kernel = splash_attention_kernel.make_splash_mha(
        mask=mask, head_shards=1, q_seq_shards=1, block_sizes=block_sizes,
        attn_logits_soft_cap=cfg.attn_logits_soft_cap,
        **({'residual_checkpoint_name': residual_name} if residual_name else {}))
    if negative_diagonal:
      from layers.bam_attention_diagonal_zero import make_negative_diagonal_splash
      kernel = make_negative_diagonal_splash(kernel)
    return jax.vmap(kernel)(q, k, v, segment_ids=segments)

  if segments is None:
    out = shard_map(body, mesh=mesh, in_specs=(axis_names,) * 3, out_specs=out_names,
                    check_rep=False)(q, k, v)
  else:
    out = shard_map(body, mesh=mesh, in_specs=(axis_names, axis_names, axis_names, segment_names),
                    out_specs=out_names, check_rep=False)(q, k, v, segments)
  return jnp.transpose(out, (0, 2, 1, 3))


def _update_bam_matrix(M_in, dM, lambda_decay):
  """Apply constant retention to the old state, then add the new write."""
  return jnp.asarray(lambda_decay, dtype=M_in.dtype) * M_in + dM


@dataclasses.dataclass(frozen=True)
class _BamReadArm:
  """One generalized column-read address per token and head."""
  name: str
  k_dim: int
  v_dim: int
  num_heads: int
  read_side: str = 'col'
  prune_row: bool = True
  key_mode: str = 'rms_gate'
  key_scale: float = 1.0
  rms_epsilon: float = 1e-6
  rms_statistics_dtype: Any = jnp.float32
  gate_activation: Any = jax.nn.sigmoid
  implementation: str = 'mul_reduce_btn'


def bam_read(M, key, arm, *, gate_logits=None, column_weight_multiplier=None):
  """Column read M @ address; normalize and gate each head's address."""
  with jax.named_scope('bam/read_key_transform'):
    direction = normalizations.rms_norm(
        key, dtype=key.dtype, epsilon=arm.rms_epsilon,
        statistics_dtype=arm.rms_statistics_dtype)
    scale = jnp.asarray(arm.key_scale, dtype=key.dtype)
    address = direction if gate_logits is None else scale * arm.gate_activation(gate_logits) * direction
  if column_weight_multiplier is not None:
    address = address * column_weight_multiplier.astype(address.dtype)
  with jax.named_scope('bam/read_m_contract'):
    if arm.implementation == 'dot_btn':
      with jax.named_scope('bam/contract_1a_col'):
        content = jnp.einsum('btkv,btnv->btnk', M, address)
    elif arm.implementation == 'mul_reduce_btn':
      content = jnp.sum(M[:, :, None] * address[..., None, :], axis=-1)
    else:
      raise ValueError(f'Unknown column read implementation: {arm.implementation}')
  return content, jnp.zeros(content.shape[:-1] + (M.shape[-1],), dtype=content.dtype)


def _pack_fetched_bam_heads(read, num_query_heads, head_dim):
  """Group adjacent fetched-read heads and pad the packed MHA-head tail."""
  fetched_heads = read.shape[-2]
  if fetched_heads < num_query_heads or fetched_heads % num_query_heads:
    raise ValueError(
        f'fetched BAM heads ({fetched_heads}) must be a positive multiple of '
        f'MHA heads ({num_query_heads})')
  heads_per_query = fetched_heads // num_query_heads
  packed_width = heads_per_query * read.shape[-1]
  if packed_width > head_dim:
    raise ValueError(
        f'{heads_per_query} fetched BAM heads need {packed_width} coordinates '
        f'per MHA head, but head_dim={head_dim}')
  read = read.reshape(
      read.shape[:-2] + (num_query_heads, packed_width))
  if packed_width < head_dim:
    read = jnp.pad(
        read, [(0, 0)] * (read.ndim - 1)
        + [(0, head_dim - packed_width)])
  return read


def bam_m_shift_valid_mask(shape, segment_ids=None, positions=None):
  valid = jnp.broadcast_to(jnp.arange(shape[1])[None, :] > 0, shape[:2])
  if segment_ids is not None:
    prev_segment = jnp.concatenate((segment_ids[:, :1], segment_ids[:, :-1]), axis=1)
    valid &= (segment_ids == prev_segment) & (segment_ids != 0)
  if positions is not None:
    prev_position = jnp.concatenate((positions[:, :1], positions[:, :-1]), axis=1)
    valid &= (positions == prev_position + 1)
  return valid


def mix_bam_previous_token(M, gate, segment_ids=None, positions=None):
  """Columnwise previous-token gate and causal shift of an immutable layer-input M [B,T,K,V].

  Each source is the previous token before any mixing. Invalid/padded/segment
  boundaries keep the current state. No sequence-prefix recurrence is used.
  """
  assert M.ndim == 4 and gate.shape == M.shape[:2] + (M.shape[-1],)
  previous = jnp.concatenate((M[:, :1], M[:, :-1]), axis=1)
  valid = bam_m_shift_valid_mask(M.shape, segment_ids, positions)
  g = gate.astype(M.dtype)[..., None, :]
  mixed = (jnp.asarray(1, M.dtype) - g) * M + g * previous
  return jnp.where(valid[..., None, None], mixed, M)


def bam_m_shift_health(M, mixed, gate, segment_ids=None, positions=None, *, column_metrics=True):
  """All valid-token gate stats; M health samples at most32 positions/sequence.

  mixed may be a callable (indices -> mixed M at those positions) so the full mix need not exist."""
  valid = bam_m_shift_valid_mask(M.shape, segment_ids, positions)
  weights = valid.astype(jnp.float32)
  count = jnp.maximum(jnp.sum(weights), 1)
  g = gate.astype(jnp.float32)
  column_mean = jnp.sum(g * weights[..., None], axis=(0, 1)) / count
  column_square = jnp.sum(g * g * weights[..., None], axis=(0, 1)) / count
  columns = (jnp.stack((column_mean, jnp.sqrt(jnp.maximum(column_square - column_mean**2, 0))))
             if column_metrics else None)
  mean = jnp.mean(column_mean)
  std = jnp.sqrt(jnp.maximum(jnp.mean(column_square) - mean**2, 0))
  fraction = lambda condition: jnp.sum(condition * weights[..., None]) / (count * g.shape[-1])
  lo = jnp.where(jnp.any(valid), jnp.min(jnp.where(valid[..., None], g, jnp.inf)), 0)
  hi = jnp.where(jnp.any(valid), jnp.max(jnp.where(valid[..., None], g, -jnp.inf)), 0)
  stats = jnp.stack((mean, std, lo, hi, fraction(g < .01), fraction(g > .99),
                     fraction(g > .5), jnp.mean(weights)))
  indices = jnp.linspace(1, max(M.shape[1]-1, 1), min(32, max(M.shape[1]-1, 1)), dtype=jnp.int32)
  indices = jnp.minimum(indices, M.shape[1]-1)
  current = M[:, indices].astype(jnp.float32)
  previous = M[:, jnp.maximum(indices-1, 0)].astype(jnp.float32)
  result = (mixed(indices) if callable(mixed) else mixed[:, indices]).astype(jnp.float32)
  sample_weights = weights[:, indices]
  sample_count = jnp.maximum(jnp.sum(sample_weights), 1)
  average = lambda v: jnp.sum(v * sample_weights) / sample_count
  squared = lambda v: jnp.mean(v*v, axis=(-2, -1))
  current_sq, previous_sq = squared(current), squared(previous)
  current_rms = jnp.sqrt(average(current_sq))
  previous_rms = jnp.sqrt(average(previous_sq))
  mixed_rms = jnp.sqrt(average(squared(result)))
  delta_rms = jnp.sqrt(average(squared(result-current)))
  cosine = jnp.mean(current*previous, axis=(-2, -1)) / jnp.maximum(jnp.sqrt(current_sq*previous_sq), 1e-12)
  state = jnp.stack((current_rms, previous_rms, mixed_rms,
                     mixed_rms / jnp.maximum(current_rms, 1e-12),
                     delta_rms / jnp.maximum(current_rms, 1e-12), average(cosine)))
  return stats, columns, state


class _RMSNormMultiplier(nn.Module):
  """RMSNorm's per-feature multiplier alone: the same 'scale' parameter as get_rmsnorm(name, cfg)."""
  config: Any
  features: int

  @nn.compact
  def __call__(self):
    cfg = self.config
    direct = getattr(cfg, 'direct_scale')
    init = nn.initializers.ones if direct else nn.initializers.zeros
    scale = self.param('scale', nn.with_logical_partitioning(init, ('norm',)), (self.features,), cfg.weight_dtype)
    scale = jnp.asarray(scale, cfg.dtype)
    return scale if direct else scale + 1.0


class _QKNormMultipliers(nn.Module):
  """Parameters of dc.QKNorm (qk_norm/{q_norm,k_norm}/scale) as multipliers, without normalizing anything."""
  config: Any
  features: int

  @nn.compact
  def __call__(self):
    return (_RMSNormMultiplier(self.config, self.features, name='q_norm')(),
            _RMSNormMultiplier(self.config, self.features, name='k_norm')())


class BamAttention(Attention):
  """V5: generalized column Q/K/V/O reads and independent attention/MLP M writes."""

  layer_mode: str = 'local_qk+local_v+local_o'
  read_side: str = 'col'
  layer_inx: int = 0  # Static layer index for per-layer read settings.
  bam_k: int = 32
  bam_v: int = 32

  def _attention_logit_scale_dim(self, qk_width):
    """Keep the Q/K temperature independent of the V/output head width."""
    dim = getattr(self.config, 'bam_qk_logit_scale_dim', None)
    if dim == 'qk_dim':
      return qk_width
    return self.head_dim if dim is None else dim


  def setup(self):
    super().setup()  # Standard Q/K projections, RoPE and output projection.
    cfg = self.config
    validate_bam_config(cfg, layer_mode=self.layer_mode)
    self._read_bam_v = self.bam_v
    self._standard_qk_width = cfg.bam_standard_qk_dim
    self._qk_col_width = cfg.bam_local_qk_col_output_dim or self.bam_k
    self._local_qk_output_width = self._qk_col_width
    self._partial_rope = cfg.bam_partial_rope
    self._partial_rope_nope_dim = cfg.bam_partial_rope_nope_dim or self._qk_col_width
    self._query_chunk_size = cfg.query_chunk_size if self.attention_kernel == 'dot_product_chunk' else None
    self._has_write = (cfg.bam_attention_write_layers is None
                       or self.layer_inx in cfg.bam_attention_write_layers)
    self._concat_health = bool(cfg.bam_record_concat_health)
    self._force_activation_dtype = bool(cfg.bam_force_activation_dtype)
    self._replicate_ploc_up = bool(cfg.bam_replicate_ploc_up)
    self._read_key_scale = float(cfg.bam_read_key_scale)
    self._rms_epsilon = float(cfg.normalization_layer_epsilon)
    self._read_key_epsilon = float(cfg.bam_read_key_epsilon or cfg.normalization_layer_epsilon)
    self._fetched_read_key_epsilon = float(cfg.bam_fetched_read_key_epsilon or self._read_key_epsilon)
    self._read_gate_init = cfg.bam_read_gate_init
    self._read_gate_activation = jax.nn.sigmoid
    self._read_rms_statistics_dtype = (jnp.float32 if cfg.bam_read_rms_statistics_dtype == 'float32' else self.dtype)
    self._write_rms_statistics_dtype = (jnp.float32 if cfg.bam_write_rms_statistics_dtype == 'float32' else self.dtype)
    self._read_implementation = cfg.bam_read_implementation
    self._m_read_norm = cfg.bam_m_read_norm
    self._full_m_read_rank = cfg.bam_local_full_m_read_bottleneck_dim
    self._write_v_bottleneck_dim = cfg.bam_write_v_bottleneck_dim
    self._write_outer_implementation = cfg.bam_write_outer_implementation
    self._write_data_rms = cfg.bam_write_data_rms
    from layers.bam_write_content import divisor
    self._attention_content_divisor = divisor(cfg, 'attention')
    self._mlp_content_divisor = divisor(cfg, 'mlp')
    self._m_shift_mode = cfg.bam_m_shift_mode or 'none'
    self._record_fetched_read_health_metrics = False
    zeros_init = initializers.contant_dense_init(0.0)
    reg_init = initializers.get_bam_init(cfg, fallback=self.kernel_init)

    # Preserve parameter names and creation order for existing checkpoints and seeds.
    if self._m_shift_mode != 'none':
      opening = float(cfg.bam_m_shift_init)
      self.m_shift_gate = DenseGeneral(
          features=self.bam_v, axis=-1, use_bias=False, kernel_init=zeros_init,
          kernel_axes=('embed', 'v_factor'), dtype=self.dtype, weight_dtype=self.weight_dtype,
          quant=self.quant, matmul_precision=cfg.matmul_precision, name='M_shift_gate')
      self.m_shift_bias = self.param(
          'M_shift_gate_b0', nn.with_logical_partitioning(
              nn.initializers.constant(math.log(opening / (1 - opening))), ('v_factor',)),
          (self.bam_v,), self.weight_dtype)

    settings = dict(prune_row=True, key_mode='rms_gate', key_scale=self._read_key_scale,
                    rms_epsilon=self._read_key_epsilon,
                    rms_statistics_dtype=self._read_rms_statistics_dtype,
                    gate_activation=self._read_gate_activation, implementation=self._read_implementation)
    self._fetched_arm = _BamReadArm(name='f', k_dim=self.bam_k, v_dim=self.bam_v,
        num_heads=self.num_query_heads, read_side='col',
        **dict(settings, rms_epsilon=self._fetched_read_key_epsilon))
    self._direct_qk_c8_arm = _BamReadArm(name='q', k_dim=self.bam_k, v_dim=self.bam_v,
        num_heads=self.num_query_heads, read_side='col', **settings)

    def add_read_gate(name, features, kernel_axes, bias_axes, opening):
      setattr(self, name, DenseGeneral(
          features=features, axis=-1, kernel_init=zeros_init, kernel_axes=kernel_axes,
          dtype=self.dtype, weight_dtype=self.weight_dtype, name=name,
          quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=False))
      setattr(self, name + '_b0', self.param(
          name + '_b0', nn.with_logical_partitioning(
              nn.initializers.constant(math.log(opening / (1.0 - opening))), bias_axes),
          features, self.weight_dtype))

    opening = (float(self._read_gate_init) if self._read_gate_init is not None
               else math.sqrt(self._read_key_epsilon) / self._read_key_scale)
    fetched_opening = cfg.bam_fetched_read_gate_init or opening
    self._add_full_m_key_projection('W_R', (self.num_query_heads, cfg.bam_n_f, self.bam_v),
                                  ('embed', 'q_heads', 'fetch', 'kv'), reg_init)
    add_read_gate('W_R_gate', (self.num_query_heads, cfg.bam_n_f, 1),
                  ('embed', 'q_heads', 'fetch', None), ('q_heads', 'fetch', None), fetched_opening)
    for arm in ('q', 'k'):
      setattr(self, 'static_' + arm + '_key', self.param(
          'static_' + arm + '_key', nn.with_logical_partitioning(zeros_init, ('v_factor', 'q_heads')),
          (self.bam_v, self.num_query_heads), self.weight_dtype))
    for arm in ('v', 'o'):
      setattr(self, 'static_' + arm + '_key', self.param(
          'static_' + arm + '_key', nn.with_logical_partitioning(
              nn.initializers.normal(self.bam_v ** -0.5) if arm == 'v' else zeros_init,
              ('v_factor', 'q_heads')), (self.bam_v, self.num_query_heads), self.weight_dtype))

    if self._has_write:
      self.P_loc_down = DenseGeneral(
          features=self._write_v_bottleneck_dim, axis=-1, kernel_init=reg_init,
          kernel_axes=('embed', None), dtype=self.dtype, weight_dtype=self.weight_dtype,
          name='P_loc_down', quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=False)
      self.P_loc_up = DenseGeneral(
          features=(self.num_query_heads, self.bam_v), axis=-1, kernel_init=reg_init,
          kernel_axes=(None if self._replicate_ploc_up else 'embed', 'q_heads', 'v_factor'),
          dtype=self.dtype, weight_dtype=self.weight_dtype, name='P_loc_up', quant=self.quant,
          matmul_precision=cfg.matmul_precision, use_bias=True)
      self.W_gw = DenseGeneral(features=(self.num_query_heads,), axis=-1, kernel_init=reg_init,
          kernel_axes=('embed', 'q_heads'), dtype=self.dtype, weight_dtype=self.weight_dtype,
          name='W_gw', quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=False)
      eps = float(cfg.bam_write_eps)
      self.gw_b0 = self.param('gw_b0', nn.with_logical_partitioning(
          nn.initializers.constant(math.log(eps / (1.0 - eps))), ('q_heads',)),
          (self.num_query_heads,), self.weight_dtype)
      self.write_data_norm = GroupedRMSNorm(
          scale_shape=(self.num_query_heads, self.bam_k),
          epsilon=initializers.get_bam_write_epsilon(cfg), dtype=self.dtype,
          statistics_dtype=self._write_rms_statistics_dtype, weight_dtype=self.weight_dtype,
          kernel_axes=('q_heads', 'kv'), scale_init=None, name='write_data_norm')
      self.write_address_norm = GroupedRMSNorm(
          scale_shape=(self.num_query_heads, self.bam_v),
          epsilon=initializers.get_bam_write_epsilon(cfg), dtype=self.dtype,
          statistics_dtype=self._write_rms_statistics_dtype, weight_dtype=self.weight_dtype,
          kernel_axes=('q_heads', 'kv'), scale_init=None, use_bias=False, name='write_address_norm')

    add_read_gate('W_lv_gate', (self.num_query_heads, 1),
                  ('embed', 'q_heads', None), ('q_heads', None), fetched_opening)
    for arm in ('q', 'k'):
      self._add_full_m_key_projection('W_l' + arm + '_c8', (self.num_query_heads, self.bam_v),
                                    ('embed', 'q_heads', 'kv'), reg_init)
      add_read_gate('W_l' + arm + '_gate', (self.num_query_heads, 1),
                    ('embed', 'q_heads', None), ('q_heads', None), opening)
    for arm, shape, axes in (
        ('q', (self.num_query_heads, self.bam_v), ('q_heads', 'kv')),
        ('k', (self.num_query_heads, self.bam_v), ('q_heads', 'kv')),
        ('vo', (self.num_query_heads, cfg.bam_n_f, self.bam_v), ('q_heads', 'fetch', 'kv'))):
      setattr(self, 'general_' + arm + '_pre_bias', self._constant_general_parameter(
          'general_' + arm + '_pre_bias', shape, axes))
    for arm in ('q', 'k', 'v', 'o'):
      if arm in ('q', 'k') and not cfg.bam_qk_static_read_gate:
        continue
      self._add_static_gate('general_' + arm + '_static_gate', cfg.bam_static_read_gate_init,
                            self.num_query_heads)

    # V and O start from identical directions but have independent up projections.
    def clone_vo_init(_key, shape, dtype, *unused, **unused_kwargs):
      source = nn.meta.unbox(self.variables['params']['W_R_up']['kernel'])
      return jnp.reshape(source[:, -self.num_query_heads:, 0, :], shape).astype(dtype)
    self._add_full_m_key_projection('W_lv_c8', (self.num_query_heads, self.bam_v),
                                  ('embed', 'q_heads', 'kv'), clone_vo_init)
    self.general_v_pre_bias = self._constant_general_parameter(
        'general_v_pre_bias', (self.num_query_heads, self.bam_v), ('q_heads', 'kv'))


  def _constant_general_parameter(self, name, shape, axes, value=0.):
    # Constants need no random key; keep the parent's init RNG counters intact.
    return nn.unbox(self.variable('params', name, nn.with_logical_partitioning(
        lambda: jnp.full(shape, value, self.weight_dtype), axes)).value)


  def _add_static_gate(self, name, opening, num_heads=None):
    cfg = self.config
    assert 0 < opening < 1.1
    num_heads = self.num_query_heads if num_heads is None else num_heads
    setattr(self, name, DenseGeneral(
        features=(num_heads,), axis=-1, use_bias=False,
        kernel_init=initializers.contant_dense_init(0.0),
        kernel_axes=('embed', 'q_heads'), dtype=self.dtype, weight_dtype=self.weight_dtype,
        quant=self.quant, matmul_precision=cfg.matmul_precision, name=name))
    setattr(self, name + '_bias', self._constant_general_parameter(
        name + '_bias', (num_heads,), ('q_heads',),
        math.log(opening / (1.1 - opening))))

  def _static_gate(self, name, x):
    return 1.1 * jax.nn.sigmoid(
        getattr(self, name)(x) + jnp.asarray(getattr(self, name + '_bias'), self.dtype))

  def _record_general_gate(self, name, gate):
    if self._concat_health:
      g = gate.astype(jnp.float32)
      self.sow('intermediates', 'concat_general_' + name + '_gate', jnp.stack((
          jnp.mean(g), jnp.std(g), jnp.mean(g < .05),
          jnp.mean(g > .5), jnp.mean(g > .95))))

  def _record_general_key_norm(self, name, key):
    if self._concat_health:
      rms = jnp.sqrt(jnp.mean(key.astype(jnp.float32)**2, axis=-1))
      self.sow('intermediates', 'concat_general_' + name + '_key_norm', jnp.stack((
          jnp.mean(rms), jnp.min(rms), jnp.mean(rms < .01))))

  def _record_general_alignment(self, name, a, b):
    if self._concat_health:
      a, b = a.astype(jnp.float32), b.astype(jnp.float32)
      cosine = jnp.sum(a*b, axis=-1) / jnp.maximum(
          jnp.sqrt(jnp.sum(a*a, axis=-1)*jnp.sum(b*b, axis=-1)), 1e-12)
      self.sow('intermediates', 'concat_general_' + name + '_alignment', jnp.stack((
          jnp.mean(cosine), jnp.mean(jnp.abs(cosine)), jnp.mean(cosine**2))))

  def _add_full_m_key_projection(self, name, features, axes, initializer):
    """Independent GELU bottleneck per Q/K/VO arm; VO shares one up output."""
    cfg = self.config
    for suffix, output, kernel_axes in (
        ('down', self._full_m_read_rank, ('embed', None)),
        ('up', features, ('embed',) + axes[1:])):
      parameter_name = ('full_m_read_down' if suffix == 'down'
                        else f'{name}_{suffix}')
      if suffix == 'down' and hasattr(self, parameter_name):
        continue
      setattr(self, parameter_name, DenseGeneral(
          features=output, axis=-1, kernel_init=initializer,
          kernel_axes=kernel_axes, dtype=self.dtype, weight_dtype=self.weight_dtype,
          name=parameter_name, quant=self.quant,
          matmul_precision=cfg.matmul_precision, use_bias=False))

  def _project_column_read_key(self, name, x, shared_hidden=None):
    hidden = nn.gelu(self.full_m_read_down(x)) if shared_hidden is None else shared_hidden
    key = getattr(self, f'{name}_up')(hidden)
    arm = {'W_lq_c8': 'q', 'W_lk_c8': 'k', 'W_R': 'vo', 'W_lv_c8': 'v'}[name]
    bias = jnp.asarray(getattr(self, 'general_' + arm + '_pre_bias'), key.dtype)
    self._record_concat_amplitude('general_' + arm + '_pre_bias', bias, key)
    self._record_general_key_norm(arm + '_before_bias', key)
    key = key + bias
    self._record_general_key_norm(arm + '_after_bias', key)
    if self._concat_health:
      normalized = key * jax.lax.rsqrt(jnp.mean(
          key.astype(self._read_rms_statistics_dtype)**2, axis=-1, keepdims=True)
          + self._read_key_epsilon).astype(key.dtype)
      targets = ((('o',))) if arm == 'vo' else (arm,)
      for target in targets:
        post = getattr(self, 'static_' + target + '_key').T.astype(key.dtype)
        if arm == 'vo':
          post = post[:, None, :]
        self._record_concat_amplitude('general_' + target + '_post_key', post, normalized)
        self._record_general_alignment(target + '_key', post, normalized)
    return key


  def _record_concat_gate(self, name, logits):
    if not self._concat_health:
      return
    gate = self._read_gate_activation(logits.astype(jnp.float32))
    self.sow('intermediates', 'concat_' + name + '_gate', jnp.stack((
        jnp.mean(gate), jnp.std(gate), jnp.mean(gate < .05),
        jnp.mean(gate > .5), jnp.mean(gate > .95))))

  def _record_concat_amplitude(self, name, bam, standard):
    if not self._concat_health:
      return
    rms = lambda x: jnp.sqrt(jnp.mean(jnp.square(x.astype(jnp.float32))))
    b, s = rms(bam), rms(standard)
    self.sow('intermediates', 'concat_' + name + '_amplitude',
             jnp.stack((b, s, b / jnp.maximum(s, 1e-12))))

  def _record_concat_qk_scores(self, query, key, q_bam, k_bam, segment_ids=None):
    if not self._concat_health or not getattr(self.config, 'bam_concat_qk', False):
      return
    # One sequence, 16 evenly-spaced tokens, all heads; no full quadratic probe.
    ix = jnp.linspace(0, query.shape[1] - 1, min(16, query.shape[1])).astype(jnp.int32)
    mask = (ix[:, None] >= ix[None, :])[None]
    if segment_ids is not None:
      segments = segment_ids[0, ix]
      mask &= (segments[:, None] == segments[None, :])[None]
    counts = jnp.sum(mask, axis=-1, keepdims=True)
    scale_dim = self._attention_logit_scale_dim(query.shape[-1] + q_bam.shape[-1])
    def score_rms(q, k):
      q, k = q[0, ix].astype(jnp.float32), k[0, ix].astype(jnp.float32)
      score = jnp.einsum('thd,shd->hts', q, k) / math.sqrt(scale_dim)
      mean = jnp.sum(jnp.where(mask, score, 0.), axis=-1, keepdims=True) / counts
      return jnp.sqrt(jnp.sum(jnp.where(mask, (score-mean)**2, 0.)) /
                      (jnp.sum(mask) * score.shape[0]))
    original_width = self.head_dim - (self._standard_qk_width or 0)
    if self._standard_qk_width is not None and q_bam.shape[-1] > original_width:
      tail = score_rms(q_bam[..., original_width:], k_bam[..., original_width:])
      prefix = score_rms(q_bam[..., :original_width], k_bam[..., :original_width])
      self.sow('intermediates', 'concat_qk_extra_scores',
               jnp.stack((tail, prefix, tail / jnp.maximum(prefix, 1e-12))))
    std, bam = score_rms(query, key), score_rms(q_bam, k_bam)
    self.sow('intermediates', 'concat_qk_scores',
             jnp.stack((bam, std, bam / jnp.maximum(std, 1e-12))))


  def _project_read_gate_logits(
      self, gate_name, x, squeeze_fetch_axis=False):
    with jax.named_scope("bam/read_gate_projection"):
      gate_bias = getattr(self, f'{gate_name}_b0')
      if self._force_activation_dtype:
        gate_bias = jnp.asarray(gate_bias, self.dtype)
      candidate_logits = getattr(self, gate_name)(x) + gate_bias
    if squeeze_fetch_axis:
      candidate_logits = jnp.squeeze(candidate_logits, axis=-2)
    return candidate_logits

  # ---- Local reads (q / k / v): one packed projection, one read primitive. ----


  def _static_column(self, M, arm, x=None, *, column_weight_multiplier=None):
    key = getattr(self, 'static_' + arm + '_key').astype(M.dtype)
    if column_weight_multiplier is None:
      read = jnp.einsum('btkv,vn->btnk', M, key)
    else:
      assert arm == 'o'
      read = jnp.einsum('btkv,btnv->btnk', M, key.T * column_weight_multiplier.astype(M.dtype))
    if (arm not in ('q', 'k') or self.config.bam_qk_static_read_gate):
      assert x is not None
      gate = self._static_gate('general_' + arm + '_static_gate', x)
      self._record_general_gate(arm + '_static', gate)
      # Preserve the trained SOTA's 1.1*sigmoid gate and initial-amplitude compensation.
      read = read * (gate / self.config.bam_static_read_gate_init)[..., None]
    return read

  def _read_direct_qk_c8(self, name, M, compressed_M, x, shared_hidden=None):
    """Independent per-head compressed read plus the original full-M static read."""
    key = self._project_column_read_key(f'W_l{name}_c8', x, shared_hidden)
    logits = self._project_read_gate_logits(f'W_l{name}_gate', x)
    self._record_concat_gate('local_' + name, logits)
    col, _ = bam_read(compressed_M, key, self._direct_qk_c8_arm, gate_logits=logits)
    col = col[..., :self._qk_col_width]
    static = self._static_column(M[..., :self._qk_col_width, :], name, x)
    self._record_concat_amplitude('static_' + name, static, col)
    col = col + static
    return _pack_fetched_bam_heads(col, self.num_query_heads, self.head_dim)


  def _independent_local_vo(self, M, x, compressed_M=None, shared_hidden=None,
                            *, o_column_weight_multiplier=None, return_o_logits=False):
    """Contract one ungated C8 read, then apply destination-specific gates."""
    if compressed_M is None:
      compressed_M = M
    read, o_logits = self._read_fetched_m(
        compressed_M, x, ungated=True, shared_hidden=shared_hidden,
        column_weight_multiplier=o_column_weight_multiplier)
    v_logits = self._project_read_gate_logits('W_lv_gate', x)
    self._record_concat_gate('local_v', v_logits)
    v_key = self._project_column_read_key('W_lv_c8', x, shared_hidden)
    v_read = bam_read(compressed_M, v_key, self._fetched_arm)
    if self._concat_health:
      v, o = (self._read_gate_activation(z.astype(jnp.float32)) for z in (v_logits, o_logits))
      vc, oc = v - jnp.mean(v), o - jnp.mean(o)
      correlation = jnp.mean(vc * oc) / jnp.maximum(jnp.sqrt(jnp.mean(vc * vc) * jnp.mean(oc * oc)), 1e-12)
      self.sow('intermediates', 'concat_vo_gate_pair', jnp.stack((
          jnp.mean(jnp.abs(v-o)), jnp.sqrt(jnp.mean((v-o)**2)), correlation)))
    outputs = (self._gate_local_output(v_read, v_logits), self._gate_local_output(read, o_logits))
    return outputs + (o_logits,) if return_o_logits else outputs


  def _matrix_for_read(self, M_in):
    """Select the configured read-side view without changing the raw matrix stream."""
    if M_in is None or self._m_read_norm == 'none':
      return M_in
    # M-shift is applied before this read-only view. Keep statistics in FP32
    # and preserve the raw matrix stream for attention/private-MLP writes.
    matrix = M_in.astype(jnp.float32)
    normalized = matrix * jax.lax.rsqrt(
        jnp.mean(jnp.square(matrix), axis=(-2, -1), keepdims=True) + self._rms_epsilon)
    return normalized.astype(M_in.dtype)

  def _add_local_qk(self, query, key, q_local, k_local, segment_ids=None):
    """Inject LocalQK into the standard Q/K arm."""
    self._record_concat_amplitude('local_q', q_local[..., :self._qk_col_width], query)
    self._record_concat_amplitude('local_k', k_local[..., :self._qk_col_width], key)
    self._record_concat_qk_scores(query, key, q_local[..., :self._qk_col_width], k_local[..., :self._qk_col_width], segment_ids)
    return (jnp.concatenate((q_local[..., :self._qk_col_width], query), axis=-1),
            jnp.concatenate((k_local[..., :self._qk_col_width], key), axis=-1))


  def _apply_partial_rope(self, x, positions, name):
    """Keep the LocalQK footprint position-free; rotate its unused head tail."""
    assert self._partial_rope_nope_dim is not None
    nope = x[..., :self._partial_rope_nope_dim]
    rope = x[..., self._partial_rope_nope_dim:]
    rope = self.apply_rotary_embedding(
        rope, positions, name=name, embedding_dims=rope.shape[-1])
    return jnp.concatenate((nope, rope), axis=-1)

  def _write_address(self, x, shared_hidden=None):
    """Reuse the reader's activated down projection when read/write sharing is enabled."""
    hidden = self.P_loc_down(x)
    hidden = nn.gelu(hidden)
    return self.P_loc_up(hidden)

  def _write(self, o_head, x, M_in, shared_hidden=None):
    """Write primitive (§4.2 safe write: aggregated U (outer) local V). o_head: [b,t,n,d] head output (pre W_O).

    Per-record factor normalization (§4.6.5 write-side per-record factor norm): the address
    factor is RMS-normalized over its head-dim axis; the data factor is normally normalized too,
    but may retain its raw magnitude for ablation. rms(u) = u·rsqrt(mean(u²,-1)+eps).
    """
    cfg = self.config
    if self._force_activation_dtype:
      assert M_in.dtype == self.dtype, (M_in.dtype, self.dtype)
      assert o_head.dtype == self.dtype, (o_head.dtype, self.dtype)
    # V1 default
    u1 = o_head[..., :self.bam_k]  # U factor [b,t,n,k]
    u2 = self._write_address(x, shared_hidden)
    write_gate_bias = self.gw_b0
    if self._force_activation_dtype:
      write_gate_bias = jnp.asarray(write_gate_bias, self.dtype)
    gate = jax.nn.sigmoid(self.W_gw(x) + write_gate_bias)  # [b,t,n]
    g = gate
    u1_norm = self._attention_write_content(u1)
    u2_norm = self.write_address_norm(u2)
    gated_u1 = g[..., None] * u1_norm
    with jax.named_scope("bam/write_outer"):
      if self._write_outer_implementation == 'dot':
        dM = jnp.einsum('btnk,btnv->btkv', gated_u1, u2_norm)
      else:
        assert self._write_outer_implementation == 'mul_reduce'  # XD
        dM = jnp.sum(
            gated_u1[..., None] * u2_norm[..., None, :], axis=-3)
    if self._force_activation_dtype:
      assert dM.dtype == self.dtype, (dM.dtype, self.dtype)
    M_out = _update_bam_matrix(M_in, dM, cfg.bam_lambda_decay)
    if self._force_activation_dtype:
      assert M_out.dtype == self.dtype, (M_out.dtype, self.dtype)
    return M_out, gate

  def _attention_write_content(self, content):
    if self._attention_content_divisor is not None:
      from layers.bam_write_content import fixed_content
      content = fixed_content(content, self._attention_content_divisor)
    elif self._write_data_rms:
      content = self.write_data_norm(content)
    return content

  def _deferred_write_factors(self, o_head, x, shared_hidden=None):
    """Same factors as _write, without forming its outer product yet."""
    u1 = o_head[..., :self.bam_k]
    u2 = self._write_address(x, shared_hidden)
    bias = jnp.asarray(self.gw_b0, self.dtype) if self._force_activation_dtype else self.gw_b0
    gate = jax.nn.sigmoid(self.W_gw(x) + bias)
    scale = 1.0
    content = self._attention_write_content(u1)
    return scale * gate[..., None] * content, self.write_address_norm(u2), gate

  def merge_mlp_write(self, mlp_head, mlp_gate, factors, M_in, independent_address=None):
    """Merge attention/MLP writes, then decay the carried matrix exactly once."""
    from layers.bam_write_content import fixed_content
    content = (fixed_content(mlp_head, self._mlp_content_divisor)
               if self._mlp_content_divisor is not None else
               self.write_data_norm(mlp_head)
               if self._write_data_rms else mlp_head)
    scale = 1.0
    mlp_content = scale * mlp_gate[..., None] * content
    if factors is None:
      if independent_address is None:
        raise ValueError('MLP-only M write requires an independent address')
      mlp_address = self.write_address_norm(independent_address)
      with jax.named_scope('bam/mlp_independent_write_outer'):
        if self._write_outer_implementation == 'dot':
          dM = jnp.einsum('btnk,btnv->btkv', mlp_content, mlp_address)
        else:
          dM = jnp.sum(mlp_content[..., None] * mlp_address[..., None, :], axis=-3)
      return _update_bam_matrix(M_in, dM, self.config.bam_lambda_decay)
    attention_content, address, attention_gate = factors
    if independent_address is None:
      combined_content = attention_content + mlp_content
      with jax.named_scope('bam/write_outer'):
        if self._write_outer_implementation == 'dot':
          dM = jnp.einsum('btnk,btnv->btkv', combined_content, address)
        else:
          dM = jnp.sum(combined_content[..., None] * address[..., None, :], axis=-3)
    else:
      mlp_address = self.write_address_norm(independent_address)
      with jax.named_scope('bam/write_outer'):
        if self._write_outer_implementation == 'dot':
          dM = jnp.einsum('btnk,btnv->btkv', attention_content, address)
        else:
          dM = jnp.sum(attention_content[..., None] * address[..., None, :], axis=-3)
      with jax.named_scope('bam/mlp_independent_write_outer'):
        if self._write_outer_implementation == 'dot':
          dM = dM + jnp.einsum('btnk,btnv->btkv', mlp_content, mlp_address)
        else:
          dM = dM + jnp.sum(
              mlp_content[..., None] * mlp_address[..., None, :], axis=-3)
      if self._concat_health and independent_address is not None:
        a, b = address.astype(jnp.float32), mlp_address.astype(jnp.float32)
        cosine = jnp.sum(a*b, axis=-1) / jnp.maximum(
            jnp.sqrt(jnp.sum(a*a, axis=-1)*jnp.sum(b*b, axis=-1)), 1e-12)
        self.sow('intermediates', 'concat_mlp_address_alignment', jnp.stack((
            jnp.mean(cosine), jnp.mean(jnp.abs(cosine)), jnp.mean(cosine**2))))
    if self._concat_health:
      for name, gate in [('attention_write', attention_gate), ('mlp_write', mlp_gate)]:
        g = gate.astype(jnp.float32)
        self.sow('intermediates', 'concat_' + name + '_gate', jnp.stack((
            jnp.mean(g), jnp.std(g), jnp.mean(g < .05),
            jnp.mean(g > .5), jnp.mean(g > .95))))
      # Factor health never reconstructs separate dynamic outer products.
      self._record_concat_amplitude('mlp_write_content', mlp_content, attention_content)
      self._record_concat_amplitude('mlp_write_raw_output', mlp_head, content)
      self._record_concat_amplitude('combined_write', dM, M_in)
    return _update_bam_matrix(M_in, dM, self.config.bam_lambda_decay)


  def _read_fetched_m(self, matrix, inputs, *, ungated=True, shared_hidden=None,
                      column_weight_multiplier=None):
    """Local O's dynamic column read; V has its independent address."""
    with jax.named_scope('bam/read_fetched_m'):
      logits = self._project_read_gate_logits('W_R_gate', inputs, squeeze_fetch_axis=True)
      self._record_concat_gate('local_o', logits)
      with jax.named_scope('bam/read_key_projection'):
        key = jnp.squeeze(self._project_column_read_key('W_R', inputs, shared_hidden), axis=-2)
      read = bam_read(matrix, key, self._fetched_arm,
                      column_weight_multiplier=column_weight_multiplier)
      return read, logits

  def _gate_local_output(self, read, logits):
    # V/O gates follow contraction on the JAX path; keep this BF16 ordering.
    gates = self._read_key_scale * self._read_gate_activation(logits)
    return _pack_fetched_bam_heads(read[0] * gates, self.num_query_heads, self.head_dim)

  def _attention_block(
      self, query, key, value, decoder_segment_ids, *, q0, s0, window_size,
      ):
    """Apply one dense/chunk attention block and its optional BAM fetch."""
    q1 = q0 + query.shape[1]
    s1 = s0 + key.shape[1]
    target = jnp.arange(q0, q1)[:, None]
    source = jnp.arange(s0, s1)[None, :]
    valid = (source <= target)[None]
    if window_size < s1:
      valid &= (source > target - window_size)
    if decoder_segment_ids is not None:
      valid &= (
          decoder_segment_ids[:, q0:q1, None]
          == decoder_segment_ids[:, None, s0:s1])

    cfg = self.config
    y_std, alpha = _attention_op(
        query, key, value, valid,
        attn_logits_soft_cap=cfg.attn_logits_soft_cap,
        float32_logits=cfg.float32_logits,
        negative_diagonal_mask=(source == target) if getattr(cfg, 'bam_attention_negative_diagonal', False) else None)
    return y_std, None

  def _query_chunk_op(self, query, key, value, decoder_segment_ids, window_size):
    """Causal query chunks using the same local attention computation."""
    t = query.shape[1]
    chunk_size = int(self._query_chunk_size)
    assert t % chunk_size == 0
    chunks = []
    for q0 in range(0, t, chunk_size):
      q1 = q0 + chunk_size
      s0 = max(0, q0 - window_size) if window_size < t else 0
      chunk, _ = self._attention_block(
          query[:, q0:q1], key[:, s0:q1], value[:, s0:q1],
          decoder_segment_ids, q0=q0, s0=s0, window_size=window_size)
      chunks.append(chunk)
    return jnp.concatenate(chunks, axis=1), None

  def _check_pallas_core(self):
    """Execution restrictions beyond the shared V5 circuit validation."""
    if self._concat_health:
      raise ValueError('Detailed read health requires the JAX reference backend')
    if float(self.config.bam_lambda_decay) != 1.0:
      raise ValueError('The Pallas core requires unit M retention')

  def _fused_input_projections(self, x, names):
    """All named DenseGeneral projections of x as ONE dot over concatenated kernels.

    Same parameters, casts, precision, kernel-gradient scales and biases as calling each module;
    the reverse then reads x once and writes one dx instead of one partial dx per projection.
    Returns {name: [B,T,*features]}."""
    params = self.variables['params']
    kernels, shapes, biases = [], [], []
    for name in names:
      mod = getattr(self, name, None)
      kernel = jnp.asarray(nn.meta.unbox(params[name]['kernel']), self.dtype)
      scale = getattr(mod, 'kernel_gradient_scale', 1.0) if mod is not None else 1.0
      if scale != 1.0:
        kernel_sg = jax.lax.stop_gradient(kernel)
        kernel = kernel_sg + scale * (kernel - kernel_sg)
      shapes.append(kernel.shape[1:])
      kernels.append(kernel.reshape(kernel.shape[0], -1))
      bias = params[name].get('bias') if hasattr(params[name], 'get') else None
      biases.append(None if bias is None else jnp.asarray(nn.meta.unbox(bias), self.dtype))
    weight = jnp.concatenate(kernels, axis=1)
    y = jax.lax.dot_general(jnp.asarray(x, self.dtype), weight, (((x.ndim - 1,), (0,)), ((), ())),
                            precision=jax.lax.Precision(self.config.matmul_precision))
    # One split (its transpose is one concatenate) instead of per-piece slices, whose reverses
    # were zero-padded full-width adds (pad_add_fusion, ~28 ms/step on v5p-8).
    widths = [kernel.shape[1] for kernel in kernels]
    cuts = [sum(widths[:i + 1]) for i in range(len(widths) - 1)]
    parts = jnp.split(y, cuts, axis=-1)
    out = {}
    for name, shape, part, bias in zip(names, shapes, parts, biases):
      part = part.reshape(x.shape[:-1] + shape)
      out[name] = part if bias is None else part + bias
    return out

  def _pallas_core_call(self, inputs_q, inputs_positions, decoder_segment_ids, M_in, is_global,
                        standard_qk_width, deterministic=False, common_vo_scale=None, common_qk_scale=None):
    """Fused-kernel forward; M_in is token-minor [B,V,K,T] (token-major [B,T,V,K] for v8t).

    Returns (out, M_in, write factors)."""
    from layers import bam_kernels as bam_pallas
    cfg = self.config
    self._check_pallas_core()
    if cfg.bam_pallas_body != 'v8t':
      raise ValueError('Independent VO keys require the v8t fused read')
    b_, t_ = inputs_q.shape[:2]
    fused = bool(getattr(cfg, 'bam_pallas_fused_inputs', False)) and not self.is_initializing()
    static_gate_names = tuple(f'general_{a}_static_gate' for a in 'qkvo'
                              if a not in 'qk' or cfg.bam_qk_static_read_gate)
    if fused:
      gate_names = ('W_lq_gate', 'W_lk_gate', 'W_R_gate', 'W_lv_gate')
      key_names = ('full_m_read_down',)
      shift_names = ('M_shift_gate',) if self._m_shift_mode != 'none' else ()
      write_names = ('P_loc_down', 'W_gw')
      if not self._has_write:
        write_names = ()
      raw = self._fused_input_projections(
          inputs_q, ('query', 'key')
          + key_names + gate_names + static_gate_names + shift_names
          + write_names)
      query = raw['query'].reshape(b_, t_, self.num_query_heads, standard_qk_width)
      key = raw['key'].reshape(b_, t_, self.num_kv_heads, standard_qk_width)
    else:
      query = self.query_projection(inputs_q, standard_qk_width)
      key = self.kv_projection(inputs_q, proj_name='key', projection_dim=standard_qk_width)
    full_qk_norm = cfg.qk_norm and cfg.bam_qk_norm_after_concat
    full_qk_width = self._qk_col_width + standard_qk_width
    if not full_qk_norm:
      query, key = dc.QKNorm(cfg, name='qk_norm')(query, key)
      query = self.apply_rotary_embedding(query, inputs_positions, name='query_rotary',
                                          embedding_dims=standard_qk_width)
      key = self.apply_rotary_embedding(key, inputs_positions, name='key_rotary',
                                        embedding_dims=standard_qk_width)
    else:
      # Full-head QKNorm then RoPE on the standard tail, rearranged exactly: the RMS statistic is one
      # scalar per token/head and RoPE is linear, so RoPE(s*x/r) = RoPE(s_tail*x_tail)/r. Scale and rotate
      # the narrow standard arm here; after the read only 1/r and the local scales remain (no post-read
      # RoPE on a 10-wide minor dimension, no relayout). r uses the unscaled, unrotated head.
      head_scale = _QKNormMultipliers(cfg, full_qk_width, name='qk_norm')()
      tail = slice(self._qk_col_width, None)
      standard_sumsq = tuple(jnp.sum(jnp.square(x.astype(jnp.float32)), axis=-1) for x in (query, key))
      query = self.apply_rotary_embedding(query * head_scale[0][tail], inputs_positions, name='query_rotary',
                                          embedding_dims=standard_qk_width)
      key = self.apply_rotary_embedding(key * head_scale[1][tail], inputs_positions, name='key_rotary',
                                        embedding_dims=standard_qk_width)

    def logits(name, squeeze):
      if fused:     # same bias handling as _project_read_gate_logits
        gate_bias = getattr(self, f'{name}_b0')
        if self._force_activation_dtype:
          gate_bias = jnp.asarray(gate_bias, self.dtype)
        gate = raw[name] + gate_bias
        gate = jnp.squeeze(gate, axis=-2) if squeeze else gate
      else:
        gate = self._project_read_gate_logits(name, inputs_q, squeeze_fetch_axis=squeeze)
      gate = gate
      return jnp.reshape(gate, inputs_q.shape[:2] + (self.num_query_heads,))

    packed_qkvo = None
    hidden = nn.gelu(raw['full_m_read_down'] if fused else self.full_m_read_down(inputs_q))
    if fused:
      up_names = ('W_lq_c8_up', 'W_lk_c8_up', 'W_R_up')
      up_names += ('W_lv_c8_up',)
      up = self._fused_input_projections(hidden, up_names)
      pre = lambda arm: ((jnp.asarray(getattr(self, 'general_' + arm + '_pre_bias'), hidden.dtype)))
      q_key = up['W_lq_c8_up'] + pre('q')
      k_key = up['W_lk_c8_up'] + pre('k')
      vo_key = up['W_R_up'] + pre('vo')
    else:
      q_key = self._project_column_read_key('W_lq_c8', inputs_q, hidden)
      k_key = self._project_column_read_key('W_lk_c8', inputs_q, hidden)
      vo_key = self._project_column_read_key('W_R', inputs_q, hidden)
    vo_key = jnp.squeeze(vo_key, axis=-2)
    v_key = (up['W_lv_c8_up'] + pre('v') if fused else
             self._project_column_read_key('W_lv_c8', inputs_q, hidden))
    # v8t accepts two key slabs, V then O, instead of a shared VO slab.
    vo_key = jnp.concatenate((v_key,
                              vo_key), axis=-2)
    if fused and getattr(cfg, 'bam_pallas_packed_read_keys', False):
      # The up projections as one einsum straight into v8t's key layout: [B,C,4N,T] (q, k, v, o) with
      # independent V/O keys, [B,C,3N,T] (q, k, vo) with a shared VO key. No split, concat or transpose
      # copies in either direction. Same kernels, casts, precision and kernel-gradient scales as
      # _fused_input_projections.
      params = self.variables['params']
      kernels = []
      arms = (('q', 'k', 'v', 'vo'))
      names = {'q': 'W_lq_c8_up', 'k': 'W_lk_c8_up', 'v': 'W_lv_c8_up', 'vo': 'W_R_up'}
      for name in (names[arm] for arm in arms):
        kernel = jnp.asarray(nn.meta.unbox(params[name]['kernel']), self.dtype)
        scale = getattr(getattr(self, name, None), 'kernel_gradient_scale', 1.0)
        if scale != 1.0:
          kernel_sg = jax.lax.stop_gradient(kernel)
          kernel = kernel_sg + scale * (kernel - kernel_sg)
        assert params[name].get('bias') is None if hasattr(params[name], 'get') else True, name
        kernels.append(kernel.reshape(kernel.shape[0], -1, kernel.shape[-1]))      # [D', N, V]
      def arm_bias(arm):   # [N, V] (the O pre-bias carries a singleton fetch axis; 0 without general reads)
        b_ = jnp.asarray(pre(arm), hidden.dtype)
        return b_.reshape(-1, self.bam_v) if b_.ndim else jnp.broadcast_to(b_, (self.num_query_heads, self.bam_v))
      bias = jnp.concatenate([arm_bias(arm) for arm in arms], axis=0)               # [S*N, V]
      # The pre-biases as an extra contraction row against a constant-one feature: added and
      # differentiated inside the matmul, with no separate [B,C,S*N,T] add or token reduction.
      ones = jnp.ones(hidden.shape[:-1] + (1,), hidden.dtype)
      weight = jnp.concatenate((jnp.concatenate(kernels, axis=1), bias[None].astype(self.dtype)), axis=0)
      packed_qkvo = jnp.einsum('btd,drc->bcrt', jnp.concatenate((hidden, ones), axis=-1), weight,
                               precision=jax.lax.Precision(cfg.matmul_precision))
    compression = jnp.eye(self.bam_v, dtype=self.static_q_key.dtype)
    static_gates = None
    init = float(cfg.bam_static_read_gate_init)
    def static_multiplier(arm):   # an ungated static Q/K read passes a constant 1 to the kernel
      name = f'general_{arm}_static_gate'
      if name not in static_gate_names:
        return jnp.ones(inputs_q.shape[:2] + (self.num_query_heads,), self.dtype)
      gate = (1.1 * jax.nn.sigmoid(raw[name] + jnp.asarray(getattr(self, name + '_bias'), self.dtype))
              if fused else self._static_gate(name, inputs_q))
      return gate / init
    static_gates = tuple(static_multiplier(arm) for arm in 'qkvo')
    body = 'v8t'
    fused_qk_norm = full_qk_norm and bool(getattr(cfg, 'bam_pallas_fused_qk_norm', False))
    # The read is linear in each V/O weight (raw static key + post-RMS dynamic scale), so v8t folds the
    # common scalar into both instead of a [B,T,N,V] epilogue that cannot fuse into Splash.
    fold_vo = common_vo_scale is not None
    vo_static = (lambda x: x * common_vo_scale.astype(x.dtype)) if fold_vo else (lambda x: x)
    # Fold the common local Q/K scalar into both static and post-RMS dynamic read weights.
    # With fused full-head QKNorm, the resulting local rows enter its RMS statistic already scaled;
    # the standard/RoPE arm stays unchanged. This also keeps zero/negative scalar gradients intact.
    fold_qk = common_qk_scale is not None and (not full_qk_norm or fused_qk_norm)
    qk_static = (lambda x: x * common_qk_scale.astype(x.dtype)) if fold_qk else (lambda x: x)
    weight = bam_pallas.static_weight(
        qk_static(self.static_q_key), qk_static(self.static_k_key),
        ((vo_static(self.static_v_key))),
        vo_static(self.static_o_key), compression)
    read_kwargs = {}
    if self._m_shift_mode != 'none':
      # Same previous-token share as mix_bam_previous_token (cast to the M dtype, zero where invalid);
      # the read kernel mixes M_t + g'(M_{t-1} - M_t) per V row; writes still accumulate onto M_in.
      with jax.named_scope('bam/m_shift'):
        shift_logits = raw['M_shift_gate'] if fused else self.m_shift_gate(inputs_q)
        shift_logits = shift_logits.astype(jnp.float32) + self.m_shift_bias.astype(jnp.float32)
        valid = bam_m_shift_valid_mask(inputs_q.shape, decoder_segment_ids, inputs_positions)
        shift = jax.nn.sigmoid(shift_logits).astype(M_in.dtype)
        read_kwargs['shift'] = jnp.where(valid[..., None], shift, jnp.zeros_like(shift))
        if getattr(cfg, 'bam_record_m_shift_health', False):
          # Compact summary as the XLA path records it; M is token-major [B,T,V,K], so the
          # (masked) share broadcasts over K and the mix is built only at the sampled positions.
          def mixed(indices, M=M_in, g=read_kwargs['shift']):
            current, previous = M[:, indices], M[:, jnp.maximum(indices-1, 0)]
            share = g[:, indices][..., None]
            return (jnp.asarray(1, M.dtype) - share) * current + share * previous
          stats, _, state = bam_m_shift_health(M_in, mixed, jax.nn.sigmoid(shift_logits),
                                               decoder_segment_ids, inputs_positions,
                                               column_metrics=False)
          self.sow('intermediates', 'm_shift_m_summary',
                   jnp.stack((stats[0], stats[1], stats[4], stats[5], state[4])))
    if self._m_read_norm == 'rms':
      # _matrix_for_read inside the v8t read: one FP32 RMS per token over the (shifted) matrix.
      read_kwargs['m_norm'] = self._rms_epsilon
    # Common scalars are passed per head ([N], constant over V): the reverse then keeps only a
    # c-summed gradient instead of per-c scratch (the 2.8B fused reverse sits at the VMEM limit).
    if fold_vo:
      for arm in ('v', 'o'):
        gamma = read_kwargs.get(arm + '_dynamic_scale')
        read_kwargs[arm + '_dynamic_scale'] = common_vo_scale.astype(jnp.float32) * (
            jnp.ones(q_key.shape[2:3], jnp.float32) if gamma is None else gamma.astype(jnp.float32))
    if fold_qk:
      for arm in ('q', 'k'):
        read_kwargs[arm + '_dynamic_scale'] = common_qk_scale.astype(jnp.float32) * jnp.ones(q_key.shape[2:3], jnp.float32)
    if getattr(cfg, 'bam_pallas_read_chunk', None):
      read_kwargs['chunk'] = int(cfg.bam_pallas_read_chunk)
    if getattr(cfg, 'bam_pallas_read_xhat_double_buffer', False):
      read_kwargs['xhat_double_buffer'] = True
    if packed_qkvo is not None:
      if body != 'v8t':
        raise ValueError('bam_pallas_packed_read_keys requires v8t')
      read_kwargs['keys_cnt'] = packed_qkvo
    if fused_qk_norm:
      # Same full-head QKNorm, computed in the read epilogue: the kernel normalizes the complete
      # row and multiplies by [local scales, ones] (Q also by the logit scale), so neither the
      # norm epilogue nor the query division runs in XLA between the read and attention.
      if body != 'v8t':
        raise ValueError('bam_pallas_fused_qk_norm requires v8t')
      cols = self._qk_col_width
      ones = jnp.ones((standard_qk_width,), jnp.float32)
      m_q, m_k = (jnp.concatenate((x[:cols].astype(jnp.float32), ones)) for x in head_scale)
      m_q = m_q / math.sqrt(self._attention_logit_scale_dim(full_qk_width))
      read_kwargs['qk_norm'] = (standard_sumsq[0], standard_sumsq[1], m_q, m_k,
                                float(cfg.normalization_layer_epsilon))
    # Full-head norm needs the unscaled, unrotated concatenated Q/K. In that case
    # attention scaling must follow QKNorm, rather than being folded into the read.
    q_scale = 1.0 / math.sqrt(self._attention_logit_scale_dim(full_qk_width)) if not full_qk_norm else 1.0
    o_read_logits = logits('W_R_gate', True)
    with jax.named_scope('bam/pallas_read'):
      outs = bam_pallas.read(
          M_in, weight, q_key, logits('W_lq_gate', False), k_key, logits('W_lk_gate', False),
          vo_key, o_read_logits, logits('W_lv_gate', False), query, key,
          qk_cols=self._qk_col_width, read_epsilon=self._read_key_epsilon,
          key_scale=self._read_key_scale,
          forward_tile=int(getattr(cfg, 'bam_pallas_read_tile', None) or 128),
          reverse_tile=int(getattr(cfg, 'bam_pallas_read_reverse_tile', None) or 128),
          vmem_mib=getattr(cfg, 'bam_pallas_vmem_mib', None),
          interpret=bool(getattr(cfg, 'bam_pallas_interpret', False)),
          q_scale=q_scale, static_gates=static_gates,
          **read_kwargs)
    query, key, value, local_output = outs[:4]
    if common_qk_scale is not None and not full_qk_norm and not fold_qk:
      # Complete local/no-position prefix only; standard RoPE Q/K is untouched.
      # Full-head QKNorm already incorporates this scale in its norm epilogue.
      def scale_local(x):
        prefix = (x[..., :self._qk_col_width].astype(jnp.float32) * common_qk_scale).astype(x.dtype)
        return jnp.concatenate((prefix, x[..., self._qk_col_width:]), axis=-1)
      query, key = scale_local(query), scale_local(key)
    if common_vo_scale is not None and not fold_vo:
      value = (value.astype(jnp.float32) * common_vo_scale).astype(value.dtype)
      local_output = (local_output.astype(jnp.float32) * common_vo_scale).astype(local_output.dtype)
    M_in = outs[4]
    if full_qk_norm and not fused_qk_norm:
      local = slice(None, self._qk_col_width)
      eps = float(cfg.normalization_layer_epsilon)

      def finish(x, scale, standard_ss):
        # Keep the read output in its own dtype/layout: accumulate the local sum of squares in f32
        # inside the reduction and apply 1/r and the local scales as one elementwise pass, so no f32
        # copy of the [B,T,N,K] output is materialized (that copy plus its relayout cost ~170 ms/step).
        local_ss = jnp.sum(jnp.square(x[..., local].astype(jnp.float32)), axis=-1)
        local_scale = scale[local]
        if common_qk_scale is not None:
          # Scale complete local Q/K before the full-head RMS statistic. Fuse with
          # the existing norm epilogue; the narrow standard/RoPE arm stays unscaled.
          local_ss = local_ss * jnp.square(common_qk_scale.astype(jnp.float32))
          local_scale = local_scale * common_qk_scale.astype(local_scale.dtype)
        inv_rms = lax.rsqrt((local_ss + standard_ss) / full_qk_width + eps).astype(x.dtype)
        multiplier = jnp.concatenate((local_scale, jnp.ones((standard_qk_width,), scale.dtype)))
        return x * inv_rms[..., None] * multiplier.astype(x.dtype)

      query = finish(query, head_scale[0], standard_sumsq[0])
      key = finish(key, head_scale[1], standard_sumsq[1])
    query = nn.with_logical_constraint(query, self.query_axis_names)
    key = nn.with_logical_constraint(key, self.key_axis_names)
    value = nn.with_logical_constraint(value, self.value_axis_names)
    if q_scale == 1.0 and not fused_qk_norm:
      query = query / jnp.sqrt(self._attention_logit_scale_dim(query.shape[-1])).astype(self.dtype)
    t = query.shape[1]
    local_window = t if (is_global or self.sliding_window_size is None) else min(t, int(self.sliding_window_size))
    if _bam_splash_enabled(cfg, t):
      y_std = _bam_splash_attention(query, key, value, decoder_segment_ids, cfg, self.mesh,
                                    window=local_window, seq_minor=_bam_splash_seq_minor(cfg, False))
    elif self._query_chunk_size is not None:
      y_std, _ = self._query_chunk_op(query, key, value, decoder_segment_ids, local_window)
    else:
      y_std, _ = self._attention_block(query, key, value, decoder_segment_ids,
                                       q0=0, s0=0, window_size=local_window)
    o_head = y_std + local_output
    if not self._has_write:
      out = nn.with_logical_constraint(o_head, self.out_axis_names)
      return self.out_projection(inputs_q.shape[-1], out), M_in, None
    if fused:
      address = self.P_loc_up(nn.gelu(raw['P_loc_down']))
    else:
      address = self._write_address(inputs_q)
    if fused:
      write_logits = raw['W_gw'] + jnp.asarray(self.gw_b0, self.dtype)
    else:
      write_logits = self.W_gw(inputs_q) + jnp.asarray(self.gw_b0, self.dtype)
    out = nn.with_logical_constraint(o_head, self.out_axis_names)
    write_content = o_head
    return self.out_projection(inputs_q.shape[-1], out), M_in, (write_content, write_logits, address)

  @nn.compact
  def __call__(
      self,
      inputs_q: Array,
      inputs_kv: Array,
      inputs_positions: Array,
      decoder_segment_ids: Array | None = None,
      decoder_input_tokens: Array | None = None,
      *,
      model_mode: str = common_types.MODEL_MODE_TRAIN,
      deterministic: bool = False,
      eos_sum: Array | None = None,
      deep_embedding: Array | None = None,
      M_in: Array | None = None,
      is_global: Array | bool | None = None,
      layer_index: Array | int | None = None,
      defer_write: bool = False,
  ):
    """BAM forward. Returns (out, M_out): out [b,t,emb_dim], M_out [b,t,k,v].

    v0.1 supports train mode and n==n_kv only; prefill/decode error out, deferred to v0.2.
    """
    cfg = self.config
    assert model_mode == common_types.MODEL_MODE_TRAIN, "BamAttention v0.1 supports train mode only"
    assert self.num_query_heads == self.num_kv_heads, "BamAttention v0.1 requires n==n_kv (no GQA)"

    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)
    M_read = M_in
    # The fused core (v8t) computes the same gate and mixes inside its read kernel.
    if self._m_shift_mode != 'none' and not getattr(cfg, 'bam_pallas_core', False):
      assert M_in is not None
      with jax.named_scope('bam/m_shift'):
        logits = self.m_shift_gate(inputs_q).astype(jnp.float32) + self.m_shift_bias.astype(jnp.float32)
        gate = jax.nn.sigmoid(logits)  # Gate is the previous-token share.
        M_read = mix_bam_previous_token(M_in, gate, decoder_segment_ids, inputs_positions)
        if getattr(cfg, 'bam_record_m_shift_health', False):
          compact = bool(getattr(cfg, 'bam_m_shift_compact_health', False))
          stats, columns, state = bam_m_shift_health(M_in, M_read, gate, decoder_segment_ids,
                                                    inputs_positions, column_metrics=not compact)
          if compact:   # Five scalars per layer; the fused core records the same summary.
            self.sow('intermediates', 'm_shift_m_summary',
                     jnp.stack((stats[0], stats[1], stats[4], stats[5], state[4])))
          else:
            self.sow('intermediates', 'm_shift_gate_stats', stats)
            self.sow('intermediates', 'm_shift_gate_columns', columns)
            self.sow('intermediates', 'm_shift_state_stats', state)


    full_m_shared_hidden = nn.gelu(self.full_m_read_down(inputs_q))

    assert 0 < self._qk_col_width <= self.head_dim
    standard_qk_width = self._standard_qk_width or (self.head_dim - self._qk_col_width)
    full_qk_norm = cfg.qk_norm and cfg.bam_qk_norm_after_concat
    common_vo_scale = None
    if getattr(cfg, 'bam_common_vo_read_scale', False):
      actual_layer = self.layer_inx if layer_index is None else layer_index
      initial_scale = getattr(cfg, 'bam_common_vo_read_scale_init', None)
      if initial_scale is None:
        initial_scale = jax.lax.rsqrt(1.0 + 2.0 * jnp.asarray(actual_layer, jnp.float32))
      common_vo_scale = self._constant_general_parameter(
          'common_vo_read_scale', (1,), (None,), initial_scale)[0]
    common_qk_scale = None
    if getattr(cfg, 'bam_common_qk_read_scale', False):
      if cfg.qk_norm and not full_qk_norm:
        raise ValueError('Common Q/K scale requires concatenated local Q/K; enabled QKNorm must cover the full head')
      common_qk_scale = self._constant_general_parameter(
          'common_qk_read_scale', (1,), (None,), 1.0)[0]
    if getattr(cfg, 'bam_pallas_core', False):
      return self._pallas_core_call(
          inputs_q, inputs_positions, decoder_segment_ids, M_in, is_global,
          standard_qk_width, deterministic=deterministic, common_vo_scale=common_vo_scale,
          common_qk_scale=common_qk_scale)
    # ---- QKV projection + QKNorm + RoPE ----
    query = self.query_projection(inputs_q,
                                  standard_qk_width)
    key = self.kv_projection(inputs_kv, proj_name="key",
                             projection_dim=standard_qk_width)
    value = jnp.zeros(inputs_kv.shape[:-1] + (self.num_kv_heads, self.head_dim), self.dtype)

    Mh = None
    if not full_qk_norm:
      query, key = dc.QKNorm(cfg, name='qk_norm')(query, key)
      query = self.apply_rotary_embedding(query, inputs_positions, name='query_rotary',
                                          embedding_dims=standard_qk_width)
      key = self.apply_rotary_embedding(key, inputs_positions, name='key_rotary',
                                        embedding_dims=standard_qk_width)

    # Full-head QKNorm waits for this M read; compatibility/additive paths already rotated.
    local_compressed_M = None
    if Mh is None:
      with jax.named_scope("bam/normalize_m"):
        Mh = self._matrix_for_read(M_read)
    with jax.named_scope("bam/read_local_m_for_qk"):
      assert Mh is not None, "local_qk read requires M_in"
      local_compressed_M = Mh
      qk_compressed_M = local_compressed_M
      q_local = self._read_direct_qk_c8('q', Mh, qk_compressed_M, inputs_q, full_m_shared_hidden)
      k_local = self._read_direct_qk_c8('k', Mh, qk_compressed_M, inputs_q, full_m_shared_hidden)
      if common_qk_scale is not None:
        q_local = (q_local.astype(jnp.float32) * common_qk_scale).astype(q_local.dtype)
        k_local = (k_local.astype(jnp.float32) * common_qk_scale).astype(k_local.dtype)
      query, key = self._add_local_qk(query, key, q_local, k_local, decoder_segment_ids)

    if full_qk_norm:
      query, key = dc.QKNorm(cfg, name='qk_norm')(query, key)
      query = self._apply_partial_rope(query, inputs_positions, name='query_rotary')
      key = self._apply_partial_rope(key, inputs_positions, name='key_rotary')

    query = nn.with_logical_constraint(query, self.query_axis_names)
    key = nn.with_logical_constraint(key, self.key_axis_names)
    value = nn.with_logical_constraint(value, self.value_axis_names)

    o_column_weight_multiplier = None
    local_output = None
    if Mh is None:
      Mh = self._matrix_for_read(M_read)
    vo_outputs = self._independent_local_vo(
        Mh, inputs_q, local_compressed_M, full_m_shared_hidden,
        o_column_weight_multiplier=o_column_weight_multiplier,
        return_o_logits=False)
    v_local, local_output = vo_outputs[:2]
    static_v = self._static_column(Mh, 'v', inputs_q)
    self._record_concat_amplitude('static_v', static_v, v_local)
    v_local = v_local + static_v
    static_o = self._static_column(
        Mh, 'o', inputs_q, column_weight_multiplier=o_column_weight_multiplier)
    self._record_concat_amplitude('static_o', static_o, local_output)
    local_output = local_output + static_o
    if self._concat_health:
      self.sow('intermediates', 'concat_local_v_content', jnp.stack((
          jnp.sqrt(jnp.mean(v_local.astype(jnp.float32) ** 2)),)))
    value = value + v_local

    if common_vo_scale is not None:
      value = (value.astype(jnp.float32) * common_vo_scale).astype(value.dtype)
      local_output = (local_output.astype(jnp.float32) * common_vo_scale).astype(local_output.dtype)
    query = query / jnp.sqrt(self._attention_logit_scale_dim(query.shape[-1])).astype(self.dtype)
    if cfg.float32_qk_product:
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)

    _, t, _, _ = query.shape

    local_window = (
        t if self.sliding_window_size is None
        else min(t, int(self.sliding_window_size)))
    if is_global:
      local_window = t
    if _bam_splash_enabled(self.config, t):
      y_std, Mbar = _bam_splash_attention(
          query, key, value, decoder_segment_ids, self.config, self.mesh, window=local_window,
          seq_minor=_bam_splash_seq_minor(self.config, True)), None
    elif self._query_chunk_size is not None:
      y_std, Mbar = self._query_chunk_op(
          query, key, value, decoder_segment_ids, local_window,
          )
    else:
      y_std, Mbar = self._attention_block(
          query, key, value, decoder_segment_ids,
          q0=0, s0=0, window_size=local_window,
          )

    if local_output is not None:
      sl = slice(None, self.bam_k)
      self._record_concat_amplitude('local_o', local_output[..., sl], y_std[..., sl])
    if local_output is not None:
      o_head = y_std if local_output is None else y_std + local_output
    if defer_write:
      assert self._has_write and M_in is not None
      factors = self._deferred_write_factors(o_head, inputs_q, full_m_shared_hidden)
      out = nn.with_logical_constraint(o_head, self.out_axis_names)
      return self.out_projection(inputs_q.shape[-1], out), M_in, factors
    if self._has_write:
      assert M_in is not None, "write primitive requires M_in"
      with jax.named_scope("bam/write_m"):
        M_out, _ = self._write(o_head, inputs_q, M_in, full_m_shared_hidden)
    else:
      M_out = M_in

    out = nn.with_logical_constraint(o_head, self.out_axis_names)
    return self.out_projection(inputs_q.shape[-1], out), M_out
