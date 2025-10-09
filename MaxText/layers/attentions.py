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

import enum
import functools
import math
from typing import Any, Optional, Tuple

from flax import linen as nn
import jax
from jax import lax
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import shard_map
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask
import jax.numpy as jnp
from einops import rearrange, repeat

import common_types
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
from layers import head_pool

import maxtext_utils
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
PRNGKey = common_types.PRNGKey

DenseGeneral = linears.DenseGeneral
RMSNorm = linears.RMSNorm
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

# Used to pass in splash attention block sizes from config.
global_block_q = 0
global_block_kv = 0
global_block_kv_compute = 0
global_block_q_dkv = 0
global_block_kv_dkv = 0
global_block_kv_dkv_compute = 0
global_block_q_dq = 0
global_block_kv_dq = 0
global_use_fused_bwd_kernel = False
global_q_layout = ""
global_k_layout = ""
global_v_layout = ""

nd_dense_init = initializers.nd_dense_init
shard_map = shard_map.shard_map

NormalInitializer = initializers.nd_dense_init_normal

dynamic_vector_slice_in_dim = jax.vmap(lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


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
  query_chunk_size: Optional[int] = None
  key_wise: bool = False

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
      attn_mask=None,
      attn_bias=None,
      sinks=None,
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
                                # key_wise=self.key_wise,
                                query_chunk_size=self.query_chunk_size)(
                                                    query, key, value, decoder_segment_ids, model_mode, attn_mask=attn_mask,
                                                    attn_bias=attn_bias, sinks=sinks)

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
    local_max = jnp.max(attn_weights, axis=-1, keepdims=True)
    local_exps = jnp.exp(attn_weights - local_max)
    local_sum = jnp.sum(local_exps, axis=-1, keepdims=True)

    local_sum = jnp.moveaxis(local_sum, -2, 1)
    local_max = jnp.moveaxis(local_max, -2, 1)

    local_max = jnp.reshape(local_max, (local_max.shape[0], local_max.shape[1], local_max.shape[2] * local_max.shape[3], 1))
    local_sum = jnp.reshape(local_sum, (local_sum.shape[0], local_sum.shape[1], local_sum.shape[2] * local_sum.shape[3], 1))

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
    attn_weights = self.qk_product(query, key, q_seq_len, model_mode)

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
    if key.shape != value.shape and self.config.attention_type != AttentionType.MLA.value and self.config.qk_head_dim is None:
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
  def __call__(self, query, key, value, decoder_segment_ids, model_mode, *args, input_q=None, input_kv=None, hidden_states=None,
               attn_mask=None, attn_bias=None, sinks=None): # lsp
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
        attn_mask=attn_mask,
        attn_bias=attn_bias,
        sinks=sinks
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
        attn_mask=attn_mask,
    )

    if ar_unnormalized_output is not None:
      unnormalized_outputs = [prefill_unnormalized_output, ar_unnormalized_output]
      exponentials_maxes = [prefill_exponentials_max, ar_exponentials_max]
      exponentials_sums = [prefill_exponentials_sum, ar_exponentials_sum]
      return self.normalize_attention(unnormalized_outputs, exponentials_maxes, exponentials_sums)
    else:
      return prefill_unnormalized_output / prefill_exponentials_sum

def segsum(x):  # adapted from https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py#L23-L32
    T = x.shape[-1]
    x = repeat(x, "... T -> ... T S", S=T)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=-1)
    x = jnp.where(mask, x, 0)
    x_segsum = jnp.cumsum(x, axis=-2)
    # XD: leave to attention mask
    # mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    # x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
    return x_segsum

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
  layer_inx: int = 0
  use_kv_shift: bool = False
  use_alibi: bool = False
  use_postnorm: bool = False
  query_chunk_size: Optional[int] = None
  use_dc: bool = False
  use_v_gate: bool = False
  key_wise: bool = False  

  def setup(self):
    # use_attn_sink = self.config.use_attn_sink if getattr(self.config, 'use_attn_sink', False) else self.use_attn_sink
    self.sinks = self.param('sinks_bias',  # XD: append bias to name to skip weight decay
        nn.with_logical_partitioning(
            nn.initializers.zeros,
            ('heads',)  # XD: need check
        ),
        (self.num_query_heads,),
        self.dtype
    ) if self.config.use_attn_sink else None

    if self.use_dc:
      self.attention_op = dc.AttentionOp(self.config, self.quant, self.sliding_window_size, query_chunk_size=self.query_chunk_size, key_wise=self.key_wise,
                                        num_query_heads=self.num_query_heads, num_kv_heads=self.num_kv_heads)
    else:
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
        query_chunk_size=self.query_chunk_size,
    )

    if self.config.merge_kvshift_vr:
      self.kv_shift_vr = kv_shift.KVshiftVR(config=self.config,mesh=self.mesh, quant=self.quant, kernel_init=self.kernel_init)
    else:
      if self.use_kv_shift:
        self.kv_shift = kv_shift.KVshift(config=self.config,mesh=self.mesh, quant=self.quant, kernel_init=self.kernel_init,
                                         num_kv_heads=self.num_kv_heads)
      
      if self.config.value_residual_learning:
        self.value_residual = kv_shift.ValueResidual(config=self.config,mesh=self.mesh, quant=self.quant, kernel_init=self.kernel_init)
    
    if self.config.use_head_pool:
      self.head_pool = head_pool.HeadPool(config=self.config,mesh=self.mesh, quant=self.quant, kernel_init=self.kernel_init)

    cfg = self.config
    if self.use_postnorm:
      self.mixv_postnorm = True if self.config.mixv_postnorm is None else self.config.mixv_postnorm
      self.o_postnorm = False if self.config.o_postnorm is None else self.config.o_postnorm
      assert int(self.o_postnorm) + int(self.mixv_postnorm) == 1
      norm_kwargs = {
                  "dtype": cfg.dtype,
                  "weight_dtype": cfg.weight_dtype,
                  "name": "post_norm",
                  "epsilon": cfg.normalization_layer_epsilon,
                  "scale_init": jax.nn.initializers.constant(self.config.postnorm_scale_init),
                  }
      self.post_norm = normalizations.get_rmsnorm(**norm_kwargs)
    
    if self.config.key_lora_dim:
      num_k_head = self.config.num_k_head if self.config.num_k_head is not None else self.num_kv_heads
      kernel_init_shard = nn.with_logical_partitioning(self.kernel_init, ('embed', None, None))
      self.wk_a = self.param('wk_a',kernel_init_shard, (self.config.emb_dim, num_k_head, self.config.key_lora_dim), self.weight_dtype)
      kernel_init_shard = nn.with_logical_partitioning(self.kernel_init, (None, None, None))
      self.wk_b = self.param('wk_b',kernel_init_shard, (self.config.key_lora_dim, num_k_head, self.head_dim), self.weight_dtype)
    if self.config.value_lora_dim:
      kernel_init_shard = nn.with_logical_partitioning(self.kernel_init, ('embed', None, None))
      self.wv_a = self.param('wv_a',kernel_init_shard, (self.config.emb_dim, self.num_kv_heads, self.config.value_lora_dim), self.weight_dtype)
      kernel_init_shard = nn.with_logical_partitioning(self.kernel_init, (None, None, None))
      self.wv_b = self.param('wv_b',kernel_init_shard, (self.config.value_lora_dim, self.num_kv_heads, self.head_dim), self.weight_dtype)

  def query_projection(self, inputs_q: Array) -> Array:
    """Query projection."""

    # NOTE: T5 does not explicitly rescale the attention logits by
    #       1/sqrt(depth_kq)!  This is folded into the initializers of the
    #       linear transformations, which is equivalent under Adafactor.
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)

    # def query_init(*args):
    #   # pylint: disable=no-value-for-parameter
    #   return self.kernel_init(*args) / depth_scaling

    head_dim = self.config.qk_head_dim if self.config.qk_head_dim else self.head_dim
    query_proj = DenseGeneral(
        features=(self.num_query_heads, head_dim),
        axis=-1,
        kernel_init=self.kernel_init, # lsp
        kernel_axes=("embed", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="query",
        quant=self.quant,
        use_bias=self.config.use_bias,
        matmul_precision=self.config.matmul_precision,
    )(inputs_q)
    return query_proj

  def kv_projection(self, inputs_kv: Array, proj_name: str) -> Array:
    """Projection for Key and Value.

    Args:
      inputs_kv: inputs_kv: key/values of shape `[batch, kv_length,
        num_kv_heads, kv_dim]`.
      proj_name: name of projection, `key` or `value`.

    Returns:
      Projection of key or value, in shape of `[batch, kv_length, head_dim]`.
    """
    if self.num_kv_heads == -1:
      raise ValueError("num_kv_heads is not defined.")

    if self.num_query_heads % self.num_kv_heads != 0:
      raise ValueError("Invalid num_kv_heads for GQA.")

    kernel_axes = ("embed", "kv_heads", "kv_head_dim")

    head_dim = self.head_dim
    if proj_name == 'key' and self.config.qk_head_dim:
      head_dim = self.config.qk_head_dim 
    elif proj_name == 'value' and self.config.vo_head_dim:
      head_dim = self.config.vo_head_dim
    
    if proj_name == 'key' and self.config.key_lora_dim:
      hid = jnp.einsum('BTD,DNK->BTNK', inputs_kv, self.wk_a)
      kv_proj = jnp.einsum('BTNK, KNd->BTNd', hid, self.wk_b)
    elif proj_name == 'value' and self.config.value_lora_dim:      
      hid = jnp.einsum('BTD,DNK->BTNK', inputs_kv, self.wv_a)
      if self.config.value_lora_norm:
        hid = RMSNorm(dtype=self.config.dtype,weight_dtype=self.config.weight_dtype, name="v_lora_norm", 
              epsilon=self.config.normalization_layer_epsilon, kernel_axes=("norm",), scale_init=nn.initializers.ones,
              )(hid)
      kv_proj = jnp.einsum('BTNK, KNd->BTNd', hid, self.wv_b)
    else:
      kv_proj = DenseGeneral(
          features=(self.num_kv_heads, head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=kernel_axes,
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name=proj_name,
          quant=self.quant,
          use_bias=self.config.use_bias,
          matmul_precision=self.config.matmul_precision,
      )(inputs_kv)
    return kv_proj

  def vgate_projection(self, inputs_q: Array) -> Array:
    G = int(self.config.vo_head_dim / self.head_dim)
    vgate = DenseGeneral(
        features=(self.num_kv_heads, G), # DNG
        axis=-1,
        kernel_init=self.kernel_init, # lsp
        kernel_axes=("embed", "q_heads", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="vgate",
        quant=self.quant,
        use_bias=self.config.use_bias,
        matmul_precision=self.config.matmul_precision,
    )(inputs_q)
    return vgate

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
        use_bias=self.config.use_bias,
        matmul_precision=self.config.matmul_precision,
    )(inputs)
    qkv_proj = checkpoint_name(qkv_proj, "qkv_proj")
    query, key, value = qkv_proj[:, :, 0, ...], qkv_proj[:, :, 1, ...], qkv_proj[:, :, 2, ...]
    return query, key, value

  def out_projection(self, output_dim: int, out: Array) -> Array:
    out_proj = DenseGeneral(
        features=output_dim,
        axis=(-2, -1),
        kernel_init=self.kernel_init,
        kernel_axes=("heads", "kv", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="out",
        quant=self.quant,
        use_bias=self.config.use_bias,
        matmul_precision=self.config.matmul_precision,
    )(out)
    return out_proj

  def apply_rotary_embedding(self, inputs: Array, inputs_positions: Array, name: str):
    """Applies rotary embeddings, handling different model types.

    Args:
      inputs: The input tensor to apply rotary embeddings to.
      inputs_positions: The positions of the inputs.
      name: A name for the embedding layer.

    Returns:
      The input tensor with rotary embeddings applied.
    """
    if self.config.attention_type == AttentionType.MLA.value:
      # For MLA attention RoPE is applied to only `self.qk_rope_head_dim` portion the heads.
      rope_embedding_dims = self.qk_rope_head_dim
    else:
      rope_embedding_dims = self.config.qk_head_dim if self.config.qk_head_dim else self.head_dim

    rope_embedding_dims = int(rope_embedding_dims * self.config.rope_ratio)

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
          name=name,
      )
    if self.config.rope_ratio == 1:
      inputs = rotary_embedding(inputs, inputs_positions)
    else:
      inputs = jnp.concatenate([rotary_embedding(inputs[...,:rope_embedding_dims], inputs_positions), inputs[..., rope_embedding_dims:]], axis=-1)
    return inputs

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
      hidden_states=None,
      value_residual=None,
      ffn_act=None,
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
    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)

    # max_logging.log(f'num_kv_heads: {self.num_kv_heads}, use_v_gate: {self.use_v_gate}, key_wise: {self.key_wise}')
    # apply projection.
    if self.config.fused_qkv:
      query, key, value = self.qkv_projection(inputs_q, proj_name="qkv_proj")
      inputs_k = inputs_v = inputs_kv
    elif self.config.dense_conn and self.config.dynamic_dense_type == 'qkvm':
        assert isinstance(inputs_kv, (tuple, list)) and len(inputs_kv) == 2
        inputs_k, inputs_v = inputs_kv
        query = self.query_projection(inputs_q)
        key = self.kv_projection(inputs_k, proj_name="key")
        value = self.kv_projection(inputs_v, proj_name="value")
    elif self.config.inner_ffn_way is not None:
      if self.config.inner_ffn_way == 'q':
        inputs_k = inputs_v = hidden_states
      elif self.config.inner_ffn_way == 'k':
        inputs_q = inputs_v = hidden_states
        inputs_k = inputs_kv
      elif self.config.inner_ffn_way == 'v':
        inputs_q = inputs_k = hidden_states
        inputs_v = inputs_kv
      query = self.query_projection(inputs_q)
      key = self.kv_projection(inputs_k, proj_name="key")
      value = self.kv_projection(inputs_v, proj_name="value")
    else:
      query = self.query_projection(inputs_q)
      key = self.kv_projection(inputs_kv, proj_name="key")
      value = self.kv_projection(inputs_kv, proj_name="value")
      inputs_k = inputs_v = inputs_kv

    if self.config.merge_kvshift_vr:
      if self.layer_inx == 0:
          value_residual = value
      inputs_k, inputs_v = inputs_kv if isinstance(inputs_kv, (tuple, list)) and len(inputs_kv) == 2 else (inputs_kv, inputs_kv)
      key, value = self.kv_shift_vr(key, value, value_residual, inputs_k=inputs_k, inputs_v=inputs_v, inputs_m=hidden_states)
    else:
      if self.config.value_residual_learning:
        if self.layer_inx == 0:
          value_residual = value
        else:
          inputs_k, inputs_v = inputs_kv if isinstance(inputs_kv, (tuple, list)) and len(inputs_kv) == 2 else (inputs_kv, inputs_kv)
          value = self.value_residual(inputs_v, value, value_residual, inputs_m=hidden_states)

      if self.use_kv_shift:
        inputs_k, inputs_v = inputs_kv if isinstance(inputs_kv, (tuple, list)) and len(inputs_kv) == 2 else (inputs_kv, inputs_kv)
        query, key, value, shifted_inputs_k, shifted_inputs_v = self.kv_shift(
          inputs_q, query, key, value, inputs_k=inputs_k, inputs_v=inputs_v, inputs_m=hidden_states)
    
    if self.config.use_head_pool:
      query, key, value, o_out, ow = self.head_pool(inputs_q, query, key, value, inputs_m=hidden_states, ffn_act=ffn_act)

    query, key = dc.QKNorm(self.config, name='qk_norm')(query, key) # lsp

    # apply ROPE
    if not self.use_alibi:
      query = self.apply_rotary_embedding(query, inputs_positions, name="query_rotary")
      key = self.apply_rotary_embedding(key, inputs_positions, name="key_rotary")


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

    assert not self.config.quantize_kvcache or self.kv_quant

     # lsp
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query /= depth_scaling

    if self.num_query_heads > self.num_kv_heads: # GQA
      n_expands = self.num_query_heads // self.num_kv_heads
      if key.shape[-2] < self.num_query_heads: # for K lora
        key = jnp.repeat(key, n_expands, axis=-2) # BSNd
      if value.shape[-2] < self.num_query_heads:  # for V lora
        value = jnp.repeat(value, n_expands, axis=-2) 
    
    if self.config.use_k_gate:
      k_gate = DenseGeneral((self.num_query_heads,),dtype=self.dtype,weight_dtype=self.weight_dtype,quant=self.quant,
            kernel_init=self.kernel_init,kernel_axes=('embed', None),name="k_gate",
        )(shifted_inputs_k if self.confg.use_shifted_k_gate_inputs else inputs_k)
      k_gate = jax.nn.tanh(k_gate) + 1  if self.config.k_gate_tanh else jax.nn.sigmoid(k_gate) 
      key = key * k_gate[...,None] # BSND, BSN1->BSND

    if self.use_v_gate:
      use_v_gate_bias = self.config.use_v_gate_bias if self.config.use_v_gate_bias is not None else False
      v_gate = DenseGeneral((self.num_query_heads,),dtype=self.dtype,weight_dtype=self.weight_dtype,quant=self.quant,
            kernel_init=self.kernel_init,kernel_axes=('embed', None),name="v_gate", use_bias=use_v_gate_bias,
        )(shifted_inputs_v if self.config.use_shifted_v_gate_inputs else inputs_v)
      v_gate = jax.nn.tanh(v_gate) + 1  if self.config.v_gate_tanh else jax.nn.sigmoid(v_gate) 
      value = value * v_gate[...,None] # BSND, BSN1->BSND

    attn_bias = None
    kwargs = dict(dtype=self.dtype, weight_dtype=self.weight_dtype, quant=self.quant)
    if getattr(self.config, 'use_fox', False):  # forgetting transformer
      x = {'q': inputs_q, 'k': inputs_k, 'v': inputs_v, 'm': hidden_states}[self.config.fgate_input] \
        if self.config.fgate_input is not None else inputs_kv
      fgate_logit = DenseGeneral((self.num_query_heads,), kernel_axes=('embed', 'heads'), name="fgate_proj",
                                 kernel_init=self.kernel_init, use_bias=True,
                                 bias_init=initializers.constant_init(self.config.fgate_bias_init)
                                 if self.config.fgate_bias_init is not None else None, **kwargs)(x)
      fgate_logit = rearrange(fgate_logit, "B T N -> B N T")
      log_fgate = jax.nn.log_sigmoid(fgate_logit.astype(jnp.float32))
      forget_bias = segsum(log_fgate)  # BNTS
      attn_bias = -forget_bias

    if getattr(self.config, 'use_selective_attn', False):  # https://arxiv.org/abs/2410.02703
      head_dim = self.config.qk_head_dim or self.head_dim
      q, k = [DenseGeneral((head_dim,), kernel_axes=('embed', 'kv'), name=f"selective_attn_proj_{name}", 
             kernel_init=self.kernel_init, use_bias=self.config.use_bias, **kwargs)(x)
             for name, x in [("query", inputs_q), ("key", inputs_k)]]
      S = jnp.einsum("B T D, B S D -> B T S", q, k) / head_dim**0.5
      # adapted from https://github.com/fangyuan-ksgk/selective-attention-transformer/blob/main/model/model.py#L197-L204
      S = jax.nn.relu(S)  # Only positive selection | (bs, seqlen, seqlen)
      S = S.at[..., 0].set(0)  # S[..., 0] = 0  # Do not mask <BOS> | first token in sequence is not masked (beginning of sequence)
      S = (1 - jnp.eye(S.shape[-1])) * S  # TS*BTS->BTS. Do not mask self | zero-out diagonal elements
      S = jnp.roll(S, 1, axis=-2)  # each token is able to mask next token's attention (Key step)
      S = S.at[..., 0, :].set(0)  # S[..., 0, :] = 0 # roll operation creates redundant beginning mask, so we zero-out the first row
      S = S.astype(jnp.float32)
      selective_mask = jnp.cumsum(S, axis=-1)  # Accumulate selection mask for each token (decided by all its previous token's)
      if getattr(self.config, 'selective_attn_dynamic_qw', False):
        qw = DenseGeneral((self.num_query_heads,), kernel_axes=('embed', 'kv'), name=f"selective_attn_dyn_qw_proj", use_bias=True, 
                          kernel_init=initializers.nd_dense_init_normal(0.001), bias_init=initializers.constant_init(1.0), **kwargs)(inputs_q)
        # selective_mask = selective_mask * rearrange(qw, 'B T N -> B N T 1')  # B1TS*BNT1->BNTS
        selective_mask = jnp.einsum('BTS,BTN->BNTS', selective_mask, qw)
      elif getattr(self.config, 'selective_attn_dynamic_kw', False):
        kw = DenseGeneral((self.num_query_heads,), kernel_axes=('embed', 'kv'), name=f"selective_attn_dyn_kw_proj", use_bias=True, 
                          kernel_init=initializers.nd_dense_init_normal(0.001), bias_init=initializers.constant_init(1.0), **kwargs)(inputs_k)
        # selective_mask = selective_mask * rearrange(kw, 'B S N -> B N 1 S')  # B1TS*BN1S->BNTS
        selective_mask = jnp.einsum('BTS,BSN->BNTS', selective_mask, kw)
      else:
        selective_mask = selective_mask[:, None]  # BTS->B1TS
      attn_bias = -selective_mask if attn_bias is None else attn_bias - selective_mask
    if attn_bias is not None:
      attn_bias = nn.with_logical_constraint(attn_bias, ('activation_batch', 'heads', 'activation_length', None),)  # XD: necessary?

    if self.config.record_internal_nn_metrics:
      if self.config.sigmoid_attention:
        self.sow('intermediates', 'q_norm_stat', maxtext_utils.l2norm(query))
        self.sow('intermediates', 'k_norm_stat', maxtext_utils.l2norm(key))
        attn_logits = jnp.tril(jnp.einsum('B T N D, B S N D -> B N T S', query, key) - 7.62)
        self.sow('intermediates', 'attn_logits_max', attn_logits.max())
        self.sow('intermediates', 'attn_logits_min', -1 * (-attn_logits).max())
        self.sow('intermediates', 'attn_logits_mean', attn_logits.mean())

    out = self.attention_op(query, key, value, decoder_segment_ids, model_mode, inputs_q, inputs_kv,
                            hidden_states=hidden_states, attn_bias=attn_bias, sinks=self.sinks)

    if self.config.use_o_gate:
      o_gate = DenseGeneral((self.config.num_out_heads,),dtype=self.dtype,weight_dtype=self.weight_dtype,quant=self.quant,
            kernel_init=self.kernel_init,kernel_axes=('embed', None),name="o_gate",)(inputs_q) # BTD,DN->BTN
      o_gate = jax.nn.tanh(o_gate) + 1  if self.config.o_gate_tanh else jax.nn.sigmoid(o_gate)
      out = jnp.repeat(out, self.config.num_out_heads // out.shape[-2] , axis=-2) 
      out = out * o_gate[...,None] # BSND, BSN1->BSND

    if self.config.o_gate_hidden_dim:
      o_gate_hidden = DenseGeneral((self.config.o_gate_hidden_dim,),dtype=self.dtype,weight_dtype=self.weight_dtype,quant=self.quant,
            kernel_init=self.kernel_init,kernel_axes=('embed', None), name="o_gate_proj_1",)(
            hidden_states if self.config.o_gate_use_inputs_m else inputs_q)
      if self.config.o_gate_hidden_act == 'sigmoid': o_gate_hidden = jax.nn.sigmoid(o_gate_hidden)
      o_gate = DenseGeneral((inputs_q.shape[-1],),dtype=self.dtype,weight_dtype=self.weight_dtype,quant=self.quant,
            kernel_init=self.kernel_init,kernel_axes=(None, 'embed'), name="o_gate_proj_2",)(o_gate_hidden)
      if self.config.o_gate_act == 'sigmoid': o_gate = jax.nn.sigmoid(o_gate)
      out = out * rearrange(o_gate, 'B T (N D) -> B T N D', N=self.config.num_query_heads)

    mixed_v = out
    if self.config.use_head_pool:
      if not self.config.hp_ablate_o and o_out is not None and not self.config.hp_o_shortcut:
        out = out + o_out
      if self.config.hp_o_transform:
        out = jnp.einsum("BTND,NM->BTMD", out, self.head_pool.sw_o)
      if self.config.hp_o_shortcut:
        out = out + o_out

    if self.use_postnorm and self.mixv_postnorm:
      b, t, n, d = out.shape
      out = jnp.reshape(out,(b,t,n*d))
      out = self.post_norm(out) # BTNd -> BT(Nd)-> BTNd
      out = jnp.reshape(out,(b,t,n,d))

    out = nn.with_logical_constraint(out, self.out_axis_names)

    if self.config.mixed_v_act:
      out = jax.nn.silu(out)

    if self.config.sub_head_gate:
      vgate = jax.nn.silu(self.vgate_projection(inputs_q)) # BTNG
      out = rearrange(out, 'B T N (G D) -> B T N G D', G=vgate.shape[-1]) * vgate[..., None]
      out = rearrange(out, 'B T N G D -> B T N (G D)')

    # apply output projection,  output dim is set to the input dim.
    out = self.out_projection(inputs_q.shape[-1], out)

    if self.config.use_head_pool and self.config.hp_out_proj:
      if self.config.hp_dynamic_mixed_v:
        ow = ow + self.head_pool.mixed_v_proj(mixed_v) # BTNd, dM-> BTNM 
        out = out + self.head_pool.out_proj(jnp.einsum("BTND,BTNM->BTMD", mixed_v, ow))
      else:
        out = out + self.head_pool.out_proj(mixed_v)

    if self.use_postnorm and self.o_postnorm: # hybrid norm
      out = self.post_norm(out, dynamic=self.config.attn_postnorm_dynamic) 
    out = checkpoint_name(out, "out_proj")
    return out, value_residual


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

    mla_rope_groups = self.config.mla_rope_groups if self.config.mla_rope_groups is not None else 1 
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
      self.q_norm = RMSNorm(
          dtype=self.config.dtype,
          weight_dtype=self.config.weight_dtype,
          name="q_norm",
          epsilon=self.config.normalization_layer_epsilon,
          kernel_axes=("norm",),
      )
      self.wq_b = DenseGeneral(
          features=(self.num_query_heads, self.qk_head_dim),
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("q_lora", "q_heads", "embed"), # TODO: fix sharding 
          # kernel_axes=("q_lora", "q_heads", "kv"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="wq_b",
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
      )

    # KV LoRA path.
    if self.config.mla_k_hidnrom:
      self.k_hidnorm = RMSNorm(
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        name="k_hidnorm",
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
    )
    if (self.config.dense_conn and self.config.dynamic_dense_type == 'qkvm') or self.config.mla_k_hidnrom:
      self.wkv_a_k = DenseGeneral(
          features=self.qk_rope_head_dim * mla_rope_groups,
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "kv_lora"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="wkv_a_k",
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
      )
      self.wkv_a_v = DenseGeneral(
          features=self.kv_lora_rank,
          axis=-1,
          kernel_init=self.kernel_init,
          kernel_axes=("embed", "kv_lora"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="wkv_a_v",
          quant=self.quant,
          matmul_precision=self.config.matmul_precision,
      )
    else:
      self.wkv_a = DenseGeneral(
        features=self.kv_lora_rank + self.qk_rope_head_dim * mla_rope_groups,
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("embed", "kv_lora"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="wkv_a",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
      )
    self.kv_norm = RMSNorm(
        dtype=self.config.dtype,
        weight_dtype=self.config.weight_dtype,
        name="kv_norm",
        epsilon=self.config.normalization_layer_epsilon,
        kernel_axes=("norm",),
        scale_init=nn.initializers.ones if self.config.mla_kv_norm_learnable else None,
    )
    self.wkv_b = DenseGeneral(
        features=(self.num_query_heads, (self.qk_nope_head_dim + self.v_head_dim)), # KNd, 192*16*(48+128) 
        axis=-1,
        kernel_init=self.kernel_init,
        kernel_axes=("kv_lora", "kv_heads", "embed"), # TODO: fix sharding 
        # kernel_axes=("kv_lora", "kv_heads", "kv_head_dim"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="wkv_b",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )

    if self.config.mla_num_groups is not None:
      num_heads_per_group = self.num_query_heads // self.config.mla_num_groups
      kernel_init_shard = nn.with_logical_partitioning(NormalInitializer(0.006), (None, None, None, 'embed'))
      shape = (self.config.mla_num_groups, self.kv_lora_rank // self.config.mla_num_groups, num_heads_per_group, self.qk_nope_head_dim + self.v_head_dim)
      self.wkv_b_kernel = self.param('wkv_b_kernel', kernel_init_shard, shape, self.weight_dtype) # NKMd, 4*(192//4)*4*(48+128), delta= 192*12*(48+128) 

    # Set softmax scaling.
    self.softmax_scale = self.qk_head_dim**0.5
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

  def mla_kv_projection(self, inputs: Array, inputs_positions: Array, hidden_states=None) -> Tuple[Array, Array]:
    """MLA key/value projection with integrated rotary embedding."""
    if isinstance(inputs, (list, tuple)):
      inputs_k, inputs_v = inputs
      low_rank_rope = self.wkv_a_k(inputs_k)
      low_rank_main = self.wkv_a_v(inputs_v)
    elif self.config.mla_k_hidnrom:
      low_rank_rope = self.wkv_a_k(self.k_hidnorm(hidden_states))
      low_rank_main = self.wkv_a_v(inputs)
    else:
      low_rank = self.wkv_a(inputs)
      low_rank_main, low_rank_rope = jnp.split(low_rank, [self.kv_lora_rank], axis=-1)

    if self.config.mla_num_groups is not None:
      low_rank_main = rearrange(low_rank_main, 'B T (N K) -> B T N K', N=self.config.mla_num_groups) # BTK -> BTNK
      low_rank_main = self.kv_norm(low_rank_main)
      kv_out = jnp.einsum('BTNK, NKMd -> BTMNd', low_rank_main, self.wkv_b_kernel)
      kv_out = rearrange(kv_out, 'B T M N d -> B T (M N) d')
    else:
      low_rank_main = self.kv_norm(low_rank_main)
      # Note: cache `low_rank_main` and `low_rank_rope` for inference.
      kv_out = self.wkv_b(low_rank_main)
    
    # Split kv_out into key_nope and value parts.
    key_nope, value = jnp.split(kv_out, [self.qk_nope_head_dim], axis=-1)

    # Apply rotary embedding to key_rope.
    if self.config.mla_rope_groups is not None:
      key_rope = rearrange(low_rank_rope, 'B T (G K) -> B T G K', G=self.config.mla_rope_groups) # BTK -> BTGK
      key_rope = self.apply_rotary_embedding(key_rope, inputs_positions, name="key_rope")
      key_rope = jnp.repeat(key_rope, self.num_query_heads//self.config.mla_rope_groups, axis=-2) # BTGK -> BT(GN)K
    else:
      key_rope = jnp.expand_dims(low_rank_rope, axis=2) # BTK -> BT1K
      key_rope = self.apply_rotary_embedding(key_rope, inputs_positions, name="key_rope")
      key_rope = jnp.broadcast_to(key_rope, (key_nope.shape[0], key_nope.shape[1], self.num_query_heads, key_rope.shape[3]))

    key = jnp.concatenate([key_nope, key_rope], axis=-1)
    return key, value

  def out_projection(self, output_dim: int, out: Array) -> Array:
    out_proj = DenseGeneral(
        features=output_dim,
        axis=(-2, -1),
        kernel_init=initializers.contant_dense_init(0.0) if self.config.mla_out_zero_init else self.kernel_init,
        kernel_axes=("heads", "kv", "embed"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="out",
        quant=self.quant,
        matmul_precision=self.config.matmul_precision,
    )(out)
    return out_proj

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
      hidden_states=None,
      value_residual=None,
      ffn_act=None,
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
    key, value = self.mla_kv_projection(inputs_kv, inputs_positions, hidden_states=hidden_states)

    if self.config.record_internal_nn_metrics:
      self.sow('intermediates', 'q_norm_stat', maxtext_utils.l2norm(query))
      self.sow('intermediates', 'k_norm_stat', maxtext_utils.l2norm(key))

    if model_mode == common_types.MODEL_MODE_PREFILL:
      query = nn.with_logical_constraint(query, self.prefill_query_axis_names)
      key = nn.with_logical_constraint(key, self.prefill_key_axis_names)
      value = nn.with_logical_constraint(value, self.prefill_value_axis_names)
    else:
      query = nn.with_logical_constraint(query, self.query_axis_names)
      key = nn.with_logical_constraint(key, self.key_axis_names)
      value = nn.with_logical_constraint(value, self.value_axis_names)

    if self.config.use_v_gate:
      v_gate = DenseGeneral((self.num_query_heads,),dtype=self.dtype,weight_dtype=self.weight_dtype,quant=self.quant,
            kernel_init=self.kernel_init,kernel_axes=('embed', None),name="v_gate",
        )(inputs_q)
      v_gate = jax.nn.tanh(v_gate) + 1  if self.config.v_gate_tanh else jax.nn.sigmoid(v_gate) 
      value = value * v_gate[...,None] # BSND, BSN1->BSND

    query = checkpoint_name(query, "query_proj")
    key = checkpoint_name(key, "key_proj")
    value = checkpoint_name(value, "value_proj")

    out = self.attention_op(query, key, value, decoder_segment_ids, model_mode, input_q=inputs_q, input_kv=inputs_kv)
    out = nn.with_logical_constraint(out, self.out_axis_names)
    out = self.out_projection(inputs_q.shape[-1], out)
    return out, value_residual


def batch_take(inputs, indices=None, axis=0):
  if indices is None:
    return inputs
  return jax.vmap(lambda a,b: jnp.take(a, b, axis=axis))(inputs, indices) # BTD -> BtND

# https://github.com/jax-ml/jax/issues/17844#issuecomment-2241236592
def batch_scatter_add(inputs, indices, updates):
  return jax.vmap(lambda a,b,c:  a.at[b,:].add(c))(inputs, indices, updates)
  # return jax.vmap(lambda a,b,c: jax.lax.scatter_add(a, b, c))(inputs, indices, updates)

def cal_mosa_mask(q_indices, k_indices, dtype=jnp.dtype("float32")):
    attn_mask = q_indices[...,None] >= k_indices[...,None,:]
    large_negative_number = accelerator.get_large_negative_number(dtype)
    attn_mask = attn_mask.astype(dtype)
    attn_mask = jnp.where((attn_mask > 0.5), attn_mask, large_negative_number)
    return attn_mask


class MoSA(Attention):
  """Mixture of Sparse Attention (MoSA) https://arxiv.org/abs/2505.00315 """

  mosa_num_query_heads: int = 0
  mosa_num_groups: Optional[int] = None
  mosa_num_kv_heads: int = 0
  mosa_topk: int = 256
  mosa_num_routers: int = 1 # 1: shared qk indexs, 2: seperate qk indexs 
  mosa_mode: str = 'topk' # 'relue'
 

  def setup(self):
    super().setup()

    self.mosa_head_sparse = self.config.mosa_head_sparse
    self.mosa_head_topk = self.config.mosa_head_topk

    if self.mosa_num_groups is None: 
      self.mosa_num_groups = self.mosa_num_query_heads
    # else:
    #   assert self.config.dc_num_groups == 1
    
    assert self.mosa_num_query_heads % self.mosa_num_groups == 0 
    assert self.mosa_num_kv_heads % self.mosa_num_groups == 0
    if self.config.dc_num_groups is not None:
      assert self.mosa_num_groups == self.config.dc_num_groups


    kernel_init_shard = nn.with_logical_partitioning(NormalInitializer(0.006), (None, None, 'embed', None))
    self.mosa_num_query_heads_per_group = self.mosa_num_query_heads // self.mosa_num_groups
    shape = (self.mosa_num_groups, self.mosa_num_query_heads_per_group, self.config.emb_dim, self.head_dim)
    self.query_weight = self.param('query_weight', kernel_init_shard, shape, self.weight_dtype) # DdN
    self.out_weight = self.param('out_weight', kernel_init_shard, shape, self.weight_dtype)    

    self.mosa_num_kv_heads_per_group = self.mosa_num_kv_heads // self.mosa_num_groups
    shape = (self.mosa_num_groups, self.mosa_num_kv_heads_per_group, self.config.emb_dim, self.head_dim)
    self.key_weight = self.param('key_weight', kernel_init_shard, shape, self.weight_dtype)
    self.value_weight = self.param('value_weight', kernel_init_shard, shape, self.weight_dtype)

    if self.config.use_head_emb:
      kernel_init_shard = nn.with_logical_partitioning(initializers.constant_init(0), (None, None, 'embed'))
      shape = (2, self.mosa_num_query_heads, self.config.emb_dim)
      self.head_emb = self.param('head_emb', kernel_init_shard, shape, self.weight_dtype) # CND

  def mosa_router(self, x: Array,):
    if self.config.mosa_router_hid_dim:
      x = DenseGeneral(
            (self.config.mosa_router_hid_dim),
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            quant=self.quant,
            kernel_init=self.kernel_init,
            kernel_axes=("embed", None),
            name="mosa_router_proj_up",
            matmul_precision=self.config.matmul_precision,
        )(x)
      x = jax.nn.gelu(x)
    num_heads = self.mosa_num_groups if not self.mosa_head_sparse else self.mosa_num_query_heads **2 
    logits = DenseGeneral(
            (num_heads, self.mosa_num_routers),
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            quant=self.quant,
            kernel_init=self.kernel_init,
            kernel_axes=(None, None, None) if self.config.mosa_router_hid_dim else ("embed", None, None),
            name="mosa_router",
            matmul_precision=self.config.matmul_precision,
        )(x) # BTD, DGC -> BTGC
    return logits 

  def select(self, logits: Array):
    C,B,N,T=logits.shape
    # select the first token and other topk-1 tokens
    topk = self.mosa_topk - 1
    gate_q, q_indices = jax.lax.top_k(logits[0,:,:,1:], topk) # BNT -> BNt
    gate_q = jnp.concatenate([logits[0,:,:,:1], gate_q], axis=-1).transpose((0,2,1)) # BNt->BtN
    q_indices = jnp.concatenate([jnp.zeros((B,N,1),dtype=q_indices.dtype),q_indices+1], axis=-1).transpose((0,2,1)) # BNt->BtN
    if self.mosa_num_routers == 1: 
      gate_k, k_indices = gate_q, q_indices
    else:
      gate_k, k_indices = jax.lax.top_k(logits[1,:,:,1:], topk) # BNT -> BNt
      gate_k = jnp.concatenate([logits[1,:,:,:1], gate_k], axis=-1).transpose((0,2,1)) # BNt->BtN
      k_indices = jnp.concatenate([jnp.zeros((B,N,1),dtype=k_indices.dtype),k_indices+1], axis=-1).transpose((0,2,1)) # BNt->BtN
    return gate_q, q_indices, gate_k, k_indices

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
      hidden_states=None,
      value_residual=None,
      ffn_act=None,
  ):

    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)
     
    logits = self.mosa_router(inputs_q)

    # max_logging.log(f"mosa_mode: {self.mosa_mode}")


    if self.mosa_mode == 'topk':
      logits = jax.nn.sigmoid(logits) 
      if self.mosa_head_sparse:
        head_scores = logits.mean(axis=(1,-1)) # BTGC->BG
        head_scores, head_indices = jax.lax.top_k(head_scores, self.mosa_head_topk) # BG -> BK
        qk_head_indices, ov_head_indices = head_indices // self.mosa_num_query_heads, head_indices % self.mosa_num_query_heads
        logits = batch_take(logits, indices=head_indices, axis=1) # BTGC->BTKC
      logits = rearrange(logits, 'B T G C -> C B G T')
      gate_q, q_indices, gate_k, k_indices = self.select(logits)
      if self.config.mosa_sqrt_gate:
        gate_q = jnp.sqrt(gate_q)
        gate_k = jnp.sqrt(gate_k)
    elif self.mosa_mode == 'relu':
      logits = rearrange(logits, 'B T G C -> C B T G')
      if self.config.mosa_num_routers == 2:
        gate = jax.nn.relu(jax.lax.tanh(logits))
        if self.config.mosa_sqrt_gate:
          gate = jnp.sqrt(gate)
        gate_q, gate_k = gate[0], gate[1]
      else: 
        gate_q = jax.nn.relu(jax.lax.tanh(logits))[0] # BTN
        gate_k = None
      self.sow('intermediates', 'relu_gate_q_ratio', (gate_q>0).astype(gate_q.dtype).mean())
      self.sow('intermediates', 'relu_gate_k_ratio', 1 if gate_k is None else (gate_k>0).astype(gate_k.dtype).mean())
      q_indices, k_indices = None, None
    elif self.mosa_mode == 'full_gated':
      logits = rearrange(logits, 'B T G C -> C B T G')
      if self.config.mosa_num_routers == 2:
        gate = jax.nn.sigmoid(logits)
        if self.config.mosa_sqrt_gate:
          gate = jnp.sqrt(gate)
        gate_q, gate_k = gate[0], gate[1]
      else:
        gate_q = jax.nn.sigmoid(logits)[0] 
        gate_k = None
      q_indices, k_indices = None, None
  
    # max_logging.log(f"q_indices, k_indices shape: {q_indices.shape}, {k_indices.shape}")

    _inputs_q, _inputs_kv  = None, None
    if not self.mosa_head_sparse:
      _inputs_kv = inputs_kv[0] if isinstance(inputs_kv, (tuple, list)) else inputs_kv
      _inputs_q = batch_take(inputs_q, indices=q_indices) # BTD, BtN -> BtND
      _inputs_kv = batch_take(_inputs_kv, indices=k_indices) # BTD, BtN -> BtN

    # max_logging.log(f"_inputs_q, _inputs_kv shape: {_inputs_q.shape}, {_inputs_kv.shape}")
    if q_indices is not None:
      if self.mosa_head_sparse: # BTD-> BTKND
        select_matmul = lambda input, w, t_idx, k_idx: jnp.einsum('tKD,KNDd->tKNd', input[t_idx,:], w[k_idx,:])
        query = jax.vmap(select_matmul, in_axes=(0,None,0,0))(inputs_q, self.query_weight, q_indices, qk_head_indices)
        key = jax.vmap(select_matmul, in_axes=(0,None,0,0))(inputs_q, self.key_weight, k_indices, qk_head_indices)
        value = jax.vmap(select_matmul, in_axes=(0,None,0,0))(inputs_q, self.value_weight, k_indices, ov_head_indices)
      else:
        query = jnp.einsum('BtGD,GNDd->BtGNd', _inputs_q, self.query_weight)
        key = jnp.einsum('BtGD,GNDd->BtGNd', _inputs_kv, self.key_weight)
        value = jnp.einsum('BtGD,GNDd->BtGNd', _inputs_kv, self.value_weight)
    else:
      query = jnp.einsum('BtD,GNDd->BtGNd', _inputs_q, self.query_weight)
      key = jnp.einsum('BtD,GNDd->BtGNd', _inputs_kv, self.key_weight)
      value = jnp.einsum('BtD,GNDd->BtGNd', _inputs_kv, self.value_weight)

    # if self.mosa_head_sparse:
    #   query = batch_take(query, indices=qk_head_indices, axis=1) # BtGNd [BK] -> BtKNd
    #   key = batch_take(key, indices=qk_head_indices, axis=1) # BtGNd [BK] -> BtKNd
    #   value = batch_take(value, indices=ov_head_indices, axis=1) # BtGNd [BK] -> BtKNd

    query = rearrange(query, 'B t G N d -> B t (G N) d')
    key = rearrange(key, 'B t G N d -> B t (G N) d')
    value = rearrange(value, 'B t G N d -> B t (G N) d')

    value_residual = None

    # query, key = dc.QKNorm(self.config, name='qk_norm')(query, key) # lsp

    # apply ROPE
    q_inputs_positions = inputs_positions if q_indices is None else batch_take(inputs_positions, indices=q_indices)[...,None] 
    k_inputs_positions = inputs_positions if k_indices is None else batch_take(inputs_positions, indices=k_indices)[...,None] 
    if self.mosa_num_groups > 1:
      q_inputs_positions = q_inputs_positions.repeat(self.mosa_num_query_heads_per_group, axis=-2)
      k_inputs_positions = k_inputs_positions.repeat(self.mosa_num_kv_heads_per_group, axis=-2)
    query = self.apply_rotary_embedding(query, q_inputs_positions, name="query_rotary") # BNts
    key = self.apply_rotary_embedding(key, k_inputs_positions, name="key_rotary")

    if self.config.mosa_num_routers == 2:
      value = gate_k.repeat(self.mosa_num_kv_heads_per_group, axis=-1)[...,None] * value # (BtG -> Bt(GN))1, Bt(GN)d->Bt(GN)d

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

    assert not self.config.quantize_kvcache or self.kv_quant

     # lsp
    depth_scaling = jnp.sqrt(self.head_dim).astype(self.dtype)
    query /= depth_scaling

    # if self.num_query_heads > self.num_kv_heads: # GQA
    #   n_expands = self.num_query_heads // self.num_kv_heads
    #   key = jnp.repeat(key, n_expands, axis=-2) # BSNd
    #   value = jnp.repeat(value, n_expands, axis=-2) 

    # if self.config.record_internal_nn_metrics:
    #   if self.config.sigmoid_attention:
    #     self.sow('intermediates', 'q_norm_stat', maxtext_utils.l2norm(query))
    #     self.sow('intermediates', 'k_norm_stat', maxtext_utils.l2norm(key))
    #     attn_logits = jnp.tril(jnp.einsum('B T N D, B S N D -> B N T S', query, key) - 7.62)
    #     self.sow('intermediates', 'attn_logits_max', attn_logits.max())
    #     self.sow('intermediates', 'attn_logits_min', -1 * (-attn_logits).max())
    #     self.sow('intermediates', 'attn_logits_mean', attn_logits.mean())

    if q_indices is not None:
      attn_mask = cal_mosa_mask(q_indices.transpose(0,2,1), k_indices.transpose(0,2,1))
      attn_mask = attn_mask.repeat(self.mosa_num_query_heads_per_group, axis=1) # BNt, BNs-> BNts -> B(GN)ts
    else:
      attn_mask = None
    # attn_mask = q_indices.transpose(0,2,1)[...,None] >= k_indices.transpose(0,2,1)[...,None,:] # BNt1 >= BN1s -> BNts

    max_logging.log(f"query, key, value shape: {query.shape}, {key.shape}, {value.shape}")
    out = self.attention_op(query, key, value, decoder_segment_ids, model_mode, _inputs_q, _inputs_kv, hidden_states=hidden_states, attn_mask=attn_mask)

    out = nn.with_logical_constraint(out, self.out_axis_names)
  
    # apply output projection,  output dim is set to the input dim.
    out = gate_q.repeat(self.mosa_num_query_heads_per_group, axis=-1)[...,None] * out # (BtG -> Bt(GN))1, BtNd -> BtNd
    # if self.mosa_head_sparse:
    #   out = out * head_scores[:,None,:,None] # BtND, B1K1

    out = rearrange(out, 'B t (G N) d -> B t G N d', N=self.mosa_num_query_heads_per_group)
    if q_indices is not None:
      if self.mosa_head_sparse: # BtKNd, BKNDd -> BtKD
        sparse_out = jax.vmap(lambda o, w, idx:jnp.einsum('tKNd,KNDd->tKD', o, w[idx,:]), in_axes=(0,None,0))(out, self.out_weight, ov_head_indices)
      else:
        sparse_out = jnp.einsum('BtGNd,GNDd->BtGD', out, self.out_weight)
    else:
      sparse_out = jnp.einsum('BtGNd,GNDd->BtD', out, self.out_weight)

    if self.config.use_head_emb and gate_k is not None and gate_q is not None: # add head emb for activated tokens  
      if q_indices is not None:
        sparse_out = sparse_out + jnp.einsum('BTN,ND->BTND', gate_q, self.head_emb[0]) + jnp.einsum('BTN,ND->BTND', gate_k, self.head_emb[1])
      else:
        sparse_out = sparse_out + jnp.einsum('BTN,ND->BTD', gate_q, self.head_emb[0]) + jnp.einsum('BTN,ND->BTD', gate_k, self.head_emb[1])

    if q_indices is not None:
      B,t,N,D= sparse_out.shape
      out = jnp.zeros_like(inputs_q)
      out = batch_scatter_add(out, q_indices.reshape(B,-1), sparse_out.reshape(B,-1,D))
    else:
      out = sparse_out
   
    out = checkpoint_name(out, "out_proj")
    return out, value_residual
