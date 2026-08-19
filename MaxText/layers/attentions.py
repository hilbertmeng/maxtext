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
from einops import rearrange

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
      

  def query_projection(self, inputs_q: Array) -> Array:
    """Query projection."""

    if self.config.opt_type == 'muon':
      kernel_axes=("embed", "mlp")
      features = (self.num_query_heads * self.head_dim, )
    else:
      kernel_axes=("embed", "q_heads", "kv")
      features = (self.num_query_heads, self.head_dim)

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
      output = output.reshape(b, t, self.num_query_heads, self.head_dim)
    return output

  def kv_projection(self, inputs_kv: Array, proj_name: str) -> Array:
    """Projection for Key and Value.

    Args:
      inputs_kv: inputs_kv: key/values of shape `[batch, kv_length,
        num_kv_heads, kv_dim]`.
      proj_name: name of projection, `key` or `value`.

    Returns:
      Projection of key or value, in shape of `[batch, kv_length, head_dim]`.
    """
    b, t, h = inputs_kv.shape

    if self.num_kv_heads == -1:
      raise ValueError("num_kv_heads is not defined.")

    if self.num_query_heads % self.num_kv_heads != 0:
      raise ValueError("Invalid num_kv_heads for GQA.")
    
    num_kv_heads = self.num_kv_heads
    if self.config.opt_type == 'muon':
      kernel_axes = ("embed", "mlp")
      features=(num_kv_heads * self.head_dim, )
    else:
      kernel_axes = ("embed", "kv_heads", "kv_head_dim")
      features=(num_kv_heads, self.head_dim) 
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
      kv_proj = kv_proj.reshape(b, t, num_kv_heads, self.head_dim)
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
    if self.config.opt_type == 'muon':
      kernel_shape = (self.head_dim * self.num_query_heads, output_dim)
      kernel_in_axis = (0, )
      kernel_out_axis = (1, )
      kernel = self.param(
          "out",
          nn.with_logical_partitioning(self.kernel_init, ("mlp", "embed")),
          kernel_shape,
          self.weight_dtype,
          kernel_in_axis,
          kernel_out_axis,
      )
      kernel = kernel.reshape(self.num_query_heads, self.head_dim, output_dim)
      kernel = jnp.asarray(kernel, self.dtype)
      out_proj = jnp.einsum("btnh,nhd->btd", out, kernel)
    else:
      out_proj = DenseGeneral(
          features=output_dim,
          axis=(-2, -1),
          kernel_init=self.kernel_init,
          kernel_axes=("heads", "kv", "embed"),
          dtype=self.dtype,
          weight_dtype=self.weight_dtype,
          name="out",
          quant=self.quant,
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

    # apply ROPE
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
# v0.1 scope: non-scan path, train mode, codebook + local + full read, n == n_kv (no GQA)
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
        x, dtype=self.dtype, epsilon=self.epsilon)
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


def _shared_bam_fetch_alpha(alpha, query, key, attn_mask, n_f, mode,
                            diagonal_yield, attn_logits_soft_cap, float32_logits):
  """Reuse standard Q/K for BAM fetch, with selectable equivalent implementations."""
  if mode == 'legacy':
    fetch_alpha = alpha * (
        1 - jnp.eye(alpha.shape[-2], alpha.shape[-1], dtype=alpha.dtype)
    ) if diagonal_yield else alpha
    return fetch_alpha[:, :n_f]

  if mode == 'compact':
    fetch_alpha = alpha[:, :n_f]
  elif mode == 'recompute':
    fetch_logits = jnp.einsum('btfd,bsfd->bfts', query[:, :, :n_f], key[:, :, :n_f])
    if attn_logits_soft_cap:
      fetch_logits = jnp.tanh(fetch_logits / attn_logits_soft_cap) * attn_logits_soft_cap
    if attn_mask is not None:
      fetch_logits = apply_mask_to_logits(fetch_logits, attn_mask)
    if float32_logits:
      fetch_logits = fetch_logits.astype(jnp.float32)
    fetch_alpha = jax.nn.softmax(fetch_logits, axis=-1)
  else:
    raise ValueError(f'Unknown shared BAM fetch mode: {mode}')

  if diagonal_yield:
    fetch_alpha = fetch_alpha * (
        1 - jnp.eye(fetch_alpha.shape[-2], fetch_alpha.shape[-1], dtype=fetch_alpha.dtype))
  return fetch_alpha


def _dynamic_mixed_bam_fetch_alpha(
    alpha, mix_logits, diagonal_yield, weight_mode='softmax',
    *, rms_epsilon, return_aux=False):
  """Build one BAM fetch route as a token-wise mixture of all MHA heads."""
  mix_logits, mix_weights = _dynamic_bam_fetch_mix_weights(
      mix_logits, alpha.dtype, weight_mode, rms_epsilon=rms_epsilon)

  # A restricted router consumes the first N standard attention maps.  Head
  # identity is learned jointly; "first" is only a deterministic subset.
  fetch_alpha = jnp.einsum(
      'bnts,btn->bts', alpha[:, :mix_weights.shape[-1]], mix_weights)
  pre_diagonal_alpha = fetch_alpha[:, None]
  if diagonal_yield:
    fetch_alpha = fetch_alpha * (
        1 - jnp.eye(fetch_alpha.shape[-2], fetch_alpha.shape[-1], dtype=fetch_alpha.dtype))
  fetch_alpha = fetch_alpha[:, None]  # bts->bfts
  if return_aux:
    return fetch_alpha, mix_logits, mix_weights, pre_diagonal_alpha
  return fetch_alpha


def _dynamic_bam_fetch_mix_weights(
    mix_logits, alpha_dtype, weight_mode='softmax', *, rms_epsilon):
  """Normalize token-wise MHA-head coefficients without materializing alpha."""
  mix_logits = jnp.asarray(mix_logits, jnp.float32)
  if weight_mode == 'softmax':
    mix_weights = jax.nn.softmax(mix_logits, axis=-1)
  elif weight_mode == 'rms':  # V1 default
    # RMSNorm alone has L2 norm sqrt(num_heads); divide it out so the signed
    # coefficient vector has unit L2 norm and does not gain amplitude for free.
    normalized = normalizations.rms_norm(
        mix_logits, dtype=alpha_dtype, epsilon=rms_epsilon)
    mix_weights = normalized / jnp.sqrt(mix_logits.shape[-1])
  else:
    raise ValueError(f'Unknown dynamic fetch weight mode: {weight_mode}')
  return mix_logits, jnp.asarray(mix_weights, alpha_dtype)


def _attention_op(
    query, key, value, valid, *, attn_logits_soft_cap=0.0,
    float32_logits=False):
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
  with jax.named_scope("attention/softmax"):
    alpha = jax.nn.softmax(logits, axis=-1)
  with jax.named_scope("attention/av"):
    y_std = jnp.einsum('bnqs,bsnd->bqnd', alpha, value)
  return y_std, alpha


def _bam_fetch_op(
    alpha, fetch_state, *, mix_weights=None, diagonal_mask=None,
    diagonal_indices=None):
  """Optionally mix MHA routes, set the local coefficient, and fetch M."""
  if diagonal_mask is not None and diagonal_indices is not None:
    raise ValueError('Specify either diagonal_mask or diagonal_indices, not both')
  with jax.named_scope("bam/mix_alpha"):
    fetch_alpha = (
        jnp.einsum(
            'bnqs,bqn->bqs', alpha[:, :mix_weights.shape[-1]], mix_weights)
        if mix_weights is not None else alpha)
    fetch_alpha_pre_diagonal = fetch_alpha
    if diagonal_mask is not None:
      fetch_alpha = jnp.where(
          diagonal_mask[None], jnp.asarray(1, fetch_alpha.dtype), fetch_alpha)
    elif diagonal_indices is not None:
      query_indices, source_indices = diagonal_indices
      fetch_alpha = fetch_alpha.at[:, query_indices, source_indices].set(
          jnp.asarray(1, fetch_alpha.dtype))
  with jax.named_scope("bam/fetch_m"):
    if fetch_alpha.ndim == 4:
      Mbar = jnp.einsum('bfqs,bskv->bfqkv', fetch_alpha, fetch_state)
    else:
      Mbar = jnp.einsum('bqs,bskv->bqkv', fetch_alpha, fetch_state)
  return Mbar, fetch_alpha, fetch_alpha_pre_diagonal


def _sliding_window_bam_fetch_alpha(
    alpha, window_size, prefix_size=None, source_positions=None):
  """Keep a recent window and optional segment-local prefix, without renormalizing."""
  if window_size <= 0:
    raise ValueError(f'BAM fetch sliding-window size must be positive, got {window_size}')
  target = jnp.arange(alpha.shape[-2])[:, None]
  source = jnp.arange(alpha.shape[-1])[None, :]
  mask = (source <= target) & (source > target - window_size)
  if prefix_size is not None:
    if prefix_size <= 0:
      raise ValueError(f'BAM fetch prefix size must be positive, got {prefix_size}')
    if source_positions is None:
      raise ValueError('source_positions are required when retaining a BAM fetch prefix')
    mask = mask[None, None] | (source_positions[:, None, None, :] < prefix_size)
  return jnp.where(mask, alpha, jnp.asarray(0, alpha.dtype))


def _temporal_block_bam_fetch(
    alpha, M, positions, segment_ids, block_size, mode, recent_window_size=None):
  """Approximate old full-fetch states with causal, segment-aware block summaries.

  Completed blocks store either their mean matrix or an orthogonal constant/linear
  least-squares pair.  The current block remains exact.  With ``recent_window_size``,
  only blocks ending before the exact recent window are compressed; the boundary
  block also remains exact, avoiding future leakage and partial-block ambiguity.
  """
  if alpha.ndim != 4 or M.ndim != 4:
    raise ValueError(f'temporal block fetch expects [b,f,t,s] and [b,s,k,v], got {alpha.shape}, {M.shape}')
  if alpha.shape[0] != M.shape[0] or alpha.shape[-1] != M.shape[1]:
    raise ValueError(f'incompatible temporal block fetch shapes: {alpha.shape}, {M.shape}')
  if alpha.shape[-2] != M.shape[1] or positions.shape != segment_ids.shape:
    raise ValueError('temporal block fetch currently requires self-attention with aligned positions/segments')
  if block_size <= 1:
    raise ValueError(f'temporal block size must exceed one, got {block_size}')
  if mode not in ('mean', 'linear'):
    raise ValueError(f'unknown temporal block mode: {mode}')
  if recent_window_size is not None and recent_window_size <= 0:
    raise ValueError(f'recent window must be positive, got {recent_window_size}')

  batch, _, target_length, source_length = alpha.shape
  del batch
  source_index = jnp.arange(source_length)[None, :]
  previous_segment = jnp.concatenate(
      (segment_ids[:, :1], segment_ids[:, :-1]), axis=1)
  is_segment_start = (source_index == 0) | (segment_ids != previous_segment) | (positions == 0)
  segment_start = jax.lax.associative_scan(
      jnp.maximum, jnp.where(is_segment_start, source_index, 0), axis=1)
  group_ids = segment_start + positions // block_size
  valid_source = segment_ids != 0

  def segment_sum(data):
    return jax.vmap(
        lambda values, ids: jax.ops.segment_sum(
            values, ids, num_segments=source_length)
    )(data, group_ids)

  # Accumulate summaries in f32, then store/read them in the matrix-stream dtype.
  M_f32 = jnp.asarray(M, jnp.float32)
  valid_f32 = valid_source.astype(jnp.float32)
  counts = segment_sum(valid_f32)[..., None, None]
  mean_matrix = segment_sum(M_f32 * valid_f32[..., None, None]) / jnp.maximum(counts, 1.0)
  mean_matrix = jnp.asarray(mean_matrix, M.dtype)

  within_block = positions % block_size
  linear_coordinate = (
      2.0 * within_block.astype(jnp.float32) / float(block_size - 1) - 1.0)
  if mode == 'linear':
    linear_energy = segment_sum(
        jnp.square(linear_coordinate) * valid_f32)[..., None, None]
    linear_matrix = segment_sum(
        M_f32 * (linear_coordinate * valid_f32)[..., None, None]
    ) / jnp.maximum(linear_energy, 1.0)
    linear_matrix = jnp.asarray(linear_matrix, M.dtype)
  else:
    linear_matrix = None

  query_group = group_ids[:, :target_length]
  same_segment = (
      segment_ids[:, :target_length, None] == segment_ids[:, None, :])
  same_segment &= segment_ids[:, :target_length, None] != 0
  if recent_window_size is None:
    compress = same_segment & (
        query_group[:, :, None] != group_ids[:, None, :])
  else:
    source_block_end = (positions // block_size + 1) * block_size - 1
    cutoff = positions[:, :target_length, None] - recent_window_size
    compress = same_segment & (source_block_end[:, None, :] <= cutoff)

  compressed_alpha = alpha * compress[:, None, :, :].astype(alpha.dtype)
  exact_alpha = alpha - compressed_alpha
  grouped_alpha = segment_sum(
      jnp.transpose(compressed_alpha, (0, 3, 1, 2)))
  grouped_alpha = jnp.transpose(grouped_alpha, (0, 2, 3, 1))
  compressed_fetch = jnp.einsum('bftg,bgkv->bftkv', grouped_alpha, mean_matrix)
  if mode == 'linear':
    grouped_linear_alpha = segment_sum(jnp.transpose(
        compressed_alpha * linear_coordinate[:, None, None, :].astype(alpha.dtype),
        (0, 3, 1, 2)))
    grouped_linear_alpha = jnp.transpose(grouped_linear_alpha, (0, 2, 3, 1))
    compressed_fetch = compressed_fetch + jnp.einsum(
        'bftg,bgkv->bftkv', grouped_linear_alpha, linear_matrix)
  exact_fetch = jnp.einsum('bfts,bskv->bftkv', exact_alpha, M)
  return exact_fetch + compressed_fetch


def _select_bam_write_source(source, y_std, y_codebook, y_full, y_local_o, y_all=None):
  """Select direct U-write content without changing the residual-stream output."""
  if source == 'std+cross+local_o':  # V1 default
    return y_all if y_all is not None else y_std + y_codebook + y_full + y_local_o
  if source == 'std+cross':
    return y_std + y_codebook + y_full
  if source == 'std':
    return y_std
  raise ValueError(f'Unknown BAM write source: {source}')


def _mix_bam_write_v(x_v, o_head, bam_k, mix_scale, bias):
  """Mix local and attention-output V factors with a per-head affine selector."""
  o_v = o_head[..., bam_k:bam_k + x_v.shape[-1]]
  if o_v.shape != x_v.shape:
    raise ValueError(f'BAM write-V shapes differ: x={x_v.shape}, o={o_v.shape}')
  x_scale = mix_scale[:, 0][None, None, :, None]
  o_scale = mix_scale[:, 1][None, None, :, None]
  return x_scale * x_v + o_scale * o_v + bias


def _update_bam_matrix(M_in, dM, lambda_decay, forget_logits=None):
  """Apply optional token-wise forgetting to old state, then add the new write."""
  retention = jnp.asarray(lambda_decay, dtype=M_in.dtype)
  forget_gate = None
  if forget_logits is not None:
    forget_gate = jax.nn.sigmoid(forget_logits)
    retention = retention * (1.0 - forget_gate[..., None])
  return retention * M_in + dM, forget_gate


def _transform_bam_read_key(
    r, mode='none', scale=1.0, *, rms_epsilon,
    rms_statistics_dtype=jnp.float32, gate_logits=None,
    learned_rms_norm=None, use_learned_rms=False):
  """Apply a side-local health transform to a runtime BAM read key.

  `soft_rms_cap` is identity to first order at zero and caps the key RMS at `scale`.
  `rms_gate` factors the key into an RMS-normalized direction and a bounded, learned
  amplitude `scale * sigmoid(gate_logits)`.  Callers normalize row and column keys
  separately so one side cannot hide the other side's scale.
  """
  if mode == 'none':
    return r
  scale = jnp.asarray(scale, dtype=r.dtype)
  if mode == 'soft_rms_cap':
    return r * scale * jax.lax.rsqrt(jnp.mean(r ** 2, axis=-1, keepdims=True) + scale ** 2)
  if mode == 'rms_gate':  # V1 default
    if gate_logits is None:
      raise ValueError('rms_gate requires gate logits')
    direction = normalizations.rms_norm(
        r, dtype=r.dtype, epsilon=rms_epsilon,
        statistics_dtype=rms_statistics_dtype)
    if learned_rms_norm is not None:
      learned_direction = learned_rms_norm(r)
      if use_learned_rms:
        direction = learned_direction
    return scale * jax.nn.sigmoid(gate_logits) * direction
  raise ValueError(f'Unknown BAM read-key transform: {mode}')


def _project_bam_read_keys(
    row_width, x, W_R, *, rms_epsilon,
    rms_statistics_dtype=jnp.float32, key_mode='none', key_scale=1.0,
    key_gate_logits=None, key_row_norm=None, key_col_norm=None,
    use_learned_key_norm=False):
  """Project and independently transform the row/column runtime read keys."""
  with jax.named_scope("bam/read_key_projection"):
    projected_key = W_R(x) if callable(W_R) else jnp.broadcast_to(
        W_R, x.shape[:-1] + W_R.shape)
    raw_row, raw_col = jnp.split(projected_key, [row_width], axis=-1)
  with jax.named_scope("bam/read_key_transform"):
    if key_gate_logits is None:
      row_gate = col_gate = None
    else:
      row_gate, col_gate = jnp.split(key_gate_logits, 2, axis=-1)
    r_row = _transform_bam_read_key(
        raw_row, key_mode, key_scale, rms_epsilon=rms_epsilon,
        rms_statistics_dtype=rms_statistics_dtype,
        gate_logits=row_gate, learned_rms_norm=key_row_norm,
        use_learned_rms=use_learned_key_norm)
    r_col = _transform_bam_read_key(
        raw_col, key_mode, key_scale, rms_epsilon=rms_epsilon,
        rms_statistics_dtype=rms_statistics_dtype,
        gate_logits=col_gate, learned_rms_norm=key_col_norm,
        use_learned_rms=use_learned_key_norm)
  return raw_row, raw_col, r_row, r_col


def _contract_bam_read_sides(
    Mc, Mr, r_row, r_col, per_head, implementation, read_side='both'):
  """Contract the selected side(s) of M and return the two output halves."""
  if read_side not in ('both', 'row', 'col'):
    raise ValueError(f'Unknown BAM read side: {read_side}')
  f = 'f' if Mc.ndim == 5 else ''
  n = 'n' if per_head else ''
  if implementation == 'dot_bnt':
    y_u = (jnp.einsum(f'b{f}tkv,bt{n}{f}v->b{n}tk', Mc, r_col)
           if read_side in ('both', 'col') else None)
    y_v = (jnp.einsum(f'b{f}tkv,bt{n}{f}k->b{n}tv', Mr, r_row)
           if read_side in ('both', 'row') else None)
    if y_u is None:
      y_u = jnp.zeros(y_v.shape[:-1] + (Mc.shape[-2],), dtype=y_v.dtype)
    if y_v is None:
      y_v = jnp.zeros(y_u.shape[:-1] + (Mr.shape[-1],), dtype=y_u.dtype)
    return y_u, y_v
  if implementation == 'dot_btn':
    y_u = y_v = None
    if read_side in ('both', 'col'):
      with jax.named_scope("bam/contract_1a_col"):
        y_u = jnp.einsum(f'b{f}tkv,bt{n}{f}v->bt{n}k', Mc, r_col)
    if read_side in ('both', 'row'):
      with jax.named_scope("bam/contract_1b_row"):
        y_v = jnp.einsum(f'b{f}tkv,bt{n}{f}k->bt{n}v', Mr, r_row)
    if y_u is None:
      y_u = jnp.zeros(y_v.shape[:-1] + (Mc.shape[-2],), dtype=y_v.dtype)
    if y_v is None:
      y_v = jnp.zeros(y_u.shape[:-1] + (Mr.shape[-1],), dtype=y_u.dtype)
    return y_u, y_v
  if implementation != 'mul_reduce_btn':
    raise ValueError(f'Unknown BAM read implementation: {implementation}')
  # V1 default

  # Spell the two contractions as broadcast multiply+reduce; on TPU this lowering
  # is measurably faster than dot_general for the current BAM read shapes.
  if not per_head:
    r_row = r_row[:, :, None]
    r_col = r_col[:, :, None]
  if Mc.ndim == 4:
    y_u = (jnp.sum(Mc[:, :, None] * r_col[..., None, :], axis=-1)
           if read_side in ('both', 'col') else None)
    y_v = (jnp.sum(Mr[:, :, None] * r_row[..., :, None], axis=-2)
           if read_side in ('both', 'row') else None)
  else:
    mc = jnp.transpose(Mc, (0, 2, 1, 3, 4))  # [b,t,f,k,v]
    mr = jnp.transpose(Mr, (0, 2, 1, 3, 4))
    y_u = (jnp.sum(mc[:, :, None] * r_col[..., None, :], axis=(-3, -1))
           if read_side in ('both', 'col') else None)
    y_v = (jnp.sum(mr[:, :, None] * r_row[..., :, None], axis=(-3, -2))
           if read_side in ('both', 'row') else None)
  if y_u is None:
    y_u = jnp.zeros(y_v.shape[:-1] + (Mc.shape[-2],), dtype=y_v.dtype)
  if y_v is None:
    y_v = jnp.zeros(y_u.shape[:-1] + (Mr.shape[-1],), dtype=y_u.dtype)
  if not per_head:
    y_u = jnp.squeeze(y_u, axis=-2)
    y_v = jnp.squeeze(y_v, axis=-2)
  return y_u, y_v


def _contract_bam_read(
    Mc, Mr, r_row, r_col, per_head, implementation, read_side='both'):
  """Contract the selected side(s) of M; omitted output halves are zero."""
  y_u, y_v = _contract_bam_read_sides(
      Mc, Mr, r_row, r_col, per_head, implementation, read_side)
  return jnp.concatenate([y_u, y_v], axis=-1)


def _fit_bam_read_to_head(read, bam_k, head_dim, v_adapter=None):
  """Map a bilateral [K-side, V-side] BAM read into one attention head."""
  if not 0 < bam_k < head_dim:
    raise ValueError(f'bam_k={bam_k} must be smaller than head_dim={head_dim}')
  y_k, y_v = jnp.split(read, [bam_k], axis=-1)
  target_v_dim = head_dim - bam_k
  if y_v.shape[-1] > target_v_dim:
    if v_adapter is None:
      raise ValueError(
          f'BAM V-side width {y_v.shape[-1]} exceeds available head tail '
          f'{target_v_dim} without an adapter')
    y_v = jnp.einsum(
        '...nv,nvd->...nd', y_v, v_adapter.astype(y_v.dtype))
  elif y_v.shape[-1] < target_v_dim:
    y_v = jnp.pad(
        y_v, [(0, 0)] * (y_v.ndim - 1)
        + [(0, target_v_dim - y_v.shape[-1])])
  return jnp.concatenate((y_k, y_v), axis=-1)


def _scale_bam_read_side_gradients(read, bam_k, col_scale, row_scale):
  """Preserve the forward read while scaling each side's backward path.

  The leading K-side output is M @ r_col (column/address lookup); the
  remaining output is M.T @ r_row (row/data lookup).  This hook is used only
  by the standalone attribution diagnostic.
  """
  y_col, y_row = jnp.split(read, [bam_k], axis=-1)

  def scale_gradient(value, scale):
    stopped = jax.lax.stop_gradient(value)
    return value + (jnp.asarray(scale, value.dtype) - 1) * (value - stopped)

  return jnp.concatenate(
      (scale_gradient(y_col, col_scale), scale_gradient(y_row, row_scale)),
      axis=-1)


def _bam_readout_query_indices(length, count=16):
  """Evenly sample query endpoints for the isolated readout diagnostic."""
  if length % count:
    raise ValueError(f'readout diagnostic needs length divisible by {count}, got {length}')
  return (jnp.arange(count) + 1) * (length // count) - 1


def bam_read(M, x, W_R, R=None, *, key_mode='none', key_scale=1.0,
             rms_epsilon,
             rms_statistics_dtype=jnp.float32,
             key_gate_logits=None, return_key_stages=False,
             key_row_norm=None, key_col_norm=None,
             use_learned_key_norm=False, implementation='dot_bnt',
             read_side='both', return_sides=False):
  """Unified read primitive (§4.6.1) — every read mode shares this.

  Projection -> split row/col -> bilateral contraction -> merge (§4.3 type alignment).
  M: single tensor [b,{f},t,k,v] (U⊗V same) OR (M_col, M_row) tuple (codebook read uses
     (Zcᵀ, Zr), each side its own matrix state). The f axis appears iff M is 5-D
     (cross-token fetch); it is summed into the contraction (Σ_f).
  W_R: DenseGeneral x -> {n}{f}(A_r+A_c), or a static key parameter with those trailing
     axes. Static keys are broadcast over x's leading [b,t] axes. MaxText emits
     [b,t,(n),(f,),kv] (head after t), consumed natively by the einsum (key subscript
     bt{n}{f}{side}). Split widths adapt to M's k/v (or C/C for codebook).
  R: [n,d,d] rematrix given => shared tier (key has no head axis, read once then per-head
     rematrix); R is None => per-head tier (key carries head axis, no rematrix).
  `dot_bnt` preserves the historical [b,n,t,d] result. `dot_btn` and
  `mul_reduce_btn` return [b,t,n,d], eliminating the callers' immediate transpose.
  `return_sides=True` returns the row/column contractions separately before concatenation.
  """
  Mc, Mr = M if isinstance(M, tuple) else (M, M)
  raw_row, raw_col, r_row, r_col = _project_bam_read_keys(
      Mr.shape[-2], x, W_R, key_mode=key_mode, key_scale=key_scale,
      rms_epsilon=rms_epsilon, rms_statistics_dtype=rms_statistics_dtype,
      key_gate_logits=key_gate_logits,
      key_row_norm=key_row_norm, key_col_norm=key_col_norm,
      use_learned_key_norm=use_learned_key_norm)
  with jax.named_scope("bam/read_m_contract"):
    y_u, y_v = _contract_bam_read_sides(
        Mc, Mr, r_row, r_col, R is None, implementation, read_side)
  if return_sides:
    if R is not None:
      raise ValueError('return_sides requires a direct per-head BAM read')
    y = (y_u, y_v)
  else:
    y = jnp.concatenate([y_u, y_v], axis=-1)
  if R is not None and not return_sides:
    with jax.named_scope("bam/read_rematrix"):
      y = (jnp.einsum('btd,nde->bnte', y, R) if implementation == 'dot_bnt'
           else jnp.einsum('btd,nde->btne', y, R))
  if return_key_stages:
    post_rms_row = (key_row_norm(raw_row) if key_row_norm is not None
                    else normalizations.rms_norm(
                        raw_row, dtype=raw_row.dtype,
                        epsilon=rms_epsilon,
                        statistics_dtype=rms_statistics_dtype))
    post_rms_col = (key_col_norm(raw_col) if key_col_norm is not None
                    else normalizations.rms_norm(
                        raw_col, dtype=raw_col.dtype,
                        epsilon=rms_epsilon,
                        statistics_dtype=rms_statistics_dtype))
    if not use_learned_key_norm:
      post_rms_row = normalizations.rms_norm(
          raw_row, dtype=raw_row.dtype, epsilon=rms_epsilon,
          statistics_dtype=rms_statistics_dtype)
      post_rms_col = normalizations.rms_norm(
          raw_col, dtype=raw_col.dtype, epsilon=rms_epsilon,
          statistics_dtype=rms_statistics_dtype)
    return y, {
        "pre_rms": jnp.concatenate((raw_row, raw_col), axis=-1),
        "post_rms_pre_gate": jnp.concatenate((post_rms_row, post_rms_col), axis=-1),
        "post_gate": jnp.concatenate((r_row, r_col), axis=-1),
    }
  return y


def factorized_head_bam_read(
    M, x, W_R, W_head_mix, *, rms_epsilon,
    rms_statistics_dtype=jnp.float32,
    key_mode='none', key_scale=1.0,
    key_gate_logits=None, key_row_norm=None,
    key_col_norm=None, use_learned_key_norm=False,
    implementation='dot_bnt', read_side='both', output_layout='bnt',
    return_aux=False):
  """Read once with a shared runtime key, then route it dynamically across heads.

  For each side, the effective per-head key is the rank-1 factorization
  ``key[b,t,n,:] = head_mix[b,t,n] * shared_key[b,t,:]``.  Bilinearity lets the
  implementation contract M with the shared key once and apply the signed head
  coefficients afterwards, without materializing per-head keys.

  M must be the local matrix state [b,t,k,v].  W_R maps x -> k+v and W_head_mix
  maps x -> [n,2], with independent row/column head coefficients.  Coefficients
  receive parameter-free RMS normalization over the head axis, so every head has
  O(1) typical scale; unlike fetch-alpha mixing, this expansion intentionally does
  not divide by sqrt(n).
  """
  if M.ndim != 4:
    raise ValueError(f'factorized local BAM read expects [b,t,k,v], got {M.shape}')
  if read_side not in ('both', 'row', 'col'):
    raise ValueError(f'Unknown BAM read side: {read_side}')
  if output_layout not in ('bnt', 'btn'):
    raise ValueError(f'Unknown factorized BAM output layout: {output_layout}')
  _, _, r_row, r_col = _project_bam_read_keys(
      M.shape[-2], x, W_R, key_mode=key_mode, key_scale=key_scale,
      rms_epsilon=rms_epsilon, rms_statistics_dtype=rms_statistics_dtype,
      key_gate_logits=key_gate_logits,
      key_row_norm=key_row_norm, key_col_norm=key_col_norm,
      use_learned_key_norm=use_learned_key_norm)

  with jax.named_scope("bam/read_m_contract"):
    y_u = y_v = None
    if implementation in ('dot_bnt', 'dot_btn'):
      if read_side in ('both', 'col'):
        y_u = jnp.einsum('btkv,btv->btk', M, r_col)
      if read_side in ('both', 'row'):
        y_v = jnp.einsum('btkv,btk->btv', M, r_row)
    elif implementation == 'mul_reduce_btn':  # V1 default
      if read_side in ('both', 'col'):
        y_u = jnp.sum(M * r_col[..., None, :], axis=-1)
      if read_side in ('both', 'row'):
        y_v = jnp.sum(M * r_row[..., :, None], axis=-2)
    else:
      raise ValueError(f'Unknown BAM read implementation: {implementation}')
  with jax.named_scope("bam/read_head_mix_projection"):
    raw_head_mix = W_head_mix(x)
  with jax.named_scope("bam/read_head_mix_transform"):
    if raw_head_mix.ndim != 4 or raw_head_mix.shape[-1] != 2:
      raise ValueError(
          f'factorized head mix expects [b,t,n,2], got {raw_head_mix.shape}')
    output_dtype = y_u.dtype if y_u is not None else y_v.dtype
    head_mix = normalizations.rms_norm(
        raw_head_mix, dtype=output_dtype, epsilon=rms_epsilon, axis=-2)
    row_mix, col_mix = head_mix[..., 0], head_mix[..., 1]
  pre_mix_read = jnp.concatenate([
      y_u if y_u is not None else jnp.zeros(
          y_v.shape[:-1] + (M.shape[-2],), dtype=y_v.dtype),
      y_v if y_v is not None else jnp.zeros(
          y_u.shape[:-1] + (M.shape[-1],), dtype=y_u.dtype),
  ], axis=-1)
  with jax.named_scope("bam/read_head_mix_expand"):
    if output_layout == 'bnt':
      y_u = (jnp.einsum('btk,btn->bntk', y_u, col_mix)
             if y_u is not None else None)
      y_v = (jnp.einsum('btv,btn->bntv', y_v, row_mix)
             if y_v is not None else None)
    else:  # V1 default
      y_u = (jnp.einsum('btk,btn->btnk', y_u, col_mix)
             if y_u is not None else None)
      y_v = (jnp.einsum('btv,btn->btnv', y_v, row_mix)
             if y_v is not None else None)
    if y_u is None:
      y_u = jnp.zeros(y_v.shape[:-1] + (M.shape[-2],), dtype=y_v.dtype)
    if y_v is None:
      y_v = jnp.zeros(y_u.shape[:-1] + (M.shape[-1],), dtype=y_u.dtype)
    output = jnp.concatenate([y_u, y_v], axis=-1)
    if return_aux:
      return output, {
          'post_gate_key': jnp.concatenate((r_row, r_col), axis=-1),
          'head_mix': head_mix,
          'pre_mix_read': pre_mix_read,
      }
    return output


def _packed_factorized_local_qk_init(kernel_init, num_heads, key_width):
  """Pack Q/K key, gate, and head-mix kernels while preserving their initializers."""
  mix_width = 2 * num_heads
  packed_width = 2 * (key_width + 2 + mix_width)

  def init_fn(key, shape, dtype, _in_axis=0, _out_axis=1):
    if len(shape) != 2 or shape[-1] != packed_width:
      raise ValueError(
          f'packed local-QK kernel expects [embed,{packed_width}], got {shape}')
    q_mix_key, k_mix_key = jax.random.split(key)
    zeros = lambda width: jnp.zeros((shape[0], width), dtype)
    mix_shape = (shape[0], num_heads, 2)
    q_mix = kernel_init(q_mix_key, mix_shape, dtype, 0, (1, 2)).reshape(
        shape[0], mix_width)
    k_mix = kernel_init(k_mix_key, mix_shape, dtype, 0, (1, 2)).reshape(
        shape[0], mix_width)
    return jnp.concatenate((
        zeros(key_width), zeros(2), q_mix,
        zeros(key_width), zeros(2), k_mix,
    ), axis=-1)

  return init_fn


def codebook_read(alpha_f, M, x, rho_u, rho_v, W_beta, *, key_mode='none',
                  key_scale=1.0, rms_epsilon,
                  rms_statistics_dtype=jnp.float32,
                  key_gate_logits=None,
                  key_row_norm=None, key_col_norm=None,
                  use_learned_key_norm=False, source_implementation='dot',
                  read_implementation='dot_btn'):
  """Codebook read (§4.6.2): keys constrained to span(ρ) cut the transfer width k*v -> C(k+v).

  Source side = ρ pre-contraction (static keys, no per-token key materialization — the savings).
  Destination side = a literal bam_read((Zcᵀ, Zr), x, W_beta) with β as the runtime key.
  rho_u: [C, k] (row-read codebook); rho_v: [C, v] (col-read codebook) — side-independent (§4.1).
  """
  if alpha_f.shape[1] != 1:
    raise ValueError(f'optimized codebook read requires one fetch route, got {alpha_f.shape}')
  with jax.named_scope('bam/codebook_source_reduce'):
    if source_implementation == 'dot':
      Yr = jnp.einsum('bskv,ck->bscv', M, rho_u)  # M^T rho_u
      Yc = jnp.einsum('bskv,cv->bsck', M, rho_v)  # M rho_v
    elif source_implementation == 'mul_reduce':
      expanded_M = M[:, :, None]
      Yr = jnp.sum(expanded_M * rho_u[None, None, :, :, None], axis=-2)
      Yc = jnp.sum(expanded_M * rho_v[None, None, :, None, :], axis=-1)
    else:
      raise ValueError(f'Unknown codebook source implementation: {source_implementation}')

  # Keep the true T x S contraction as a dense dot. Flattening C and content makes
  # its transfer width explicit and removes the length-one fetch axis.
  with jax.named_scope('bam/codebook_fetch'):
    b, s, c, d = (*Yc.shape[:3], Yc.shape[-1] + Yr.shape[-1])
    source = jnp.concatenate([Yc, Yr], axis=-1).reshape(b, s, c * d)
    Z = jnp.einsum('bts,bsd->btd', alpha_f[:, 0], source).reshape(
        b, alpha_f.shape[-2], c, d)

  with jax.named_scope('bam/codebook_read'):
    projection = lambda z: jnp.squeeze(W_beta(z), axis=-2)
    _, _, beta_row, beta_col = _project_bam_read_keys(
        c, x, projection, key_mode=key_mode, key_scale=key_scale,
        rms_epsilon=rms_epsilon, rms_statistics_dtype=rms_statistics_dtype,
        key_gate_logits=key_gate_logits,
        key_row_norm=key_row_norm, key_col_norm=key_col_norm,
        use_learned_key_norm=use_learned_key_norm)
    k = M.shape[-2]
    Zc, Zr = Z[..., :k], Z[..., k:]
    if read_implementation in ('dot_bnt', 'dot_btn'):
      y_u = jnp.einsum('btck,btnc->btnk', Zc, beta_col)
      y_v = jnp.einsum('btcv,btnc->btnv', Zr, beta_row)
    elif read_implementation == 'mul_reduce_btn':
      y_u = jnp.sum(Zc[:, :, None] * beta_col[..., None], axis=-2)
      y_v = jnp.sum(Zr[:, :, None] * beta_row[..., None], axis=-2)
    else:
      raise ValueError(f'Unknown BAM read implementation: {read_implementation}')
    return jnp.concatenate([y_u, y_v], axis=-1)


class BamAttention(Attention):
  """BAM Attention: standard MHA plus a matrix residual stream M, a write primitive, and a read primitive.

  v0.1: train mode, n==n_kv, non-scan. Per-layer read mode is set by layer_mode
  ('codebook' / 'local_qk' / 'local_v' / 'full' / 'local_o' / 'write' / 'none',
  combinable e.g. 'codebook+local_qk+local_o').  `write` keeps the M update but performs
  no BAM read.
  Asymmetric init: read side zero-initialized => start is bit-identical to standard MHA;
  write-gate bias=logit(eps) slightly open => M accumulates immediately. 'full' is the
  full-read oracle, for codebook-read correctness check only; most expensive transfer, not
  for production. 'local_qk' = route branch (Q/K injection, shared tier + R_q/R_k rematrix).
  'local_v' = source-summary branch (factorized local-M read injected before alpha-weighted V).
  'local_o' = content branch (O local read, per-head tier, merges into o). Common prefix
  'local_' marks both as local reads (alpha:=delta, read own M_in, zero transfer); the
  suffix names the use point (Q/K vs o) — the axis that actually distinguishes the two.
  Normalization (§4.6.5, param-free): read side — one RMS scalar per token over (k,v), all
  reads consume Mh; write side — per-record factor rms before the outer product (gate is the
  sole magnitude channel). Readout RMSNorm on y_bam is NOT adopted.
  """

  layer_mode: str = 'none'      # per-layer read mode (each layer is a separate instance under non-scan)
  read_side: str = 'both'       # both | row (M^T r_row) | col (M r_col)
  bam_k: int = 32
  bam_v: int = 32

  def setup(self):
    super().setup()             # reuse attention_op / projections / rope / out_projection
    cfg = self.config
    self._mha_control = bool(getattr(cfg, 'bam_mha_control', False))
    self._query_chunk_size = (
        cfg.query_chunk_size if self.attention_kernel == 'dot_product_chunk' else None)
    self._query_chunk_implementation = cfg.bam_query_chunk_implementation
    if self._mha_control:
      assert self.layer_mode == 'none', 'BAM MHA control must disable every BAM layer mode'
      assert not cfg.bam_diagnostics, 'BAM MHA control must not expose BAM diagnostics'
      if self._query_chunk_size is not None:
        assert self._query_chunk_size > 0
        assert cfg.max_target_length % self._query_chunk_size == 0, (
            'BAM MHA-control query chunk size must divide max_target_length')
      self._mode = set()
      self._has_write = False
      return

    assert 0 < self.bam_k < self.head_dim, (
        f'bam_k({self.bam_k}) must fit inside head_dim({self.head_dim})')
    self._mode = set(self.layer_mode.replace('+', ' ').split())
    self._has_write = self.layer_mode != 'none'
    assert self._mode <= {
        'codebook', 'local_qk', 'local_v', 'full', 'local_o', 'write', 'none'}
    assert self.read_side in ('both', 'row', 'col')

    # DenseGeneral passes in_axis/out_axis to its initializer in addition to
    # key/shape/dtype. Use MaxText's N-D constant wrapper so the zero-init read
    # kernels work both in DenseGeneral and in direct self.param calls below.
    zeros_init = initializers.contant_dense_init(0.0)
    orth_init = nn.initializers.orthogonal()
    reg_init = self.kernel_init

    self._read_key_mode = cfg.bam_read_key_mode
    self._read_key_scale = float(cfg.bam_read_key_scale)
    self._rms_epsilon = float(cfg.normalization_layer_epsilon)
    self._read_key_epsilon = float(
        cfg.bam_read_key_epsilon
        if cfg.bam_read_key_epsilon is not None
        else cfg.normalization_layer_epsilon)
    self._read_gate_init = (
        None if cfg.bam_read_gate_init is None else float(cfg.bam_read_gate_init))
    def resolve_rms_statistics_dtype(mode):
      assert mode in ('float32', 'activation')
      return jnp.float32 if mode == 'float32' else self.dtype

    self._read_rms_statistics_dtype = resolve_rms_statistics_dtype(
        cfg.bam_read_rms_statistics_dtype)
    self._write_rms_statistics_dtype = resolve_rms_statistics_dtype(
        cfg.bam_write_rms_statistics_dtype)
    self._create_grouped_rw_norm = bool(cfg.bam_create_grouped_rw_norm_params)
    self._use_grouped_rw_norm = bool(cfg.bam_use_grouped_rw_norm)
    self._use_native_grouped_read_norm = bool(
        cfg.bam_use_native_grouped_read_norm)
    self._local_qk_key_mode = cfg.bam_local_qk_key_mode
    self._factorized_head_output_layout = cfg.bam_factorized_head_output_layout
    self._pack_factorized_local_qk = bool(cfg.bam_pack_factorized_local_qk)
    self._batch_factorized_local_qk_read = bool(
        cfg.bam_batch_factorized_local_qk_read)
    self._readout_attribution = bool(cfg.bam_readout_attribution)
    self._replicate_ploc_up = bool(cfg.bam_replicate_ploc_up)
    self._local_qk_injection = cfg.bam_local_qk_injection
    self._local_qk_rope_pairing = cfg.bam_local_qk_rope_pairing
    self._force_activation_dtype = bool(cfg.bam_force_activation_dtype)
    self._shared_fetch_mode = cfg.bam_shared_fetch_mode
    self._fetch_mix_num_heads = (
        self.num_query_heads
        if cfg.bam_fetch_mix_num_heads is None
        else int(cfg.bam_fetch_mix_num_heads))
    assert 0 < self._fetch_mix_num_heads <= self.num_query_heads
    self._fetch_sliding_window_size = cfg.bam_fetch_sliding_window_size
    self._fetch_sliding_window_prefix_size = getattr(
        cfg, 'bam_fetch_sliding_window_prefix_size', None)
    self._fetch_temporal_block_size = cfg.bam_fetch_temporal_block_size
    self._fetch_temporal_block_mode = cfg.bam_fetch_temporal_block_mode
    self._fetch_temporal_recent_window_size = cfg.bam_fetch_temporal_recent_window_size
    self._codebook_source_implementation = cfg.bam_codebook_source_implementation
    self._codebook_read_implementation = cfg.bam_codebook_read_implementation
    self._share_full_local_read = cfg.bam_share_full_local_read
    self._combine_full_local_read = cfg.bam_combine_full_local_read
    self._fetch_diagonal_one = cfg.bam_fetch_diagonal_one
    self._read_implementation = cfg.bam_read_implementation
    self._m_read_norm = cfg.bam_m_read_norm
    self._squeeze_single_fetch_read = cfg.bam_squeeze_single_fetch_read
    self._fetch_read_bottleneck_dim = getattr(
        cfg, 'bam_fetch_read_bottleneck_dim', None)
    self._fetch_read_bottleneck_activation = getattr(
        cfg, 'bam_fetch_read_bottleneck_activation', 'none')
    self._abs_k_dim = (
        getattr(cfg, 'bam_abs_k_compression_dim', None)
        if 'full' in self._mode else None)
    self._abs_k_col_output = getattr(cfg, 'bam_abs_k_col_output', 'direct')
    self._abs_v_dim = (
        getattr(cfg, 'bam_abs_v_compression_dim', None)
        if 'full' in self._mode else None)
    self._abs_v_row_output = getattr(cfg, 'bam_abs_v_row_output', 'direct')
    self._abs_v_source_implementation = getattr(
        cfg, 'bam_abs_v_source_implementation', 'dot')
    if (self._shared_fetch_mode in ('dynamic_mix', 'dynamic_rms_mix')
        and {'full', 'codebook'} & self._mode):
      assert not cfg.bam_dedicated_fetch, 'dynamic head mixing and dedicated fetch are exclusive'
      assert cfg.bam_n_f == 1, 'dynamic head mixing produces exactly one fetch route'
    self._write_v_mode = cfg.bam_write_v_mode
    self._write_u2_norm = cfg.bam_write_u2_norm
    self._write_v_bottleneck_dim = cfg.bam_write_v_bottleneck_dim
    self._write_v_bottleneck_activation = cfg.bam_write_v_bottleneck_activation
    self._write_outer_implementation = cfg.bam_write_outer_implementation
    self._forget_mode = cfg.bam_forget_mode
    assert self._read_key_mode in ('none', 'soft_rms_cap', 'rms_gate')
    assert self._local_qk_key_mode in (
        'shared', 'factorized', 'per_head', 'per_head_static')
    assert self._factorized_head_output_layout in ('bnt', 'btn')
    assert not self._pack_factorized_local_qk or (
        'local_qk' in self._mode
        and self._local_qk_key_mode == 'factorized'
        and cfg.bam_create_read_gate_params), (
            'packed local-QK requires factorized keys with explicit read gates')
    assert not self._batch_factorized_local_qk_read or (
        self._pack_factorized_local_qk
        and self._factorized_head_output_layout == 'btn'), (
            'batched local-QK reads require packed projections and btn output')
    assert self._local_qk_injection in ('post_rope', 'pre_qknorm_rope')
    assert self._local_qk_injection == 'post_rope' or 'local_qk' in self._mode
    assert self._local_qk_rope_pairing in ('split_half', 'adjacent')
    assert self._local_qk_rope_pairing == 'split_half' or (
        self._local_qk_injection == 'pre_qknorm_rope' and not cfg.qk_norm), (
            'adjacent Q/K RoPE requires pre-RoPE LocalQK injection with QKNorm disabled')
    assert self._codebook_source_implementation in ('dot', 'mul_reduce')
    assert self._codebook_read_implementation in ('dot_bnt', 'dot_btn', 'mul_reduce_btn')
    assert self._shared_fetch_mode in (
        'legacy', 'compact', 'recompute', 'dynamic_mix', 'dynamic_rms_mix')
    if self._fetch_sliding_window_size is not None:
      assert self._fetch_sliding_window_size > 0
      assert not cfg.bam_dedicated_fetch
      assert self._shared_fetch_mode in ('dynamic_mix', 'dynamic_rms_mix'), (
          'BAM fetch sliding window currently masks the post-mix fetch alpha')
    if self._fetch_sliding_window_prefix_size is not None:
      assert self._fetch_sliding_window_prefix_size > 0
      assert self._fetch_sliding_window_size is not None, (
          'BAM fetch prefix retention requires a sliding window')
    if self._fetch_temporal_block_size is not None:
      assert self._fetch_temporal_block_size > 1
      assert self._fetch_temporal_block_mode in ('mean', 'linear')
      assert 'full' in self._mode, 'temporal block compression requires full fetch'
    else:
      assert self._fetch_temporal_block_mode == 'none'
      assert self._fetch_temporal_recent_window_size is None
    assert self._write_v_mode in ('x', 'x_bias', 'mix', 'o_tail', 'static')
    assert self._write_u2_norm in ('rms', 'grouped_rms_bias')
    assert self._write_u2_norm == 'rms' or self._write_v_mode == 'o_tail'
    assert not (self._write_u2_norm != 'rms' and self._create_grouped_rw_norm)
    assert self._write_v_bottleneck_activation in ('none', 'gelu')
    if self._write_v_bottleneck_dim is None:
      assert self._write_v_bottleneck_activation == 'none'
    else:
      assert 0 < self._write_v_bottleneck_dim < cfg.emb_dim
      assert self._write_v_mode in ('x', 'x_bias')
    assert not self._replicate_ploc_up or self._write_v_bottleneck_dim is not None, (
        'replicated P_loc_up requires a write-V bottleneck')
    assert self._write_outer_implementation in ('dot', 'mul_reduce')
    assert self._m_read_norm in ('rms', 'none')
    assert self._forget_mode in ('constant', 'dynamic')
    assert self._read_implementation in ('dot_bnt', 'dot_btn', 'mul_reduce_btn')
    assert self._query_chunk_implementation in (
        'legacy', 'no_remat', 'deferred_read', 'diag_select', 'optimized')
    assert self._fetch_read_bottleneck_activation in ('none', 'gelu')
    if self._fetch_read_bottleneck_dim is None:
      assert self._fetch_read_bottleneck_activation == 'none'
    else:
      assert 0 < self._fetch_read_bottleneck_dim < cfg.emb_dim
      assert 'full' in self._mode
    assert self._abs_k_col_output in ('direct', 'project')
    assert self._abs_v_row_output in ('direct', 'project')
    assert self._abs_v_source_implementation in ('dot', 'mul_reduce')
    if self._abs_k_dim is not None:
      assert 0 < self._abs_k_dim < self.bam_k
      assert self._abs_v_dim is not None, (
          'absolute-K compression currently pairs with absolute-V compression')
    if self._abs_v_dim is not None:
      assert 0 < self._abs_v_dim < self.bam_v
      assert 'full' in self._mode
      assert self._combine_full_local_read or (
          self._fetch_diagonal_one and 'local_o' not in self._mode), (
              'absolute-V compression requires combined local/full or strict diagonal-one read')
    if 'full' in self._mode:
      full_v_output_dim = (
          self.bam_v
          if self._abs_v_dim is None or self._abs_v_row_output == 'project'
          else self._abs_v_dim)
      assert full_v_output_dim <= self.head_dim - self.bam_k, (
          f'fetched BAM output needs {self.bam_k}+{full_v_output_dim} head '
          f'coordinates, but head_dim={self.head_dim}')
    assert self._read_key_scale > 0.0
    assert self._rms_epsilon > 0.0
    assert self._read_key_epsilon > 0.0
    assert self._read_gate_init is None or 0.0 < self._read_gate_init < 1.0
    assert self._read_key_mode != 'rms_gate' or cfg.bam_create_read_gate_params
    assert not self._use_grouped_rw_norm or self._create_grouped_rw_norm
    assert not self._create_grouped_rw_norm or self._read_key_mode == 'rms_gate'
    assert not self._use_native_grouped_read_norm or self._read_key_mode == 'rms_gate'
    assert not (self._use_native_grouped_read_norm and self._create_grouped_rw_norm)
    assert not self._create_grouped_rw_norm or (
        'local_qk' not in self._mode or self._local_qk_key_mode == 'per_head'), (
            'per-head learned local_qk normalization requires per-head runtime keys')
    if self._share_full_local_read and {'full', 'local_o'} & self._mode:
      assert {'full', 'local_o'} <= self._mode, (
          'shared full/local read projections require both full and local_o')
      assert cfg.bam_n_f == 1, (
          'shared full/local read projections require exactly one full fetch')
    assert not self._combine_full_local_read or self._share_full_local_read, (
        'combining full/local reads requires shared projections')
    assert not self._combine_full_local_read or cfg.bam_write_source != 'std+cross', (
        "combined read cannot isolate cross-only content for bam_write_source='std+cross'")
    if self._fetch_diagonal_one and {'full', 'codebook'} & self._mode:
      assert {'full', 'codebook'} & self._mode and 'local_o' not in self._mode, (
          'direct fetch diagonal one requires full/codebook without a local_o branch')
      assert cfg.bam_n_f == 1, 'direct fetch diagonal one requires exactly one fetch route'
      assert not cfg.bam_keep_fetch_diagonal, (
          'direct fetch diagonal one and keep_fetch_diagonal are mutually exclusive')
    if self._query_chunk_size is not None:
      assert self._query_chunk_size > 0
      assert cfg.max_target_length % self._query_chunk_size == 0, (
          'BAM query chunk size must divide max_target_length')
      assert self._mode in ({'local_qk'}, {'local_qk', 'full'}), (
          'QChunk BAM supports V2 local_qk layers with optional full fetch')
      if 'full' in self._mode:
        assert self._query_chunk_implementation == 'optimized'
        assert self._shared_fetch_mode == 'dynamic_rms_mix'
        assert cfg.bam_n_f == 1 and self._fetch_diagonal_one
        assert not cfg.bam_dedicated_fetch
        assert self._fetch_sliding_window_size is None
        assert self._fetch_temporal_block_size is None
        assert self._read_implementation in ('dot_btn', 'mul_reduce_btn')
      assert not cfg.bam_diagnostics
    if self._squeeze_single_fetch_read and 'full' in self._mode:
      assert 'full' in self._mode and cfg.bam_n_f == 1, (
          'single-fetch squeeze requires one full-read route')
    assert not cfg.bam_dedicated_fetch or 'full' in self._mode, (
        'bam_dedicated_fetch currently requires the full read mode')
    def add_read_gate(name, features, kernel_axes, bias_axes, initial_gate):
      """Create a zero-kernel semantic gate with an explicitly calibrated bias."""
      if not cfg.bam_create_read_gate_params:
        return
      assert 0.0 < initial_gate < 1.0
      setattr(self, name, DenseGeneral(
          features=features, axis=-1, kernel_init=zeros_init, kernel_axes=kernel_axes,
          dtype=self.dtype, weight_dtype=self.weight_dtype, name=name,
          quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=False))
      bias_value = math.log(initial_gate / (1.0 - initial_gate))
      setattr(self, f'{name}_b0', self.param(
          f'{name}_b0',
          nn.with_logical_partitioning(
              lambda key, shape, dtype: jnp.full(shape, bias_value, dtype), bias_axes),
          features, self.weight_dtype))

    def add_grouped_read_norms(name, row_shape, col_shape, row_axes, col_axes):
      if not (self._create_grouped_rw_norm or self._use_native_grouped_read_norm):
        return
      setattr(self, f'{name}_row_norm', GroupedRMSNorm(
          scale_shape=row_shape, epsilon=self._read_key_epsilon, dtype=self.dtype,
          weight_dtype=self.weight_dtype, kernel_axes=row_axes,
          name=f'{name}_row_norm'))
      setattr(self, f'{name}_col_norm', GroupedRMSNorm(
          scale_shape=col_shape, epsilon=self._read_key_epsilon, dtype=self.dtype,
          weight_dtype=self.weight_dtype, kernel_axes=col_axes,
          name=f'{name}_col_norm'))

    zero_key_gate_init = (
        self._read_gate_init
        if self._read_gate_init is not None
        else math.sqrt(self._read_key_epsilon) / self._read_key_scale)
    assert zero_key_gate_init < 1.0

    if 'codebook' in self._mode:
      self.W_beta = DenseGeneral(
          features=(self.num_query_heads, cfg.bam_n_f, 2 * cfg.bam_C), axis=-1,
          kernel_init=zeros_init, kernel_axes=("embed", "q_heads", "fetch", "code"),
          dtype=self.dtype, weight_dtype=self.weight_dtype, name="W_beta",
          quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=False)
      self.rho_u = self.param('rho_u', orth_init, (cfg.bam_C, self.bam_k), self.weight_dtype)
      self.rho_v = self.param('rho_v', orth_init, (cfg.bam_C, self.bam_v), self.weight_dtype)
      add_read_gate('W_beta_gate', (self.num_query_heads, cfg.bam_n_f, 2),
                    ('embed', 'q_heads', 'fetch', None), ('q_heads', 'fetch', None),
                    zero_key_gate_init)
      add_grouped_read_norms(
          'W_beta', (self.num_query_heads, cfg.bam_n_f, cfg.bam_C),
          (self.num_query_heads, cfg.bam_n_f, cfg.bam_C),
          ('q_heads', 'fetch', 'kv'), ('q_heads', 'fetch', 'kv'))

    if 'full' in self._mode:
      read_k_dim = self._abs_k_dim or self.bam_k
      read_v_dim = self._abs_v_dim or self.bam_v

      def decoder_init(_key, shape, dtype):
        direct = jnp.eye(shape[-2], shape[-1], dtype=dtype)
        return jnp.broadcast_to(direct, shape)

      if self._abs_v_dim is not None:
        # M itself remains [k,v] across layers.  Only the historical cache/read view is
        # projected on its V axis; the target-side column key is generated directly
        # in the compressed space.
        self.abs_v_cache_projection = self.param(
            'abs_v_cache_projection',
            nn.with_logical_partitioning(orth_init, ('v_factor', 'kv')),
            (self.bam_v, self._abs_v_dim), self.weight_dtype)
        # Create the decoder in both paired arms so their parameter trees and initializers
        # match.  The direct arm deliberately leaves it unused.
        self.abs_v_row_decoder = self.param(
            'abs_v_row_decoder',
            nn.with_logical_partitioning(
                decoder_init, ('q_heads', 'kv', 'v_factor')),
            (self.num_query_heads, self._abs_v_dim, self.bam_v), self.weight_dtype)

      if self._abs_k_dim is not None:
        # Compress only the historical cache/read view; the cross-layer M stream
        # remains full [k,v].  V is compressed first at runtime because it is the
        # narrower axis, then this projection maps K to its cached width.
        self.abs_k_cache_projection = self.param(
            'abs_k_cache_projection',
            nn.with_logical_partitioning(orth_init, ('v_factor', 'kv')),
            (self.bam_k, self._abs_k_dim), self.weight_dtype)
        # The identity-prefix initializer makes the project arm exactly match the
        # direct zero-pad arm at step zero while retaining identical parameter trees.
        self.abs_k_col_decoder = self.param(
            'abs_k_col_decoder',
            nn.with_logical_partitioning(
                decoder_init, ('q_heads', 'kv', 'v_factor')),
            (self.num_query_heads, self._abs_k_dim, self.bam_k), self.weight_dtype)

      # Joint target-side read key is generated directly in both cached spaces.
      read_features = (self.num_query_heads, cfg.bam_n_f, read_k_dim + read_v_dim)
      if self._fetch_read_bottleneck_dim is None:
        self.W_R = DenseGeneral(
            features=read_features, axis=-1, kernel_init=zeros_init,
            kernel_axes=("embed", "q_heads", "fetch", "kv"),
            dtype=self.dtype, weight_dtype=self.weight_dtype, name="W_R",
            quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=False)
      else:
        self.W_R_down = DenseGeneral(
            features=self._fetch_read_bottleneck_dim, axis=-1,
            kernel_init=reg_init, kernel_axes=("embed", None),
            dtype=self.dtype, weight_dtype=self.weight_dtype, name="W_R_down",
            quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=False)
        self.W_R_up = DenseGeneral(
            features=read_features, axis=-1, kernel_init=zeros_init,
            kernel_axes=("embed", "q_heads", "fetch", "kv"),
            dtype=self.dtype, weight_dtype=self.weight_dtype, name="W_R_up",
            quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=False)
      add_read_gate('W_R_gate', (self.num_query_heads, cfg.bam_n_f, 2),
                    ('embed', 'q_heads', 'fetch', None), ('q_heads', 'fetch', None),
                    zero_key_gate_init)
      add_grouped_read_norms(
          'W_R', (self.num_query_heads, cfg.bam_n_f, read_k_dim),
          (self.num_query_heads, cfg.bam_n_f, read_v_dim),
          ('q_heads', 'fetch', 'kv'), ('q_heads', 'fetch', 'kv'))

      if cfg.bam_dedicated_fetch:
        # Capability-ceiling router: fetch patterns no longer have to borrow the first
        # n_f standard MHA heads and can specialize without sacrificing MHA behavior.
        self.fetch_query = DenseGeneral(
            features=(cfg.bam_n_f, self.head_dim), axis=-1, kernel_init=reg_init,
            kernel_axes=('embed', 'q_heads', 'kv'), dtype=self.dtype,
            weight_dtype=self.weight_dtype, name='fetch_query', quant=self.quant,
            matmul_precision=cfg.matmul_precision, use_bias=cfg.qkv_bias)
        self.fetch_key = DenseGeneral(
            features=(cfg.bam_n_f, self.head_dim), axis=-1, kernel_init=reg_init,
            kernel_axes=('embed', 'kv_heads', 'kv_head_dim'), dtype=self.dtype,
            weight_dtype=self.weight_dtype, name='fetch_key', quant=self.quant,
            matmul_precision=cfg.matmul_precision, use_bias=cfg.qkv_bias)

    if (self._shared_fetch_mode in ('dynamic_mix', 'dynamic_rms_mix')
        and {'full', 'codebook'} & self._mode):
      # Softmax starts from a uniform convex mixture. Signed RMS mixing needs a
      # regular-initialized direction because RMSNorm at an all-zero vector is singular;
      # the content-read projection remains zero-init, preserving the exact MHA start.
      mix_init = zeros_init if self._shared_fetch_mode == 'dynamic_mix' else reg_init
      self.fetch_head_mix = DenseGeneral(
          features=self._fetch_mix_num_heads, axis=-1, kernel_init=mix_init,
          kernel_axes=('embed', 'q_heads'), dtype=self.dtype,
          weight_dtype=self.weight_dtype, name='fetch_head_mix', quant=self.quant,
          matmul_precision=cfg.matmul_precision, use_bias=True)

    if 'local_qk' in self._mode:
      if self._local_qk_key_mode == 'per_head_static':
        # Bias-only endpoint of the per-head runtime-key projection: one learned row/column
        # key per layer, head, and Q/K use point. Zero init preserves the exact MHA start;
        # no RMS read gate is created or applied, so the zero-point Jacobian stays one.
        for _name in ("W_lq", "W_lk"):
          setattr(self, _name, self.param(
              _name,
              nn.with_logical_partitioning(zeros_init, ("q_heads", "kv")),
              (self.num_query_heads, self.bam_k + self.bam_v),
              self.weight_dtype,
          ))
      elif self._local_qk_key_mode == 'per_head':
        # Capability ablation symmetric with local_o: each head gets its own
        # runtime row/column key and gate, with no post-read static rematrix.
        # Zero-init keys preserve the exact MHA starting function.
        for _name in ("W_lq", "W_lk"):
          setattr(self, _name, DenseGeneral(
              features=(self.num_query_heads, self.bam_k + self.bam_v), axis=-1,
              kernel_init=zeros_init, kernel_axes=("embed", "q_heads", "kv"),
              dtype=self.dtype, weight_dtype=self.weight_dtype, name=_name,
              quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=True))
          add_read_gate(f'{_name}_gate', (self.num_query_heads, 2),
                        ('embed', 'q_heads', None), ('q_heads', None),
                        zero_key_gate_init)
          add_grouped_read_norms(
              _name, (self.num_query_heads, self.bam_k),
              (self.num_query_heads, self.bam_v),
              ('q_heads', 'kv'), ('q_heads', 'kv'))
      elif self._local_qk_key_mode == 'factorized':
        # One zero-init runtime row/column key per Q/K use point, dynamically
        # distributed over heads by independent signed row/column coefficients.
        # The read is performed once per side; no static dxd rematrix is needed.
        if self._pack_factorized_local_qk:
          key_width = self.bam_k + self.bam_v
          packed_width = 2 * (key_width + 2 + 2 * self.num_query_heads)
          self.W_local_qk_packed = DenseGeneral(
              features=packed_width, axis=-1,
              kernel_init=_packed_factorized_local_qk_init(
                  reg_init, self.num_query_heads, key_width),
              kernel_axes=("embed", None), dtype=self.dtype,
              weight_dtype=self.weight_dtype, name="W_local_qk_packed",
              quant=self.quant, matmul_precision=cfg.matmul_precision,
              use_bias=False)
          bias_value = math.log(
              zero_key_gate_init / (1.0 - zero_key_gate_init))
          if self._batch_factorized_local_qk_read:
            self.W_local_qk_bias = self.param(
                'W_local_qk_bias',
                nn.with_logical_partitioning(zeros_init, (None, 'kv')),
                (2, key_width), self.weight_dtype)
            self.W_local_qk_gate_b0 = self.param(
                'W_local_qk_gate_b0',
                nn.with_logical_partitioning(
                    lambda key, shape, dtype: jnp.full(
                        shape, bias_value, dtype), (None, None)),
                (2, 2), self.weight_dtype)
            add_grouped_read_norms(
                'W_local_qk', (2, self.bam_k), (2, self.bam_v),
                (None, 'kv'), (None, 'kv'))
          else:
            for _name in ("W_lq", "W_lk"):
              setattr(self, f'{_name}_bias', self.param(
                  f'{_name}_bias',
                  nn.with_logical_partitioning(zeros_init, ('kv',)),
                  (key_width,), self.weight_dtype))
              setattr(self, f'{_name}_gate_b0', self.param(
                  f'{_name}_gate_b0',
                  nn.with_logical_partitioning(
                      lambda key, shape, dtype: jnp.full(
                          shape, bias_value, dtype), (None,)),
                  (2,), self.weight_dtype))
              add_grouped_read_norms(
                  _name, (self.bam_k,), (self.bam_v,), ('kv',), ('kv',))
        else:
          for _name in ("W_lq", "W_lk"):
            setattr(self, _name, DenseGeneral(
                features=(self.bam_k + self.bam_v,), axis=-1, kernel_init=zeros_init,
                kernel_axes=("embed", "kv"), dtype=self.dtype,
                weight_dtype=self.weight_dtype, name=_name, quant=self.quant,
                matmul_precision=cfg.matmul_precision, use_bias=True))
            add_read_gate(f'{_name}_gate', (2,), ('embed', None), (None,),
                          zero_key_gate_init)
            setattr(self, f'{_name}_head_mix', DenseGeneral(
                features=(self.num_query_heads, 2), axis=-1, kernel_init=reg_init,
                kernel_axes=("embed", "q_heads", None), dtype=self.dtype,
                weight_dtype=self.weight_dtype, name=f'{_name}_head_mix',
                quant=self.quant, matmul_precision=cfg.matmul_precision,
                use_bias=False))
            add_grouped_read_norms(
                _name, (self.bam_k,), (self.bam_v,), ('kv',), ('kv',))
      else:
        # Default tier: one shared runtime key per use point, then a per-head
        # zero-init static rematrix gates and remixes the readout.
        for _name in ("W_lq", "W_lk"):
          setattr(self, _name, DenseGeneral(
              features=(self.bam_k + self.bam_v,), axis=-1, kernel_init=reg_init,
              kernel_axes=("embed", "kv"), dtype=self.dtype,
              weight_dtype=self.weight_dtype, name=_name, quant=self.quant,
              matmul_precision=cfg.matmul_precision, use_bias=True))
          add_read_gate(f'{_name}_gate', (2,), ('embed', None), (None,), 0.1)
        for _name in ("R_q", "R_k"):
          # R is large enough across 24 unrolled layers that leaving it without
          # logical axes breaches MaxText's sharding audit.
          setattr(self, _name, self.param(
              _name,
              nn.with_logical_partitioning(zeros_init, ("q_heads", "embed", "kv")),
              (self.num_query_heads, self.head_dim, self.head_dim),
              self.weight_dtype,
          ))

      local_v_output_dim = self.head_dim - self.bam_k
      if self.bam_v > local_v_output_dim:
        def local_v_adapter_init(key, shape, dtype):
          return reg_init(key, shape, dtype, 1, 2)
        for _name in ('local_q_v_adapter', 'local_k_v_adapter'):
          setattr(self, _name, self.param(
              _name,
              nn.with_logical_partitioning(
                  local_v_adapter_init, ('q_heads', 'v_factor', 'kv')),
              (self.num_query_heads, self.bam_v, local_v_output_dim),
              self.weight_dtype,
          ))

    if 'local_o' in self._mode and not self._share_full_local_read:
      # local_o = content branch (§4.6.5, 2026-08-01): per-head tier (R is None),
      # W_Ro: D->n(k+v) zero-init (gates readout to zero at start). Reads out [U;V] d-wide
      # and merges into o via integration point 1 (y_bam -> o). alpha:=delta degenerate
      # (fetch free); same-layer FFN sees the readout via residual.
      self.W_Ro = DenseGeneral(
          features=(self.num_query_heads, self.bam_k + self.bam_v), axis=-1,
          kernel_init=zeros_init, kernel_axes=("embed", "q_heads", "kv"),
          dtype=self.dtype, weight_dtype=self.weight_dtype, name="W_Ro",
          quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=False)
      add_read_gate('W_Ro_gate', (self.num_query_heads, 2),
                    ('embed', 'q_heads', None), ('q_heads', None), zero_key_gate_init)
      add_grouped_read_norms(
          'W_Ro', (self.num_query_heads, self.bam_k),
          (self.num_query_heads, self.bam_v),
          ('q_heads', 'kv'), ('q_heads', 'kv'))

    if self._has_write:
      if cfg.bam_write_u_proj or cfg.bam_create_write_u_proj_params:
        def write_u_init(key, shape, dtype):
          return reg_init(key, shape, dtype, 1, 2)
        self.P_agg_u = self.param(
            'P_agg_u',
            nn.with_logical_partitioning(write_u_init, ('q_heads', 'embed', 'v_factor')),
            (self.num_query_heads, self.head_dim, self.bam_k), self.weight_dtype)
      # Write anchor P_loc: V factor (default agg_u@loc_v), regular init
      loc_v = self.bam_v if cfg.bam_write_form == 'agg_u@loc_v' else self.bam_k
      assert self._write_v_mode != 'mix' or loc_v == self.bam_v
      if self._write_v_mode == 'o_tail':
        assert self.head_dim - self.bam_k == loc_v, (
            'o_tail write requires the o_head tail width to equal the V-factor width')
      elif self._write_v_mode == 'static':
        # One fixed V-side write direction per head.  Scale the orthogonal rows to unit
        # raw RMS so the existing write-side RMS reparameterization starts near identity.
        self.S_v = self.param(
            'S_v',
            nn.with_logical_partitioning(
                nn.initializers.orthogonal(math.sqrt(loc_v)),
                ('q_heads', 'v_factor')),
            (self.num_query_heads, loc_v), self.weight_dtype)
      elif self._write_v_bottleneck_dim is not None:
        self.P_loc_down = DenseGeneral(
            features=self._write_v_bottleneck_dim, axis=-1,
            kernel_init=reg_init, kernel_axes=("embed", None),
            dtype=self.dtype, weight_dtype=self.weight_dtype,
            name="P_loc_down", quant=self.quant,
            matmul_precision=cfg.matmul_precision, use_bias=False)
        self.P_loc_up = DenseGeneral(
            features=(self.num_query_heads, loc_v), axis=-1,
            kernel_init=reg_init,
            kernel_axes=(
                None if self._replicate_ploc_up else "embed",
                "q_heads", "v_factor"),
            dtype=self.dtype, weight_dtype=self.weight_dtype,
            name="P_loc_up", quant=self.quant,
            matmul_precision=cfg.matmul_precision,
            use_bias=self._write_v_mode == 'x_bias')
      else:
        self.P_loc = DenseGeneral(features=(self.num_query_heads, loc_v), axis=-1,
            kernel_init=reg_init, kernel_axes=("embed", "q_heads", "v_factor"),
            dtype=self.dtype, weight_dtype=self.weight_dtype, name="P_loc",
            quant=self.quant, matmul_precision=cfg.matmul_precision,
            use_bias=self._write_v_mode == 'x_bias')
      # Write gate g_write: regular kernel, bias = logit(eps) explicitly slightly open
      self.W_gw = DenseGeneral(features=(self.num_query_heads,), axis=-1, kernel_init=reg_init,
          kernel_axes=("embed", "q_heads"), dtype=self.dtype, weight_dtype=self.weight_dtype,
          name="W_gw", quant=self.quant, matmul_precision=cfg.matmul_precision, use_bias=False)
      eps = float(cfg.bam_write_eps)
      self.gw_b0 = self.param('gw_b0',
          nn.with_logical_partitioning(
              lambda key, shape, dtype: jnp.full(shape, math.log(eps / (1.0 - eps))),
              ("q_heads",)),
          (self.num_query_heads,), self.weight_dtype)
      if self._write_v_mode == 'mix':
        self.write_v_mix_scale = self.param(
            'write_v_mix_scale',
            nn.with_logical_partitioning(
                lambda key, shape, dtype: jnp.broadcast_to(
                    jnp.asarray([1.0, 0.0], dtype=dtype), shape),
                ('q_heads', None)),
            (self.num_query_heads, 2), self.weight_dtype)
        self.write_v_bias = self.param(
            'write_v_bias',
            nn.with_logical_partitioning(
                lambda key, shape, dtype: jnp.zeros(shape, dtype),
                ('q_heads', 'v_factor')),
            (self.num_query_heads, loc_v), self.weight_dtype)
      if self._forget_mode == 'dynamic':
        forget_init = float(cfg.bam_forget_init)
        assert 0.0 < forget_init < 1.0
        self.W_forget_gate = DenseGeneral(
            features=(1,), axis=-1, kernel_init=zeros_init,
            kernel_axes=('embed', None), dtype=self.dtype,
            weight_dtype=self.weight_dtype, name='W_forget_gate', quant=self.quant,
            matmul_precision=cfg.matmul_precision, use_bias=False)
        forget_bias = math.log(forget_init / (1.0 - forget_init))
        self.W_forget_gate_b0 = self.param(
            'W_forget_gate_b0',
            nn.with_logical_partitioning(
                lambda key, shape, dtype: jnp.full(shape, forget_bias, dtype),
                (None,)),
            (1,), self.weight_dtype)
      if self._create_grouped_rw_norm:
        self.write_u1_norm = GroupedRMSNorm(
            scale_shape=(self.num_query_heads, self.bam_k),
            epsilon=self._rms_epsilon, dtype=self.dtype,
            weight_dtype=self.weight_dtype, kernel_axes=('q_heads', 'kv'),
            name='write_u1_norm')
        self.write_u2_norm = GroupedRMSNorm(
            scale_shape=(self.num_query_heads, loc_v),
            epsilon=self._rms_epsilon, dtype=self.dtype,
            weight_dtype=self.weight_dtype, kernel_axes=('q_heads', 'kv'),
            name='write_u2_norm')
      elif self._write_u2_norm == 'grouped_rms_bias':
        self.write_u2_norm = GroupedRMSNorm(
            scale_shape=(self.num_query_heads, loc_v),
            epsilon=self._rms_epsilon, dtype=self.dtype,
            weight_dtype=self.weight_dtype, kernel_axes=('q_heads', 'kv'),
            use_bias=True, name='write_u2_norm')

    if 'local_v' in self._mode:
      # Source-local matrix summary injected into the standard V stream.  Match
      # FactorizedLocalQK: one shared bilateral key, then signed dynamic head routing.
      self.W_lv = DenseGeneral(
          features=(self.bam_k + self.bam_v,), axis=-1, kernel_init=zeros_init,
          kernel_axes=("embed", "kv"), dtype=self.dtype,
          weight_dtype=self.weight_dtype, name="W_lv", quant=self.quant,
          matmul_precision=cfg.matmul_precision, use_bias=True)
      add_read_gate('W_lv_gate', (2,), ('embed', None), (None,), zero_key_gate_init)
      self.W_lv_head_mix = DenseGeneral(
          features=(self.num_query_heads, 2), axis=-1, kernel_init=reg_init,
          kernel_axes=("embed", "q_heads", None), dtype=self.dtype,
          weight_dtype=self.weight_dtype, name="W_lv_head_mix", quant=self.quant,
          matmul_precision=cfg.matmul_precision, use_bias=False)
      add_grouped_read_norms(
          'W_lv', (self.bam_k,), (self.bam_v,), ('kv',), ('kv',))

  def _project_full_read_key(self, x):
    if self._fetch_read_bottleneck_dim is None:
      return self.W_R(x)
    x = self.W_R_down(x)
    if self._fetch_read_bottleneck_activation == 'gelu':
      x = nn.gelu(x)
    return self.W_R_up(x)

  def _read_key_kwargs(self, gate_name, x, squeeze_fetch_axis=False):
    candidate_logits = None
    if self.config.bam_create_read_gate_params:
      with jax.named_scope("bam/read_gate_projection"):
        gate_bias = getattr(self, f'{gate_name}_b0')
        if self._force_activation_dtype:
          gate_bias = jnp.asarray(gate_bias, self.dtype)
        candidate_logits = getattr(self, gate_name)(x) + gate_bias
        if squeeze_fetch_axis:
          candidate_logits = jnp.squeeze(candidate_logits, axis=-2)
    return self._read_key_kwargs_from_logits(
        gate_name.removesuffix('_gate'), candidate_logits,
        squeeze_fetch_axis=squeeze_fetch_axis)

  def _read_key_kwargs_from_logits(
      self, projection_name, candidate_logits, squeeze_fetch_axis=False):
    gate_logits = (
        candidate_logits if self._read_key_mode == 'rms_gate' else None)
    kwargs = dict(
        key_mode=self._read_key_mode,
        key_scale=self._read_key_scale,
        rms_epsilon=self._read_key_epsilon,
        rms_statistics_dtype=self._read_rms_statistics_dtype,
        key_gate_logits=gate_logits,
    )
    if self._create_grouped_rw_norm or self._use_native_grouped_read_norm:
      row_norm = getattr(self, f'{projection_name}_row_norm')
      col_norm = getattr(self, f'{projection_name}_col_norm')
      if squeeze_fetch_axis:
        raw_row_norm = row_norm
        raw_col_norm = col_norm
        row_norm = lambda z: jnp.squeeze(raw_row_norm(z[..., None, :]), axis=-2)
        col_norm = lambda z: jnp.squeeze(raw_col_norm(z[..., None, :]), axis=-2)
      kwargs.update(
          key_row_norm=row_norm,
          key_col_norm=col_norm,
          use_learned_key_norm=(
              self._use_grouped_rw_norm or self._use_native_grouped_read_norm),
      )
    return kwargs

  def _fit_local_qk_reads(self, q_local, k_local):
    """Place K-side first and adapt/pad V-side into the remaining head width."""
    q_adapter = getattr(self, 'local_q_v_adapter', None)
    k_adapter = getattr(self, 'local_k_v_adapter', None)
    return (
        _fit_bam_read_to_head(q_local, self.bam_k, self.head_dim, q_adapter),
        _fit_bam_read_to_head(k_local, self.bam_k, self.head_dim, k_adapter),
    )

  def _read_local_qk(self, Mh, inputs_q):
    """Read the local matrix into Q/K; callers choose the injection point."""
    if self._pack_factorized_local_qk:
      with jax.named_scope("bam/local_qk_packed_projection"):
        packed = self.W_local_qk_packed(inputs_q)
      key_width = self.bam_k + self.bam_v
      mix_width = 2 * self.num_query_heads
      if self._batch_factorized_local_qk_read:
        slot_width = key_width + 2 + mix_width
        packed = packed.reshape(packed.shape[:-1] + (2, slot_width))
        qk_key = packed[..., :key_width]
        qk_gate = packed[..., key_width:key_width + 2]
        qk_mix = packed[..., key_width + 2:].reshape(
            packed.shape[:-2] + (2, self.num_query_heads, 2))
        qk_key = qk_key + jnp.asarray(
            self.W_local_qk_bias, qk_key.dtype)
        qk_gate = qk_gate + jnp.asarray(
            self.W_local_qk_gate_b0, qk_gate.dtype)
        qk_u, qk_v = bam_read(
            Mh, inputs_q, lambda _x: qk_key, None,
            **self._read_key_kwargs_from_logits('W_local_qk', qk_gate),
            implementation=self._read_implementation, read_side=self.read_side,
            return_sides=True)
        if self._read_implementation == 'dot_bnt':
          qk_u = rearrange(qk_u, 'b q t d -> b t q d')
          qk_v = rearrange(qk_v, 'b q t d -> b t q d')
        qk_mix = normalizations.rms_norm(
            qk_mix, dtype=qk_u.dtype, epsilon=self._read_key_epsilon,
            axis=-2)
        row_mix, col_mix = qk_mix[..., 0], qk_mix[..., 1]
        qk_u = jnp.einsum('btqk,btqn->btqnk', qk_u, col_mix)
        qk_v = jnp.einsum('btqv,btqn->btqnv', qk_v, row_mix)
        qk_local = jnp.concatenate((qk_u, qk_v), axis=-1)
        return qk_local[:, :, 0], qk_local[:, :, 1]

      split_points = (
          key_width,
          key_width + 2,
          key_width + 2 + mix_width,
          2 * key_width + 2 + mix_width,
          2 * key_width + 4 + mix_width,
      )
      q_key, q_gate, q_mix, k_key, k_gate, k_mix = jnp.split(
          packed, split_points, axis=-1)
      q_mix = q_mix.reshape(q_mix.shape[:-1] + (self.num_query_heads, 2))
      k_mix = k_mix.reshape(k_mix.shape[:-1] + (self.num_query_heads, 2))
      q_key = q_key + jnp.asarray(self.W_lq_bias, q_key.dtype)
      k_key = k_key + jnp.asarray(self.W_lk_bias, k_key.dtype)
      q_gate = q_gate + jnp.asarray(self.W_lq_gate_b0, q_gate.dtype)
      k_gate = k_gate + jnp.asarray(self.W_lk_gate_b0, k_gate.dtype)
      capture_readout = self._readout_attribution and not self.is_initializing()
      q_local = factorized_head_bam_read(
          Mh, inputs_q, lambda _x: q_key, lambda _x: q_mix,
          **self._read_key_kwargs_from_logits('W_lq', q_gate),
          implementation=self._read_implementation, read_side=self.read_side,
          output_layout=self._factorized_head_output_layout,
          return_aux=capture_readout)
      k_local = factorized_head_bam_read(
          Mh, inputs_q, lambda _x: k_key, lambda _x: k_mix,
          **self._read_key_kwargs_from_logits('W_lk', k_gate),
          implementation=self._read_implementation, read_side=self.read_side,
          output_layout=self._factorized_head_output_layout,
          return_aux=capture_readout)
      if capture_readout:
        q_local, q_aux = q_local
        k_local, k_aux = k_local
        query_indices = _bam_readout_query_indices(inputs_q.shape[1])
        for prefix, aux in (('local_q', q_aux), ('local_k', k_aux)):
          self.sow(
              'bam_readout', f'{prefix}_post_gate_key',
              aux['post_gate_key'][:, query_indices])
          self.sow(
              'bam_readout', f'{prefix}_head_mix',
              aux['head_mix'][:, query_indices])
          self.sow(
              'bam_readout', f'{prefix}_pre_mix_read',
              aux['pre_mix_read'][:, query_indices])
      if self._factorized_head_output_layout == 'bnt':
        q_local = rearrange(q_local, 'b n t d -> b t n d')
        k_local = rearrange(k_local, 'b n t d -> b t n d')
      return self._fit_local_qk_reads(q_local, k_local)

    local_qk_q_kwargs = (
        {
            'rms_epsilon': self._read_key_epsilon,
            'rms_statistics_dtype': self._read_rms_statistics_dtype,
        }
        if self._local_qk_key_mode == 'per_head_static'
        else self._read_key_kwargs('W_lq_gate', inputs_q))
    local_qk_k_kwargs = (
        {
            'rms_epsilon': self._read_key_epsilon,
            'rms_statistics_dtype': self._read_rms_statistics_dtype,
        }
        if self._local_qk_key_mode == 'per_head_static'
        else self._read_key_kwargs('W_lk_gate', inputs_q))
    if self._local_qk_key_mode == 'factorized':   # V1 default
      q_local = factorized_head_bam_read(
          Mh, inputs_q, self.W_lq, self.W_lq_head_mix,
          **local_qk_q_kwargs, implementation=self._read_implementation,
          read_side=self.read_side,
          output_layout=self._factorized_head_output_layout)
      k_local = factorized_head_bam_read(
          Mh, inputs_q, self.W_lk, self.W_lk_head_mix,
          **local_qk_k_kwargs, implementation=self._read_implementation,
          read_side=self.read_side,
          output_layout=self._factorized_head_output_layout)
      if self._factorized_head_output_layout == 'bnt':
        q_local = rearrange(q_local, 'b n t d -> b t n d')
        k_local = rearrange(k_local, 'b n t d -> b t n d')
      return self._fit_local_qk_reads(q_local, k_local)

    local_qk_per_head = self._local_qk_key_mode in ('per_head', 'per_head_static')
    local_qk_R_q = None if local_qk_per_head else self.R_q
    local_qk_R_k = None if local_qk_per_head else self.R_k
    q_local = bam_read(
        Mh, inputs_q, self.W_lq, local_qk_R_q, **local_qk_q_kwargs,
        implementation=self._read_implementation, read_side=self.read_side)
    k_local = bam_read(
        Mh, inputs_q, self.W_lk, local_qk_R_k, **local_qk_k_kwargs,
        implementation=self._read_implementation, read_side=self.read_side)
    if self._read_implementation == 'dot_bnt':
      q_local = rearrange(q_local, 'b n t d -> b t n d')
      k_local = rearrange(k_local, 'b n t d -> b t n d')
    return self._fit_local_qk_reads(q_local, k_local)

  def _read_local_v(self, Mh, inputs_kv):
    """Summarize each source-local matrix into the corresponding standard V."""
    v_local = factorized_head_bam_read(
        Mh, inputs_kv, self.W_lv, self.W_lv_head_mix,
        **self._read_key_kwargs('W_lv_gate', inputs_kv),
        implementation=self._read_implementation, read_side=self.read_side)
    return rearrange(v_local, 'b n t d -> b t n d')

  def _matrix_for_read(self, M_in):
    """Select the configured read-side view without changing the raw matrix stream."""
    if M_in is None or self._m_read_norm == 'none':
      return M_in
    return M_in * jax.lax.rsqrt(
        jnp.mean(M_in ** 2, axis=(-2, -1), keepdims=True) + self._rms_epsilon)

  def _apply_adjacent_rope(self, x, positions, name):
    """Apply the standard RoPE frequencies to adjacent coordinate pairs."""
    packed = jnp.concatenate([x[..., 0::2], x[..., 1::2]], axis=-1)
    rotated = self.apply_rotary_embedding(packed, positions, name=name)
    first, second = jnp.split(rotated, 2, axis=-1)
    return jnp.stack([first, second], axis=-1).reshape(x.shape)

  def _write(self, o_head, x, M_in):
    """Write primitive (§4.2 safe write: aggregated U (outer) local V). o_head: [b,t,n,d] head output (pre W_O).

    Per-record factor normalization (§4.6.5 write-side per-record factor norm): each factor is RMS-normalized
    over its head-dim axis before the outer product, so a single record has O(1) energy and the
    gate is the sole magnitude channel (admission semantics). rms(u) = u·rsqrt(mean(u²,-1)+eps).
    """
    cfg = self.config
    if self._force_activation_dtype:
      assert M_in.dtype == self.dtype, (M_in.dtype, self.dtype)
      assert o_head.dtype == self.dtype, (o_head.dtype, self.dtype)
    if cfg.bam_write_u_proj:  # XD: needed by adaptation of pretrained models
      write_u_proj = self.P_agg_u
      if self._force_activation_dtype:
        write_u_proj = jnp.asarray(write_u_proj, self.dtype)
      u1 = jnp.einsum('btnd,ndk->btnk', o_head, write_u_proj)
    else:  # V1 default
      u1 = o_head[..., :self.bam_k]                        # U factor [b,t,n,k]
    if self._write_v_mode == 'o_tail':
      u2 = o_head[..., self.bam_k:]
    elif self._write_v_mode == 'static':
      u2 = self.S_v
      if self._force_activation_dtype:
        u2 = jnp.asarray(u2, self.dtype)
    else:  # V1 default
      if self._write_v_bottleneck_dim is None:
        x_v = self.P_loc(x)
      else:  # V1 default
        x_v = self.P_loc_down(x)
        if self._write_v_bottleneck_activation == 'gelu':
          x_v = nn.gelu(x_v)
        x_v = self.P_loc_up(x_v)
      u2 = x_v
    if self._write_v_mode == 'mix':
      write_v_mix_scale = self.write_v_mix_scale
      write_v_bias = self.write_v_bias
      if self._force_activation_dtype:
        write_v_mix_scale = jnp.asarray(write_v_mix_scale, self.dtype)
        write_v_bias = jnp.asarray(write_v_bias, self.dtype)
      u2 = _mix_bam_write_v(
          x_v, o_head, self.bam_k, write_v_mix_scale, write_v_bias)
    write_gate_bias = self.gw_b0
    if self._force_activation_dtype:
      write_gate_bias = jnp.asarray(write_gate_bias, self.dtype)
    gate = jax.nn.sigmoid(self.W_gw(x) + write_gate_bias)  # [b,t,n]
    g = gate
    if cfg.bam_sqrt_n_scale:
      # With per-record rms, each head's record is unit energy, so |M| ~ n * Σg. Scaling the
      # gate by 1/sqrt(n) damps each head's write so |M| ~ sqrt(n) — head-count-invariant
      # dynamics, analogous to attention's 1/sqrt(d). No-op at n==1.
      g = g * (1.0 / jnp.sqrt(self.num_query_heads))
    u1_norm = normalizations.rms_norm(
        u1, dtype=self.dtype, epsilon=self._rms_epsilon,
        statistics_dtype=self._write_rms_statistics_dtype)
    u2_norm = normalizations.rms_norm(
        u2, dtype=self.dtype, epsilon=self._rms_epsilon,
        statistics_dtype=self._write_rms_statistics_dtype)
    if self._create_grouped_rw_norm:
      learned_u1_norm = self.write_u1_norm(u1)
      learned_u2_norm = self.write_u2_norm(u2)
      if self._use_grouped_rw_norm:
        u1_norm = learned_u1_norm
        u2_norm = learned_u2_norm
    elif self._write_u2_norm == 'grouped_rms_bias':
      u2_norm = self.write_u2_norm(u2)
    gated_u1 = g[..., None] * u1_norm
    if self._readout_attribution and not self.is_initializing():
      # The gradient of this per-record scale is exactly
      # <dL/ddM, g * rms(u1) outer rms(u2)> without exporting dL/ddM itself.
      record_scale = self.perturb(
          'write_record_scale', jnp.zeros(g.shape, jnp.float32))
      gated_u1 = gated_u1 * jnp.asarray(
          1 + record_scale[..., None], gated_u1.dtype)
      self.sow('bam_readout', 'write_u1_norm', u1_norm)
      self.sow('bam_readout', 'write_u2_norm', u2_norm)
      self.sow('bam_readout', 'write_scale', g)
      self.sow('bam_readout', 'write_gate', gate)
    with jax.named_scope("bam/write_outer"):
      if self._write_outer_implementation == 'dot':
        if self._write_v_mode == 'static':
          dM = jnp.einsum('btnk,nv->btkv', gated_u1, u2_norm)
        else:
          dM = jnp.einsum('btnk,btnv->btkv', gated_u1, u2_norm)
      elif self._write_v_mode == 'static':
        dM = jnp.sum(
            gated_u1[..., None] * u2_norm[None, None, :, None, :], axis=-3)
      else:
        assert self._write_outer_implementation == 'mul_reduce'  # XD
        dM = jnp.sum(
            gated_u1[..., None] * u2_norm[..., None, :], axis=-3)
    if self._force_activation_dtype:
      assert dM.dtype == self.dtype, (dM.dtype, self.dtype)
    forget_logits = None
    if self._forget_mode == 'dynamic':
      forget_gate_bias = self.W_forget_gate_b0
      if self._force_activation_dtype:
        forget_gate_bias = jnp.asarray(forget_gate_bias, self.dtype)
      forget_logits = self.W_forget_gate(x) + forget_gate_bias
    M_out, forget_gate = _update_bam_matrix(
        M_in, dM, cfg.bam_lambda_decay, forget_logits)
    if self._force_activation_dtype:
      assert M_out.dtype == self.dtype, (M_out.dtype, self.dtype)
    return M_out, gate, forget_gate

  def _compress_full_fetch_state(self, Mh):
    """Project only the historical full-read cache view; keep cross-layer M full."""
    fetch_state = Mh
    if self._abs_v_dim is not None:
      with jax.named_scope("bam/compress_abs_v_cache"):
        projection = self.abs_v_cache_projection.astype(fetch_state.dtype)
        if self._abs_v_source_implementation == 'dot':
          fetch_state = jnp.einsum('bskv,vc->bskc', fetch_state, projection)
        else:
          fetch_state = jnp.sum(
              fetch_state[..., None] * projection[None, None, None, :, :],
              axis=-2)
    if self._abs_k_dim is not None:
      with jax.named_scope("bam/compress_abs_k_cache"):
        projection = self.abs_k_cache_projection.astype(fetch_state.dtype)
        if self._abs_v_source_implementation == 'dot':
          fetch_state = jnp.einsum('bskc,kp->bspc', fetch_state, projection)
        else:
          fetch_state = jnp.sum(
              fetch_state[..., :, None, :]
              * projection[None, None, :, :, None], axis=-3)
    return fetch_state

  def _expand_full_read(self, full_read):
    """Restore compressed read sides and place them in one attention head."""
    read_k_dim = self._abs_k_dim or self.bam_k
    y_k, y_v = jnp.split(full_read, [read_k_dim], axis=-1)

    def decode(y, decoder):
      decoder = decoder.astype(y.dtype)
      if self._read_implementation == 'dot_bnt':
        return jnp.einsum('bntc,ncd->bntd', y, decoder)
      return jnp.einsum('btnc,ncd->btnd', y, decoder)

    if self._abs_k_dim is not None:
      if self._abs_k_col_output == 'project':
        y_k = decode(y_k, self.abs_k_col_decoder)
      else:
        y_k = jnp.pad(
            y_k, [(0, 0)] * (y_k.ndim - 1)
            + [(0, self.bam_k - self._abs_k_dim)])
    if self._abs_v_dim is not None:
      if self._abs_v_row_output == 'project':
        y_v = decode(y_v, self.abs_v_row_decoder)
      # Direct mode keeps only the compressed coordinates. Padding the complete
      # [K-side, V-side] result below is equivalent when k+v==head_dim and also
      # supports k+v>head_dim without an unnecessary fetched-read decoder.
    return _fit_bam_read_to_head(
        jnp.concatenate((y_k, y_v), axis=-1), self.bam_k, self.head_dim)

  def _query_chunk_op(
      self, query, key, value, decoder_segment_ids, *, with_full_read=False,
      Mh=None, inputs_q=None, is_global=None, window_size_override=None,
      prepared=None):
    """Apply shared chunked attention, optionally consuming alpha for V2 fetch."""
    cfg = self.config
    _, t, _, _ = query.shape
    chunk_size = int(self._query_chunk_size)
    assert t % chunk_size == 0
    if with_full_read and prepared is None:
      assert Mh is not None and inputs_q is not None
      with jax.named_scope("bam/mix_alpha_projection"):
        _, mix_weights = _dynamic_bam_fetch_mix_weights(
            self.fetch_head_mix(inputs_q), query.dtype, 'rms',
            rms_epsilon=self._rms_epsilon)

      fetch_state = self._compress_full_fetch_state(Mh)

      # Parameterized projections must stay outside the runtime L/G cond.  A
      # Linen module call inside lax.cond leaks initialization tracers.
      with jax.named_scope("bam/read_fetched_m"):
        projected_read_key = self._project_full_read_key(inputs_q)
        read_key_kwargs = self._read_key_kwargs('W_R_gate', inputs_q)
        _, _, read_row, read_col = _project_bam_read_keys(
            self._abs_k_dim or self.bam_k,
            inputs_q, lambda _x: projected_read_key,
            **read_key_kwargs)
      prepared = (mix_weights, fetch_state, read_row, read_col)
    elif with_full_read:
      mix_weights, fetch_state, read_row, read_col = prepared

    if is_global is not None:
      local_window = min(t, int(self.sliding_window_size))
      return lax.cond(
          is_global,
          lambda _: self._query_chunk_op(
              query, key, value, decoder_segment_ids,
              with_full_read=with_full_read, Mh=Mh, inputs_q=inputs_q,
              window_size_override=t, prepared=prepared),
          lambda _: self._query_chunk_op(
              query, key, value, decoder_segment_ids,
              with_full_read=with_full_read, Mh=Mh, inputs_q=inputs_q,
              window_size_override=local_window, prepared=prepared),
          operand=None)
    window_size = (
        window_size_override if window_size_override is not None else
        (t if self.sliding_window_size is None
         else min(t, int(self.sliding_window_size))))
    template_target = jnp.arange(t - chunk_size, t)[:, None]
    template_source = jnp.arange(t)[None, :]
    mask_template = template_source <= template_target
    diagonal_template = template_source == template_target
    if window_size < t:
      mask_template &= template_source > template_target - window_size

    y_std_chunks = []
    Mbar_chunks = []
    for q0 in range(0, t, chunk_size):
      q1 = q0 + chunk_size
      s0 = max(0, q0 - window_size) if window_size < t else 0
      s1 = q1
      valid = mask_template[:, t - (s1 - s0):][None]
      if decoder_segment_ids is not None:
        valid = valid & (
            decoder_segment_ids[:, q0:q1, None]
            == decoder_segment_ids[:, None, s0:s1])
      y_std_chunk, alpha = _attention_op(
          query[:, q0:q1], key[:, s0:s1], value[:, s0:s1], valid,
          attn_logits_soft_cap=cfg.attn_logits_soft_cap,
          float32_logits=cfg.float32_logits)
      y_std_chunks.append(y_std_chunk)
      if with_full_read:
        Mbar_chunk, _, _ = _bam_fetch_op(
            alpha, fetch_state[:, s0:s1],
            mix_weights=mix_weights[:, q0:q1],
            diagonal_mask=diagonal_template[:, t - (s1 - s0):])
        Mbar_chunks.append(Mbar_chunk)

    y_std = jnp.concatenate(y_std_chunks, axis=1)
    if with_full_read:
      Mbar = jnp.concatenate(Mbar_chunks, axis=1)
      with jax.named_scope("bam/read_fetched_m"):
        full_read = _contract_bam_read(
            Mbar[:, None], Mbar[:, None], read_row, read_col, True,
            self._read_implementation, self.read_side)
        y_full = self._expand_full_read(full_read)
      return y_std, y_full
    return y_std

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
  ):
    """BAM forward. Returns (out, M_out): out [b,t,emb_dim], M_out [b,t,k,v].

    v0.1 supports train mode and n==n_kv only; prefill/decode error out, deferred to v0.2.
    """
    cfg = self.config
    assert model_mode == common_types.MODEL_MODE_TRAIN, "BamAttention v0.1 supports train mode only"
    assert self.num_query_heads == self.num_kv_heads, "BamAttention v0.1 requires n==n_kv (no GQA)"

    read_gradient_scales = None
    if self._readout_attribution and not self.is_initializing():
      read_gradient_scales = (
          self.perturb('read_col_gradient_scale', jnp.zeros((), jnp.float32)),
          self.perturb('read_row_gradient_scale', jnp.zeros((), jnp.float32)),
      )

    def mask_read_side_gradients(read):
      if read_gradient_scales is None:
        return read
      return _scale_bam_read_side_gradients(
          read, self.bam_k, *read_gradient_scales)

    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)

    # ---- QKV projection (reuse parent) + optional pre-RoPE LocalQK + QKNorm + RoPE ----
    if cfg.fused_qkv:
      query, key, value = self.qkv_projection(inputs_q, proj_name="qkv_proj")
    else:
      query = self.query_projection(inputs_q)
      key = self.kv_projection(inputs_kv, proj_name="key")
      value = self.kv_projection(inputs_kv, proj_name="value")

    Mh = None
    if 'local_qk' in self._mode and self._local_qk_injection == 'pre_qknorm_rope':
      assert M_in is not None, "local_qk read requires M_in"
      with jax.named_scope("bam/normalize_m"):
        Mh = self._matrix_for_read(M_in)
      with jax.named_scope("bam/read_local_m_for_qk"):
        assert Mh is not None, "local_qk read requires M_in"
        q_local, k_local = self._read_local_qk(Mh, inputs_q)
        q_local = mask_read_side_gradients(q_local)
        k_local = mask_read_side_gradients(k_local)
        query = query + q_local
        key = key + k_local

    query, key = dc.QKNorm(cfg, name='qk_norm')(query, key)
    if not self._mha_control and self._local_qk_rope_pairing == 'adjacent':
      query = self._apply_adjacent_rope(query, inputs_positions, name='query_rotary')
      key = self._apply_adjacent_rope(key, inputs_positions, name='key_rotary')
    else:
      query = self.apply_rotary_embedding(query, inputs_positions, name="query_rotary")
      key = self.apply_rotary_embedding(key, inputs_positions, name="key_rotary")

    if self._mha_control:
      query = nn.with_logical_constraint(query, self.query_axis_names)
      key = nn.with_logical_constraint(key, self.key_axis_names)
      value = nn.with_logical_constraint(value, self.value_axis_names)
      query = query / jnp.sqrt(self.head_dim).astype(self.dtype)
      if cfg.float32_qk_product:
        query = query.astype(jnp.float32)
        key = key.astype(jnp.float32)
      if self._query_chunk_size is not None:
        y_std = self._query_chunk_op(
            query, key, value, decoder_segment_ids, is_global=is_global)
      else:
        attn_mask = self.attention_op.generate_attention_mask(
            query, key, decoder_segment_ids, model_mode)
        valid = None
        if attn_mask is not None:
          attn_mask = jnp.squeeze(attn_mask, axis=2)
          valid = jnp.squeeze(
              attn_mask >= DEFAULT_MASK_VALUE * 0.5, axis=1)
        y_std, _ = _attention_op(
            query, key, value, valid,
            attn_logits_soft_cap=cfg.attn_logits_soft_cap,
            float32_logits=cfg.float32_logits)
      y_std = nn.with_logical_constraint(y_std, self.out_axis_names)
      return self.out_projection(inputs_q.shape[-1], y_std), M_in

    fetch_query = fetch_key = None
    if cfg.bam_dedicated_fetch:
      fetch_query = self.fetch_query(inputs_q)
      fetch_key = self.fetch_key(inputs_kv)
      fetch_query, fetch_key = dc.QKNorm(cfg, name='fetch_qk_norm')(fetch_query, fetch_key)
      fetch_query = self.apply_rotary_embedding(
          fetch_query, inputs_positions, name='fetch_query_rotary')
      fetch_key = self.apply_rotary_embedding(
          fetch_key, inputs_positions, name='fetch_key_rotary')
    # Diagnostics retain only the tensors needed to reconstruct route/read behavior outside
    # this module. No statistics are computed here; the collection is mutable only in the
    # standalone diagnostic runner, so normal train/eval executions pay no output cost.
    query_std, key_std = query, key

    # ---- Read-side normalization (§4.6.5 read-side whole-matrix one scalar): one RMS scalar per token over
    # (k,v); all read paths consume Mh. No-op when M_in is all zero (start). Write side keeps
    # bare accumulation on raw M_in. Param-free, pre-LN-style matrix-stream shift. ----
    if Mh is None:
      with jax.named_scope("bam/normalize_m"):
        Mh = self._matrix_for_read(M_in)

    # ---- post-RoPE local_qk (historical route branch, before alpha) ----
    if 'local_qk' in self._mode and self._local_qk_injection == 'post_rope':  # V1 default
      with jax.named_scope("bam/read_local_m_for_qk"):
        assert Mh is not None, "local_qk read requires M_in"
        q_local, k_local = self._read_local_qk(Mh, inputs_q)
        q_local = mask_read_side_gradients(q_local)
        k_local = mask_read_side_gradients(k_local)
        query = query + q_local
        key = key + k_local
    if 'local_v' in self._mode:
      with jax.named_scope("bam/read_local_m_for_v"):
        assert Mh is not None, "local_v read requires M_in"
        value = value + self._read_local_v(Mh, inputs_kv)
    query_route, key_route = query, key

    query = nn.with_logical_constraint(query, self.query_axis_names)
    key = nn.with_logical_constraint(key, self.key_axis_names)
    value = nn.with_logical_constraint(value, self.value_axis_names)

    # ---- alpha = softmax(QK/sqrt(d) + mask) (full sequence in train, n==n_kv) ----
    query = query / jnp.sqrt(self.head_dim).astype(self.dtype)
    if cfg.float32_qk_product:
      query = query.astype(jnp.float32); key = key.astype(jnp.float32)
    if self._query_chunk_size is not None:
      assert cfg.bam_write_source == 'std+cross+local_o'
      if 'full' in self._mode:
        y_std, y_full = self._query_chunk_op(
            query, key, value, decoder_segment_ids, with_full_read=True,
            Mh=Mh, inputs_q=inputs_q, is_global=is_global)
        o_head = y_std + y_full
      else:
        y_std = self._query_chunk_op(
            query, key, value, decoder_segment_ids, is_global=is_global)
        o_head = y_std
      with jax.named_scope("bam/write_m"):
        M_out, _, _ = self._write(o_head, inputs_q, M_in)
      out = nn.with_logical_constraint(o_head, self.out_axis_names)
      return self.out_projection(inputs_q.shape[-1], out), M_out

    attn_mask = self.attention_op.generate_attention_mask(query, key, decoder_segment_ids, model_mode)
    valid = None
    if attn_mask is not None:
      # AttentionOp keeps the GQA group axis in its mask because the standard
      # qk_product has shape [b, n_kv, groups, t, s]. BAM v0.1 requires
      # n == n_kv and collapses that singleton group axis in logits, so collapse
      # the matching mask axis as well; otherwise broadcasting creates a bogus
      # five-dimensional alpha tensor [b, b, n, t, s].
      attn_mask = jnp.squeeze(attn_mask, axis=2)              # [b,1,t,s]
      valid = jnp.squeeze(
          attn_mask >= DEFAULT_MASK_VALUE * 0.5, axis=1)      # [b,t,s]
    y_std, alpha = _attention_op(
        query, key, value, valid,
        attn_logits_soft_cap=cfg.attn_logits_soft_cap,
        float32_logits=cfg.float32_logits)
    # Diagonal yield (§4.6.5): when local_o is on, the softmax-fetch patterns (full/codebook) zero
    # their alpha diagonal without renormalizing — the leftover mass 1−α_tt becomes a soft
    # "abstain-from-self" gate (attention-sink style, zero-param). sparse (block granularity —
    # masking self hits block neighbors) and prefix (inclusive scan — "contains self" is part of
    # the LA exact subfamily) are untouched; local_o itself is fetch-identity (α=δ); when local_o
    # is off the alpha diagonal is the local_o quota-squeeze approximation and must stay.
    fetch_alpha = None
    fetch_mix_logits = None
    fetch_mix_weights = None
    fetch_alpha_pre_diagonal = None
    diagonal_yield = 'local_o' in self._mode and not cfg.bam_keep_fetch_diagonal
    if {'full', 'codebook'} & self._mode:  # V1 default
      with jax.named_scope("bam/mix_alpha"):
        if cfg.bam_dedicated_fetch:
          fetch_query = fetch_query / jnp.sqrt(self.head_dim).astype(self.dtype)
          if cfg.float32_qk_product:
            fetch_query = fetch_query.astype(jnp.float32)
            fetch_key = fetch_key.astype(jnp.float32)
          fetch_logits = jnp.einsum('btfd,bsfd->bfts', fetch_query, fetch_key)
          if cfg.attn_logits_soft_cap:
            fetch_logits = jnp.tanh(fetch_logits / cfg.attn_logits_soft_cap) * cfg.attn_logits_soft_cap
          if attn_mask is not None:
            fetch_logits = apply_mask_to_logits(fetch_logits, attn_mask)
          if cfg.float32_logits:
            fetch_logits = fetch_logits.astype(jnp.float32)
          fetch_alpha = jax.nn.softmax(fetch_logits, axis=-1)
          if diagonal_yield:
            fetch_alpha = fetch_alpha * (
                1 - jnp.eye(fetch_alpha.shape[-2], fetch_alpha.shape[-1], dtype=fetch_alpha.dtype))
        elif self._shared_fetch_mode in ('dynamic_mix', 'dynamic_rms_mix'):  # V1 default
          dynamic_fetch = _dynamic_mixed_bam_fetch_alpha(
              alpha, self.fetch_head_mix(inputs_q), diagonal_yield,
              'rms' if self._shared_fetch_mode == 'dynamic_rms_mix' else 'softmax',
              rms_epsilon=self._rms_epsilon,
              return_aux=cfg.bam_diagnostics)
          if cfg.bam_diagnostics:
            (fetch_alpha, fetch_mix_logits, fetch_mix_weights,
             fetch_alpha_pre_diagonal) = dynamic_fetch
          else:
            fetch_alpha = dynamic_fetch
        else:
          fetch_alpha = _shared_bam_fetch_alpha(
              alpha, query, key, attn_mask, cfg.bam_n_f, self._shared_fetch_mode,
              diagonal_yield, cfg.attn_logits_soft_cap, cfg.float32_logits)
        if self._fetch_sliding_window_size is not None:
          fetch_alpha = _sliding_window_bam_fetch_alpha(
              fetch_alpha, self._fetch_sliding_window_size,
              self._fetch_sliding_window_prefix_size, inputs_positions)
        if self._fetch_diagonal_one:  # V1 default
          diagonal = jnp.arange(min(fetch_alpha.shape[-2:]))
          fetch_alpha = fetch_alpha.at[..., diagonal, diagonal].set(
              jnp.asarray(1, dtype=fetch_alpha.dtype))

    # ---- BAM read (all modes share the unified bam_read primitive, §4.6.1) ----
    y_codebook = jnp.zeros_like(y_std)
    y_full = jnp.zeros_like(y_std)
    y_local_o = jnp.zeros_like(y_std)
    Mbar = None
    y_bam = 0.0
    capture_read_key_stages = cfg.bam_diagnostics and not self.is_initializing()
    read_key_stages = {}
    if 'codebook' in self._mode:
      assert Mh is not None, "codebook read requires M_in"
      y_codebook = codebook_read(
          fetch_alpha, Mh, inputs_q, self.rho_u, self.rho_v, self.W_beta,
          **self._read_key_kwargs('W_beta_gate', inputs_q, squeeze_fetch_axis=True),
          source_implementation=self._codebook_source_implementation,
          read_implementation=self._codebook_read_implementation)
      y_bam = y_codebook
    if 'full' in self._mode:  # V1 default
      assert Mh is not None, "full read requires M_in"
      with jax.named_scope("bam/fetch_m"):
        fetch_state = self._compress_full_fetch_state(Mh)
        if self._fetch_temporal_block_size is not None:
          Mbar_fetch = _temporal_block_bam_fetch(
              fetch_alpha, fetch_state, inputs_positions, decoder_segment_ids,
              self._fetch_temporal_block_size, self._fetch_temporal_block_mode,
              self._fetch_temporal_recent_window_size)
        else:
          Mbar_fetch, _, _ = _bam_fetch_op(fetch_alpha, fetch_state)
        Mbar = Mbar_fetch
        if self._combine_full_local_read:
          # The shared runtime key makes Read linear in M.  Because fetch_alpha has
          # yielded its diagonal, adding normalized local Mh here exactly replaces
          # the separate local_o read with a fixed local coefficient of one.
          Mbar = Mbar + fetch_state[:, None]
      with jax.named_scope("bam/read_fetched_m"):
        full_read_projection = self._project_full_read_key
        full_read_kwargs = self._read_key_kwargs('W_R_gate', inputs_q)
        if self._squeeze_single_fetch_read:
          Mbar = jnp.squeeze(Mbar, axis=1)
          full_read_projection = lambda x: jnp.squeeze(
              self._project_full_read_key(x), axis=-2)
          full_read_kwargs = self._read_key_kwargs(
              'W_R_gate', inputs_q, squeeze_fetch_axis=True)
        full_read = bam_read(
            Mbar, inputs_q, full_read_projection, None,
            **full_read_kwargs, return_key_stages=capture_read_key_stages,
            implementation=self._read_implementation, read_side=self.read_side)
        if capture_read_key_stages:
          full_read, full_key_stages = full_read
          read_key_stages.update({
              f"read_key_W_R_{stage}": key for stage, key in full_key_stages.items()})
        full_read = self._expand_full_read(full_read)
        y_full = (rearrange(full_read, 'b n t d -> b t n d')
                  if self._read_implementation == 'dot_bnt' else full_read)
        y_full = mask_read_side_gradients(y_full)
        y_bam = y_bam + y_full
    if 'local_o' in self._mode and not self._combine_full_local_read:
      assert Mh is not None, "local_o read requires M_in"
      if self._share_full_local_read:
        local_read_projection = lambda x: jnp.squeeze(
            self._project_full_read_key(x), axis=-2)
        local_read_key_kwargs = self._read_key_kwargs(
            'W_R_gate', inputs_q, squeeze_fetch_axis=True)
      else:
        local_read_projection = self.W_Ro
        local_read_key_kwargs = self._read_key_kwargs('W_Ro_gate', inputs_q)
      local_o_read = bam_read(
          Mh, inputs_q, local_read_projection, None,
          **local_read_key_kwargs,
          return_key_stages=capture_read_key_stages,
          implementation=self._read_implementation, read_side=self.read_side)
      if capture_read_key_stages:
        local_o_read, local_o_key_stages = local_o_read
        read_key_stages.update({
            f"read_key_W_Ro_{stage}": key for stage, key in local_o_key_stages.items()})
      y_local_o = (rearrange(local_o_read, 'b n t d -> b t n d')
                   if self._read_implementation == 'dot_bnt' else local_o_read)
      y_bam = y_bam + y_local_o

    o_head = y_std + y_bam                                  # [b,t,n,d]

    # Select only the direct source of the matrix-stream U write. The residual-stream
    # output above remains unchanged, so these arms isolate read-to-write recirculation.
    write_o_head = _select_bam_write_source(
        cfg.bam_write_source, y_std, y_codebook, y_full, y_local_o, o_head)

    # ---- Write primitive (update BAM state first, then project to the residual stream) ----
    if self._has_write:  # V1 default
      assert M_in is not None, "write primitive requires M_in"
      with jax.named_scope("bam/write_m"):
        M_out, write_gate, forget_gate = self._write(write_o_head, inputs_q, M_in)
    else:
      M_out = M_in
      write_gate = None
      forget_gate = None

    if self._readout_attribution and not self.is_initializing():
      query_indices = _bam_readout_query_indices(inputs_q.shape[1])
      if fetch_alpha is not None:
        alpha_rows = jnp.squeeze(fetch_alpha, axis=1)[:, query_indices]
        # Signed dynamic mixing is materially more diffuse than one MHA head;
        # keep enough support for the P2 reconstruction and record the exact
        # 99%-absolute-mass support needed at every sampled read site.
        top_count = min(1536, alpha_rows.shape[-1])
        _, source_indices = jax.lax.top_k(jnp.abs(alpha_rows), top_count)
        source_weights = jnp.take_along_axis(
            alpha_rows, source_indices, axis=-1)
        sorted_abs = jnp.sort(jnp.abs(alpha_rows), axis=-1)[..., ::-1]
        total_abs = jnp.sum(sorted_abs, axis=-1)
        support_99 = 1 + jnp.sum(
            jnp.cumsum(sorted_abs, axis=-1)
            < 0.99 * total_abs[..., None], axis=-1)
        self.sow('bam_readout', 'fetch_source_indices', source_indices)
        self.sow('bam_readout', 'fetch_source_weights', source_weights)
        self.sow('bam_readout', 'fetch_support_99_count', support_99)
        self.sow(
            'bam_readout', 'fetch_retained_abs_mass',
            jnp.sum(jnp.abs(source_weights), axis=-1)
            / jnp.maximum(jnp.sum(jnp.abs(alpha_rows), axis=-1), 1e-12))
      if 'read_key_W_R_post_gate' in read_key_stages:
        full_key = read_key_stages['read_key_W_R_post_gate']
        if full_key.ndim == 5:
          full_key = jnp.squeeze(full_key, axis=-2)
        self.sow(
            'bam_readout', 'full_post_gate_key',
            full_key[:, query_indices])
      self.sow('bam_readout', 'y_full', y_full[:, query_indices])
    elif cfg.bam_diagnostics and not self.is_initializing():
      # Raw diagnostic interface only. Keep policy/statistics in bam_diagnostics.py so new
      # questions do not accumulate analysis logic in the production attention path.
      raw_tensors = {
          "M_in": M_in,
          "M_out": M_out,
          "query_std": query_std,
          "key_std": key_std,
          "query_route": query_route,
          "key_route": key_route,
          "y_std": y_std,
          "y_codebook": y_codebook,
          "y_full": y_full,
          "y_local_o": y_local_o,
      }
      if fetch_alpha is not None:
        raw_tensors["fetch_alpha"] = fetch_alpha
      if fetch_alpha_pre_diagonal is not None:
        raw_tensors["fetch_alpha_pre_diagonal"] = fetch_alpha_pre_diagonal
      if fetch_mix_logits is not None:
        raw_tensors["fetch_mix_logits"] = fetch_mix_logits
      if fetch_mix_weights is not None:
        raw_tensors["fetch_mix_weights"] = fetch_mix_weights
      if Mbar_fetch is not None:
        raw_tensors["Mbar_fetch"] = Mbar_fetch
      if Mbar is not None:
        raw_tensors["Mbar"] = Mbar
      if write_gate is not None:
        raw_tensors["write_gate"] = write_gate
      if forget_gate is not None:
        raw_tensors["forget_gate"] = forget_gate
      raw_tensors.update(read_key_stages)
      for raw_name, raw_value in raw_tensors.items():
        self.sow("bam_raw", raw_name, raw_value)

    out = nn.with_logical_constraint(o_head, self.out_axis_names)
    out = self.out_projection(inputs_q.shape[-1], out)

    return out, M_out
