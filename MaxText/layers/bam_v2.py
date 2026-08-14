# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal BAM V2 attention path.

This module is reduced from ``tmp/maxtext`` commit ``1afd9425``, specifically
``layers/attentions.py``'s ``BamAttention`` as reached by
``exp.BamLlama2MediumV2``.  Branches that V2 cannot execute (ablation modes,
codebook/local-v/local-o reads, dedicated/temporal/window fetch, learned
grouped norms, dynamic forgetting, alternate write forms and diagnostics) are
intentionally absent.  Parameter names, shapes and initializer ordering remain
the same as that reference path so checkpoints and deterministic alignment are
reviewable.
"""

import math

from flax import linen as nn
import jax
import jax.numpy as jnp

import common_types
from layers import attentions
from layers import dc
from layers import initializers
from layers import linears
from layers import normalizations


Array = common_types.Array
DenseGeneral = linears.DenseGeneral


def _rms_norm(x, *, dtype, epsilon, axis=-1, statistics_dtype=None):
  """Reference param-free RMS helper absent from this older target baseline."""
  statistics_dtype = statistics_dtype or dtype
  statistics = jnp.asarray(x, statistics_dtype)
  mean_square = jnp.mean(jnp.square(statistics), axis=axis, keepdims=True)
  return jnp.asarray(statistics * jax.lax.rsqrt(mean_square + epsilon), dtype)


def _rms_gated_key(raw_key, gate_logits, *, split_at, scale, epsilon, dtype):
  """Reference ``_project_bam_read_keys`` + ``rms_gate`` specialization."""
  raw_row, raw_col = jnp.split(raw_key, [split_at], axis=-1)
  row_gate, col_gate = jnp.split(gate_logits, 2, axis=-1)

  def transform(key, gate):
    direction = _rms_norm(key, dtype=key.dtype, epsilon=epsilon, statistics_dtype=jnp.float32)
    return jnp.asarray(scale, key.dtype) * jax.nn.sigmoid(gate) * direction

  return transform(raw_row, row_gate), transform(raw_col, col_gate)


def _factorized_local_read(
    matrix, raw_key, gate_logits, raw_head_mix, *, bam_k, scale, epsilon, output_dtype
):
  """Reference ``factorized_head_bam_read`` fixed to V2's both/BTN/mul path."""
  row_key, col_key = _rms_gated_key(
      raw_key,
      gate_logits,
      split_at=bam_k,
      scale=scale,
      epsilon=epsilon,
      dtype=output_dtype,
  )
  y_u = jnp.sum(matrix * col_key[..., None, :], axis=-1)
  y_v = jnp.sum(matrix * row_key[..., :, None], axis=-2)
  head_mix = _rms_norm(raw_head_mix, dtype=y_u.dtype, epsilon=epsilon, axis=-2)
  row_mix, col_mix = head_mix[..., 0], head_mix[..., 1]
  y_u = jnp.einsum("btk,btn->btnk", y_u, col_mix)
  y_v = jnp.einsum("btv,btn->btnv", y_v, row_mix)
  return jnp.concatenate((y_u, y_v), axis=-1)


def _packed_local_qk_init(kernel_init, num_query_heads, num_kv_heads, key_width):
  """Reference packed initializer, including its two-key PRNG consumption."""
  query_mix_width = 2 * num_query_heads
  kv_mix_width = 2 * num_kv_heads
  packed_width = 2 * (key_width + 2) + query_mix_width + kv_mix_width

  def init_fn(key, shape, dtype, _in_axis=0, _out_axis=1):
    if len(shape) != 2 or shape[-1] != packed_width:
      raise ValueError(f"packed local-QK kernel expects [embed,{packed_width}], got {shape}")
    q_mix_key, k_mix_key = jax.random.split(key)
    zeros = lambda width: jnp.zeros((shape[0], width), dtype)
    q_mix_shape = (shape[0], num_query_heads, 2)
    kv_mix_shape = (shape[0], num_kv_heads, 2)
    q_mix = kernel_init(q_mix_key, q_mix_shape, dtype, 0, (1, 2)).reshape(shape[0], query_mix_width)
    k_mix = kernel_init(k_mix_key, kv_mix_shape, dtype, 0, (1, 2)).reshape(shape[0], kv_mix_width)
    return jnp.concatenate(
        (zeros(key_width), zeros(2), q_mix, zeros(key_width), zeros(2), k_mix), axis=-1
    )

  return init_fn


class BamV2Attention(attentions.Attention):
  """The exact BAM attention variant selected by ``BamLlama2MediumV2``."""

  bam_k: int = 32
  bam_v: int = 32

  def setup(self):
    super().setup()
    cfg = self.config
    if self.bam_k + self.bam_v != self.head_dim:
      raise ValueError("BAM V2 requires bam_k + bam_v == head_dim")
    if self.num_query_heads % self.num_kv_heads != 0:
      raise ValueError("BAM V2 requires num_query_heads divisible by num_kv_heads")
    if self.use_kv_shift or getattr(cfg, "use_o_shift", False):
      raise ValueError("BAM V2 is not defined with KV/O shift")
    if uses_dcmha := attentions.uses_dcmha_attention(
        cfg, self.sliding_window_size, self.attention_kernel
    ):
      raise ValueError(f"BAM V2 is not defined with DCMHA (uses_dcmha={uses_dcmha})")
    if cfg.bam_adaptation_postnorm and not cfg.bam_adaptation:
      raise ValueError("bam_adaptation_postnorm requires bam_adaptation")

    reg_init = self.kernel_init
    zeros_init = initializers.contant_dense_init(0.0)
    key_width = self.bam_k + self.bam_v
    packed_width = 2 * (key_width + 2) + 2 * (self.num_query_heads + self.num_kv_heads)
    read_gate_init = float(cfg.bam_read_gate_init)
    gate_bias = math.log(read_gate_init / (1.0 - read_gate_init))

    # Source: absolute-V full-read branch.  The direct decoder is deliberately
    # created but unused: the reference V2 does this to preserve its checkpoint tree.
    self.abs_v_cache_projection = self.param(
        "abs_v_cache_projection",
        nn.with_logical_partitioning(nn.initializers.orthogonal(), ("v_factor", "kv")),
        (self.bam_v, cfg.bam_abs_v_compression_dim),
        self.weight_dtype,
    )

    def decoder_init(_key, shape, dtype):
      return jnp.broadcast_to(jnp.eye(shape[-2], shape[-1], dtype=dtype), shape)

    self.abs_v_row_decoder = self.param(
        "abs_v_row_decoder",
        nn.with_logical_partitioning(decoder_init, ("q_heads", "kv", "v_factor")),
        (self.num_query_heads, cfg.bam_abs_v_compression_dim, self.bam_v),
        self.weight_dtype,
    )
    read_width = self.bam_k + cfg.bam_abs_v_compression_dim
    self.W_R = DenseGeneral(
        features=(self.num_query_heads, 1, read_width),
        axis=-1,
        kernel_init=zeros_init,
        kernel_axes=("embed", "q_heads", "fetch", "kv"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="W_R",
        quant=self.quant,
        matmul_precision=cfg.matmul_precision,
        use_bias=False,
    )
    self.W_R_gate = DenseGeneral(
        features=(self.num_query_heads, 1, 2),
        axis=-1,
        kernel_init=zeros_init,
        kernel_axes=("embed", "q_heads", "fetch", None),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="W_R_gate",
        quant=self.quant,
        matmul_precision=cfg.matmul_precision,
        use_bias=False,
    )
    self.W_R_gate_b0 = self.param(
        "W_R_gate_b0",
        nn.with_logical_partitioning(
            lambda key, shape, dtype: jnp.full(shape, gate_bias, dtype),
            ("q_heads", "fetch", None),
        ),
        (self.num_query_heads, 1, 2),
        self.weight_dtype,
    )
    self.fetch_head_mix = DenseGeneral(
        features=self.num_query_heads,
        axis=-1,
        kernel_init=reg_init,
        kernel_axes=("embed", "q_heads"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="fetch_head_mix",
        quant=self.quant,
        matmul_precision=cfg.matmul_precision,
        use_bias=True,
    )

    # Source: BamAttention.setup(), packed factorized LocalQK branch.  It is
    # declared after full-read/fetch exactly as in the reference initializer.
    self.W_local_qk_packed = DenseGeneral(
        features=packed_width,
        axis=-1,
        kernel_init=_packed_local_qk_init(
            reg_init, self.num_query_heads, self.num_kv_heads, key_width
        ),
        kernel_axes=("embed", None),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="W_local_qk_packed",
        quant=self.quant,
        matmul_precision=cfg.matmul_precision,
        use_bias=False,
    )
    for name in ("W_lq", "W_lk"):
      setattr(
          self,
          f"{name}_bias",
          self.param(
              f"{name}_bias",
              nn.with_logical_partitioning(zeros_init, ("kv",)),
              (key_width,),
              self.weight_dtype,
          ),
      )
      setattr(
          self,
          f"{name}_gate_b0",
          self.param(
              f"{name}_gate_b0",
              nn.with_logical_partitioning(
                  lambda key, shape, dtype: jnp.full(shape, gate_bias, dtype), (None,)
              ),
              (2,),
              self.weight_dtype,
          ),
      )

    # Adaptation-only learned U factor.  Source: the reference BamAttention
    # ``bam_write_u_proj`` branch, collapsed behind one V2 capability flag.
    # Keeping this conditional preserves the exact default parameter tree.
    if cfg.bam_adaptation:
      def write_u_init(key, shape, dtype):
        return reg_init(key, shape, dtype, 1, 2)

      self.P_agg_u = self.param(
          "P_agg_u",
          nn.with_logical_partitioning(write_u_init, ("q_heads", "embed", "v_factor")),
          (self.num_query_heads, self.head_dim, self.bam_k),
          self.weight_dtype,
      )

    # Source: Direct P_loc r=256 GELU x_bias write, without unused projected-U.
    self.P_loc_down = DenseGeneral(
        features=cfg.bam_write_v_bottleneck_dim,
        axis=-1,
        kernel_init=reg_init,
        kernel_axes=("embed", None),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="P_loc_down",
        quant=self.quant,
        matmul_precision=cfg.matmul_precision,
        use_bias=False,
    )
    self.P_loc_up = DenseGeneral(
        features=(self.num_query_heads, self.bam_v),
        axis=-1,
        kernel_init=reg_init,
        kernel_axes=("embed", "q_heads", "v_factor"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="P_loc_up",
        quant=self.quant,
        matmul_precision=cfg.matmul_precision,
        use_bias=True,
    )
    self.W_gw = DenseGeneral(
        features=(self.num_query_heads,),
        axis=-1,
        kernel_init=reg_init,
        kernel_axes=("embed", "q_heads"),
        dtype=self.dtype,
        weight_dtype=self.weight_dtype,
        name="W_gw",
        quant=self.quant,
        matmul_precision=cfg.matmul_precision,
        use_bias=False,
    )
    write_eps = float(cfg.bam_write_eps)
    write_bias = math.log(write_eps / (1.0 - write_eps))
    self.gw_b0 = self.param(
        "gw_b0",
        nn.with_logical_partitioning(
            lambda key, shape, dtype: jnp.full(shape, write_bias, dtype), ("q_heads",)
        ),
        (self.num_query_heads,),
        self.weight_dtype,
    )

  def _local_qk(self, matrix, inputs):
    packed = self.W_local_qk_packed(inputs)
    key_width = self.bam_k + self.bam_v
    query_mix_width = 2 * self.num_query_heads
    kv_mix_width = 2 * self.num_kv_heads
    split_points = (
        key_width,
        key_width + 2,
        key_width + 2 + query_mix_width,
        2 * key_width + 2 + query_mix_width,
        2 * key_width + 4 + query_mix_width,
    )
    q_key, q_gate, q_mix, k_key, k_gate, k_mix = jnp.split(packed, split_points, axis=-1)
    q_mix = q_mix.reshape(q_mix.shape[:-1] + (self.num_query_heads, 2))
    k_mix = k_mix.reshape(k_mix.shape[:-1] + (self.num_kv_heads, 2))
    q_key = q_key + jnp.asarray(self.W_lq_bias, q_key.dtype)
    k_key = k_key + jnp.asarray(self.W_lk_bias, k_key.dtype)
    q_gate = q_gate + jnp.asarray(self.W_lq_gate_b0, q_gate.dtype)
    k_gate = k_gate + jnp.asarray(self.W_lk_gate_b0, k_gate.dtype)
    kwargs = dict(
        bam_k=self.bam_k,
        scale=float(self.config.bam_read_key_scale),
        epsilon=float(self.config.bam_read_key_epsilon),
        output_dtype=self.dtype,
    )
    return (
        _factorized_local_read(matrix, q_key, q_gate, q_mix, **kwargs),
        _factorized_local_read(matrix, k_key, k_gate, k_mix, **kwargs),
    )

  def _full_read(self, matrix, alpha, inputs):
    mix_logits = jnp.asarray(self.fetch_head_mix(inputs), jnp.float32)
    mix_weights = _rms_norm(
        mix_logits,
        dtype=alpha.dtype,
        epsilon=float(self.config.normalization_layer_epsilon),
    ) / jnp.sqrt(self.num_query_heads)
    fetch_alpha = jnp.einsum("bnts,btn->bts", alpha, mix_weights)[:, None]
    diagonal = jnp.arange(min(fetch_alpha.shape[-2:]))
    fetch_alpha = fetch_alpha.at[..., diagonal, diagonal].set(jnp.asarray(1, fetch_alpha.dtype))

    projection = jnp.asarray(self.abs_v_cache_projection, matrix.dtype)
    fetch_state = jnp.einsum("bskv,vc->bskc", matrix, projection)
    fetched = jnp.einsum("bfts,bskv->bftkv", fetch_alpha, fetch_state)
    raw_key = self.W_R(inputs)
    gate_logits = self.W_R_gate(inputs) + jnp.asarray(self.W_R_gate_b0, self.dtype)
    row_key, col_key = _rms_gated_key(
        raw_key,
        gate_logits,
        split_at=self.bam_k,
        scale=float(self.config.bam_read_key_scale),
        epsilon=float(self.config.bam_read_key_epsilon),
        dtype=self.dtype,
    )
    # Reference ``_contract_bam_read(..., mul_reduce_btn)`` with one fetch axis.
    fetched_btn = jnp.transpose(fetched, (0, 2, 1, 3, 4))
    y_u = jnp.sum(fetched_btn[:, :, None] * col_key[..., None, :], axis=(-3, -1))
    y_v = jnp.sum(fetched_btn[:, :, None] * row_key[..., :, None], axis=(-3, -2))
    if self.config.bam_adaptation:
      # The decoder already exists in the default checkpoint tree.  Adaptation
      # executes it as a per-head C->V linear projection instead of padding the
      # compressed row read with zeros.
      decoder = jnp.asarray(self.abs_v_row_decoder, y_v.dtype)
      y_v = jnp.einsum("btnc,ncv->btnv", y_v, decoder)
      return jnp.concatenate((y_u, y_v), axis=-1)
    full_read = jnp.concatenate((y_u, y_v), axis=-1)
    return jnp.pad(full_read, ((0, 0), (0, 0), (0, 0), (0, self.bam_v - y_v.shape[-1])))

  def _write(self, o_head, inputs, matrix):
    if self.config.bam_adaptation:
      projection = jnp.asarray(self.P_agg_u, o_head.dtype)
      u1 = jnp.einsum("btnd,ndk->btnk", o_head, projection)
    else:
      u1 = o_head[..., :self.bam_k]
    u2 = self.P_loc_up(nn.gelu(self.P_loc_down(inputs)))
    gate = jax.nn.sigmoid(self.W_gw(inputs) + jnp.asarray(self.gw_b0, self.dtype))
    u1 = _rms_norm(
        u1, dtype=self.dtype, epsilon=float(self.config.normalization_layer_epsilon), statistics_dtype=jnp.float32
    )
    u2 = _rms_norm(
        u2, dtype=self.dtype, epsilon=float(self.config.normalization_layer_epsilon), statistics_dtype=jnp.float32
    )
    delta = jnp.sum((gate[..., None] * u1)[..., None] * u2[..., None, :], axis=-3)
    return jnp.asarray(self.config.bam_lambda_decay, matrix.dtype) * matrix + delta

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
  ):
    """Run the V2 train-only forward and return ``(residual_output, M_out)``."""
    del decoder_input_tokens, deterministic, eos_sum, deep_embedding
    if model_mode != common_types.MODEL_MODE_TRAIN:
      raise ValueError("BAM V2 supports training mode only")
    if M_in is None:
      raise ValueError("BAM V2 requires an incoming matrix stream")
    if M_in.dtype != self.dtype:
      raise TypeError(f"BAM V2 matrix dtype {M_in.dtype} must equal activation dtype {self.dtype}")

    inputs_q = nn.with_logical_constraint(inputs_q, self.input_axis_names)
    inputs_kv = nn.with_logical_constraint(inputs_kv, self.input_axis_names)
    if self.config.fused_qkv:
      query, key, value = self.qkv_projection(inputs_q, proj_name="qkv_proj")
    else:
      query = self.query_projection(inputs_q)
      key = self.kv_projection(inputs_kv, proj_name="key")
      value = self.kv_projection(inputs_kv, proj_name="value")
    query, key = dc.QKNorm(self.config, name="qk_norm")(query, key)
    query = self.apply_rotary_embedding(query, inputs_positions, name="query_rotary")
    key = self.apply_rotary_embedding(key, inputs_positions, name="key_rotary")
    q_local, k_local = self._local_qk(M_in, inputs_q)
    if self.config.bam_adaptation_postnorm:
      norm_kwargs = dict(
          direct_scale=True,
          scale_init=nn.initializers.constant(0.001),
      )
      q_local = normalizations.get_rmsnorm(
          "rms_norm_q", self.config, **norm_kwargs
      )(q_local)
      k_local = normalizations.get_rmsnorm(
          "rms_norm_k", self.config, **norm_kwargs
      )(k_local)
    query, key = query + q_local, key + k_local
    query = nn.with_logical_constraint(query, self.query_axis_names)
    key = nn.with_logical_constraint(key, self.key_axis_names)
    value = nn.with_logical_constraint(value, self.value_axis_names)

    query = query / jnp.sqrt(self.head_dim).astype(self.dtype)
    if self.float32_qk_product:
      query, key = query.astype(jnp.float32), key.astype(jnp.float32)
    batch, query_length, query_heads, depth = query.shape
    kv_heads = key.shape[-2]
    query_groups = query_heads // kv_heads
    grouped_query = query.reshape(batch, query_length, kv_heads, query_groups, depth)
    logits = jnp.einsum("btkgd,bskd->bkgts", grouped_query, key)
    logits = logits.reshape(batch, query_heads, query_length, key.shape[1])
    if self.config.attn_logits_soft_cap:
      logits = jnp.tanh(logits / self.config.attn_logits_soft_cap) * self.config.attn_logits_soft_cap
    mask = self.attention_op.generate_attention_mask(query, key, decoder_segment_ids, model_mode)
    if mask is not None:
      logits = attentions.apply_mask_to_logits(logits, jnp.squeeze(mask, axis=2))
    if self.float32_logits:
      logits = logits.astype(jnp.float32)
    alpha = jax.nn.softmax(logits, axis=-1)
    grouped_alpha = alpha.reshape(batch, kv_heads, query_groups, query_length, key.shape[1])
    y_std = jnp.einsum("bkgts,bskd->btkgd", grouped_alpha, value)
    y_std = y_std.reshape(batch, query_length, query_heads, depth)
    y_full = self._full_read(M_in, alpha, inputs_q)
    if self.config.bam_adaptation_postnorm:
      y_full = normalizations.get_rmsnorm(
          "rms_norm_o",
          self.config,
          direct_scale=True,
          scale_init=nn.initializers.constant(0.001),
      )(y_full)
    o_head = y_std + y_full
    M_out = self._write(o_head, inputs_q, M_in)
    out = self.out_projection(inputs_q.shape[-1], nn.with_logical_constraint(o_head, self.out_axis_names))
    return out, M_out
