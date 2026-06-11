from typing import Any, Optional

import jax
from flax import linen as nn
import jax.numpy as jnp
from jax.sharding import Mesh

from layers import initializers
from layers import normalizations
from layers import linears
from layers import quantizations

Quant = quantizations.AqtQuantization
NdInitializer = initializers.NdInitializer
nd_dense_init = initializers.nd_dense_init


def shift_1d(inputs, offset: int, axis: int):
  paddings = [
      ((max(offset, 0), -min(offset, 0)) if i == axis else (0, 0))
      for i in range(len(inputs.shape))
  ]
  input_length = jnp.shape(inputs)[axis]
  padded_inputs = jnp.pad(inputs, paddings, mode='edge')
  if offset > 0:
    output = jax.lax.slice_in_dim(
        padded_inputs, start_index=0, limit_index=input_length, axis=axis
    )
  else:
    output = jax.lax.slice_in_dim(
        padded_inputs,
        start_index=-offset,
        limit_index=input_length - offset,
        axis=axis,
    )
  return output


def gather_by_seq_index(inputs, indices):
  """Gather [B, T, ...] inputs at per-example sequence indices [B, T]."""
  indices = jnp.clip(indices, 0, inputs.shape[1] - 1)
  index_shape = indices.shape + (1,) * (inputs.ndim - indices.ndim)
  indices = jnp.reshape(indices, index_shape)
  indices = jnp.broadcast_to(indices, inputs.shape[:2] + inputs.shape[2:])
  return jnp.take_along_axis(inputs, indices, axis=1)


def _position_to_seq_index_table(positions, token_valid, max_position):
  batch, length = positions.shape
  positions = positions.astype(jnp.int32)
  seq_idx = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32)[None, :], (batch, length))
  batch_idx = jnp.broadcast_to(jnp.arange(batch, dtype=jnp.int32)[:, None], (batch, length))
  in_range = (positions >= 0) & (positions < max_position) & token_valid
  scatter_positions = jnp.where(in_range, positions, max_position)
  table = jnp.full((batch, max_position), -1, dtype=jnp.int32)
  return table.at[batch_idx, scatter_positions].set(seq_idx, mode="drop")


def arc_2d_causal_shift_plan(
    positions,
    decoder_segment_ids=None,
    *,
    row_stride=32,
    grid_size=1024,
    marker_position=16383,
    max_position=16384,
):
  """Build reusable ARC-aware causal 2-D neighbor indices and masks.

  Sources are ordered as previous sequence token, top-left, top, top-right,
  and self. Top-neighbor lookups use ARC grid-aligned position IDs.
  """
  batch, length = positions.shape
  positions = positions.astype(jnp.int32)
  seq_idx = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32)[None, :], (batch, length))
  if decoder_segment_ids is None:
    token_valid = jnp.ones((batch, length), dtype=jnp.bool_)
  else:
    token_valid = decoder_segment_ids > 0

  pos_to_seq = _position_to_seq_index_table(positions, token_valid, max_position)

  def lookup_position(source_positions):
    source_positions = source_positions.astype(jnp.int32)
    in_range = (source_positions >= 0) & (source_positions < max_position)
    clipped = jnp.clip(source_positions, 0, max_position - 1)
    source_idx = jnp.take_along_axis(pos_to_seq, clipped, axis=1)
    valid = in_range & (source_idx >= 0) & (source_idx <= seq_idx)
    return source_idx, valid

  prev_idx = jnp.maximum(seq_idx - 1, 0)
  prev_token_valid = gather_by_seq_index(token_valid[..., None], prev_idx)[..., 0]
  prev_valid = token_valid & (seq_idx > 0) & prev_token_valid

  local_position = positions % grid_size
  row = local_position // row_stride
  col = local_position % row_stride
  target_block = positions // grid_size
  is_grid_token = token_valid & (positions >= 0) & (positions < marker_position)
  top_source_positions = (
      positions - row_stride - 1,
      positions - row_stride,
      positions - row_stride + 1,
  )

  top_indices = []
  top_valids = []
  for source_pos, col_valid in zip(
      top_source_positions,
      (col > 0, jnp.ones_like(col, dtype=jnp.bool_), col + 1 < row_stride),
  ):
    source_idx, source_valid = lookup_position(source_pos)
    same_grid = (source_pos // grid_size) == target_block
    top_indices.append(source_idx)
    top_valids.append(is_grid_token & (row > 0) & col_valid & same_grid & source_valid)

  self_idx = seq_idx
  self_valid = jnp.ones_like(token_valid)
  source_indices = jnp.stack((prev_idx, *top_indices, self_idx), axis=-1)
  source_valid = jnp.stack((prev_valid, *top_valids, self_valid), axis=-1)
  return source_indices, source_valid


def apply_arc_2d_causal_shift(inputs, logits, source_indices, source_valid, softmax=True):
  logits = logits.astype(jnp.float32)
  num_sources = source_indices.shape[-1]
  if softmax:
    masked_logits = jnp.where(source_valid[:, :, None, :], logits, jnp.asarray(-1.0e9, dtype=jnp.float32))
    weights = jax.nn.softmax(masked_logits, axis=-1).astype(inputs.dtype)
    out = jnp.zeros_like(inputs)
    for i in range(num_sources):
      source = inputs if i == num_sources - 1 else gather_by_seq_index(inputs, source_indices[..., i])
      out = out + source * weights[..., i][..., None]
  else:
    weights = jnp.where(source_valid[:, :, None, :], logits, jnp.asarray(0.0, dtype=jnp.float32)).astype(inputs.dtype)
    out = inputs
    for i in range(num_sources):
      source = inputs if i == num_sources - 1 else gather_by_seq_index(inputs, source_indices[..., i])
      out = out + source * weights[..., i][..., None]
  return out


def arc_2d_causal_shift(
    inputs,
    logits,
    positions,
    decoder_segment_ids=None,
    *,
    row_stride=32,
    grid_size=1024,
    marker_position=16383,
    max_position=16384,
):
  """Apply ARC-aware dynamic causal 2-D shift to one projected tensor."""
  source_indices, source_valid = arc_2d_causal_shift_plan(
      positions,
      decoder_segment_ids=decoder_segment_ids,
      row_stride=row_stride,
      grid_size=grid_size,
      marker_position=marker_position,
      max_position=max_position,
  )
  return apply_arc_2d_causal_shift(inputs, logits, source_indices, source_valid)


def arc_2d_causal_shift_kv(
    key,
    value,
    key_logits,
    value_logits,
    positions,
    decoder_segment_ids=None,
    *,
    row_stride=32,
    grid_size=1024,
    marker_position=16383,
    max_position=16384,
    softmax=True,
):
  """Apply ARC-aware dynamic causal 2-D shift to K/V with shared neighbors."""
  source_indices, source_valid = arc_2d_causal_shift_plan(
      positions,
      decoder_segment_ids=decoder_segment_ids,
      row_stride=row_stride,
      grid_size=grid_size,
      marker_position=marker_position,
      max_position=max_position,
  )
  key = apply_arc_2d_causal_shift(key, key_logits, source_indices, source_valid, softmax=softmax)
  value = apply_arc_2d_causal_shift(value, value_logits, source_indices, source_valid, softmax=softmax)
  return key, value


class KVshift(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  num_kv_heads: int = 16
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")
  num_kv_heads: int = None
  
  def setup(self):
    cfg = self.config
    norm_kwargs = {
                "dtype": cfg.dtype,
                "weight_dtype": cfg.weight_dtype,
                "epsilon": cfg.normalization_layer_epsilon,
                }
    if not cfg.kv_shift_skip_knorm:
      self.kv_shift_norm = normalizations.get_rmsnorm("kv_shift_knorm", cfg)
    self.kv_shift_prenorm = normalizations.get_rmsnorm("kv_shift_prenorm", cfg)
    
    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    self.q_shift = cfg.use_q_shift
    self.num_shifts = 2 if not self.q_shift else 3
    self.kv_shift_hidden_way = cfg.kv_shift_hidden_way
    self.kv_shift_mode = getattr(cfg, "kv_shift_mode", "1d")
    self.kv_shift_arc_2d_sources = 5

    if self.kv_shift_mode == "arc_2d":
      for mode in "kv":
        setattr(self, f'dw_proj_{mode}', linears.DenseGeneral(
                                    (self.num_kv_heads, self.kv_shift_arc_2d_sources),
                                    kernel_init=initializers.contant_dense_init(0.0),
                                    kernel_axes=('embed', "kv_heads", "kv_shift_sources"),
                                    use_bias=False,
                                    name=f'kv_shift_2d_proj_{mode}',
                                    **kwargs))
      return
    
    if self.kv_shift_hidden_way in ['kv', 'qkv'] and cfg.kv_shift_flash: # kv
      for mode in self.kv_shift_hidden_way:
        setattr(self, f'dw_proj_{mode}', linears.DenseGeneral(
                                    (self.num_kv_heads, ),
                                    kernel_init=initializers.contant_dense_init(0.0),
                                    kernel_axes=('embed', "kv_heads"),
                                    use_bias=False,
                                    name=f'kv_shift_proj_{mode}',
                                    **kwargs))
    else:
      self.dw_proj = linears.DenseGeneral(
                                    (self.num_kv_heads * self.num_shifts, ),
                                      kernel_init=initializers.contant_dense_init(0.0),
                                      kernel_axes=('embed', "kv_heads"),
                                      use_bias=False,
                                      name='kv_shift_proj',
                                      **kwargs)
      
  @nn.compact
  def __call__(
      self,
      inputs_q, # BTD
      query, # BTND
      key, # BTND
      value, # BTND 
      inputs_k=None, # BTD
      inputs_v=None, # BTD
      inputs_m=None, # BTD
      inputs_positions=None, # BT
      decoder_segment_ids=None, # BT
      kv_shift_plan=None, # (source_indices, source_valid)
  ):
    inputs = inputs_q

    if self.kv_shift_mode == "arc_2d":
      if inputs_positions is None:
        raise ValueError("kv_shift_mode='arc_2d' requires inputs_positions.")
      kg = self.dw_proj_k(inputs_k)
      vg = self.dw_proj_v(inputs_v)
      shift_kwargs = dict(
          row_stride=getattr(self.config, "kv_shift_arc_row_stride", 32),
          grid_size=getattr(self.config, "kv_shift_arc_grid_size", 1024),
          marker_position=getattr(self.config, "kv_shift_arc_marker_position", 16383),
          max_position=getattr(
              self.config,
              "kv_shift_arc_max_position",
              getattr(self.config, "rope_max_position", 16384),
          ),
      )
      arc_2d_softmax = getattr(self.config, "kv_shift_arc_2d_softmax", True)
      arc_2d_softmax = True if arc_2d_softmax is None else arc_2d_softmax
      if kv_shift_plan is None:
        key, value = arc_2d_causal_shift_kv(
            key,
            value,
            kg,
            vg,
            inputs_positions,
            decoder_segment_ids=decoder_segment_ids,
            softmax=arc_2d_softmax,
            **shift_kwargs,
        )
      else:
        source_indices, source_valid = kv_shift_plan
        key = apply_arc_2d_causal_shift(key, kg, source_indices, source_valid, softmax=arc_2d_softmax)
        value = apply_arc_2d_causal_shift(value, vg, source_indices, source_valid, softmax=arc_2d_softmax)
    elif self.config.kv_shift_flash:
      kg = jax.nn.sigmoid(self.dw_proj_k(inputs_k))[..., jnp.newaxis]
      vg = jax.nn.sigmoid(self.dw_proj_v(inputs_v))[..., jnp.newaxis]
      key = key * kg + (1-kg) * shift_1d(key, offset=1, axis=1)
      value = value * vg + (1-vg) * shift_1d(value, offset=1, axis=1)

    else:
      dw = jax.nn.sigmoid(self.dw_proj(inputs[:,1:]))
      dw = dw.reshape(*dw.shape[:-1], -1, self.num_shifts)
      kg, vg = dw[...,:1], dw[...,1:] # B(T-1)N1
      key = key.at[:, 1:].set( key[:,1:] * kg + (1-kg) * key[:,:-1]) 
      value = value.at[:, 1:].set( value[:,1:] * vg + (1-vg) * value[:,:-1])

    if not self.config.kv_shift_skip_knorm:
      key = self.kv_shift_norm(key)

    return query, key, value


class Oshift(nn.Module):
  config: Any
  mesh: Mesh
  quant: Optional[Quant] = None
  num_query_heads: int = None
  kernel_init: NdInitializer = nd_dense_init(1.0, "fan_in", "normal")

  def setup(self):
    cfg = self.config
    kwargs = dict(dtype=cfg.dtype, weight_dtype=cfg.weight_dtype, quant=self.quant)
    self.dw_proj_o = linears.DenseGeneral(
        (self.num_query_heads,),
        kernel_init=initializers.contant_dense_init(0.0),
        kernel_axes=("embed", "heads"),
        use_bias=False,
        name="o_shift_proj",
        **kwargs,
    )

  @nn.compact
  def __call__(self, inputs_q, out):
    og = jax.nn.sigmoid(self.dw_proj_o(inputs_q))[..., jnp.newaxis]
    out = out * og + (1 - og) * shift_1d(out, offset=1, axis=1)
    return out
