
import enum
import functools
import math
from typing import Any, Optional, Tuple

from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import common_types
import max_logging
from layers import quantizations

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
KVQuant = quantizations.KVQuant

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def apply_mask_to_logits(logits: Array, mask: Array):
  return jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), logits, DEFAULT_MASK_VALUE)


def get_large_negative_number(dtype: jnp.dtype) -> Array:
    """Returns a large negative value for the given dtype."""
    # -0.7 is a float64 in Jax. Explicit cast output to target dtype.
    if jnp.issubdtype(dtype, jnp.inexact):
      dtype_max = jnp.finfo(dtype).max
    elif jnp.issubdtype(dtype, jnp.integer):
      dtype_max = jnp.iinfo(dtype).max
    else:
      raise ValueError('Unsupported dtype for inputs.')
    return jnp.asarray(-0.7 * dtype_max, dtype=dtype)


def _compute_slide_attn_mask(w, window_size, length: int, dtype: jnp.dtype = jnp.bfloat16, squeeze: bool = False) -> Array:
  """
  w: query chunk size
  window_size: window size
  length: query length that before split
  dtype: query dtype
  """
  if w is None:
    w = length
  if window_size is None:
    offset = length - w
  else:
    offset = min(window_size, length - w)
  x = jnp.ones([w, w + offset])
  m1 = jnp.triu(x, k=offset + 1)
  if window_size is not None:
    if window_size < length - w:
        m2 = jnp.tril(x, k=0)
    else:
        m2 = jnp.tril(x, k=length - window_size - w)
    m = m1 + m2
  else:
    m = m1
  large_negative_number = get_large_negative_number(dtype)
  m = m.astype(dtype)
  m = jnp.where((m > 0.5), large_negative_number, m)
  if squeeze:
    return m
  else:
    return m[jnp.newaxis, jnp.newaxis, ...]


class QChunk(nn.Module):
  config: Config
  sliding_window_size: int
  kv_quant: Optional[KVQuant] = None

  def setup(self):
    cfg = self.config
    self.query_chunk_size = cfg.query_chunk_size
    self.float32_qk_product = cfg.float32_qk_product
    self.float32_logits = cfg.float32_logits
    self.post_compose = cfg.post_compose
    self.pre_compose = cfg.pre_compose
    self.dtype = cfg.dtype
    self.num_kv_heads = cfg.num_kv_heads

  def check_attention_inputs(self, query: Array, key: Array, value: Array) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, "k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

  def qk_product(self, query: Array, key: Array) -> Array:
    einsum = jnp.einsum
    if self.kv_quant: # true when quantize_kvcache set true
      einsum = self.kv_quant.einsum_fn_with_rhs_qtensor(key)
    b, t, n, d = query.shape  
    n_kv = key.shape[-2]
    assert n_kv == self.num_kv_heads
    query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
    result = einsum("btkgd,bskd->bkgts", query, key)
    return result

  def _apply_attention_dot(
      self,
      query: Array, 
      key: Array,   
      value: Array, 
      attn_mask: Array | None,
      pre_proj_dw_args: tuple = (),
      post_proj_dw_args: tuple = (),
      pre_proj_layer = None,
      post_proj_layer = None,
  ):
    """Apply Attention."""
    if self.float32_qk_product:
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)
    # bnts -> bkgts
    attn_weights = self.qk_product(query, key)
    attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
   
    if self.config.pre_compose:
       # 5 demonsion
      pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
      attn_weights = pre_proj_layer(attn_weights, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)

    attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
    # apply attention mask
    if attn_mask is not None:
      attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
    if self.config.float32_logits:
          attn_weights = attn_weights.astype(jnp.float32)
    # normalize the attention weights
    probs = jax.nn.softmax(attn_weights).astype(self.dtype) # bkgts
    probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)

    if self.config.post_compose:
      post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = post_proj_dw_args
      probs = post_proj_layer(probs, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)

    probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)
    # Casting softmaxt computation for float32 for model stability.
    probs = probs.astype(self.dtype)
    if attn_mask is not None:
      probs = jnp.where((attn_mask >= DEFAULT_MASK_VALUE * 0.5), probs, 0.)
    output = jnp.einsum('bkgts,bskh->btkgh', probs, value) # add group
    b, t, n_kv, g, h = output.shape
    output = jnp.reshape(output, (b, t, n_kv * g, h))
    output = nn.with_logical_constraint(output, ('activation_batch', 'activation_length', 'heads', 'mlp'),)
    return output

  @nn.compact
  def __call__(
    self,
    query: Array, 
    key: Array,   
    value: Array, 
    decoder_segment_ids: Array | None,  # attention mask
    model_mode: str = common_types.MODEL_MODE_TRAIN,
    pre_proj_dw_args = None,
    post_proj_dw_args = None,
    pre_proj_layer = None,
    post_proj_layer = None,
):
    self.check_attention_inputs(query, key, value)

    b, t, n, _ = query.shape
    h = value.shape[-1]
    s = key.shape[1]
    attn_mask = _compute_slide_attn_mask(self.query_chunk_size, self.sliding_window_size, t, query.dtype)

    if self.query_chunk_size is None:
        encoded = self._apply_attention_dot(
            query, key, value, attn_mask,  
            pre_proj_dw_args=pre_proj_dw_args, 
            post_proj_dw_args=post_proj_dw_args, 
            )
    else:
        max_logging.log(f'Use Query chunk to Accelerate. query_chunk_size: {self.query_chunk_size}')
        w = self.query_chunk_size
        assert t % w == 0, f'{t} % {w} != 0'
        encoded = jnp.zeros((b, t, n, h), dtype=value.dtype)
        for i in range(t // w):
            start, stop = i * w, (i + 1) * w
            kv_start = max(0, stop - w - self.sliding_window_size) if self.sliding_window_size is not None else 0
            _query = query[:, start : stop]
            _key, _value = key[:, kv_start : stop], value[:, kv_start : stop]
            _attn_mask = attn_mask[..., -_key.shape[1]:]
            def slice_dw(qw1, qw2, kw1, kw2, qdd, kdd):
                return (qw1[:, start : stop] if qw1 is not None else None,
                    qw2[:, start : stop] if qw2 is not None else None,
                    kw1[:, kv_start : stop] if kw1 is not None else None,
                    kw2[:, kv_start : stop] if kw2 is not None else None,
                    qdd[:, start : stop] if qdd is not None else None,
                    kdd[:, kv_start : stop] if kdd is not None else None)
            
            _pre_proj_dw_args = None if pre_proj_dw_args is None else slice_dw(*pre_proj_dw_args)
            _post_proj_dw_args = None if post_proj_dw_args is None else slice_dw(*post_proj_dw_args)
            _encoded = self._apply_attention_dot(_query, _key, _value, _attn_mask, 
                                                _pre_proj_dw_args, _post_proj_dw_args,
                                                pre_proj_layer, post_proj_layer)
            encoded = encoded.at[:, start : stop].set(_encoded)
    return encoded, None, None
