
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
from functools import partial

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


def make_causal_dual_mask(query_chunk_size: int, window_size: int, dtype=jnp.float32):
    large_n = get_large_negative_number(dtype)
    tril = jnp.tril(jnp.ones((query_chunk_size, query_chunk_size), dtype=bool))
    left = jnp.where(tril, 0., large_n).astype(dtype)
    right = jnp.full((query_chunk_size, window_size), large_n, dtype)
    mask = jnp.concatenate([left, right], axis=-1)[None, None]
    return mask


def make_fix_mask(qcs: int, sws: int, seq_length: int, dtype=jnp.bfloat16):
  NEG_INF = get_large_negative_number(dtype) 
  # qcs: query_chunk_size, sws: sliding_window_size
  n = qcs + sws
  row = jnp.arange(n)[:, None]     # (n,1)
  col = jnp.arange(n)[None, :]     # (1,n)
  upper_right = col > row
  in_bottom_left_block = (row >= sws) & (col < qcs)
  lower_left_of_block   = (row - sws) >= col       # 使其成为该区块的左下三角
  bottom_left_tri       = in_bottom_left_block & lower_left_of_block
  mask_bool = upper_right | bottom_left_tri
  m = jnp.where(mask_bool, NEG_INF, 0.).astype(dtype)
  return m[None, None, :seq_length, : seq_length]

@jax.jit
def _apply_attention_dot2(
    query: Array, 
    key: Array,   
    value: Array, 
    attn_mask: Array | None,
    pre_proj_dw_args: tuple = (),
    post_proj_dw_args: tuple = (),
):
  dtype = jnp.bfloat16
  # bnts -> bkgts
  einsum = jnp.einsum
  b, t, n, d = query.shape  
  n_kv = key.shape[-2]
  query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
  attn_weights = einsum("btkgd,bskd->bkgts", query, key)

  attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', None, 'activation_length', None),)
  
  # if self.config.pre_compose:
  #    # 5 demonsion
  #   pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
  #   attn_weights = pre_proj_layer(attn_weights, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)

  attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', None, 'activation_length', None),)
  # apply attention mask
  if attn_mask is not None:
    attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
  
  attn_weights = attn_weights.astype(jnp.float32)
  # normalize the attention weights
  probs = jax.nn.softmax(attn_weights).astype(dtype) # bkgts
  probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)

  # if self.config.post_compose:
  #   post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = post_proj_dw_args
  #   probs = post_proj_layer(probs, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)

  probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)
  # Casting softmaxt computation for float32 for model stability.
  probs = probs.astype(dtype)
  if attn_mask is not None:
    probs = jnp.where((attn_mask >= DEFAULT_MASK_VALUE * 0.5), probs, 0.)
  output = jnp.einsum('bkgts,bskh->btkgh', probs, value) # add group
  b, t, n_kv, g, h = output.shape
  output = jnp.reshape(output, (b, t, n_kv * g, h))
  output = nn.with_logical_constraint(output, ('activation_batch', 'activation_length', 'heads', 'mlp'),)
  return output


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
    attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', None, 'activation_length', None),)
   
    if self.config.pre_compose:
       # 5 demonsion
      pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
      attn_weights = pre_proj_layer(attn_weights, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)

    attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', None, 'activation_length', None),)
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
  
  def _attention_with_scan(
      self,
      query, key, value,
      sliding_window_size: int | None,
      pre_proj_dw_args: Array | None,
      post_proj_dw_args: Array | None,
      pre_proj_layer = None,
      post_proj_layer = None
  ):
    b, t, n, h = query.shape
    w  = self.query_chunk_size
    assert t % w == 0, f"{t} % {w} != 0"
    query = query.astype(jnp.bfloat16)
    key = key.astype(jnp.bfloat16)
    value = value.astype(jnp.bfloat16)
    encoded0 = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)

    num_steps = t // w
    idxs = jnp.arange(num_steps, dtype=jnp.int32)
    window_len = w + sliding_window_size if sliding_window_size < t else t
    fix_masks = make_fix_mask(w, sliding_window_size, t, jnp.bfloat16)

    @jax.jit
    def step(carry, i):
        encoded = carry
        start = i * w
        stop  = start + w
        kv_start = jnp.maximum(0, stop - w - sliding_window_size) if sliding_window_size < t else 0
        mask_start = jnp.minimum(i * w, sliding_window_size)
        _query = lax.dynamic_slice(query, (0, start, 0, 0), (b, w, n, h))
        _key   = lax.dynamic_slice_in_dim(key, kv_start, window_len, axis=1)
        _value = lax.dynamic_slice_in_dim(value, kv_start, window_len, axis=1)
        _attn_mask = lax.dynamic_slice_in_dim(fix_masks, mask_start, w, axis=2)

        def slice_dw(qw1, qw2, kw1, kw2, qdd, kdd):
            tensors = (qw1, qw2, kw1, kw2, qdd, kdd)
            starts  = (start, start, kv_start, kv_start, start, kv_start)
            sizes   = (w, w, window_len, window_len, w, window_len)
            def _slice(t, s, length):
                return None if t is None else lax.dynamic_slice_in_dim(
                    t, s, length, axis=1)
            return tuple(jax.tree_util.tree_map(_slice, tensors, starts, sizes))
              
        _pre_proj_dw_args = None if pre_proj_dw_args is None else slice_dw(*pre_proj_dw_args)
        _post_proj_dw_args = None if post_proj_dw_args is None else slice_dw(*post_proj_dw_args)
        _encoded = self._apply_attention_dot(_query, _key, _value, _attn_mask, 
                                              _pre_proj_dw_args, _post_proj_dw_args,
                                              pre_proj_layer, post_proj_layer)
        encoded = lax.dynamic_update_slice(encoded, _encoded, (0, start, 0, 0))
        return encoded, None

    encoded_final, _ = lax.scan(step, encoded0, idxs)
    return encoded_final
  
  def _attention_with_scan2(
    self,
    query, key, value,
    sliding_window_size: int | None,
    pre_proj_dw_args: Array | None = None,
    post_proj_dw_args: Array | None = None,
    pre_proj_layer=None,
    post_proj_layer=None,
):
    b, t, n, h = query.shape
    w = self.query_chunk_size
    assert t % w == 0, f"{t} % {w} != 0"

    num_steps = t // w
    window_len = w + sliding_window_size if sliding_window_size < t else t

    # 一次性生成每个 step 的 kv/mask 偏移
    step_idx = jnp.arange(num_steps, dtype=jnp.int32)
    kv_start = jnp.maximum(0, step_idx * w + w - sliding_window_size)
    mask_start = jnp.minimum(step_idx * w, sliding_window_size)
    fix_masks = make_fix_mask(w, sliding_window_size, t, jnp.bfloat16)

    def _slice_in_dim(x, start, length):
        return lax.dynamic_slice_in_dim(x, start, length, axis=1)

    def step(_, inputs):
        i, kv_s, mask_s = inputs 
        start = i * w
        _q = _slice_in_dim(query, start, w)
        _k = _slice_in_dim(key,   kv_s,  window_len)
        _v = _slice_in_dim(value, kv_s,  window_len)
        _m = lax.dynamic_slice_in_dim(fix_masks, mask_s, w, axis=2)

        def slice_dw(qw1, qw2, kw1, kw2, qdd, kdd):
            tensors = (qw1, qw2, kw1, kw2, qdd, kdd)
            starts  = (start, start, kv_start, kv_start, start, kv_start)
            sizes   = (w, w, window_len, window_len, w, window_len)
            def _slice(t, s, length):
                return None if t is None else lax.dynamic_slice_in_dim(
                    t, s, length, axis=1)
            return tuple(jax.tree_util.tree_map(_slice, tensors, starts, sizes))

        _pre_dw  = None if pre_proj_dw_args  is None else slice_dw(*pre_proj_dw_args)
        _post_dw = None if post_proj_dw_args is None else slice_dw(*post_proj_dw_args)

        # 真正的 attention 计算
        _encoded = self._apply_attention_dot(
            _q, _k, _v, _m,
            _pre_dw, _post_dw,
            pre_proj_layer, post_proj_layer,
        )
        return None, _encoded            # 把结果作为 ys 输出

    # 把三串偏移捆成 xs 让 scan 分步读
    _, encoded_chunks = lax.scan(step, None, (step_idx, kv_start, mask_start))
    return encoded_chunks.reshape(b, t, n, h)
  
  def _attention_with_fori2(
      self,
      query, key, value,
      sliding_window_size: int | None,
      pre_proj_dw_args: Array | None,
      post_proj_dw_args: Array | None,
      pre_proj_layer = None,
      post_proj_layer = None
  ):
    b, t, n, h = query.shape
    w  = self.query_chunk_size
    assert t % w == 0, f"{t} % {w} != 0"
    query = query.astype(jnp.bfloat16)
    key = key.astype(jnp.bfloat16)
    value = value.astype(jnp.bfloat16)
    encoded0 = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)

    num_steps = t // w
    idxs = jnp.arange(num_steps, dtype=jnp.int32)
    window_len = w + sliding_window_size if sliding_window_size < t else t
    fix_masks = make_fix_mask(w, sliding_window_size, t, jnp.bfloat16)

    @jax.jit
    def step(i, carry):
        encoded = carry
        start = i * w
        stop  = start + w
        kv_start = jnp.maximum(0, stop - w - sliding_window_size) if sliding_window_size < t else 0
        mask_start = jnp.minimum(i * w, sliding_window_size)
        _query = lax.dynamic_slice(query, (0, start, 0, 0), (b, w, n, h))
        _key   = lax.dynamic_slice_in_dim(key, kv_start, window_len, axis=1)
        _value = lax.dynamic_slice_in_dim(value, kv_start, window_len, axis=1)
        _attn_mask = lax.dynamic_slice_in_dim(fix_masks, mask_start, w, axis=2)

        def slice_dw(qw1, qw2, kw1, kw2, qdd, kdd):
            tensors = (qw1, qw2, kw1, kw2, qdd, kdd)
            starts  = (start, start, kv_start, kv_start, start, kv_start)
            sizes   = (w, w, window_len, window_len, w, window_len)
            def _slice(t, s, length):
                return None if t is None else lax.dynamic_slice_in_dim(
                    t, s, length, axis=1)
            return tuple(jax.tree_util.tree_map(_slice, tensors, starts, sizes))
              
        _pre_proj_dw_args = None if pre_proj_dw_args is None else slice_dw(*pre_proj_dw_args)
        _post_proj_dw_args = None if post_proj_dw_args is None else slice_dw(*post_proj_dw_args)
        _encoded = self._apply_attention_dot(_query, _key, _value, _attn_mask, 
                                              _pre_proj_dw_args, _post_proj_dw_args,
                                              pre_proj_layer, post_proj_layer)
        encoded = lax.dynamic_update_slice(encoded, _encoded, (0, start, 0, 0))
        return encoded
    # encoded_final = lax.fori_loop(0, num_steps, step, encoded0)
    # encoded_final, _ = lax.scan(step, encoded0, idxs)
    RematStep = nn.remat(step,
                        prevent_cse=True,
                        # policy= jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims, #  默认的 policy=None 会让 JAX 自己决定，通常是合理的
                        policy=None,
                        static_argnums=(0, ),  # Deterministic and model mode are static arguments.
                        )
    
    for i in range(num_steps):
       encoded0 = RematStep(i, encoded0)
    return encoded0
  
  def _attention_with_remat(
      self,
      query, key, value,
      sliding_window_size: int | None,
      pre_proj_dw_args: Array | None,
      post_proj_dw_args: Array | None,
      pre_proj_layer = None,
      post_proj_layer = None
  ):
    b, t, n, h = query.shape
    w  = self.query_chunk_size
    assert t % w == 0, f"{t} % {w} != 0"
    query = query.astype(jnp.bfloat16)
    key = key.astype(jnp.bfloat16)
    value = value.astype(jnp.bfloat16)
    num_steps = t // w
    fix_masks = make_fix_mask(w, sliding_window_size, t, jnp.bfloat16)
    attn_mask = _compute_slide_attn_mask(self.query_chunk_size, sliding_window_size, t, query.dtype)
    # encoded0传入chunk_attn比append再cat更省1G显存
    encoded0 = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)
    def chunk_attn(i, carry):
        encoded = carry
        start, stop = i * w, (i + 1) * w
        kv_start = max(0, stop - w - sliding_window_size) if sliding_window_size < t else 0
        if not self.config.fix_key_mask_shape or sliding_window_size == t:
          kv_stop = stop
          _attn_mask = attn_mask[..., kv_start - stop:]
        else:
          mask_start = min(i * w, sliding_window_size)
          mask_stop = min((i+1) * w, w + sliding_window_size)
          _attn_mask = fix_masks[:, :, mask_start: mask_stop]
          kv_stop = kv_start + w + sliding_window_size

        _query = query[:, start : stop]
        _key, _value = key[:, kv_start : kv_stop], value[:, kv_start : kv_stop]

        def slice_dw(qw1, qw2, kw1, kw2, qdd, kdd):
          return (qw1[:, start : stop] if qw1 is not None else None,
              qw2[:, start : stop] if qw2 is not None else None,
              kw1[:, kv_start : kv_stop] if kw1 is not None else None,
              kw2[:, kv_start : kv_stop] if kw2 is not None else None,
              qdd[:, start : stop] if qdd is not None else None,
              kdd[:, kv_start : kv_stop] if kdd is not None else None)
              
        _pre_proj_dw_args = None if pre_proj_dw_args is None else slice_dw(*pre_proj_dw_args)
        _post_proj_dw_args = None if post_proj_dw_args is None else slice_dw(*post_proj_dw_args)
        _encoded = self._apply_attention_dot(_query, _key, _value, _attn_mask, 
                                              _pre_proj_dw_args, _post_proj_dw_args,
                                              pre_proj_layer, post_proj_layer)
        encoded = lax.dynamic_update_slice(encoded, _encoded, (0, start, 0, 0)) # 比at性能稍好，但差不多
        return encoded
    RematChunkAttn = jax.checkpoint(chunk_attn,
                        prevent_cse=True, # suggest true, save more hbm memory
                        policy=None,
                        static_argnums=(0, ),
                        )
    for i in range(num_steps):
       encoded0 = RematChunkAttn(i, encoded0)
    return encoded0

  @nn.compact
  def __call__(
    self,
    query: Array, 
    key: Array,   
    value: Array, 
    decoder_segment_ids: Array | None,  # attention mask
    model_mode: str = common_types.MODEL_MODE_TRAIN,
    eos_sum = None,
    pre_proj_dw_args = None,
    post_proj_dw_args = None,
    pre_proj_layer = None,
    post_proj_layer = None,
):
    self.check_attention_inputs(query, key, value)

    b, t, n, _ = query.shape
    h = value.shape[-1]
    print(f'eos_sum: {eos_sum}')

    sliding_window_size = t if self.sliding_window_size is None else self.sliding_window_size
    # Attention mask compute
    
    if self.query_chunk_size is None:
      attn_mask = _compute_slide_attn_mask(self.query_chunk_size, sliding_window_size, t, query.dtype)
      encoded = self._apply_attention_dot(
            query, key, value, attn_mask,  
            pre_proj_dw_args=pre_proj_dw_args, 
            post_proj_dw_args=post_proj_dw_args, 
            )
    elif self.config.query_chunk_scan:
      encoded = self._attention_with_remat(
            query, key, value,
            sliding_window_size,
            pre_proj_dw_args,
            post_proj_dw_args,
            pre_proj_layer,
            post_proj_layer
            )
    else:
      attn_mask = _compute_slide_attn_mask(self.query_chunk_size, sliding_window_size, t, query.dtype)
      max_logging.log(f'Use Query chunk to Accelerate. query_chunk_size: {self.query_chunk_size}')
      w = self.query_chunk_size

      if self.config.fix_key_mask_shape:
        fix_masks = make_fix_mask(w, sliding_window_size, t, query.dtype)

      assert t % w == 0, f'{t} % {w} != 0'
      encoded = jnp.zeros((b, t, n, h), dtype=value.dtype)
      for i in range(t // w):
          start, stop = i * w, (i + 1) * w
          kv_start = max(0, stop - w - sliding_window_size) if sliding_window_size < t else 0
          if not self.config.fix_key_mask_shape or sliding_window_size == t:
            kv_stop = stop
            _attn_mask = attn_mask[..., kv_start - stop:]
          else:
            mask_start = min(i * w, sliding_window_size)
            mask_stop = min((i+1) * w, w + sliding_window_size)
            _attn_mask = fix_masks[:, :, mask_start: mask_stop]
            kv_stop = kv_start + w + sliding_window_size

          _query = query[:, start : stop]
          _key, _value = key[:, kv_start : kv_stop], value[:, kv_start : kv_stop]

          def slice_dw(qw1, qw2, kw1, kw2, qdd, kdd):
            return (qw1[:, start : stop] if qw1 is not None else None,
                qw2[:, start : stop] if qw2 is not None else None,
                kw1[:, kv_start : kv_stop] if kw1 is not None else None,
                kw2[:, kv_start : kv_stop] if kw2 is not None else None,
                qdd[:, start : stop] if qdd is not None else None,
                kdd[:, kv_start : kv_stop] if kdd is not None else None)
          
          _pre_proj_dw_args = None if pre_proj_dw_args is None else slice_dw(*pre_proj_dw_args)
          _post_proj_dw_args = None if post_proj_dw_args is None else slice_dw(*post_proj_dw_args)
          _encoded = self._apply_attention_dot(_query, _key, _value, _attn_mask, 
                                              _pre_proj_dw_args, _post_proj_dw_args,
                                              pre_proj_layer, post_proj_layer)
          encoded = encoded.at[:, start : stop].set(_encoded)
    return encoded, None, None
  

    