
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
from einops import rearrange
import aqt.jax.v2.aqt_dot_general as aqt
import aqt.jax.v2.config as aqt_config

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


def comoute_dynamic_attn_mask(global_chunk_sizes, i, dtype=jnp.bfloat16, squeeze=False):
  last_p = global_chunk_sizes[i-1]
  cur_p = global_chunk_sizes[i]
  m = jnp.triu(jnp.ones([cur_p - last_p, cur_p]), k=last_p+1)
  large_negative_number = get_large_negative_number(dtype)
  m = m.astype(dtype)
  m = jnp.where((m > 0.5), large_negative_number, m)
  if squeeze:
    return m
  else:
    return m[jnp.newaxis, jnp.newaxis, ...]


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


def _apply_attention_dot(
      carry: Array | None,
      args,
      pre_proj_layer = None,
      post_proj_layer = None,
  ):

  def qk_product(query: Array, key: Array) -> Array:
      einsum = jnp.einsum
      b, t, n, d = query.shape  
      n_kv = key.shape[-2]
      query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
      result = einsum("btkgd,bskd->bkgts", query, key)
      return result

  query, key, value, attn_mask = args[:4]
  dtype = jnp.bfloat16
  attn_weights = qk_product(query, key)

  pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = args[4:10]
  attn_weights = pre_proj_layer(attn_weights, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)

  attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', None, 'activation_length', None),)
  # apply attention mask
  if attn_mask is not None:
    attn_weights = apply_mask_to_logits(attn_weights, attn_mask)

  # normalize the attention weights
  probs = jax.nn.softmax(attn_weights).astype(dtype) # bkgts
  probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)

  post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = args[10: 16]
  probs = post_proj_layer(probs, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)

  probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)
  # Casting softmaxt computation for float32 for model stability.
  probs = probs.astype(dtype)
  if attn_mask is not None:
    probs = jnp.where((attn_mask >= DEFAULT_MASK_VALUE * 0.5), probs, 0.)
  output = jnp.einsum('bkgts,bskh->btkgh', probs, value) # add group
  b, t, n_kv, g, h = output.shape
  output = jnp.reshape(output, (b, t, n_kv * g, h))
  output = nn.with_logical_constraint(output, ('activation_batch', 'activation_length', 'heads', 'mlp'),)
  return None, output


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

  def qk_product2(self, query: Array, key: Array) -> Array:
    einsum = jnp.einsum
    if self.kv_quant: # true when quantize_kvcache set true
      einsum = self.kv_quant.einsum_fn_with_rhs_qtensor(key)
    b, t, n, d = query.shape  
    n_kv = key.shape[-2]
    assert n_kv == self.num_kv_heads
    query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
    result = einsum("btkgd,bskd->bkgts", query, key)
    return result
  
  def qk_product(self, query: Array, key: Array) -> Array:
    dot_general = aqt.dot_general_make(8, 8)
    b, t, n, d = query.shape  
    n_kv = key.shape[-2]
    assert n_kv == self.num_kv_heads
    query = jnp.reshape(query, (b, t, n_kv, n // n_kv, d))
    result = jnp.einsum("btkgd,bskd->bkgts", query, key, _dot_general=dot_general.__call__)
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

  def _attention_with_parallel(
      self,
      query, key, value, attn_mask,
      sliding_window_size: int | None,
      pre_proj_dw_args: Array | None,
      post_proj_dw_args: Array | None,
      pre_proj_layer = None,
      post_proj_layer = None,
      remat = False,
      parallel_method: str = 'fori',
  ):
    # assert self.config.fix_key_mask_shape
    b, t, n, h = query.shape
    w  = self.query_chunk_size
    assert t % w == 0, f"{t} % {w} != 0"
    num_steps = t // w
    window_len = w + sliding_window_size if sliding_window_size < t else t

    if pre_proj_dw_args is not None:
      qw1, qw2, kw1, kw2, qdd, kdd = pre_proj_dw_args

    if post_proj_dw_args is not None:
      pqw1, pqw2, pkw1, pkw2, pqdd, pkdd = post_proj_dw_args

    def body(*args):
      if parallel_method == 'fori':
        i, carry = args
      else:
        carry, i = args

      encoded = carry
      start, stop = i * w, (i + 1) * w
      kv_start = jnp.maximum(0, stop - w - sliding_window_size) if sliding_window_size < t else 0
      mask_start = jnp.minimum(i * w, sliding_window_size)
      _query = lax.dynamic_slice(query, (0, start, 0, 0), (b, w, n, h))
      _key   = lax.dynamic_slice_in_dim(key, kv_start, window_len, axis=1)
      _value = lax.dynamic_slice_in_dim(value, kv_start, window_len, axis=1)
      _attn_mask = lax.dynamic_slice_in_dim(attn_mask, mask_start, w, axis=2)

      def _safe_slice(tensor, s, length):
          return None if tensor is None else lax.dynamic_slice_in_dim(tensor, s, length, axis=1)

      _pre_proj_dw_args, _post_proj_dw_args = None, None
      _pre_proj_dw_args = (
              _safe_slice(qw1, start,     w),
              _safe_slice(qw2, start,     w),
              _safe_slice(kw1, kv_start,  window_len),
              _safe_slice(kw2, kv_start,  window_len),
              _safe_slice(qdd, start,     w),
              _safe_slice(kdd, kv_start,  window_len),
          )
      _post_proj_dw_args = (
              _safe_slice(pqw1, start,     w),
              _safe_slice(pqw2, start,     w),
              _safe_slice(pkw1, kv_start,  window_len),
              _safe_slice(pkw2, kv_start,  window_len),
              _safe_slice(pqdd, start,     w),
              _safe_slice(pkdd, kv_start,  window_len),
          )
      _encoded = self._apply_attention_dot(_query, _key, _value, _attn_mask, 
                                            _pre_proj_dw_args, _post_proj_dw_args,
                                            pre_proj_layer, post_proj_layer)
      if parallel_method == 'vmap':
          return _encoded
      
      encoded = lax.dynamic_update_slice(encoded, _encoded, (0, start, 0, 0))

      if parallel_method == 'fori':
        return encoded
      
      return encoded, None
    
    RematBody = jax.checkpoint(body, 
                               prevent_cse=False if parallel_method == 'scan' else True, # attn scan prevent cse use False
                               policy=None) if remat else body
    if parallel_method == 'vmap':
       # (num_steps, b, t, n, h)
      encoded0 = jax.vmap(RematBody)(None, jnp.arange(num_steps, dtype=jnp.int32))
      encoded0 = rearrange(encoded0, 'n B T N H -> B (n T) N H ', n=num_steps)
    elif parallel_method == 'fori':
      encoded0 = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)
      encoded0 = lax.fori_loop(0, num_steps, RematBody, encoded0)
    else:
      encoded0 = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)
      encoded0, _ = lax.scan(f=RematBody, init=encoded0, xs=jnp.arange(num_steps))
    return encoded0

  def _attention_with_parallel_split(
      self,
      query, key, value, attn_mask,
      sliding_window_size: int | None,
      pre_proj_dw_args: Array | None,
      post_proj_dw_args: Array | None,
      pre_proj_layer = None,
      post_proj_layer = None,
      remat = False,
  ):
    b, t, n, h = query.shape
    w  = self.query_chunk_size
    assert t % w == 0, f"{t} % {w} != 0"
    num_steps = t // w
    window_len = w + sliding_window_size if sliding_window_size < t else t

    if pre_proj_dw_args is not None:
      qw1, qw2, kw1, kw2, qdd, kdd = pre_proj_dw_args

    if post_proj_dw_args is not None:
      pqw1, pqw2, pkw1, pkw2, pqdd, pkdd = post_proj_dw_args

    qw1s, qw2s, kw1s, kw2s, qdds, kdds = [[] for i in  range(6)]
    pqw1s, pqw2s, pkw1s, pkw2s, pqdds, pkdds = [[] for i in  range(6)]
    queries, keys, values, attn_masks = [[] for i in  range(4)]

    for i in range(num_steps):
      start, stop = i * w, (i + 1) * w
      kv_start = max(0, start - sliding_window_size) if sliding_window_size < t else 0
      mask_start = min(i * w, sliding_window_size)
      mask_stop = min((i+1) * w, w + sliding_window_size)
      _attn_mask = attn_mask[:, :, mask_start: mask_stop]
      kv_stop = kv_start + w + sliding_window_size

      _query = query[:, start : stop]
      _key, _value = key[:, kv_start : kv_stop], value[:, kv_start : kv_stop]

      queries.append(_query)
      keys.append(_key)
      values.append(_value)
      attn_masks.append(_attn_mask)

      def _safe_slice(tensor, s, length):
          return None if tensor is None else lax.dynamic_slice_in_dim(tensor, s, length, axis=1)

      _qw1 = _safe_slice(qw1, start,     w)
      _qw2 = _safe_slice(qw2, start,     w)
      _kw1 = _safe_slice(kw1, kv_start,  window_len)
      _kw2 = _safe_slice(kw2, kv_start,  window_len)
      _qdd = _safe_slice(qdd, start,     w)
      _kdd = _safe_slice(kdd, kv_start,     window_len)

      qw1s.append(_qw1)
      qw2s.append(_qw2)
      kw1s.append(_kw1)
      kw2s.append(_kw2)
      qdds.append(_qdd)
      kdds.append(_kdd)

      _pqw1 = _safe_slice(pqw1, start,     w)
      _pqw2 = _safe_slice(pqw2, start,     w)
      _pkw1 = _safe_slice(pkw1, kv_start,  window_len)
      _pkw2 = _safe_slice(pkw2, kv_start,  window_len)
      _pqdd = _safe_slice(pqdd, start,     w)
      _pkdd = _safe_slice(pkdd, kv_start,     window_len)

      pqw1s.append(_pqw1)
      pqw2s.append(_pqw2)
      pkw1s.append(_pkw1)
      pkw2s.append(_pkw2)
      pqdds.append(_pqdd)
      pkdds.append(_pkdd)

    queries = jnp.stack(queries)
    keys = jnp.stack(keys)
    values = jnp.stack(values)
    attn_masks = jnp.stack(attn_masks)

    qw1s = jnp.stack(qw1s)
    qw2s = jnp.stack(qw2s)
    kw1s = jnp.stack(kw1s)
    kw2s = jnp.stack(kw2s)
    qdds = jnp.stack(qdds)
    kdds = jnp.stack(kdds)

    pqw1s = jnp.stack(pqw1s)
    pqw2s = jnp.stack(pqw2s)
    pkw1s = jnp.stack(pkw1s)
    pkw2s = jnp.stack(pkw2s)
    pqdds = jnp.stack(pqdds)
    pkdds = jnp.stack(pkdds)
  
    inputs = [queries, keys, values, attn_masks, qw1s, qw2s, kw1s, kw2s, qdds, kdds, pqw1s, pqw2s, pkw1s, pkw2s, pqdds, pkdds]
    
    apply_attention_dot_func = partial(_apply_attention_dot, pre_proj_layer=pre_proj_layer, post_proj_layer=post_proj_layer)

    Remat_apply_attention_dot_func = jax.checkpoint(apply_attention_dot_func, 
                               prevent_cse=False, # attn scan prevent cse use False
                               policy=None) if remat else apply_attention_dot_func

    _, encoded0 = lax.scan(f=Remat_apply_attention_dot_func, init=None, xs=inputs)
    # (num_steps, b, t, n, h)
    encoded0 = rearrange(encoded0, 'n B T N H -> B (n T) N H ', n=num_steps)

    return encoded0

  def _attention_with_remat(
      self,
      query, key, value, attn_mask,
      sliding_window_size: int | None,
      pre_proj_dw_args: Array | None,
      post_proj_dw_args: Array | None,
      pre_proj_layer = None,
      post_proj_layer = None,
      remat: bool = False
  ):
    b, t, n, h = query.shape
    
    if self.config.balanced_attn:
      w = self.config.local_chunk_sizes if sliding_window_size < t else self.config.global_chunk_sizes 
      num_steps = len(w) - 1
      assert w[0] == 0 and w[-1] == t
    else:
      w  = self.query_chunk_size
      assert t % w == 0, f"{t} % {w} != 0"
      num_steps = t // w
      
    print(f'sliding_window_size: {sliding_window_size} query_chunk_sizes: {w}')
    # encoded0传入chunk_attn比append再cat更省1G显存
    encoded0 = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)
    def chunk_attn(i, carry):
        encoded = carry
        if self.config.balanced_attn:
          start, stop = w[i], w[i + 1]
        else:
          start, stop = i * w, (i + 1) * w

        kv_start = max(0, start - sliding_window_size) if sliding_window_size < t else 0
        if not self.config.fix_key_mask_shape:
          kv_stop = stop
          if attn_mask is None:
            _attn_mask = comoute_dynamic_attn_mask(self.config.global_chunk_sizes, i+1)
          else:
            _attn_mask = attn_mask[..., kv_start - stop:]
        else:
          mask_start = min(i * w, sliding_window_size)
          mask_stop = min((i+1) * w, w + sliding_window_size)
          _attn_mask = attn_mask[:, :, mask_start: mask_stop]
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
                        prevent_cse=True, # no scan, so suggest true, save more hbm memory
                        policy=None,
                        static_argnums=(0, ),
                        ) if remat else chunk_attn
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
    def update_mask(v, atten_mask):
      # 当设置Windows大于8192时，自动根据数据中的eos数量，来决定Windows是否重置为4096
      offset = 1 - 8192 - self.query_chunk_size
      atten_mask = atten_mask.at[..., :offset].set(v)
      return atten_mask
    
    self.check_attention_inputs(query, key, value)

    b, t, _, _ = query.shape
    sliding_window_size = t if self.sliding_window_size is None else min(t, self.sliding_window_size)
    if self.config.fix_key_mask_shape:
      attn_mask = make_fix_mask(self.query_chunk_size, sliding_window_size, t, query.dtype)
      assert eos_sum is None and not self.config.mix_attn # not attn scan don't support mix_attn
    else:
      if eos_sum is None:
        if self.config.balanced_attn:
          if sliding_window_size == t:
            attn_mask = None
          else:
            q = self.config.local_chunk_sizes[1]
            attn_mask = _compute_slide_attn_mask(q, sliding_window_size, t, query.dtype)
        else:
            attn_mask = _compute_slide_attn_mask(self.query_chunk_size, sliding_window_size, t, query.dtype)
      else:
        if sliding_window_size < self.config.max_target_length // 3: # 1, 1 t s
          attn_mask = _compute_slide_attn_mask(self.query_chunk_size, sliding_window_size, t, query.dtype)
          attn_mask = attn_mask[:, jnp.newaxis]
        else:
          attn_mask = _compute_slide_attn_mask(self.query_chunk_size, sliding_window_size, t, query.dtype, squeeze=True)
          attn_mask = jax.lax.broadcast(attn_mask, (b, )) # b x qchunk x s
          large_negative_number = get_large_negative_number(attn_mask.dtype)
          eos_sum_mask = large_negative_number * eos_sum
          attn_mask = jax.vmap(update_mask, in_axes=0, out_axes=0)(eos_sum_mask, attn_mask)
          attn_mask = nn.with_logical_constraint(attn_mask, ('activation_batch', 'activation_length', None),)
          attn_mask = attn_mask[:, jnp.newaxis, jnp.newaxis, ...] # bts -> bnts #  (4, 1, 512, 2048)

    if self.query_chunk_size is None:
      assert not self.config.fix_key_mask_shape
      encoded = self._apply_attention_dot(
              query, key, value, attn_mask,  
              pre_proj_dw_args=pre_proj_dw_args, 
              post_proj_dw_args=post_proj_dw_args, 
              )
    else:
      args = (query, key, value, attn_mask, sliding_window_size, pre_proj_dw_args, post_proj_dw_args, pre_proj_layer, post_proj_layer)
      if self.config.query_chunk_method == 'scan': # need huge hbm, only support fix key mask
        assert self.config.fix_key_mask_shape
        encoded = self._attention_with_parallel(*args, remat=False, parallel_method='scan')
      elif self.config.query_chunk_method == 'remat_scan':
        assert self.config.fix_key_mask_shape
        encoded = self._attention_with_parallel(*args, remat=True, parallel_method='scan')
      # best branch
      elif self.config.query_chunk_method == 'remat': # support fix/dynamic key mask
        if sliding_window_size == t:
          encoded = self._attention_with_remat(*args, remat=True)
        else:
          attn_mask = make_fix_mask(self.query_chunk_size, sliding_window_size, t, query.dtype)
          encoded = self._attention_with_parallel(*args, remat=True, parallel_method='fori') # fori remat=True more quick than fori remat=False?
      elif self.config.query_chunk_method == 'vmap':
        encoded = self._attention_with_parallel(*args, parallel_method='vmap')
      elif self.config.query_chunk_method == 'remat_vmap':
        encoded = self._attention_with_parallel(*args,  remat=True, parallel_method='vmap')
      elif self.config.query_chunk_method == 'remat_fori':
        encoded = self._attention_with_parallel(*args,  remat=True, parallel_method='fori')
      elif self.config.query_chunk_method == 'remat_scan_split':
        encoded = self._attention_with_parallel_split(*args,  remat=True)
      else:                                           # support fix/dynamic key mask
        encoded = self._attention_with_remat(*args, remat=False)
    return encoded, None, None