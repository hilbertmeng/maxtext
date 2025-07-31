
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
from einops import rearrange

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

  def _attention_with_parallel(
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
    encoded0 = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)

    def body(carry, i):
        encoded = carry
        start, stop = i * w, (i + 1) * w
        kv_start = jnp.maximum(0, stop - w - sliding_window_size) if sliding_window_size < t else 0
        mask_start = jnp.minimum(i * w, sliding_window_size)
        _query = lax.dynamic_slice(query, (0, start, 0, 0), (b, w, n, h))
        _key   = lax.dynamic_slice_in_dim(key, kv_start, window_len, axis=1)
        _value = lax.dynamic_slice_in_dim(value, kv_start, window_len, axis=1)
        _attn_mask = lax.dynamic_slice_in_dim(attn_mask, mask_start, w, axis=2)

        _pre_proj_dw_args, _post_proj_dw_args = None, None
        def _safe_slice(tensor, s, length):
            return None if tensor is None else lax.dynamic_slice_in_dim(tensor, s, length, axis=1)

        if pre_proj_dw_args is not None:
            qw1, qw2, kw1, kw2, qdd, kdd = pre_proj_dw_args
            _pre_proj_dw_args = (
                _safe_slice(qw1, start,     w),
                _safe_slice(qw2, start,     w),
                _safe_slice(kw1, kv_start,  window_len),
                _safe_slice(kw2, kv_start,  window_len),
                _safe_slice(qdd, start,     w),
                _safe_slice(kdd, kv_start,  window_len),
            )
        if post_proj_dw_args is not None:
            qw1, qw2, kw1, kw2, qdd, kdd = post_proj_dw_args
            _post_proj_dw_args = (
                _safe_slice(qw1, start,     w),
                _safe_slice(qw2, start,     w),
                _safe_slice(kw1, kv_start,  window_len),
                _safe_slice(kw2, kv_start,  window_len),
                _safe_slice(qdd, start,     w),
                _safe_slice(kdd, kv_start,  window_len),
            )
        _encoded = self._apply_attention_dot(_query, _key, _value, _attn_mask, 
                                              _pre_proj_dw_args, _post_proj_dw_args,
                                              pre_proj_layer, post_proj_layer)
        encoded = lax.dynamic_update_slice(encoded, _encoded, (0, start, 0, 0))
        return encoded, None
    
    RematBody = jax.checkpoint(body, 
                               prevent_cse=True if parallel_method == 'vmap' else False, # attn scan prevent cse use False
                               policy=None) if remat else body
    encoded0, _ = lax.scan(f=RematBody, init=encoded0, xs=jnp.arange(num_steps))
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
    w  = self.query_chunk_size
    assert t % w == 0, f"{t} % {w} != 0"
    num_steps = t // w
    # encoded0传入chunk_attn比append再cat更省1G显存
    encoded0 = jnp.zeros((b, t, n, h), dtype=jnp.bfloat16)
    def chunk_attn(i, carry):
        encoded = carry
        start, stop = i * w, (i + 1) * w
        kv_start = max(0, stop - w - sliding_window_size) if sliding_window_size < t else 0
        kv_stop = stop
        _attn_mask = attn_mask[..., kv_start - stop:]
        
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
      offset = 1 - 4096 - self.query_chunk_size
      atten_mask = atten_mask.at[..., :offset].set(v)
      return atten_mask
    
    self.check_attention_inputs(query, key, value)

    b, t, n, h = query.shape
    print(f'eos_sum: {eos_sum}')
    sliding_window_size = t if self.sliding_window_size is None else min(t, self.sliding_window_size)
    if eos_sum is None:
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



  #provide chunk function -> provide q, k, v and parameters in chunk
    if self.query_chunk_size is None:
      encoded = self._apply_attention_dot(
              query, key, value, attn_mask,  
              pre_proj_dw_args=pre_proj_dw_args, 
              post_proj_dw_args=post_proj_dw_args, 
              )
    else:
      args = (query, key, value, attn_mask, sliding_window_size, pre_proj_dw_args, post_proj_dw_args, pre_proj_layer, post_proj_layer)
      # best branch
      if self.config.query_chunk_method == 'remat': # support fix/dynamic key mask
        print(f'query_chunk_method: remat...')
        if sliding_window_size == t:
          encoded = self._attention_with_remat(*args, remat=True)
        else:
          attn_mask = make_fix_mask(self.query_chunk_size, sliding_window_size, t, query.dtype)
          encoded = self._attention_with_parallel(*args, remat=True) # fori remat=True more quick than fori remat=False?
      else:                                           # support fix/dynamic key mask
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


Q_CHUNK_SIZE = 128
K_CHUNK_SIZE = 128


#for replacing apply_attention_dot
def flash_attention_chunk(self,
      query, key, value, attn_mask,
      sliding_window_size: int | None,
      pre_proj_dw_args: Array | None,
      post_proj_dw_args: Array | None,
      pre_proj_layer = None,
      post_proj_layer = None,
      remat = False,):
  
  pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
  post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = post_proj_dw_args
  I_dim = pre_qw1.shape[-1]
  batch_size, seq_len_t, num_heads, dim = query.shape
  seq_len_k = key.shape[1]
  
  
  #chunk scanner, outer loop for q chunk
  def chunk_scanner_pre(chunk_idx, _):
    chunk_sizes = min(Q_CHUNK_SIZE, seq_len_t)

    q_chunk = lax.dynamic_slice(query, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, num_heads, dim))
    
    #chunk along q dimension
    pre_qw1_chunk = lax.dynamic_slice(pre_qw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, num_heads, I_dim))
    pre_qw2_chunk = lax.dynamic_slice(pre_qw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, I_dim, num_heads))
    pre_qdd_chunk = lax.dynamic_slice(pre_qdd, (0, chunk_idx, 0), slice_sizes = (batch_size, chunk_sizes, num_heads))

    return (chunk_idx + chunk_sizes, query_chunk_pre_helper(chunk_idx, q_chunk, key, value, pre_qw1_chunk, pre_qw2_chunk, pre_qdd_chunk,
                                                        pre_kw1, pre_kw2,pre_kdd, pre_proj_layer,chunk_sizes, attn_mask, seq_len_t))
    
  def chunk_scanner_post(chunk_idx, _):
    chunk_sizes = min(Q_CHUNK_SIZE, seq_len_t)
    post_qw1_chunk = lax.dynamic_slice(post_qw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, num_heads, I_dim))
    post_qw2_chunk = lax.dynamic_slice(post_qw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, chunk_sizes, I_dim, num_heads))
    post_qdd_chunk = lax.dynamic_slice(post_qdd, (0, chunk_idx, 0), slice_sizes = (batch_size, chunk_sizes, num_heads))
    
    return (chunk_idx + chunk_sizes, query_chunk_post_helper(chunk_idx, seq_len_t, key, value, post_qw1_chunk, post_qw2_chunk, post_qdd_chunk,
                            post_kw1, post_kw2, post_kdd, post_proj_layer, attn_mask))
  
  
  _, pre_weights = lax.scan(chunk_scanner_pre, init = 0, xs = None, length = math.ceil(seq_len_t /Q_CHUNK_SIZE))
  
  probs = jax.nn.softmax(pre_weights).astype(jnp.dtype)
  probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)
  _, out = lax.scan(chunk_scanner_post, init = 0, xs = None, length = math.ceil(seq_len_k / K_CHUNK_SIZE))
    
  return out  
  
  

#return chunk_idx, output
#main inputs: q_chunk, k, and pre_weights, pre_layer, attn_mask
def query_chunk_pre_helper(self, chunk_idx, q_chunk, k, v, pre_qw1_chunk, pre_qw2_chunk, pre_qdd_chunk,
                          pre_kw1, pre_kw2,pre_kdd, pre_proj_layer, q_chunk_size, attn_mask, q_len):
    
    _, batch_size, num_heads, dim= q_chunk.shape
    I_dim = pre_qw1_chunk.shape[-1]
    seq_len_k = k.shape[1] 
    k_chunk_size = min(K_CHUNK_SIZE, seq_len_k)
    
    def chunk_scanner(carries, _):
        chunk_idx, out = carries
        
        k_chunk = lax.dynamic_slice(k, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, dim))

        chunk_attn_mask = lax.dynamic_slice(attn_mask, (0, chunk_idx), slice_sizes = (q_len, k_chunk_size))
        #get attention scores
        chunk_attn_weights = self.qk_product(q_chunk, k_chunk)
        chunk_attn_weights = nn.with_logical_constraint(chunk_attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
        
        #chunk for k dim
        if self.config.pre_compose:
            pre_kw1_chunk = lax.dynamic_slice(pre_kw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, I_dim))
            pre_kw2_chunk = lax.dynamic_slice(pre_kw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, I_dim, num_heads))
            pre_kdd_chunk = lax.dynamic_slice(pre_kdd, (0, chunk_idx, 0), slice_sizes = (batch_size, k_chunk_size, num_heads))
            #compose layer
            chunk_attn_weights = pre_proj_layer(chunk_attn_weights, pre_qw1_chunk, 
                                                pre_qw2_chunk, pre_kw1_chunk, pre_kw2_chunk, 
                                                pre_qdd_chunk, pre_kdd_chunk)
        chunk_attn_weights = nn.with_logical_constraint(chunk_attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
        if attn_mask is not None:
          chunk_attn_weights = apply_mask_to_logits(chunk_attn_weights, chunk_attn_mask)
        if self.config.float32_logits:
          chunk_attn_weights = chunk_attn_weights.astype(jnp.float32)
          
        return chunk_attn_weights
      
    out = jnp.zeros((batch_size, num_heads, q_chunk_size, k_chunk_size))
    _, out = lax.scan(chunk_scanner, init = (0, out), xs = None, length = math.ceil(seq_len_k / K_CHUNK_SIZE))
    return out
          
          
      
def query_chunk_post_helper(chunk_idx, q_len, k, v, post_qw1_chunk, post_qw2_chunk, post_qdd_chunk,
                            post_kw1, post_kw2, post_kdd, post_proj_layer, attn_mask, probs):

    batch_size, seq_len_k, num_heads, v_dim = k.shape
    I_dim = I_dim = post_qw1_chunk.shape[-1]
    k_chunk_size = min(K_CHUNK_SIZE, seq_len_k)
    q_chunk_size = min(Q_CHUNK_SIZE, q_len)
    
    def chunk_scanner(chunk_idx, _):
      
      v_chunk = lax.dynamic_slice(v, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, v_dim))
      chunk_attn_mask = lax.dynamic_slice(attn_mask, (0, chunk_idx), slice_sizes = (q_len, k_chunk_size))
      
      post_kw1_chunk = lax.dynamic_slice(post_kw1, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, num_heads, I_dim))
      post_kw2_chunk = lax.dynamic_slice(post_kw2, (0, chunk_idx, 0, 0), slice_sizes = (batch_size, k_chunk_size, I_dim, num_heads))
      post_kdd_chunk = lax.dynamic_slice(post_kdd, (0, chunk_idx, 0), slice_sizes = (batch_size, k_chunk_size, num_heads))
      probs = post_proj_layer(probs, post_qw1_chunk, post_qw2_chunk, post_kw1_chunk, post_kw2_chunk, post_qdd_chunk, post_kdd_chunk)
      
      probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_kv_heads', None, 'activation_length', None),)
      probs = probs.astype(self.dtype)
      if attn_mask is not None:
        probs = jnp.where((chunk_attn_mask >= DEFAULT_MASK_VALUE * 0.5), probs, 0.)
      output = jnp.einsum('bkgts,bskh->btkgh', probs, v_chunk)
      b, t, n_kv, g, h = output.shape
      output = jnp.reshape(output, (b, t, n_kv * g, h))
      output = nn.with_logical_constraint(output, ('activation_batch', 'activation_length', 'heads', 'mlp'),)
      return output
    
    out = jnp.zeros((batch_size, num_heads, q_chunk_size, k_chunk_size))
    _, out = lax.scan(chunk_scanner, init = (0, out), xs = None, length = math.ceil(seq_len_k / K_CHUNK_SIZE))
    return out
    

      