
import enum
import functools
import math
from typing import Any, Optional, Tuple

from flax import linen as nn
import jax
from jax import lax
import jax.numpy as jnp
import einops
import common_types
import max_logging
from layers.embeddings import get_alibi_mask


Array = common_types.Array
Config = common_types.Config
DType = common_types.DType

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


def _compute_slide_attn_mask(w, window_size, length: int, dtype: jnp.dtype = jnp.bfloat16, squeeze: bool = False, unmask_bos: bool = False, mask_current_token: bool = False) -> Array:
  """
  w: query chunk size
  window_size: window size
  length: query length that before split
  dtype: query dtype
  """
  if w is None:
    w = length
  if unmask_bos and w == length:
    if window_size is not None:
      window_size = window_size - 1 # spare one token for bos
  if window_size is None:
    offset = length - w
  else:
    offset = min(window_size, length - w)
  x = jnp.ones([w, w + offset])
  m1 = jnp.triu(x, k=offset + 1 - int(mask_current_token))
  if window_size is not None:
    if window_size < length - w:
        m2 = jnp.tril(x, k=0)
    else:
        m2 = jnp.tril(x, k=length - window_size - w)
    m = m1 + m2
  else:
    m = m1
  if unmask_bos and w == length: # unmask the first token when query_chunk_size == seq_len
    m = m.at[:, 0].set(0) 
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
  query_chunk_size: Optional[int] = None

  def setup(self):
    cfg = self.config
    # self.query_chunk_size = cfg.query_chunk_size
    self.float32_qk_product = cfg.float32_qk_product
    self.float32_logits = cfg.float32_logits
    self.post_compose = cfg.post_compose
    self.pre_compose = cfg.pre_compose
    self.dtype = cfg.dtype
    self.num_kv_heads = cfg.num_kv_heads
    self.use_alibi = cfg.use_alibi 
    if self.use_alibi:
       mode = cfg.alibi_mode if cfg.alibi_mode else 'sigmoid_attention' 
       self.alibi_mask = get_alibi_mask(cfg.base_num_query_heads, cfg.max_target_length, mode=mode)
    self.sigmoid_attention = cfg.sigmoid_attention
    self.sigmoid_bias = -math.log(cfg.max_target_length) + self.config.sigmoid_bias if self.config.use_sigmoid_bias else 0

    if self.config.sigmoid_bias_learnable:
      self.sigmoid_bias_learn = self.param(
          "sigmoid_bias_learn",
          nn.with_logical_partitioning(nn.initializers.zeros, ("norm",)),
          (1,),
          getattr(cfg, 'weight_dtype', jnp.float32),
      )

  def check_attention_inputs(self, query: Array, key: Array, value: Array) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, "k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert key.shape[-2] == value.shape[-2], "k, v num_kv_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."
    assert query.shape[-1] == key.shape[-1], "q, k depths must match."

  def qk_product(self, query: Array, key: Array) -> Array:
    b, t, n, d = query.shape  
    n_kv = key.shape[-2]
    # assert n_kv == self.num_kv_heads
    # normal: b t n d
    result = jnp.einsum('btnd,bsnd->bnts', query, key)
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
      alibi_mask=None,
      attn_bias=None,
      sinks=None,
  ):
    """Apply Attention."""
    if self.float32_qk_product:
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)
    # bnts
    attn_weights = self.qk_product(query, key)
    attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
   
    if self.config.pre_compose and pre_proj_dw_args is not None:
       # 5 demonsion
      pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
      attn_weights = pre_proj_layer(attn_weights, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)

    attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
   
    if self.use_alibi:
      attn_weights = attn_weights + alibi_mask
      # attn_weights = attn_weights + 2 + alibi_mask 
      # attn_weights = attn_weights / 4 + alibi_mask
      # attn_weights = attn_weights * jnp.exp(alibi_mask)
    if self.sigmoid_attention:
      attn_weights = attn_weights + self.sigmoid_bias
      if self.config.sigmoid_bias_learnable:
        attn_weights = attn_weights + self.sigmoid_bias_learn
    if attn_bias is not None:
      attn_weights = attn_weights.astype(jnp.float32) + attn_bias
   
    # apply attention mask
    if attn_mask is not None:
      attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
    
    if sinks is not None:
      batch_size, num_heads, seq_len, kv_len = attn_weights.shape
      sink_logits = einops.repeat(sinks, 'N -> B N T 1', B=batch_size, T=seq_len)
      sink_logits = nn.with_logical_constraint(sink_logits, 
        ('activation_batch', 'heads', 'activation_length', None),)  # XD: necceary?
      attn_weights = jnp.concatenate([attn_weights, sink_logits], axis=-1)
    
    if self.config.float32_logits:
      attn_weights = attn_weights.astype(jnp.float32)
    # normalize the attention weights
    if self.sigmoid_attention:
      probs = jax.nn.sigmoid(attn_weights).astype(self.dtype)
      # probs = attn_weights.astype(self.dtype)
    else:
      probs = jax.nn.softmax(attn_weights).astype(self.dtype)
    
    if sinks is not None: probs = probs[..., :-1]
    
    probs = nn.with_logical_constraint(probs, ('activation_batch', 'heads', 'activation_length', None),)

    if self.config.post_compose and post_proj_dw_args is not None:
      post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = post_proj_dw_args
      probs = post_proj_layer(probs, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)

    probs = nn.with_logical_constraint(probs, ('activation_batch', 'heads', 'activation_length', None),)
    # Casting softmaxt computation for float32 for model stability.
    probs = probs.astype(self.dtype)
    if attn_mask is not None:
      probs = jnp.where((attn_mask >= DEFAULT_MASK_VALUE * 0.5), probs, 0.)
    output = jnp.einsum('bnts,bsnh->btnh', probs, value)
    probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_length', 'heads', 'mlp'),)
    return output

  def _attention_parallel_remat(
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
    b, t, n, h = query.shape
    w  = self.query_chunk_size
    assert t % w == 0, f"{t} % {w} != 0"
    num_steps = t // w
    window_len = w + sliding_window_size if sliding_window_size < t else t

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
    attn_mask=None,
    attn_bias=None,
    sinks=None,
):
    self.check_attention_inputs(query, key, value)

    b, t, n, _ = query.shape
    h = value.shape[-1]
    s = key.shape[1]
    if attn_mask is None:
      if self.config.query_chunk_method is not None and 'parallel' in self.config.query_chunk_method and self.sliding_window_size is not None and self.sliding_window_size < t:
        attn_mask = make_fix_mask(self.query_chunk_size, self.sliding_window_size, t, query.dtype)
      else:
        attn_mask = _compute_slide_attn_mask(self.query_chunk_size, self.sliding_window_size, t, query.dtype, unmask_bos=self.config.unmask_bos, mask_current_token=self.config.mask_current_token)

    if self.query_chunk_size is None:
        encoded = self._apply_attention_dot(
            query, key, value, attn_mask,  
            pre_proj_dw_args=pre_proj_dw_args, 
            post_proj_dw_args=post_proj_dw_args, 
            pre_proj_layer=pre_proj_layer,
            post_proj_layer=post_proj_layer,
            sinks=sinks,
            )
    else:
        # max_logging.log(f'Use Query chunk to Accelerate. query_chunk_size: {self.query_chunk_size}')
        w = self.query_chunk_size
        assert t % w == 0, f'{t} % {w} != 0'
        # if self.config.chunk_scan:
        #   carrys = []
        #   for i in range(t // w):
        #       start, stop = i * w, (i + 1) * w
        #       kv_start = max(0, stop - w - self.sliding_window_size) if self.sliding_window_size is not None else 0              
        #       carrys.append((start, stop, kv_start))
        #   def qchunk_scan(carry, xs):
        #     # start, stop = i * w, (i + 1) * w
        #     # kv_start = jax.lax.max(0, stop - w - self.sliding_window_size) if self.sliding_window_size is not None else 0
        #     # start, _, _ = carry[0]
        #     start = carry
        #     stop = start + w
        #     kv_start = 0
        #     # stop = self.sliding_window_size or t
        #     _query = jax.lax.dynamic_slice_in_dim(query, start, w, axis=1)
        #     _key = jax.lax.dynamic_slice_in_dim(key, kv_start, stop-kv_start, axis=1)
        #     _value = jax.lax.dynamic_slice_in_dim(value, kv_start, stop-kv_start, axis=1)
        #     _attn_mask = jax.lax.dynamic_slice_in_dim(attn_mask, stop-kv_start, stop-kv_start, axis=-1)
        #     # alibi_mask = jnp.array(self.alibi_mask[..., start : stop, -_key.shape[1]:]) if self.use_alibi else None
        #     alibi_mask = None
        #     def slice_dw(qw1, qw2, kw1, kw2, qdd, kdd):
        #         return (jax.lax.dynamic_slice_in_dim(qw1, start, w, axis=1) if qw1 is not None else None,
        #             jax.lax.dynamic_slice_in_dim(qw2, start, w, axis=1)  if qw2 is not None else None,
        #             jax.lax.dynamic_slice_in_dim(kw1, kv_start, stop-kv_start, axis=1) if kw1 is not None else None,
        #             jax.lax.dynamic_slice_in_dim(kw2, kv_start, stop-kv_start, axis=1) if kw2 is not None else None,
        #             jax.lax.dynamic_slice_in_dim(qdd, start, w, axis=1) if qdd is not None else None,
        #             jax.lax.dynamic_slice_in_dim(kdd, kv_start, stop-kv_start, axis=1) if kdd is not None else None)
            
        #     _pre_proj_dw_args = None if pre_proj_dw_args is None else slice_dw(*pre_proj_dw_args)
        #     _post_proj_dw_args = None if post_proj_dw_args is None else slice_dw(*post_proj_dw_args)
        #     _encoded = self._apply_attention_dot(_query, _key, _value, _attn_mask, 
        #                                           _pre_proj_dw_args, _post_proj_dw_args,
        #                                           pre_proj_layer, post_proj_layer, alibi_mask=alibi_mask)
        #     return carry + w, _encoded
        #   _, encoded = jax.lax.scan(qchunk_scan, init=0, xs=None, length=math.ceil(t // w))
        #   # scan_fn = nn.scan(
        #   # decoder_layer,
        #   # variable_axes={
        #   #     "params": params_spec,
        #   #     "cache": cache_spec,
        #   #     "intermediates": 0,
        #   #     "aqt": 0,
        #   #     "_overwrite_with_gradient": 0,
        #   # },
        #   # split_rngs={
        #   #     "params": True,
        #   #     "dropout": cfg.enable_dropout,
        #   # },
        #   # in_axes=(
        #   #     nn.broadcast,
        #   #     nn.broadcast,
        #   #     nn.broadcast,
        #   #     nn.broadcast,
        #   # ),
        #   # length=length,
        #   # metadata_params={nn.PARTITION_NAME: metdata_axis_name},
        #   # )
        #   # scan_fn(config=cfg, mesh=mesh, name=metdata_axis_name, quant=self.quant)

        #   # _, encoded = nn.scan(qchunk_scan, init=0, xs=carrys, length=math.ceil(t // w))
        #   encoded = encoded.reshape(b,t,n,h) 
        if self.config.query_chunk_method is not None and 'parallel' in self.config.query_chunk_method and self.sliding_window_size is not None and self.sliding_window_size < t:
          remat = True if 'remat' in self.config.query_chunk_method else False
          print(f'query_chunk_method: {self.config.query_chunk_method} t: {t}')
          args = (query, key, value, attn_mask, self.sliding_window_size, pre_proj_dw_args, post_proj_dw_args, pre_proj_layer, post_proj_layer)
          print(f'Local Attn Parallel.... remat is {remat} sliding_window_size: {self.sliding_window_size}')
          encoded = self._attention_parallel_remat(*args, remat=remat, parallel_method='scan')
        else:
          encoded = jnp.zeros((b, t, n, h), dtype=value.dtype)
          for i in range(t // w):
              start, stop = i * w, (i + 1) * w
              kv_start = max(0, stop - w - self.sliding_window_size) if self.sliding_window_size is not None else 0
              _query = query[:, start : stop]
              _key, _value = key[:, kv_start : stop], value[:, kv_start : stop]
              _attn_mask = attn_mask[..., -_key.shape[1]:]
              alibi_mask = jnp.array(self.alibi_mask[..., start : stop, -_key.shape[1]:]) if self.use_alibi else None
              _attn_bias = attn_bias[..., start : stop, -_key.shape[1]:] if attn_bias is not None else None
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
                                                  pre_proj_layer, post_proj_layer,
                                                  alibi_mask=alibi_mask, attn_bias=_attn_bias,
                                                  sinks=sinks)
              encoded = encoded.at[:, start : stop].set(_encoded)
    return encoded, None, None
