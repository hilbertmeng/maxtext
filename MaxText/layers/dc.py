from typing import Optional, Any
import math

import jax
from flax import linen as nn
from jax.sharding import Mesh
import jax.numpy as jnp
from einops import rearrange

from layers import normalizations
from layers import quantizations
from layers import linears
from layers import initializers
import max_logging
import common_types
from flax.linen.linear import PrecisionLike

Dtype = Any
Array = common_types.Array
DType = common_types.DType
Mesh = common_types.Mesh
RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

NormalInitializer = initializers.nd_dense_init_normal

class QKNorm(nn.Module):
  config: Any

  @nn.compact
  def __call__(self, query, key):
    if self.config.qk_norm:
      max_logging.log('qk norm is True', debug=self.config.debug)
      query = RMSNorm(
        weight_dtype=self.config.weight_dtype,
        dtype=self.config.dtype,
        name=f'q_norm',
        kernel_axes=('norm',),
        epsilon=self.config.normalization_layer_epsilon,
        )(query)
      key = RMSNorm(
        weight_dtype=self.config.weight_dtype,
        dtype=self.config.dtype,
        name=f'k_norm',
        kernel_axes=('norm',),
        epsilon=self.config.normalization_layer_epsilon,
        )(key)
    return query, key


class DynamicWeightProjection(nn.Module):
  weight_dtype: Optional[Dtype] = jnp.float32
  dtype: Optional[Dtype] = None
  precision: PrecisionLike = None
  n_splits: int = None  # 可以理解为pre_w1, pre_w2, post_w1, post_w2
  num_heads: int = 0
  num_groups: int = 1
  input_dim: int = None
  dynamic_w_init: float = None
  dynamic_d_init: float = None
  dynamic_squeeze_ratio: int = None
  decompose_dynamic_w: bool = True
  dynamic_w_hidden_dim: int = None
  merge_dynamic_w_hidden: bool = False
  deterministic: bool = False
  dynamic_dropout_rate: Optional[float] = None
  quant: Optional[Quant] = None

  def setup(self) -> None:
    self.num_heads_per_group = self.num_heads // self.num_groups
    kwargs = dict(
      dtype=self.dtype,
      weight_dtype=self.weight_dtype, 
      use_bias=False,
    )

    if self.dynamic_w_init is not None:
      # dynamic_hidden_dim： 2
      dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio \
        if self.dynamic_squeeze_ratio is not None else 2
      # '12x4096x1x4x128'
      self.dw1 = linears.DenseGeneral(
                        features=(self.num_groups, self.n_splits, self.dynamic_w_hidden_dim),  
                        quant=self.quant,  # 0.00014
                        kernel_init=NormalInitializer(math.sqrt(2.0 / (self.input_dim + self.dynamic_w_hidden_dim))), 
                        kernel_axes=('embed', None, 'heads', 'mlp'),
                        **kwargs)
      self.dw_hidden_activation = nn.gelu
      # self.dynamic_w_hidden_dim: num_heads_per_group * I * 2 = 32 * 2 * 2 = 128
      G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
      I = dynamic_hidden_dim * 2  # 2 * 2
      shape = [G, self.n_splits, K, I, M]
      kernel_init_shard = nn.with_logical_partitioning(NormalInitializer(self.dynamic_w_init), (None, 'data', 'fsdp', None, 'tensor'))
      self.qkw = self.param('qkw',kernel_init_shard, shape, self.weight_dtype)
  
    if self.dynamic_d_init is not None:
      self.dd = linears.DenseGeneral(
                        features=(self.num_groups, 
                        self.num_heads_per_group * self.n_splits), 
                        quant=self.quant,
                        kernel_init=NormalInitializer(self.dynamic_d_init), 
                        kernel_axes=('embed', None, 'mlp'),
                        **kwargs
                        )

    self.dw_activation = nn.tanh
    # RMSNormScale, compare to RMSNorm. it remove scale
    self.dw1_norm = nn.RMSNorm(
                      use_scale=False,
                      param_dtype=self.weight_dtype,
                      dtype=self.dtype,
                       )
    if self.dynamic_dropout_rate is not None:
      self.dropout = nn.Dropout(self.dynamic_dropout_rate)

  def __call__(self, query_vec):
    qkw_kernel = jnp.asarray(self.qkw, self.dtype) # lsp
    if self.n_splits == 2:
      dw_hidden = self.dw_hidden_activation(self.dw1(query_vec))   # BTG2,64
      if self.dynamic_dropout_rate is not None:
        dw_hidden = self.dropout(dw_hidden, deterministic=self.deterministic)
      # C: n_split,  K -> M
      w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, qkw_kernel), 2, axis=-2)
      w1 = self.dw1_norm(w1)
      pre_w1, post_w1 = unbind(w1, 2, axis=3) # BTG2IM->[BTGIM]*2
      pre_w2, post_w2 = unbind(w2, 2, axis=3)

      dd = self.dd(query_vec)
      dd = self.dw_activation(dd)
      if self.dynamic_dropout_rate is not None:
        dd = self.dropout(dd, deterministic=self.deterministic)
      pre_dd, post_dd = jnp.split(dd, 2, axis=-1)
      return (pre_w1, pre_w2, pre_dd), (post_w1, post_w2, post_dd)
    else:
      dw_hidden = self.dw_hidden_activation(self.dw1(query_vec))
      if self.dynamic_dropout_rate is not None:
        dw_hidden = self.dropout(dw_hidden, deterministic=self.deterministic)
      # dw_hidden: b * t * 1 * n_split * 128  qkw_kernel: 1 * n_split * 128 * I(4) * 128
      w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, qkw_kernel), 2, axis=-2)
      w1 = self.dw1_norm(w1)
      pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, axis=3) # BTG4IM->[BTGIM]*4
      pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, axis=3)

      dd = self.dd(query_vec)
      dd = self.dw_activation(dd)
      if self.dynamic_dropout_rate is not None:
        dd = self.dropout(dd, deterministic=self.deterministic)
      pre_qdd, pre_kdd, post_qdd, post_kdd = jnp.split(dd, 4, axis=-1)
      return (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd), \
        (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)


class CrossHeadProjection(nn.Module):
  dtype: Optional[Dtype] = None
  weight_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  squeeze_ratio: float = None
  num_heads: int = 0
  num_groups: int = 0
  relative_scale: float = 0.1
  static_proj: bool = True
  query_wise: bool = True
  key_wise: bool = True
  input_activation_cls: Optional[str] = None
  use_input_bias: bool = True
  left_mul: bool = False
  init: Optional[str] = None
  residual: bool = True
  query_input_dim: int = None # medium: 1024
  key_input_dim: int = None # medium: 1024
  dynamic_w_hidden_dim: int = None # medium: 64
  loop_over_dynamic_hd: bool = True
  decompose_dynamic_w: bool = True

  def setup(self) -> None:
    self.num_heads_per_group = self.num_heads // self.num_groups
    kwargs = dict(
                  dtype=self.dtype,
                  weight_dtype=self.weight_dtype,
                  use_bias=False,
                  precision=self.precision,
                )
    def init_fn(out_dim, in_dim=None):
      if self.init is not None: 
        return self.init
      if in_dim is None: 
        in_dim = self.num_heads_per_group
      if not self.residual or in_dim == self.num_heads_per_group and in_dim > out_dim: # ffn.w1
        relative_scale = 1.0
      elif in_dim in [self.query_input_dim, self.key_input_dim] and \
        self.dynamic_w_hidden_dim and out_dim in [self.dynamic_w_hidden_dim, self.dynamic_w_hidden_dim * 2]:
        relative_scale = 1.
      elif out_dim == self.num_heads_per_group and in_dim <= out_dim:  # ffn.w2 or w
        relative_scale = 0.1
      else:
        assert False, f'[{in_dim}, {out_dim}]'
      return math.sqrt(2.0 / (in_dim + out_dim)) * relative_scale

    if self.static_proj:
      if self.squeeze_ratio is None:
        shape=[self.num_groups, self.num_heads_per_group, self.num_heads_per_group]
        scale = init_fn(self.num_heads_per_group)
        self.w = self.param('w', NormalInitializer(scale), shape, self.weight_dtype)
      else:
        self.hidden_dim = self.num_heads_per_group // self.squeeze_ratio
        shape=[self.num_groups, self.num_heads_per_group, self.hidden_dim]
        scale = init_fn(self.hidden_dim)
        self.w1 = self.param('w1', NormalInitializer(scale), shape, self.weight_dtype)

        shape=[self.num_groups, self.hidden_dim, self.num_heads_per_group]
        scale = init_fn(self.num_heads_per_group, in_dim=self.hidden_dim)
        self.w1 = self.param('w2', NormalInitializer(scale), shape, self.weight_dtype)

  def __call__(self, inputs, qw1 = None, qw2 = None, kw1 = None, kw2 = None, qdd = None, kdd = None):
    shape = inputs.shape  #  (16, 16, 4097, 4097)
    assert inputs.shape[1] == self.num_heads

    inputs = rearrange(inputs, 'B (G M) T S -> B G M T S', G=self.num_groups)
    inputs_label = 'BGMTS'
    out_label = inputs_label.replace('M', 'N') #  BGNTS
    exp = f'{inputs_label},GMN->{out_label}' #  'BGMTS'  GMN   BGNTS

    ret = inputs
    # This op I/O too many, loss is lower but speed lower than remove it. suggest remove it
    # ret += jnp.einsum('BGMTS,GMN->BGNTS', inputs, self.w)
    if self.static_proj:
      if self.squeeze_ratio is None: # None
        w = self.w
        _inputs = inputs
        if self.input_activation_cls is not None: # None
          if self.use_input_bias: _inputs += self.ib if self.transpose else jnp.expand_dims(self.ib, axis=(2, 3))
          _inputs = self.input_activation(_inputs)
        ret += jnp.einsum(exp, _inputs, w) if not self.left_mul else jnp.einsum(exp, w, _inputs)
      else:
        hidden = jnp.einsum(exp, inputs, self.w1) if not self.left_mul else jnp.einsum(exp, self.w1, inputs)
        if self.squeeze_gate_activation_cls is not None:
          hidden = hidden * self.gate_activation(jnp.einsum(exp, inputs, self.w1g))
        else:
          hidden = self.activation(hidden)
        ret += jnp.einsum(exp, hidden, self.w2) if not self.left_mul else jnp.einsum(exp, self.w2, ret)

    if qw1 is not None: # BTGIM
      hidden_sym = 'I'; hidden_label = inputs_label.replace('M', 'I')  # 'BGITS'
      for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]):
        dw_label = f'B{sym}G{hidden_sym}M' if w1.shape[-1] == self.num_heads_per_group \
          else f'B{sym}GM{hidden_sym}'  # BTGIM
        dynamic_hidden_dim = w1.shape[dw_label.index(hidden_sym)] # 2, 就是w1的I的值
        eqn1 = f'{inputs_label},{dw_label}->{hidden_label}' # 'BGMTS,BTGMI->BGITS'  lsp: 'BGMTS,BTGIM->BGITS'
        eqn2 = f'{hidden_label},{dw_label}->{inputs_label}' # 'BGITS,BTGMI->BGMTS'  lsp: 'BGITS,BTGIM->BGMTS'
        if sym == 'T' and self.query_wise or sym == 'S' and self.key_wise:
          # dynamic_hidden_dim: I -> 2 lsp here
          if self.loop_over_dynamic_hd and dynamic_hidden_dim <= 2:  # 循环算
            for i in range(dynamic_hidden_dim):
              if dw_label[-1] == hidden_sym:
                hidden = jnp.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i])
                out = jnp.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i])
              else:
                # lsp
                assert dw_label[-2] == hidden_sym, dw_label
                # 'BGMTS,BTGM->BGTS' # head融合 1次
                hidden = jnp.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i, :])
                # 'BGTS,BTGM->BGTS' # # head融合 2次
                out = jnp.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i, :])
              ret = ret + out
          else: # 整块算
            # 'BGMTS,BTGIM->BGITS'
            hidden = jnp.einsum(eqn1, inputs, w1)
            if self.decompose_dynamic_w:  # true
              # 'BGITS,BTGIM->BGMTS'  -> out: BGMTS , ret: BGNTS, M == N
              out = jnp.einsum(eqn2, hidden, w2)
              ret = ret + out
            else:
              ret = ret + hidden

    if qdd is not None:  # 对logits做二次修改
      for sym, dd in zip(['T', 'S'], [qdd, kdd]):
        dd_label = f'B{sym}GM'
        if sym == 'T' and self.query_wise or sym == 'S' and self.key_wise or \
              not self.query_wise and not self.key_wise:
          # 'BGMTS', B(T/S)GM
          dout = jnp.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs, dd)
          ret = ret + dout
    return jnp.reshape(ret, shape)  # BGMTS->BNTS


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


def unbind(ary, n, axis=0):
  return [jnp.squeeze(a, axis=axis) for a in jnp.split(ary, n, axis=axis)]


def apply_mask_to_logits(logits: Array, mask: Array):
  return jnp.where((mask >= DEFAULT_MASK_VALUE * 0.5), logits, DEFAULT_MASK_VALUE)


class AttentionOp(nn.Module):

  config: Any
  quant: Optional[Quant] = None
  sliding_window_size: int|None = None

  def setup(self):
    cfg = self.config
    self.num_query_heads = cfg.num_query_heads
    self.num_kv_heads = cfg.num_kv_heads
    self.head_dim = cfg.head_dim
    self.is_cross_attention = False
    self.num_groups = self.num_kv_heads // self.num_query_heads
    self.dtype = cfg.dtype
    self.weight_dtype = cfg.weight_dtype
    self.precision = None
    self.deterministic = False
    self.dynamic_dropout_rate = 0.0
    self.static_proj = cfg.static_proj
    self.loop_over_dynamic_hd = True
    self.query_chunk_size = cfg.query_chunk_size
    self.float32_qk_product = cfg.float32_qk_product
    self.pre_compose = cfg.pre_compose
    self.post_compose = cfg.post_compose

    input_dim = self.num_query_heads * self.head_dim
    I = 2
    num_heads_per_group = self.num_query_heads // self.num_groups
    dynamic_w_hidden_dim = num_heads_per_group * I * 2
    if cfg.pre_compose or cfg.post_compose:
      if self.is_cross_attention:
        for name in ['q_dyn_w_proj', 'k_dyn_w_proj']:
          setattr(self, name, DynamicWeightProjection(
            num_heads=self.num_query_heads, num_groups=self.num_groups,
            input_dim=self.num_query_heads * self.head_dim, n_splits=2,
            dynamic_w_init=math.sqrt(1 / dynamic_w_hidden_dim) * 2 / (num_heads_per_group + I) * 0.01,
            dynamic_d_init=math.sqrt(2 / (input_dim + num_heads_per_group)) * 0.005,
            dynamic_squeeze_ratio=num_heads_per_group // I,
            dynamic_w_hidden_dim=dynamic_w_hidden_dim,
            dtype=self.dtype, weight_dtype=self.weight_dtype, precision=self.precision,
            deterministic=self.deterministic,
            dynamic_dropout_rate=self.dynamic_dropout_rate,
            quant=self.quant,
          ))
      else:
        self.dyn_w_proj = DynamicWeightProjection(
          num_heads=self.num_query_heads, num_groups=self.num_groups,
          input_dim=self.num_query_heads * self.head_dim, n_splits=4,
          dynamic_w_init=math.sqrt(1 / dynamic_w_hidden_dim) * 2 / (num_heads_per_group + I) * 0.01,
          dynamic_d_init=math.sqrt(2 / (input_dim + num_heads_per_group)) * 0.005,
          dynamic_squeeze_ratio=num_heads_per_group // I,
          dynamic_w_hidden_dim=dynamic_w_hidden_dim,
          dtype=self.dtype, weight_dtype=self.weight_dtype, precision=self.precision,
          deterministic=self.deterministic,
          dynamic_dropout_rate=self.dynamic_dropout_rate,
          quant=self.quant,
          )

      self.pre_proj = CrossHeadProjection(
        dtype=self.dtype, 
        weight_dtype=self.weight_dtype, 
        precision=self.precision,
        num_heads=self.num_query_heads, 
        num_groups=self.num_groups,
        static_proj=self.static_proj,
        query_input_dim=input_dim,
        key_input_dim=input_dim,
        dynamic_w_hidden_dim=dynamic_w_hidden_dim,
        loop_over_dynamic_hd=self.loop_over_dynamic_hd
        )

      self.post_proj = CrossHeadProjection(
        dtype=self.dtype, 
        weight_dtype=self.weight_dtype, 
        precision=self.precision,
        num_heads=self.num_query_heads, 
        num_groups=self.num_groups,
        static_proj=self.static_proj,
        query_input_dim=input_dim,
        key_input_dim=input_dim,
        dynamic_w_hidden_dim=dynamic_w_hidden_dim,
        loop_over_dynamic_hd=self.loop_over_dynamic_hd
        )

  def check_attention_inputs(
    self,
    query: Array,
    key: Array,
    value: Array) -> None:
    """Check attention inputs."""

    assert key.ndim == value.ndim, 'k, v must have same rank.'
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
        'q, k, v batch dims must match.')
    assert key.shape[-2] == value.shape[-2], ('k, v num_kv_heads must match.')
    assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'
    
  @nn.compact
  def __call__(
    self,
    query: Array, 
    key: Array,   
    value: Array, 
    decoder_segment_ids: Array | None,  # attention mask
    model_mode: str = common_types.MODEL_MODE_TRAIN,
    input_q: Array = None,
    input_kv: Array = None,
):
    cfg = self.config
    self.check_attention_inputs(query, key, value)

    b, t, n, _ = query.shape
    h = value.shape[-1]
    s = key.shape[1]
    attn_mask = _compute_slide_attn_mask(self.query_chunk_size, self.sliding_window_size, t, query.dtype)

    if cfg.pre_compose or cfg.post_compose:
        if hasattr(self, 'dyn_w_proj'):
            pre_proj_dw_args, post_proj_dw_args = self.dyn_w_proj(input_q)
        else:
            if hasattr(self, 'dyn_w_pre_proj'):
                pre_proj_dw_args = self.dyn_w_pre_proj(input_q)
            if hasattr(self, 'dyn_w_post_proj'):
                post_proj_dw_args = self.dyn_w_post_proj(input_kv)
    else:
        pre_proj_dw_args, post_proj_dw_args = (None, ) * 6, (None, ) * 6

    if self.query_chunk_size is None:
        encoded = self._apply_attention_dot(
            query, key, value, attn_mask,  
            pre_proj_dw_args=pre_proj_dw_args, 
            post_proj_dw_args=post_proj_dw_args, 
            )
    else:
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
            _pre_proj_dw_args = slice_dw(*pre_proj_dw_args)
            _post_proj_dw_args = slice_dw(*post_proj_dw_args)
            _encoded = self._apply_attention_dot(_query, _key, _value, _attn_mask, _pre_proj_dw_args, _post_proj_dw_args)
            encoded = encoded.at[:, start : stop].set(_encoded)
    return encoded

  def qk_product(self, query: Array, key: Array) -> Array:
    b, t, n, d = query.shape  
    n_kv = key.shape[-2]
    assert n_kv == self.num_kv_heads
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
  ):
    """Apply Attention."""
    if self.float32_qk_product:
      query = query.astype(jnp.float32)
      key = key.astype(jnp.float32)
    # bnts
    attn_weights = self.qk_product(query, key)
    attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
   
    if self.config.pre_compose:
       # 5 demonsion
      pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd = pre_proj_dw_args
      attn_weights = self.pre_proj(attn_weights, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)

    attn_weights = nn.with_logical_constraint(attn_weights, ('activation_batch', 'heads', 'activation_length', None),)
    # apply attention mask
    if attn_mask is not None:
      attn_weights = apply_mask_to_logits(attn_weights, attn_mask)
    if self.config.float32_logits:
          attn_weights = attn_weights.astype(jnp.float32)
    # normalize the attention weights
    probs = jax.nn.softmax(attn_weights).astype(self.dtype)
    probs = nn.with_logical_constraint(probs, ('activation_batch', 'heads', 'activation_length', None),)

    if self.post_compose:
      post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd = post_proj_dw_args
      probs = self.post_proj(probs, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)

    probs = nn.with_logical_constraint(probs, ('activation_batch', 'heads', 'activation_length', None),)
    # Casting softmaxt computation for float32 for model stability.
    probs = probs.astype(self.dtype)
    if attn_mask is not None:
      probs = jnp.where((attn_mask >= DEFAULT_MASK_VALUE * 0.5), probs, 0.)
    output = jnp.einsum('bnts,bsnh->btnh', probs, value)
    probs = nn.with_logical_constraint(probs, ('activation_batch', 'activation_length', 'heads', 'mlp'),)
    return output