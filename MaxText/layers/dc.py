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
from layers import accelerator

Dtype = Any
Array = common_types.Array
DType = common_types.DType
Mesh = common_types.Mesh
RMSNorm = normalizations.RMSNorm
Quant = quantizations.AqtQuantization
DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NormalInitializer = initializers.nd_dense_init_normal


def unbind(ary, n, axis=0):
  return [jnp.squeeze(a, axis=axis) for a in jnp.split(ary, n, axis=axis)]



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
  dc_share_all_dw_hidden: bool = False
  dc_share_prepost_dw_hidden: bool = False
  dc_dw2_zero_init: bool = False
  use_dw_bias: bool = False
  use_dd_bias: bool = False

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
      # for dc mosa 
      self.dw1_kernel = self.param('dw1_kernel',nn.with_logical_partitioning(NormalInitializer(math.sqrt(2.0 / (self.input_dim + self.dynamic_w_hidden_dim))), ('embed', None, 'heads', 'mlp')),
                        (self.input_dim, self.num_groups, self.n_splits, self.dynamic_w_hidden_dim), self.weight_dtype) # DGCK
      self.dw_hidden_activation = nn.gelu
      # self.dynamic_w_hidden_dim: num_heads_per_group * I * 2 = 32 * 2 * 2 = 128
      G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
      I = dynamic_hidden_dim * 2  # 2 * 2
      if self.dc_share_all_dw_hidden:
        shape = [G, 1, K * self.n_splits, I * self.n_splits, M]
      elif self.dc_share_prepost_dw_hidden:
        shape = [G, self.n_splits // 2, K * 2, I * 2, M]
      else:
        shape = [G, self.n_splits, K, I, M]
      kernel_init_shard = nn.with_logical_partitioning(NormalInitializer(self.dynamic_w_init), (None, 'data', 'fsdp', None, 'tensor'))
      if self.dc_dw2_zero_init:
        shape = (shape[0], shape[1], shape[2], shape[3]//2, shape[4]) 
        self.qkw1 = self.param('qkw1',kernel_init_shard, shape, self.weight_dtype)
        self.qkw2 = self.param('qkw2',nn.with_logical_partitioning(initializers.constant_init(0.0), (None, 'data', 'fsdp', None, 'tensor')), shape, self.weight_dtype)
      else:
        self.qkw = self.param('qkw',kernel_init_shard, shape, self.weight_dtype)
      
      if self.use_dw_bias:
        assert self.dc_share_prepost_dw_hidden
        bias_shape = [G, self.n_splits, I // 2, M] # CIM
        self.w1_bias =  self.param('w1_bias',nn.with_logical_partitioning(initializers.constant_init(0.0), (None, None, None, 'kv_heads')), bias_shape, self.weight_dtype)
        self.w2_bias =  self.param('w2_bias',nn.with_logical_partitioning(initializers.constant_init(0.0), (None, None, None, 'kv_heads')), bias_shape, self.weight_dtype)


    if self.dynamic_d_init is not None:
      self.dd = linears.DenseGeneral(
                        features=(self.num_groups, 
                        self.num_heads_per_group * self.n_splits), 
                        quant=self.quant,
                        kernel_init=NormalInitializer(self.dynamic_d_init), 
                        kernel_axes=('embed', None, 'mlp'),
                        **kwargs
                        )
      self.dd_kernel = self.param('dd_kernel',nn.with_logical_partitioning(NormalInitializer(self.dynamic_d_init), ('embed', None, 'mlp')),
                        (self.input_dim, self.num_groups, self.num_heads_per_group * self.n_splits), self.weight_dtype) # DGK
      if self.use_dd_bias: 
        bias_shape = [self.num_heads_per_group * self.n_splits]
        self.dd_bias =  self.param('dd_bias',nn.with_logical_partitioning(initializers.constant_init(0.0), (None,)), bias_shape, self.weight_dtype)


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
    qkw_kernel = jnp.asarray(self.qkw, self.dtype) if not self.dc_dw2_zero_init else jnp.asarray(jnp.concatenate([self.qkw1, self.qkw2], axis=-2), self.dtype) # lsp
    if self.n_splits == 2:
      skip_norm_tanh = False # default: False
      if len(query_vec.shape) == 4: # for dc mosa, BtGD
        dw_hidden = self.dw_hidden_activation(jnp.einsum('BTGD, DGCK->BTGCK', query_vec, self.dw1_kernel))   # BTG2,64
      else: # dcmha
        dw_hidden = self.dw_hidden_activation(self.dw1(query_vec))   # BTG2,64
      if self.dynamic_dropout_rate is not None:
        dw_hidden = self.dropout(dw_hidden, deterministic=self.deterministic)

      if self.dc_share_all_dw_hidden:
        dw_hidden = rearrange(dw_hidden, 'B T G C K -> B T G 1 (C K)')
        w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, qkw_kernel), 2, axis=-2)
        w1, w2 = [rearrange(w, 'B T G 1 (C I) M -> B T G C I M', C=self.n_splits) for w in [w1, w2]]
      elif self.dc_share_prepost_dw_hidden:
        dw_hidden = rearrange(dw_hidden, 'B T G (C P) K -> B T G C (P K)', P=2) # 
        w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, qkw_kernel), 2, axis=-2)  
        w1, w2 = [rearrange(w, 'B T G C (P I) M -> B T G (P C) I M', P=2) for w in [w1, w2]]
      else:
        # C: n_split,  K -> M
        w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, qkw_kernel), 2, axis=-2)
      if self.use_dw_bias:
        w1 = w1 + self.w1_bias[None,None]
        w2 = w2 + self.w2_bias[None,None]
      if not skip_norm_tanh:
        w1 = self.dw1_norm(w1)
      pre_w1, post_w1 = unbind(w1, 2, axis=3) # BTG2IM->[BTGIM]*2
      pre_w2, post_w2 = unbind(w2, 2, axis=3)

      if len(query_vec.shape) == 4: # for dc mosa, BtGD
        dd = jnp.einsum('BTGD,DGK->BTGK', query_vec, self.dd_kernel)
      else: #dcmha
        dd = self.dd(query_vec)
      if self.use_dd_bias:
        dd = dd + self.dd_bias[None, None]
      if not skip_norm_tanh:
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
      if self.dc_share_all_dw_hidden:
        dw_hidden = rearrange(dw_hidden, 'B T G C K -> B T G 1 (C K)')
        w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, qkw_kernel), 2, axis=-2)
        w1, w2 = [rearrange(w, 'B T G 1 (C I) M -> B T G C I M', C=self.n_splits) for w in [w1, w2]]
      elif self.dc_share_prepost_dw_hidden:
        dw_hidden = rearrange(dw_hidden, 'B T G (C P) K -> B T G C (P K)', P=2) # C means q/k , P means pre/post
        w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, qkw_kernel), 2, axis=-2)
        w1, w2 = [rearrange(w, 'B T G C (P I) M -> B T G (P C) I M', P=2) for w in [w1, w2]]
      else:
        # C: n_split,  K -> M
        w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, qkw_kernel), 2, axis=-2)        
      if self.use_dw_bias:
        w1 = w1 + self.w1_bias[None,None,None]
        w2 = w2 + self.w2_bias[None,None,None]
      w1 = self.dw1_norm(w1)
      pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, axis=3) # BTG4IM->[BTGIM]*4
      pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, axis=3)

      dd = self.dd(query_vec)
      if self.use_dd_bias:
        dd = dd + self.dd_bias[None, None]
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
        self.w1 = self.param('w1', NormalInitializer(scale), shape, self.weight_dtype) # GMI
        shape=[self.num_groups, self.hidden_dim, self.num_heads_per_group] # GIM
        scale = init_fn(self.num_heads_per_group, in_dim=self.hidden_dim)
        self.w2 = self.param('w2', NormalInitializer(scale), shape, self.weight_dtype)

  # @jax.jit(inline=False)
  def apply_sw(self, ret, exp, _inputs, w):
    # ret += jnp.einsum(exp, _inputs, w) # BGMTS,GMN->BGNTS
    s = ret.shape[-1]
    res = jnp.zeros(ret.shape, dtype=ret.dtype)
    s_chunk = 128
    for i in range(s//s_chunk):
      res = res.at[:,:,:,:,i*s_chunk:(i+1)*s_chunk].set(jnp.einsum(exp, _inputs[:,:,:,:,i*s_chunk:(i+1)*s_chunk], w))
    return ret + res

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

        # ret = self.apply_sw(ret, exp, _inputs, w)
        ret += jnp.einsum(exp, _inputs, w) if not self.left_mul else jnp.einsum(exp, w, _inputs) # BGMTS,GMN->BGNTS
        # ret += rearrange( rearrange(_inputs, 'B G M T S -> B T S (G M)') @ w[0], 'B T S (G N) -> B G N T S', G=self.num_groups)
        # BTSM, MN-> BTSN
        # assert not self.left_mul and self.num_groups == 1
        # ret += sum([ _inputs[:,:,hidx:hidx+1] * w[None,:,hidx,:,None,None] for hidx in range(w.shape[-1])])  #BG1TS,1GN11->BGNTS         # BGMTS, GMN, BGNTS
      else:
        if self.loop_over_dynamic_hd:
          for i in range(self.w1.shape[-1]):
            hidden = jnp.einsum('BGMTS,GMI->BGITS', inputs, self.w1[:,:,i:i+1])
            ret +=  jnp.einsum('BGITS,GIM->BGMTS', hidden, self.w2[:,i:i+1])
        else:
          hidden = jnp.einsum(exp, inputs, self.w1) if not self.left_mul else jnp.einsum(exp, self.w1, inputs)
          ret += jnp.einsum(exp, hidden, self.w2) if not self.left_mul else jnp.einsum(exp, self.w2, ret)

    if qw1 is not None: # BTGIM
      hidden_sym = 'I'; hidden_label = inputs_label.replace('M', 'I')  # 'BGITS'
      for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]):
        if w1 is None: continue
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

    if qdd is not None or kdd is not None:  # 对logits做二次修改
      for sym, dd in zip(['T', 'S'], [qdd, kdd]):
        if dd is None: continue
        dd_label = f'B{sym}GM'
        if sym == 'T' and self.query_wise or sym == 'S' and self.key_wise or \
              not self.query_wise and not self.key_wise:
          # 'BGMTS', B(T/S)GM
          dout = jnp.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs, dd)
          ret = ret + dout
    return jnp.reshape(ret, shape)  # BGMTS->BNTS


class AttentionOp(nn.Module):

  config: Any
  quant: Optional[Quant] = None
  sliding_window_size: int|None = None
  query_chunk_size: int|None = None
  num_query_heads: int = 0 
  num_kv_heads: int = 0
  key_wise: bool = False

  def setup(self):
    cfg = self.config
    self.head_dim = cfg.head_dim
    self.is_cross_attention = False
    self.num_groups = 1 if cfg.dc_num_groups is None else cfg.dc_num_groups
    self.dtype = cfg.dtype
    self.weight_dtype = cfg.weight_dtype
    self.precision = None
    self.deterministic = False
    self.dynamic_dropout_rate = 0.0
    self.pre_static_proj = cfg.pre_static_proj if cfg.pre_static_proj is not None else cfg.static_proj
    self.post_static_proj = cfg.post_static_proj if cfg.post_static_proj is not None else cfg.static_proj
    self.loop_over_dynamic_hd = cfg.loop_over_dynamic_hd
    self.float32_qk_product = cfg.float32_qk_product
    self.pre_compose = cfg.pre_compose
    self.post_compose = cfg.post_compose
    self.seperate_qk_dw_proj = cfg.seperate_qk_dw_proj


    cfg = self.config
    norm_kwargs = {
                "dtype": cfg.dtype,
                "weight_dtype": cfg.weight_dtype,
                "epsilon": cfg.normalization_layer_epsilon,
                }
    if self.config.use_dc_prenorm:
      self.dc_prenorm = normalizations.get_rmsnorm(name="dc_prenorm", **norm_kwargs)
    
    if self.config.dc_hidden_way == 'qk':
      assert self.config.seperate_qk_dw_proj

    input_dim = self.num_query_heads * self.head_dim
    I = 2
    num_heads_per_group = self.num_query_heads // self.num_groups
    dynamic_w_hidden_dim = num_heads_per_group * I * 2
    if cfg.pre_compose or cfg.post_compose:
      if self.is_cross_attention or self.seperate_qk_dw_proj:
        for name, use in [('q_dyn_w_proj', cfg.query_wise), ('k_dyn_w_proj', self.key_wise)]:
          if not use: continue
          setattr(self, name, DynamicWeightProjection(
            num_heads=self.num_query_heads, num_groups=self.num_groups,
            input_dim=self.config.emb_dim, n_splits=2,
            dynamic_w_init=math.sqrt(1 / dynamic_w_hidden_dim) * 2 / (num_heads_per_group + I) * 0.01,
            dynamic_d_init=math.sqrt(2 / (input_dim + num_heads_per_group)) * 0.005,
            dynamic_squeeze_ratio=num_heads_per_group // I,
            dynamic_w_hidden_dim=dynamic_w_hidden_dim,
            dtype=self.dtype, weight_dtype=self.weight_dtype, precision=self.precision,
            deterministic=self.deterministic,
            dynamic_dropout_rate=self.dynamic_dropout_rate,
            quant=self.quant,
            dc_share_all_dw_hidden=self.config.dc_share_all_dw_hidden,
            dc_share_prepost_dw_hidden=self.config.dc_share_prepost_dw_hidden,
            dc_dw2_zero_init=self.config.dc_dw2_zero_init,
            use_dw_bias=self.config.use_dw_bias,
            use_dd_bias=self.config.use_dd_bias,
          ))
      else:
        self.dyn_w_proj = DynamicWeightProjection(
          num_heads=self.num_query_heads, num_groups=self.num_groups,
          input_dim=self.config.emb_dim, n_splits=4,
          dynamic_w_init=math.sqrt(1 / dynamic_w_hidden_dim) * 2 / (num_heads_per_group + I) * 0.01,
          dynamic_d_init=math.sqrt(2 / (input_dim + num_heads_per_group)) * 0.005,
          dynamic_squeeze_ratio=num_heads_per_group // I,
          dynamic_w_hidden_dim=dynamic_w_hidden_dim,
          dtype=self.dtype, weight_dtype=self.weight_dtype, precision=self.precision,
          deterministic=self.deterministic,
          dynamic_dropout_rate=self.dynamic_dropout_rate,
          quant=self.quant,
          dc_share_all_dw_hidden=self.config.dc_share_all_dw_hidden,
          dc_share_prepost_dw_hidden=self.config.dc_share_prepost_dw_hidden,
          dc_dw2_zero_init=self.config.dc_dw2_zero_init,
          use_dw_bias=self.config.use_dw_bias,
          use_dd_bias=self.config.use_dd_bias,
          )

      self.pre_proj = CrossHeadProjection(
        dtype=self.dtype, 
        weight_dtype=self.weight_dtype, 
        precision=self.precision,
        num_heads=self.num_query_heads, 
        num_groups=self.num_groups,
        static_proj=self.pre_static_proj,
        query_wise=cfg.query_wise,
        key_wise=self.key_wise,
        query_input_dim=input_dim,
        key_input_dim=input_dim,
        dynamic_w_hidden_dim=dynamic_w_hidden_dim,
        loop_over_dynamic_hd=self.loop_over_dynamic_hd,
        squeeze_ratio=self.config.sw_squeeze_ratio,
        )

      self.post_proj = CrossHeadProjection(
        dtype=self.dtype, 
        weight_dtype=self.weight_dtype, 
        precision=self.precision,
        num_heads=self.num_query_heads, 
        num_groups=self.num_groups,
        static_proj=self.post_static_proj,
        query_input_dim=input_dim,
        key_input_dim=input_dim,
        query_wise=cfg.query_wise,
        key_wise=self.key_wise,
        dynamic_w_hidden_dim=dynamic_w_hidden_dim,
        loop_over_dynamic_hd=self.loop_over_dynamic_hd,
        squeeze_ratio=self.config.sw_squeeze_ratio,
        )

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
    hidden_states: Array = None, # inputs_m
    attn_mask=None,
):
    cfg = self.config

    if self.config.use_dc_prenorm:
      hidden_states = self.dc_prenorm(hidden_states)

    pre_proj_dw_args, post_proj_dw_args = (None, ) * 6, (None, ) * 6
    if cfg.pre_compose or cfg.post_compose:
        if hasattr(self, 'dyn_w_proj'):
            if self.config.dc_hidden_way == 'm':
              pre_proj_dw_args, post_proj_dw_args = self.dyn_w_proj(hidden_states)
            else:
              pre_proj_dw_args, post_proj_dw_args = self.dyn_w_proj(input_q)
        elif self.seperate_qk_dw_proj and (hasattr(self, 'q_dyn_w_proj') or hasattr(self, 'k_dyn_w_proj')):
          if self.config.query_wise:
            pre_q_dw_args, post_q_dw_args = self.q_dyn_w_proj(input_q)
          else:
            pre_q_dw_args, post_q_dw_args = (None, ) * 3, (None, ) * 3
          if self.key_wise:
            _key_vec = input_kv[0] if isinstance(input_kv, (tuple, list)) and len(input_kv) == 2 else input_kv
            pre_k_dw_args, post_k_dw_args = self.k_dyn_w_proj(_key_vec)
          else:
            pre_k_dw_args, post_k_dw_args = (None, ) * 3, (None, ) * 3
          pre_proj_dw_args = pre_q_dw_args[:2] + pre_k_dw_args[:2] + pre_q_dw_args[2:] + pre_k_dw_args[2:]
          post_proj_dw_args = post_q_dw_args[:2] + post_k_dw_args[:2] + post_q_dw_args[2:] + post_k_dw_args[2:]
        else:
            if hasattr(self, 'dyn_w_pre_proj'):
                pre_proj_dw_args = self.dyn_w_pre_proj(input_q)
            if hasattr(self, 'dyn_w_post_proj'):
                post_proj_dw_args = self.dyn_w_post_proj(input_kv)
    
    if self.config.ablate_kw:
      # pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd
      if self.config.ablate_prekdd:
        pre_proj_dw_args = pre_proj_dw_args[:2] + (None, None,) + (pre_proj_dw_args[-2], None,)
        post_proj_dw_args = post_proj_dw_args[:2] + (None, None,) + post_proj_dw_args[-2:]
      else:
        pre_proj_dw_args = pre_proj_dw_args[:2] + (None, None,) + pre_proj_dw_args[-2:]
        post_proj_dw_args = post_proj_dw_args[:2] + (None, None,) + post_proj_dw_args[-2:]

    # if self.config.kdd_only:
    #   # pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd
    #   if self.config.pre_compose:
    #     pre_proj_dw_args = (None, ) * 5 +  (pre_proj_dw_args[-1], )
    #     post_proj_dw_args = (None, ) * 6
    #   elif self.config.post_compose:
    #     pre_proj_dw_args = (None, ) * 6 
    #     post_proj_dw_args = (None, ) * 5 + (post_proj_dw_args[-1], )      
        
    outputs, _, _ = accelerator.QChunk(cfg, self.sliding_window_size, query_chunk_size=self.query_chunk_size)(query, key, value, decoder_segment_ids, model_mode, 
                            pre_proj_dw_args, post_proj_dw_args, 
                            pre_proj_layer=self.pre_proj,
                            post_proj_layer=self.post_proj,
                            attn_mask=attn_mask,
                            )
    return outputs
