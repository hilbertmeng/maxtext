"""PyTorch BAM full-M generalized-read decoder (inference port of the MaxText implementation).

Port of the executed path of MaxText `BamAttention` / `SubDecoderLayer` / `EmbeddingBamWrite`
(runtime b83223f) for `BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile`;
the fused Pallas v8t core computes the same equations. Notation per token: M [K, V] (K = data/column
output, V = address), N heads.

  read key   r = norm(up(gelu(down(x))) + pre_bias)                     (one shared `down`)
  Q/K read   col = M r * 0.1 sigmoid(gate), truncated to 72 columns, + static read M[:72] s * g_static
  VO read    y = M norm(r_vo);  V = 0.1 sigmoid(lv) y + M s_v g_v ;  O_loc = 0.1 sigmoid(lo) y + M s_o g_o
  attention  Q = [col_q | RoPE(Wq x)], K = [col_k | RoPE(Wk x)], V as above;  o = softmax(QK^T/sqrt d) V + O_loc
  write      M += sum_n sigmoid(gw) norm(o[:K]) (x) norm(P_loc(x))   (+ the MLP write on its layers)

M is per token (it never mixes positions), so incremental decoding caches only attention K/V.

`config.bam_enabled=False` selects the parameter-matched MHA control (`Llama2XLPropTruePileMHA`,
runtime e30c1b8): the same decoder with standard Q/K/V projections, RoPE over the whole head and
no M stream, as in MaxText where both models share the fusion decoder layer.
"""
import importlib
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import GenerationMixin, PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:   # Triton split-KV decode kernel (CUDA); without it decoding falls back to SDPA
  from .bam_decode_attention import decode_attention as _decode_attention
except ImportError:
  try:
    _decode_attention = importlib.import_module('bam_decode_attention').decode_attention
  except Exception:  # pylint: disable=broad-except
    _decode_attention = None

try:   # Triton per-token M read/write kernels (CUDA); without them the batched-GEMM path is used
  from .bam_m_kernels import (bam_decode_read as _bam_decode_read, bam_read as _bam_read,
                              bam_rope_tail as _bam_rope_tail, bam_write as _bam_write,
                              bam_write_ohead as _bam_write_ohead, bam_gemv as _bam_gemv)
except ImportError:
  try:
    _m = importlib.import_module('bam_m_kernels')
    _bam_read, _bam_write, _bam_decode_read, _bam_rope_tail, _bam_write_ohead, _bam_gemv = (
        _m.bam_read, _m.bam_write, _m.bam_decode_read, _m.bam_rope_tail, _m.bam_write_ohead, _m.bam_gemv)
  except Exception:  # pylint: disable=broad-except
    _bam_read = _bam_write = _bam_decode_read = _bam_rope_tail = _bam_write_ohead = _bam_gemv = None

DECODE_TOKENS = 64    # below this many tokens per step: per-arm read programs with RoPE fused
FUSED_UP_TOKENS = 4   # up to this many: the read kernel also applies GELU + the key up-projection (A800: one
                      # token 4.8 us; at 16 tokens re-reading the up weights per token loses to one GEMV)

try:
  from .configuration_bam import BamConfig
except ImportError:      # repo scripts import this file as a top-level module (not via trust_remote_code)
  BamConfig = importlib.import_module('configuration_bam').BamConfig


def position_ids_of(positions):
  return positions[0] if isinstance(positions, tuple) else positions


def rms_norm(x, eps):
  """Parameter-free RMS norm over the last axis with FP32 statistics, returned in x.dtype."""
  f = x.float()
  return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def gelu(x):
  return F.gelu(x, approximate='tanh')     # flax nn.gelu default


class BamRMSNorm(nn.Module):
  """MaxText RMSNorm: zero-initialized scale applied as (1 + scale)."""

  def __init__(self, dim, eps):
    super().__init__()
    self.weight = nn.Parameter(torch.zeros(dim))
    self.eps = eps

  def forward(self, x):
    return rms_norm(x, self.eps) * (self.weight.to(x.dtype) + 1)


class HeadLinear(nn.Linear):
  """Linear map to [..., *out_shape] (MaxText DenseGeneral with multi-axis features)."""

  def __init__(self, in_features, out_shape, bias=False):
    self.out_shape = tuple(out_shape)
    super().__init__(in_features, math.prod(self.out_shape), bias=bias)

  def forward(self, x):
    return super().forward(x).unflatten(-1, self.out_shape)


class BamRotary(nn.Module):
  """MaxText RotaryEmbedding (half-split, not interleaved) over the standard Q/K tail.

  Timescales are Python constants (not a buffer: non-persistent buffers stay uninitialized when
  transformers >= 5 builds the model on the meta device; not computed in forward: torch.compile would
  inline the power into every rotated element, 4x slower for the MHA control)."""

  def __init__(self, dim, min_timescale, max_timescale):
    super().__init__()
    self.timescale = [min_timescale * (max_timescale / min_timescale) ** (2 * i / dim) for i in range(dim // 2)]

  def table(self, positions, dtype):
    """(position ids, cos, sin [B, T, 1, D/2]): computed once per forward and shared by all layers."""
    timescale = torch.tensor(self.timescale, device=positions.device, dtype=torch.float32)
    angle = positions.float()[:, :, None, None] / timescale
    return positions, torch.cos(angle).to(dtype), torch.sin(angle).to(dtype)

  def forward(self, x, positions):
    """x [B, T, N, D]; positions [B, T] ids or a (ids, cos, sin) table from `table`."""
    if not isinstance(positions, tuple):
      positions = self.table(positions, x.dtype)
    _, cos, sin = positions
    a, b = x.chunk(2, dim=-1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), dim=-1)


class GatedLinear(nn.Module):
  """Gate logits = W x + b (the bias kept separately as in MaxText)."""

  def __init__(self, in_features, out_shape):
    super().__init__()
    self.proj = HeadLinear(in_features, out_shape)
    self.bias = nn.Parameter(torch.zeros(out_shape))

  def forward(self, x):
    return self.proj(x) + self.bias.to(x.dtype)


ALIGN = 64    # GEMM dims padded to a multiple of 64: misaligned (e.g. 6140, 1940) halve cuBLAS throughput


def pad_rows(w, multiple=ALIGN):
  extra = -w.shape[0] % multiple
  return torch.cat((w, w.new_zeros((extra,) + w.shape[1:]))) if extra else w


def fused_linear(*parts):
  """One nn.Linear computing [p(x) for p in parts] (same input); biases are added in its epilogue.
  The output is zero-padded to a multiple of ALIGN; callers split off the real width.

  parts: (weight [out, in], bias [out] or None) pairs."""
  weight = pad_rows(torch.cat([w for w, _ in parts], 0))
  bias = pad_rows(torch.cat([b if b is not None else torch.zeros(w.shape[0], dtype=w.dtype, device=w.device)
                             for w, b in parts]))
  layer = nn.Linear(weight.shape[1], weight.shape[0], bias=any(b is not None for _, b in parts),
                    device=weight.device, dtype=weight.dtype)
  layer.weight.data.copy_(weight)
  if layer.bias is not None:
    layer.bias.data.copy_(bias)
  return layer


def batched_write(M, content, address):
  """M + sum_n content[n,k] address[n,v] per token, as one baddbmm. M [B,T,K,V]."""
  b, t = M.shape[:2]
  return torch.baddbmm(M.reshape(b * t, *M.shape[2:]), content.reshape(b * t, -1, content.shape[-1]).transpose(1, 2),
                       address.reshape(b * t, -1, address.shape[-1])).reshape(M.shape)


def pending_args(flat_m, groups, b, t, eps):
  """Kernel arguments of the previous layer's deferred write groups (content or (attention, local_o) pair,
  gate logits, raw address); dummies when there is none."""
  if not groups:
    d = flat_m
    return (d, d, d[..., 0], d, d, d[..., 0], d, 0, False, eps)
  rows = lambda z: z.reshape(b * t, *z.shape[2:])
  content, gate, address = groups[0]
  has_local = isinstance(content, tuple)
  c1, l1 = (rows(content[0]), rows(content[1])) if has_local else (rows(content), rows(content))
  g1, a1 = gate.reshape(b * t, -1), rows(address)
  if len(groups) > 1:
    c2, g2, a2 = rows(groups[1][0]), groups[1][1].reshape(b * t, -1), rows(groups[1][2])
  else:
    c2, g2, a2 = c1, g1, a1
  return (c1, l1, g1, a1, c2, g2, a2, len(groups), has_local, eps)


def triton_write(M, *groups):
  """groups of raw (content [B,T,N,K] or (attention, local_o) pair, gate logits [B,T,N], address [B,T,N,V]);
  one or two groups (attention write, MLP write)."""
  b, t = M.shape[:2]
  rows = lambda z: z.reshape(b * t, *z.shape[2:])
  content, gate, address = groups[0]
  has_local = isinstance(content, tuple)
  c1, l1 = (rows(content[0]), rows(content[1])) if has_local else (rows(content), rows(content))
  g1, a1 = gate.reshape(b * t, -1), rows(address)
  if len(groups) > 1:
    c2, g2, a2 = rows(groups[1][0]), groups[1][1].reshape(b * t, -1), rows(groups[1][2])
  else:
    c2, g2, a2 = c1, g1, a1
  out = _bam_write(rows(M), c1, l1, g1, a1, c2, g2, a2, has_local, len(groups) > 1, 1e-6)
  return out.reshape(M.shape)


def gate_part(g):
  """(weight, bias) of a GatedLinear."""
  return g.proj.weight, g.bias.reshape(-1)


class PaddedHeadLinear(nn.Module):
  """HeadLinear with its output zero-padded to ALIGN for the GEMM, sliced back (inference)."""

  def __init__(self, layer: HeadLinear):
    super().__init__()
    self.linear = fused_linear((layer.weight, layer.bias))
    self.out_features, self.out_shape = layer.out_features, layer.out_shape

  def forward(self, x):
    return self.linear(x)[..., :self.out_features].unflatten(-1, self.out_shape)


class BamEmbeddingWrite(nn.Module):
  """Initial M from one gated outer-product write of the token embedding."""

  def __init__(self, config: BamConfig):
    super().__init__()
    n, h = config.num_attention_heads, config.hidden_size
    self.data = HeadLinear(h, (n, config.bam_k))
    self.address_down = nn.Linear(h, config.embedding_write_bottleneck_dim, bias=False)
    self.address_up = HeadLinear(config.embedding_write_bottleneck_dim, (n, config.bam_v), bias=True)
    self.gate = GatedLinear(h, (n,))
    self.eps = config.rms_norm_eps

  def forward(self, x):
    gate = torch.sigmoid(self.gate(x))
    data = rms_norm(self.data(x), self.eps)
    address = rms_norm(self.address_up(gelu(self.address_down(x))), self.eps)
    return torch.einsum('btnk,btnv->btkv', gate[..., None] * data, address)


class StaticKVCache(nn.Module):
  """Preallocated K/V [B,N,S,D] per layer with a device-side write position.

  Shapes and addresses stay fixed and the position advances in place, so a decode step can be
  captured by CUDA graphs (torch.compile mode='reduce-overhead' / 'max-autotune').
  """

  def __init__(self, config: BamConfig, batch_size, max_len, dtype, device):
    super().__init__()
    shape = (batch_size, config.num_attention_heads, max_len, config.head_dim)
    for i in range(config.num_hidden_layers):
      self.register_buffer(f'k{i}', torch.zeros(shape, dtype=dtype, device=device), persistent=False)
      self.register_buffer(f'v{i}', torch.zeros(shape, dtype=dtype, device=device), persistent=False)
    self.register_buffer('pos', torch.zeros((), dtype=torch.long, device=device), persistent=False)
    self.max_len = max_len

  def reset(self, pos=0):
    self.pos.fill_(pos)

  def attend(self, q, k, v, layer_idx, scale=1.0):
    """q/k/v [B,N,T,D]; writes k/v at pos..pos+T and attends causally over the valid prefix."""
    t = q.shape[2]
    idx = self.pos + torch.arange(t, device=q.device)
    kb, vb = getattr(self, f'k{layer_idx}'), getattr(self, f'v{layer_idx}')
    kb.index_copy_(2, idx, k)
    vb.index_copy_(2, idx, v)
    if t == 1 and q.is_cuda and _decode_attention is not None:     # split-KV Triton decode kernel
      lengths = (self.pos + 1).to(torch.int32).repeat(q.shape[0])
      return _decode_attention(q * scale if scale != 1.0 else q, kb, vb, lengths)
    mask = torch.arange(self.max_len, device=q.device)[None, :] <= idx[:, None]     # [T,S]
    return F.scaled_dot_product_attention(q, kb, vb, attn_mask=mask[None, None], scale=scale)


def causal_attention(q, k, v, layer_idx, attention_mask=None, past_key_value: Optional[Cache] = None, scale=1.0):
  """q/k/v [B,T,N,D]; logits are q.k * scale (1.0 when q is pre-scaled); returns [B,T,N,D].
  Appends k/v to the cache."""
  q, k, v = (z.transpose(1, 2) for z in (q, k, v))           # [B,N,T,D]
  if isinstance(past_key_value, StaticKVCache):
    return past_key_value.attend(q, k, v, layer_idx, scale).transpose(1, 2)
  if past_key_value is not None:
    k, v = past_key_value.update(k, v, layer_idx)
  if attention_mask is None:
    y = F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] > 1, scale=scale)
  else:
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=scale)
  return y.transpose(1, 2)


class MhaAttention(nn.Module):
  """MaxText `Attention` as executed by the MHA control: Q/K/V projections, full-head RoPE."""

  def __init__(self, config: BamConfig, layer_idx: int):
    super().__init__()
    self.layer_idx = layer_idx
    h, n, d = config.hidden_size, config.num_attention_heads, config.head_dim
    self.head_dim = d
    self.q_proj = HeadLinear(h, (n, d))
    self.k_proj = HeadLinear(h, (n, d))
    self.v_proj = HeadLinear(h, (n, d))
    self.rotary = BamRotary(d, config.rope_min_timescale, config.rope_max_timescale)
    self.o_proj = nn.Linear(n * d, h, bias=False)
    self.fused_qkv = None

  def fuse(self):
    """Q/K/V as one GEMM (inference; the separate parameters stay for checkpoint I/O)."""
    self.fused_qkv = fused_linear(*[(m.weight, None) for m in (self.q_proj, self.k_proj, self.v_proj)])

  def forward(self, x, M, positions, attention_mask=None, past_key_value=None, defer_write=False):
    if self.fused_qkv is not None:
      q, k, v = self.fused_qkv(x)[..., :3 * self.q_proj.out_features].unflatten(-1, (3, -1, self.head_dim)).unbind(-3)
    else:
      q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    q = self.rotary(q, positions)
    k = self.rotary(k, positions)
    if self.fused_qkv is not None:    # inference path: the softmax scale is applied by the attention kernel
      y = causal_attention(q, k, v, self.layer_idx, attention_mask, past_key_value, scale=self.head_dim ** -0.5)
      return self.o_proj(y.flatten(-2)), M
    q = q / torch.tensor(math.sqrt(self.head_dim), dtype=q.dtype)
    y = causal_attention(q, k, v, self.layer_idx, attention_mask, past_key_value)
    return self.o_proj(y.flatten(-2)), M


class BamAttention(nn.Module):

  def __init__(self, config: BamConfig, layer_idx: int):
    super().__init__()
    self.layer_idx = layer_idx
    self.last_layer = layer_idx == config.num_hidden_layers - 1     # its write has no reader
    h, n, d = config.hidden_size, config.num_attention_heads, config.head_dim
    v, r = config.bam_v, config.read_bottleneck_dim
    self.num_heads, self.head_dim, self.bam_k = n, d, config.bam_k
    self.qk_cols = config.qk_col_dim
    self.read_eps, self.read_scale = config.read_key_eps, config.read_key_scale
    self.eps = config.rms_norm_eps
    self.static_gate_init = config.static_read_gate_init
    # Standard (rotated) Q/K tail.
    self.q_proj = HeadLinear(h, (n, config.standard_qk_dim))
    self.k_proj = HeadLinear(h, (n, config.standard_qk_dim))
    self.rotary = BamRotary(config.standard_qk_dim, config.rope_min_timescale, config.rope_max_timescale)
    assert config.rope_min_timescale == 1.0       # the Triton read applies max_ts ** (2i / R)
    self.rotary_max_timescale = float(config.rope_max_timescale)
    # Read keys: shared GELU bottleneck, per-arm up projection + pre-bias.
    self.read_down = nn.Linear(h, r, bias=False)
    self.q_read_up = HeadLinear(r, (n, v))
    self.k_read_up = HeadLinear(r, (n, v))
    self.vo_read_up = HeadLinear(r, (n, v))
    self.q_read_pre_bias = nn.Parameter(torch.zeros(n, v))
    self.k_read_pre_bias = nn.Parameter(torch.zeros(n, v))
    self.vo_read_pre_bias = nn.Parameter(torch.zeros(n, v))
    # Read gates (dynamic) and static-read gates.
    self.q_read_gate = GatedLinear(h, (n,))
    self.k_read_gate = GatedLinear(h, (n,))
    self.v_read_gate = GatedLinear(h, (n,))
    self.o_read_gate = GatedLinear(h, (n,))
    self.q_static_gate = GatedLinear(h, (n,))
    self.k_static_gate = GatedLinear(h, (n,))
    self.v_static_gate = GatedLinear(h, (n,))
    self.o_static_gate = GatedLinear(h, (n,))
    # Static read keys [V, N].
    self.q_static_key = nn.Parameter(torch.zeros(v, n))
    self.k_static_key = nn.Parameter(torch.zeros(v, n))
    self.v_static_key = nn.Parameter(torch.zeros(v, n))
    self.o_static_key = nn.Parameter(torch.zeros(v, n))
    # Attention write.
    self.write_address_down = nn.Linear(h, config.write_bottleneck_dim, bias=False)
    self.write_address_up = HeadLinear(config.write_bottleneck_dim, (n, v), bias=True)
    self.write_gate = GatedLinear(h, (n,))
    self.o_proj = nn.Linear(n * d, h, bias=False)
    self.fused_x = self.fused_up = None

  GATES = ('q_read', 'k_read', 'v_read', 'o_read', 'q_static', 'k_static', 'v_static', 'o_static')

  def fuse(self, triton_m=False):
    """Every projection of x as one GEMM, the three read-key ups as another (inference).
    triton_m: per-token M read/write in the fused Triton kernels (bam_m_kernels)."""
    n = self.num_heads
    self.triton_m = triton_m and _bam_read is not None
    self._input_norm = ()     # (input RMSNorm,) when the layer passes x un-normalized (set by the layer)
    self.register_buffer('static_keys', torch.stack(
        [self.q_static_key, self.k_static_key, self.v_static_key, self.o_static_key]).contiguous(), persistent=False)
    gates = [gate_part(getattr(self, g + '_gate')) for g in self.GATES]
    self.fused_x = fused_linear((self.q_proj.weight, None), (self.k_proj.weight, None), (self.read_down.weight, None),
                                *gates, (self.write_address_down.weight, None), gate_part(self.write_gate))
    self._x_sizes = [self.q_proj.out_features, self.k_proj.out_features, self.read_down.out_features,
                     8 * n, self.write_address_down.out_features, n]
    self._x_sizes.append(self.fused_x.out_features - sum(self._x_sizes))      # alignment padding
    self.fused_up = fused_linear(*[(up.weight, b.reshape(-1)) for up, b in (
        (self.q_read_up, self.q_read_pre_bias), (self.k_read_up, self.k_read_pre_bias),
        (self.vo_read_up, self.vo_read_pre_bias))])
    self.padded_write_address_up = PaddedHeadLinear(self.write_address_up)

  def _projections(self, x):
    """Returns q_std, k_std [.,N,Ds]; read keys q/k/vo [.,N,V]; gate logits {name: [.,N]};
    write address hidden [., Rw] (pre-GELU) and write-gate logits [.,N]."""
    n = self.num_heads
    if self.fused_x is not None:
      if self._input_norm and x.is_cuda and x.shape[0] * x.shape[1] == 1:   # decode bs1: norm in the GEMV
        norm = self._input_norm[0]
        xs = _bam_gemv(x.reshape(-1), None, norm.weight, self.fused_x.weight, self.fused_x.bias, None,
                       norm.eps).reshape(1, 1, -1)
      else:
        xs = self.fused_x(self._input_norm[0](x) if self._input_norm else x)
      qs, ks, down, gates, wdown, wgate, _ = xs.split(self._x_sizes, -1)
      decode = self.triton_m and x.is_cuda and x.shape[0] * x.shape[1] <= FUSED_UP_TOKENS
      # the decode kernel applies GELU and the key up-projection itself
      keys = None if decode else self.fused_up(gelu(down))[..., :3 * self.q_read_up.out_features]
      raw = (keys, gates, xs[..., :2 * self._x_sizes[0]], down)  # views for the Triton read (std Q|K adjacent)
      keys = None if decode else keys.unflatten(-1, (3, n, -1)).unbind(-3)
      gates = dict(zip(self.GATES, gates.unflatten(-1, (8, n)).unbind(-2)))
      return qs.unflatten(-1, (n, -1)), ks.unflatten(-1, (n, -1)), keys, gates, wdown, wgate, raw
    hidden = gelu(self.read_down(x))
    keys = (self.q_read_up(hidden) + self.q_read_pre_bias.to(x.dtype),
            self.k_read_up(hidden) + self.k_read_pre_bias.to(x.dtype),
            self.vo_read_up(hidden) + self.vo_read_pre_bias.to(x.dtype))
    gates = {g: getattr(self, g + '_gate')(x) for g in self.GATES}
    return self.q_proj(x), self.k_proj(x), keys, gates, self.write_address_down(x), self.write_gate(x), None

  def _static_read(self, M, key, gate):
    """M [B,T,K',V] . s [V,N] -> [B,T,N,K'], times 1.1 sigmoid(gate) / init."""
    read = torch.einsum('btkv,vn->btnk', M, key.to(M.dtype))
    return read * (1.1 * torch.sigmoid(gate) / self.static_gate_init)[..., None]

  def _qk_read(self, M, key, gate, static_key, static_gate):
    r = self.read_scale * torch.sigmoid(gate)[..., None] * rms_norm(key, self.read_eps)
    col = torch.einsum('btkv,btnv->btnk', M, r)[..., :self.qk_cols]
    return col + self._static_read(M[:, :, :self.qk_cols], static_key, static_gate)

  def _stacked_reads(self, M, q_key, k_key, vo_key, g):
    """All reads of the layer as one per-token GEMM Y_t = X_t M_t^T (every read is linear in M_t):
    rows of X_t [4N, V] are q/k/v/o; static reads enter as gate * static key."""
    sg = lambda name, key: (1.1 * torch.sigmoid(g[name]) / self.static_gate_init)[..., None] * key.T.to(M.dtype)
    dyn = lambda name, key: (self.read_scale * torch.sigmoid(g[name]))[..., None] * key
    vo = rms_norm(vo_key, self.read_eps)
    x = torch.stack((dyn('q_read', rms_norm(q_key, self.read_eps)) + sg('q_static', self.q_static_key),
                     dyn('k_read', rms_norm(k_key, self.read_eps)) + sg('k_static', self.k_static_key),
                     dyn('v_read', vo) + sg('v_static', self.v_static_key),
                     dyn('o_read', vo) + sg('o_static', self.o_static_key)), dim=-3)       # [B,T,4,N,V]
    b, t = M.shape[:2]
    y = torch.matmul(x.reshape(b * t, -1, x.shape[-1]), M.reshape(b * t, *M.shape[2:]).transpose(1, 2))
    return y.reshape(b, t, 4, self.num_heads, -1).unbind(2)                                 # 4 x [B,T,N,K]

  def forward(self, x, M, positions, attention_mask=None, past_key_value: Optional[Cache] = None,
              defer_write=False, residual=None):
    """x [B,T,H] (pre-normed unless the layer set _input_norm), M [B,T,K,V]. Returns (out, M_out or write
    factors); out includes `residual` when one is given (decode bs1, Triton path)."""
    qs, ks, keys, g, wdown, wgate, raw = self._projections(x)
    if self.fused_x is not None and self.triton_m and x.is_cuda:     # fused Triton M kernels
      # M arrives as (M, deferred write groups of the previous layer); the read applies them (one pass over
      # M per layer) and this layer's write is deferred in turn. The last layer's write has no reader.
      M, groups = M if isinstance(M, tuple) else (M, None)
      b, t = x.shape[:2]
      n = self.num_heads
      std = raw[2].reshape(b * t, 2, n, -1)                       # unrotated standard Q/K, one view
      flat_m = M.reshape(b * t, *M.shape[2:])
      gates = raw[1].reshape(b * t, 8, n)
      decode = b * t < DECODE_TOKENS
      if groups and not decode:    # prefill: a separate write kernel keeps the 3-stage read pipeline (A800)
        M = triton_write(M, *groups)
        flat_m, groups = M.reshape(b * t, *M.shape[2:]), None
      pend = pending_args(flat_m, groups, b, t, self.eps)
      address = None
      if decode and b * t <= FUSED_UP_TOKENS:   # + GELU, key up-projection, and this layer's write address
        wa = self.padded_write_address_up.linear
        y, m_new, address = _bam_decode_read(
            raw[3].reshape(b * t, -1), self.fused_up.weight, self.fused_up.bias, flat_m, gates, self.static_keys,
            std, position_ids_of(positions), self.read_eps, self.read_scale, self.static_gate_init, self.qk_cols,
            1.0 / math.sqrt(self.head_dim), self.rotary_max_timescale, *pend, wdown.reshape(b * t, -1),
            wa.weight, wa.bias)
        address = address.reshape(b, t, n, -1)
      else:                                     # decode: RoPE/q scale in the kernel; prefill: in torch
        y, m_new = _bam_read(flat_m, raw[0].reshape(b * t, 3, n, -1), gates, self.static_keys, std,
                             position_ids_of(positions).reshape(-1).contiguous(), self.read_eps, self.read_scale,
                             self.static_gate_init, self.qk_cols, 1.0 / math.sqrt(self.head_dim),
                             self.rotary_max_timescale, decode, *pend)
      if groups:
        M = m_new.reshape(M.shape)
      if not decode:    # RoPE tail written in place (no q/k concatenation); the softmax scale goes to SDPA
        _, cos, sin = positions if isinstance(positions, tuple) else self.rotary.table(positions, y.dtype)
        _bam_rope_tail(y, std, cos.reshape(b * t, -1).contiguous(), sin.reshape(b * t, -1).contiguous(),
                       self.qk_cols)
      q, k, v, local_o = y.reshape(b, t, 4, n, -1).unbind(2)
      attn = causal_attention(q, k, v, self.layer_idx, attention_mask, past_key_value,
                              scale=1.0 if decode else self.head_dim ** -0.5)
      if address is None:
        address = self.padded_write_address_up(gelu(wdown))
      if decode:      # the local-O add happens inside the next read kernel
        if residual is not None:    # bs1: (attn + local_o) -> o_proj -> + residual in one GEMV
          out = _bam_gemv(attn.reshape(-1), local_o.reshape(-1), None, self.o_proj.weight, None,
                          residual.reshape(-1), 0.).reshape(residual.shape)
        else:
          out = self.o_proj((attn + local_o).flatten(-2))
        return out, (M, [((attn, local_o), wgate, address)])
      if not defer_write and not self.last_layer:
        # prefill, no MLP write: write now; the write kernel also emits o_head = attn + local_o
        m_out, o_head = _bam_write_ohead(M.reshape(b * t, *M.shape[2:]), attn.reshape(b * t, n, -1),
                                         local_o.reshape(b * t, n, -1), wgate.reshape(b * t, -1),
                                         address.reshape(b * t, n, -1), self.eps)
        return self.o_proj(o_head.reshape(b, t, -1)), (m_out.reshape(M.shape), [])
      o_head = attn + local_o
      return self.o_proj(o_head.flatten(-2)), (M, [] if self.last_layer else [(o_head, wgate, address)])
    q_key, k_key, vo_key = keys
    if self.fused_x is not None:      # inference path: stacked reads, one batched write
      yq, yk, v, local_o = self._stacked_reads(M, q_key, k_key, vo_key, g)
      q = torch.cat((yq[..., :self.qk_cols], self.rotary(qs, positions)), dim=-1)
      k = torch.cat((yk[..., :self.qk_cols], self.rotary(ks, positions)), dim=-1)
      q = q / torch.tensor(math.sqrt(self.head_dim), dtype=q.dtype)
      o_head = causal_attention(q, k, v, self.layer_idx, attention_mask, past_key_value) + local_o
      factors = (torch.sigmoid(wgate)[..., None] * rms_norm(o_head[..., :self.bam_k], self.eps),
                 rms_norm(self.padded_write_address_up(gelu(wdown)), self.eps))
      out = self.o_proj(o_head.flatten(-2))
      return out, factors if defer_write else batched_write(M, *factors)
    q = torch.cat((self._qk_read(M, q_key, g['q_read'], self.q_static_key, g['q_static']),
                   self.rotary(qs, positions)), dim=-1)
    k = torch.cat((self._qk_read(M, k_key, g['k_read'], self.k_static_key, g['k_static']),
                   self.rotary(ks, positions)), dim=-1)
    # One ungated VO column read, then destination-specific gates; plus static V/O reads.
    vo = torch.einsum('btkv,btnv->btnk', M, rms_norm(vo_key, self.read_eps))
    v = vo * (self.read_scale * torch.sigmoid(g['v_read']))[..., None]
    v = v + self._static_read(M, self.v_static_key, g['v_static'])
    local_o = vo * (self.read_scale * torch.sigmoid(g['o_read']))[..., None]
    local_o = local_o + self._static_read(M, self.o_static_key, g['o_static'])
    q = q / torch.tensor(math.sqrt(self.head_dim), dtype=q.dtype)

    o_head = causal_attention(q, k, v, self.layer_idx, attention_mask, past_key_value) + local_o

    content = rms_norm(o_head[..., :self.bam_k], self.eps)
    address = rms_norm(self.write_address_up(gelu(wdown)), self.eps)
    gated = torch.sigmoid(wgate)[..., None] * content
    out = self.o_proj(o_head.flatten(-2))
    if defer_write:
      return out, (gated, address)
    return out, M + torch.einsum('btnk,btnv->btkv', gated, address)


class BamMLP(nn.Module):

  def __init__(self, hidden_size, intermediate_size):
    super().__init__()
    self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)   # MaxText wi_0
    self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)     # wi_1
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)   # wo
    self.fused = None
    self._sizes = None
    self._norm = ()     # (post-attention RMSNorm,) when the layer passes x un-normalized (set by the layer)

  def fuse(self, extra=()):
    """gate/up as one GEMM with the intermediate width zero-padded to a multiple of ALIGN (silu(0) * 0 = 0,
    zero down columns: exact). Extra (weight, bias) projections of the same input (the MLP write's gate and
    address) form a second small GEMM: appended to the big one they leave a partial tile (A800: 874 us vs
    784 + 64 us at 4096 tokens)."""
    gate, up = pad_rows(self.gate_proj.weight), pad_rows(self.up_proj.weight)
    width = gate.shape[0]
    # One weight [gate | up | extra]: prefill runs the gate/up rows and the extra rows as two GEMMs (row
    # slices are views); decode (a GEMV, no tile waste) runs all rows at once and saves a launch.
    self.fused = fused_linear((gate, None), (up, None), *extra)
    self._width = width
    self._extra_sizes = [w.shape[0] for w, _ in extra]
    if extra:
      self._extra_sizes.append(self.fused.out_features - 2 * width - sum(self._extra_sizes))
    self._sizes = [width, width]
    down = self.down_proj.weight
    self.fused_down = nn.Linear(width, down.shape[0], bias=False, device=down.device, dtype=down.dtype)
    self.fused_down.weight.data.zero_()
    self.fused_down.weight.data[:, :down.shape[1]] = down

  def forward(self, x):
    """Returns (output, extra projections of x)."""
    if self.fused is None:
      return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)), ()
    w, b = self.fused.weight, self.fused.bias
    if self._norm and x.is_cuda and x.shape[0] * x.shape[1] == 1:    # decode bs1: norm in the GEMV
      norm = self._norm[0]
      h = _bam_gemv(x.reshape(-1), None, norm.weight, w, b, None, norm.eps).reshape(1, 1, -1)
      gate, up, *extra = h.split(self._sizes + self._extra_sizes, -1)
      return self.fused_down(F.silu(gate) * up), list(extra[:-1])
    if self._norm:
      x = self._norm[0](x)
    if not self._extra_sizes:
      gate, up = F.linear(x, w, b).split(self._sizes, -1)
      extra = ()
    elif x.shape[0] * x.shape[1] < DECODE_TOKENS:
      gate, up, *extra = F.linear(x, w, b).split(self._sizes + self._extra_sizes, -1)
      extra = extra[:-1]
    else:
      cut = 2 * self._width
      gate, up = F.linear(x, w[:cut], None if b is None else b[:cut]).split(self._sizes, -1)
      extra = F.linear(x, w[cut:], None if b is None else b[cut:]).split(self._extra_sizes, -1)[:-1]
    return self.fused_down(F.silu(gate) * up), list(extra)


class BamMLPWrite(nn.Module):
  """Second write of the layer's MLP output into M with its own gate and address."""

  def __init__(self, config: BamConfig):
    super().__init__()
    n = config.num_attention_heads
    self.gate = GatedLinear(config.hidden_size, (n,))
    self.address_down = nn.Linear(config.hidden_size, config.mlp_write_address_dim, bias=False)
    self.address_up = HeadLinear(config.mlp_write_address_dim, (n, config.bam_v), bias=True)
    self.num_heads, self.bam_k, self.eps = n, config.bam_k, config.rms_norm_eps

  def projections(self):
    """(weight, bias) pairs fused into the MLP input GEMM: gate logits, address_down."""
    self.padded_address_up = PaddedHeadLinear(self.address_up)
    return [gate_part(self.gate), (self.address_down.weight, None)]

  def forward(self, hidden, mlp_out, M, attention_factors, fused=()):
    logits, down = fused if fused else (self.gate(hidden), self.address_down(hidden))
    if fused and isinstance(attention_factors, tuple) and isinstance(attention_factors[1], list):
      # Triton path: (M, deferred groups); append the MLP write group, applied by the next layer's read
      M_state, groups = attention_factors
      mlp_address = self.padded_address_up(gelu(down))
      return (M_state, groups + [(mlp_out.unflatten(-1, (self.num_heads, self.bam_k)), logits, mlp_address)])
    gated, address = attention_factors
    content = rms_norm(mlp_out.unflatten(-1, (self.num_heads, self.bam_k)), self.eps)
    mlp_content = torch.sigmoid(logits)[..., None] * content
    mlp_address = rms_norm((self.padded_address_up if fused else self.address_up)(gelu(down)), self.eps)
    if fused:     # one batched write for both groups
      return batched_write(M, torch.cat((gated, mlp_content), -2), torch.cat((address, mlp_address), -2))
    return M + (torch.einsum('btnk,btnv->btkv', gated, address)
                + torch.einsum('btnk,btnv->btkv', mlp_content, mlp_address))


class BamDecoderLayer(nn.Module):

  def __init__(self, config: BamConfig, layer_idx: int):
    super().__init__()
    self.input_layernorm = BamRMSNorm(config.hidden_size, config.rms_norm_eps)
    self.self_attn = (BamAttention if config.bam_enabled else MhaAttention)(config, layer_idx)
    self.post_attention_layernorm = BamRMSNorm(config.hidden_size, config.rms_norm_eps)
    self.mlp = BamMLP(config.hidden_size, config.intermediate_sizes[layer_idx])
    self.mlp_write = BamMLPWrite(config) if layer_idx in config.mlp_write_layers else None

  def forward(self, x, M, positions, attention_mask=None, past_key_value=None):
    # decode bs1 on the Triton path: the norms run inside the GEMV kernels, o_proj adds the residual
    bs1 = bool(getattr(self.self_attn, '_input_norm', ())) and x.is_cuda and x.shape[0] * x.shape[1] == 1
    normed = x if getattr(self.self_attn, '_input_norm', ()) else self.input_layernorm(x)
    attn, M_or_factors = self.self_attn(normed, M, positions, attention_mask, past_key_value,
                                        defer_write=self.mlp_write is not None, **({'residual': x} if bs1 else {}))
    x = attn if bs1 else x + attn
    hidden = x if self.mlp._norm else self.post_attention_layernorm(x)
    mlp_out, extra = self.mlp(hidden)
    M = (M_or_factors if self.mlp_write is None
         else self.mlp_write(hidden, mlp_out, M, M_or_factors, extra))
    return mlp_out + x, M

  def fuse(self, triton_m=False):
    if isinstance(self.self_attn, BamAttention):
      self.self_attn.fuse(triton_m)
      if self.self_attn.triton_m and _bam_gemv is not None:
        self.self_attn._input_norm = (self.input_layernorm,)   # tuple: not registered as a submodule
        self.mlp._norm = (self.post_attention_layernorm,)
    else:
      self.self_attn.fuse()
    self.mlp.fuse(self.mlp_write.projections() if self.mlp_write is not None else ())


class BamPreTrainedModel(PreTrainedModel):
  config_class = BamConfig
  base_model_prefix = 'model'
  _no_split_modules = ['BamDecoderLayer']
  _supports_cache_class = True
  supports_gradient_checkpointing = False

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(0.0, self.config.initializer_range)
      if getattr(module, 'bias', None) is not None:
        module.bias.data.zero_()


class BamModel(BamPreTrainedModel):

  def __init__(self, config: BamConfig):
    super().__init__(config)
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self.embedding_write = BamEmbeddingWrite(config) if config.bam_enabled else None
    self.layers = nn.ModuleList(BamDecoderLayer(config, i) for i in range(config.num_hidden_layers))
    self.norm = BamRMSNorm(config.hidden_size, config.rms_norm_eps)
    self.post_init()

  def get_input_embeddings(self):
    return self.embed_tokens

  def set_input_embeddings(self, value):
    self.embed_tokens = value

  def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
              inputs_embeds=None, use_cache=None, output_hidden_states=None, return_dict=None, **kwargs):
    use_cache = self.config.use_cache if use_cache is None else use_cache
    return_dict = getattr(self.config, 'return_dict', True) if return_dict is None else return_dict
    x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
    b, t = x.shape[:2]
    if isinstance(past_key_values, StaticKVCache):     # fixed-shape path (CUDA graphs)
      cache = past_key_values
      positions = (cache.pos + torch.arange(t, device=x.device))[None].expand(b, t)
      positions = self.layers[0].self_attn.rotary.table(positions, x.dtype)
      M = self.embedding_write(x) if self.embedding_write is not None else None
      for layer in self.layers:
        x, M = layer(x, M, positions, None, cache)
      cache.pos.add_(t)
      x = self.norm(x)
      return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=cache)
    if use_cache and not isinstance(past_key_values, Cache):
      legacy = past_key_values
      past_key_values = DynamicCache()
      if legacy is not None:      # tuple of per-layer (key, value) (transformers 4.x legacy format)
        for i, (key, value) in enumerate(legacy):
          past_key_values.update(key, value, i)
    past = past_key_values.get_seq_length() if past_key_values is not None else 0
    if position_ids is None:
      if attention_mask is not None:      # left padding: positions count real tokens only
        position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -t:]
      else:
        position_ids = torch.arange(past, past + t, device=x.device)[None].expand(b, t)
    mask = None
    if attention_mask is not None or (past > 0 and t > 1):
      s = past + t
      keep = torch.ones(b, s, dtype=torch.bool, device=x.device) if attention_mask is None \
          else attention_mask[:, -s:].bool()
      causal = torch.arange(s, device=x.device)[None, :] <= torch.arange(past, s, device=x.device)[:, None]
      allowed = causal[None] & keep[:, None, :]
      allowed |= ~allowed.any(-1, keepdim=True)               # fully padded query rows: avoid NaN
      mask = allowed[:, None]                                  # [B,1,T,S] bool
    M = self.embedding_write(x) if self.embedding_write is not None else None
    hidden_states = (x,) if output_hidden_states else None
    positions = self.layers[0].self_attn.rotary.table(position_ids, x.dtype)    # shared by all layers
    for layer in self.layers:
      x, M = layer(x, M, positions, mask, past_key_values if use_cache else None)
      if output_hidden_states:
        hidden_states += (x,)
    x = self.norm(x)
    if not return_dict:
      return (x, past_key_values if use_cache else None, hidden_states)
    return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values if use_cache else None,
                                   hidden_states=hidden_states)


class BamForCausalLM(BamPreTrainedModel, GenerationMixin):
  _tied_weights_keys = []

  def __init__(self, config: BamConfig):
    super().__init__(config)
    self.model = BamModel(config)
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.post_init()

  def get_input_embeddings(self):
    return self.model.embed_tokens

  def set_input_embeddings(self, value):
    self.model.embed_tokens = value

  def get_output_embeddings(self):
    return self.lm_head

  @torch.no_grad()
  def fuse_projections(self, triton_m=True):
    """Inference-time GEMM fusion: every projection sharing an input becomes one GEMM (both models);
    BAM's per-token M reads/writes use the fused Triton kernels when available (triton_m)."""
    for layer in self.model.layers:
      layer.fuse(triton_m)
    return self

  def set_output_embeddings(self, new_embeddings):
    self.lm_head = new_embeddings

  def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
              inputs_embeds=None, labels=None, use_cache=None, output_hidden_states=None, return_dict=None,
              logits_to_keep=0, **kwargs) -> Union[Tuple, CausalLMOutputWithPast]:
    """logits_to_keep > 0 projects only the last positions (prefill for generation)."""
    return_dict = getattr(self.config, 'return_dict', True) if return_dict is None else return_dict
    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                         past_key_values=past_key_values, inputs_embeds=inputs_embeds, use_cache=use_cache,
                         output_hidden_states=output_hidden_states, return_dict=True)
    hidden = outputs.last_hidden_state
    logits = self.lm_head(hidden[:, -logits_to_keep:] if logits_to_keep else hidden)
    loss = None
    if labels is not None:
      loss = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)
    if not return_dict:
      out = (logits, outputs.past_key_values, outputs.hidden_states)
      return (loss,) + out if loss is not None else out
    return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values,
                                  hidden_states=outputs.hidden_states)

  def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None,
                                    inputs_embeds=None, cache_position=None, **kwargs):
    past = 0
    if past_key_values is not None:
      past = (past_key_values.get_seq_length() if isinstance(past_key_values, Cache)
              else past_key_values[0][0].shape[2])
      if past > 0:
        input_ids = input_ids[:, past:]
    position_ids = None
    if attention_mask is not None:
      position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)[:, -input_ids.shape[1]:]
    if inputs_embeds is not None and past == 0:
      model_inputs = {'inputs_embeds': inputs_embeds}
    else:
      model_inputs = {'input_ids': input_ids}
    model_inputs.update(past_key_values=past_key_values, attention_mask=attention_mask,
                        position_ids=position_ids, use_cache=kwargs.get('use_cache', True))
    return model_inputs

  def _reorder_cache(self, past_key_values, beam_idx):
    if isinstance(past_key_values, Cache):
      past_key_values.reorder_cache(beam_idx)
      return past_key_values
    return tuple(tuple(s.index_select(0, beam_idx.to(s.device)) for s in layer) for layer in past_key_values)
