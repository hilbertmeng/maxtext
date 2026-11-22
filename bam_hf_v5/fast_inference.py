"""Inference fast path for the Llama-backbone LoomFormer V5 port and its MHA control (BamV5ForCausalLM).

`build(hf_model)` wraps a loaded BF16 CUDA model in a module with the 2.8B port's interface:

  state = model.new_state(batch, max_len)      # static K/V (+ LoomFormer per-layer last-token M) caches
  logits = model.prefill(ids, state)           # [B, 1, vocab] logits of the last position
  logits = model.decode(ids[:, -1:], state)    # one token per row; state.pos advances in place

Shapes and buffer addresses stay fixed during decoding, so `decode` can be captured by CUDA graphs.
Components both models have use one implementation (inductor GEMMs, RMSNorm, SwiGLU, RoPE tables, the
static caches and decode attention of bam_hf_pythia, SDPA prefill attention); only LoomFormer-specific
parts differ. LoomFormer algebra (exact up to rounding, checked by check_fast.py):
- The whole-M RMS of the read view only scales the read outputs: read(Mh) = read(M~) / rms(M~).
- Every read is linear in the key: the Q/K pre-read gate, the V/O post-read gate and the static read
  combine into one key per arm and head, R = g rms(r) + g_s s, so all four arms are one per-token
  matmul M~ [K, V] @ R [V, 4N].
- The attention and MLP writes are one per-token matmul [K, 2N] @ [2N, V].
- Strict causality: queries 1..T-1 attend causally to keys 0..T-2; a decode step attends to the cache
  before appending its own key.
"""
import importlib.util
import math
import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

_PYTHIA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bam_hf_pythia')
sys.path.insert(0, _PYTHIA)
from decode_attention import attend_and_append  # noqa: E402

from decode_attention import decode_attention  # noqa: E402

_spec = importlib.util.spec_from_file_location('bam_hf_pythia_fast_inference', os.path.join(_PYTHIA, 'fast_inference.py'))
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
State, _cat_linear, _p, _prefill_attention = _shared.State, _shared._cat_linear, _shared._p, _shared._prefill_attention
_pad_rows = _shared._pad_rows
pythia_kernels = _shared.bam_kernels
_spec = importlib.util.spec_from_file_location('bam_hf_v5_kernels',
                                               os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bam_kernels.py'))
bam_kernels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bam_kernels)

KERNELS = os.environ.get('BAM_KERNELS', '1') == '1'


@torch.library.custom_op('bam_v5::gemv_cols', mutates_args=())
def gemv_cols(srcs: list[torch.Tensor], starts: list[int], ws: list[torch.Tensor], bs: list[Optional[torch.Tensor]],
              acts: list[int], br: int, bk: int, warps: int) -> list[torch.Tensor]:
  """bam_hf_pythia one-token grouped GEMV of x_i = srcs_i[:, starts_i:starts_i + K_i] (whole projection outputs
  passed in: column slices into a custom op are copied by inductor when their producer is a fused kernel)."""
  xs = [x[:, a:a + w.shape[1]] for x, a, w in zip(srcs, starts, ws)]
  return pythia_kernels.gemv(xs, ws, bs, acts, br, bk, warps)


@gemv_cols.register_fake
def _(srcs, starts, ws, bs, acts, br, bk, warps):
  return [x.new_empty((1, w.shape[0])) for x, w in zip(srcs, ws)]


class PaddedState(State):
  """State with cache rows padded to a multiple of 16 elements (32-byte sectors; both models): unaligned rows
  (e.g. 120 or 75 BF16 values) cost decode attention up to ~35% of its bandwidth (A800). k(i) / v(i) are the
  used views."""

  def __init__(self, layers, batch, heads, max_len, dqk, dv, m_shape=None, **kw):
    pad = lambda w: -(-w // 16) * 16
    # LoomFormer: m(i) [B, 2, K, V] holds the layer-input M of the token at position q in slot q % 2 (a decode
    # step reads its predecessor's slot while the read kernel writes its own)
    super().__init__(layers, batch, heads, max_len, pad(dqk), pad(dv), m_shape and (2, *m_shape), **kw)
    self.dqk, self.dv = dqk, dv

  def m_prev(self, i, pos):
    return self.m(i).index_select(1, ((pos + 1) % 2).view(1))[:, 0]

  def m_store(self, i, q, value):
    self.m(i).index_copy_(1, (q % 2).view(1), value[:, None])

  def k(self, i):
    return super().k(i)[..., :self.dqk]

  def v(self, i):
    return super().v(i)[..., :self.dv]


def _rmsw(x, w, eps):
  """Decoder RMSNorm (FP32 statistics, direct multiplier w), in x.dtype."""
  f = x.float()
  return (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps) * w).to(x.dtype)


def _rms(x, eps):
  f = x.float()
  return f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)


def _rope_tables(dim, c, device, dtype):
  frac = 2 * torch.arange(dim // 2, dtype=torch.float32) / dim
  timescale = c.rope_min_timescale * (c.rope_max_timescale / c.rope_min_timescale) ** frac
  ang = torch.arange(c.max_position_embeddings * 4, dtype=torch.float32)[:, None] / timescale
  return ang.cos().to(dtype).to(device), ang.sin().to(dtype).to(device)


def _rope_tail(x, start, cos, sin):
  """Half-split RoPE on x[..., start:], the leading coordinates unchanged."""
  h = (x.shape[-1] - start) // 2
  a, b = x[..., start:start + h], x[..., start + h:]
  return torch.cat((x[..., :start], a * cos - b * sin, b * cos + a * sin), -1)


def _strict_attention_cudnn(q, k, v):
  """Strictly causal attention (query i sees keys < i; the first query 0), q/k [B,T,N,Dq] (q pre-scaled),
  v [B,T,N,Dv], widths multiples of 8. cuDNN SDPA takes Dq != Dv: LoomFormer's 120 / 96 at near-FLOP-ratio
  speed (flash pads both to 128: ~1.4x slower)."""
  q, k, v = (z.transpose(1, 2) for z in (q, k, v))
  with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
    y = F.scaled_dot_product_attention(q[:, :, 1:], k[:, :, :-1], v[:, :, :-1], is_causal=True, scale=1.0)
  return F.pad(y, (0, 0, 1, 0)).transpose(1, 2)


def _swiglu(gu, inter):
  return F.silu(gu[..., :inter]) * gu[..., inter:2 * inter]


def _padded_ffn(gate, up, down, multiple=64):
  """SwiGLU weights with the hidden width zero-padded to a multiple (exact: silu(0) * 0 = 0); unaligned
  widths (e.g. 3317) make the down-projection rows unaligned and its GEMV/GEMM much slower."""
  pad = -gate.weight.shape[0] % multiple
  return (F.pad(gate.weight, (0, 0, 0, pad)), F.pad(up.weight, (0, 0, 0, pad)), F.pad(down.weight, (0, pad)),
          gate.weight.shape[0] + pad)


class _Base(nn.Module):

  def __init__(self, hf):
    super().__init__()
    c = hf.config
    self.config = c
    m = hf.model
    self.d, self.n, self.hd, self.eps = c.hidden_size, c.num_attention_heads, c.head_dim, c.rms_norm_eps
    self.embed = _p(m.embed_tokens.weight)
    self.lm_head = _p(hf.lm_head.weight)
    self.norm_w = _p(m.norm.weight.float())

  def _logits(self, x, b):
    x = _rmsw(x, self.norm_w, self.eps)
    if x.dtype == torch.float32:
      return F.linear(x, self.lm_head)
    return torch.mm(x.flatten(0, 1), self.lm_head.t(), out_dtype=torch.float32).view(b, x.shape[1], -1)

  def _mlp(self, p, h):
    return F.linear(_swiglu(F.linear(h, p.gu_w), p.inter), p.down_w)

  def prefill(self, ids, state=None, last_only=True):
    return self._forward(ids, state, decode=False, last_only=last_only)

  def decode(self, ids, state):
    return self._forward(ids, state, decode=True)


class FastMHA(_Base):
  """Llama2 Prop MHA control: RoPE on the trailing head_dim - mha_nope_dim coordinates."""

  def __init__(self, hf):
    super().__init__(hf)
    c = hf.config
    self.nope = c.mha_nope_dim
    cos, sin = _rope_tables(self.hd - self.nope, c, self.embed.device, self.embed.dtype)
    self.register_buffer('rope_cos', cos, persistent=False)
    self.register_buffer('rope_sin', sin, persistent=False)
    self.layers = nn.ModuleList()
    for layer in hf.model.layers:
      a, p = layer.self_attn, nn.Module()
      p.qkv_w = _p(torch.cat((a.query.weight, a.key.weight, a.value.weight), 0))
      p.out_w = _p(a.out.weight)
      g, u, dw, p.inter = _padded_ffn(layer.mlp.gate, layer.mlp.up, layer.mlp.down)
      p.gu_w, p.down_w = _p(torch.cat((g, u), 0)), _p(dw)
      p.ln1_w, p.ln2_w = _p(layer.input_layernorm.weight.float()), _p(layer.post_attention_layernorm.weight.float())
      self.layers.append(p)

  def new_state(self, batch, max_len, layout='rows'):
    return PaddedState(len(self.layers), batch, self.n, max_len, self.hd, self.hd, dtype=self.embed.dtype,
                 device=self.embed.device, layout=layout)

  def _forward(self, ids, state, decode, last_only=True):
    b, t = ids.shape
    n, hd = self.n, self.hd
    x = F.embedding(ids, self.embed)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    lengths = (pos + 1).to(torch.int32).expand(b).contiguous() if decode else None
    positions = pos + torch.arange(t, device=ids.device)
    cos, sin = self.rope_cos[positions][:, None], self.rope_sin[positions][:, None]
    scale = hd ** -0.5
    for i, p in enumerate(self.layers):
      qkv = F.linear(_rmsw(x, p.ln1_w, self.eps), p.qkv_w)
      q, k, v = qkv.view(b, t, 3, n, hd).unbind(2)
      q, k = _rope_tail(q, self.nope, cos, sin), _rope_tail(k, self.nope, cos, sin)
      if decode:
        y = attend_and_append((q * scale)[:, 0], state.k(i), state.v(i), k[:, 0], v[:, 0], pos, lengths, False,
                              state.layout)[:, None]
      else:
        if state is not None:
          state.store_prefill(i, k.transpose(1, 2), v.transpose(1, 2))
        y = _prefill_attention(q * scale, k, v, strict=False)
      x = x + F.linear(y.reshape(b, t, n * hd), p.out_w)
      x = x + self._mlp(p, _rmsw(x, p.ln2_w, self.eps))
    if state is not None:
      state.pos.add_(t)
    return self._logits(x[:, -1:] if last_only else x, b)


class FastBamV5(_Base):
  """LoomFormer V5 (Llama backbone): M-shift, whole-M read RMS, generalized reads, attention + MLP writes."""

  def __init__(self, hf):
    super().__init__(hf)
    c = hf.config
    m = hf.model
    self.kd, self.vd, self.s = c.bam_k, c.bam_v, c.standard_qk_dim
    self.key_eps, self.write_eps, self.key_scale = c.read_key_eps, c.write_eps, c.read_key_scale
    self.static_scale = 1.1 / c.static_read_gate_init
    ew = m.embedding_write
    self.e_w, self.e_b = _cat_linear([ew.data, ew.address_down, ew.gate])
    self.e_up_w, self.e_up_b = _p(ew.address_up.weight), _p(ew.address_up.bias)
    self.e_tau, self.e_bottleneck = ew.tau, ew.address_down.weight.shape[0]
    cos, sin = _rope_tables(self.s, c, self.embed.device, self.embed.dtype)
    self.register_buffer('rope_cos', cos, persistent=False)
    self.register_buffer('rope_sin', sin, persistent=False)
    self.layers = nn.ModuleList()
    for layer in m.layers:
      a, p = layer.self_attn, nn.Module()
      p.has_write, p.has_mlp_write = a.has_write, layer.has_mlp_write
      xs = [a.m_shift, a.query, a.key, a.read_down] + ([a.write_down] if a.has_write else [])
      xs += [getattr(a, f'{arm}_gate') for arm in 'qkvo'] + [getattr(a, f'{arm}_static_gate') for arm in 'qkvo']
      xs += [a.write_gate] if a.has_write else []
      p.x_w, p.x_b = _cat_linear(xs)
      p.split_x = [l.weight.shape[0] for l in xs]
      names = ['shift', 'std', 'std', 'read_down'] + (['write_down'] if a.has_write else []) + ['gates'] * 8
      names += ['write_gate'] if a.has_write else []
      p.off_x, o = {}, 0
      for name, w in zip(names, p.split_x):
        p.off_x[name] = (p.off_x.get(name, (o,))[0], o + w)
        o += w
      p.up_w, p.up_b = _cat_linear([getattr(a, f'{arm}_key_up') for arm in 'qkvo'])
      p.static_key = _p(torch.stack([getattr(a, f'{arm}_static_key').float().t() for arm in 'qkvo']))   # [4,N,V]
      p.static_key_rows = _p(p.static_key.flatten(0, 1).to(self.embed.dtype).contiguous())          # [4N,V]
      p.qk_scale, p.vo_scale = float(a.qk_scale.float()), float(a.vo_scale.float())
      p.out_w = _p(a.out.weight)
      if a.has_write:
        p.wup_w, p.wup_b = _p(a.write_up.weight), _p(a.write_up.bias)
      g, u, dw, p.inter = _padded_ffn(layer.mlp.gate, layer.mlp.up, layer.mlp.down)
      hs = [g, u]
      hb = [g.new_zeros(2 * p.inter)]
      if layer.has_mlp_write:
        mg, mu, mdw, p.mem_inter = _padded_ffn(layer.memory_mlp.gate, layer.memory_mlp.up, layer.memory_mlp.down)
        hs += [mg, mu, layer.mlp_write_gate.weight, layer.mlp_address_down.weight]
        hb += [g.new_zeros(2 * p.mem_inter), layer.mlp_write_gate.bias,
               g.new_zeros(layer.mlp_address_down.weight.shape[0])]
        p.mem_down_w = _p(mdw)
        p.addr_up_w, p.addr_up_b = _p(layer.mlp_address_up.weight), _p(layer.mlp_address_up.bias)
        p.tau = layer.tau
      p.split_h = [w.shape[0] for w in hs]
      p.h_w, p.h_b = _p(_pad_rows(torch.cat(hs, 0))), _p(_pad_rows(torch.cat(hb)))
      p.down_w = _p(dw)
      p.ln1_w, p.ln2_w = _p(layer.input_layernorm.weight.float()), _p(layer.post_attention_layernorm.weight.float())
      self.layers.append(p)

  def new_state(self, batch, max_len, layout='rows'):
    return PaddedState(len(self.layers), batch, self.n, max_len, self.kd + self.s, self.kd, (self.kd, self.vd),
                 dtype=self.embed.dtype, device=self.embed.device, layout=layout)

  def _heads(self, x):
    return x.unflatten(-1, (self.n, -1))

  def _embedding_write(self, e):
    n, kd = self.n, self.kd
    z = F.linear(e, self.e_w, self.e_b)
    data = (z[..., :n * kd].float() / self.e_tau).to(e.dtype)
    hid = F.gelu(z[..., n * kd:n * kd + self.e_bottleneck], approximate='tanh')
    gate = torch.sigmoid(z[..., n * kd + self.e_bottleneck:n * kd + self.e_bottleneck + n])
    address = _rms(self._heads(F.linear(hid, self.e_up_w, self.e_up_b)), self.write_eps).to(e.dtype)
    return torch.matmul((gate[..., None] * self._heads(data)).transpose(-1, -2), address)   # [B,T,K,V]

  def _layer(self, i, p, x, M, M_prev, valid, cos, sin, state, lengths, decode):
    b, t = x.shape[:2]
    n, kd, vd, dt = self.n, self.kd, self.vd, x.dtype
    z = F.linear(_rmsw(x, p.ln1_w, self.eps), p.x_w, p.x_b)
    parts = list(torch.split(z[..., :sum(p.split_x)], p.split_x, -1))
    shift, q_std, k_std, read_down = parts[:4]
    rest = parts[4:]
    write_down = rest.pop(0) if p.has_write else None
    gates, static_gates = torch.stack(rest[:4], -2), torch.stack(rest[4:8], -2)        # [B,T,4,N]
    write_gate = rest[8] if p.has_write else None
    up = F.linear(F.gelu(read_down, approximate='tanh'), p.up_w, p.up_b)[..., :4 * n * vd].unflatten(-1, (4, n, vd))
    keys = (self.key_scale * torch.sigmoid(gates.float()))[..., None] * _rms(up, self.key_eps)
    keys = keys + (self.static_scale * torch.sigmoid(static_gates.float()))[..., None] * p.static_key
    s = torch.sigmoid(shift.float()).to(M.dtype)[..., None, :]
    mixed = torch.where(valid[..., None, None], (1 - s) * M + s * M_prev, M)
    f = mixed.float()
    inv = torch.rsqrt(f.pow(2).mean((-2, -1)) + self.eps)                             # [B,T]
    reads = torch.matmul(mixed, keys.to(M.dtype).flatten(2, 3).transpose(-1, -2))     # [B,T,K,4N]
    reads = (reads.float() * inv[..., None, None]).transpose(-1, -2).unflatten(-2, (4, n))   # [B,T,4,N,K]
    q_loc, k_loc, value, local_o = reads.unbind(2)
    q = torch.cat(((q_loc * p.qk_scale).to(dt), self._heads(q_std)), -1)
    k = torch.cat(((k_loc * p.qk_scale).to(dt), self._heads(k_std)), -1)
    q = _rope_tail(q, kd, cos, sin) / torch.tensor(math.sqrt(self.hd), dtype=dt)
    k = _rope_tail(k, kd, cos, sin)
    value, local_o = (value * p.vo_scale).to(dt).contiguous(), (local_o * p.vo_scale).to(dt)
    if decode:
      y = attend_and_append(q[:, 0], state.k(i), state.v(i), k[:, 0], value[:, 0], state.pos, lengths, True,
                            state.layout)[:, None]
    else:
      if state is not None:
        state.store_prefill(i, k.transpose(1, 2), value.transpose(1, 2))
      y = _prefill_attention(q, k, value, strict=True)
    o_head = y + local_o
    x = x + F.linear(o_head.flatten(-2), p.out_w)
    hz = F.linear(_rmsw(x, p.ln2_w, self.eps), p.h_w, p.h_b)
    sh = p.split_h
    mlp = F.linear(_swiglu(hz, p.inter), p.down_w)
    if p.has_write:
      content = torch.sigmoid(write_gate)[..., None] * _rms(o_head, self.write_eps).to(dt)
      address = _rms(self._heads(F.linear(F.gelu(write_down, approximate='tanh'), p.wup_w, p.wup_b)),
                     self.write_eps).to(dt)
      if p.has_mlp_write:
        o = 2 * p.inter
        mem = F.linear(_swiglu(hz[..., o:o + 2 * p.mem_inter], p.mem_inter), p.mem_down_w)
        memory = self._heads(mem if p.tau == 1.0 else (mem.float() / p.tau).to(dt))
        o += 2 * p.mem_inter
        mgate, adown = hz[..., o:o + sh[4]], hz[..., o + sh[4]:o + sh[4] + sh[5]]
        mlp_content = torch.sigmoid(mgate)[..., None] * memory
        mlp_address = _rms(self._heads(F.linear(F.gelu(adown, approximate='tanh'), p.addr_up_w, p.addr_up_b)),
                           self.write_eps).to(dt)
        content, address = torch.cat((content, mlp_content), -2), torch.cat((address, mlp_address), -2)
      M = M + torch.matmul(content.transpose(-1, -2), address)
    return x + mlp, M

  def _forward(self, ids, state, decode, last_only=True):
    b, t = ids.shape
    x = F.embedding(ids, self.embed)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    lengths = pos.to(torch.int32).expand(b).contiguous() if decode else None     # strictly causal
    positions = pos + torch.arange(t, device=ids.device)
    cos, sin = self.rope_cos[positions][:, None], self.rope_sin[positions][:, None]
    if decode:
      valid = (pos > 0).expand(b, t)
    else:
      valid = torch.ones(b, t, dtype=torch.bool, device=ids.device)
      valid[:, 0] = False
    M = self._embedding_write(x)
    for i, p in enumerate(self.layers):
      M_in = M
      M_prev = state.m_prev(i, pos)[:, None] if decode else torch.cat((M[:, :1], M[:, :-1]), 1)
      x, M = self._layer(i, p, x, M, M_prev, valid, cos, sin, state, lengths, decode)
      if state is not None:
        state.m_store(i, pos + t - 1, M_in[:, -1])
    if state is not None:
      state.pos.add_(t)
    return self._logits(x[:, -1:] if last_only else x, b)

  def _forward_kernels(self, ids, state, decode, last_only=True):
    """Triton path (BF16): each layer's read fused with the previous layer's write (bam_kernels); BAM-only
    small linears grouped at decode. Shared components as in FastMHA."""
    b, t = ids.shape
    rows = b * t
    n, kd, vd, s, d = self.n, self.kd, self.vd, self.s, self.d
    x = F.embedding(ids, self.embed).view(rows, d)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    prev_valid = (pos > 0).to(torch.int32)
    lengths = pos.to(torch.int32).expand(b).contiguous() if decode else None     # strictly causal
    M = self._embedding_write(x).reshape(rows, kd * vd)
    q_mult = 1.0 / math.sqrt(self.hd)
    # read-kernel programs (row chunks, arm groups per program, warps): A800 sweep at Medium / Large / XL shapes
    if not decode:
      sub, groups, warps = 1, 2, 2
    elif rows == 1:
      sub, groups, warps = 4 if n > 16 else 2, 1, 4
    else:
      sub, groups, warps = (2, 1, 4) if rows <= 16 else (1, 2, 4)
    pending = None
    for i, p in enumerate(self.layers):
      z = F.linear(_rmsw(x, p.ln1_w, self.eps), p.x_w, p.x_b)
      o = p.off_x
      rd = z[:, o['read_down'][0]:o['read_down'][1]]
      wup = None
      if decode and rows == 1:
        if p.has_write:
          up, wup = gemv_cols([z, z], [o['read_down'][0], o['write_down'][0]], [p.up_w, p.wup_w], [p.up_b, p.wup_b],
                              [1, 1], 16, 256, 4)
        else:
          up, = gemv_cols([z], [o['read_down'][0]], [p.up_w], [p.up_b], [1], 16, 256, 4)
      elif decode and p.has_write:     # multi-token decode: one GELU pass, both GEMMs in one launch
        g = F.gelu(z[:, o['read_down'][0]:o['write_down'][1]], approximate='tanh')
        kr = p.up_w.shape[1]
        up, wup = pythia_kernels.grouped_gemm([g[:, :kr], g[:, kr:]], [p.up_w, p.wup_w], [p.up_b, p.wup_b], [0, 0],
                                              None, 128, 64, 4 if rows <= 16 else 8, 3)
      else:
        up = F.linear(F.gelu(rd, approximate='tanh'), p.up_w, p.up_b)
      write, (g1_off, g2_off) = (pending[:6], pending[6:]) if pending is not None else ((M,) * 6, (0, 0))
      m_new, qk, vo = bam_kernels.fused_read(
          M, *write, state.m(i) if decode else M, prev_valid, pos, z, z, up, z, p.static_key_rows, self.rope_cos,
          self.rope_sin, b, n, kd, vd, s, pending is not None, decode, self.write_eps, self.eps, self.key_eps,
          self.key_scale, self.static_scale, p.qk_scale, p.vo_scale, q_mult, sub, groups, warps,
          state.k(i) if decode else None, state.v(i) if decode else None, g1_off, g2_off, o['shift'][0],
          o['std'][0], o['gates'][0])
      if pending is not None:
        M = m_new
      if state is not None and not (decode and pending is not None):   # else the read kernel stored it
        state.m_store(i, pos + t - 1, M.view(b, t, kd, vd)[:, -1])
      qw, vw = qk.shape[-1], vo.shape[-1]
      q, k = (qk[:, j].view(b, t, n, qw) for j in (0, 1))       # zero-padded widths
      value, local_o = (vo[:, j].view(b, t, n, vw)[..., :kd] for j in (0, 1))
      if decode:     # the read kernel already wrote this token's key and value to the caches
        y = decode_attention(q[:, 0, :, None, :kd + s], state.k(i), state.v(i), lengths,
                             state.layout)[:, None, :, 0]
      else:
        if state is not None:
          state.store_prefill(i, k[..., :kd + s].transpose(1, 2), value.transpose(1, 2))
        y = _strict_attention_cudnn(q, k, vo[:, 0].view(b, t, n, vw))[..., :kd]
      o_head = (y + local_o).reshape(rows, n * kd)
      x = x + F.linear(o_head, p.out_w)
      hz = F.linear(_rmsw(x, p.ln2_w, self.eps), p.h_w, p.h_b)
      mlp = F.linear(_swiglu(hz, p.inter), p.down_w)
      if p.has_write:
        assert p.has_mlp_write and p.tau == 1.0
        sh, o2 = p.split_h, 2 * p.inter
        o3 = o2 + 2 * p.mem_inter
        mem_in, mgate, adown = _swiglu(hz[:, o2:o3], p.mem_inter), hz[:, o3:o3 + sh[4]], \
            hz[:, o3 + sh[4]:o3 + sh[4] + sh[5]]
        if decode and rows == 1:
          mem, addr = gemv_cols([mem_in, hz], [0, o3 + sh[4]], [p.mem_down_w, p.addr_up_w], [None, p.addr_up_b],
                                [0, 1], 4, 1024, 4)
        elif decode:
          mem, addr = pythia_kernels.grouped_gemm([mem_in, F.gelu(adown, approximate='tanh')],
                                                  [p.mem_down_w, p.addr_up_w], [None, p.addr_up_b], [0, 0], None,
                                                  32 if rows <= 16 else 64, 128, 4, 3)
        else:
          mem = F.linear(mem_in, p.mem_down_w)
          addr = F.linear(F.gelu(adown, approximate='tanh'), p.addr_up_w, p.addr_up_b)
        if wup is None:
          wd = z[:, o['write_down'][0]:o['write_down'][1]]
          wup = F.linear(F.gelu(wd, approximate='tanh'), p.wup_w, p.wup_b)
        pending = (o_head, z, wup, mem, hz, addr, o['write_gate'][0], o3)
      else:
        pending = None
      x = x + mlp
    if state is not None:
      state.pos.add_(t)
    x = x.view(b, t, d)
    return self._logits(x[:, -1:] if last_only else x, b)

  def prefill(self, ids, state=None, last_only=True):
    if KERNELS and self.embed.dtype == torch.bfloat16:
      return self._forward_kernels(ids, state, decode=False, last_only=last_only)
    return self._forward(ids, state, decode=False, last_only=last_only)

  def decode(self, ids, state):
    if KERNELS and self.embed.dtype == torch.bfloat16:
      return self._forward_kernels(ids, state, decode=True)
    return self._forward(ids, state, decode=True)


def build(hf):
  return (FastBamV5 if hf.config.bam_enabled else FastMHA)(hf).eval()
