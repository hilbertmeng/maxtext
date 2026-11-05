"""Inference fast path for the Pythia unit-H BAM port and the Pythia-2.8B (GPTNeoX) baseline.

`build(hf_model)` wraps a loaded HF model (BamPythiaForCausalLM or GPTNeoXForCausalLM, BF16, CUDA) in
an inference module with the same interface for both:

  state = model.new_state(batch, max_len)      # static K/V (+ BAM per-layer last-token M) caches
  logits = model.prefill(ids, state)           # [B, 1, vocab] logits of the last position
  logits = model.decode(ids[:, -1:], state)    # one token per row; state.pos advances in place

Shapes and buffer addresses stay fixed during decoding, so `decode` can be captured by CUDA graphs
(torch.compile mode='max-autotune' / 'reduce-overhead'). Batches are unpadded (all rows share a
position). Components both models have use the same implementation (inductor GEMMs, split-KV Triton
decode attention, RoPE tables); BAM-specific parts use the Triton kernels of bam_kernels.py and
prefill_attention.py (FastBamPythia.kernels; the pure-torch BAM path stays as a reference).

BAM algebra used here (exact up to rounding, checked by `check_fast.py` against the reference path):
- The whole-M RMS of the read view only scales the read outputs: read(Mh) = read(M~) / rms(M~).
- Every read is linear in the key: the Q/K pre-read gate, the V/O post-read gate and the static read
  combine into one key per arm and head, R = g rms(r) + g_s s, so all four arms are one per-token
  matmul M~ [K, V] @ R [V, 4N].
- The attention and MLP writes are one per-token matmul [K, 2N] @ [2N, V].
- Strict causality: queries 1..T-1 attend causally to keys 0..T-2 (the first query has no keys); a
  decode step attends to the cache before appending its own key.
"""
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

try:
  from . import bam_kernels
  from .decode_attention import attend_and_append, decode_attention, kt_cache_shape, kt_from_rows
  from .prefill_attention import strict_causal_attention
except ImportError:
  import bam_kernels
  from decode_attention import attend_and_append, decode_attention, kt_cache_shape, kt_from_rows
  from prefill_attention import strict_causal_attention


def _pad_rows(w, multiple=64):
  pad = -w.shape[0] % multiple
  return F.pad(w, (0, 0, 0, pad)) if pad and w.dim() == 2 else (F.pad(w, (0, pad)) if pad else w)


def _cat_linear(linears, multiple=64):
  """Concatenate Linear layers along the output axis (missing biases are zero); pad rows."""
  w = torch.cat([l.weight for l in linears], 0)
  b = torch.cat([l.bias if l.bias is not None else l.weight.new_zeros(l.weight.shape[0]) for l in linears])
  return nn.Parameter(_pad_rows(w, multiple), requires_grad=False), nn.Parameter(_pad_rows(b, multiple),
                                                                                  requires_grad=False)


def _p(t):
  return nn.Parameter(t.detach().clone(), requires_grad=False)


def _ln(x, w, b, eps=1e-5):
  return F.layer_norm(x.float(), (x.shape[-1],), w, b, eps).to(x.dtype)


def _rms(x, eps):
  f = x.float()
  return f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + eps)


class State(nn.Module):
  """Static caches, one buffer per layer (an in-place write into a slice of one stacked buffer is
  functionalized by inductor into a copy of the whole buffer). pos (device scalar) = tokens already
  cached = position of the next token."""

  def __init__(self, layers, batch, heads, max_len, dqk, dv, m_shape=None, dtype=torch.bfloat16, device='cuda',
               layout='rows'):
    super().__init__()
    self.max_len, self.layers, self.batch, self.layout = max_len, layers, batch, layout
    kshape = kt_cache_shape(batch, heads, max_len, dqk) if layout == 'kt' else (batch, heads, max_len, dqk)
    for i in range(layers):
      self.register_buffer(f'k{i}', torch.zeros(kshape, dtype=dtype, device=device))
      self.register_buffer(f'v{i}', torch.zeros(batch, heads, max_len, dv, dtype=dtype, device=device))
      if m_shape is not None:     # the last token's layer-input M (BAM M-shift)
        self.register_buffer(f'm{i}', torch.zeros(batch, *m_shape, dtype=dtype, device=device))
    self.register_buffer('pos', torch.zeros((), dtype=torch.long, device=device))

  def k(self, i):
    return getattr(self, f'k{i}')

  def v(self, i):
    return getattr(self, f'v{i}')

  def m(self, i):
    return getattr(self, f'm{i}')

  def reset(self, pos=0):
    self.pos.fill_(pos)

  def store_prefill(self, i, k, v):
    """Keys k [B,N,T,Dq] and values v [B,N,T,Dv] of positions 0..T-1 into layer i."""
    t = k.shape[2]
    if self.layout == 'kt':
      kb = kt_from_rows(k)
      self.k(i)[:, :, :kb.shape[2]] = kb
    else:
      self.k(i)[:, :, :t] = k
    self.v(i)[:, :, :t] = v

  def load_rows(self, src, row):
    """Copy a smaller-batch state (same position) into rows row.. (prefill large batches in pieces)."""
    for name, buf in self.named_buffers():
      if name != 'pos':
        buf[row:row + src.batch].copy_(getattr(src, name))
    self.pos.copy_(src.pos)


# Triton GEMVs for the linears both models share (qkv/x, out/dense, fc_in/h, fc_out) at one-token decode.
# Off: shared components use the same inductor kernels in both models. Measured on (A800, bs1): Pythia
# 5.24 -> 4.46 ms/token, BAM 5.88 -> 5.29 (BAM/Pythia 0.891 -> 0.843). BAM-only small linears are
# always grouped (bam_kernels.gemv).
SHARED_GEMV = os.environ.get('SHARED_GEMV', '0') == '1'
PAD80 = os.environ.get('PAD80', '0') == '1'     # Pythia 80-wide heads: padding to 96 is slower in-graph (A800)


def _prefill_attention(q, k, v, strict):
  """q/k [B,T,N,Dq] (q pre-scaled), v [B,T,N,Dv] -> [B,T,N,Dv] via SDPA flash (head dims padded)."""
  dv = v.shape[-1]
  width = max(q.shape[-1], dv)
  if width % 8 or (width == 80 and PAD80):
    width = 96 if width <= 96 else 128
  q, k, v = ((F.pad(z, (0, width - z.shape[-1])) if z.shape[-1] != width else z).transpose(1, 2) for z in (q, k, v))
  if not strict:
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)[..., :dv].transpose(1, 2)
  y = F.scaled_dot_product_attention(q[:, :, 1:], k[:, :, :-1], v[:, :, :-1], is_causal=True, scale=1.0)
  return F.pad(y[..., :dv], (0, 0, 1, 0)).transpose(1, 2)        # first query: no keys -> 0


class FastPythia(nn.Module):
  """GPTNeoX (parallel residual, partial NeoX RoPE)."""

  def __init__(self, hf):
    super().__init__()
    c = hf.config
    self.config = c
    m = hf.gpt_neox
    self.d, self.n, self.hd = c.hidden_size, c.num_attention_heads, c.hidden_size // c.num_attention_heads
    self.rot = int(self.hd * c.rope_parameters.get('partial_rotary_factor', 1.0))
    self.eps = c.layer_norm_eps
    self.embed = _p(m.embed_in.weight)
    self.lm_head = _p(hf.get_output_embeddings().weight)
    self.norm_w, self.norm_b = _p(m.final_layer_norm.weight.float()), _p(m.final_layer_norm.bias.float())
    inv = 1.0 / (c.rope_parameters['rope_theta'] ** (torch.arange(0, self.rot, 2, dtype=torch.float) / self.rot))
    self.register_buffer('inv_freq', inv.to(self.embed.device), persistent=False)
    ang = torch.arange(c.max_position_embeddings * 4, dtype=torch.float32)[:, None] * inv
    # RoPE tables (indexed, not recomputed: inductor otherwise fuses cos/sin into every layer's kernel)
    self.register_buffer('rope_cos', ang.cos().to(self.embed.dtype).to(self.embed.device), persistent=False)
    self.register_buffer('rope_sin', ang.sin().to(self.embed.dtype).to(self.embed.device), persistent=False)
    self.layers = nn.ModuleList()
    for layer in m.layers:
      p = nn.Module()
      a = layer.attention
      w = a.query_key_value.weight.view(self.n, 3, self.hd, self.d).transpose(0, 1).reshape(3 * self.d, self.d)
      b = a.query_key_value.bias.view(self.n, 3, self.hd).transpose(0, 1).reshape(-1)
      p.qkv_w, p.qkv_b = _p(w), _p(b)
      p.dense_w, p.dense_b = _p(a.dense.weight), _p(a.dense.bias)
      p.fc_in_w, p.fc_in_b = _p(layer.mlp.dense_h_to_4h.weight), _p(layer.mlp.dense_h_to_4h.bias)
      p.fc_out_w, p.fc_out_b = _p(layer.mlp.dense_4h_to_h.weight), _p(layer.mlp.dense_4h_to_h.bias)
      # one-token decode: dense and fc_out as one GEMV over [attention | gelu(h)] (parallel residual)
      p.out_cat_w = _p(torch.cat((p.dense_w, p.fc_out_w), 1))
      p.out_cat_b = _p(p.dense_b + p.fc_out_b)
      for name, ln in (('ln1', layer.input_layernorm), ('ln2', layer.post_attention_layernorm)):
        setattr(p, name + '_w', _p(ln.weight.float()))
        setattr(p, name + '_b', _p(ln.bias.float()))
      self.layers.append(p)

  def new_state(self, batch, max_len, layout='rows'):
    # 'rows' is the fastest Pythia key layout in the compiled decode step at B = 1, 16, 64 (in isolation
    # 'chunked' (B = 1) and 'kt' (B = 64) were a few percent faster, not in the full step)
    return State(len(self.layers), batch, self.n, max_len, self.hd, self.hd, device=self.embed.device, layout=layout)

  def _rope(self, x, cos, sin):
    r, h = self.rot, self.rot // 2
    a, b = x[..., :h], x[..., h:r]
    return torch.cat((a * cos - b * sin, b * cos + a * sin, x[..., r:]), -1)

  def _forward(self, ids, state, decode, last_only=True):
    b, t = ids.shape
    x = F.embedding(ids, self.embed)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    lengths = (pos + 1).to(torch.int32).expand(b).contiguous() if decode else None
    positions = pos + torch.arange(t, device=ids.device)
    cos, sin = self.rope_cos[positions][:, None], self.rope_sin[positions][:, None]
    scale = self.hd ** -0.5
    gemv = decode and b == 1 and SHARED_GEMV      # one-token decode: Triton GEMVs (LayerNorms, residual fused)
    for i, p in enumerate(self.layers):
      if gemv:
        qkv, hpre = bam_kernels.gemv([x[:, 0], x[:, 0]], [p.qkv_w, p.fc_in_w], [p.qkv_b, p.fc_in_b], [0, 0], 8, 512,
                                     4, ln_ws=[p.ln1_w, p.ln2_w], ln_bs=[p.ln1_b, p.ln2_b], ln_eps=self.eps)
      else:
        qkv = F.linear(_ln(x, p.ln1_w, p.ln1_b, self.eps), p.qkv_w, p.qkv_b)
      q, k, v = qkv.view(b, t, 3, self.n, self.hd).unbind(2)
      q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
      if decode:
        y = attend_and_append((q * scale)[:, 0], state.k(i), state.v(i), k[:, 0], v[:, 0], pos, lengths, False, state.layout)[:, None]
      else:
        if state is not None:
          state.store_prefill(i, k.transpose(1, 2), v.transpose(1, 2))
        y = _prefill_attention(q * scale, k, v, strict=False)
      if gemv:
        x, = bam_kernels.gemv([y.reshape(1, self.d)], [p.out_cat_w], [p.out_cat_b], [2], 8, 512, 8,
                              x2s=[hpre], ress=[x[:, 0]], act_froms=[self.d])
        x = x[:, None]
        continue
      attn = F.linear(y.reshape(b, t, self.d), p.dense_w, p.dense_b)
      h = _ln(x, p.ln2_w, p.ln2_b, self.eps)
      mlp = F.linear(F.gelu(F.linear(h, p.fc_in_w, p.fc_in_b)), p.fc_out_w, p.fc_out_b)
      x = x + attn + mlp
    if state is not None:
      state.pos.add_(t)
    x = _ln(x[:, -1:] if last_only else x, self.norm_w, self.norm_b, self.eps)
    return F.linear(x, self.lm_head)

  def prefill(self, ids, state=None, last_only=True):
    return self._forward(ids, state, decode=False, last_only=last_only)

  def decode(self, ids, state):
    return self._forward(ids, state, decode=True)


class FastBamPythia(nn.Module):
  """Unit-H BAM with M-shift and whole-M read RMS (see the module docstring for the algebra)."""

  def __init__(self, hf):
    super().__init__()
    c = hf.config
    self.config = c
    m = hf.model
    self.d, self.n, self.kd, self.vd, self.s = c.hidden_size, c.num_attention_heads, c.bam_k, c.bam_v, \
        c.standard_qk_dim
    self.eps, self.key_eps, self.write_eps, self.key_scale = c.layer_norm_eps, c.read_key_eps, c.write_eps, \
        c.read_key_scale
    self.static_scale = 1.1 / c.static_read_gate_init
    self.embed = _p(m.embed_tokens.weight)
    self.lm_head = _p(hf.lm_head.weight)
    self.norm_w, self.norm_b = _p(m.norm.weight.float()), _p(m.norm.bias.float())
    ew = m.embedding_write
    self.e_w, self.e_b = _cat_linear([ew.data, ew.address_down, ew.gate])
    self.e_up_w, self.e_up_b = _p(ew.address_up.weight), _p(ew.address_up.bias)
    self.e_tau, self.e_bottleneck = ew.tau, ew.address_down.weight.shape[0]
    frac = 2 * torch.arange(self.s // 2, dtype=torch.float32) / self.s
    timescale = c.rope_min_timescale * (c.rope_max_timescale / c.rope_min_timescale) ** frac
    self.register_buffer('inv_timescale', (1.0 / timescale).to(self.embed.device), persistent=False)
    ang = torch.arange(c.max_position_embeddings * 4, dtype=torch.float32)[:, None] / timescale
    self.register_buffer('rope_cos', ang.cos().to(self.embed.dtype).to(self.embed.device), persistent=False)
    self.register_buffer('rope_sin', ang.sin().to(self.embed.dtype).to(self.embed.device), persistent=False)
    self.kernels = True
    self.layers = nn.ModuleList()
    for layer in m.layers:
      a = layer.self_attn
      p = nn.Module()
      p.has_write = a.has_write
      p.has_mlp_write = layer.has_mlp_write
      xs = [a.m_shift, a.query, a.key, a.read_down] + ([a.write_down] if a.has_write else [])
      xs += [getattr(a, f'{arm}_gate') for arm in 'qkvo'] + [getattr(a, f'{arm}_static_gate') for arm in 'qkvo']
      xs += [a.write_gate] if a.has_write else []
      p.x_w, p.x_b = _cat_linear(xs)
      p.split_x = [l.weight.shape[0] for l in xs]
      ups = [getattr(a, f'{arm}_key_up') for arm in 'qkvo']
      p.up_w, p.up_b = _cat_linear(ups)
      # static keys [V, N] per arm -> [4, N, V]
      p.static_key = _p(torch.stack([getattr(a, f'{arm}_static_key').float().t() for arm in 'qkvo']))
      p.qk_scale, p.vo_scale = float(a.qk_scale.float()), float(a.vo_scale.float())
      p.q_norm, p.k_norm = _p(a.q_norm.weight), _p(a.k_norm.weight)
      p.norm_w = _p(torch.stack((a.q_norm.weight, a.k_norm.weight)).float())        # kernels: [2, 100]
      p.static_key_rows = _p(p.static_key.flatten(0, 1).to(self.embed.dtype).contiguous())  # kernels: [4N, V]
      offsets = [0]
      for w in p.split_x:
        offsets.append(offsets[-1] + w)
      names = ['shift', 'q_std', 'k_std', 'read_down'] + (['write_down'] if a.has_write else []) + \
          ['gates'] * 8 + (['write_gate'] if a.has_write else [])
      p.off_x = {}
      for name, o, w in zip(names, offsets, p.split_x):
        if name not in p.off_x:
          p.off_x[name] = (o, o + w)
        else:
          p.off_x[name] = (p.off_x[name][0], o + w)
      p.out_w, p.out_b = _p(a.out.weight), _p(a.out.bias)
      if a.has_write:
        p.wup_w, p.wup_b = _p(a.write_up.weight), _p(a.write_up.bias)
      hs = [layer.mlp.fc_in]
      if layer.has_mlp_write:
        hs += [layer.memory_mlp.fc_in, layer.mlp_write_gate, layer.mlp_address_down]
        p.mem_out_w = _p(layer.memory_mlp.fc_out.weight)
        p.addr_up_w, p.addr_up_b = _p(layer.mlp_address_up.weight), _p(layer.mlp_address_up.bias)
        p.tau = layer.tau
      p.h_w, p.h_b = _cat_linear(hs)
      p.split_h = [l.weight.shape[0] for l in hs]
      p.fc_out_w, p.fc_out_b = _p(layer.mlp.fc_out.weight), _p(layer.mlp.fc_out.bias)
      for name, ln in (('ln1', layer.input_layernorm), ('ln2', layer.post_attention_layernorm)):
        setattr(p, name + '_w', _p(ln.weight.float()))
        setattr(p, name + '_b', _p(ln.bias.float()))
      self.layers.append(p)

  def new_state(self, batch, max_len, layout='kt'):
    return State(len(self.layers), batch, self.n, max_len, self.kd + self.s, self.kd, (self.kd, self.vd),
                 device=self.embed.device, layout=layout)

  def _heads(self, x):
    return x.unflatten(-1, (self.n, -1))

  def _embedding_write(self, e):
    n, kd = self.n, self.kd
    z = F.linear(e, self.e_w, self.e_b)
    data = (z[..., :n * kd].float() / self.e_tau).to(e.dtype)
    hid = F.gelu(z[..., n * kd:n * kd + self.e_bottleneck], approximate='tanh')
    gate = torch.sigmoid(z[..., n * kd + self.e_bottleneck:n * kd + self.e_bottleneck + n])
    address = _rms(self._heads(F.linear(hid, self.e_up_w, self.e_up_b)), self.write_eps).to(e.dtype)
    content = gate[..., None] * self._heads(data)                     # [B,T,N,K]
    return torch.matmul(content.transpose(-1, -2), address)           # [B,T,K,V]

  def _rope(self, x, cos, sin):
    kd, h = self.kd, self.s // 2
    a, b = x[..., kd:kd + h], x[..., kd + h:]
    return torch.cat((x[..., :kd], a * cos - b * sin, b * cos + a * sin), -1)

  def _layer(self, i, p, x, M, M_prev, valid, cos, sin, state, positions, decode):
    b, t = x.shape[:2]
    n, kd, vd = self.n, self.kd, self.vd
    xn = _ln(x, p.ln1_w, p.ln1_b, self.eps)
    z = F.linear(xn, p.x_w, p.x_b)
    parts = list(torch.split(z[..., :sum(p.split_x)], p.split_x, -1))
    shift, q_std, k_std, read_down = parts[:4]
    rest = parts[4:]
    write_down = rest.pop(0) if p.has_write else None
    gates = torch.stack(rest[:4], -2)                                   # [B,T,4,N]
    static_gates = torch.stack(rest[4:8], -2)
    write_gate = rest[8] if p.has_write else None
    # read keys of the four arms, the static reads folded in
    up = F.linear(F.gelu(read_down, approximate='tanh'), p.up_w, p.up_b).unflatten(-1, (4, n, vd))
    keys = (self.key_scale * torch.sigmoid(gates.float()))[..., None] * _rms(up, self.key_eps)
    keys = keys + (self.static_scale * torch.sigmoid(static_gates.float()))[..., None] * p.static_key
    # read view: M-shift blend, whole-M RMS applied to the outputs
    s = torch.sigmoid(shift.float()).to(M.dtype)[..., None, :]
    mixed = torch.where(valid[..., None, None], (1 - s) * M + s * M_prev, M)
    f = mixed.float()
    inv = torch.rsqrt(f.pow(2).mean((-2, -1)) + self.eps)              # [B,T]
    reads = torch.matmul(mixed, keys.to(M.dtype).flatten(2, 3).transpose(-1, -2))   # [B,T,K,4N]
    reads = (reads.float() * inv[..., None, None]).transpose(-1, -2).unflatten(-2, (4, n))   # [B,T,4,N,K]
    q_loc, k_loc, value, local_o = reads.unbind(2)
    dt = x.dtype
    q = torch.cat(((q_loc * p.qk_scale).to(dt), self._heads(q_std)), -1)
    k = torch.cat(((k_loc * p.qk_scale).to(dt), self._heads(k_std)), -1)
    q = (_rms(q, self.eps).to(dt) * p.q_norm)
    k = (_rms(k, self.eps).to(dt) * p.k_norm)
    q = self._rope(q, cos, sin) / torch.tensor(math.sqrt(kd + self.s), dtype=dt)
    k = self._rope(k, cos, sin)
    value = (value * p.vo_scale).to(dt)
    local_o = (local_o * p.vo_scale).to(dt)
    if decode:
      y = attend_and_append(q[:, 0], state.k(i), state.v(i), k[:, 0], value[:, 0], state.pos, lengths, True, state.layout)[:, None]
    else:
      if state is not None:
        state.store_prefill(i, k.transpose(1, 2), value.transpose(1, 2))
      y = _prefill_attention(q, k, value, strict=True)
    o_head = y + local_o
    x = x + F.linear(o_head.flatten(-2), p.out_w, p.out_b)
    h = _ln(x, p.ln2_w, p.ln2_b, self.eps)
    hz = F.linear(h, p.h_w, p.h_b)
    hparts = torch.split(hz[..., :sum(p.split_h)], p.split_h, -1)
    mlp = F.linear(F.gelu(hparts[0]), p.fc_out_w, p.fc_out_b)
    if p.has_write:
      content = torch.sigmoid(write_gate)[..., None] * _rms(o_head, self.write_eps).to(dt)
      address = _rms(self._heads(F.linear(F.gelu(write_down, approximate='tanh'), p.wup_w, p.wup_b)),
                     self.write_eps).to(dt)
      if p.has_mlp_write:
        mem_in, mgate, adown = hparts[1:]
        memory = self._heads(F.linear(F.gelu(mem_in), p.mem_out_w))
        if p.tau != 1.0:
          memory = (memory.float() / p.tau).to(dt)
        mlp_content = torch.sigmoid(mgate)[..., None] * memory
        mlp_address = _rms(self._heads(F.linear(F.gelu(adown, approximate='tanh'), p.addr_up_w, p.addr_up_b)),
                           self.write_eps).to(dt)
        content = torch.cat((content, mlp_content), -2)
        address = torch.cat((address, mlp_address), -2)
      M = M + torch.matmul(content.transpose(-1, -2), address)
    return x + mlp, M

  def _forward(self, ids, state, decode, last_only=True):
    b, t = ids.shape
    x = F.embedding(ids, self.embed)
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    lengths = pos.to(torch.int32).expand(b).contiguous() if decode else None     # strictly causal
    positions = pos + torch.arange(t, device=ids.device)
    ang = positions[:, None].float() * self.inv_timescale
    cos, sin = ang.cos().to(x.dtype)[:, None], ang.sin().to(x.dtype)[:, None]
    if decode:
      valid = (pos > 0).expand(b, t)
    else:
      valid = torch.ones(b, t, dtype=torch.bool, device=ids.device)
      valid[:, 0] = False
    M = self._embedding_write(x)
    for i, p in enumerate(self.layers):
      M_in = M
      M_prev = state.m(i)[:, None] if decode else torch.cat((M[:, :1], M[:, :-1]), 1)
      x, M = self._layer(i, p, x, M, M_prev, valid, cos, sin, state, positions, decode)
      if state is not None:
        state.m(i).copy_(M_in[:, -1])         # after the layer: M_prev may alias state.m(i)
    if state is not None:
      state.pos.add_(t)
    x = _ln(x[:, -1:] if last_only else x, self.norm_w, self.norm_b, self.eps)
    return torch.mm(x.flatten(0, 1), self.lm_head.t(), out_dtype=torch.float32).view(b, x.shape[1], -1)

  def _forward_kernels(self, ids, state, decode, last_only=True):
    """Triton path: each layer's read is fused with the previous layer's write (bam_kernels)."""
    b, t = ids.shape
    rows = b * t
    n, kd, vd, d = self.n, self.kd, self.vd, self.d
    x = F.embedding(ids, self.embed).view(rows, d)
    dt = x.dtype
    pos = state.pos if state is not None else torch.zeros((), dtype=torch.long, device=ids.device)
    prev_valid = (pos > 0).to(torch.int32)
    lengths = pos.to(torch.int32).expand(b).contiguous() if decode else None     # strictly causal
    M = self._embedding_write(x).reshape(rows, kd * vd)
    pending = None
    for i, p in enumerate(self.layers):
      # one-token decode: Triton GEMVs; BAM-only small linears grouped, shared ones only with SHARED_GEMV
      small = decode and b == 1
      gemv = small and SHARED_GEMV
      if gemv:
        z, = bam_kernels.gemv([x], [p.x_w], [p.x_b], [0], 8, 512, 4, ln_ws=[p.ln1_w], ln_bs=[p.ln1_b],
                              ln_eps=self.eps)
      else:
        z = F.linear(_ln(x, p.ln1_w, p.ln1_b, self.eps), p.x_w, p.x_b)
      o = p.off_x
      rd = z[:, o['read_down'][0]:o['read_down'][1]]
      if small and p.has_write:
        up, wup = bam_kernels.gemv([rd, z[:, o['write_down'][0]:o['write_down'][1]]], [p.up_w, p.wup_w],
                                   [p.up_b, p.wup_b], [1, 1], 32, 128, 4)
      elif small:
        up, = bam_kernels.gemv([rd], [p.up_w], [p.up_b], [1], 32, 128, 4)
      elif decode and p.has_write:     # multi-token decode: one GELU pass, both GEMMs in one launch
        assert o['write_down'][0] == o['read_down'][1]
        g = F.gelu(z[:, o['read_down'][0]:o['write_down'][1]], approximate='tanh')
        kr = p.up_w.shape[1]
        up, wup = bam_kernels.grouped_gemm([g[:, :kr], g[:, kr:]], [p.up_w, p.wup_w], [p.up_b, p.wup_b], [0, 0],
                                           None, 128, 64, 4 if rows <= 16 else 8, 3)
      else:
        up = F.linear(F.gelu(rd, approximate='tanh'), p.up_w, p.up_b)
        wup = None
      write = pending if pending is not None else (M,) * 6
      m_prev = state.m(i) if decode else M
      m_new, qk, vo = bam_kernels.fused_read(
          M, *write, m_prev, prev_valid, pos, z[:, :o['shift'][1]], z[:, o['q_std'][0]:o['k_std'][1]], up,
          z[:, o['gates'][0]:o['gates'][1]], p.static_key_rows, p.norm_w, self.rope_cos, self.rope_sin,
          b, pending is not None, decode, self.write_eps, self.eps, self.key_eps, self.key_scale,
          self.static_scale, p.qk_scale, p.vo_scale, 1.0 / math.sqrt(kd + self.s), 1 if decode else 8,
          state.k(i) if decode else None, state.v(i) if decode else None)
      if pending is not None:
        M = m_new
      if state is not None:
        state.m(i).copy_(M.view(b, t, kd * vd)[:, -1].view(b, kd, vd))
      q, k = (qk[:, j, :, :kd + self.s].view(b, t, n, kd + self.s) for j in (0, 1))
      value, local_o = vo[:, 0].view(b, t, n, kd), vo[:, 1].view(b, t, n, kd)
      if decode:
        # the read kernel already wrote this token's key and value to the caches
        y = decode_attention(q[:, 0, :, None], state.k(i), state.v(i), lengths, state.layout)[:, None, :, 0]
      else:
        if state is not None:
          state.store_prefill(i, k.transpose(1, 2), value.transpose(1, 2))
        y = None
        o_head = strict_causal_attention(q, k, value, local_o).view(rows, n * kd)
      if y is not None:
        o_head = (y + local_o).view(rows, n * kd)
      sh = p.split_h
      if gemv:       # one token: raw projections, activations in the GEMV prologues, residuals in epilogues
        x, = bam_kernels.gemv([o_head], [p.out_w], [p.out_b], [0], 4, 1024, 2, ress=[x])
        hz, = bam_kernels.gemv([x], [p.h_w], [p.h_b], [0], 16, 256, 4, ln_ws=[p.ln2_w], ln_bs=[p.ln2_b],
                               ln_eps=self.eps)
        if p.has_write:
          assert p.has_mlp_write and p.tau == 1.0
          o2, o3 = sh[0] + sh[1], sh[0] + sh[1] + sh[2]
          x, mem, addr = bam_kernels.gemv(
              [hz[:, :sh[0]], hz[:, sh[0]:o2], hz[:, o3:o3 + sh[3]]], [p.fc_out_w, p.mem_out_w, p.addr_up_w],
              [p.fc_out_b, None, p.addr_up_b], [2, 2, 1], 8, 1024, 8, ress=[x, None, None])
          pending = (o_head, z[:, o['write_gate'][0]:o['write_gate'][1]], wup, mem, hz[:, o2:o3], addr)
        else:
          x, = bam_kernels.gemv([hz[:, :sh[0]]], [p.fc_out_w], [p.fc_out_b], [2], 8, 1024, 8, ress=[x])
          pending = None
        continue
      x = x + F.linear(o_head, p.out_w, p.out_b)
      h = _ln(x, p.ln2_w, p.ln2_b, self.eps)
      if small:      # shared linears by inductor, the BAM-only memory-MLP output and write address grouped
        hz = F.linear(h, p.h_w, p.h_b)
        mlp = F.linear(F.gelu(hz[:, :sh[0]]), p.fc_out_w, p.fc_out_b)
        if p.has_write:
          assert p.has_mlp_write and p.tau == 1.0
          o2, o3 = sh[0] + sh[1], sh[0] + sh[1] + sh[2]
          mem, addr = bam_kernels.gemv([hz[:, sh[0]:o2], hz[:, o3:o3 + sh[3]]], [p.mem_out_w, p.addr_up_w],
                                       [None, p.addr_up_b], [2, 1], 16, 1024, 8)
          pending = (o_head, z[:, o['write_gate'][0]:o['write_gate'][1]], wup, mem, hz[:, o2:o3], addr)
        else:
          pending = None
        x = x + mlp
        continue
      # one GEMM for every consumer of h (main MLP, memory MLP, write gate, write address): LayerNorm fused
      # as its prologue. (Splitting off the exact-GELU columns to get a fused GELU epilogue did not: inductor
      # still runs the GELU as its own kernel, and h then had to be materialized for the second GEMM.)
      hz = F.linear(h, p.h_w, p.h_b)
      mlp = F.linear(F.gelu(hz[:, :sh[0]]), p.fc_out_w, p.fc_out_b)
      if p.has_write:
        assert p.has_mlp_write and p.tau == 1.0
        o2, o3 = sh[0] + sh[1], sh[0] + sh[1] + sh[2]
        mem_in, adown = F.gelu(hz[:, sh[0]:o2]), F.gelu(hz[:, o3:o3 + sh[3]], approximate='tanh')
        if decode:   # multi-token decode: the memory-MLP output and write address in one launch
          mem, addr = bam_kernels.grouped_gemm([mem_in, adown], [p.mem_out_w, p.addr_up_w], [None, p.addr_up_b],
                                               [0, 0], None, 32 if rows <= 16 else 64, 128, 4, 3)
        else:
          mem, addr = F.linear(mem_in, p.mem_out_w), F.linear(adown, p.addr_up_w, p.addr_up_b)
        if wup is None:      # prefill (multi-token decode computed it with the read keys)
          wd = z[:, o['write_down'][0]:o['write_down'][1]]
          wup = F.linear(F.gelu(wd, approximate='tanh'), p.wup_w, p.wup_b)
        pending = (o_head, z[:, o['write_gate'][0]:o['write_gate'][1]], wup, mem, hz[:, o2:o3], addr)
      else:
        pending = None
      x = x + mlp
    if state is not None:
      state.pos.add_(t)
    x = x.view(b, t, d)
    x = _ln(x[:, -1:] if last_only else x, self.norm_w, self.norm_b, self.eps)
    return torch.mm(x.flatten(0, 1), self.lm_head.t(), out_dtype=torch.float32).view(b, x.shape[1], -1)

  def prefill(self, ids, state=None, last_only=True):
    if self.kernels:
      return self._forward_kernels(ids, state, decode=False, last_only=last_only)
    return self._forward(ids, state, decode=False, last_only=last_only)

  def decode(self, ids, state):
    if self.kernels:
      return self._forward_kernels(ids, state, decode=True)
    return self._forward(ids, state, decode=True)


def build(hf):
  kind = hf.config.model_type
  model = {'bam_pythia': FastBamPythia, 'gpt_neox': FastPythia}[kind](hf)
  return model.eval()
