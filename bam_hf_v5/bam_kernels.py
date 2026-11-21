"""Triton kernels for the per-token M work of the LoomFormer V5 fast path (shapes generic: K > 64, any V, S, N).

`fused_read`: the previous layer's write and this layer's read in one pass over M.

  M_new = M_old + sum_n sigmoid(g1_n) rms(c1_n) (x) rms(a1_n) + sum_n sigmoid(g2_n) c2_n (x) rms(a2_n)
  M~    = M_new + s (M_prev_new - M_new)           (per V column, s = sigmoid(shift); t > 0 only)
  R     = c_key sigmoid(g_a) rms(up_a) + c_static sigmoid(h_a) S_a      (arms q, k, v, o; [4N, V])
  read  = R @ M~^T / rms(M~)                       ([4N, K])
  q, k  = RoPE_tail([c_qk read_{q,k} | std_{q,k}]), q * q_mult;  v, o_loc = c_vo read_{v,o}

Programs: axis 0 = (arm group (q, k) or (v, o), row chunk), axis 1 = token. M tiles are [64, VP] and
[K1P, VP] (K = 64 + tail, V padded to a power of two, masked). One-token decode fuses the previous
layer's write (program 0 stores M_new) and writes the token's key and value into the rows-layout caches.

Layouts (row = token, unit inner stride): M [K*V] (k major); content [N, K]; address [N, V]; up [4N, V];
gates [4N] dynamic then [4N] static (pre-activation); std [2N, S] (q then k); qk out [2, N, QW]
(K + S used); vo out [2, N, K].
"""
from typing import Optional

import torch
import triton
import triton.language as tl


def _pow2(x):
  return max(16, triton.next_power_of_2(x))


@triton.jit
def _load_m(P, K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, VP: tl.constexpr):
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, K1P)
  v = tl.arange(0, VP)
  vm = v < V
  m0 = tl.load(P + k0[:, None] * V + v[None, :], mask=vm[None, :], other=0.)
  m1 = tl.load(P + k1[:, None] * V + v[None, :], mask=(k1 < K)[:, None] & vm[None, :], other=0.)
  return m0, m1


@triton.jit
def _store_m(P, m0, m1, K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, VP: tl.constexpr):
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, K1P)
  v = tl.arange(0, VP)
  vm = v < V
  tl.store(P + k0[:, None] * V + v[None, :], m0, mask=vm[None, :])
  tl.store(P + k1[:, None] * V + v[None, :], m1, mask=(k1 < K)[:, None] & vm[None, :])


@triton.jit
def _write_pair(row, C, s_c, G, s_g, A, s_a, write_eps, RMS_C: tl.constexpr, N: tl.constexpr, NP: tl.constexpr,
                K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, VP: tl.constexpr):
  """sum_n sigmoid(g_n) c_n (x) rms(a_n) of one token as fp32 tiles [64, VP], [K1P, VP]; c_n RMS-normalized
  over K when RMS_C."""
  n = tl.arange(0, NP)
  nm = n < N
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, K1P)
  v = tl.arange(0, VP)
  c0 = tl.load(C + row * s_c + n[:, None] * K + k0[None, :], mask=nm[:, None], other=0.).to(tl.float32)
  c1 = tl.load(C + row * s_c + n[:, None] * K + k1[None, :], mask=nm[:, None] & (k1 < K)[None, :],
               other=0.).to(tl.float32)
  sc = tl.sigmoid(tl.load(G + row * s_g + n, mask=nm, other=0.).to(tl.float32))
  if RMS_C:
    sc = sc * tl.rsqrt((tl.sum(c0 * c0, 1) + tl.sum(c1 * c1, 1)) / K + write_eps)
  a = tl.load(A + row * s_a + n[:, None] * V + v[None, :], mask=nm[:, None] & (v < V)[None, :],
              other=0.).to(tl.float32)
  a = (a * tl.rsqrt(tl.sum(a * a, 1) / V + write_eps)[:, None]).to(tl.bfloat16)
  c0 = (c0 * sc[:, None]).to(tl.bfloat16)
  c1 = (c1 * sc[:, None]).to(tl.bfloat16)
  return tl.dot(tl.trans(c0), a), tl.dot(tl.trans(c1), a)


@triton.jit
def _new_m(row, MOLD, s_mo, C1, s_c1, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, write_eps,
           N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr,
           VP: tl.constexpr):
  m0, m1 = _load_m(MOLD + row * s_mo, K, K1P, V, VP)
  d0, d1 = _write_pair(row, C1, s_c1, G1, s_g1, A1, s_a1, write_eps, True, N, NP, K, K1P, V, VP)     # attention
  e0, e1 = _write_pair(row, C2, s_c2, G2, s_g2, A2, s_a2, write_eps, False, N, NP, K, K1P, V, VP)    # MLP
  m0 = (m0.to(tl.float32) + (d0 + e0)).to(tl.bfloat16)
  m1 = (m1.to(tl.float32) + (d1 + e1)).to(tl.bfloat16)
  return m0, m1


@triton.jit
def _read_kernel(
    M, s_m, MW, C1, s_c1, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, write_eps, MPREV, s_mp, PREV_VALID, POS,
    SHIFT, s_sh, STD, s_std, UP, s_up, GATES, s_gt, SKEY, COS, SIN,
    QK, s_qk, VO, s_vo, KC, skb, skh, sks, VC, svb, svh, svs,
    T, eps, key_eps, key_scale, static_scale, qk_scale, vo_scale, q_mult,
    HAS_PREV: tl.constexpr, FUSE_WRITE: tl.constexpr, CACHE: tl.constexpr, SUB: tl.constexpr,
    N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
    H: tl.constexpr, HP: tl.constexpr, QW: tl.constexpr, CR: tl.constexpr, CRP: tl.constexpr):
  """Axis 0 = group * SUB + row chunk (CR of the group's 2N (arm, head) rows), axis 1 = token row."""
  group = tl.program_id(0) // SUB
  ci = tl.arange(0, CRP)
  c = (tl.program_id(0) % SUB) * CR + ci                  # row within the group
  cm = (ci < CR) & (c < 2 * N)
  cg = group * 2 * N + c                                  # row within the 4N (arm, head) rows
  row = tl.program_id(1)
  seq = row // T
  t = row % T
  pos = tl.load(POS) + t
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, K1P)
  km1 = k1 < K
  v = tl.arange(0, VP)
  vm = v < V
  # read keys (independent of M: issued first so their loads overlap the M work)
  sk = tl.load(SKEY + cg[:, None] * V + v[None, :], mask=cm[:, None] & vm[None, :], other=0.).to(tl.float32)
  up = tl.load(UP + row * s_up + cg[:, None] * V + v[None, :], mask=cm[:, None] & vm[None, :],
               other=0.).to(tl.float32)
  gd = key_scale * tl.sigmoid(tl.load(GATES + row * s_gt + cg, mask=cm, other=0.).to(tl.float32))
  gs = static_scale * tl.sigmoid(tl.load(GATES + row * s_gt + 4 * N + cg, mask=cm, other=0.).to(tl.float32))
  r = (up * (gd * tl.rsqrt(tl.sum(up * up, 1) / V + key_eps))[:, None] + gs[:, None] * sk).to(tl.bfloat16)
  # read view
  if FUSE_WRITE:
    m0, m1 = _new_m(row, M, s_m, C1, s_c1, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, write_eps,
                    N, NP, K, K1P, V, VP)
    if tl.program_id(0) == 0:
      _store_m(MW + row * (K * V), m0, m1, K, K1P, V, VP)
  else:
    m0, m1 = _load_m(M + row * s_m, K, K1P, V, VP)
  f0 = m0.to(tl.float32)
  f1 = m1.to(tl.float32)
  if t > 0:
    pv = 1
    pp = M + (row - 1) * s_m
  elif HAS_PREV:
    pv = tl.load(PREV_VALID)
    pp = MPREV + seq * s_mp
  else:
    pv = 0
    pp = M + row * s_m
  if pv != 0:
    p0, p1 = _load_m(pp, K, K1P, V, VP)
    s = tl.sigmoid(tl.load(SHIFT + row * s_sh + v, mask=vm, other=0.).to(tl.float32))
    f0 = (f0 + s[None, :] * (p0.to(tl.float32) - f0)).to(tl.bfloat16).to(tl.float32)
    f1 = (f1 + s[None, :] * (p1.to(tl.float32) - f1)).to(tl.bfloat16).to(tl.float32)
  inv = tl.rsqrt((tl.sum(tl.sum(f0 * f0, 1), 0) + tl.sum(tl.sum(f1 * f1, 1), 0)) / (K * V) + eps)
  o0 = tl.dot(r, tl.trans(f0.to(tl.bfloat16)))          # [CRP, 64]
  o1 = tl.dot(r, tl.trans(f1.to(tl.bfloat16)))          # [CRP, K1P]
  head = c % N
  if group == 0:     # q, k: [c_qk read | std] with the RoPE tail, q scaled
    arm = c // N
    qm = tl.where(arm == 0, q_mult, 1.0)
    sq = inv * qk_scale * qm
    y0 = (o0 * sq[:, None]).to(tl.bfloat16)
    y1 = (o1 * sq[:, None]).to(tl.bfloat16)
    j = tl.arange(0, HP)
    jm = j < H
    sa = tl.load(STD + row * s_std + c[:, None] * (2 * H) + j[None, :], mask=cm[:, None] & jm[None, :],
                 other=0.).to(tl.float32)
    sb = tl.load(STD + row * s_std + c[:, None] * (2 * H) + H + j[None, :], mask=cm[:, None] & jm[None, :],
                 other=0.).to(tl.float32)
    cs = tl.load(COS + pos * H + j, mask=jm, other=0.).to(tl.float32)
    sn = tl.load(SIN + pos * H + j, mask=jm, other=0.).to(tl.float32)
    ra = ((sa * cs[None, :] - sb * sn[None, :]) * qm[:, None]).to(tl.bfloat16)
    rb = ((sb * cs[None, :] + sa * sn[None, :]) * qm[:, None]).to(tl.bfloat16)
    base = QK + row * s_qk + c[:, None] * QW
    sm = cm[:, None]
    tl.store(base + k0[None, :], y0, mask=sm)
    tl.store(base + k1[None, :], y1, mask=sm & km1[None, :])
    tl.store(base + K + j[None, :], ra, mask=sm & jm[None, :])
    tl.store(base + K + H + j[None, :], rb, mask=sm & jm[None, :])
    if CACHE:      # one-token decode: the key rows go to the key cache at position pos
      kmask = cm[:, None] & (arm == 1)[:, None]
      kb = KC + seq * skb + head[:, None] * skh + pos * sks
      tl.store(kb + k0[None, :], y0, mask=kmask)
      tl.store(kb + k1[None, :], y1, mask=kmask & km1[None, :])
      tl.store(kb + K + j[None, :], ra, mask=kmask & jm[None, :])
      tl.store(kb + K + H + j[None, :], rb, mask=kmask & jm[None, :])
  else:              # v, local o
    sv = inv * vo_scale
    y0 = (o0 * sv).to(tl.bfloat16)
    y1 = (o1 * sv).to(tl.bfloat16)
    base = VO + row * s_vo + c[:, None] * K
    sm = cm[:, None]
    tl.store(base + k0[None, :], y0, mask=sm)
    tl.store(base + k1[None, :], y1, mask=sm & km1[None, :])
    if CACHE:      # the value rows go to the value cache
      vmask = cm[:, None] & (c < N)[:, None]
      vb = VC + seq * svb + head[:, None] * svh + pos * svs
      tl.store(vb + k0[None, :], y0, mask=vmask)
      tl.store(vb + k1[None, :], y1, mask=vmask & km1[None, :])


@triton.jit
def _write_kernel(MOLD, s_mo, MW, C1, s_c1, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, write_eps,
                  N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr,
                  VP: tl.constexpr):
  row = tl.program_id(0)
  m0, m1 = _new_m(row, MOLD, s_mo, C1, s_c1, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, write_eps,
                  N, NP, K, K1P, V, VP)
  _store_m(MW + row * (K * V), m0, m1, K, K1P, V, VP)


def qk_width(k, s):
  return -(-(k + s) // 8) * 8


@torch.library.custom_op('bam_v5::fused_read', mutates_args=())
def fused_read(m_old: torch.Tensor, c1: torch.Tensor, g1: torch.Tensor, a1: torch.Tensor, c2: torch.Tensor,
               g2: torch.Tensor, a2: torch.Tensor, m_prev: torch.Tensor, prev_valid: torch.Tensor,
               pos: torch.Tensor, shift: torch.Tensor, std: torch.Tensor, up: torch.Tensor, gates: torch.Tensor,
               static_key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               batch: int, n: int, k: int, v: int, s: int, has_write: bool, has_prev: bool, write_eps: float,
               eps: float, key_eps: float, key_scale: float, static_scale: float, qk_scale: float, vo_scale: float,
               q_mult: float, sub: int, warps: int, k_cache: Optional[torch.Tensor] = None,
               v_cache: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Token rows r = b * T + t (2-D [rows, features], unit inner stride). Returns (m_new [rows, K*V]
  (empty when not has_write), qk [rows, 2, N, QW] (K + S used), vo [rows, 2, N, K]).

  k_cache [B,N,S,Dq] / v_cache [B,N,S,K] (rows layout, one-token decode): the token's key and value are
  written at position pos. Not declared as a mutation (see decode_attention.attend_and_append): within a
  decode step only this op writes a layer's caches, and the strictly causal attention that follows (it
  consumes this op's output) reads positions < pos only."""
  rows = m_old.shape[0]
  t = rows // batch
  qw = qk_width(k, s)
  qk = m_old.new_empty((rows, 2, n, qw))
  vo = m_old.new_empty((rows, 2, n, k))
  m_new = m_old.new_empty((rows if has_write else 0, k * v))
  dummy = m_old
  meta = dict(N=n, NP=_pow2(n), K=k, K1P=_pow2(k - 64), V=v, VP=_pow2(v))
  fuse = has_write and t == 1
  if has_write and not fuse:     # prefill: a separate write pass (fused, the read kernel spills)
    _write_kernel[(rows,)](m_old, m_old.stride(0), m_new, c1, c1.stride(0), g1, g1.stride(0), a1, a1.stride(0),
                           c2, c2.stride(0), g2, g2.stride(0), a2, a2.stride(0), write_eps, **meta, num_warps=4)
    m_old = m_new
  if not fuse:
    c1 = g1 = a1 = c2 = g2 = a2 = dummy
  if not has_prev:
    m_prev = dummy
  cache = k_cache is not None
  assert not cache or t == 1
  kc = k_cache if cache else dummy
  vc = v_cache if cache else dummy
  cr = -(-2 * n // sub)
  _read_kernel[(2 * sub, rows)](
      m_old, m_old.stride(0), m_new if fuse else dummy, c1, c1.stride(0), g1, g1.stride(0), a1, a1.stride(0), c2,
      c2.stride(0), g2, g2.stride(0), a2, a2.stride(0), write_eps, m_prev, m_prev.stride(0), prev_valid, pos,
      shift, shift.stride(0), std, std.stride(0), up, up.stride(0), gates, gates.stride(0), static_key, cos, sin,
      qk, qk.stride(0), vo, vo.stride(0),
      kc, kc.stride(0) if cache else 0, kc.stride(1) if cache else 0, kc.stride(2) if cache else 0,
      vc, vc.stride(0) if cache else 0, vc.stride(1) if cache else 0, vc.stride(2) if cache else 0,
      t, eps, key_eps, key_scale, static_scale, qk_scale, vo_scale, q_mult,
      HAS_PREV=has_prev, FUSE_WRITE=fuse, CACHE=cache, SUB=sub, H=s // 2, HP=_pow2(s // 2), QW=qw, CR=cr,
      CRP=_pow2(cr), **meta, num_warps=warps)
  return m_new, qk, vo


@fused_read.register_fake
def _(m_old, c1, g1, a1, c2, g2, a2, m_prev, prev_valid, pos, shift, std, up, gates, static_key, cos, sin,
      batch, n, k, v, s, has_write, has_prev, write_eps, eps, key_eps, key_scale, static_scale, qk_scale, vo_scale,
      q_mult, sub, warps, k_cache=None, v_cache=None):
  rows = m_old.shape[0]
  return (m_old.new_empty((rows if has_write else 0, k * v)), m_old.new_empty((rows, 2, n, qk_width(k, s))),
          m_old.new_empty((rows, 2, n, k)))
