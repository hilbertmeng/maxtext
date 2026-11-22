"""Triton kernels for the per-token M work of the LoomFormer V5 fast path (shapes generic: K > 64, any V, S, N).

`fused_read`: the previous layer's write and this layer's read in one pass over M.

  M_new = M_old + sum_n sigmoid(g1_n) rms(c1_n) (x) rms(a1_n) + sum_n sigmoid(g2_n) c2_n (x) rms(a2_n)
  M~    = M_new + s (M_prev_new - M_new)           (per V column, s = sigmoid(shift); t > 0 only)
  R     = c_key sigmoid(g_a) rms(up_a) + c_static sigmoid(h_a) S_a      (arms q, k, v, o; [4N, V])
  read  = R @ M~^T / rms(M~)                       ([4N, K])
  q, k  = RoPE_tail([c_qk read_{q,k} | std_{q,k}]), q * q_mult;  v, o_loc = c_vo read_{v,o}

Programs: axis 0 = (arm group (q, k) or (v, o), row chunk), axis 1 = token. M is held as tiles
(64 + K1P) x (V0 + V1) (K = 64 + tail, V = V0 + tail, tails padded to powers of two and masked). One-token decode fuses the previous
layer's write (program 0 stores M_new) and writes the token's key and value into the rows-layout caches.

Layouts (row = token, unit inner stride): M [K*V] (k major); content [N, K]; address [N, V]; up [4N, V];
gates [4N] dynamic then [4N] static (pre-activation); std [2N, S] (q then k); qk out [2, N, QW]
(K + S used); vo out [2, N, VW] (K used); padding columns are zero.
"""
from typing import Optional

import torch
import triton
import triton.language as tl


def _pow2(x):
  return max(16, triton.next_power_of_2(x))


def _vsplit(v):
  """V columns as V0 + V1 tiles (powers of two >= 16, V1 masked; 0 = absent): 40 -> 32 + 16, 32 -> 32."""
  v0 = 1 << (v.bit_length() - 1)
  if v <= 16 or v0 == v:
    return _pow2(v), 0
  return v0, _pow2(v - v0)


@triton.jit
def _load_m(P, K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr):
  """M tiles (k 64 + K1P) x (v V0 + V1) of one token; V1 == 0: the v1 tiles are zero placeholders."""
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, K1P)
  km = (k1 < K)[:, None]
  a = tl.arange(0, V0)
  am = (a < V)[None, :]
  m00 = tl.load(P + k0[:, None] * V + a[None, :], mask=am, other=0.)
  m10 = tl.load(P + k1[:, None] * V + a[None, :], mask=km & am, other=0.)
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = (b < V)[None, :]
    m01 = tl.load(P + k0[:, None] * V + b[None, :], mask=bm, other=0.)
    m11 = tl.load(P + k1[:, None] * V + b[None, :], mask=km & bm, other=0.)
  else:
    m01 = tl.zeros((64, 16), P.dtype.element_ty)
    m11 = tl.zeros((K1P, 16), P.dtype.element_ty)
  return m00, m01, m10, m11


@triton.jit
def _store_m(P, m00, m01, m10, m11, K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, V0: tl.constexpr,
             V1: tl.constexpr):
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, K1P)
  km = (k1 < K)[:, None]
  a = tl.arange(0, V0)
  am = (a < V)[None, :]
  tl.store(P + k0[:, None] * V + a[None, :], m00, mask=am)
  tl.store(P + k1[:, None] * V + a[None, :], m10, mask=km & am)
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = (b < V)[None, :]
    tl.store(P + k0[:, None] * V + b[None, :], m01, mask=bm)
    tl.store(P + k1[:, None] * V + b[None, :], m11, mask=km & bm)


@triton.jit
def _write_pair(row, C, s_c, CB, s_cb, h_cb, G, s_g, GB, A, s_a, write_eps, HAS_CB: tl.constexpr, RMS_C: tl.constexpr, N: tl.constexpr, NP: tl.constexpr,
                K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr):
  """sum_n sigmoid(g_n) c_n (x) rms(a_n) of one token as fp32 tiles (as _load_m); c_n RMS-normalized over K
  when RMS_C."""
  n = tl.arange(0, NP)
  nm = n < N
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, K1P)
  c0 = tl.load(C + row * s_c + n[:, None] * K + k0[None, :], mask=nm[:, None], other=0.).to(tl.float32)
  c1 = tl.load(C + row * s_c + n[:, None] * K + k1[None, :], mask=nm[:, None] & (k1 < K)[None, :],
               other=0.).to(tl.float32)
  if HAS_CB:     # content = C + CB (attention output + local O, the o_head sum rounded as materialized)
    c0 = (c0 + tl.load(CB + row * s_cb + n[:, None] * h_cb + k0[None, :], mask=nm[:, None],
                       other=0.).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    c1 = (c1 + tl.load(CB + row * s_cb + n[:, None] * h_cb + k1[None, :], mask=nm[:, None] & (k1 < K)[None, :],
                       other=0.).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
  sc = tl.sigmoid(tl.load(G + row * s_g + n, mask=nm, other=0.).to(tl.float32) +
                  tl.load(GB + n, mask=nm, other=0.).to(tl.float32))
  if RMS_C:
    sc = sc * tl.rsqrt((tl.sum(c0 * c0, 1) + tl.sum(c1 * c1, 1)) / K + write_eps)
  c0 = (c0 * sc[:, None]).to(tl.bfloat16)
  c1 = (c1 * sc[:, None]).to(tl.bfloat16)
  a = tl.arange(0, V0)
  a0 = tl.load(A + row * s_a + n[:, None] * V + a[None, :], mask=nm[:, None] & (a < V)[None, :],
               other=0.).to(tl.float32)
  ss = tl.sum(a0 * a0, 1)
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    a1 = tl.load(A + row * s_a + n[:, None] * V + b[None, :], mask=nm[:, None] & (b < V)[None, :],
                 other=0.).to(tl.float32)
    ss += tl.sum(a1 * a1, 1)
  rr = tl.rsqrt(ss / V + write_eps)[:, None]
  a0 = (a0 * rr).to(tl.bfloat16)
  if V1 > 0:
    a1 = (a1 * rr).to(tl.bfloat16)
    d01 = tl.dot(tl.trans(c0), a1)
    d11 = tl.dot(tl.trans(c1), a1)
  else:
    d01 = tl.zeros((64, 16), tl.float32)
    d11 = tl.zeros((K1P, 16), tl.float32)
  return tl.dot(tl.trans(c0), a0), d01, tl.dot(tl.trans(c1), a0), d11


@triton.jit
def _new_m(row, MOLD, s_mo, C1, s_c1, C1B, s_c1b, h_c1b, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, WB,
           write_eps, HAS_CB: tl.constexpr,
           N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr,
           V0: tl.constexpr, V1: tl.constexpr):
  m00, m01, m10, m11 = _load_m(MOLD + row * s_mo, K, K1P, V, V0, V1)
  d00, d01, d10, d11 = _write_pair(row, C1, s_c1, C1B, s_c1b, h_c1b, G1, s_g1, WB, A1, s_a1, write_eps, HAS_CB,
                                   True, N, NP, K, K1P, V, V0, V1)
  e00, e01, e10, e11 = _write_pair(row, C2, s_c2, C2, 0, 0, G2, s_g2, WB + N, A2, s_a2, write_eps, False,
                                   False, N, NP, K, K1P, V, V0, V1)
  m00 = (m00.to(tl.float32) + (d00 + e00)).to(tl.bfloat16)
  m10 = (m10.to(tl.float32) + (d10 + e10)).to(tl.bfloat16)
  if V1 > 0:
    m01 = (m01.to(tl.float32) + (d01 + e01)).to(tl.bfloat16)
    m11 = (m11.to(tl.float32) + (d11 + e11)).to(tl.bfloat16)
  return m00, m01, m10, m11


@triton.jit
def _group_keys(group, row, c, cm, UP, s_up, GATES, s_gt, GB, SKEY, key_eps, key_scale, static_scale, N: tl.constexpr,
                V: tl.constexpr, V0: tl.constexpr, V1: tl.constexpr, CRP: tl.constexpr):
  """Read keys R (bf16, V0 + V1 column tiles) of rows c of arm group `group`: independent of M."""
  cg = group * 2 * N + c                                  # row within the 4N (arm, head) rows
  rm = cm[:, None]
  gd = key_scale * tl.sigmoid(tl.load(GATES + row * s_gt + cg, mask=cm, other=0.).to(tl.float32) +
                              tl.load(GB + cg, mask=cm, other=0.).to(tl.float32))
  gs = static_scale * tl.sigmoid(tl.load(GATES + row * s_gt + 4 * N + cg, mask=cm, other=0.).to(tl.float32) +
                                 tl.load(GB + 4 * N + cg, mask=cm, other=0.).to(tl.float32))
  a = tl.arange(0, V0)
  am = rm & (a < V)[None, :]
  sk0 = tl.load(SKEY + cg[:, None] * V + a[None, :], mask=am, other=0.).to(tl.float32)
  up0 = tl.load(UP + row * s_up + cg[:, None] * V + a[None, :], mask=am, other=0.).to(tl.float32)
  ss = tl.sum(up0 * up0, 1)
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    bm = rm & (b < V)[None, :]
    sk1 = tl.load(SKEY + cg[:, None] * V + b[None, :], mask=bm, other=0.).to(tl.float32)
    up1 = tl.load(UP + row * s_up + cg[:, None] * V + b[None, :], mask=bm, other=0.).to(tl.float32)
    ss += tl.sum(up1 * up1, 1)
  gk = (gd * tl.rsqrt(ss / V + key_eps))[:, None]
  r0 = (up0 * gk + gs[:, None] * sk0).to(tl.bfloat16)
  if V1 > 0:
    r1 = (up1 * gk + gs[:, None] * sk1).to(tl.bfloat16)
  else:
    r1 = tl.zeros((CRP, 16), tl.bfloat16)
  return r0, r1


@triton.jit
def _group_read(group, row, seq, pos, c, cm, r0, r1, f00, f01, f10, f11, inv, STD, s_std, COS, SIN,
                QK, s_qk, VO, s_vo, KC, skb, skh, sks, VC, svb, svh, svs, qk_scale, vo_scale, q_mult,
                CACHE: tl.constexpr, N: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr, V1: tl.constexpr,
                H: tl.constexpr, HP: tl.constexpr, QW: tl.constexpr, VW: tl.constexpr, CRP: tl.constexpr):
  """Rows c (of the group's 2N (arm, head) rows) of arm group 0 (q, k) or 1 (v, local o) for one token,
  from its read keys, the bf16 read view tiles and its inverse RMS."""
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, K1P)
  km1 = k1 < K
  o0 = tl.dot(r0, tl.trans(f00))          # [CRP, 64]
  o1 = tl.dot(r0, tl.trans(f10))          # [CRP, K1P]
  if V1 > 0:
    o0 += tl.dot(r1, tl.trans(f01))
    o1 += tl.dot(r1, tl.trans(f11))
  head = c % N
  sm = cm[:, None]
  if group == 0:     # q, k: [c_qk read | std] with the RoPE tail, q scaled
    arm = c // N
    qm = tl.where(arm == 0, q_mult, 1.0)
    sq = inv * qk_scale * qm
    y0 = (o0 * sq[:, None]).to(tl.bfloat16)
    y1 = (o1 * sq[:, None]).to(tl.bfloat16)
    j = tl.arange(0, HP)
    jm = j < H
    sa = tl.load(STD + row * s_std + c[:, None] * (2 * H) + j[None, :], mask=sm & jm[None, :],
                 other=0.).to(tl.float32)
    sb = tl.load(STD + row * s_std + c[:, None] * (2 * H) + H + j[None, :], mask=sm & jm[None, :],
                 other=0.).to(tl.float32)
    cs = tl.load(COS + pos * H + j, mask=jm, other=0.).to(tl.float32)
    sn = tl.load(SIN + pos * H + j, mask=jm, other=0.).to(tl.float32)
    ra = ((sa * cs[None, :] - sb * sn[None, :]) * qm[:, None]).to(tl.bfloat16)
    rb = ((sb * cs[None, :] + sa * sn[None, :]) * qm[:, None]).to(tl.bfloat16)
    base = QK + row * s_qk + c[:, None] * QW
    tl.store(base + k0[None, :], y0, mask=sm)
    tl.store(base + k1[None, :], y1, mask=sm & km1[None, :])
    tl.store(base + K + j[None, :], ra, mask=sm & jm[None, :])
    tl.store(base + K + H + j[None, :], rb, mask=sm & jm[None, :])
    if QW > K + 2 * H:     # zero padding (prefill attention reads QW columns)
      zp = tl.arange(0, 8)
      tl.store(base + K + 2 * H + zp[None, :], tl.zeros((CRP, 8), tl.bfloat16),
               mask=sm & (zp[None, :] < QW - K - 2 * H))
    if CACHE:      # one-token decode: the key rows go to the key cache at position pos
      kmask = sm & (arm == 1)[:, None]
      kb = KC + seq * skb + head[:, None] * skh + pos * sks
      tl.store(kb + k0[None, :], y0, mask=kmask)
      tl.store(kb + k1[None, :], y1, mask=kmask & km1[None, :])
      tl.store(kb + K + j[None, :], ra, mask=kmask & jm[None, :])
      tl.store(kb + K + H + j[None, :], rb, mask=kmask & jm[None, :])
  else:              # v, local o
    sv = inv * vo_scale
    y0 = (o0 * sv).to(tl.bfloat16)
    y1 = (o1 * sv).to(tl.bfloat16)
    base = VO + row * s_vo + c[:, None] * VW
    tl.store(base + k0[None, :], y0, mask=sm)
    tl.store(base + k1[None, :], y1, mask=sm & (k1 < VW)[None, :])     # columns K..VW-1 are zero
    if CACHE:      # the value rows go to the value cache
      vmask = sm & (c < N)[:, None]
      vb = VC + seq * svb + head[:, None] * svh + pos * svs
      tl.store(vb + k0[None, :], y0, mask=vmask)
      tl.store(vb + k1[None, :], y1, mask=vmask & km1[None, :])


@triton.jit
def _read_kernel(
    M, s_m, MW, C1, s_c1, C1B, s_c1b, h_c1b, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, write_eps, MPREV, s_mp,
    PREV_VALID, POS,
    SHIFT, s_sh, STD, s_std, UP, s_up, GATES, s_gt, XB, WB, SKEY, COS, SIN,
    QK, s_qk, VO, s_vo, KC, skb, skh, sks, VC, svb, svh, svs,
    T, eps, key_eps, key_scale, static_scale, qk_scale, vo_scale, q_mult,
    HAS_PREV: tl.constexpr, FUSE_WRITE: tl.constexpr, HAS_CB: tl.constexpr, CACHE: tl.constexpr, SUB: tl.constexpr, GROUPS: tl.constexpr,
    N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr, V0: tl.constexpr,
    V1: tl.constexpr, H: tl.constexpr, HP: tl.constexpr, QW: tl.constexpr, VW: tl.constexpr, CR: tl.constexpr,
    CRP: tl.constexpr):
  """Axis 0 = (GROUPS == 1: arm group) * SUB + row chunk (CR of a group's 2N (arm, head) rows), axis 1 = token
  row. GROUPS == 2: one program reads both groups from one read view."""
  ci = tl.arange(0, CRP)
  c = (tl.program_id(0) % SUB) * CR + ci                  # row within the group
  cm = (ci < CR) & (c < 2 * N)
  row = tl.program_id(1)
  seq = row // T
  t = row % T
  pos = tl.load(POS) + t
  g0 = (tl.program_id(0) // SUB) * GROUPS
  ra0, ra1 = _group_keys(g0, row, c, cm, UP, s_up, GATES, s_gt, XB + V, SKEY, key_eps, key_scale, static_scale, N, V, V0, V1,
                         CRP)
  if GROUPS == 2:
    rb0, rb1 = _group_keys(g0 + 1, row, c, cm, UP, s_up, GATES, s_gt, XB + V, SKEY, key_eps, key_scale, static_scale, N, V,
                           V0, V1, CRP)
  # the predecessor's M (issued first: independent of this token's write); pv = 0 -> s = 0, exact no-op
  if t > 0:
    pv = 1.0
    pp = M + (row - 1) * s_m
  elif HAS_PREV:
    pv = tl.load(PREV_VALID).to(tl.float32)
    pp = MPREV + seq * s_mp + ((pos + 1) % 2) * (K * V)
  else:
    pv = 0.0
    pp = M + row * s_m
  p00, p01, p10, p11 = _load_m(pp, K, K1P, V, V0, V1)
  a = tl.arange(0, V0)
  s0 = (pv * tl.sigmoid(tl.load(SHIFT + row * s_sh + a, mask=a < V, other=0.).to(tl.float32) +
                       tl.load(XB + a, mask=a < V, other=0.).to(tl.float32)))[None, :]
  if V1 > 0:
    b = V0 + tl.arange(0, V1)
    s1 = (pv * tl.sigmoid(tl.load(SHIFT + row * s_sh + b, mask=b < V, other=0.).to(tl.float32) +
                         tl.load(XB + b, mask=b < V, other=0.).to(tl.float32)))[None, :]
  if FUSE_WRITE:
    m00, m01, m10, m11 = _new_m(row, M, s_m, C1, s_c1, C1B, s_c1b, h_c1b, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2,
                                s_a2, WB, write_eps, HAS_CB,
                                N, NP, K, K1P, V, V0, V1)
    if tl.program_id(0) == 0:
      _store_m(MW + row * (K * V), m00, m01, m10, m11, K, K1P, V, V0, V1)
      if HAS_PREV:     # decode: also into the state slot of this position (the predecessor's is read above)
        _store_m(MPREV + seq * s_mp + (pos % 2) * (K * V), m00, m01, m10, m11, K, K1P, V, V0, V1)
  else:
    m00, m01, m10, m11 = _load_m(M + row * s_m, K, K1P, V, V0, V1)
  f00 = m00.to(tl.float32)
  f10 = m10.to(tl.float32)
  f00 = (f00 + s0 * (p00.to(tl.float32) - f00)).to(tl.bfloat16).to(tl.float32)
  f10 = (f10 + s0 * (p10.to(tl.float32) - f10)).to(tl.bfloat16).to(tl.float32)
  f01 = m01.to(tl.float32)
  f11 = m11.to(tl.float32)
  if V1 > 0:
    f01 = (f01 + s1 * (p01.to(tl.float32) - f01)).to(tl.bfloat16).to(tl.float32)
    f11 = (f11 + s1 * (p11.to(tl.float32) - f11)).to(tl.bfloat16).to(tl.float32)
  ss = tl.sum(tl.sum(f00 * f00, 1), 0) + tl.sum(tl.sum(f10 * f10, 1), 0)
  if V1 > 0:
    ss += tl.sum(tl.sum(f01 * f01, 1), 0) + tl.sum(tl.sum(f11 * f11, 1), 0)
  inv = tl.rsqrt(ss / (K * V) + eps)
  fb00 = f00.to(tl.bfloat16)
  fb01 = f01.to(tl.bfloat16)
  fb10 = f10.to(tl.bfloat16)
  fb11 = f11.to(tl.bfloat16)
  _group_read(g0, row, seq, pos, c, cm, ra0, ra1, fb00, fb01, fb10, fb11, inv, STD, s_std, COS, SIN, QK, s_qk, VO,
              s_vo, KC, skb, skh, sks, VC, svb, svh, svs, qk_scale, vo_scale, q_mult, CACHE, N, K, K1P, V1, H, HP, QW,
              VW, CRP)
  if GROUPS == 2:
    _group_read(g0 + 1, row, seq, pos, c, cm, rb0, rb1, fb00, fb01, fb10, fb11, inv, STD, s_std, COS, SIN, QK, s_qk,
                VO, s_vo, KC, skb, skh, sks, VC, svb, svh, svs, qk_scale, vo_scale, q_mult, CACHE, N, K, K1P, V1, H,
                HP, QW, VW, CRP)


@triton.jit
def _write_kernel(MOLD, s_mo, MW, C1, s_c1, C1B, s_c1b, h_c1b, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, WB,
                  write_eps, HAS_CB: tl.constexpr,
                  N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, K1P: tl.constexpr, V: tl.constexpr,
                  V0: tl.constexpr, V1: tl.constexpr):
  row = tl.program_id(0)
  m00, m01, m10, m11 = _new_m(row, MOLD, s_mo, C1, s_c1, C1B, s_c1b, h_c1b, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2,
                              A2, s_a2, WB, write_eps, HAS_CB, N, NP, K, K1P, V, V0, V1)
  _store_m(MW + row * (K * V), m00, m01, m10, m11, K, K1P, V, V0, V1)


def _round8(x):
  return -(-x // 8) * 8


def qk_width(k, s):
  return _round8(k + s)


@torch.library.custom_op('bam_v5::fused_read', mutates_args=())
def fused_read(m_old: torch.Tensor, c1: torch.Tensor, g1: torch.Tensor, a1: torch.Tensor, c2: torch.Tensor,
               g2: torch.Tensor, a2: torch.Tensor, m_prev: torch.Tensor, prev_valid: torch.Tensor,
               pos: torch.Tensor, shift: torch.Tensor, std: torch.Tensor, up: torch.Tensor, gates: torch.Tensor,
               static_key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, xb: torch.Tensor, wb: torch.Tensor,
               batch: int, n: int, k: int, v: int, s: int, has_write: bool, has_prev: bool, write_eps: float,
               eps: float, key_eps: float, key_scale: float, static_scale: float, qk_scale: float, vo_scale: float,
               q_mult: float, sub: int, groups: int, warps: int, k_cache: Optional[torch.Tensor] = None,
               v_cache: Optional[torch.Tensor] = None, g1_off: int = 0, g2_off: int = 0, shift_off: int = 0,
               std_off: int = 0, gates_off: int = 0, c1_vo: Optional[torch.Tensor] = None
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Token rows r = b * T + t (2-D [rows, features], unit inner stride). Returns (m_new [rows, K*V]
  (empty when not has_write), qk [rows, 2, N, QW] (K + S used), vo [rows, 2, N, VW] (K used)); widths
  rounded up to multiples of 8, zero-padded.

  One-token decode (has_prev): m_prev [B, 2, K*V] holds the layer-input M of position q in slot q % 2; the
  predecessor's slot is read and, with has_write, M_new stored into slot pos % 2. k_cache [B,N,S,Dq] /
  v_cache [B,N,S,K] (rows layout): the token's key and value are written at position pos. Not declared as
  mutations (see decode_attention.attend_and_append): within a decode step only this op writes a layer's
  caches and M slots, and the strictly causal attention that follows (it consumes this op's output) reads
  positions < pos only."""
  # xb = [shift bias (V) | read-gate biases (8N)] of this layer, wb = [attention | MLP write-gate bias (N each)]
  # of the pending write: the projections themselves are bias-free mm (as MHA's)
  # g1, g2, shift, std, gates may be whole projection outputs with the column offsets *_off (column slices
  # passed into a custom op are copied by inductor when their producer is a fused kernel)
  g1, g2, shift, std, gates = (x[:, o:] for x, o in ((g1, g1_off), (g2, g2_off), (shift, shift_off), (std, std_off),
                                                     (gates, gates_off)))
  # c1_vo: the pending layer's vo output; the attention-write content is then c1 + its local O (vo[:, 1])
  cb = c1_vo[:, 1] if c1_vo is not None else m_old
  cbs = (cb.stride(0), cb.stride(1)) if c1_vo is not None else (0, 0)
  rows = m_old.shape[0]
  t = rows // batch
  qw = qk_width(k, s)
  qk = m_old.new_empty((rows, 2, n, qw))
  vo = m_old.new_empty((rows, 2, n, _round8(k)))
  m_new = m_old.new_empty((rows if has_write else 0, k * v))
  dummy = m_old
  v0, v1 = _vsplit(v)
  meta = dict(N=n, NP=_pow2(n), K=k, K1P=_pow2(k - 64), V=v, V0=v0, V1=v1)
  fuse = has_write and t == 1
  if has_write and not fuse:     # prefill: a separate write pass (fused, the read kernel spills)
    _write_kernel[(rows,)](m_old, m_old.stride(0), m_new, c1, c1.stride(0), cb, *cbs, g1, g1.stride(0), a1, a1.stride(0),
                           c2, c2.stride(0), g2, g2.stride(0), a2, a2.stride(0), wb, write_eps, c1_vo is not None, **meta,
                           num_warps=1 if v1 == 0 and v0 <= 32 else 2)
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
  _read_kernel[(2 // groups * sub, rows)](
      m_old, m_old.stride(0), m_new if fuse else dummy, c1, c1.stride(0), cb, *cbs, g1, g1.stride(0), a1, a1.stride(0), c2,
      c2.stride(0), g2, g2.stride(0), a2, a2.stride(0), write_eps, m_prev, m_prev.stride(0), prev_valid, pos,
      shift, shift.stride(0), std, std.stride(0), up, up.stride(0), gates, gates.stride(0), xb, wb, static_key, cos,
      sin,
      qk, qk.stride(0), vo, vo.stride(0),
      kc, kc.stride(0) if cache else 0, kc.stride(1) if cache else 0, kc.stride(2) if cache else 0,
      vc, vc.stride(0) if cache else 0, vc.stride(1) if cache else 0, vc.stride(2) if cache else 0,
      t, eps, key_eps, key_scale, static_scale, qk_scale, vo_scale, q_mult,
      HAS_PREV=has_prev, FUSE_WRITE=fuse, HAS_CB=fuse and c1_vo is not None, CACHE=cache, SUB=sub, GROUPS=groups, H=s // 2, HP=_pow2(s // 2),
      QW=qw, VW=_round8(k), CR=cr, CRP=_pow2(cr), **meta, num_warps=warps)
  return m_new, qk, vo


@fused_read.register_fake
def _(m_old, c1, g1, a1, c2, g2, a2, m_prev, prev_valid, pos, shift, std, up, gates, static_key, cos, sin, xb, wb,
      batch, n, k, v, s, has_write, has_prev, write_eps, eps, key_eps, key_scale, static_scale, qk_scale, vo_scale,
      q_mult, sub, groups, warps, k_cache=None, v_cache=None, g1_off=0, g2_off=0, shift_off=0, std_off=0,
      gates_off=0, c1_vo=None):
  rows = m_old.shape[0]
  return (m_old.new_empty((rows if has_write else 0, k * v)), m_old.new_empty((rows, 2, n, qk_width(k, s))),
          m_old.new_empty((rows, 2, n, _round8(k))))


@triton.jit
def _gemv_part(pid, X, XO, XI, W, B, Y, R, K, ACT: tl.constexpr, HAS_B: tl.constexpr, BR: tl.constexpr,
               BK: tl.constexpr, LNW=None, eps=0.):
  """Rows pid*BR.. of y = W f(x) (+ b), x = X[XO:XO + K]; f: 0 identity, 1 tanh GELU, 3 SwiGLU
  silu(X[XO + k]) * X[XO + XI + k], 4 RMSNorm (fp32 statistics, weight LNW); intermediate values rounded to
  bf16 as when materialized."""
  r = pid * BR + tl.arange(0, BR)
  rm = r < R
  acc = tl.zeros((BR,), tl.float32)
  if ACT == 4:
    sq = tl.zeros((BK,), tl.float32)
    for k0 in range(0, K, BK):
      kk = k0 + tl.arange(0, BK)
      xs = tl.load(X + XO + kk, mask=kk < K, other=0.).to(tl.float32)
      sq += xs * xs
    rstd = tl.rsqrt(tl.sum(sq, 0) / K + eps)
  for k0 in range(0, K, BK):
    kk = k0 + tl.arange(0, BK)
    km = kk < K
    x = tl.load(X + XO + kk, mask=km, other=0.).to(tl.float32)
    if ACT == 4:
      x = (x * rstd * tl.load(LNW + kk, mask=km, other=0.)).to(tl.bfloat16).to(tl.float32)
    if ACT == 1:
      inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
      e = tl.exp(-2.0 * tl.abs(inner))
      th = (1 - e) / (1 + e)
      x = (0.5 * x * (1 + tl.where(inner >= 0, th, -th))).to(tl.bfloat16).to(tl.float32)
    elif ACT == 3:
      u = tl.load(X + XO + XI + kk, mask=km, other=0.).to(tl.float32)
      x = ((x * tl.sigmoid(x)).to(tl.bfloat16).to(tl.float32) * u).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + r[:, None] * K + kk[None, :], mask=rm[:, None] & km[None, :], other=0.).to(tl.float32)
    acc += tl.sum(w * x[None, :], 1)
  if HAS_B:
    acc += tl.load(B + r, mask=rm, other=0.).to(tl.float32)
  tl.store(Y + r, acc.to(tl.bfloat16), mask=rm)


@triton.jit
def _gemv2_kernel(X1, XO1, XI1, W1, B1, Y1, R1, K1, P1, X2, XO2, XI2, W2, B2, Y2, R2, K2,
                  ACT1: tl.constexpr, HB1: tl.constexpr, ACT2: tl.constexpr, HB2: tl.constexpr, BR: tl.constexpr,
                  BK: tl.constexpr):
  pid = tl.program_id(0)
  if pid < P1:
    _gemv_part(pid, X1, XO1, XI1, W1, B1, Y1, R1, K1, ACT1, HB1, BR, BK)
  else:
    _gemv_part(pid - P1, X2, XO2, XI2, W2, B2, Y2, R2, K2, ACT2, HB2, BR, BK)


@torch.library.custom_op('bam_v5::gemv2', mutates_args=())
def gemv2(x1: torch.Tensor, xo1: int, xi1: int, w1: torch.Tensor, b1: Optional[torch.Tensor], act1: int,
          x2: torch.Tensor, xo2: int, xi2: int, w2: torch.Tensor, b2: Optional[torch.Tensor], act2: int,
          br: int, bk: int, warps: int) -> tuple[torch.Tensor, torch.Tensor]:
  """Two one-token linears y_i = W_i f_i(x_i[xo_i:]) (+ b_i) in one launch (BAM-only decode projections);
  x_i [1, *] whole projection outputs, column offset xo_i, SwiGLU (act 3) up half at xo_i + xi_i."""
  y1, y2 = x1.new_empty((1, w1.shape[0])), x2.new_empty((1, w2.shape[0]))
  p1 = triton.cdiv(w1.shape[0], br)
  _gemv2_kernel[(p1 + triton.cdiv(w2.shape[0], br),)](
      x1, xo1, xi1, w1, b1 if b1 is not None else w1, y1, w1.shape[0], w1.shape[1], p1,
      x2, xo2, xi2, w2, b2 if b2 is not None else w2, y2, w2.shape[0], w2.shape[1],
      ACT1=act1, HB1=b1 is not None, ACT2=act2, HB2=b2 is not None, BR=br, BK=bk, num_warps=warps)
  return y1, y2


@gemv2.register_fake
def _(x1, xo1, xi1, w1, b1, act1, x2, xo2, xi2, w2, b2, act2, br, bk, warps):
  return x1.new_empty((1, w1.shape[0])), x2.new_empty((1, w2.shape[0]))


@triton.jit
def _gemv_rms_kernel(X, LNW, eps, W, B, Y, R, K, HAS_B: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
  _gemv_part(tl.program_id(0), X, 0, 0, W, B, Y, R, K, 4, HAS_B, BR, BK, LNW, eps)


@torch.library.custom_op('bam_v5::gemv_rms', mutates_args=())
def gemv_rms(x: torch.Tensor, ln_w: torch.Tensor, eps: float, w: torch.Tensor, b: Optional[torch.Tensor], br: int,
             bk: int, warps: int) -> torch.Tensor:
  """One-token y = W RMSNorm(x) (+ b), x [1, K] (RMSNorm weight ln_w fp32): LoomFormer's biased input projection
  (inductor sends addmm to cuBLAS with a separate RMSNorm kernel)."""
  y = x.new_empty((1, w.shape[0]))
  _gemv_rms_kernel[(triton.cdiv(w.shape[0], br),)](x, ln_w, eps, w, b if b is not None else w, y, w.shape[0],
                                                   w.shape[1], HAS_B=b is not None, BR=br, BK=bk, num_warps=warps)
  return y


@gemv_rms.register_fake
def _(x, ln_w, eps, w, b, br, bk, warps):
  return x.new_empty((1, w.shape[0]))
