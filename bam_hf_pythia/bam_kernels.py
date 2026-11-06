"""Triton kernels for the per-token M work of the Pythia unit-H BAM fast path.

`fused_read`: the previous layer's write and this layer's read in one pass over M.

  M_new = M_old + sum_n sigmoid(g1_n) rms(c1_n) (x) rms(a1_n) + sum_n sigmoid(g2_n) c2_n (x) rms(a2_n)
  M~    = M_new + s (M_prev_new - M_new)           (per V column, s = sigmoid(shift); t > 0 only)
  R     = 0.1 sigmoid(g_a) rms(up_a) + (1.1 / 0.99) sigmoid(h_a) S_a      (arms q, k, v, o; [4N, V])
  read  = R @ M~^T / rms(M~)                       ([4N, K])
  q, k  = RoPE_tail(QKNorm([c_qk read_{q,k} | std_{q,k}])), q / sqrt(100);  v, o_loc = c_vo read_{v,o}

Programs: axis 0 = arm group ((q, k) or (v, o)), fastest, so both groups of a run share M in L2;
axis 1 = a run of TOK consecutive tokens of one sequence. The previous token's M_new stays in
registers across the run (only the first token of a run recomputes its predecessor). Group 0
stores M_new (the carried M), which the next layer's call reads as M_old.

Layouts (row = token): M [K*V] (k major); content [N, K]; address [N, V]; up [4N, V]; gates
[4N] dynamic then [4N] static (pre-activation); std [2N, 20] (q then k); qk out [2, N, 100];
vo out [2, N, 80]. K = 80 is handled as 64 + 16 rows, the 20 standard coordinates as 2 x 10.
"""
from typing import Optional

import torch
import triton
import triton.language as tl

QK_WIDTH = 112     # stored Q/K row width (100 used): aligned rows make the attention loads vectorize
import os
READ_WARPS = int(os.environ.get("BAM_READ_WARPS", 4))
WRITE_WARPS = int(os.environ.get("BAM_WRITE_WARPS", 2))
WRITE_TW = int(os.environ.get("BAM_WRITE_TW", 1))
READ_GROUPS = int(os.environ.get("BAM_READ_GROUPS", 1))
DEC_GROUPS = int(os.environ.get("BAM_DEC_GROUPS", 1))
DEC_WARPS = int(os.environ.get("BAM_DEC_WARPS", 4))
READ_TOK = int(os.environ.get("BAM_READ_TOK", 1))
READ_SUB_DECODE = int(os.environ.get("BAM_READ_SUB", 4))
READ_SUB_PREFILL = int(os.environ.get("BAM_READ_SUB_PREFILL", 1))
FUSE_MAX_ROWS = int(os.environ.get("BAM_FUSE_MAX_ROWS", 1 << 30))


@triton.jit
def _write_pair(row, C, s_c, G, s_g, A, s_a, write_eps, RMS_C: tl.constexpr, N: tl.constexpr):
  """One gated outer-product write of a token: tiles [64, 64] and [16, 64] (fp32) of
  sum_n sigmoid(g_n) c_n (x) rms(a_n), c_n RMS-normalized over K when RMS_C."""
  n = tl.arange(0, N)
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, 16)
  v = tl.arange(0, 64)
  c0 = tl.load(C + row * s_c + n[:, None] * 80 + k0[None, :]).to(tl.float32)        # [N, 64k]
  c1 = tl.load(C + row * s_c + n[:, None] * 80 + k1[None, :]).to(tl.float32)        # [N, 16k]
  sc = tl.sigmoid(tl.load(G + row * s_g + n).to(tl.float32))
  if RMS_C:
    sc = sc * tl.rsqrt((tl.sum(c0 * c0, 1) + tl.sum(c1 * c1, 1)) / 80 + write_eps)
  a = tl.load(A + row * s_a + n[:, None] * 64 + v[None, :]).to(tl.float32)          # [N, 64v]
  a = (a * tl.rsqrt(tl.sum(a * a, 1) / 64 + write_eps)[:, None]).to(tl.bfloat16)
  c0 = (c0 * sc[:, None]).to(tl.bfloat16)
  c1 = (c1 * sc[:, None]).to(tl.bfloat16)
  return tl.dot(tl.trans(c0), a), tl.dot(tl.trans(c1), a)


@triton.jit
def _new_m(row, MOLD, s_mo, MW, C1, s_c1, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, write_eps,
           HAS_WRITE: tl.constexpr, N: tl.constexpr):
  """M_new tiles [64, 64] and [16, 64] (bf16) of one token."""
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, 16)
  v = tl.arange(0, 64)
  m0 = tl.load(MOLD + row * s_mo + k0[:, None] * 64 + v[None, :])
  m1 = tl.load(MOLD + row * s_mo + k1[:, None] * 64 + v[None, :])
  if HAS_WRITE:
    d0, d1 = _write_pair(row, C1, s_c1, G1, s_g1, A1, s_a1, write_eps, True, N)        # attention write
    e0, e1 = _write_pair(row, C2, s_c2, G2, s_g2, A2, s_a2, write_eps, False, N)       # MLP write
    m0 = (m0.to(tl.float32) + (d0 + e0)).to(tl.bfloat16)
    m1 = (m1.to(tl.float32) + (d1 + e1)).to(tl.bfloat16)
  return m0, m1


@triton.jit
def _group_key(group, row, sk, UP, s_up, GATES, s_gt, key_eps, key_scale, static_scale, c_start, N: tl.constexpr,
               CR: tl.constexpr):
  """Read keys R [CR, V] of rows c_start.. of an arm group: gated RMS-normalized key + gated static key.
  Independent of M: computed first so its loads overlap the M work."""
  v = tl.arange(0, 64)
  cg = group * 2 * N + c_start + tl.arange(0, CR)
  up = tl.load(UP + row * s_up + cg[:, None] * 64 + v[None, :]).to(tl.float32)
  gd = key_scale * tl.sigmoid(tl.load(GATES + row * s_gt + cg).to(tl.float32))
  gs = static_scale * tl.sigmoid(tl.load(GATES + row * s_gt + 4 * N + cg).to(tl.float32))
  r = up * (gd * tl.rsqrt(tl.sum(up * up, 1) / 64 + key_eps))[:, None] + gs[:, None] * sk.to(tl.float32)
  return r.to(tl.bfloat16)


@triton.jit
def _group_read(group, row, t, pos0, f0, f1, inv, r, NORMW, STD, s_std, COS, SIN,
                QK, s_qk, VO, s_vo, eps, qk_scale, vo_scale, q_mult, c_start,
                KC, skb, skh, skblk, VC, svb, svh, svs, seq, TAIL, stb, sth,
                N: tl.constexpr, QW: tl.constexpr, CR: tl.constexpr, CACHE: tl.constexpr, KTB: tl.constexpr):
  """Read of one arm group (0: q, k with QKNorm + RoPE tail; 1: v, local o) for one token, rows
  c_start .. c_start + CR of the group's 2N (arm, head) rows, with keys r from _group_key."""
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, 16)
  c = c_start + tl.arange(0, CR)
  o0 = tl.dot(r, tl.trans(f0))       # [2N, 64k]
  o1 = tl.dot(r, tl.trans(f1))       # [2N, 16k]
  if group == 0:
    o0 = o0 * (inv * qk_scale)
    o1 = o1 * (inv * qk_scale)
    j = tl.arange(0, 16)
    jm = j < 10
    sa = tl.load(STD + row * s_std + c[:, None] * 20 + j[None, :], mask=jm[None, :], other=0.).to(tl.float32)
    sb = tl.load(STD + row * s_std + c[:, None] * 20 + 10 + j[None, :], mask=jm[None, :], other=0.).to(tl.float32)
    ssq = tl.sum(o0 * o0, 1) + tl.sum(o1 * o1, 1) + tl.sum(sa * sa, 1) + tl.sum(sb * sb, 1)
    rr = tl.rsqrt(ssq / 100 + eps)
    arm = c // N
    isq = (arm == 0)[:, None]
    w0 = tl.where(isq, tl.load(NORMW + k0)[None, :], tl.load(NORMW + 100 + k0)[None, :])
    w1 = tl.where(isq, tl.load(NORMW + k1)[None, :], tl.load(NORMW + 100 + k1)[None, :])
    wa = tl.where(isq, tl.load(NORMW + 80 + j, mask=jm, other=0.)[None, :],
                  tl.load(NORMW + 180 + j, mask=jm, other=0.)[None, :])
    wb = tl.where(isq, tl.load(NORMW + 90 + j, mask=jm, other=0.)[None, :],
                  tl.load(NORMW + 190 + j, mask=jm, other=0.)[None, :])
    qm = tl.where(arm == 0, q_mult, 1.0)
    rq = rr * qm                       # q also carries 1 / sqrt(100)
    y0 = o0 * rq[:, None] * w0
    y1 = o1 * rq[:, None] * w1
    ya = sa * rq[:, None] * wa
    yb = sb * rq[:, None] * wb
    cs = tl.load(COS + (pos0 + t) * 10 + j, mask=jm, other=0.).to(tl.float32)
    sn = tl.load(SIN + (pos0 + t) * 10 + j, mask=jm, other=0.).to(tl.float32)
    ra = ya * cs[None, :] - yb * sn[None, :]
    rb = yb * cs[None, :] + ya * sn[None, :]
    base = QK + row * s_qk + c[:, None] * QW
    tl.store(base + k0[None, :], y0.to(tl.bfloat16))
    tl.store(base + k1[None, :], y1.to(tl.bfloat16))
    tl.store(base + 80 + j[None, :], ra.to(tl.bfloat16), mask=jm[None, :])
    tl.store(base + 90 + j[None, :], rb.to(tl.bfloat16), mask=jm[None, :])
    if CACHE:     # decode: the key rows go to the tail block (rows layout, slot (pos0 // KTB) % 2)
      isk = (arm == 1)[:, None]
      kb = TAIL + seq * stb + (c % N)[:, None] * sth + ((pos0 // KTB) % 2) * (KTB * 100) + (pos0 % KTB) * 100
      tl.store(kb + k0[None, :], y0.to(tl.bfloat16), mask=isk)
      tl.store(kb + k1[None, :], y1.to(tl.bfloat16), mask=isk)
      tl.store(kb + 80 + j[None, :], ra.to(tl.bfloat16), mask=isk & jm[None, :])
      tl.store(kb + 90 + j[None, :], rb.to(tl.bfloat16), mask=isk & jm[None, :])
  else:
    sc = inv * vo_scale
    base = VO + row * s_vo + c[:, None] * 80
    tl.store(base + k0[None, :], (o0 * sc).to(tl.bfloat16))
    tl.store(base + k1[None, :], (o1 * sc).to(tl.bfloat16))
    if CACHE:     # decode: the value rows also go to the value cache at position pos0
      isv = (c < N)[:, None]
      vb = VC + seq * svb + (c % N)[:, None] * svh + pos0 * svs
      tl.store(vb + k0[None, :], (o0 * sc).to(tl.bfloat16), mask=isv)
      tl.store(vb + k1[None, :], (o1 * sc).to(tl.bfloat16), mask=isv)


@triton.jit
def _read_kernel(
    M, s_m, MW, C1, s_c1, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, write_eps, MPREV, s_mp, PREV_VALID, POS,
    SHIFT, s_sh, STD, s_std, UP, s_up, GATES, s_gt, SKEY, NORMW, COS, SIN,
    QK, s_qk, VO, s_vo, KC, skb, skh, skblk, VC, svb, svh, svs, TAIL, stb, sth,
    T, ROWS, eps, key_eps, key_scale, static_scale, qk_scale, vo_scale, q_mult,
    HAS_PREV: tl.constexpr, N: tl.constexpr, QW: tl.constexpr, GROUPS: tl.constexpr, TOK: tl.constexpr,
    FUSE_WRITE: tl.constexpr, SUB: tl.constexpr, CACHE: tl.constexpr, KTB: tl.constexpr):
  """TOK tokens per program along axis 1 (sharing the static keys); axis 0 = (arm group, SUB row
  chunk) when GROUPS == 1 (both groups, SUB = 1 otherwise). Each program recomputes the read view
  (one-token decode: SUB > 1 spreads a token over more programs)."""
  CR: tl.constexpr = 2 * N // SUB
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, 16)
  v = tl.arange(0, 64)
  sub = tl.program_id(0) % SUB
  c_start = sub * CR
  c = c_start + tl.arange(0, CR)
  pos0 = tl.load(POS)
  g0 = tl.program_id(0) // SUB if GROUPS == 1 else 0
  sk_a = tl.load(SKEY + (g0 * 2 * N + c)[:, None] * 64 + v[None, :])
  if GROUPS == 2:
    sk_b = tl.load(SKEY + (2 * N + c)[:, None] * 64 + v[None, :])
  else:
    sk_b = sk_a
  for i in range(TOK):
    row = tl.program_id(1) * TOK + i
    if row < ROWS:
      seq = row // T
      t = row % T
      if GROUPS == 2:
        r_a = _group_key(0, row, sk_a, UP, s_up, GATES, s_gt, key_eps, key_scale, static_scale, c_start, N, CR)
        r_b = _group_key(1, row, sk_b, UP, s_up, GATES, s_gt, key_eps, key_scale, static_scale, c_start, N, CR)
      else:
        r_a = _group_key(g0, row, sk_a, UP, s_up, GATES, s_gt, key_eps, key_scale, static_scale, c_start, N, CR)
        r_b = r_a
      if FUSE_WRITE:     # one-token rows (decode): this layer's input M = M + the previous layer's write
        m0, m1 = _new_m(row, M, s_m, MW, C1, s_c1, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, write_eps, True,
                        N)
        if tl.program_id(0) == 0:
          tl.store(MW + row * 5120 + k0[:, None] * 64 + v[None, :], m0)
          tl.store(MW + row * 5120 + k1[:, None] * 64 + v[None, :], m1)
        f0 = m0.to(tl.float32)
        f1 = m1.to(tl.float32)
      else:
        f0 = tl.load(M + row * s_m + k0[:, None] * 64 + v[None, :]).to(tl.float32)
        f1 = tl.load(M + row * s_m + k1[:, None] * 64 + v[None, :]).to(tl.float32)
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
        p0 = tl.load(pp + k0[:, None] * 64 + v[None, :]).to(tl.float32)
        p1 = tl.load(pp + k1[:, None] * 64 + v[None, :]).to(tl.float32)
        s = tl.sigmoid(tl.load(SHIFT + row * s_sh + v).to(tl.float32))
        f0 = f0 + s[None, :] * (p0 - f0)
        f1 = f1 + s[None, :] * (p1 - f1)
      inv = tl.rsqrt((tl.sum(tl.sum(f0 * f0, 1), 0) + tl.sum(tl.sum(f1 * f1, 1), 0)) / 5120 + eps)
      f0 = f0.to(tl.bfloat16)
      f1 = f1.to(tl.bfloat16)
      if CACHE and tl.program_id(0) == 0:
        # amortized block move: at step r of block j the previous block's head r moves from its tail slot into
        # the transposed cache (done by r = N - 1 < 64, before step 0 of block j + 1 reuses the slot; until
        # then attention reads the previous block from the tail too)
        r_step = pos0 % KTB
        if pos0 >= KTB and r_step < N:
          pj = tl.arange(0, KTB)
          dd = tl.arange(0, 128)
          src = ((pos0 // KTB - 1) % 2) * (KTB * 100)
          blk = tl.load(TAIL + seq * stb + r_step * sth + src + pj[:, None] * 100 + dd[None, :], mask=dd[None, :] < 100)
          tl.store(KC + seq * skb + r_step * skh + (pos0 // KTB - 1) * skblk + dd[None, :] * KTB + pj[:, None], blk,
                   mask=dd[None, :] < 100)
      if GROUPS == 2:
        _group_read(0, row, t, pos0, f0, f1, inv, r_a, NORMW, STD, s_std, COS, SIN, QK,
                    s_qk, VO, s_vo, eps, qk_scale, vo_scale, q_mult, c_start, KC, skb, skh, skblk, VC, svb, svh, svs, seq, TAIL, stb, sth, N, QW, CR, CACHE, KTB)
        _group_read(1, row, t, pos0, f0, f1, inv, r_b, NORMW, STD, s_std, COS, SIN, QK,
                    s_qk, VO, s_vo, eps, qk_scale, vo_scale, q_mult, c_start, KC, skb, skh, skblk, VC, svb, svh, svs, seq, TAIL, stb, sth, N, QW, CR, CACHE, KTB)
      else:
        if g0 == 0:
          _group_read(0, row, t, pos0, f0, f1, inv, r_a, NORMW, STD, s_std, COS, SIN, QK,
                      s_qk, VO, s_vo, eps, qk_scale, vo_scale, q_mult, c_start, KC, skb, skh, skblk, VC, svb, svh, svs, seq, TAIL, stb, sth, N, QW, CR, CACHE, KTB)
        else:
          _group_read(1, row, t, pos0, f0, f1, inv, r_a, NORMW, STD, s_std, COS, SIN, QK,
                      s_qk, VO, s_vo, eps, qk_scale, vo_scale, q_mult, c_start, KC, skb, skh, skblk, VC, svb, svh, svs, seq, TAIL, stb, sth, N, QW, CR, CACHE, KTB)


@triton.jit
def _write_kernel(MOLD, s_mo, MW, s_mw, C1, s_c1, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, rows, write_eps,
                  N: tl.constexpr, TW: tl.constexpr):
  k0 = tl.arange(0, 64)
  k1 = 64 + tl.arange(0, 16)
  v = tl.arange(0, 64)
  for i in range(TW):
    row = tl.program_id(0) * TW + i
    if row < rows:
      m0, m1 = _new_m(row, MOLD, s_mo, MW, C1, s_c1, G1, s_g1, A1, s_a1, C2, s_c2, G2, s_g2, A2, s_a2, write_eps,
                      True, N)
      tl.store(MW + row * s_mw + k0[:, None] * 64 + v[None, :], m0)
      tl.store(MW + row * s_mw + k1[:, None] * 64 + v[None, :], m1)


@torch.library.custom_op('bam_pythia::fused_read', mutates_args=())
def fused_read(m_old: torch.Tensor, c1: torch.Tensor, g1: torch.Tensor, a1: torch.Tensor, c2: torch.Tensor,
               g2: torch.Tensor, a2: torch.Tensor, m_prev: torch.Tensor, prev_valid: torch.Tensor,
               pos: torch.Tensor, shift: torch.Tensor, std: torch.Tensor, up: torch.Tensor, gates: torch.Tensor,
               static_key: torch.Tensor, norm_w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               batch: int, has_write: bool, has_prev: bool, write_eps: float, eps: float, key_eps: float,
               key_scale: float, static_scale: float, qk_scale: float, vo_scale: float, q_mult: float,
               tok: int, k_cache: Optional[torch.Tensor] = None, v_cache: Optional[torch.Tensor] = None,
               k_tail: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Token rows r = b * T + t (all 2-D [rows, features], unit inner stride). Returns
  (m_new [rows, K*V] (empty when not has_write), qk [rows, 2, N, 112] (100 used), vo [rows, 2, N, 80]).

  k_cache [B,N,S/64,100,64] ('kt' layout), k_tail [B,N,2,64,100], v_cache [B,N,S,80] (one-token decode):
  the token's key goes to tail slot (pos // 64) % 2 (moved into k_cache one head per step during the next
  block) and its value to v_cache. Not declared as a mutation (see decode_attention.
  attend_and_append): within a decode step only this op writes a layer's caches, and the strictly causal
  attention that follows (it consumes this op's output) reads positions < pos only."""
  rows = m_old.shape[0]
  t = rows // batch
  n = static_key.shape[0] // 4
  qk = m_old.new_empty((rows, 2, n, QK_WIDTH))
  vo = m_old.new_empty((rows, 2, n, 80))
  m_new = m_old.new_empty((rows if has_write else 0, 80 * 64))
  dummy = m_old
  fuse = has_write and t == 1 and rows <= FUSE_MAX_ROWS   # decode: the write in the read kernel
  if has_write and not fuse:   # separate pass: fused with the read, the kernel needs 255 registers (low occupancy)
    tw = WRITE_TW
    _write_kernel[(triton.cdiv(rows, tw),)](
        m_old, m_old.stride(0), m_new, 80 * 64, c1, c1.stride(0), g1, g1.stride(0), a1, a1.stride(0), c2,
        c2.stride(0), g2, g2.stride(0), a2, a2.stride(0), rows, write_eps, N=n, TW=tw, num_warps=WRITE_WARPS)
    m_old = m_new
  if not fuse:
    c1 = g1 = a1 = c2 = g2 = a2 = dummy
  if not has_prev:
    m_prev = dummy
  if t > 1:
    groups, warps = READ_GROUPS, READ_WARPS
  elif 'BAM_DEC_GROUPS' in os.environ:
    groups, warps = DEC_GROUPS, DEC_WARPS
  else:   # one-token decode, A800 sweep (us per layer at B = 1 / 16 / 64: 9.1 / 10.1 / 13.5): latency-bound
    # small batches spread a token over more programs; large ones compute the write once per token
    groups, warps = (1, 4) if rows <= 16 else (2, 8)
  tok = READ_TOK
  if groups == 2:
    sub = 1
  elif t > 1:
    sub = READ_SUB_PREFILL
  elif 'BAM_READ_SUB' in os.environ:
    sub = READ_SUB_DECODE
  else:
    sub = 4 if rows <= 4 else 2
  cache = k_cache is not None
  assert not cache or t == 1
  kc = k_cache if cache else dummy
  vc = v_cache if cache else dummy
  tl_tail = k_tail if cache else dummy
  assert not cache or (k_tail.shape[-3:] == (2, kc.shape[-1], 100) and k_tail.is_contiguous())
  _read_kernel[((3 - groups) * sub, triton.cdiv(rows, tok))](
      m_old, m_old.stride(0), m_new if fuse else dummy, c1, c1.stride(0), g1, g1.stride(0), a1, a1.stride(0), c2,
      c2.stride(0), g2, g2.stride(0), a2, a2.stride(0), write_eps, m_prev, m_prev.stride(0), prev_valid, pos,
      shift, shift.stride(0), std, std.stride(0), up, up.stride(0), gates, gates.stride(0), static_key, norm_w,
      cos, sin, qk, qk.stride(0), vo, vo.stride(0),
      kc, kc.stride(0), kc.stride(1), kc.stride(2) if cache else 0, vc, vc.stride(0), vc.stride(1),
      vc.stride(2) if cache else 0, tl_tail, tl_tail.stride(0), tl_tail.stride(1),
      t, rows, eps, key_eps, key_scale, static_scale, qk_scale, vo_scale, q_mult,
      HAS_PREV=has_prev, N=n, QW=QK_WIDTH, GROUPS=groups, TOK=tok, FUSE_WRITE=fuse, SUB=sub,
      CACHE=cache, KTB=kc.shape[-1] if cache else 64, num_warps=warps)
  return m_new, qk, vo


@fused_read.register_fake
def _(m_old, c1, g1, a1, c2, g2, a2, m_prev, prev_valid, pos, shift, std, up, gates, static_key, norm_w, cos, sin,
      batch, has_write, has_prev, write_eps, eps, key_eps, key_scale, static_scale, qk_scale, vo_scale, q_mult,
      tok, k_cache=None, v_cache=None, k_tail=None):
  rows = m_old.shape[0]
  n = static_key.shape[0] // 4
  return (m_old.new_empty((rows if has_write else 0, 80 * 64)), m_old.new_empty((rows, 2, n, QK_WIDTH)),
          m_old.new_empty((rows, 2, n, 80)))


@triton.jit
def _act(x, ACT: tl.constexpr):
  if ACT == 1:      # tanh-approximate GELU (flax nn.gelu)
    inner = 0.7978845608028654 * (x + 0.044715 * x * x * x)
    e = tl.exp(-2.0 * tl.abs(inner))
    th = (1 - e) / (1 + e)
    th = tl.where(inner >= 0, th, -th)
    x = 0.5 * x * (1 + th)
  elif ACT == 2:    # exact GELU
    x = 0.5 * x * (1 + tl.erf(x * 0.7071067811865476))
  return x


@triton.jit
def _gemv_problem(pid, X, X2, KS, LNW, LNB, W, B, RES, Y, R, K, ln_eps, ACT: tl.constexpr, ACT_FROM,
                  HAS_LN: tl.constexpr, HAS_X2: tl.constexpr, HAS_B: tl.constexpr, HAS_RES: tl.constexpr,
                  BR: tl.constexpr, BK: tl.constexpr):
  """Rows pid*BR.. of y = W @ f(x) (+ b) (+ res). f: LayerNorm (input dtype rounding) when HAS_LN, then
  act on columns >= ACT_FROM. x = [X | X2] split at KS when HAS_X2 (KS a multiple of BK)."""
  r = pid * BR + tl.arange(0, BR)
  rm = r < R
  if HAS_LN:
    s1 = tl.zeros((BK,), tl.float32)
    s2 = tl.zeros((BK,), tl.float32)
    for k0 in range(0, K, BK):
      kk = k0 + tl.arange(0, BK)
      x = tl.load(X + kk, mask=kk < K, other=0.).to(tl.float32)
      s1 += x
      s2 += x * x
    mean = tl.sum(s1, 0) / K
    rstd = tl.rsqrt(tl.maximum(tl.sum(s2, 0) / K - mean * mean, 0.) + ln_eps)
  acc = tl.zeros((BR,), tl.float32)
  for k0 in range(0, K, BK):
    kk = k0 + tl.arange(0, BK)
    km = kk < K
    if HAS_X2:
      if k0 < KS:
        x = tl.load(X + kk, mask=km, other=0.).to(tl.float32)
      else:
        x = tl.load(X2 + kk - KS, mask=km, other=0.).to(tl.float32)
    else:
      x = tl.load(X + kk, mask=km, other=0.).to(tl.float32)
    if HAS_LN:
      x = ((x - mean) * rstd * tl.load(LNW + kk, mask=km, other=0.) + tl.load(LNB + kk, mask=km, other=0.))
      x = x.to(tl.bfloat16).to(tl.float32)
    if ACT != 0:
      x = tl.where(kk >= ACT_FROM, _act(x, ACT).to(tl.bfloat16).to(tl.float32), x)
    w = tl.load(W + r[:, None] * K + kk[None, :], mask=rm[:, None] & km[None, :], other=0.).to(tl.float32)
    acc += tl.sum(w * x[None, :], 1)
  if HAS_B:
    acc += tl.load(B + r, mask=rm, other=0.).to(tl.float32)
  if HAS_RES:
    acc = acc.to(Y.dtype.element_ty).to(tl.float32) + tl.load(RES + r, mask=rm, other=0.).to(tl.float32)
  tl.store(Y + r, acc.to(Y.dtype.element_ty), mask=rm)


@triton.jit
def _grouped_gemv_kernel(X1, X21, KS1, LW1, LB1, W1, B1, RS1, Y1, R1, K1, AF1, P1,
                         X2, X22, KS2, LW2, LB2, W2, B2, RS2, Y2, R2, K2, AF2, P2,
                         X3, X23, KS3, LW3, LB3, W3, B3, RS3, Y3, R3, K3, AF3, ln_eps,
                         ACT1: tl.constexpr, ACT2: tl.constexpr, ACT3: tl.constexpr,
                         F1: tl.constexpr, F2: tl.constexpr, F3: tl.constexpr,
                         NP: tl.constexpr, BR: tl.constexpr, BK: tl.constexpr):
  """Up to three one-token linears in one launch; Fi = (has_ln, has_x2, has_b, has_res) bits 1, 2, 4, 8."""
  pid = tl.program_id(0)
  if pid < P1:
    _gemv_problem(pid, X1, X21, KS1, LW1, LB1, W1, B1, RS1, Y1, R1, K1, ln_eps, ACT1, AF1, (F1 & 1) != 0,
                  (F1 & 2) != 0, (F1 & 4) != 0, (F1 & 8) != 0, BR, BK)
  elif NP == 1:
    pass
  elif pid < P1 + P2:
    _gemv_problem(pid - P1, X2, X22, KS2, LW2, LB2, W2, B2, RS2, Y2, R2, K2, ln_eps, ACT2, AF2, (F2 & 1) != 0,
                  (F2 & 2) != 0, (F2 & 4) != 0, (F2 & 8) != 0, BR, BK)
  elif NP == 3:
    _gemv_problem(pid - P1 - P2, X3, X23, KS3, LW3, LB3, W3, B3, RS3, Y3, R3, K3, ln_eps, ACT3, AF3,
                  (F3 & 1) != 0, (F3 & 2) != 0, (F3 & 4) != 0, (F3 & 8) != 0, BR, BK)




@torch.library.custom_op('bam_pythia::gemv', mutates_args=())
def _gemv_op(xs: list[torch.Tensor], ws: list[torch.Tensor], bs: list[Optional[torch.Tensor]], acts: list[int],
             br: int, bk: int, x2s: list[Optional[torch.Tensor]], ln_ws: list[Optional[torch.Tensor]],
             ln_bs: list[Optional[torch.Tensor]], ress: list[Optional[torch.Tensor]], act_froms: list[int],
             ln_eps: float, warps: int) -> list[torch.Tensor]:
  """One-token linears y_i = W_i f_i(x_i) + b_i (+ res_i), 1 to 3 problems in one launch (see `gemv`).

  x_i [1, K_i] (or [x_i | x2_i] along K), W_i [R_i, K_i] contiguous. f_i: LayerNorm(ln_w_i, ln_b_i) if
  given, then act_i (0 none, 1 tanh GELU, 2 exact GELU) on columns >= act_from_i. Intermediate values
  are rounded to BF16 as when they are materialized."""
  n = len(xs)
  ys = [x.new_empty((1, w.shape[0])) for x, w in zip(xs, ws)]
  args, flags, ps = [], [], []
  for i in range(3):
    j = min(i, n - 1)
    x, w, y = xs[j], ws[j], ys[j]
    x2, lw, lb, b, res = x2s[j], ln_ws[j], ln_bs[j], bs[j], ress[j]
    p = triton.cdiv(w.shape[0], br) if i < n else 0
    ks = x.shape[-1] if x2 is not None else 0
    assert x2 is None or ks % bk == 0
    flags.append((lw is not None) | (x2 is not None) << 1 | (b is not None) << 2 | (res is not None) << 3)
    args += [x, x2 if x2 is not None else x, ks, lw if lw is not None else w, lb if lb is not None else w, w,
             b if b is not None else w, res if res is not None else w, y, w.shape[0], w.shape[1], act_froms[j]]
    if i < 2:
      args.append(p)
    ps.append(p)
  _grouped_gemv_kernel[(sum(ps),)](*args, ln_eps, ACT1=acts[0], ACT2=acts[min(1, n - 1)], ACT3=acts[min(2, n - 1)],
                                   F1=flags[0], F2=flags[1], F3=flags[2], NP=n, BR=br, BK=bk, num_warps=warps,
                                   num_stages=2)
  return ys


@_gemv_op.register_fake
def _(xs, ws, bs, acts, br, bk, x2s, ln_ws, ln_bs, ress, act_froms, ln_eps, warps):
  return [x.new_empty((1, w.shape[0])) for x, w in zip(xs, ws)]


def gemv(xs, ws, bs, acts, br=16, bk=256, warps=4, x2s=None, ln_ws=None, ln_bs=None, ress=None, act_froms=None,
         ln_eps=1e-5):
  """Block sizes (br, bk, warps) from an A800 sweep per problem shape (see fast_inference)."""
  none = [None] * len(xs)
  return _gemv_op(xs, ws, bs, acts, br, bk, x2s or none, ln_ws or none, ln_bs or none, ress or none,
                  act_froms or [0] * len(xs), ln_eps, warps)




@triton.jit
def _gemm_problem(pid, X, s_x, W, B, RES, s_r, Y, s_y, M, R, K, ACT: tl.constexpr, HAS_B: tl.constexpr,
                  HAS_RES: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
  """Columns pid*BN.. of y [M, R] = act(x) [M, K] @ W^T (+ b) (+ res), all M <= BM rows in one tile."""
  m = tl.arange(0, BM)
  mm = m < M
  r = pid * BN + tl.arange(0, BN)
  rm = r < R
  acc = tl.zeros((BM, BN), tl.float32)
  for k0 in range(0, K, BK):
    kk = k0 + tl.arange(0, BK)
    km = kk < K
    x = tl.load(X + m[:, None] * s_x + kk[None, :], mask=mm[:, None] & km[None, :], other=0.)
    if ACT != 0:
      x = _act(x.to(tl.float32), ACT).to(tl.bfloat16)
    w = tl.load(W + r[:, None] * K + kk[None, :], mask=rm[:, None] & km[None, :], other=0.)
    acc = tl.dot(x, tl.trans(w), acc)
  if HAS_B:
    acc += tl.load(B + r, mask=rm, other=0.).to(tl.float32)[None, :]
  if HAS_RES:
    acc = acc.to(Y.dtype.element_ty).to(tl.float32) + tl.load(RES + m[:, None] * s_r + r[None, :],
                                                               mask=mm[:, None] & rm[None, :], other=0.).to(tl.float32)
  tl.store(Y + m[:, None] * s_y + r[None, :], acc.to(Y.dtype.element_ty), mask=mm[:, None] & rm[None, :])


@triton.jit
def _grouped_gemm_kernel(X1, sx1, W1, B1, RS1, sr1, Y1, R1, K1, P1, X2, sx2, W2, B2, RS2, sr2, Y2, R2, K2, P2,
                         X3, sx3, W3, B3, RS3, sr3, Y3, R3, K3, M,
                         ACT1: tl.constexpr, ACT2: tl.constexpr, ACT3: tl.constexpr, F1: tl.constexpr,
                         F2: tl.constexpr, F3: tl.constexpr, NP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                         BK: tl.constexpr):
  """Up to three small-M linears in one launch; Fi bits: 1 has_b, 2 has_res. Outputs are contiguous."""
  pid = tl.program_id(0)
  if pid < P1:
    _gemm_problem(pid, X1, sx1, W1, B1, RS1, sr1, Y1, R1, M, R1, K1, ACT1, (F1 & 1) != 0, (F1 & 2) != 0, BM, BN, BK)
  elif NP == 1:
    pass
  elif pid < P1 + P2:
    _gemm_problem(pid - P1, X2, sx2, W2, B2, RS2, sr2, Y2, R2, M, R2, K2, ACT2, (F2 & 1) != 0, (F2 & 2) != 0, BM,
                  BN, BK)
  elif NP == 3:
    _gemm_problem(pid - P1 - P2, X3, sx3, W3, B3, RS3, sr3, Y3, R3, M, R3, K3, ACT3, (F3 & 1) != 0, (F3 & 2) != 0,
                  BM, BN, BK)


@torch.library.custom_op('bam_pythia::grouped_gemm', mutates_args=())
def _grouped_gemm_op(xs: list[torch.Tensor], ws: list[torch.Tensor], bs: list[Optional[torch.Tensor]],
                     acts: list[int], ress: list[Optional[torch.Tensor]], bn: int, bk: int, warps: int,
                     stages: int) -> list[torch.Tensor]:
  m = xs[0].shape[0]
  n = len(xs)
  ys = [x.new_empty((m, w.shape[0])) for x, w in zip(xs, ws)]
  args, flags, ps = [], [], []
  for i in range(3):
    j = min(i, n - 1)
    x, w, b, res, y = xs[j], ws[j], bs[j], ress[j], ys[j]
    assert x.stride(-1) == 1 and w.is_contiguous()
    p = triton.cdiv(w.shape[0], bn) if i < n else 0
    flags.append((b is not None) | (res is not None) << 1)
    args += [x, x.stride(0), w, b if b is not None else w, res if res is not None else w,
             res.stride(0) if res is not None else 0, y, w.shape[0], w.shape[1]] + ([p] if i < 2 else [])
    ps.append(p)
  bm = max(16, triton.next_power_of_2(m))
  _grouped_gemm_kernel[(sum(ps),)](*args, m, ACT1=acts[0], ACT2=acts[min(1, n - 1)], ACT3=acts[min(2, n - 1)],
                                   F1=flags[0], F2=flags[1], F3=flags[2], NP=n, BM=bm, BN=bn, BK=bk,
                                   num_warps=warps, num_stages=stages)
  return ys


@_grouped_gemm_op.register_fake
def _(xs, ws, bs, acts, ress, bn, bk, warps, stages):
  return [x.new_empty((x.shape[0], w.shape[0])) for x, w in zip(xs, ws)]


def grouped_gemm(xs, ws, bs, acts, ress=None, bn=64, bk=64, warps=4, stages=3):
  """Small-M (<= 64 rows) linears y_i = act_i(x_i) @ W_i^T + b_i (+ res_i) in one launch (1 to 3 problems)."""
  return _grouped_gemm_op(xs, ws, bs, acts, ress or [None] * len(xs), bn, bk, warps, stages)
