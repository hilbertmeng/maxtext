"""Fused Triton kernels for BAM's per-token matrix reads and writes (the GPU analogue of the v8t core).

read:  Y_t[a] = X_t[a] M_t^T for the four arms a = q, k, v, o, where
       X_t[a][n, v] = 0.1 sigmoid(g_dyn[a, n]) rms(key_a[n, :])[v] + 1.1 sigmoid(g_stat[a, n]) / 0.99 S_a[v, n]
       (q, k use their own keys; v and o share the VO key). One program per token: M_t, raw keys and gate
       logits are read once, X never leaves registers, the contraction runs on tensor cores (tl.dot).
write: M'_t = M_t + sum_g C_g^T A_g with C_g = sigmoid(gate_g) rms(content_g) [N, K], A_g = rms(address_g) [N, V]
       for one or two groups (attention write, plus the MLP write on its layers).
Inputs may be strided views (e.g. column slices of a fused projection output). Accumulation is FP32.
"""
import math
from typing import Tuple

import torch
import triton
import triton.language as tl


DECODE_BC, DECODE_WARPS = 256, 8      # A800 sweep at T=1: 5.1 us (no pending write), 7.8 us (one group)


def _prec(m):
  return 'ieee' if m.dtype == torch.float32 else 'tf32'     # tf32 is a no-op for bf16 operands


@triton.jit
def _store_rows(y, a, n, k, t, Y, s_y_t, STD, POS, s_st, s_sn, s_sarm, nm, km, N: tl.constexpr, K: tl.constexpr,
                QCOLS: tl.constexpr, R: tl.constexpr, RB: tl.constexpr, QSCALE: tl.constexpr,
                LOG_MAX_TS: tl.constexpr):
  """Store rows (arm a, head n) of token t. Q/K arms (a < 2) keep the M-read columns < QCOLS and take the
  rotated standard coordinates in columns >= QCOLS (half-split RoPE over R dims, timescale max_ts ** (2i/R));
  Q is scaled by QSCALE. The tail is computed on a narrow [rows, RB] tile and stored separately."""
  scale = tl.where(a == 0, QSCALE, 1.0)
  qk = a < 2
  row = Y + t * s_y_t + a * N * K + n * K
  keep = nm[:, None] & km[None, :] & ~(qk[:, None] & (k >= QCOLS)[None, :])
  tl.store(row[:, None] + k[None, :], (y * scale[:, None]).to(Y.dtype.element_ty), mask=keep)
  j = tl.arange(0, RB)
  half = R // 2
  i = tl.where(j < half, j, j - half)
  partner = tl.where(j < half, j + half, j - half)
  tail = (qk & nm)[:, None] & (j < R)[None, :]
  base = STD + t * s_st + tl.minimum(a, 1)[:, None] * s_sarm + n[:, None] * s_sn
  s1 = tl.load(base + j[None, :], mask=tail, other=0.).to(tl.float32)
  s2 = tl.load(base + partner[None, :], mask=tail, other=0.).to(tl.float32)
  pos = tl.load(POS + t).to(tl.float32)
  ang = pos / tl.exp(LOG_MAX_TS * 2.0 * i.to(tl.float32) / R)
  rot = s1 * tl.cos(ang)[None, :] + tl.where(j < half, -1.0, 1.0)[None, :] * s2 * tl.sin(ang)[None, :]
  tl.store(row[:, None] + QCOLS + j[None, :], (rot * scale[:, None]).to(Y.dtype.element_ty), mask=tail)


@triton.jit
def _group(C, CL, G, A, t, s_ct, s_cn, s_clt, s_cln, s_gt, s_at, s_an, n, k, kb, v, nm, km, kbm, vm,
           K: tl.constexpr, V: tl.constexpr, EPS: tl.constexpr, HAS_LOCAL: tl.constexpr, SPLITK: tl.constexpr):
  """Gated RMS content rows restricted to the k block kb (statistics over all K) and RMS address rows.
  Without SPLITK the block is all of K and the content is loaded once."""
  c = tl.load(C + t * s_ct + n[:, None] * s_cn + k[None, :], mask=nm[:, None] & km[None, :], other=0.).to(tl.float32)
  if HAS_LOCAL:      # content = attention output + local O (added here instead of in a separate kernel)
    c += tl.load(CL + t * s_clt + n[:, None] * s_cln + k[None, :], mask=nm[:, None] & km[None, :], other=0.).to(tl.float32)
  if SPLITK:
    cb = tl.load(C + t * s_ct + n[:, None] * s_cn + kb[None, :], mask=nm[:, None] & kbm[None, :], other=0.).to(tl.float32)
    if HAS_LOCAL:
      cb += tl.load(CL + t * s_clt + n[:, None] * s_cln + kb[None, :], mask=nm[:, None] & kbm[None, :],
                    other=0.).to(tl.float32)
  else:
    cb = c
  g = tl.load(G + t * s_gt + n, mask=nm, other=0.).to(tl.float32)
  cb = cb * (tl.sigmoid(g) * tl.rsqrt(tl.sum(c * c, axis=1) / K + EPS))[:, None]
  a = tl.load(A + t * s_at + n[:, None] * s_an + v[None, :], mask=nm[:, None] & vm[None, :], other=0.).to(tl.float32)
  a = a * tl.rsqrt(tl.sum(a * a, axis=1) / V + EPS)[:, None]
  return cb, a


@triton.jit
def _load_m(M, MOUT, t, store, s_mt, C1, CL1, G1, A1, C2, G2, A2, s_c1t, s_c1n, s_cl1t, s_cl1n, s_g1t, s_a1t,
            s_a1n, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n, k, v, km, vm, N: tl.constexpr, K: tl.constexpr,
            V: tl.constexpr, BN: tl.constexpr, EPS: tl.constexpr, PENDING: tl.constexpr, HAS_LOCAL: tl.constexpr,
            PREC: tl.constexpr):
  """M_t [BK, BV] for the read. With PENDING groups (the previous layer's deferred write) the update
  M + sum_g C_g^T A_g is applied in registers and, if `store`, written to MOUT (the layer's output M)."""
  mask = km[:, None] & vm[None, :]
  m = tl.load(M + t * s_mt + k[:, None] * V + v[None, :], mask=mask, other=0.)
  if PENDING > 0:
    dt = m.dtype
    n = tl.arange(0, BN)
    nm = n < N
    c, a = _group(C1, CL1, G1, A1, t, s_c1t, s_c1n, s_cl1t, s_cl1n, s_g1t, s_a1t, s_a1n, n, k, k, v, nm, km, km, vm,
                  K, V, EPS, HAS_LOCAL, False)
    acc = tl.dot(tl.trans(c.to(dt)), a.to(dt), m.to(tl.float32), input_precision=PREC)
    if PENDING > 1:
      c, a = _group(C2, C2, G2, A2, t, s_c2t, s_c2n, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n, n, k, k, v, nm, km, km, vm,
                    K, V, EPS, False, False)
      acc = tl.dot(tl.trans(c.to(dt)), a.to(dt), acc, input_precision=PREC)
    m = acc.to(dt)
    if store:
      tl.store(MOUT + t * s_mt + k[:, None] * V + v[None, :], m, mask=mask)
  return m


@triton.jit
def _read_kernel(M, KEYS, GATES, S, Y, T, STD, POS, MOUT, C1, CL1, G1, A1, C2, G2, A2,
                 s_mt, s_kt, s_karm, s_kn, s_gt, s_y_t, s_st, s_sn, s_sarm,
                 s_c1t, s_c1n, s_cl1t, s_cl1n, s_g1t, s_a1t, s_a1n, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n,
                 N: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                 EPS: tl.constexpr, SCALE: tl.constexpr, STATIC_DIV: tl.constexpr, PREC: tl.constexpr,
                 QCOLS: tl.constexpr, R: tl.constexpr, RB: tl.constexpr, QSCALE: tl.constexpr,
                 LOG_MAX_TS: tl.constexpr, ROPE: tl.constexpr, TB: tl.constexpr, STAGES: tl.constexpr,
                 PENDING: tl.constexpr, HAS_LOCAL: tl.constexpr, PEPS: tl.constexpr):
  """Program (token block, arm): TB tokens, software-pipelined; one [BN,BV]x[BV,BK] tensor-core
  product per token. M_t is shared by the four arm programs through L2."""
  a = tl.program_id(0)                            # arm fastest: the 4 programs of a token block run together (L2)
  blk = tl.program_id(1)
  n = tl.arange(0, BN)
  k = tl.arange(0, BK)
  v = tl.arange(0, BV)
  nm, km, vm = n < N, k < K, v < V
  arm = tl.minimum(a, 2)                          # v and o share the VO key
  st = tl.load(S + a * V * N + v[None, :] * N + n[:, None], mask=nm[:, None] & vm[None, :], other=0.).to(tl.float32)
  for t in tl.range(blk * TB, tl.minimum(blk * TB + TB, T), num_stages=STAGES):
    mkv = _load_m(M, MOUT, t, a == 0, s_mt, C1, CL1, G1, A1, C2, G2, A2, s_c1t, s_c1n, s_cl1t, s_cl1n, s_g1t,
                s_a1t, s_a1n, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n, k, v, km, vm, N, K, V, BN, PEPS, PENDING,
                HAS_LOCAL, PREC)
    key = tl.load(KEYS + t * s_kt + arm * s_karm + n[:, None] * s_kn + v[None, :],
                  mask=nm[:, None] & vm[None, :], other=0.).to(tl.float32)
    gd = tl.load(GATES + t * s_gt + a * N + n, mask=nm, other=0.).to(tl.float32)
    gs = tl.load(GATES + t * s_gt + (4 + a) * N + n, mask=nm, other=0.).to(tl.float32)
    inv = tl.rsqrt(tl.sum(key * key, axis=1) / V + EPS)
    x = (SCALE * tl.sigmoid(gd) * inv)[:, None] * key + (1.1 * tl.sigmoid(gs) / STATIC_DIV)[:, None] * st
    y = tl.dot(x.to(mkv.dtype), tl.trans(mkv), input_precision=PREC)     # [BN, BK] fp32
    if ROPE:
      _store_rows(y, tl.zeros_like(n) + a, n, k, t, Y, s_y_t, STD, POS, s_st, s_sn, s_sarm, nm, km, N, K,
                  QCOLS, R, RB, QSCALE, LOG_MAX_TS)
    else:
      tl.store(Y + t * s_y_t + a * N * K + n[:, None] * K + k[None, :], y.to(Y.dtype.element_ty),
               mask=nm[:, None] & km[None, :])


@triton.jit
def _read_kernel_stacked(M, KEYS, GATES, S, Y, T, STD, POS, MOUT, C1, CL1, G1, A1, C2, G2, A2,
                         s_mt, s_kt, s_karm, s_kn, s_gt, s_y_t, s_st, s_sn, s_sarm,
                         s_c1t, s_c1n, s_cl1t, s_cl1n, s_g1t, s_a1t, s_a1n, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n,
                         N: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                         BN: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                         EPS: tl.constexpr, SCALE: tl.constexpr, STATIC_DIV: tl.constexpr, PREC: tl.constexpr,
                         QCOLS: tl.constexpr, R: tl.constexpr, RB: tl.constexpr, QSCALE: tl.constexpr,
                         LOG_MAX_TS: tl.constexpr, ROPE: tl.constexpr, TB: tl.constexpr, STAGES: tl.constexpr,
                         PENDING: tl.constexpr, HAS_LOCAL: tl.constexpr, PEPS: tl.constexpr):
  """Program = TB tokens; per token one [4*BN, BV] x [BV, BK] product with the four arms stacked as rows
  (row r: arm r // BN, head r % BN), so M_t is read once and X is gathered in one 2-D load."""
  r = tl.arange(0, 4 * BN)
  a = r // BN
  n = r % BN
  k = tl.arange(0, BK)
  v = tl.arange(0, BV)
  nm, km, vm = n < N, k < K, v < V
  arm = tl.minimum(a, 2)
  st = tl.load(S + a[:, None] * V * N + v[None, :] * N + n[:, None], mask=nm[:, None] & vm[None, :], other=0.).to(tl.float32)
  for t in tl.range(tl.program_id(0) * TB, tl.minimum(tl.program_id(0) * TB + TB, T), num_stages=STAGES):
    mkv = _load_m(M, MOUT, t, True, s_mt, C1, CL1, G1, A1, C2, G2, A2, s_c1t, s_c1n, s_cl1t, s_cl1n, s_g1t,
                s_a1t, s_a1n, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n, k, v, km, vm, N, K, V, BN, PEPS, PENDING,
                HAS_LOCAL, PREC)
    key = tl.load(KEYS + t * s_kt + arm[:, None] * s_karm + n[:, None] * s_kn + v[None, :],
                  mask=nm[:, None] & vm[None, :], other=0.).to(tl.float32)
    gd = tl.load(GATES + t * s_gt + a * N + n, mask=nm, other=0.).to(tl.float32)
    gs = tl.load(GATES + t * s_gt + (4 + a) * N + n, mask=nm, other=0.).to(tl.float32)
    inv = tl.rsqrt(tl.sum(key * key, axis=1) / V + EPS)
    x = (SCALE * tl.sigmoid(gd) * inv)[:, None] * key + (1.1 * tl.sigmoid(gs) / STATIC_DIV)[:, None] * st
    y = tl.dot(x.to(mkv.dtype), tl.trans(mkv), input_precision=PREC)
    if ROPE:
      _store_rows(y, a, n, k, t, Y, s_y_t, STD, POS, s_st, s_sn, s_sarm, nm, km, N, K, QCOLS, R, RB, QSCALE,
                  LOG_MAX_TS)
    else:      # q/k columns >= QCOLS are left for bam_rope_tail
      tl.store(Y + t * s_y_t + a[:, None] * N * K + n[:, None] * K + k[None, :], y.to(Y.dtype.element_ty),
               mask=nm[:, None] & km[None, :] & ~((a < 2)[:, None] & (k >= QCOLS)[None, :]))


@triton.jit
def _read_kernel_khalf(M, KEYS, GATES, S, Y, T, s_mt, s_kt, s_karm, s_kn, s_gt, s_y_t,
                       N: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BN: tl.constexpr, KH: tl.constexpr,
                       NKH: tl.constexpr, BV: tl.constexpr, QCOLS: tl.constexpr, EPS: tl.constexpr,
                       SCALE: tl.constexpr, STATIC_DIV: tl.constexpr, PREC: tl.constexpr, TB: tl.constexpr,
                       STAGES: tl.constexpr, ARMS: tl.constexpr):
  """Prefill read (no deferred write, RoPE applied afterwards). Program (arm group, token block), the arm
  group fastest so a token's programs share M_t through L2; each builds X rows [ARMS*BN, BV] once and
  produces the output columns in NKH chunks of KH. The single-program [4*BN, 128] FP32 accumulator needed
  255 registers (spills, one program per SM). A800, T=4096: 146 us (stacked) -> 123 (K halves) -> 110 us
  (two arms per program, one warp)."""
  r = tl.arange(0, ARMS * BN)
  a = tl.program_id(0) * ARMS + r // BN
  n = r % BN
  v = tl.arange(0, BV)
  nm, vm = n < N, v < V
  arm = tl.minimum(a, 2)
  for t in tl.range(tl.program_id(1) * TB, tl.minimum(tl.program_id(1) * TB + TB, T), num_stages=STAGES):
    key = tl.load(KEYS + t * s_kt + arm[:, None] * s_karm + n[:, None] * s_kn + v[None, :],
                  mask=nm[:, None] & vm[None, :], other=0.).to(tl.float32)
    gd = tl.load(GATES + t * s_gt + a * N + n, mask=nm, other=0.).to(tl.float32)
    gs = tl.load(GATES + t * s_gt + (4 + a) * N + n, mask=nm, other=0.).to(tl.float32)
    st = tl.load(S + a[:, None] * V * N + v[None, :] * N + n[:, None], mask=nm[:, None] & vm[None, :],
                 other=0.).to(tl.float32)
    inv = tl.rsqrt(tl.sum(key * key, axis=1) / V + EPS)
    x = (SCALE * tl.sigmoid(gd) * inv)[:, None] * key + (1.1 * tl.sigmoid(gs) / STATIC_DIV)[:, None] * st
    for hk in tl.static_range(NKH):
      k = hk * KH + tl.arange(0, KH)
      km = k < K
      mkv = tl.load(M + t * s_mt + k[:, None] * V + v[None, :], mask=km[:, None] & vm[None, :], other=0.)
      y = tl.dot(x.to(mkv.dtype), tl.trans(mkv), input_precision=PREC)
      keep = nm[:, None] & km[None, :] & ~((a < 2)[:, None] & (k >= QCOLS)[None, :])   # q/k tails: RoPE later
      tl.store(Y + t * s_y_t + a[:, None] * N * K + n[:, None] * K + k[None, :], y.to(Y.dtype.element_ty),
               mask=keep)


def _pending(m, c1, l1, g1, a1, c2, g2, a2, pending):
  """Output M buffer and pointer/stride arguments of the deferred write (dummies when pending == 0)."""
  used = ((c1, l1, g1, a1) if pending else ()) + ((c2, g2, a2) if pending > 1 else ())
  for x in used:
    assert x.stride(-1) == 1
  m_out = torch.empty_like(m) if pending else m.new_empty((0,))
  ptrs = (m_out if pending else m, c1, l1, g1, a1, c2, g2, a2)
  strides = (c1.stride(0), c1.stride(1), l1.stride(0), l1.stride(1), g1.stride(0), a1.stride(0), a1.stride(1),
             c2.stride(0), c2.stride(1), g2.stride(0), a2.stride(0), a2.stride(1))
  return m_out, ptrs, strides


@torch.library.custom_op('bam::read', mutates_args=())
def bam_read(m: torch.Tensor, keys: torch.Tensor, gates: torch.Tensor, static_keys: torch.Tensor,
             std: torch.Tensor, positions: torch.Tensor, eps: float, scale: float, static_div: float,
             qk_cols: int, q_scale: float, max_timescale: float, rope: bool,
             c1: torch.Tensor, l1: torch.Tensor, g1: torch.Tensor, a1: torch.Tensor, c2: torch.Tensor,
             g2: torch.Tensor, a2: torch.Tensor, pending: int, has_local: bool,
             write_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
  """m [T,K,V] contiguous; keys [T,3,N,V] (q, k, vo; any token/arm/head strides, unit V stride);
  gates [T,8,N] logits (dyn q,k,v,o then static q,k,v,o; unit stride within the 8N block);
  static_keys [4,V,N] contiguous; std [T,2,N,R] unrotated standard Q/K coordinates (unit R stride);
  positions [T] int. With `rope`, returns attention-ready Y [T,4,N,K]: q (scaled by q_scale), k, v, local_o
  with q/k columns >= qk_cols holding the rotated standard coordinates (fewer kernels when decoding);
  without, Y is the raw read (q/k columns >= qk_cols are unused) and the caller applies RoPE.
  pending (0, 1, 2) deferred-write groups of the previous layer (content c1 (+ l1 if has_local), gate logits
  g1, raw address a1; second group c2, g2, a2) are applied to M first; returns (Y, M') with M' empty if none."""
  t, kdim, vdim = m.shape
  n = keys.shape[2]
  assert m.is_contiguous() and static_keys.is_contiguous() and keys.stride(-1) == 1
  assert gates.stride(-1) == 1 and gates.stride(-2) == n
  y = torch.empty((t, 4, n, kdim), device=m.device, dtype=m.dtype)
  m_out, ptrs, strides = _pending(m, c1, l1, g1, a1, c2, g2, a2, pending)
  if not t:
    return y, m_out
  assert std.stride(-1) == 1 and positions.is_contiguous()
  args = (m, keys, gates, static_keys, y, t, std, positions, *ptrs, m.stride(0), keys.stride(0), keys.stride(1),
          keys.stride(2), gates.stride(0), y.stride(0), std.stride(0), std.stride(2), std.stride(1), *strides)
  meta = dict(N=n, K=kdim, V=vdim, BN=max(16, triton.next_power_of_2(n)), BK=triton.next_power_of_2(kdim),
              BV=max(16, triton.next_power_of_2(vdim)), EPS=eps, SCALE=scale, STATIC_DIV=static_div, PREC=_prec(m),
              QCOLS=qk_cols, R=kdim - qk_cols, RB=triton.next_power_of_2(kdim - qk_cols), QSCALE=q_scale, LOG_MAX_TS=math.log(max_timescale), ROPE=rope,
              PENDING=pending, HAS_LOCAL=has_local, PEPS=write_eps)
  if t < 64:     # decode: one program per (arm, token) for latency (A800: 3.7 us at T=1)
    _read_kernel[(4, t)](*args, **meta, TB=1, STAGES=1, num_warps=4)
  elif not rope and not pending:     # prefill path
    kh, arms = 64, 2
    _read_kernel_khalf[(4 // arms, t)](m, keys, gates, static_keys, y, t, m.stride(0), keys.stride(0), keys.stride(1),
                             keys.stride(2), gates.stride(0), y.stride(0), N=n, K=kdim, V=vdim,
                             BN=max(16, triton.next_power_of_2(n)), KH=kh, NKH=triton.cdiv(kdim, kh),
                             BV=max(16, triton.next_power_of_2(vdim)), QCOLS=qk_cols, EPS=eps, SCALE=scale,
                             STATIC_DIV=static_div, PREC=_prec(m), TB=1, STAGES=1, ARMS=arms, num_warps=1)
  else:          # four arms stacked into one product per token, M_t read once
    for stages in ((2, 1) if rope else (3, 2, 1)):   # shared-memory budget differs (A800 164 KB, RTX 6000D 99 KB)
      try:
        _read_kernel_stacked[(triton.cdiv(t, 4),)](*args, **meta, TB=4, STAGES=stages, num_warps=4)
        break
      except triton.runtime.errors.OutOfResources:
        if stages == 1:
          raise
  return y, m_out


@bam_read.register_fake
def _(m, keys, gates, static_keys, std, positions, eps, scale, static_div, qk_cols, q_scale, max_timescale, rope,
      c1, l1, g1, a1, c2, g2, a2, pending, has_local, write_eps):
  return (m.new_empty((m.shape[0], 4, keys.shape[2], m.shape[1])),
          torch.empty_like(m) if pending else m.new_empty((0,)))


@triton.jit
def _gelu_tanh(h):
  z = 0.7978845608028654 * (h + 0.044715 * h * h * h)
  return 0.5 * h * (1.0 + (2.0 * tl.sigmoid(2.0 * z) - 1.0))


@triton.jit
def _decode_read_kernel(HID, W, B, M, GATES, S, STD, POS, Y, MOUT, C1, CL1, G1, A1, C2, G2, A2,
                        WHID, WW, WB, ADDR,
                        s_ht, s_mt, s_gt, s_st, s_sn, s_sarm, s_pb, s_pt, s_y_t, TLEN,
                        s_wht, s_adt,
                        s_c1t, s_c1n, s_cl1t, s_cl1n, s_g1t, s_a1t, s_a1n, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n,
                        N: tl.constexpr, K: tl.constexpr, V: tl.constexpr, H: tl.constexpr,
                        BK: tl.constexpr, BV: tl.constexpr, BH: tl.constexpr, BC: tl.constexpr,
                        EPS: tl.constexpr, SCALE: tl.constexpr, STATIC_DIV: tl.constexpr,
                        QCOLS: tl.constexpr, R: tl.constexpr, RB: tl.constexpr, QSCALE: tl.constexpr,
                        LOG_MAX_TS: tl.constexpr, BN: tl.constexpr, PENDING: tl.constexpr,
                        HAS_LOCAL: tl.constexpr, PEPS: tl.constexpr, PREC: tl.constexpr,
                        WH: tl.constexpr, BWH: tl.constexpr):
  """Decode read, program (head n, key arm ka, token t): gelu(hidden) -> this head's 40 key rows of the
  fused up projection (+ pre-bias) -> RMS, gates, static key -> rows of X_t M_t^T (VO key: v and o rows)
  -> RoPE tail and q scale; the previous layer's deferred write is applied to M first. Programs with
  ka == 3 instead compute head n's raw write address W_addr gelu(write hidden) + b of this layer.
  Replaces the GELU, key/address up-projection GEMV and read kernels of a decode step."""
  n = tl.program_id(0)
  ka = tl.program_id(1)
  t = tl.program_id(2)
  v = tl.arange(0, BV)
  vm = v < V
  if ka == 3:
    wa = tl.zeros((BV,), tl.float32)
    for c0 in tl.static_range(0, BWH, BC):
      c = c0 + tl.arange(0, BC)
      cm = c < WH
      h = _gelu_tanh(tl.load(WHID + t * s_wht + c, mask=cm, other=0.).to(tl.float32))
      w = tl.load(WW + (n * V + v[:, None]) * WH + c[None, :], mask=vm[:, None] & cm[None, :], other=0.)
      wa += tl.sum(w.to(tl.float32) * h[None, :], axis=1)
    wa += tl.load(WB + n * V + v, mask=vm, other=0.).to(tl.float32)
    tl.store(ADDR + t * s_adt + n * V + v, wa.to(ADDR.dtype.element_ty), mask=vm)
  else:
    k = tl.arange(0, BK)
    km = k < K
    # Issue every load that does not depend on the key GEMV first (M and the deferred-write factors), so
    # their latency overlaps the GEMV instead of following it.
    mraw = tl.load(M + t * s_mt + k[:, None] * V + v[None, :], mask=km[:, None] & vm[None, :], other=0.)
    if PENDING > 0:
      hn = tl.arange(0, BN)
      hm = hn < N
      cmask = hm[:, None] & km[None, :]
      amask = hm[:, None] & vm[None, :]
      rc1 = tl.load(C1 + t * s_c1t + hn[:, None] * s_c1n + k[None, :], mask=cmask, other=0.)
      rl1 = tl.load(CL1 + t * s_cl1t + hn[:, None] * s_cl1n + k[None, :], mask=cmask & HAS_LOCAL, other=0.)
      rg1 = tl.load(G1 + t * s_g1t + hn, mask=hm, other=0.)
      ra1 = tl.load(A1 + t * s_a1t + hn[:, None] * s_a1n + v[None, :], mask=amask, other=0.)
      if PENDING > 1:
        rc2 = tl.load(C2 + t * s_c2t + hn[:, None] * s_c2n + k[None, :], mask=cmask, other=0.)
        rg2 = tl.load(G2 + t * s_g2t + hn, mask=hm, other=0.)
        ra2 = tl.load(A2 + t * s_a2t + hn[:, None] * s_a2n + v[None, :], mask=amask, other=0.)
    key = tl.zeros((BV,), tl.float32)
    for c0 in tl.static_range(0, BH, BC):
      c = c0 + tl.arange(0, BC)
      cm = c < H
      h = _gelu_tanh(tl.load(HID + t * s_ht + c, mask=cm, other=0.).to(tl.float32))
      w = tl.load(W + ((ka * N + n) * V + v[:, None]) * H + c[None, :], mask=vm[:, None] & cm[None, :], other=0.)
      key += tl.sum(w.to(tl.float32) * h[None, :], axis=1)
    key += tl.load(B + (ka * N + n) * V + v, mask=vm, other=0.).to(tl.float32)
    key = tl.where(vm, key, 0.)
    key = key * tl.rsqrt(tl.sum(key * key, axis=0) / V + EPS)
    if PENDING > 0:       # M' = M + sum_g C_g^T A_g (C: gated RMS content, A: RMS address)
      dt = mraw.dtype
      cc = rc1.to(tl.float32) + rl1.to(tl.float32)
      cc = cc * (tl.sigmoid(rg1.to(tl.float32)) * tl.rsqrt(tl.sum(cc * cc, axis=1) / K + PEPS))[:, None]
      aa = ra1.to(tl.float32)
      aa = aa * tl.rsqrt(tl.sum(aa * aa, axis=1) / V + PEPS)[:, None]
      acc = tl.dot(tl.trans(cc.to(dt)), aa.to(dt), mraw.to(tl.float32), input_precision=PREC)
      if PENDING > 1:
        cc = rc2.to(tl.float32)
        cc = cc * (tl.sigmoid(rg2.to(tl.float32)) * tl.rsqrt(tl.sum(cc * cc, axis=1) / K + PEPS))[:, None]
        aa = ra2.to(tl.float32)
        aa = aa * tl.rsqrt(tl.sum(aa * aa, axis=1) / V + PEPS)[:, None]
        acc = tl.dot(tl.trans(cc.to(dt)), aa.to(dt), acc, input_precision=PREC)
      mkv = acc.to(dt)
      if (n == 0) & (ka == 0):
        tl.store(MOUT + t * s_mt + k[:, None] * V + v[None, :], mkv, mask=km[:, None] & vm[None, :])
    else:
      mkv = mraw
    for a2 in tl.static_range(2):
      a = tl.minimum(ka + a2, 3)
      if (a2 == 0) | (ka == 2):            # q, k: one row; VO key: v and o rows
        gd = tl.load(GATES + t * s_gt + a * N + n).to(tl.float32)
        gs = tl.load(GATES + t * s_gt + (4 + a) * N + n).to(tl.float32)
        st = tl.load(S + a * V * N + v * N + n, mask=vm, other=0.).to(tl.float32)
        x = SCALE * tl.sigmoid(gd) * key + (1.1 * tl.sigmoid(gs) / STATIC_DIV) * st
        y = tl.sum(mkv.to(tl.float32) * x[None, :], axis=1)              # [BK]
        row = Y + t * s_y_t + (a * N + n) * K
        if ka < 2:
          scale = tl.where(a == 0, QSCALE, 1.0)
          tl.store(row + k, (y * scale).to(Y.dtype.element_ty), mask=k < QCOLS)
          j = tl.arange(0, RB)
          half = R // 2
          i = tl.where(j < half, j, j - half)
          partner = tl.where(j < half, j + half, j - half)
          jm = j < R
          base = STD + t * s_st + a * s_sarm + n * s_sn
          s1 = tl.load(base + j, mask=jm, other=0.).to(tl.float32)
          s2 = tl.load(base + partner, mask=jm, other=0.).to(tl.float32)
          pos = tl.load(POS + (t // TLEN) * s_pb + (t % TLEN) * s_pt).to(tl.float32)
          ang = pos / tl.exp(LOG_MAX_TS * 2.0 * i.to(tl.float32) / R)
          rot = s1 * tl.cos(ang) + tl.where(j < half, -1.0, 1.0) * s2 * tl.sin(ang)
          tl.store(row + QCOLS + j, (rot * scale).to(Y.dtype.element_ty), mask=jm)
        else:
          tl.store(row + k, y.to(Y.dtype.element_ty), mask=km)


@torch.library.custom_op('bam::decode_read', mutates_args=())
def bam_decode_read(hidden: torch.Tensor, up_weight: torch.Tensor, up_bias: torch.Tensor, m: torch.Tensor,
                    gates: torch.Tensor, static_keys: torch.Tensor, std: torch.Tensor, positions: torch.Tensor,
                    eps: float, scale: float, static_div: float, qk_cols: int, q_scale: float,
                    max_timescale: float, c1: torch.Tensor, l1: torch.Tensor, g1: torch.Tensor, a1: torch.Tensor,
                    c2: torch.Tensor, g2: torch.Tensor, a2: torch.Tensor, pending: int, has_local: bool,
                    write_eps: float, write_hidden: torch.Tensor, write_weight: torch.Tensor,
                    write_bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Decode-step read with the key up-projection fused in.
  hidden [T,H] pre-GELU read bottleneck (unit last stride); up_weight [3*N*V, H] / up_bias [3*N*V] (q, k, vo
  keys; pre-biases folded); m [T,K,V] contiguous; gates [T,8,N]; static_keys [4,V,N]; std [T,2,N,R];
  positions [B,Tlen] (T = B*Tlen; any strides). Returns attention-ready Y [T,4,N,K] (RoPE, q scale applied)."""
  t, kdim, vdim = m.shape
  n = gates.shape[2]
  h = hidden.shape[1]
  assert m.is_contiguous() and up_weight.is_contiguous() and static_keys.is_contiguous()
  assert hidden.stride(-1) == 1 and gates.stride(-1) == 1 and gates.stride(-2) == n and std.stride(-1) == 1
  y = torch.empty((t, 4, n, kdim), device=m.device, dtype=m.dtype)
  addr = torch.empty((t, n, vdim), device=m.device, dtype=m.dtype)
  m_out, ptrs, strides = _pending(m, c1, l1, g1, a1, c2, g2, a2, pending)
  assert write_hidden.stride(-1) == 1 and write_weight.is_contiguous()
  if t:
    _decode_read_kernel[(n, 4, t)](
        hidden, up_weight, up_bias, m, gates, static_keys, std, positions, y, *ptrs,
        write_hidden, write_weight, write_bias, addr,
        hidden.stride(0), m.stride(0), gates.stride(0), std.stride(0), std.stride(2), std.stride(1),
        positions.stride(0), positions.stride(1), y.stride(0), positions.shape[1],
        write_hidden.stride(0), addr.stride(0), *strides,
        N=n, K=kdim, V=vdim, H=h, BK=triton.next_power_of_2(kdim), BV=max(16, triton.next_power_of_2(vdim)),
        BH=triton.next_power_of_2(h), BC=DECODE_BC, EPS=eps, SCALE=scale, STATIC_DIV=static_div, QCOLS=qk_cols,
        R=kdim - qk_cols, RB=triton.next_power_of_2(kdim - qk_cols), QSCALE=q_scale,
        LOG_MAX_TS=math.log(max_timescale), BN=max(16, triton.next_power_of_2(n)), PENDING=pending,
        HAS_LOCAL=has_local, PEPS=write_eps, PREC=_prec(m), WH=write_hidden.shape[1],
        BWH=triton.next_power_of_2(write_hidden.shape[1]), num_warps=DECODE_WARPS)
  return y, m_out, addr


@bam_decode_read.register_fake
def _(hidden, up_weight, up_bias, m, gates, static_keys, std, positions, eps, scale, static_div, qk_cols, q_scale,
      max_timescale, c1, l1, g1, a1, c2, g2, a2, pending, has_local, write_eps, write_hidden, write_weight,
      write_bias):
  return (m.new_empty((m.shape[0], 4, gates.shape[2], m.shape[1])),
          torch.empty_like(m) if pending else m.new_empty((0,)),
          m.new_empty((m.shape[0], gates.shape[2], m.shape[2])))



@triton.jit
def _rope_tail_kernel(Y, STD, POS, ROWS, s_yt, s_st, s_sn, s_sarm, N: tl.constexpr, K: tl.constexpr,
                      QCOLS: tl.constexpr, R: tl.constexpr, RB: tl.constexpr, LOG_MAX_TS: tl.constexpr,
                      BROWS: tl.constexpr):
  """Y[t, a, n, QCOLS:] = RoPE(std[t, a, n, :]) for the q, k arms, in place. Rows r = (t * 2 + a) * N + n are
  flattened across tokens so a program handles a dense [BROWS, RB] tile (bandwidth-bound)."""
  r = tl.program_id(0) * BROWS + tl.arange(0, BROWS)
  rm = r < ROWS
  t = r // (2 * N)
  a = (r // N) % 2
  n = r % N
  j = tl.arange(0, RB)
  half = R // 2
  i = tl.where(j < half, j, j - half)
  partner = tl.where(j < half, j + half, j - half)
  mask = rm[:, None] & (j < R)[None, :]
  base = STD + t[:, None] * s_st + a[:, None] * s_sarm + n[:, None] * s_sn
  s1 = tl.load(base + j[None, :], mask=mask, other=0.).to(tl.float32)
  s2 = tl.load(base + partner[None, :], mask=mask, other=0.).to(tl.float32)
  pos = tl.load(POS + t, mask=rm, other=0).to(tl.float32)
  ang = pos[:, None] * tl.exp(-LOG_MAX_TS * 2.0 * i.to(tl.float32) / R)[None, :]
  rot = s1 * tl.cos(ang) + tl.where(j < half, -1.0, 1.0)[None, :] * s2 * tl.sin(ang)
  tl.store(Y + t[:, None] * s_yt + (a * N + n)[:, None] * K + QCOLS + j[None, :], rot.to(Y.dtype.element_ty),
           mask=mask)


@torch.library.custom_op('bam::rope_tail', mutates_args=('y',))
def bam_rope_tail(y: torch.Tensor, std: torch.Tensor, positions: torch.Tensor, qk_cols: int,
                  max_timescale: float) -> None:
  """In place: q/k columns >= qk_cols of y [T,4,N,K] (from bam_read without rope) = RoPE(std [T,2,N,R])."""
  t, _, n, kdim = y.shape
  assert y.is_contiguous() and std.stride(-1) == 1 and positions.is_contiguous()
  rows = t * 2 * n
  if rows:
    brows = 32      # A800: 32 us/layer at T=4096 (a plain copy of these strided tail bytes takes 22 us)
    _rope_tail_kernel[(triton.cdiv(rows, brows),)](
        y, std, positions, rows, y.stride(0), std.stride(0), std.stride(2), std.stride(1), N=n, K=kdim,
        QCOLS=qk_cols, R=kdim - qk_cols, RB=triton.next_power_of_2(kdim - qk_cols),
        LOG_MAX_TS=math.log(max_timescale), BROWS=brows, num_warps=8)


@triton.jit
def _write_kernel(M, OUT, C1, CL1, G1, A1, C2, G2, A2, OH,
                  s_mt, s_c1t, s_c1n, s_cl1t, s_cl1n, s_g1t, s_a1t, s_a1n, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n,
                  N: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  BV: tl.constexpr, TWO: tl.constexpr, HAS_LOCAL: tl.constexpr, EPS: tl.constexpr,
                  PREC: tl.constexpr, STORE_SUM: tl.constexpr):
  """Program = token t: M'_t = M_t + sum_g C_g^T A_g (one or two groups). A token loop or K split measured
  1.6x slower on A800 (prefill, T=4096); decode-step writes are fused into the next read instead."""
  t = tl.program_id(0)
  n = tl.arange(0, BN)
  k = tl.arange(0, BK)
  v = tl.arange(0, BV)
  nm, km, vm = n < N, k < K, v < V
  mask = km[:, None] & vm[None, :]
  acc = tl.load(M + t * s_mt + k[:, None] * V + v[None, :], mask=mask, other=0.).to(tl.float32)
  dt = OUT.dtype.element_ty
  if STORE_SUM:      # o_head = attention output + local O, for the output projection (saves an add kernel)
    oh = tl.load(C1 + t * s_c1t + n[:, None] * s_c1n + k[None, :], mask=nm[:, None] & km[None, :], other=0.).to(tl.float32)
    oh += tl.load(CL1 + t * s_cl1t + n[:, None] * s_cl1n + k[None, :], mask=nm[:, None] & km[None, :],
                  other=0.).to(tl.float32)
    tl.store(OH + t * N * K + n[:, None] * K + k[None, :], oh.to(dt), mask=nm[:, None] & km[None, :])
  c, a = _group(C1, CL1, G1, A1, t, s_c1t, s_c1n, s_cl1t, s_cl1n, s_g1t, s_a1t, s_a1n, n, k, k, v, nm, km, km, vm,
                K, V, EPS, HAS_LOCAL, False)
  acc = tl.dot(tl.trans(c.to(dt)), a.to(dt), acc, input_precision=PREC)
  if TWO:
    c, a = _group(C2, C2, G2, A2, t, s_c2t, s_c2n, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n, n, k, k, v, nm, km, km, vm,
                  K, V, EPS, False, False)
    acc = tl.dot(tl.trans(c.to(dt)), a.to(dt), acc, input_precision=PREC)
  tl.store(OUT + t * s_mt + k[:, None] * V + v[None, :], acc.to(dt), mask=mask)


@torch.library.custom_op('bam::write', mutates_args=())
def bam_write(m: torch.Tensor, content: torch.Tensor, local: torch.Tensor, gate: torch.Tensor, address: torch.Tensor,
              content2: torch.Tensor, gate2: torch.Tensor, address2: torch.Tensor, has_local: bool, two: bool,
              eps: float) -> torch.Tensor:
  """m [T,K,V] contiguous; content (+ local if has_local) [T,N,K], gate [T,N] logits, address [T,N,V] raw
  (unit last strides); the second group is used iff `two`. Returns M + sum_g (sigmoid(gate) rms(content))^T
  rms(address)."""
  t, kdim, vdim = m.shape
  n = content.shape[1]
  assert m.is_contiguous()
  for x in (content, local, gate, address, content2, gate2, address2):
    assert x.stride(-1) == 1
  out = torch.empty_like(m)
  if t:
    _write_kernel[(t,)](
        m, out, content, local, gate, address, content2, gate2, address2, out,
        m.stride(0), content.stride(0), content.stride(1), local.stride(0), local.stride(1), gate.stride(0),
        address.stride(0), address.stride(1), content2.stride(0), content2.stride(1), gate2.stride(0),
        address2.stride(0), address2.stride(1),
        N=n, K=kdim, V=vdim, BN=max(16, triton.next_power_of_2(n)), BK=triton.next_power_of_2(kdim),
        BV=max(16, triton.next_power_of_2(vdim)), TWO=two, HAS_LOCAL=has_local, EPS=eps, PREC=_prec(m),
        STORE_SUM=False, num_warps=4)
  return out


@torch.library.custom_op('bam::write_ohead', mutates_args=())
def bam_write_ohead(m: torch.Tensor, attn: torch.Tensor, local: torch.Tensor, gate: torch.Tensor,
                    address: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
  """Single-group write with content attn + local; also returns o_head = attn + local [T,N,K] (prefill)."""
  t, kdim, vdim = m.shape
  n = attn.shape[1]
  assert m.is_contiguous()
  for x in (attn, local, gate, address):
    assert x.stride(-1) == 1
  out = torch.empty_like(m)
  oh = torch.empty((t, n, kdim), device=m.device, dtype=m.dtype)
  if t:
    _write_kernel[(t,)](
        m, out, attn, local, gate, address, attn, gate, address, oh,
        m.stride(0), attn.stride(0), attn.stride(1), local.stride(0), local.stride(1), gate.stride(0),
        address.stride(0), address.stride(1), attn.stride(0), attn.stride(1), gate.stride(0),
        address.stride(0), address.stride(1),
        N=n, K=kdim, V=vdim, BN=max(16, triton.next_power_of_2(n)), BK=triton.next_power_of_2(kdim),
        BV=max(16, triton.next_power_of_2(vdim)), TWO=False, HAS_LOCAL=True, EPS=eps, PREC=_prec(m),
        STORE_SUM=True, num_warps=4)
  return out, oh


@bam_write_ohead.register_fake
def _(m, attn, local, gate, address, eps):
  return torch.empty_like(m), m.new_empty((m.shape[0], attn.shape[1], m.shape[1]))


@bam_write.register_fake
def _(m, content, local, gate, address, content2, gate2, address2, has_local, two, eps):
  return torch.empty_like(m)
