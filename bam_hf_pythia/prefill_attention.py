"""Strictly causal flash attention (prefill, no cache) in Triton for BAM's Q/K width 100, V width 80.

Query i attends to keys j < i (MaxText negative-diagonal attention); the first query returns zeros.
Q/K columns are loaded as 64 + 32 + 16 blocks (the last masked to 100), V as 64 + 16, so neither
memory padding nor 128-wide dot products are needed (SDPA flash needs equal q/k/v widths).
Inputs keep the projection layout [B, T, N, D] (no transposes).
"""
import torch
import triton
import triton.language as tl


def _configs():
  out = []
  for bm, bn, w, s in [(128, 64, 8, 2), (128, 64, 4, 2), (128, 32, 4, 2), (64, 64, 4, 2), (64, 64, 4, 3),
                       (128, 64, 8, 3), (128, 128, 8, 2), (64, 32, 4, 3), (128, 32, 8, 3), (128, 64, 4, 3),
                       (128, 128, 8, 3), (64, 128, 4, 2)]:
    for qs in (0, 1):
      for vs in (0, 1):
        out.append(triton.Config({'BM': bm, 'BN': bn, 'QSPLIT': qs, 'VSPLIT': vs}, num_warps=w, num_stages=s))
  return out


@triton.jit
def _block(acc0, acc1, l_i, m_i, q0, q1, q2, K, V, kbase, vbase, skt, svt, offs_m, start, T,
           DQ: tl.constexpr, BN: tl.constexpr, MASK: tl.constexpr, QSPLIT: tl.constexpr, VSPLIT: tl.constexpr):
  offs_n = start + tl.arange(0, BN)
  nmask = offs_n < T
  kp = K + kbase + offs_n[None, :] * skt
  if QSPLIT:
    d0 = tl.arange(0, 64)
    d1 = 64 + tl.arange(0, 32)
    d2 = 96 + tl.arange(0, 16)
    k0 = tl.load(kp + d0[:, None], mask=nmask[None, :], other=0.)
    k1 = tl.load(kp + d1[:, None], mask=nmask[None, :], other=0.)
    k2 = tl.load(kp + d2[:, None], mask=nmask[None, :] & (d2[:, None] < DQ), other=0.)
    s = tl.dot(q0, k0) + tl.dot(q1, k1) + tl.dot(q2, k2)
  else:
    d = tl.arange(0, 128)
    kk = tl.load(kp + d[:, None], mask=nmask[None, :] & (d[:, None] < DQ), other=0.)
    s = tl.dot(q0, kk)
  if MASK:
    s = tl.where(offs_n[None, :] < offs_m[:, None], s, float('-inf'))
  m_new = tl.maximum(m_i, tl.max(s, 1))
  m_safe = tl.where(m_new == float('-inf'), 0., m_new)
  p = tl.math.exp2((s - m_safe[:, None]) * 1.4426950408889634)
  alpha = tl.math.exp2((m_i - m_safe) * 1.4426950408889634)
  l_i = l_i * alpha + tl.sum(p, 1)
  vp = V + vbase + offs_n[:, None] * svt
  pb = p.to(K.dtype.element_ty)
  if VSPLIT:
    v0 = tl.load(vp + tl.arange(0, 64)[None, :], mask=nmask[:, None], other=0.)
    v1 = tl.load(vp + 64 + tl.arange(0, 16)[None, :], mask=nmask[:, None], other=0.)
    acc0 = acc0 * alpha[:, None] + tl.dot(pb, v0)
    acc1 = acc1 * alpha[:, None] + tl.dot(pb, v1)
  else:
    e = tl.arange(0, 128)
    vv = tl.load(vp + e[None, :], mask=nmask[:, None] & (e[None, :] < 80), other=0.)
    acc0 = acc0 * alpha[:, None] + tl.dot(pb, vv)
  return acc0, acc1, l_i, m_new


@triton.autotune(configs=_configs(), key=['T', 'N'])
@triton.jit
def _strict_causal_kernel(Q, K, V, LO, O, sqb, sqt, sqn, skb, skt, skn, svb, svt, svn, sob, sot, son, T, N,
                          DQ: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, QSPLIT: tl.constexpr,
                          VSPLIT: tl.constexpr):
  pid_m = tl.program_id(0)
  bn = tl.program_id(1)
  b = bn // N
  n = bn % N
  offs_m = pid_m * BM + tl.arange(0, BM)
  mmask = offs_m < T
  qp = Q + b * sqb + n * sqn + offs_m[:, None] * sqt
  if QSPLIT:
    d2 = 96 + tl.arange(0, 16)
    q0 = tl.load(qp + tl.arange(0, 64)[None, :], mask=mmask[:, None], other=0.)
    q1 = tl.load(qp + 64 + tl.arange(0, 32)[None, :], mask=mmask[:, None], other=0.)
    q2 = tl.load(qp + d2[None, :], mask=mmask[:, None] & (d2[None, :] < DQ), other=0.)
  else:
    d = tl.arange(0, 128)
    q0 = tl.load(qp + d[None, :], mask=mmask[:, None] & (d[None, :] < DQ), other=0.)
    q1 = q0
    q2 = q0
  m_i = tl.full((BM,), float('-inf'), tl.float32)
  l_i = tl.zeros((BM,), tl.float32)
  if VSPLIT:
    acc0 = tl.zeros((BM, 64), tl.float32)
  else:
    acc0 = tl.zeros((BM, 128), tl.float32)
  acc1 = tl.zeros((BM, 16), tl.float32)
  kbase = b * skb + n * skn
  vbase = b * svb + n * svn
  lo = pid_m * BM               # keys below the query block: no mask
  for start in range(0, lo, BN):
    acc0, acc1, l_i, m_i = _block(acc0, acc1, l_i, m_i, q0, q1, q2, K, V, kbase, vbase, skt, svt, offs_m, start,
                                  T, DQ, BN, False, QSPLIT, VSPLIT)
  for start in range(lo, tl.minimum(lo + BM, T), BN):
    acc0, acc1, l_i, m_i = _block(acc0, acc1, l_i, m_i, q0, q1, q2, K, V, kbase, vbase, skt, svt, offs_m, start,
                                  T, DQ, BN, True, QSPLIT, VSPLIT)
  inv = 1.0 / tl.maximum(l_i, 1e-30)            # the first query has no keys: 0
  op = O + b * sob + n * son + offs_m[:, None] * sot
  lp = LO + vbase + offs_m[:, None] * svt         # local O: same layout as V; o_head = attention + local O
  if VSPLIT:
    e0 = tl.arange(0, 64)
    e1 = 64 + tl.arange(0, 16)
    y0 = (acc0 * inv[:, None]).to(O.dtype.element_ty).to(tl.float32) + tl.load(lp + e0[None, :], mask=mmask[:, None],
                                                                              other=0.).to(tl.float32)
    y1 = (acc1 * inv[:, None]).to(O.dtype.element_ty).to(tl.float32) + tl.load(lp + e1[None, :], mask=mmask[:, None],
                                                                              other=0.).to(tl.float32)
    tl.store(op + e0[None, :], y0.to(O.dtype.element_ty), mask=mmask[:, None])
    tl.store(op + e1[None, :], y1.to(O.dtype.element_ty), mask=mmask[:, None])
  else:
    e = tl.arange(0, 128)
    em = mmask[:, None] & (e[None, :] < 80)
    y = (acc0 * inv[:, None]).to(O.dtype.element_ty).to(tl.float32) + tl.load(lp + e[None, :], mask=em,
                                                                             other=0.).to(tl.float32)
    tl.store(op + e[None, :], y.to(O.dtype.element_ty), mask=em)


@torch.library.custom_op('bam_pythia::strict_causal_attention', mutates_args=())
def strict_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, local_o: torch.Tensor) -> torch.Tensor:
  """q/k [B,T,N,100] (q pre-scaled), v and local_o [B,T,N,80] (same strides); last dims contiguous.
  Returns attention + local_o [B,T,N,80] (BAM o_head)."""
  b, t, n, dq = q.shape
  assert 96 < dq <= 128 and v.shape[-1] == 80 and q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
  assert local_o.stride() == v.stride()
  out = v.new_empty((b, t, n, 80))
  grid = lambda meta: (triton.cdiv(t, meta['BM']), b * n)
  _strict_causal_kernel[grid](q, k, v, local_o, out, q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1),
                              k.stride(2), v.stride(0), v.stride(1), v.stride(2), out.stride(0), out.stride(1),
                              out.stride(2), t, n, DQ=dq)
  return out


@strict_causal_attention.register_fake
def _(q, k, v, local_o):
  return v.new_empty((*q.shape[:-1], 80))
