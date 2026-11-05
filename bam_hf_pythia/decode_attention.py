"""Split-KV decode attention (flash-decoding) in Triton, CUDA-graph safe.

One query token per sequence attends to the first `lengths[b]` entries of a preallocated cache.
The grid depends only on the cache shape, the length is read on device, so a decode step keeps
fixed shapes. Stage 1 computes per-split (max, sum, unnormalized output); stage 2 merges splits.
Q/K and V widths may differ (BAM: 100 / 80); a zero length returns zeros (BAM's strictly causal
first token). Shared by BAM and the Pythia baseline.
"""
import math
from typing import Optional
import os

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Q, K, V, LEN, OUT, MAX, SUM,
             sqb, sqh, skb, skh, sks, svb, svh, svs, sob, soh, sos,
             N, DQ: tl.constexpr, BLOCK_DQ: tl.constexpr, DV: tl.constexpr, BLOCK_DV: tl.constexpr,
             BLOCK_N: tl.constexpr, SPLIT: tl.constexpr):
  bh = tl.program_id(0)
  split = tl.program_id(1)
  b = bh // N
  h = bh % N
  length = tl.load(LEN + b)
  start = split * SPLIT
  d = tl.arange(0, BLOCK_DQ)
  dmask = d < DQ
  e = tl.arange(0, BLOCK_DV)
  emask = e < DV
  q = tl.load(Q + b * sqb + h * sqh + d, mask=dmask, other=0.).to(tl.float32)
  m = tl.full((), -1e30, tl.float32)
  l = tl.zeros((), tl.float32)
  acc = tl.zeros((BLOCK_DV,), tl.float32)
  for n0 in range(0, SPLIT, BLOCK_N):
    offs = start + n0 + tl.arange(0, BLOCK_N)
    valid = offs < length
    k = tl.load(K + b * skb + h * skh + offs[:, None] * sks + d[None, :], mask=valid[:, None] & dmask[None, :],
                other=0.).to(tl.float32)
    s = tl.sum(k * q[None, :], axis=1)
    s = tl.where(valid, s, -1e30)
    m_new = tl.maximum(m, tl.max(s, axis=0))
    p = tl.where(valid, tl.exp(s - m_new), 0.)
    alpha = tl.exp(m - m_new)
    v = tl.load(V + b * svb + h * svh + offs[:, None] * svs + e[None, :], mask=valid[:, None] & emask[None, :],
                other=0.).to(tl.float32)
    acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
    l = l * alpha + tl.sum(p, axis=0)
    m = m_new
  base = b * sob + h * soh + split * sos
  tl.store(OUT + base + e, acc, mask=emask)
  tl.store(MAX + bh * tl.num_programs(1) + split, m)
  tl.store(SUM + bh * tl.num_programs(1) + split, l)


@triton.jit
def _dot_rows(P, offs, valid, stride, x0, x1, x2, C1: tl.constexpr, C2: tl.constexpr):
  """[BLOCK_N] dot products of rows P + offs * stride with x split as 64 + C1 + C2 columns."""
  base = P + offs[:, None] * stride
  s = tl.sum(tl.load(base + tl.arange(0, 64)[None, :], mask=valid[:, None], other=0.).to(tl.float32) * x0[None, :], 1)
  if C1 > 0:
    s += tl.sum(tl.load(base + 64 + tl.arange(0, C1)[None, :], mask=valid[:, None], other=0.).to(tl.float32)
                * x1[None, :], 1)
  if C2 > 0:
    s += tl.sum(tl.load(base + 64 + C1 + tl.arange(0, C2)[None, :], mask=valid[:, None], other=0.).to(tl.float32)
                * x2[None, :], 1)
  return s


@triton.jit
def _partial_chunked(Q, K, V, LEN, OUT, MAX, SUM,
                     sqb, sqh, skb, skh, sks, svb, svh, svs, sob, soh, sos, N,
                     Q1: tl.constexpr, Q2: tl.constexpr, V1: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr):
  """Like _partial, with Q/K columns as 64 + Q1 + Q2 and V columns as 64 + V1 (no padding to 128)."""
  bh = tl.program_id(0)
  split = tl.program_id(1)
  b = bh // N
  h = bh % N
  length = tl.load(LEN + b)
  start = split * SPLIT
  qp = Q + b * sqb + h * sqh
  q0 = tl.load(qp + tl.arange(0, 64)).to(tl.float32) * 1.4426950408889634
  q1 = tl.load(qp + 64 + tl.arange(0, max(Q1, 1)), mask=tl.arange(0, max(Q1, 1)) < Q1, other=0.).to(tl.float32) * 1.4426950408889634
  q2 = tl.load(qp + 64 + Q1 + tl.arange(0, max(Q2, 1)), mask=tl.arange(0, max(Q2, 1)) < Q2,
               other=0.).to(tl.float32) * 1.4426950408889634
  m = tl.full((), -1e30, tl.float32)
  l = tl.zeros((), tl.float32)
  acc0 = tl.zeros((64,), tl.float32)
  acc1 = tl.zeros((max(V1, 1),), tl.float32)
  kp = K + b * skb + h * skh
  vp = V + b * svb + h * svh
  for n0 in range(0, SPLIT, BLOCK_N):
    offs = start + n0 + tl.arange(0, BLOCK_N)
    valid = offs < length
    s = _dot_rows(kp, offs, valid, sks, q0, q1, q2, Q1, Q2)          # log2-scaled logits
    s = tl.where(valid, s, -1e30)
    m_new = tl.maximum(m, tl.max(s, axis=0))
    p = tl.where(valid, tl.math.exp2(s - m_new), 0.)
    alpha = tl.math.exp2(m - m_new)
    vb = vp + offs[:, None] * svs
    acc0 = acc0 * alpha + tl.sum(p[:, None] * tl.load(vb + tl.arange(0, 64)[None, :], mask=valid[:, None],
                                                      other=0.).to(tl.float32), 0)
    if V1 > 0:
      acc1 = acc1 * alpha + tl.sum(p[:, None] * tl.load(vb + 64 + tl.arange(0, V1)[None, :], mask=valid[:, None],
                                                        other=0.).to(tl.float32), 0)
    l = l * alpha + tl.sum(p, axis=0)
    m = m_new
  base = OUT + b * sob + h * soh + split * sos
  tl.store(base + tl.arange(0, 64), acc0)
  if V1 > 0:
    tl.store(base + 64 + tl.arange(0, V1), acc1)
  tl.store(MAX + bh * tl.num_programs(1) + split, m * 0.6931471805599453)      # back to natural log
  tl.store(SUM + bh * tl.num_programs(1) + split, l)


@triton.jit
def _kt_update(acc, l, m, k, q, vp, svs, offs, length, e, emask):
  """Online-softmax update with one [DQ, BLOCK_N] key tile (log2-scaled q)."""
  valid = offs < length
  s = tl.sum(k * q[:, None], axis=0)
  s = tl.where(valid, s, -1e30)
  m_new = tl.maximum(m, tl.max(s, axis=0))
  p = tl.where(valid, tl.math.exp2(s - m_new), 0.)
  alpha = tl.math.exp2(m - m_new)
  v = tl.load(vp + offs[:, None] * svs + e[None, :], mask=valid[:, None] & emask[None, :], other=0.).to(tl.float32)
  return acc * alpha + tl.sum(p[:, None] * v, axis=0), l * alpha + tl.sum(p, axis=0), m_new


@triton.jit
def _partial_kt(Q, K, V, LEN, OUT, MAX, SUM,
                sqb, sqh, skb, skh, skblk, svb, svh, svs, sob, soh, sos, N, TAIL, stb, sth,
                DQ: tl.constexpr, BLOCK_DQ: tl.constexpr, DV: tl.constexpr, BLOCK_DV: tl.constexpr,
                BLOCK_N: tl.constexpr, SPLIT: tl.constexpr, HAS_TAIL: tl.constexpr):
  """Like _partial, for a key cache in blocks of BLOCK_N positions stored transposed: K[b, h, blk, d, j]
  (each tile of BLOCK_N keys is one contiguous [DQ, BLOCK_N] block: fully coalesced loads). With
  HAS_TAIL, the block holding position `length` (the one being filled) and the one before it are read
  from TAIL [B, N, 2, BLOCK_N, DQ] slots (block % 2) in rows layout (decoding writes a key there contiguously; the scattered single-position write
  into a transposed block costs ~7-15 us per layer)."""
  bh = tl.program_id(0)
  split = tl.program_id(1)
  b = bh // N
  h = bh % N
  length = tl.load(LEN + b)
  start = split * SPLIT
  d = tl.arange(0, BLOCK_DQ)
  dmask = d < DQ
  e = tl.arange(0, BLOCK_DV)
  emask = e < DV
  j = tl.arange(0, BLOCK_N)
  q = tl.load(Q + b * sqb + h * sqh + d, mask=dmask, other=0.).to(tl.float32) * 1.4426950408889634
  m = tl.full((), -1e30, tl.float32)
  l = tl.zeros((), tl.float32)
  acc = tl.zeros((BLOCK_DV,), tl.float32)
  kp = K + b * skb + h * skh
  vp = V + b * svb + h * svh
  # with a tail, the main loop (constant trip count: keeps its loads pipelined) covers the blocks before the
  # last two; the block being filled and the previous one (moved into K during the current block) are read
  # from their tail slots after it
  cb = length // BLOCK_N
  lim = tl.maximum(cb - 1, 0) * BLOCK_N if HAS_TAIL else length
  for n0 in range(0, SPLIT, BLOCK_N):
    offs = start + n0 + j
    blk = (start + n0) // BLOCK_N
    k = tl.load(kp + blk * skblk + d[:, None] * BLOCK_N + j[None, :], mask=dmask[:, None] & (offs < lim)[None, :],
                other=0.).to(tl.float32)
    acc, l, m = _kt_update(acc, l, m, k, q, vp, svs, offs, lim, e, emask)
  if HAS_TAIL:
    for back in range(2):
      tb = cb - 1 + back
      p0 = tb * BLOCK_N
      if tb >= 0 and p0 >= start and p0 < start + SPLIT:
        offs = p0 + j
        k = tl.load(TAIL + b * stb + h * sth + (tb % 2) * (BLOCK_N * DQ) + j[None, :] * DQ + d[:, None],
                    mask=dmask[:, None] & (offs < length)[None, :], other=0.).to(tl.float32)
        acc, l, m = _kt_update(acc, l, m, k, q, vp, svs, offs, length, e, emask)
  base = OUT + b * sob + h * soh + split * sos
  tl.store(base + e, acc, mask=emask)
  tl.store(MAX + bh * tl.num_programs(1) + split, m * 0.6931471805599453)
  tl.store(SUM + bh * tl.num_programs(1) + split, l)


@triton.jit
def _merge(OUT, MAX, SUM, O, sob, soh, sos, sb, sh, N, SPLITS: tl.constexpr, BLOCK_S: tl.constexpr,
           D: tl.constexpr, BLOCK_D: tl.constexpr):
  bh = tl.program_id(0)
  b = bh // N
  h = bh % N
  s = tl.arange(0, BLOCK_S)
  smask = s < SPLITS
  m = tl.load(MAX + bh * SPLITS + s, mask=smask, other=-1e30)
  l = tl.load(SUM + bh * SPLITS + s, mask=smask, other=0.)
  mx = tl.max(m, axis=0)
  w = tl.where(smask, tl.exp(m - mx), 0.)
  d = tl.arange(0, BLOCK_D)
  dmask = d < D
  part = tl.load(OUT + b * sob + h * soh + s[:, None] * sos + d[None, :], mask=smask[:, None] & dmask[None, :],
                 other=0.)
  out = tl.sum(part * w[:, None], axis=0) / tl.maximum(tl.sum(l * w, axis=0), 1e-30)   # empty: 0
  tl.store(O + b * sb + h * sh + d, out.to(O.dtype.element_ty), mask=dmask)


_SMS = {}


def _sm_count(device):
  if device not in _SMS:
    _SMS[device] = torch.cuda.get_device_properties(device).multi_processor_count
  return _SMS[device]


def _splits(batch_heads, max_len, block_n, sms, waves):
  """Enough splits to cover `waves` waves of SMs, each a multiple of block_n tokens."""
  want = max(1, min(math.ceil(waves * sms / batch_heads), math.ceil(max_len / block_n)))
  split = triton.cdiv(triton.cdiv(max_len, want), block_n) * block_n
  return triton.cdiv(max_len, split), split


KT_BLOCK = 64     # positions per block of the transposed ('kt') key cache [B, N, S / KT_BLOCK, Dq, KT_BLOCK]


def kt_cache_shape(batch, heads, max_len, dq):
  return (batch, heads, triton.cdiv(max_len, KT_BLOCK), dq, KT_BLOCK)


def kt_from_rows(k):
  """[B, N, T, Dq] keys -> [B, N, ceil(T / KT_BLOCK), Dq, KT_BLOCK] (the tail block zero-padded)."""
  b, n, t, dq = k.shape
  pad = -t % KT_BLOCK
  if pad:
    k = torch.nn.functional.pad(k, (0, 0, 0, pad))
  return k.unflatten(2, (-1, KT_BLOCK)).transpose(-1, -2)


# Key layouts: 'rows' [B,N,S,Dq] read as one 128-wide block; 'chunked' [B,N,S,Dq] read as 64 + 32/16 + 4
# column chunks; 'kt' transposed blocks (contiguous [Dq, 64] tiles). A800, 2k context, 32 heads, per layer:
#   Dq 100 / Dv 80 (BAM):     rows 19 / 300 / 1038 us at B = 1 / 16 / 64; kt 16.6 / 255 / 975
#   Dq 80 / Dv 80 (Pythia):   rows 14.5 / 216 / 951;  chunked 12.4 / 283 / 1000;  kt 15.2 / 243 / 911
@torch.library.custom_op('bam_pythia::decode_attention', mutates_args=())
def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor,
                     layout: str = 'rows', k_tail: Optional[torch.Tensor] = None) -> torch.Tensor:
  """q [B,N,1,Dq] (pre-scaled); k cache in `layout`, v [B,N,S,Dv]; lengths [B] int (valid entries).
  Returns [B,N,1,Dv]."""
  b, n, _, dim = q.shape
  dv = v.shape[-1]
  max_len = v.shape[2]
  lengths = lengths.contiguous()
  assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
  if layout == 'kt':
    block_n, waves, warps = 64, 16, 4
  else:   # A800 sweep (N=20, D=96, 4k cache): 74-83% of HBM bandwidth at B=16/64, 0.021 ms/layer at B=1.
    block_n, waves = (64, 16) if b * n < 160 else (32, 8)
    warps = 2
  splits, split = _splits(b * n, max_len, block_n, _sm_count(q.device), waves)
  part = torch.empty((b, n, splits, dv), device=q.device, dtype=torch.float32)
  mx = torch.empty((b * n * splits,), device=q.device, dtype=torch.float32)
  sm = torch.empty_like(mx)
  block_d, block_dv = triton.next_power_of_2(dim), triton.next_power_of_2(dv)
  common = (q, k, v, lengths, part, mx, sm, q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2), part.stride(0), part.stride(1), part.stride(2), n)
  if layout == 'kt':
    tail = k_tail if k_tail is not None else k
    _partial_kt[(b * n, splits)](*common, tail, tail.stride(0), tail.stride(1), DQ=dim, BLOCK_DQ=block_d, DV=dv,
                                 BLOCK_DV=block_dv, BLOCK_N=block_n, SPLIT=split, HAS_TAIL=k_tail is not None,
                                 num_warps=warps, num_stages=2)
  elif layout == 'chunked':
    q1, q2 = {100: (32, 4), 80: (16, 0), 96: (32, 0), 64: (0, 0)}[dim]
    _partial_chunked[(b * n, splits)](*common, Q1=q1, Q2=q2, V1=dv - 64, BLOCK_N=block_n, SPLIT=split,
                                      num_warps=warps, num_stages=2)
  else:
    _partial[(b * n, splits)](*common, DQ=dim, BLOCK_DQ=block_d, DV=dv, BLOCK_DV=block_dv, BLOCK_N=block_n,
                              SPLIT=split, num_warps=warps, num_stages=2)
  out = q.new_empty((b, n, 1, dv))
  _merge[(b * n,)](part, mx, sm, out, part.stride(0), part.stride(1), part.stride(2), out.stride(0), out.stride(1),
                   n, SPLITS=splits, BLOCK_S=triton.next_power_of_2(splits), D=dv, BLOCK_D=block_dv, num_warps=4)
  return out


@decode_attention.register_fake
def _(q, k, v, lengths, layout='rows', k_tail=None):
  return q.new_empty((*q.shape[:-1], v.shape[-1]))


@triton.jit
def _cache_write_kernel(KC, VC, K, V, POS, skb, skh, sks, svb, svh, svs, sk_b, sk_h, sv_b, sv_h, N,
                        DQ: tl.constexpr, BQ: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, KT: tl.constexpr,
                        KTB: tl.constexpr):
  bh = tl.program_id(0)
  b = bh // N
  h = bh % N
  pos = tl.load(POS)
  d = tl.arange(0, BQ)
  if KT:      # sks = block stride
    kaddr = KC + b * skb + h * skh + (pos // KTB) * sks + d * KTB + pos % KTB
  else:
    kaddr = KC + b * skb + h * skh + pos * sks + d
  tl.store(kaddr, tl.load(K + b * sk_b + h * sk_h + d, mask=d < DQ), mask=d < DQ)
  e = tl.arange(0, BV)
  tl.store(VC + b * svb + h * svh + pos * svs + e, tl.load(V + b * sv_b + h * sv_h + e, mask=e < DV), mask=e < DV)


def _cache_write(k_cache, v_cache, k, v, pos, kt):
  b, n, dq = k.shape
  dv = v.shape[-1]
  assert k.stride(-1) == 1 and v.stride(-1) == 1
  _cache_write_kernel[(b * n,)](k_cache, v_cache, k, v, pos, k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
                                v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), k.stride(0), k.stride(1),
                                v.stride(0), v.stride(1), n, DQ=dq, BQ=triton.next_power_of_2(dq), DV=dv,
                                BV=triton.next_power_of_2(dv), KT=kt, KTB=KT_BLOCK, num_warps=1)


@torch.library.custom_op('bam_pythia::attend_and_append', mutates_args=())
def attend_and_append(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, k: torch.Tensor,
                      v: torch.Tensor, pos: torch.Tensor, lengths: torch.Tensor, strict: bool,
                      layout: str = 'rows') -> torch.Tensor:
  """One decode step: q [B,N,Dq] (pre-scaled) attends to the first lengths[b] cache entries, and k/v
  [B,N,D] are written at pos. strict: lengths = pos, the read precedes the write (BAM); else
  lengths = pos + 1, the new key included. The cache write is not declared as a mutation: declared,
  inductor clones and copies back each whole cache every step (it does not reinplace it). This is
  sound because inside a decode step only this op touches a layer's cache, and the read happens
  before the write inside the op.
  """
  if not strict:
    _cache_write(k_cache, v_cache, k, v, pos, layout == 'kt')
  y = decode_attention(q[:, :, None], k_cache, v_cache, lengths, layout)[:, :, 0]
  if strict:
    _cache_write(k_cache, v_cache, k, v, pos, layout == 'kt')
  return y


@attend_and_append.register_fake
def _(q, k_cache, v_cache, k, v, pos, lengths, strict, layout='rows'):
  return q.new_empty((*q.shape[:-1], v_cache.shape[-1]))
