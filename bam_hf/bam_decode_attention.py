"""Split-KV decode attention (flash-decoding) in Triton, CUDA-graph safe.

One query token per sequence attends to the first `lengths[b]` entries of a preallocated cache.
The grid depends only on the cache shape, the length is read on device, so a decode step keeps
fixed shapes. Stage 1 computes per-split (max, sum, unnormalized output); stage 2 merges splits.
Shared by BAM and the MHA control.
"""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _partial(Q, K, V, LEN, OUT, MAX, SUM,
             sqb, sqh, skb, skh, sks, svb, svh, svs, sob, soh, sos,
             N, D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr, SPLIT: tl.constexpr):
  bh = tl.program_id(0)
  split = tl.program_id(1)
  b = bh // N
  h = bh % N
  length = tl.load(LEN + b)
  start = split * SPLIT
  d = tl.arange(0, BLOCK_D)
  dmask = d < D
  q = tl.load(Q + b * sqb + h * sqh + d, mask=dmask, other=0.).to(tl.float32)
  m = tl.full((), -1e30, tl.float32)
  l = tl.zeros((), tl.float32)
  acc = tl.zeros((BLOCK_D,), tl.float32)
  for n0 in range(0, SPLIT, BLOCK_N):
    offs = start + n0 + tl.arange(0, BLOCK_N)
    valid = offs < length
    kv_mask = valid[:, None] & dmask[None, :]
    k = tl.load(K + b * skb + h * skh + offs[:, None] * sks + d[None, :], mask=kv_mask, other=0.).to(tl.float32)
    s = tl.sum(k * q[None, :], axis=1)
    s = tl.where(valid, s, -1e30)
    m_new = tl.maximum(m, tl.max(s, axis=0))
    p = tl.where(valid, tl.exp(s - m_new), 0.)
    alpha = tl.exp(m - m_new)
    v = tl.load(V + b * svb + h * svh + offs[:, None] * svs + d[None, :], mask=kv_mask, other=0.).to(tl.float32)
    acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
    l = l * alpha + tl.sum(p, axis=0)
    m = m_new
  base = b * sob + h * soh + split * sos
  tl.store(OUT + base + d, acc, mask=dmask)
  tl.store(MAX + bh * tl.num_programs(1) + split, m)
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
  out = tl.sum(part * w[:, None], axis=0) / tl.sum(l * w, axis=0)
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


@torch.library.custom_op('bam::decode_attention', mutates_args=())
def decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
  """q [B,N,1,D] (pre-scaled); k/v [B,N,S,D] caches; lengths [B] int (valid entries). Returns [B,N,1,D]."""
  b, n, _, dim = q.shape
  max_len = k.shape[2]
  lengths = lengths.contiguous()
  assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
  # A800 sweep (N=20, D=96, 4k cache): 74-83% of HBM bandwidth at B=16/64, 0.021 ms/layer at B=1.
  block_n, waves = (64, 16) if b * n < 160 else (32, 8)
  splits, split = _splits(b * n, max_len, block_n, _sm_count(q.device), waves)
  part = torch.empty((b, n, splits, dim), device=q.device, dtype=torch.float32)
  mx = torch.empty((b * n * splits,), device=q.device, dtype=torch.float32)
  sm = torch.empty_like(mx)
  block_d = triton.next_power_of_2(dim)
  _partial[(b * n, splits)](
      q, k, v, lengths, part, mx, sm,
      q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1), v.stride(2),
      part.stride(0), part.stride(1), part.stride(2),
      n, D=dim, BLOCK_D=block_d, BLOCK_N=block_n, SPLIT=split, num_warps=2, num_stages=2)
  out = torch.empty_like(q)
  _merge[(b * n,)](part, mx, sm, out, part.stride(0), part.stride(1), part.stride(2), out.stride(0), out.stride(1),
                   n, SPLITS=splits, BLOCK_S=triton.next_power_of_2(splits), D=dim, BLOCK_D=block_d, num_warps=4)
  return out


@decode_attention.register_fake
def _(q, k, v, lengths):
  return torch.empty_like(q)
