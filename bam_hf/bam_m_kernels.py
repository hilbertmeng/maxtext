"""Fused Triton kernels for BAM's per-token matrix reads and writes (the GPU analogue of the v8t core).

read:  Y_t[a] = X_t[a] M_t^T for the four arms a = q, k, v, o, where
       X_t[a][n, v] = 0.1 sigmoid(g_dyn[a, n]) rms(key_a[n, :])[v] + 1.1 sigmoid(g_stat[a, n]) / 0.99 S_a[v, n]
       (q, k use their own keys; v and o share the VO key). One program per token: M_t, raw keys and gate
       logits are read once, X never leaves registers, the contraction runs on tensor cores (tl.dot).
write: M'_t = M_t + sum_g C_g^T A_g with C_g = sigmoid(gate_g) rms(content_g) [N, K], A_g = rms(address_g) [N, V]
       for one or two groups (attention write, plus the MLP write on its layers).
Inputs may be strided views (e.g. column slices of a fused projection output). Accumulation is FP32.
"""
import torch
import triton
import triton.language as tl


def _prec(m):
  return 'ieee' if m.dtype == torch.float32 else 'tf32'     # tf32 is a no-op for bf16 operands


@triton.jit
def _read_kernel(M, KEYS, GATES, S, Y, T,
                 s_mt, s_kt, s_karm, s_kn, s_gt, s_y_t,
                 N: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                 EPS: tl.constexpr, SCALE: tl.constexpr, STATIC_DIV: tl.constexpr, PREC: tl.constexpr,
                 TB: tl.constexpr, STAGES: tl.constexpr):
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
    mkv = tl.load(M + t * s_mt + k[:, None] * V + v[None, :], mask=km[:, None] & vm[None, :], other=0.)
    key = tl.load(KEYS + t * s_kt + arm * s_karm + n[:, None] * s_kn + v[None, :],
                  mask=nm[:, None] & vm[None, :], other=0.).to(tl.float32)
    gd = tl.load(GATES + t * s_gt + a * N + n, mask=nm, other=0.).to(tl.float32)
    gs = tl.load(GATES + t * s_gt + (4 + a) * N + n, mask=nm, other=0.).to(tl.float32)
    inv = tl.rsqrt(tl.sum(key * key, axis=1) / V + EPS)
    x = (SCALE * tl.sigmoid(gd) * inv)[:, None] * key + (1.1 * tl.sigmoid(gs) / STATIC_DIV)[:, None] * st
    y = tl.dot(x.to(mkv.dtype), tl.trans(mkv), input_precision=PREC)     # [BN, BK] fp32
    tl.store(Y + t * s_y_t + a * N * K + n[:, None] * K + k[None, :], y.to(Y.dtype.element_ty),
             mask=nm[:, None] & km[None, :])


@triton.jit
def _read_kernel_stacked(M, KEYS, GATES, S, Y, T,
                         s_mt, s_kt, s_karm, s_kn, s_gt, s_y_t,
                         N: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                         BN: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                         EPS: tl.constexpr, SCALE: tl.constexpr, STATIC_DIV: tl.constexpr, PREC: tl.constexpr,
                         TB: tl.constexpr, STAGES: tl.constexpr):
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
    mkv = tl.load(M + t * s_mt + k[:, None] * V + v[None, :], mask=km[:, None] & vm[None, :], other=0.)
    key = tl.load(KEYS + t * s_kt + arm[:, None] * s_karm + n[:, None] * s_kn + v[None, :],
                  mask=nm[:, None] & vm[None, :], other=0.).to(tl.float32)
    gd = tl.load(GATES + t * s_gt + a * N + n, mask=nm, other=0.).to(tl.float32)
    gs = tl.load(GATES + t * s_gt + (4 + a) * N + n, mask=nm, other=0.).to(tl.float32)
    inv = tl.rsqrt(tl.sum(key * key, axis=1) / V + EPS)
    x = (SCALE * tl.sigmoid(gd) * inv)[:, None] * key + (1.1 * tl.sigmoid(gs) / STATIC_DIV)[:, None] * st
    y = tl.dot(x.to(mkv.dtype), tl.trans(mkv), input_precision=PREC)
    tl.store(Y + t * s_y_t + a[:, None] * N * K + n[:, None] * K + k[None, :], y.to(Y.dtype.element_ty),
             mask=nm[:, None] & km[None, :])


@torch.library.custom_op('bam::read', mutates_args=())
def bam_read(m: torch.Tensor, keys: torch.Tensor, gates: torch.Tensor, static_keys: torch.Tensor,
             eps: float, scale: float, static_div: float) -> torch.Tensor:
  """m [T,K,V] contiguous; keys [T,3,N,V] (q, k, vo; any token/arm/head strides, unit V stride);
  gates [T,8,N] logits (dyn q,k,v,o then static q,k,v,o; unit stride within the 8N block);
  static_keys [4,V,N] contiguous. Returns Y [T,4,N,K]."""
  t, kdim, vdim = m.shape
  n = keys.shape[2]
  assert m.is_contiguous() and static_keys.is_contiguous() and keys.stride(-1) == 1
  assert gates.stride(-1) == 1 and gates.stride(-2) == n
  y = torch.empty((t, 4, n, kdim), device=m.device, dtype=m.dtype)
  if not t:
    return y
  args = (m, keys, gates, static_keys, y, t, m.stride(0), keys.stride(0), keys.stride(1), keys.stride(2),
          gates.stride(0), y.stride(0))
  meta = dict(N=n, K=kdim, V=vdim, BN=max(16, triton.next_power_of_2(n)), BK=triton.next_power_of_2(kdim),
              BV=max(16, triton.next_power_of_2(vdim)), EPS=eps, SCALE=scale, STATIC_DIV=static_div, PREC=_prec(m))
  if t < 64:     # decode: one program per (arm, token) for latency (A800: 3.7 us at T=1)
    _read_kernel[(4, t)](*args, **meta, TB=1, STAGES=1, num_warps=4)
  else:          # prefill: four arms stacked into one product per token, M_t read once (A800: 131 us at T=4096)
    _read_kernel_stacked[(triton.cdiv(t, 4),)](*args, **meta, TB=4, STAGES=3, num_warps=4)
  return y


@bam_read.register_fake
def _(m, keys, gates, static_keys, eps, scale, static_div):
  return m.new_empty((m.shape[0], 4, keys.shape[2], m.shape[1]))


@triton.jit
def _group(C, G, A, t, s_ct, s_cn, s_gt, s_at, s_an, n, k, v, nm, km, vm, K: tl.constexpr, V: tl.constexpr,
           EPS: tl.constexpr):
  c = tl.load(C + t * s_ct + n[:, None] * s_cn + k[None, :], mask=nm[:, None] & km[None, :], other=0.).to(tl.float32)
  g = tl.load(G + t * s_gt + n, mask=nm, other=0.).to(tl.float32)
  c = c * (tl.sigmoid(g) * tl.rsqrt(tl.sum(c * c, axis=1) / K + EPS))[:, None]
  a = tl.load(A + t * s_at + n[:, None] * s_an + v[None, :], mask=nm[:, None] & vm[None, :], other=0.).to(tl.float32)
  a = a * tl.rsqrt(tl.sum(a * a, axis=1) / V + EPS)[:, None]
  return c, a


@triton.jit
def _write_kernel(M, OUT, C1, G1, A1, C2, G2, A2,
                  s_mt, s_c1t, s_c1n, s_g1t, s_a1t, s_a1n, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n,
                  N: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                  BV: tl.constexpr, TWO: tl.constexpr, EPS: tl.constexpr, PREC: tl.constexpr):
  t = tl.program_id(0)
  n = tl.arange(0, BN)
  k = tl.arange(0, BK)
  v = tl.arange(0, BV)
  nm, km, vm = n < N, k < K, v < V
  mask = km[:, None] & vm[None, :]
  acc = tl.load(M + t * s_mt + k[:, None] * V + v[None, :], mask=mask, other=0.).to(tl.float32)
  c, a = _group(C1, G1, A1, t, s_c1t, s_c1n, s_g1t, s_a1t, s_a1n, n, k, v, nm, km, vm, K, V, EPS)
  dt = OUT.dtype.element_ty
  acc = tl.dot(tl.trans(c.to(dt)), a.to(dt), acc, input_precision=PREC)
  if TWO:
    c, a = _group(C2, G2, A2, t, s_c2t, s_c2n, s_g2t, s_a2t, s_a2n, n, k, v, nm, km, vm, K, V, EPS)
    acc = tl.dot(tl.trans(c.to(dt)), a.to(dt), acc, input_precision=PREC)
  tl.store(OUT + t * s_mt + k[:, None] * V + v[None, :], acc.to(OUT.dtype.element_ty), mask=mask)


@torch.library.custom_op('bam::write', mutates_args=())
def bam_write(m: torch.Tensor, content: torch.Tensor, gate: torch.Tensor, address: torch.Tensor,
              content2: torch.Tensor, gate2: torch.Tensor, address2: torch.Tensor, two: bool,
              eps: float) -> torch.Tensor:
  """m [T,K,V] contiguous; content [T,N,K], gate [T,N] logits, address [T,N,V] raw (unit last stride);
  the second group is used iff `two`. Returns M + sum_g (sigmoid(gate) rms(content))^T rms(address)."""
  t, kdim, vdim = m.shape
  n = content.shape[1]
  assert m.is_contiguous()
  for x in (content, gate, address, content2, gate2, address2):
    assert x.stride(-1) == 1
  out = torch.empty_like(m)
  if t:
    _write_kernel[(t,)](m, out, content, gate, address, content2, gate2, address2,
                        m.stride(0), content.stride(0), content.stride(1), gate.stride(0), address.stride(0),
                        address.stride(1), content2.stride(0), content2.stride(1), gate2.stride(0),
                        address2.stride(0), address2.stride(1),
                        N=n, K=kdim, V=vdim, BN=max(16, triton.next_power_of_2(n)), BK=triton.next_power_of_2(kdim),
                        BV=max(16, triton.next_power_of_2(vdim)), TWO=two, EPS=eps, PREC=_prec(m), num_warps=4)
  return out


@bam_write.register_fake
def _(m, content, gate, address, content2, gate2, address2, two, eps):
  return torch.empty_like(m)
