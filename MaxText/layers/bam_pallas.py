"""Fused Pallas TPU kernels for the AllLocal DirectC10 BAM core.

Two kernels touch the per-token matrix state; everything else stays in XLA.

* ``read``: M -> (Q, K, V, LocalO). One MXU dot gives the four static reads
  and the C10 compression; three VPU contractions give the dynamic direct Q/K
  reads and the shared LocalVO read; RoPE'd standard QK coordinates are placed
  into the same output heads (no separate concatenate).
* ``write``: M + sum_n A_n (x) C_n for the attention write and, on MLP-write
  layers, the independent MLP write, in one pass over M.

Inside the kernels M is token-minor ``[V, K, T]`` (tokens on SIMD lanes, the
static/compression contraction axis V leading), and the model carries it in
that layout ``[B, V, K, T]`` between layers. Small per-token inputs/outputs are
token-minor ``[B, heads, width, T]``; wrappers convert from/to the token-major
XLA tensors. Reverses are analytic; no AD runs inside a kernel. Shared-parameter
gradients accumulate on chip per batch element across token tiles.

The per-tile functions are pure jnp and double as the reference implementation.
"""
from functools import partial
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32


# ---------------------------------------------------------------------------
# Small helpers (token axis last everywhere).

def _rows(x, start, stop, axis):
  return jax.lax.slice_in_dim(x, start, stop, axis=axis)


def _norm(x, epsilon, axis):
  f = x.astype(F32)
  return f * jax.lax.rsqrt(jnp.mean(f * f, axis=axis, keepdims=True) + epsilon)


def _norm_backward(x, cotangent, epsilon, axis):
  f = x.astype(F32)
  u = cotangent.astype(F32)
  inv = jax.lax.rsqrt(jnp.mean(f * f, axis=axis, keepdims=True) + epsilon)
  return (u - f * (jnp.mean(u * f, axis=axis, keepdims=True) * inv * inv)) * inv


def _sigmoid(x):
  return jax.nn.sigmoid(x.astype(F32))


def _static_dot(weight, m):
  """[J, V] x [V, K, T] -> [J, K, T] in FP32 on the MXU."""
  v, k, t = m.shape
  return jnp.dot(weight, m.reshape(v, k * t), preferred_element_type=F32).reshape(weight.shape[0], k, t)


def _static_dot_t(weight, d):
  """[J, V]^T x [J, K, T] -> [V, K, T] in FP32."""
  j, k, t = d.shape
  return jax.lax.dot_general(weight, d.reshape(j, k * t), (((0,), (0,)), ((), ())),
                             preferred_element_type=F32).reshape(weight.shape[1], k, t)


def _weight_grad(d, m):
  """sum_{k,t} d[J,k,t] m[V,k,t] -> [J, V] FP32."""
  j, k, t = d.shape
  return jax.lax.dot_general(d.reshape(j, k * t), m.reshape(m.shape[0], k * t),
                             (((1,), (1,)), ((), ())), preferred_element_type=F32)


def _dynamic_read(mc, key):
  """y[n,k,t] = sum_c key[n,c,t] mc[c,k,t]; never materializes [n,c,k,t]."""
  acc = None
  for c in range(mc.shape[0]):
    term = _rows(key, c, c + 1, 1).astype(F32) * _rows(mc, c, c + 1, 0).astype(F32)
    acc = term if acc is None else acc + term
  return acc


def _dynamic_read_backward(mc, key, dy):
  """Returns (d mc [c,k,t], d key [n,c,t]) for _dynamic_read in FP32."""
  dyf = dy.astype(F32)
  dmc, dkey = [], []
  for c in range(mc.shape[0]):
    kc = _rows(key, c, c + 1, 1).astype(F32)            # [n,1,t]
    mcc = _rows(mc, c, c + 1, 0).astype(F32)            # [1,k,t]
    dmc.append(jnp.sum(kc * dyf, axis=0, keepdims=True))   # [1,k,t]
    dkey.append(jnp.sum(dyf * mcc, axis=1, keepdims=True))  # [n,1,t]
  return jnp.concatenate(dmc, axis=0), jnp.concatenate(dkey, axis=1)


def _pad_cols(x, width):
  if x.shape[1] == width:
    return x
  return jnp.concatenate((x, jnp.zeros((x.shape[0], width - x.shape[1], x.shape[2]), x.dtype)), axis=1)


# ---------------------------------------------------------------------------
# Read: per-tile math.
#
#   m  [V,K,T]  carried state           sw [4N+C, V] = [S_q; S_k; S_v; S_o; P]^T
#   rq, rk [N,C,T] direct Q/K keys      lq, lk [N,T] their gate logits (bias added)
#   rr [N,C,T] shared LocalVO key       lo, lv [N,T] LocalO / LocalV gate logits
#   qs, ks [N,R,T] normalized, rotated standard QK coordinates (R = D_head - QKc)

def _read_keys(rq, lq, rk, lk, rr, read_epsilon, key_scale, dtype):
  gq = key_scale * _sigmoid(lq)
  gk = key_scale * _sigmoid(lk)
  nq = _norm(rq, read_epsilon, 1).astype(dtype)
  nk = _norm(rk, read_epsilon, 1).astype(dtype)
  nr = _norm(rr, read_epsilon, 1).astype(dtype)
  kq = (gq.astype(dtype)[:, None, :] * nq).astype(dtype)
  kk = (gk.astype(dtype)[:, None, :] * nk).astype(dtype)
  return kq, kk, nr, gq, gk, nq, nk


def read_tile(m, sw, rq, lq, rk, lk, rr, lo, lv, qs, ks, *, heads, qk_cols,
              read_epsilon, key_scale):
  dt = m.dtype
  n = heads
  st = _static_dot(sw.astype(dt), m).astype(dt)
  sq, sk, sv, so = (_rows(st, i * n, (i + 1) * n, 0) for i in range(4))
  mc = _rows(st, 4 * n, st.shape[0], 0)
  kq, kk, nr, *_ = _read_keys(rq, lq, rk, lk, rr, read_epsilon, key_scale, dt)
  mcq = _rows(mc, 0, qk_cols, 1)
  q_local = (_dynamic_read(mcq, kq).astype(dt) + _rows(sq, 0, qk_cols, 1)).astype(dt)
  k_local = (_dynamic_read(mcq, kk).astype(dt) + _rows(sk, 0, qk_cols, 1)).astype(dt)
  y = _dynamic_read(mc, nr).astype(dt)
  gv = (key_scale * _sigmoid(lv)).astype(dt)[:, None, :]
  go = (key_scale * _sigmoid(lo)).astype(dt)[:, None, :]
  value = ((y * gv).astype(dt) + sv).astype(dt)
  local_o = ((y * go).astype(dt) + so).astype(dt)
  query = jnp.concatenate((q_local, qs.astype(dt)), axis=1)
  key = jnp.concatenate((k_local, ks.astype(dt)), axis=1)
  return query, key, value, local_o


def read_tile_backward(m, sw, rq, lq, rk, lk, rr, lo, lv, qs, ks, dq, dk, dv, do, *,
                       heads, qk_cols, read_epsilon, key_scale):
  """Analytic reverse of read_tile. Returns (dm, dsw[FP32], drq, dlq, drk, dlk,
  drr, dlo, dlv, dqs, dks)."""
  dt = m.dtype
  n = heads
  kdim = m.shape[1]
  st = _static_dot(sw.astype(dt), m).astype(dt)
  mc = _rows(st, 4 * n, st.shape[0], 0)
  kq, kk, nr, gq, gk, nq, nk = _read_keys(rq, lq, rk, lk, rr, read_epsilon, key_scale, dt)
  y = _dynamic_read(mc, nr).astype(dt).astype(F32)
  sgv, sgo = _sigmoid(lv), _sigmoid(lo)

  dqf, dkf, dvf, dof = (z.astype(F32) for z in (dq, dk, dv, do))
  dyq = _rows(dqf, 0, qk_cols, 1)
  dyk = _rows(dkf, 0, qk_cols, 1)
  dqs = _rows(dqf, qk_cols, dqf.shape[1], 1)
  dks = _rows(dkf, qk_cols, dkf.shape[1], 1)

  gv = (key_scale * sgv)[:, None, :]
  go = (key_scale * sgo)[:, None, :]
  dy = dvf * gv + dof * go
  dlv = key_scale * sgv * (1 - sgv) * jnp.sum(dvf * y, axis=1)
  dlo = key_scale * sgo * (1 - sgo) * jnp.sum(dof * y, axis=1)

  mcq = _rows(mc, 0, qk_cols, 1)
  dmc_q, dkq = _dynamic_read_backward(mcq, kq, dyq)
  dmc_k, dkk = _dynamic_read_backward(mcq, kk, dyk)
  dmc, dnr = _dynamic_read_backward(mc, nr, dy)
  dmc = dmc + _pad_cols(dmc_q + dmc_k, kdim)

  # kq = gq * N(rq): gate and normalization reverses.
  dlq = key_scale * _sigmoid(lq) * (1 - _sigmoid(lq)) * jnp.sum(dkq * nq.astype(F32), axis=1)
  dlk = key_scale * _sigmoid(lk) * (1 - _sigmoid(lk)) * jnp.sum(dkk * nk.astype(F32), axis=1)
  drq = _norm_backward(rq, gq[:, None, :] * dkq, read_epsilon, 1)
  drk = _norm_backward(rk, gk[:, None, :] * dkk, read_epsilon, 1)
  drr = _norm_backward(rr, dnr, read_epsilon, 1)

  dstatic = jnp.concatenate((_pad_cols(dyq, kdim), _pad_cols(dyk, kdim), dvf, dof, dmc), axis=0)
  dm = _static_dot_t(sw.astype(F32), dstatic)
  dsw = _weight_grad(dstatic, m.astype(F32))
  return dm, dsw, drq, dlq, drk, dlk, drr, dlo, dlv, dqs, dks


# ---------------------------------------------------------------------------
# Write: per-tile math. Each write group is (content [N,K,T], gate logits
# [N,T], address [N,V,T]); the content is RMS-normalized over K, gated by
# sigmoid, the address RMS-normalized over V. dM[v,k] = sum_n A[n,v] C[n,k].

def _write_factors(content, logits, address, epsilon, dtype):
  gate = _sigmoid(logits).astype(dtype)
  c = (gate[:, None, :] * _norm(content, epsilon, 1).astype(dtype)).astype(dtype)
  a = _norm(address, epsilon, 1).astype(dtype)
  return c, a


def _outer_sum(cs, as_):
  acc = None
  for c, a in zip(cs, as_):
    for i in range(c.shape[0]):
      term = _rows(a, i, i + 1, 0).reshape(a.shape[1], 1, a.shape[2]).astype(F32) * _rows(c, i, i + 1, 0).astype(F32)
      acc = term if acc is None else acc + term
  return acc


def write_tile(m, *groups, epsilon):
  dt = m.dtype
  cs, as_ = [], []
  for g in range(0, len(groups), 3):
    c, a = _write_factors(*groups[g:g + 3], epsilon, dt)
    cs.append(c)
    as_.append(a)
  delta = _outer_sum(cs, as_).astype(dt)
  return (m.astype(F32) + delta.astype(F32)).astype(dt)


def write_tile_backward(dm_out, *groups, epsilon):
  """Returns per group (dcontent, dlogits, daddress); dM_in is dm_out itself."""
  dt = dm_out.dtype
  g = dm_out.astype(F32)                                   # [V,K,T]
  out = []
  for j in range(0, len(groups), 3):
    content, logits, address = groups[j:j + 3]
    c, a = _write_factors(content, logits, address, epsilon, dt)
    dc, da = [], []
    for i in range(c.shape[0]):
      ai = _rows(a, i, i + 1, 0).reshape(a.shape[1], 1, a.shape[2]).astype(F32)   # [V,1,T]
      ci = _rows(c, i, i + 1, 0).astype(F32)                                       # [1,K,T]
      dc.append(jnp.sum(g * ai, axis=0, keepdims=True))                            # [1,K,T]
      da.append(jnp.sum(g * ci, axis=1).reshape(1, a.shape[1], a.shape[2]))       # [1,V,T]
    dc = jnp.concatenate(dc, axis=0)
    da = jnp.concatenate(da, axis=0)
    sg = _sigmoid(logits)
    nc = _norm(content, epsilon, 1).astype(dt).astype(F32)
    dlogits = sg * (1 - sg) * jnp.sum(dc * nc, axis=1)
    dcontent = _norm_backward(content, sg.astype(dt).astype(F32)[:, None, :] * dc, epsilon, 1)
    daddress = _norm_backward(address, da, epsilon, 1)
    out += [dcontent, dlogits, daddress]
  return out


# ---------------------------------------------------------------------------
# Pallas plumbing.

def _spec(shape, tile):
  return pl.BlockSpec((None,) + tuple(shape) + (tile,), lambda b, i: (b,) + (0,) * len(shape) + (i,))


def _whole(shape):
  return pl.BlockSpec(tuple(shape), lambda b, i, n=len(shape): (0,) * n)


def _params(semantics, vmem_mib):
  kwargs = dict(dimension_semantics=semantics)
  if vmem_mib:
    kwargs['vmem_limit_bytes'] = int(vmem_mib) * 1024 * 1024
  return pltpu.CompilerParams(**kwargs)


def _tile(t, tile):
  tile = min(tile, t)
  if t % tile:
    raise ValueError(f'token tile {tile} must divide sequence length {t}')
  return tile


# ---------------------------------------------------------------------------
# Register-blocked kernel bodies. They compute exactly the per-tile functions
# above, but work through VMEM refs so that each accumulator block (a few
# [K,T] FP32 slabs) stays in vector registers while every loaded factor slab is
# reused across the block.

def _row(ref, *idx):
  """One sublane row [1,T] of a ref, indexed on its leading axes."""
  *lead, r = idx
  return ref[tuple(lead) + (pl.ds(r, 1), slice(None))]


def _fill_write_factors(group_refs, c_scr, a_scr, epsilon, dt):
  off = 0
  for g in range(0, len(group_refs), 3):
    content, logits, address = group_refs[g:g + 3]
    for i in range(content.shape[0]):
      gate = _sigmoid(_row(logits, i)).astype(dt)
      c_scr[off + i] = (gate * _norm(content[i], epsilon, 0).astype(dt)).astype(dt)
      a_scr[off + i] = _norm(address[i], epsilon, 0).astype(dt)
    off += content.shape[0]
  return off


def _write_kernel(epsilon, n_groups, vb):
  def kernel(m_ref, *refs):
    group_refs = refs[:3 * n_groups]
    out_ref, c_scr, a_scr = refs[3 * n_groups:]
    dt = m_ref.dtype
    ntot = _fill_write_factors(group_refs, c_scr, a_scr, epsilon, dt)
    for v0 in range(0, m_ref.shape[0], vb):
      rows = min(vb, m_ref.shape[0] - v0)
      accs = [None] * rows
      for i in range(ntot):
        ci = c_scr[i].astype(F32)
        arow = a_scr[i, pl.ds(v0, rows), :].astype(F32)
        for j in range(rows):
          term = arow[j:j + 1] * ci
          accs[j] = term if accs[j] is None else accs[j] + term
      for j in range(rows):
        out_ref[v0 + j] = (m_ref[v0 + j].astype(F32) + accs[j].astype(dt).astype(F32)).astype(dt)
  return kernel


def _write_backward_kernel(epsilon, n_groups, nb):
  def kernel(g_ref, *refs):
    group_refs = refs[:3 * n_groups]
    outs = refs[3 * n_groups:6 * n_groups]
    c_scr, a_scr, dc_scr, da_scr = refs[6 * n_groups:]
    dt = g_ref.dtype
    v_dim = g_ref.shape[0]
    ntot = _fill_write_factors(group_refs, c_scr, a_scr, epsilon, dt)
    # dC[n,k] = sum_v A[n,v] G[v,k]: head blocks share every loaded G row slab.
    for n0 in range(0, ntot, nb):
      heads = min(nb, ntot - n0)
      accs = [None] * heads
      for v in range(v_dim):
        gv = g_ref[v].astype(F32)
        for j in range(heads):
          term = _row(a_scr, n0 + j, v).astype(F32) * gv
          accs[j] = term if accs[j] is None else accs[j] + term
      for j in range(heads):
        dc_scr[n0 + j] = accs[j]
    # dA[n,v] = sum_k G[v,k] C[n,k].
    for i in range(ntot):
      ci = c_scr[i].astype(F32)
      for v in range(v_dim):
        da_scr[i, pl.ds(v, 1), :] = jnp.sum(g_ref[v].astype(F32) * ci, axis=0, keepdims=True)
    off = 0
    for g in range(n_groups):
      content, logits, address = group_refs[3 * g:3 * g + 3]
      dcontent, dlogits, daddress = outs[3 * g:3 * g + 3]
      for i in range(content.shape[0]):
        sg = _sigmoid(_row(logits, i))
        nc = _norm(content[i], epsilon, 0).astype(dt).astype(F32)
        dci = dc_scr[off + i]
        dlogits[pl.ds(i, 1), :] = (sg * (1 - sg) * jnp.sum(dci * nc, axis=0, keepdims=True)).astype(dlogits.dtype)
        dcontent[i] = _norm_backward(content[i], sg.astype(dt).astype(F32) * dci, epsilon, 0).astype(dcontent.dtype)
        daddress[i] = _norm_backward(address[i], da_scr[off + i], epsilon, 0).astype(daddress.dtype)
      off += content.shape[0]
  return kernel


def _fill_read_state(m_ref, sw_ref, rq, lq, rk, lk, rr, st_scr, key_scr, read_epsilon, key_scale):
  dt = m_ref.dtype
  st_scr[...] = _static_dot(sw_ref[...].astype(dt), m_ref[...]).astype(dt)
  kq, kk, nr, *_ = _read_keys(rq[...], lq[...], rk[...], lk[...], rr[...], read_epsilon, key_scale, dt)
  key_scr[0] = kq
  key_scr[1] = kk
  key_scr[2] = nr


def _dynamic_block(st_scr, key_scr, which, mc0, n0, heads, cols):
  """FP32 accumulators y[n0+j, :cols] = sum_c key[which, n0+j, c] mc[c, :cols]."""
  accs = [None] * heads
  for c in range(key_scr.shape[2]):
    mcc = st_scr[mc0 + c, pl.ds(0, cols), :].astype(F32)
    for j in range(heads):
      term = _row(key_scr, which, n0 + j, c).astype(F32) * mcc
      accs[j] = term if accs[j] is None else accs[j] + term
  return accs


def _read_kernel(heads, qk_cols, read_epsilon, key_scale, nb):
  def kernel(m_ref, sw_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, q_ref, k_ref, v_ref, o_ref,
             st_scr, key_scr):
    dt = m_ref.dtype
    n = heads
    mc0 = 4 * n
    _fill_read_state(m_ref, sw_ref, rq, lq, rk, lk, rr, st_scr, key_scr, read_epsilon, key_scale)
    for which, out_ref, std_ref in ((0, q_ref, qs), (1, k_ref, ks)):
      for n0 in range(0, n, nb):
        hb = min(nb, n - n0)
        accs = _dynamic_block(st_scr, key_scr, which, mc0, n0, hb, qk_cols)
        for j in range(hb):
          h = n0 + j
          static = st_scr[which * n + h, pl.ds(0, qk_cols), :]
          out_ref[h, pl.ds(0, qk_cols), :] = (accs[j].astype(dt) + static).astype(dt)
          out_ref[h, pl.ds(qk_cols, out_ref.shape[1] - qk_cols), :] = std_ref[h].astype(dt)
    kdim = m_ref.shape[1]
    for n0 in range(0, n, nb):
      hb = min(nb, n - n0)
      accs = _dynamic_block(st_scr, key_scr, 2, mc0, n0, hb, kdim)
      for j in range(hb):
        h = n0 + j
        y = accs[j].astype(dt)
        gv = (key_scale * _sigmoid(_row(lv, h))).astype(dt)
        go = (key_scale * _sigmoid(_row(lo, h))).astype(dt)
        v_ref[h] = ((y * gv).astype(dt) + st_scr[2 * n + h]).astype(dt)
        o_ref[h] = ((y * go).astype(dt) + st_scr[3 * n + h]).astype(dt)
  return kernel


def _read_backward_kernel(heads, qk_cols, read_epsilon, key_scale):
  def kernel(m_ref, sw_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, dq, dk, dv, do,
             dm_ref, dsw_ref, drq, dlq, drk, dlk, drr, dlo, dlv, dqs, dks,
             st_scr, key_scr, d_scr, dkey_scr):
    dt = m_ref.dtype
    n = heads
    mc0 = 4 * n
    kdim = m_ref.shape[1]
    cdim = key_scr.shape[2]
    _fill_read_state(m_ref, sw_ref, rq, lq, rk, lk, rr, st_scr, key_scr, read_epsilon, key_scale)
    for c in range(cdim):
      d_scr[mc0 + c] = jnp.zeros((kdim, d_scr.shape[2]), F32)
    rest = kdim - qk_cols
    for which, d_ref, ds_ref in ((0, dq, dqs), (1, dk, dks)):
      for h in range(n):
        dy = d_ref[h, pl.ds(0, qk_cols), :].astype(F32)
        d_scr[which * n + h, pl.ds(0, qk_cols), :] = dy
        d_scr[which * n + h, pl.ds(qk_cols, rest), :] = jnp.zeros((rest, dy.shape[1]), F32)
        ds_ref[h] = d_ref[h, pl.ds(qk_cols, rest), :].astype(ds_ref.dtype)
        for c in range(cdim):
          mcc = st_scr[mc0 + c, pl.ds(0, qk_cols), :].astype(F32)
          dkey_scr[which, h, pl.ds(c, 1), :] = jnp.sum(dy * mcc, axis=0, keepdims=True)
          d_scr[mc0 + c, pl.ds(0, qk_cols), :] += _row(key_scr, which, h, c).astype(F32) * dy
    for h in range(n):
      dvh = dv[h].astype(F32)
      doh = do[h].astype(F32)
      d_scr[2 * n + h] = dvh
      d_scr[3 * n + h] = doh
      y = None
      for c in range(cdim):
        term = _row(key_scr, 2, h, c).astype(F32) * st_scr[mc0 + c].astype(F32)
        y = term if y is None else y + term
      y = y.astype(dt).astype(F32)
      sgv, sgo = _sigmoid(_row(lv, h)), _sigmoid(_row(lo, h))
      dy = dvh * (key_scale * sgv) + doh * (key_scale * sgo)
      dlv[pl.ds(h, 1), :] = (key_scale * sgv * (1 - sgv) * jnp.sum(dvh * y, axis=0, keepdims=True)).astype(dlv.dtype)
      dlo[pl.ds(h, 1), :] = (key_scale * sgo * (1 - sgo) * jnp.sum(doh * y, axis=0, keepdims=True)).astype(dlo.dtype)
      for c in range(cdim):
        mcc = st_scr[mc0 + c].astype(F32)
        dkey_scr[2, h, pl.ds(c, 1), :] = jnp.sum(dy * mcc, axis=0, keepdims=True)
        d_scr[mc0 + c] += _row(key_scr, 2, h, c).astype(F32) * dy
    # Key gate / normalization reverses on the small [N,C,T] tensors.
    for idx, (r, l, dr, dl) in enumerate(((rq, lq, drq, dlq), (rk, lk, drk, dlk))):
      sg = _sigmoid(l[...])
      nrm = _norm(r[...], read_epsilon, 1).astype(dt).astype(F32)
      dkey = dkey_scr[idx]
      dl[...] = (key_scale * sg * (1 - sg) * jnp.sum(dkey * nrm, axis=1)).astype(dl.dtype)
      dr[...] = _norm_backward(r[...], (key_scale * sg)[:, None, :] * dkey, read_epsilon, 1).astype(dr.dtype)
    drr[...] = _norm_backward(rr[...], dkey_scr[2], read_epsilon, 1).astype(drr.dtype)
    d = d_scr[...]
    dm_ref[...] = _static_dot_t(sw_ref[...].astype(F32), d).astype(dm_ref.dtype)

    @pl.when(pl.program_id(1) == 0)
    def _():
      dsw_ref[...] = jnp.zeros(dsw_ref.shape, dsw_ref.dtype)
    dsw_ref[...] += _weight_grad(d, m_ref[...].astype(F32))
  return kernel


_READ_TOKEN_ARGS = (0, 2, 3, 4, 5, 6, 7, 8, 9, 10)   # indices with a token axis


def _read_specs(args, tile):
  specs = []
  for i, x in enumerate(args):
    specs.append(_spec(x.shape[1:-1], tile) if i in _READ_TOKEN_ARGS else _whole(x.shape))
  return specs


def _read_forward_call(args, opts):
  m = args[0]
  b, v, k, t = m.shape
  n, qk_cols = opts['heads'], opts['qk_cols']
  tile = _tile(t, opts['forward_tile'])
  kw = dict(heads=n, qk_cols=qk_cols, read_epsilon=opts['read_epsilon'], key_scale=opts['key_scale'])
  scratch = ()
  if opts['body'] == 'tile':
    def kernel(*refs):
      outs = read_tile(*(r[...] for r in refs[:11]), **kw)
      for ref, val in zip(refs[11:], outs):
        ref[...] = val
  else:
    kernel = _read_kernel(n, qk_cols, opts['read_epsilon'], opts['key_scale'], opts['head_block'])
    c = args[2].shape[2]
    scratch = [pltpu.VMEM((args[1].shape[0], k, tile), m.dtype), pltpu.VMEM((3, n, c, tile), m.dtype)]

  shape = (b, n, k, t)
  return pl.pallas_call(
      kernel, grid=(b, t // tile), in_specs=_read_specs(args, tile),
      out_specs=[_spec((n, k), tile)] * 4, scratch_shapes=scratch,
      out_shape=tuple(jax.ShapeDtypeStruct(shape, m.dtype) for _ in range(4)),
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read')(*args)


def _read_backward_call(args, cts, opts):
  m, sw = args[0], args[1]
  b, v, k, t = m.shape
  n, qk_cols = opts['heads'], opts['qk_cols']
  tile = _tile(t, opts['reverse_tile'])
  kw = dict(heads=n, qk_cols=qk_cols, read_epsilon=opts['read_epsilon'], key_scale=opts['key_scale'])
  token_outs = [(m.shape[1:-1], m.dtype)] + [
      (args[i].shape[1:-1], args[i].dtype) for i in (2, 3, 4, 5, 6, 7, 8, 9, 10)]

  scratch = ()
  if opts['body'] == 'tile':
    def kernel(*refs):
      ins, grads = refs[:15], refs[15:]
      res = read_tile_backward(*(r[...] for r in ins), **kw)
      dm, dsw, rest = res[0], res[1], res[2:]
      grads[0][...] = dm.astype(grads[0].dtype)
      for ref, val in zip(grads[2:], rest):
        ref[...] = val.astype(ref.dtype)

      @pl.when(pl.program_id(1) == 0)
      def _():
        grads[1][...] = jnp.zeros(grads[1].shape, grads[1].dtype)
      grads[1][...] += dsw
  else:
    kernel = _read_backward_kernel(n, qk_cols, opts['read_epsilon'], opts['key_scale'])
    c = args[2].shape[2]
    j = sw.shape[0]
    scratch = [pltpu.VMEM((j, k, tile), m.dtype), pltpu.VMEM((3, n, c, tile), m.dtype),
               pltpu.VMEM((j, k, tile), F32), pltpu.VMEM((3, n, c, tile), F32)]

  ct_specs = [_spec((n, k), tile)] * 4
  out_specs = ([_spec(token_outs[0][0], tile),
                pl.BlockSpec((None,) + sw.shape, lambda bb, i: (bb, 0, 0))]
               + [_spec(s, tile) for s, _ in token_outs[1:]])
  out_shape = ([jax.ShapeDtypeStruct(m.shape, m.dtype), jax.ShapeDtypeStruct((b,) + sw.shape, F32)]
               + [jax.ShapeDtypeStruct((b,) + s + (t,), d) for s, d in token_outs[1:]])
  outs = pl.pallas_call(
      kernel, grid=(b, t // tile), in_specs=_read_specs(args, tile) + ct_specs,
      out_specs=out_specs, out_shape=tuple(out_shape), scratch_shapes=scratch,
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'arbitrary'), opts['vmem_mib']),
      name='bam_core_read_backward')(*args, *cts)
  dm, dsw = outs[0], jnp.sum(outs[1], axis=0).astype(sw.dtype)
  return (dm, dsw) + tuple(outs[2:])


@partial(jax.custom_vjp, nondiff_argnums=(11,))
def _read(m, sw, rq, lq, rk, lk, rr, lo, lv, qs, ks, opts):
  return _read_forward_call((m, sw, rq, lq, rk, lk, rr, lo, lv, qs, ks), dict(opts))


def _read_fwd(m, sw, rq, lq, rk, lk, rr, lo, lv, qs, ks, opts):
  args = (m, sw, rq, lq, rk, lk, rr, lo, lv, qs, ks)
  return _read_forward_call(args, dict(opts)), args


def _read_bwd(opts, args, cts):
  return _read_backward_call(args, cts, dict(opts))


_read.defvjp(_read_fwd, _read_bwd)


def _write_forward_call(m, groups, opts):
  b, v, k, t = m.shape
  tile = _tile(t, opts['forward_tile'])
  eps = opts['epsilon']
  ntot = sum(x.shape[1] for x in groups[::3])
  scratch = ()
  if opts['body'] == 'tile':
    def kernel(*refs):
      refs[-1][...] = write_tile(*(r[...] for r in refs[:-1]), epsilon=eps)
  else:
    kernel = _write_kernel(eps, len(groups) // 3, opts['row_block'])
    scratch = [pltpu.VMEM((ntot, k, tile), m.dtype), pltpu.VMEM((ntot, v, tile), m.dtype)]

  in_specs = [_spec((v, k), tile)] + [_spec(x.shape[1:-1], tile) for x in groups]
  return pl.pallas_call(
      kernel, grid=(b, t // tile), in_specs=in_specs, out_specs=_spec((v, k), tile),
      out_shape=jax.ShapeDtypeStruct(m.shape, m.dtype), scratch_shapes=scratch,
      input_output_aliases={0: 0},
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_write')(m, *groups)


def _write_backward_call(g, groups, opts):
  b, v, k, t = g.shape
  tile = _tile(t, opts['reverse_tile'])
  eps = opts['epsilon']
  ntot = sum(x.shape[1] for x in groups[::3])
  scratch = ()
  if opts['body'] == 'tile':
    def kernel(*refs):
      nin = 1 + len(groups)
      res = write_tile_backward(*(r[...] for r in refs[:nin]), epsilon=eps)
      for ref, val in zip(refs[nin:], res):
        ref[...] = val.astype(ref.dtype)
  else:
    kernel = _write_backward_kernel(eps, len(groups) // 3, opts['head_block'])
    scratch = [pltpu.VMEM((ntot, k, tile), g.dtype), pltpu.VMEM((ntot, v, tile), g.dtype),
               pltpu.VMEM((ntot, k, tile), F32), pltpu.VMEM((ntot, v, tile), F32)]

  in_specs = [_spec((v, k), tile)] + [_spec(x.shape[1:-1], tile) for x in groups]
  return pl.pallas_call(
      kernel, grid=(b, t // tile), in_specs=in_specs, scratch_shapes=scratch,
      out_specs=[_spec(x.shape[1:-1], tile) for x in groups],
      out_shape=tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in groups),
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_write_backward')(g, *groups)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _write(m, opts, *groups):
  return _write_forward_call(m, groups, dict(opts))


def _write_fwd(m, opts, *groups):
  return _write_forward_call(m, groups, dict(opts)), groups


def _write_bwd(opts, groups, g):
  return (g,) + tuple(_write_backward_call(g, groups, dict(opts)))


_write.defvjp(_write_fwd, _write_bwd)


# ---------------------------------------------------------------------------
# Public wrappers (token-major in/out except M; batch-sharded via shard_map).

def _freeze(d):
  return tuple(sorted(d.items()))


def _map_batch(fn, args, batch_args, n_out):
  from jax._src import mesh as mesh_lib
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if not mesh.axis_names or mesh.size == 1:
    return fn(*args)
  from flax import linen as nn
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P
  axes = nn.logical_to_mesh_axes(('activation_batch',))[0]
  names = (axes,) if isinstance(axes, str) else (axes or ())
  if not names or any(size > 1 and name not in names for name, size in mesh.shape.items()):
    raise ValueError('BAM Pallas core supports batch/FSDP mesh axes only')
  spec = P(axes)
  out_specs = spec if n_out == 1 else (spec,) * n_out
  return shard_map(fn, mesh=mesh, in_specs=tuple(spec if b else P() for b in batch_args),
                   out_specs=out_specs, check_rep=False)(*args)


def _minor(x):
  """[B,T,...] -> [B,...,T]."""
  return jnp.moveaxis(x, 1, -1)


def _major(x):
  return jnp.moveaxis(x, -1, 1)


def read(m, static_weight, q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits,
         q_standard, k_standard, *, qk_cols, read_epsilon, key_scale,
         forward_tile=128, reverse_tile=128, vmem_mib=None, interpret=False,
         body='blocked', head_block=4):
  """m [B,V,K,T]; keys [B,T,N,C]; logits [B,T,N]; standard QK [B,T,N,R].

  Returns token-major (query, key, value, local_o), each [B,T,N,K]."""
  heads = q_key.shape[2]
  opts = _freeze(dict(heads=heads, qk_cols=qk_cols, read_epsilon=read_epsilon,
                      key_scale=key_scale, forward_tile=forward_tile,
                      reverse_tile=reverse_tile, vmem_mib=vmem_mib, interpret=interpret,
                      body=body, head_block=head_block))

  def local(m, sw, *xs):
    xs = tuple(_minor(x) for x in xs)
    outs = _read(m, sw, *xs, opts)
    return tuple(_major(o) for o in outs)

  args = (m, static_weight, q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits,
          q_standard, k_standard)
  return _map_batch(local, args, (True, False) + (True,) * 9, 4)


def write(m, groups, *, epsilon, forward_tile=128, reverse_tile=128, vmem_mib=None,
          interpret=False, body='blocked', row_block=4, head_block=4):
  """m [B,V,K,T]; groups: sequence of (content [B,T,N,K], logits [B,T,N], address [B,T,N,V])."""
  opts = _freeze(dict(epsilon=epsilon, forward_tile=forward_tile, reverse_tile=reverse_tile,
                      vmem_mib=vmem_mib, interpret=interpret, body=body,
                      row_block=row_block, head_block=head_block))
  flat = tuple(x for grp in groups for x in grp)

  def local(m, *xs):
    return _write(m, opts, *(_minor(x) for x in xs))

  return _map_batch(local, (m,) + flat, (True,) * (1 + len(flat)), 1)


def static_weight(static_q, static_k, static_v, static_o, compression):
  """Stack [V,N] static keys and the [V,C] compression into the [4N+C, V] read weight."""
  return jnp.concatenate((static_q, static_k, static_v, static_o, compression), axis=1).T


def reference_read(m, sw, *xs, heads, qk_cols, read_epsilon, key_scale):
  """Pure-jnp token-minor reference over [B,...,T] arrays (same math as the kernel)."""
  return jax.vmap(lambda m, *x: read_tile(m, sw, *x, heads=heads, qk_cols=qk_cols,
                                          read_epsilon=read_epsilon, key_scale=key_scale))(m, *xs)


def reference_write(m, *groups, epsilon):
  return jax.vmap(lambda m, *g: write_tile(m, *g, epsilon=epsilon))(m, *groups)
