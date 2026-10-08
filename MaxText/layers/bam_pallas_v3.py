"""v3 fused BAM core: FP32 VMEM staging, reduction-free contractions, token-major big I/O.

Same equations as layers/bam_pallas.py (whose per-tile functions remain the reference).
Design rules, from v5p bundle analysis of the earlier bodies:

* BF16 HBM tiles (M, cotangents, write content) are unpacked once per tile into FP32 VMEM
  scratch. Mosaic supports sublane-strided access only for 32-bit data, so every kernel
  can then use both the k-major view (stride K) and the head-major view of one buffer.
* Every contraction is a broadcast-multiply-accumulate of register-resident slabs over a
  leading index; no sublane reductions inside contraction loops. Accumulator blocks are
  sized to stay well inside the 64 vector registers.
* Q/K/V/LocalO, their cotangents and the write content enter/leave token-major
  [T, N*K] through in-kernel 128x128 transposes, so XLA needs no transposes around them.

HBM layouts per batch element: M [V*K, T] (the carried [V,K,T]); keys [C,N,T];
logits [N,T]; standard QK [N*R,T]; Q/K/V/LocalO and cotangents [T, N*K];
write content [T, N*K]; write address [N,V,T].
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from layers.bam_pallas import (F32, _cmajor_keys, _freeze, _map_batch, _minor, _norm, _norm_backward,
                               _pad_rows, _padded_static_weight, _params, _row, _sigmoid, _spec, _tile,
                               _whole)


def _tmajor_spec(width, tile):
  return pl.BlockSpec((None, tile, width), lambda b, i: (b, i, 0))


def _load_transposed(src_ref, dst, row0, width):
  """Token-major BF16 block [t, width] -> FP32 rows dst[row0:row0+width] as [width, t]."""
  for j in range(0, width, 128):
    w = min(128, width - j)
    dst[pl.ds(row0 + j, w), :] = src_ref[:, pl.ds(j, w)].astype(F32).T


def _store_transposed(src, row0, dst_ref, width):
  """FP32 rows src[row0:row0+width] ([width, t]) -> token-major dst_ref [t, width]."""
  for j in range(0, width, 128):
    w = min(128, width - j)
    dst_ref[:, pl.ds(j, w)] = src[pl.ds(row0 + j, w), :].T.astype(dst_ref.dtype)


def _acc(a, b):
  return b if a is None else a + b


# ---------------------------------------------------------------------------
# Read.

def _read_kernel(heads, qk_cols, read_epsilon, key_scale):
  def kernel(m_ref, sw_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, q_ref, k_ref, v_ref, o_ref,
             mf, key_scr, out_scr):
    dt = m_ref.dtype
    n = heads
    np_ = _pad_rows(n)
    nk = q_ref.shape[1]
    kdim = nk // n
    vdim = m_ref.shape[0] // kdim
    cdim = rq.shape[0]
    rdim = qs.shape[0] // n
    mc0 = 4 * np_
    mf[...] = m_ref[...].astype(F32)
    kq, kk, nr, *_ = _cmajor_keys(rq[...], lq[...], rk[...], lk[...], rr[...], read_epsilon, key_scale, dt)
    key_scr[0] = kq.astype(F32)
    key_scr[1] = kk.astype(F32)
    key_scr[2] = nr.astype(F32)
    gv = (key_scale * _sigmoid(lv[...])).astype(dt).astype(F32)
    go = (key_scale * _sigmoid(lo[...])).astype(dt).astype(F32)
    sw = sw_ref[...].astype(dt)
    rnd = lambda x: x.astype(dt).astype(F32)
    for k in range(kdim):
      mk = mf[pl.ds(k, vdim, stride=kdim), :].astype(dt)
      st = rnd(jnp.dot(sw, mk, preferred_element_type=F32))
      dyn = k < qk_cols
      acc_v = acc_q = acc_k = None
      for c in range(cdim):
        row = st[mc0 + c:mc0 + c + 1]
        acc_v = _acc(acc_v, key_scr[2, c] * row)
        if dyn:
          acc_q = _acc(acc_q, key_scr[0, c] * row)
          acc_k = _acc(acc_k, key_scr[1, c] * row)
      rows = pl.ds(k, n, stride=kdim)
      y = rnd(acc_v)
      out_scr[2, rows, :] = rnd(rnd(y * gv) + st[2 * np_:2 * np_ + n])
      out_scr[3, rows, :] = rnd(rnd(y * go) + st[3 * np_:3 * np_ + n])
      if dyn:
        out_scr[0, rows, :] = rnd(rnd(acc_q) + st[0:n])
        out_scr[1, rows, :] = rnd(rnd(acc_k) + st[np_:np_ + n])
    for h in range(n):
      out_scr[0, pl.ds(h * kdim + qk_cols, rdim), :] = qs[pl.ds(h * rdim, rdim), :].astype(F32)
      out_scr[1, pl.ds(h * kdim + qk_cols, rdim), :] = ks[pl.ds(h * rdim, rdim), :].astype(F32)
    for i, ref in enumerate((q_ref, k_ref, v_ref, o_ref)):
      _store_transposed(out_scr.at[i], 0, ref, nk)
  return kernel


def _read_backward_kernel(heads, qk_cols, read_epsilon, key_scale, cb):
  def kernel(m_ref, sw_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, dq, dk, dv, do,
             dm_ref, dsw_ref, drq, dlq, drk, dlk, drr, dlo, dlv, dqs, dks,
             mf, key_scr, ct, mc_scr, dyvo, dmc_scr, d_scr, dm_scr):
    dt = m_ref.dtype
    n = heads
    np_ = _pad_rows(n)
    nk = dq.shape[1]
    kdim = nk // n
    vdim = m_ref.shape[0] // kdim
    cdim = rq.shape[0]
    rdim = qs.shape[0] // n
    rest = kdim - qk_cols
    mc0 = 4 * np_
    rnd = lambda x: x.astype(dt).astype(F32)
    mf[...] = m_ref[...].astype(F32)
    kq, kk, nr, gq, gk, nq, nkn = _cmajor_keys(rq[...], lq[...], rk[...], lk[...], rr[...],
                                               read_epsilon, key_scale, dt)
    key_scr[0] = kq.astype(F32)
    key_scr[1] = kk.astype(F32)
    key_scr[2] = nr.astype(F32)
    for i, ref in enumerate((dq, dk, dv, do)):
      _load_transposed(ref, ct.at[i], 0, nk)
    sgv, sgo = _sigmoid(lv[...]), _sigmoid(lo[...])
    gvf, gof = key_scale * sgv, key_scale * sgo
    sw = sw_ref[...].astype(dt)
    # Pass 1 (k-major): static/compressed rows, VO forward, VO cotangent and its key gradient.
    dlv_acc = dlo_acc = None
    dkv = [None] * cdim
    for k in range(kdim):
      mk = mf[pl.ds(k, vdim, stride=kdim), :].astype(dt)
      st = rnd(jnp.dot(sw, mk, preferred_element_type=F32))
      mc = st[mc0:mc0 + cdim]
      mc_scr[k] = mc
      y = None
      for c in range(cdim):
        y = _acc(y, key_scr[2, c] * mc[c:c + 1])
      y = rnd(y)
      rows = pl.ds(k, n, stride=kdim)
      dvk = ct[2, rows, :]
      dok = ct[3, rows, :]
      dlv_acc = _acc(dlv_acc, dvk * y)
      dlo_acc = _acc(dlo_acc, dok * y)
      dy = dvk * gvf + dok * gof
      dyvo[rows, :] = dy
      for c in range(cdim):
        dkv[c] = _acc(dkv[c], dy * mc[c:c + 1])
    dlv[...] = (key_scale * sgv * (1 - sgv) * dlv_acc).astype(dlv.dtype)
    dlo[...] = (key_scale * sgo * (1 - sgo) * dlo_acc).astype(dlo.dtype)
    # Pass 2 (k-major): direct Q/K key gradients; RoPE passthrough (head-major).
    dkeys = []
    for i, ds_ref in ((0, dqs), (1, dks)):
      acc = [None] * cdim
      for k in range(qk_cols):
        dyk = ct[i, pl.ds(k, n, stride=kdim), :]
        mc = mc_scr[k]
        for c in range(cdim):
          acc[c] = _acc(acc[c], dyk * mc[c:c + 1])
      dkeys.append(jnp.stack(acc))
      for h in range(n):
        ds_ref[pl.ds(h * rdim, rdim), :] = ct[i, pl.ds(h * kdim + qk_cols, rdim), :].astype(ds_ref.dtype)
    # Pass 3 (head-major): dMc[c] = sum_n key[n,c] dy[n], register accumulators per c block.
    for c0 in range(0, cdim, cb):
      cs = list(range(c0, min(c0 + cb, cdim)))
      full = {c: None for c in cs}
      part = {c: None for c in cs}
      for h in range(n):
        dyq = ct[0, pl.ds(h * kdim, qk_cols), :]
        dyk = ct[1, pl.ds(h * kdim, qk_cols), :]
        dyv = dyvo[pl.ds(h * kdim, kdim), :]
        for c in cs:
          part[c] = _acc(part[c], _row(key_scr, 0, c, h) * dyq + _row(key_scr, 1, c, h) * dyk)
          full[c] = _acc(full[c], _row(key_scr, 2, c, h) * dyv)
      for c in cs:
        dmc_scr[pl.ds(c * kdim, qk_cols), :] = full[c][:qk_cols] + part[c]
        dmc_scr[pl.ds(c * kdim + qk_cols, rest), :] = full[c][qk_cols:]
    # Key gate / normalization reverses over [C, N, T].
    for r, l, dr, dl, g, nrm, dkey in ((rq, lq, drq, dlq, gq, nq, dkeys[0]), (rk, lk, drk, dlk, gk, nkn, dkeys[1])):
      sg = _sigmoid(l[...])
      dl[...] = (key_scale * sg * (1 - sg) * jnp.sum(dkey * nrm.astype(F32), axis=0)).astype(dl.dtype)
      dr[...] = _norm_backward(r[...], g[None] * dkey, read_epsilon, 0).astype(dr.dtype)
    drr[...] = _norm_backward(rr[...], jnp.stack(dkv), read_epsilon, 0).astype(drr.dtype)
    # Pass 4 (k-major): static-read reverse, one small dot pair per key row.
    d_scr[...] = jnp.zeros(d_scr.shape, F32)
    dsw = None
    for k in range(kdim):
      rows = pl.ds(k, n, stride=kdim)
      if k < qk_cols:
        d_scr[pl.ds(0, n), :] = ct[0, rows, :]
        d_scr[pl.ds(np_, n), :] = ct[1, rows, :]
      elif k == qk_cols:
        d_scr[pl.ds(0, n), :] = jnp.zeros((n, d_scr.shape[1]), F32)
        d_scr[pl.ds(np_, n), :] = jnp.zeros((n, d_scr.shape[1]), F32)
      d_scr[pl.ds(2 * np_, n), :] = ct[2, rows, :]
      d_scr[pl.ds(3 * np_, n), :] = ct[3, rows, :]
      d_scr[pl.ds(mc0, cdim), :] = dmc_scr[pl.ds(k, cdim, stride=kdim), :]
      d = d_scr[...].astype(dt)
      dm_scr[pl.ds(k, vdim, stride=kdim), :] = jax.lax.dot_general(
          sw, d, (((0,), (0,)), ((), ())), preferred_element_type=F32)
      mk = mf[pl.ds(k, vdim, stride=kdim), :].astype(dt)
      dsw = _acc(dsw, jax.lax.dot_general(d, mk, (((1,), (1,)), ((), ())), preferred_element_type=F32))
    dm_ref[...] = dm_scr[...].astype(dm_ref.dtype)

    @pl.when(pl.program_id(1) == 0)
    def _():
      dsw_ref[...] = jnp.zeros(dsw_ref.shape, dsw_ref.dtype)
    dsw_ref[...] += dsw
  return kernel


def _read_specs(args, tile):
  m, sw = args[0], args[1]
  return [_spec(m.shape[1:-1], tile), _whole(sw.shape)] + [_spec(x.shape[1:-1], tile) for x in args[2:]]


def _read_forward_call(args, opts):
  m = args[0]
  b, vk, t = m.shape
  n, kdim = opts['heads'], opts['k_dim']
  c = args[2].shape[1]
  tile = _tile(t, opts['forward_tile'])
  nk = n * kdim
  return pl.pallas_call(
      _read_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale']),
      grid=(b, t // tile), in_specs=_read_specs(args, tile),
      out_specs=[_tmajor_spec(nk, tile)] * 4,
      out_shape=tuple(jax.ShapeDtypeStruct((b, t, nk), m.dtype) for _ in range(4)),
      scratch_shapes=[pltpu.VMEM((vk, tile), F32), pltpu.VMEM((3, c, n, tile), F32),
                      pltpu.VMEM((4, nk, tile), F32)],
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read')(*args)


def _read_backward_call(args, cts, opts):
  m, sw = args[0], args[1]
  b, vk, t = m.shape
  n, kdim = opts['heads'], opts['k_dim']
  c = args[2].shape[1]
  nk = n * kdim
  tile = _tile(t, opts['reverse_tile'])
  out_specs = ([_spec((vk,), tile), pl.BlockSpec((None,) + sw.shape, lambda bb, i: (bb, 0, 0))]
               + [_spec(x.shape[1:-1], tile) for x in args[2:]])
  out_shape = ([jax.ShapeDtypeStruct(m.shape, m.dtype), jax.ShapeDtypeStruct((b,) + sw.shape, F32)]
               + [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in args[2:]])
  scratch = [pltpu.VMEM((vk, tile), F32), pltpu.VMEM((3, c, n, tile), F32), pltpu.VMEM((4, nk, tile), F32),
             pltpu.VMEM((kdim, c, tile), F32), pltpu.VMEM((nk, tile), F32), pltpu.VMEM((c * kdim, tile), F32),
             pltpu.VMEM((sw.shape[0], tile), F32), pltpu.VMEM((vk, tile), F32)]
  outs = pl.pallas_call(
      _read_backward_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], 2),
      grid=(b, t // tile), in_specs=_read_specs(args, tile) + [_tmajor_spec(nk, tile)] * 4,
      out_specs=out_specs, out_shape=tuple(out_shape), scratch_shapes=scratch,
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'arbitrary'), opts['vmem_mib']),
      name='bam_core_read_backward')(*args, *cts)
  return (outs[0], jnp.sum(outs[1], axis=0).astype(sw.dtype)) + tuple(outs[2:])


@partial(jax.custom_vjp, nondiff_argnums=(11,))
def _read(m, sw, rq, lq, rk, lk, rr, lo, lv, qs, ks, opts):
  return _read_forward_call((m, sw, rq, lq, rk, lk, rr, lo, lv, qs, ks), dict(opts))


def _read_fwd(m, sw, rq, lq, rk, lk, rr, lo, lv, qs, ks, opts):
  args = (m, sw, rq, lq, rk, lk, rr, lo, lv, qs, ks)
  return _read_forward_call(args, dict(opts)), args


def _read_bwd(opts, args, cts):
  return _read_backward_call(args, cts, dict(opts))


_read.defvjp(_read_fwd, _read_bwd)


# ---------------------------------------------------------------------------
# Write.

def _fill_factors(groups, c_raw, c_scr, a_scr, epsilon, dt):
  """Transpose contents into c_raw, then gated/normalized C (c_scr) and A (a_scr), FP32."""
  off = 0
  for g in range(0, len(groups), 3):
    content, logits, address = groups[g:g + 3]
    nh = address.shape[0]
    kdim = content.shape[1] // nh
    _load_transposed(content, c_raw, off * kdim, nh * kdim)
    for h in range(nh):
      r = pl.ds((off + h) * kdim, kdim)
      gate = _sigmoid(_row(logits, h)).astype(dt)
      c_scr[r, :] = (gate * _norm(c_raw[r, :], epsilon, 0).astype(dt)).astype(dt).astype(F32)
      a_scr[off + h] = _norm(address[h], epsilon, 0).astype(dt).astype(F32)
    off += nh
  return off


def _write_kernel(epsilon, n_groups, vb, kb):
  def kernel(m_ref, *refs):
    groups = refs[:3 * n_groups]
    out_ref, c_raw, c_scr, a_scr = refs[3 * n_groups:]
    dt = m_ref.dtype
    ntot = _fill_factors(groups, c_raw, c_scr, a_scr, epsilon, dt)
    vdim = a_scr.shape[1]
    kdim = c_scr.shape[0] // ntot
    for v0 in range(0, vdim, vb):
      rows = min(vb, vdim - v0)
      for k0 in range(0, kdim, kb):
        kw = min(kb, kdim - k0)
        accs = [None] * rows
        for i in range(ntot):
          ci = c_scr[pl.ds(i * kdim + k0, kw), :]
          arow = a_scr[i, pl.ds(v0, rows), :]
          for j in range(rows):
            accs[j] = _acc(accs[j], arow[j:j + 1] * ci)
        for j in range(rows):
          r = pl.ds((v0 + j) * kdim + k0, kw)
          out_ref[r, :] = (m_ref[r, :].astype(F32) + accs[j].astype(dt).astype(F32)).astype(dt)
  return kernel


def _write_backward_kernel(epsilon, n_groups, nb, kb, ab):
  def kernel(g_ref, *refs):
    groups = refs[:3 * n_groups]
    outs = refs[3 * n_groups:6 * n_groups]
    gf, c_raw, c_scr, a_scr, dc_scr, da_scr = refs[6 * n_groups:]
    dt = g_ref.dtype
    gf[...] = g_ref[...].astype(F32)
    ntot = _fill_factors(groups, c_raw, c_scr, a_scr, epsilon, dt)
    vdim = a_scr.shape[1]
    kdim = c_scr.shape[0] // ntot
    # dC[i] = sum_v A[i, v] G[v]   (G[v] contiguous [K, T] slabs).
    for i0 in range(0, ntot, nb):
      hb = min(nb, ntot - i0)
      for k0 in range(0, kdim, kb):
        kw = min(kb, kdim - k0)
        accs = [None] * hb
        for v in range(vdim):
          gv = gf[pl.ds(v * kdim + k0, kw), :]
          for j in range(hb):
            accs[j] = _acc(accs[j], _row(a_scr, i0 + j, v) * gv)
        for j in range(hb):
          dc_scr[pl.ds((i0 + j) * kdim + k0, kw), :] = accs[j]
    # dA[i] = sum_k C[i, k] G[:, k]   (G[:, k] strided [V, T] slabs).
    for i0 in range(0, ntot, ab):
      hb = min(ab, ntot - i0)
      accs = [None] * hb
      for k in range(kdim):
        gk = gf[pl.ds(k, vdim, stride=kdim), :]
        for j in range(hb):
          accs[j] = _acc(accs[j], c_scr[pl.ds((i0 + j) * kdim + k, 1), :] * gk)
      for j in range(hb):
        da_scr[i0 + j] = accs[j]
    off = 0
    for g in range(n_groups):
      content, logits, address = groups[3 * g:3 * g + 3]
      dcontent, dlogits, daddress = outs[3 * g:3 * g + 3]
      nh = address.shape[0]
      for h in range(nh):
        r = pl.ds((off + h) * kdim, kdim)
        sg = _sigmoid(_row(logits, h))
        x = c_raw[r, :]
        nc = _norm(x, epsilon, 0).astype(dt).astype(F32)
        dci = dc_scr[r, :]
        dlogits[pl.ds(h, 1), :] = (sg * (1 - sg) * jnp.sum(dci * nc, axis=0, keepdims=True)).astype(dlogits.dtype)
        dc_scr[r, :] = _norm_backward(x, sg.astype(dt).astype(F32) * dci, epsilon, 0)
        daddress[h] = _norm_backward(address[h], da_scr[off + h], epsilon, 0).astype(daddress.dtype)
      _store_transposed(dc_scr, off * kdim, dcontent, nh * kdim)
      off += nh
  return kernel


def _write_group_specs(groups, tile):
  specs = []
  for content, logits, address in (groups[i:i + 3] for i in range(0, len(groups), 3)):
    specs += [_tmajor_spec(content.shape[2], tile), _spec(logits.shape[1:-1], tile),
              _spec(address.shape[1:-1], tile)]
  return specs


def _factor_scratch(groups, tile, kdim, vdim):
  ntot = sum(a.shape[1] for a in groups[2::3])
  return ntot, [pltpu.VMEM((ntot * kdim, tile), F32), pltpu.VMEM((ntot * kdim, tile), F32),
                pltpu.VMEM((ntot, vdim, tile), F32)]


def _write_forward_call(m, groups, opts):
  b, vk, t = m.shape
  tile = _tile(t, opts['forward_tile'])
  vdim = groups[2].shape[2]
  kdim = vk // vdim
  _, scratch = _factor_scratch(groups, tile, kdim, vdim)
  return pl.pallas_call(
      _write_kernel(opts['epsilon'], len(groups) // 3, opts['row_block'], opts['k_block']),
      grid=(b, t // tile), in_specs=[_spec((vk,), tile)] + _write_group_specs(groups, tile),
      out_specs=_spec((vk,), tile), out_shape=jax.ShapeDtypeStruct(m.shape, m.dtype),
      scratch_shapes=scratch, input_output_aliases={0: 0},
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_write')(m, *groups)


def _write_backward_call(g, groups, opts):
  b, vk, t = g.shape
  tile = _tile(t, opts['reverse_tile'])
  vdim = groups[2].shape[2]
  kdim = vk // vdim
  ntot, scratch = _factor_scratch(groups, tile, kdim, vdim)
  scratch = ([pltpu.VMEM((vk, tile), F32)] + scratch
             + [pltpu.VMEM((ntot * kdim, tile), F32), pltpu.VMEM((ntot, vdim, tile), F32)])
  specs = _write_group_specs(groups, tile)
  return pl.pallas_call(
      _write_backward_kernel(opts['epsilon'], len(groups) // 3, opts['head_block'], opts['k_block'],
                             opts['address_block']),
      grid=(b, t // tile), in_specs=[_spec((vk,), tile)] + specs, out_specs=specs,
      out_shape=tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in groups),
      scratch_shapes=scratch, interpret=opts['interpret'],
      compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
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
# Public wrappers (same signatures as layers.bam_pallas.read/write).

def read(m, static_weight, q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits,
         q_standard, k_standard, *, qk_cols, read_epsilon, key_scale,
         forward_tile=128, reverse_tile=128, vmem_mib=None, interpret=False, **_):
  heads = q_key.shape[2]

  def local(m, sw, qk, ql, kk, kl, vo, ol, vl, qs, ks):
    b, v, kdim, t = m.shape
    opts = _freeze(dict(heads=heads, qk_cols=qk_cols, read_epsilon=read_epsilon, key_scale=key_scale,
                        forward_tile=forward_tile, reverse_tile=reverse_tile, vmem_mib=vmem_mib,
                        interpret=interpret, k_dim=kdim))
    key = lambda x: jnp.transpose(x, (0, 3, 2, 1))               # [B,T,N,C] -> [B,C,N,T]
    std = lambda x: _minor(x).reshape(b, heads * x.shape[3], t)   # [B,T,N,R] -> [B,N*R,T]
    outs = _read(m.reshape(b, v * kdim, t), _padded_static_weight(sw, heads),
                 key(qk), _minor(ql), key(kk), _minor(kl), key(vo), _minor(ol), _minor(vl),
                 std(qs), std(ks), opts)
    return tuple(o.reshape(b, t, heads, kdim) for o in outs)

  args = (m, static_weight, q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits,
          q_standard, k_standard)
  return _map_batch(local, args, (True, False) + (True,) * 9, 4)


def write(m, groups, *, epsilon, forward_tile=128, reverse_tile=128, vmem_mib=None, interpret=False,
          row_block=4, k_block=48, head_block=4, address_block=8, **_):
  flat = tuple(x for grp in groups for x in grp)

  def local(m, *xs):
    b, v, kdim, t = m.shape
    opts = _freeze(dict(epsilon=epsilon, forward_tile=forward_tile, reverse_tile=reverse_tile,
                        vmem_mib=vmem_mib, interpret=interpret, row_block=row_block, k_block=k_block,
                        head_block=head_block, address_block=address_block))
    gs = []
    for content, logits, address in (xs[i:i + 3] for i in range(0, len(xs), 3)):
      gs += [content.reshape(b, t, -1), _minor(logits), jnp.transpose(address, (0, 2, 3, 1))]
    return _write(m.reshape(b, v * kdim, t), opts, *gs).reshape(m.shape)

  return _map_batch(local, (m,) + flat, (True,) * (1 + len(flat)), 1)
