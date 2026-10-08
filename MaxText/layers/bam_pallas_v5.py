"""v5 fused BAM core: v4 layouts with loop-structured kernel bodies.

v4 bodies were fully unrolled Python loops; libtpu's scheduler hoists loads across such
straight-line blocks and spills heavily (v5p dumps: up to 70% of stores were spills).
v5 keeps v4's k-major M and token-major big I/O but runs every long k / head loop as a
lax.fori_loop, so each scheduling region is one small iteration and accumulators are
loop-carried in vector registers. Dynamic loop indices only touch leading (untiled)
scratch dimensions; head-major <-> k-major conversions use row loads.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from layers.bam_pallas import (F32, _freeze, _map_batch, _minor, _norm, _norm_backward, _pad_rows,
                               _params, _sigmoid, _spec, _tile)
from layers.bam_pallas_v3 import _acc, _load_transposed, _store_transposed, _tmajor_spec


def _fori(n, body, init, unroll=1):
  return jax.lax.fori_loop(0, n, body, init, unroll=unroll)


# ---------------------------------------------------------------------------
# Write. Content token-major [T, N*K]; logits [N, T]; address [N, V, T]; M k-major [K, V, T].

def _fill_factors(groups, c_raw, a_scr, ckm, eps):
  """c_raw: transposed raw content, head-major rows (i*K + k). a_scr: normalized address.
  ckm[k]: gated normalized content of every head for key row k, [NPt, T] (k-major)."""
  off = 0
  kdim = ckm.shape[0]
  for g in range(0, len(groups), 3):
    content, logits, address = groups[g:g + 3]
    nh = address.shape[0]
    _load_transposed(content, c_raw, off * kdim, nh * kdim)
    for h in range(nh):
      r = pl.ds((off + h) * kdim, kdim)
      c_raw[r, :] = _sigmoid(logits[pl.ds(h, 1), :]) * _norm(c_raw[r, :], eps, 0)
      a_scr[off + h] = _norm(address[h], eps, 0)
    off += nh
  ntot = off
  npt = ckm.shape[1]

  def body(k, carry):
    rows = c_raw[pl.ds(k, ntot, stride=kdim), :]
    if npt > ntot:
      rows = jnp.concatenate((rows, jnp.zeros((npt - ntot, rows.shape[1]), F32)), axis=0)
    ckm[k] = rows
    return carry
  _fori(kdim, body, 0)
  return ntot


def _raw_content(groups, c_raw, kdim):
  off = 0
  for g in range(0, len(groups), 3):
    content, _, address = groups[g:g + 3]
    nh = address.shape[0]
    _load_transposed(content, c_raw, off * kdim, nh * kdim)
    off += nh
  return off


def _write_kernel(eps, n_groups, kb):
  def kernel(m_ref, *refs):
    groups = refs[:3 * n_groups]
    out_ref, c_raw, a_scr, ckm = refs[3 * n_groups:]
    kdim = m_ref.shape[0]
    ntot = _fill_factors(groups, c_raw, a_scr, ckm, eps)

    def body(b, carry):
      for j in range(kb):
        k = b * kb + j
        ck = ckm[k]
        acc = None
        for i in range(ntot):
          acc = _acc(acc, ck[i:i + 1] * a_scr[i])
        out_ref[k] = (m_ref[k].astype(F32) + acc).astype(out_ref.dtype)
      return carry
    _fori(kdim // kb, body, 0)
  return kernel


def _factor_scratch(groups, tile, kdim, vdim):
  ntot = sum(a.shape[1] for a in groups[2::3])
  return ntot, [pltpu.VMEM((ntot * kdim, tile), F32), pltpu.VMEM((ntot, vdim, tile), F32),
                pltpu.VMEM((kdim, _pad_rows(ntot), tile), F32)]


def _write_group_specs(groups, tile):
  specs = []
  for content, logits, address in (groups[i:i + 3] for i in range(0, len(groups), 3)):
    specs += [_tmajor_spec(content.shape[2], tile), _spec(logits.shape[1:-1], tile),
              _spec(address.shape[1:-1], tile)]
  return specs


def _write_forward_call(m, groups, opts):
  b, kdim, vdim, t = m.shape
  tile = _tile(t, opts['forward_tile'])
  _, scratch = _factor_scratch(groups, tile, kdim, vdim)
  return pl.pallas_call(
      _write_kernel(opts['epsilon'], len(groups) // 3, opts['k_block']),
      grid=(b, t // tile), in_specs=[_spec((kdim, vdim), tile)] + _write_group_specs(groups, tile),
      out_specs=_spec((kdim, vdim), tile), out_shape=jax.ShapeDtypeStruct(m.shape, m.dtype),
      scratch_shapes=scratch, input_output_aliases={0: 0},
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_write')(m, *groups)


# ---------------------------------------------------------------------------
# Read. M k-major [K, V, T]; keys [C, N, T]; logits [N, T]; standard QK [R, N, T];
# Q/K/V/LocalO and their cotangents token-major [T, N*K].

def _keys(rq, lq, rk, lk, rr, eps, scale):
  nq, nk, nr = (_norm(r, eps, 0) for r in (rq, rk, rr))
  gq = scale * _sigmoid(lq)
  gk = scale * _sigmoid(lk)
  return gq[None] * nq, gk[None] * nk, nr, gq, gk, nq, nk


def _load_headmajor(src_ref, dst, n, kdim):
  """Token-major [t, N*K] -> FP32 dst[h, kb, ks, t] (head h, key row 8*kb+ks) via 128x128 transposes."""
  nk = n * kdim
  for j0 in range(0, nk, 128):
    w = min(128, nk - j0)
    blk = src_ref[:, pl.ds(j0, w)].astype(F32).T          # rows (h, k) in head-major order
    r = 0
    while r < w:
      h, k0 = divmod(j0 + r, kdim)
      cnt = min(kdim - k0, w - r)
      dst[h, pl.ds(k0 // 8, cnt // 8)] = blk[r:r + cnt].reshape(cnt // 8, 8, blk.shape[1])
      r += cnt


def _store_kmajor_tokens(src, n, kdim, out_ref):
  """k-major FP32 src[k, h, t] (h < NP) -> token-major out_ref[t, h*K + k], 128-column chunks."""
  nk = n * kdim
  for j0 in range(0, nk, 128):
    w = min(128, nk - j0)
    pieces = []
    r = 0
    while r < w:
      h, k0 = divmod(j0 + r, kdim)
      cnt = min(kdim - k0, w - r)
      pieces.append(src[pl.ds(k0, cnt), pl.ds(h, 1), :].reshape(cnt, src.shape[2]))
      r += cnt
    blk = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=0)
    out_ref[:, pl.ds(j0, w)] = blk.T.astype(out_ref.dtype)


def _read_kernel(heads, qk_cols, eps, scale, kb):
  def kernel(m_ref, sw_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, q_ref, k_ref, v_ref, o_ref,
             key_scr, kq_s, kk_s, kv_s, ko_s):
    dt = m_ref.dtype
    n = heads
    np_ = _pad_rows(n)
    kdim = m_ref.shape[0]
    cdim = rq.shape[0]
    t = m_ref.shape[2]
    mc0 = 4 * np_
    kq, kk, nr, *_ = _keys(rq[...], lq[...], rk[...], lk[...], rr[...], eps, scale)
    key_scr[0] = kq
    key_scr[1] = kk
    key_scr[2] = nr
    gv = scale * _sigmoid(lv[...])
    go = scale * _sigmoid(lo[...])
    sw = sw_ref[...].astype(dt)
    pad = jnp.zeros((np_ - n, t), F32) if np_ > n else None
    padded = lambda x: x if pad is None else jnp.concatenate((x, pad), axis=0)

    def block(dyn_qk):
      def body(b, carry):
        k0 = b * kb + (0 if dyn_qk else qk_cols)
        st = jnp.dot(sw, jnp.concatenate([m_ref[k0 + j] for j in range(kb)], axis=1),
                     preferred_element_type=F32)
        acc = {}
        for c in range(cdim):
          keys = [key_scr[r, c] for r in range(3)]
          for j in range(kb):
            row = st[mc0 + c:mc0 + c + 1, j * t:(j + 1) * t]
            for r in (range(3) if dyn_qk else (2,)):
              acc[r, j] = _acc(acc.get((r, j)), keys[r] * row)
        for j in range(kb):
          k = k0 + j
          sj = st[:, j * t:(j + 1) * t]
          kv_s[k] = padded(acc[2, j] * gv + sj[2 * np_:2 * np_ + n])
          ko_s[k] = padded(acc[2, j] * go + sj[3 * np_:3 * np_ + n])
          if dyn_qk:
            kq_s[k] = padded(acc[0, j] + sj[0:n])
            kk_s[k] = padded(acc[1, j] + sj[np_:np_ + n])
          else:
            kq_s[k] = padded(qs[k - qk_cols].astype(F32))
            kk_s[k] = padded(ks[k - qk_cols].astype(F32))
        return carry
      return body
    _fori(qk_cols // kb, block(True), 0)
    _fori((kdim - qk_cols) // kb, block(False), 0)
    for s, ref in ((kq_s, q_ref), (kk_s, k_ref), (kv_s, v_ref), (ko_s, o_ref)):
      _store_kmajor_tokens(s, n, kdim, ref)
  return kernel


def _read_backward_kernel(heads, qk_cols, eps, scale, cb, kb, ablate=''):
  ablate = set(ablate.split(',')) if ablate else set()

  def kernel(m_ref, sw_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, dq, dk, dv, do,
             dm_ref, dsw_ref, drq, dlq, drk, dlk, drr, dlo, dlv, dqs, dks,
             key_scr, keyn, gate_s, ct, mc_s, dyvo, dmc_s):
    dt = m_ref.dtype
    n = heads
    np_ = _pad_rows(n)
    kdim, vdim, t = m_ref.shape
    cdim = rq.shape[0]
    rdim = qs.shape[0]
    mc0 = 4 * np_
    kbk = kdim // 8
    qb = qk_cols // 8
    kq, kk, nr, gq, gk, nq, nkn = _keys(rq[...], lq[...], rk[...], lk[...], rr[...], eps, scale)
    for r, x in enumerate((kq, kk, nr)):
      key_scr[r] = x
      for h in range(n):
        keyn[r, h] = x[:, h, :]
    for i, ref in enumerate((dq, dk, dv, do)):
      _load_headmajor(ref, ct.at[i], n, kdim)
    sgv, sgo = _sigmoid(lv[...]), _sigmoid(lo[...])
    gvf, gof = scale * sgv, scale * sgo
    for h in range(n):
      gate_s[0, h] = gvf[h:h + 1]
      gate_s[1, h] = gof[h:h + 1]
    sw = sw_ref[...].astype(dt)
    swc = sw[mc0:mc0 + cdim]
    pad = jnp.zeros((np_ - n, t), F32) if np_ > n else None
    padded = lambda x: x if pad is None else jnp.concatenate((x, pad), axis=0)
    zero = jnp.zeros((n, t), F32)

    def krow(i, kblk, s):
      """Cotangent i at key row 8*kblk+s for all heads: [N, T] (one row load per head)."""
      return ct[i, pl.ds(0, n), kblk, pl.ds(s, 1), :].reshape(n, t)

    # Compressed rows mc[k] = P^T M_k ([C, T]) for every key row.
    def mc_body(b, carry):
      k0 = b * kb
      st = jnp.dot(swc, jnp.concatenate([m_ref[k0 + j] for j in range(kb)], axis=1),
                   preferred_element_type=F32)
      for j in range(kb):
        mc_s[k0 + j] = st[:, j * t:(j + 1) * t]
      return carry
    if 'mc' not in ablate:
      _fori(kdim // kb, mc_body, 0)

    # VO (k-major): y, gate gradients, VO key gradient (register-carried).
    # Loop carries are padded to NP (multiple-of-8) rows: libtpu crashed on [N=20, T] carries.
    zp = jnp.zeros((np_, t), F32)
    keys_p = [[padded(key_scr[r, c]) for c in range(cdim)] for r in range(3)]
    gvp, gop = padded(gvf), padded(gof)

    def vo_body(kblk, carry):
      dlv_a, dlo_a, dkv = carry
      dkv = list(dkv)
      for s in range(8):
        mc = mc_s[kblk * 8 + s]
        y = None
        for c in range(cdim):
          y = _acc(y, keys_p[2][c] * mc[c:c + 1])
        dvk, dok = padded(krow(2, kblk, s)), padded(krow(3, kblk, s))
        dlv_a = dlv_a + dvk * y
        dlo_a = dlo_a + dok * y
        dy = dvk * gvp + dok * gop
        for c in range(cdim):
          dkv[c] = dkv[c] + dy * mc[c:c + 1]
      return dlv_a, dlo_a, tuple(dkv)
    dlv_a, dlo_a, dkv = ((zp, zp, (zp,) * cdim) if 'vo' in ablate else
                         _fori(kbk, vo_body, (zp, zp, (zp,) * cdim)))
    dlv_a, dlo_a, dkv = dlv_a[:n], dlo_a[:n], tuple(x[:n] for x in dkv)
    dlv[...] = (scale * sgv * (1 - sgv) * dlv_a).astype(dlv.dtype)
    dlo[...] = (scale * sgo * (1 - sgo) * dlo_a).astype(dlo.dtype)

    # Direct Q/K key gradients (k-major) and RoPE passthrough.
    dkeys = []
    for i, ds_ref in ((0, dqs), (1, dks)):
      def key_body(kblk, acc, i=i):
        acc = list(acc)
        for s in range(8):
          dyk = padded(krow(i, kblk, s))
          mc = mc_s[kblk * 8 + s]
          for c in range(cdim):
            acc[c] = acc[c] + dyk * mc[c:c + 1]
        return tuple(acc)
      res = (zp,) * cdim if 'key' in ablate else _fori(qb, key_body, (zp,) * cdim)
      dkeys.append(jnp.stack(tuple(x[:n] for x in res)))
      for r in range(rdim):
        k = qk_cols + r
        ds_ref[r] = krow(i, k // 8, k % 8).astype(ds_ref.dtype)

    # Head-major VO cotangent.
    def dyvo_body(h, carry):
      dyvo[h] = ct[2, h] * gate_s[0, h][None] + ct[3, h] * gate_s[1, h][None]
      return carry
    if 'dyvo' not in ablate:
      _fori(n, dyvo_body, 0)

    # dMc[c] = sum_h key[h,c] dy[h] over heads (register-carried per c block).
    for c0 in (range(0, cdim, cb) if 'dmc' not in ablate else ()):
      cs = list(range(c0, min(c0 + cb, cdim)))

      def dmc_body(h, carry, cs=cs):
        part, full = carry
        part, full = list(part), list(full)
        dyq = ct[0, h, pl.ds(0, qb)].reshape(qk_cols, t)
        dyk = ct[1, h, pl.ds(0, qb)].reshape(qk_cols, t)
        dyv = dyvo[h].reshape(kdim, t)
        kn = [keyn[r, h] for r in range(3)]
        for j, c in enumerate(cs):
          part[j] = part[j] + kn[0][c:c + 1] * dyq + kn[1][c:c + 1] * dyk
          full[j] = full[j] + kn[2][c:c + 1] * dyv
        return tuple(part), tuple(full)
      part, full = _fori(n, dmc_body, ((jnp.zeros((qk_cols, t), F32),) * len(cs),
                                       (jnp.zeros((kdim, t), F32),) * len(cs)))
      for j, c in enumerate(cs):
        tot = jnp.concatenate((full[j][:qk_cols] + part[j], full[j][qk_cols:]), axis=0)
        dmc_s[c] = tot.reshape(kbk, 8, t)

    # Key gate / normalization reverses over [C, N, T].
    for r, l, dr, dl, g, nrm, dkey in ((rq, lq, drq, dlq, gq, nq, dkeys[0]),
                                       (rk, lk, drk, dlk, gk, nkn, dkeys[1])):
      sg = _sigmoid(l[...])
      dl[...] = (scale * sg * (1 - sg) * jnp.sum(dkey * nrm, axis=0)).astype(dl.dtype)
      dr[...] = _norm_backward(r[...], g[None] * dkey, eps, 0).astype(dr.dtype)
    drr[...] = _norm_backward(rr[...], jnp.stack(dkv), eps, 0).astype(drr.dtype)

    # Static-read reverse (k-major): D_k assembled in registers, blocked dots.
    def rev_body(kblk, dsw):
      cols, mks = [], []
      for s in range(8):
        k = kblk * 8 + s
        dmc = dmc_s[pl.ds(0, cdim), kblk, pl.ds(s, 1), :].reshape(cdim, t)
        qrow = (k < qk_cols).astype(F32)
        cols.append(jnp.concatenate([padded(krow(0, kblk, s) * qrow), padded(krow(1, kblk, s) * qrow),
                                     padded(krow(2, kblk, s)), padded(krow(3, kblk, s)), dmc], axis=0))
        mks.append(m_ref[k])
      for j0 in range(0, 8, kb):
        dcat = jnp.concatenate(cols[j0:j0 + kb], axis=1).astype(dt)
        mcat = jnp.concatenate(mks[j0:j0 + kb], axis=1)
        dmk = jax.lax.dot_general(sw, dcat, (((0,), (0,)), ((), ())), preferred_element_type=F32)
        for j in range(kb):
          dm_ref[kblk * 8 + j0 + j] = dmk[:, j * t:(j + 1) * t].astype(dm_ref.dtype)
        dsw = dsw + jax.lax.dot_general(dcat, mcat, (((1,), (1,)), ((), ())), preferred_element_type=F32)
      return dsw
    dsw = jnp.zeros(sw.shape, F32) if 'rev' in ablate else _fori(kbk, rev_body, jnp.zeros(sw.shape, F32))

    @pl.when(pl.program_id(1) == 0)
    def _():
      dsw_ref[...] = jnp.zeros(dsw_ref.shape, dsw_ref.dtype)
    dsw_ref[...] += dsw
  return kernel


def _read_specs(args, tile):
  m, sw = args[0], args[1]
  return [_spec(m.shape[1:-1], tile), pl.BlockSpec(sw.shape, lambda b, i: (0, 0))] + [
      _spec(x.shape[1:-1], tile) for x in args[2:]]


def _read_forward_call(args, opts):
  m = args[0]
  b, kdim, v, t = m.shape
  n = opts['heads']
  c = args[2].shape[1]
  np_ = _pad_rows(n)
  tile = _tile(t, opts['forward_tile'])
  nk = n * kdim
  return pl.pallas_call(
      _read_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], opts['k_block']),
      grid=(b, t // tile), in_specs=_read_specs(args, tile), out_specs=[_tmajor_spec(nk, tile)] * 4,
      out_shape=tuple(jax.ShapeDtypeStruct((b, t, nk), m.dtype) for _ in range(4)),
      scratch_shapes=[pltpu.VMEM((3, c, n, tile), F32)] + [pltpu.VMEM((kdim, np_, tile), F32)] * 4,
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read')(*args)


def _read_backward_call(args, cts, opts):
  m, sw = args[0], args[1]
  b, kdim, v, t = m.shape
  n = opts['heads']
  c = args[2].shape[1]
  nk = n * kdim
  tile = _tile(t, opts['reverse_tile'])
  out_specs = ([_spec((kdim, v), tile), pl.BlockSpec((None,) + sw.shape, lambda bb, i: (bb, 0, 0))]
               + [_spec(x.shape[1:-1], tile) for x in args[2:]])
  out_shape = ([jax.ShapeDtypeStruct(m.shape, m.dtype), jax.ShapeDtypeStruct((b,) + sw.shape, F32)]
               + [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in args[2:]])
  scratch = [pltpu.VMEM((3, c, n, tile), F32), pltpu.VMEM((3, n, c, tile), F32),
             pltpu.VMEM((2, n, 1, tile), F32),
             pltpu.VMEM((4, n, kdim // 8, 8, tile), F32), pltpu.VMEM((kdim, c, tile), F32),
             pltpu.VMEM((n, kdim // 8, 8, tile), F32), pltpu.VMEM((c, kdim // 8, 8, tile), F32)]
  outs = pl.pallas_call(
      _read_backward_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], 2,
                            opts['rev_k_block'], opts.get('ablate', '')),
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


def read(m, static_weight, q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits,
         q_standard, k_standard, *, qk_cols, read_epsilon, key_scale,
         forward_tile=128, reverse_tile=128, vmem_mib=None, interpret=False, **kw):
  """m is k-major [B,K,V,T]; other arguments token-major as in layers.bam_pallas.read."""
  from layers.bam_pallas import _padded_static_weight
  heads = q_key.shape[2]

  def local(m, sw, qk, ql, kk, kl, vo, ol, vl, qs, ks):
    b, kdim, v, t = m.shape
    opts = _freeze(dict(heads=heads, qk_cols=qk_cols, read_epsilon=read_epsilon, key_scale=key_scale,
                        forward_tile=forward_tile, reverse_tile=reverse_tile, vmem_mib=vmem_mib,
                        interpret=interpret, k_block=kw.get('k_block', 2),
                        rev_k_block=kw.get('rev_k_block', 2), ablate=kw.get('ablate', '')))
    tr = lambda x: jnp.transpose(x, (0, 3, 2, 1))       # [B,T,N,X] -> [B,X,N,T]
    outs = _read(m, _padded_static_weight(sw, heads), tr(qk), _minor(ql), tr(kk), _minor(kl),
                 tr(vo), _minor(ol), _minor(vl), tr(qs), tr(ks), opts)
    return tuple(o.reshape(b, t, heads, kdim) for o in outs)

  args = (m, static_weight, q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits,
          q_standard, k_standard)
  return _map_batch(local, args, (True, False) + (True,) * 9, 4)
