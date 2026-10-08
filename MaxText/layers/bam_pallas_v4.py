"""v4 fused BAM core: k-major M, contiguous MXU static reads, one permutation by row loads.

Same equations as layers/bam_pallas.py (per-tile reference). Lessons from v5p bundle dumps:
sublane-strided stores cost one store-slot op per row (one store slot per bundle), strided
loads one load op per row (three load slots); BF16 rounding emulation of intermediates
doubles VALU work. Hence:

* M is carried k-major [B, K, V, T]: each key row's [V, T] slab is contiguous, so the static
  reads and C10 compression are one small MXU dot per k with no relayout.
* Dynamic reads accumulate per k-block with c-major keys; every store is contiguous.
* The only (k, n) -> (n, k) permutation is done by per-row strided loads that assemble
  head-major 128-row chunks for the in-kernel transposes to token-major outputs.
* Arithmetic stays FP32 inside a kernel; only kernel outputs are rounded.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from layers.bam_pallas import (F32, _freeze, _map_batch, _minor, _norm, _norm_backward, _pad_rows,
                               _padded_static_weight, _params, _sigmoid, _spec, _tile, _whole)
from layers.bam_pallas_v3 import _acc, _load_transposed, _tmajor_spec


def _keys(rq, lq, rk, lk, rr, eps, scale):
  nq, nk, nr = (_norm(r, eps, 0) for r in (rq, rk, rr))
  gq = scale * _sigmoid(lq)
  gk = scale * _sigmoid(lk)
  return gq[None] * nq, gk[None] * nk, nr, gq, gk, nq, nk


def _kmajor_to_tokens(kscr, np_, n, kdim, out_ref):
  """k-major rows kscr[k*NP + h] -> token-major out_ref[t, h*K + k] in 128-row head-major chunks."""
  nk = n * kdim
  for j0 in range(0, nk, 128):
    w = min(128, nk - j0)
    pieces = []
    r = j0
    while r < j0 + w:
      h, k0 = divmod(r, kdim)
      cnt = min(kdim - k0, j0 + w - r)
      pieces.append(kscr[pl.ds(k0 * np_ + h, cnt, stride=np_), :])
      r += cnt
    blk = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=0)
    out_ref[:, pl.ds(j0, w)] = blk.T.astype(out_ref.dtype)


def _read_kernel(heads, qk_cols, eps, scale, kb, ablate='', dot_block=16):
  def kernel(m_ref, sw_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, q_ref, k_ref, v_ref, o_ref,
             key_scr, kscr, st_scr):
    dt = m_ref.dtype
    n = heads
    np_ = _pad_rows(n)
    kdim = m_ref.shape[0]
    cdim = rq.shape[0]
    mc0 = 4 * np_
    kq, kk, nr, *_ = _keys(rq[...], lq[...], rk[...], lk[...], rr[...], eps, scale)
    key_scr[0] = kq
    key_scr[1] = kk
    key_scr[2] = nr
    gv = scale * _sigmoid(lv[...])
    go = scale * _sigmoid(lo[...])
    sw = sw_ref[...].astype(dt)
    t = m_ref.shape[2]
    # Static reads + compression: k-major slabs concatenated along lanes are free, so a few
    # large dots replace 96 tiny ones; each k's results are a lane-aligned [J', T] chunk.
    for d0 in range(0, kdim, dot_block):
      kk_ = list(range(d0, min(d0 + dot_block, kdim)))
      mcat = jnp.concatenate([m_ref[k] for k in kk_], axis=1)
      st_scr[:, pl.ds(d0 * t, len(kk_) * t)] = jnp.dot(sw, mcat, preferred_element_type=F32)
    for k0 in range(0, kdim, kb):
      ks_ = list(range(k0, min(k0 + kb, kdim)))
      lanes = {k: pl.ds(k * t, t) for k in ks_}
      acc = {(r, k): None for r in range(3) for k in ks_}
      for c in (range(cdim) if 'dyn' not in ablate else range(1)):
        keys = [key_scr[r, c] for r in range(3)]
        for k in ks_:
          row = st_scr[pl.ds(mc0 + c, 1), lanes[k]]
          for r in range(3):
            if r < 2 and k >= qk_cols:
              continue
            acc[r, k] = _acc(acc[r, k], keys[r] * row)
      for k in ks_:
        rows = pl.ds(k * np_, n)
        kscr[2, rows, :] = acc[2, k] * gv + st_scr[pl.ds(2 * np_, n), lanes[k]]
        kscr[3, rows, :] = acc[2, k] * go + st_scr[pl.ds(3 * np_, n), lanes[k]]
        if k < qk_cols:
          kscr[0, rows, :] = acc[0, k] + st_scr[pl.ds(0, n), lanes[k]]
          kscr[1, rows, :] = acc[1, k] + st_scr[pl.ds(np_, n), lanes[k]]
        else:
          kscr[0, rows, :] = qs[k - qk_cols].astype(F32)
          kscr[1, rows, :] = ks[k - qk_cols].astype(F32)
    for i, ref in enumerate((q_ref, k_ref, v_ref, o_ref)):
      if 'transpose' in ablate:
        ref[...] = jnp.zeros(ref.shape, ref.dtype)
      else:
        _kmajor_to_tokens(kscr.at[i], np_, n, kdim, ref)
  return kernel


def _read_forward_call(args, opts):
  m = args[0]
  b, kdim, v, t = m.shape
  n = opts['heads']
  c = args[2].shape[1]
  np_ = _pad_rows(n)
  tile = _tile(t, opts['forward_tile'])
  nk = n * kdim
  specs = [_spec((kdim, v), tile), _whole(args[1].shape)] + [_spec(x.shape[1:-1], tile) for x in args[2:]]
  return pl.pallas_call(
      _read_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], opts['k_block'],
                   opts.get('ablate', ''), opts.get('dot_block', 16)),
      grid=(b, t // tile), in_specs=specs, out_specs=[_tmajor_spec(nk, tile)] * 4,
      out_shape=tuple(jax.ShapeDtypeStruct((b, t, nk), m.dtype) for _ in range(4)),
      scratch_shapes=[pltpu.VMEM((3, c, n, tile), F32), pltpu.VMEM((4, kdim * np_, tile), F32),
                      pltpu.VMEM((args[1].shape[0], kdim * tile), F32)],
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read')(*args)


def _read_backward_kernel(heads, qk_cols, eps, scale, cb, kb, dot_block):
  """Reverse of _read_kernel. Cotangents are staged once head-major (transposes) and once
  k-major (one permutation by row loads); every other access is contiguous."""
  def kernel(m_ref, sw_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, dq, dk, dv, do,
             dm_ref, dsw_ref, drq, dlq, drk, dlk, drr, dlo, dlv, dqs, dks,
             key_scr, ct, ctk, st_scr, dyvo, dmcs):
    dt = m_ref.dtype
    n = heads
    np_ = _pad_rows(n)
    kdim, vdim, t = m_ref.shape
    cdim = rq.shape[0]
    rdim = qs.shape[0]
    nk = n * kdim
    mc0 = 4 * np_
    rest = kdim - qk_cols
    kq, kk, nr, gq, gk, nq, nkn = _keys(rq[...], lq[...], rk[...], lk[...], rr[...], eps, scale)
    key_scr[0] = kq
    key_scr[1] = kk
    key_scr[2] = nr
    zpad = jnp.zeros((np_ - n, t), F32)
    for i, ref in enumerate((dq, dk, dv, do)):
      _load_transposed(ref, ct.at[i], 0, nk)
      for k in range(kdim):
        rows = ct[i, pl.ds(k, n, stride=kdim), :]
        ctk[i, pl.ds(k * np_, np_), :] = jnp.concatenate((rows, zpad), axis=0) if np_ > n else rows
    sgv, sgo = _sigmoid(lv[...]), _sigmoid(lo[...])
    gvf, gof = scale * sgv, scale * sgo
    sw = sw_ref[...].astype(dt)
    swc = sw[mc0:mc0 + cdim]
    for d0 in range(0, kdim, dot_block):
      kk_ = list(range(d0, min(d0 + dot_block, kdim)))
      st_scr[:, pl.ds(d0 * t, len(kk_) * t)] = jnp.dot(
          swc, jnp.concatenate([m_ref[k] for k in kk_], axis=1), preferred_element_type=F32)
    mcrow = lambda c, k: st_scr[pl.ds(c, 1), pl.ds(k * t, t)]
    # VO (k-major): forward y, gate gradients, VO key gradient.
    dlv_acc = dlo_acc = None
    dkv = [None] * cdim
    for k in range(kdim):
      rows = pl.ds(k * np_, n)
      y = None
      for c in range(cdim):
        y = _acc(y, key_scr[2, c] * mcrow(c, k))
      dvk = ctk[2, rows, :]
      dok = ctk[3, rows, :]
      dlv_acc = _acc(dlv_acc, dvk * y)
      dlo_acc = _acc(dlo_acc, dok * y)
      dy = dvk * gvf + dok * gof
      for c in range(cdim):
        dkv[c] = _acc(dkv[c], dy * mcrow(c, k))
    dlv[...] = (scale * sgv * (1 - sgv) * dlv_acc).astype(dlv.dtype)
    dlo[...] = (scale * sgo * (1 - sgo) * dlo_acc).astype(dlo.dtype)
    # Direct Q/K key gradients (k-major) and RoPE passthrough (r-major outputs).
    dkeys = []
    for i, ds_ref in ((0, dqs), (1, dks)):
      acc = [None] * cdim
      for k in range(qk_cols):
        dyk = ctk[i, pl.ds(k * np_, n), :]
        for c in range(cdim):
          acc[c] = _acc(acc[c], dyk * mcrow(c, k))
      dkeys.append(jnp.stack(acc))
      for r in range(rdim):
        ds_ref[r] = ctk[i, pl.ds((qk_cols + r) * np_, n), :].astype(ds_ref.dtype)
    # Head-major VO cotangent and dMc[c] = sum_n key[n,c] dy[n] (register accumulators).
    for h in range(n):
      hs = pl.ds(h * kdim, kdim)
      dyvo[hs, :] = ct[2, hs, :] * gvf[h:h + 1] + ct[3, hs, :] * gof[h:h + 1]
    for c0 in range(0, cdim, cb):
      cs = list(range(c0, min(c0 + cb, cdim)))
      full = {c: None for c in cs}
      part = {c: None for c in cs}
      for h in range(n):
        dyq = ct[0, pl.ds(h * kdim, qk_cols), :]
        dyk = ct[1, pl.ds(h * kdim, qk_cols), :]
        dyv = dyvo[pl.ds(h * kdim, kdim), :]
        for c in cs:
          part[c] = _acc(part[c], key_scr[0, c, pl.ds(h, 1), :] * dyq + key_scr[1, c, pl.ds(h, 1), :] * dyk)
          full[c] = _acc(full[c], key_scr[2, c, pl.ds(h, 1), :] * dyv)
      for c in cs:
        dmcs[pl.ds(c * kdim, qk_cols), :] = full[c][:qk_cols] + part[c]
        dmcs[pl.ds(c * kdim + qk_cols, rest), :] = full[c][qk_cols:]
    # Key gate / normalization reverses over [C, N, T].
    for r, l, dr, dl, g, nrm, dkey in ((rq, lq, drq, dlq, gq, nq, dkeys[0]), (rk, lk, drk, dlk, gk, nkn, dkeys[1])):
      sg = _sigmoid(l[...])
      dl[...] = (scale * sg * (1 - sg) * jnp.sum(dkey * nrm, axis=0)).astype(dl.dtype)
      dr[...] = _norm_backward(r[...], g[None] * dkey, eps, 0).astype(dr.dtype)
    drr[...] = _norm_backward(rr[...], jnp.stack(dkv), eps, 0).astype(drr.dtype)
    # Static-read reverse: D_k assembled in registers (k-major), blocked dots over k.
    zgrp = jnp.zeros((np_, t), F32)
    dsw = None
    for k0 in range(0, kdim, kb):
      ks_ = list(range(k0, min(k0 + kb, kdim)))
      cols = []
      for k in ks_:
        grp = pl.ds(k * np_, np_)
        qk = [ctk[0, grp, :], ctk[1, grp, :]] if k < qk_cols else [zgrp, zgrp]
        cols.append(jnp.concatenate(qk + [ctk[2, grp, :], ctk[3, grp, :],
                                          dmcs[pl.ds(k, cdim, stride=kdim), :]], axis=0))
      dcat = jnp.concatenate(cols, axis=1).astype(dt)
      mcat = jnp.concatenate([m_ref[k] for k in ks_], axis=1)
      dmk = jax.lax.dot_general(sw, dcat, (((0,), (0,)), ((), ())), preferred_element_type=F32)
      for i, k in enumerate(ks_):
        dm_ref[k] = dmk[:, i * t:(i + 1) * t].astype(dm_ref.dtype)
      dsw = _acc(dsw, jax.lax.dot_general(dcat, mcat, (((1,), (1,)), ((), ())), preferred_element_type=F32))

    @pl.when(pl.program_id(1) == 0)
    def _():
      dsw_ref[...] = jnp.zeros(dsw_ref.shape, dsw_ref.dtype)
    dsw_ref[...] += dsw
  return kernel


def _read_backward_call(args, cts, opts):
  m, sw = args[0], args[1]
  b, kdim, v, t = m.shape
  n = opts['heads']
  c = args[2].shape[1]
  np_ = _pad_rows(n)
  nk = n * kdim
  tile = _tile(t, opts['reverse_tile'])
  in_specs = ([_spec((kdim, v), tile), _whole(sw.shape)] + [_spec(x.shape[1:-1], tile) for x in args[2:]]
              + [_tmajor_spec(nk, tile)] * 4)
  out_specs = ([_spec((kdim, v), tile), pl.BlockSpec((None,) + sw.shape, lambda bb, i: (bb, 0, 0))]
               + [_spec(x.shape[1:-1], tile) for x in args[2:]])
  out_shape = ([jax.ShapeDtypeStruct(m.shape, m.dtype), jax.ShapeDtypeStruct((b,) + sw.shape, F32)]
               + [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in args[2:]])
  scratch = [pltpu.VMEM((3, c, n, tile), F32), pltpu.VMEM((4, nk, tile), F32),
             pltpu.VMEM((4, kdim * np_, tile), F32), pltpu.VMEM((c, kdim * tile), F32),
             pltpu.VMEM((nk, tile), F32), pltpu.VMEM((c * kdim, tile), F32)]
  outs = pl.pallas_call(
      _read_backward_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], 2,
                            opts.get('rev_k_block', 2), opts.get('dot_block', 16)),
      grid=(b, t // tile), in_specs=in_specs, out_specs=out_specs, out_shape=tuple(out_shape),
      scratch_shapes=scratch, interpret=opts['interpret'],
      compiler_params=_params(('parallel', 'arbitrary'), opts['vmem_mib']),
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
  heads = q_key.shape[2]

  def local(m, sw, qk, ql, kk, kl, vo, ol, vl, qs, ks):
    b, kdim, v, t = m.shape
    opts = _freeze(dict(heads=heads, qk_cols=qk_cols, read_epsilon=read_epsilon, key_scale=key_scale,
                        forward_tile=forward_tile, reverse_tile=reverse_tile, vmem_mib=vmem_mib,
                        interpret=interpret, k_block=kw.get('k_block', 2), dot_block=kw.get('dot_block', 16),
                        rev_k_block=kw.get('rev_k_block', 2)))
    key = lambda x: jnp.transpose(x, (0, 3, 2, 1))       # [B,T,N,C] -> [B,C,N,T]
    std = lambda x: jnp.transpose(x, (0, 3, 2, 1))       # [B,T,N,R] -> [B,R,N,T]
    outs = _read(m, _padded_static_weight(sw, heads), key(qk), _minor(ql), key(kk), _minor(kl),
                 key(vo), _minor(ol), _minor(vl), std(qs), std(ks), opts)
    return tuple(o.reshape(b, t, heads, kdim) for o in outs)

  args = (m, static_weight, q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits,
          q_standard, k_standard)
  return _map_batch(local, args, (True, False) + (True,) * 9, 4)
