"""v7 fused BAM core: k-major end to end, no relayout in any MXU operand.

Same equations as layers/bam_pallas.py (per-tile reference). The v-major kernels keep each
head's [K,T] slab with k on sublanes, so every static MXU dot over the head/row axis j needs
a (j,k) sublane transpose ([90,96,128] -> [90,12288]): ~half of the v5p read-reverse bundles.
Here all per-token math runs on k-major [N,T] head slabs, so

* M is carried k-major [B,K,V,T]; M_k is a [V,T] slab and lane-concatenating k slabs is free,
  so static reads, compression, dM = S^T D and dS = D M^T are plain MXU dots;
* outputs and their cotangents are k-major [B,K,N,T] (XLA transposes them to token-major,
  as it does for the v-major [B,N,K,T] outputs);
* write factors stay head-major [B,N,K,T] / [B,N,V,T] (transposed by XLA, not in-kernel).

Arithmetic is FP32 inside a kernel; MXU operands and kernel outputs are BF16.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

import os as _os

from layers.bam_pallas import (F32, _acc, _freeze, _map_batch, _minor, _norm, _norm_backward,
                               _pad_rows, _padded_static_weight, _params, _sigmoid, _spec, _tile,
                               _whole)


_ABL = set(_os.environ.get('BAM_PALLAS_ABLATE', '').split(','))


def _keys(rq, lq, rk, lk, rr, eps, scale):
  """C-major keys [C,N,T]: normalized over C, gated per head."""
  nq, nk, nr = (_norm(r, eps, 0) for r in (rq, rk, rr))
  gq = scale * _sigmoid(lq)
  gk = scale * _sigmoid(lk)
  return gq[None] * nq, gk[None] * nk, nr, gq, gk, nq, nk


def _concat_k(ref, k0, k1):
  """Lane-concatenated k-major slabs ref[k0:k1] -> [rows, (k1-k0)*T]."""
  return jnp.concatenate([ref[k] for k in range(k0, k1)], axis=1)


# ---------------------------------------------------------------------------
# Read.

def _read_kernel(heads, qk_cols, eps, scale, kb, db):
  def kernel(m_ref, sw_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, q_ref, k_ref, v_ref, o_ref,
             key_scr, st_scr):
    n = heads
    np_ = _pad_rows(n)
    kdim, _, t = m_ref.shape
    cdim = rq.shape[0]
    mc0 = 4 * np_
    kq, kk, nr, *_ = _keys(rq[...], lq[...], rk[...], lk[...], rr[...], eps, scale)
    key_scr[0] = kq
    key_scr[1] = kk
    key_scr[2] = nr
    gv = scale * _sigmoid(lv[...])
    go = scale * _sigmoid(lo[...])
    sw = sw_ref[...].astype(m_ref.dtype)
    for d0 in range(0, kdim, db):
      d1 = min(d0 + db, kdim)
      st_scr[:, pl.ds(d0 * t, (d1 - d0) * t)] = jnp.dot(sw, _concat_k(m_ref, d0, d1),
                                                         preferred_element_type=F32)
    for k0 in range(0, kdim, kb):
      ks_ = range(k0, min(k0 + kb, kdim))
      acc = {}
      for c in range(cdim):
        keys = [key_scr[r, c] for r in range(3)]
        for k in ks_:
          row = st_scr[pl.ds(mc0 + c, 1), pl.ds(k * t, t)]
          for r in range(3):
            if r < 2 and k >= qk_cols:
              continue
            acc[r, k] = _acc(acc.get((r, k)), keys[r] * row)
      for k in ks_:
        lanes = pl.ds(k * t, t)
        v_ref[k] = (acc[2, k] * gv + st_scr[pl.ds(2 * np_, n), lanes]).astype(v_ref.dtype)
        o_ref[k] = (acc[2, k] * go + st_scr[pl.ds(3 * np_, n), lanes]).astype(o_ref.dtype)
        if k < qk_cols:
          q_ref[k] = (acc[0, k] + st_scr[pl.ds(0, n), lanes]).astype(q_ref.dtype)
          k_ref[k] = (acc[1, k] + st_scr[pl.ds(np_, n), lanes]).astype(k_ref.dtype)
        else:
          q_ref[k] = qs[k - qk_cols].astype(q_ref.dtype)
          k_ref[k] = ks[k - qk_cols].astype(k_ref.dtype)
  return kernel


def _read_specs(args, tile):
  m = args[0]
  return ([_spec(m.shape[1:-1], tile), _whole(args[1].shape)]
          + [_spec(x.shape[1:-1], tile) for x in args[2:]])


def _read_forward_call(args, opts):
  m, sw = args[0], args[1]
  b, kdim, _, t = m.shape
  c, n = args[2].shape[1:3]
  tile = _tile(t, opts['forward_tile'])
  out = jax.ShapeDtypeStruct((b, kdim, n, t), m.dtype)
  return pl.pallas_call(
      _read_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], opts['k_block'],
                   opts['dot_block']),
      grid=(b, t // tile), in_specs=_read_specs(args, tile), out_specs=[_spec((kdim, n), tile)] * 4,
      out_shape=(out,) * 4,
      scratch_shapes=[pltpu.VMEM((3, c, n, tile), F32), pltpu.VMEM((sw.shape[0], kdim * tile), F32)],
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read')(*args)


def _blocks(lo, hi, body, rolled):
  """Loop over 8-row k blocks: rolled (bounds Mosaic's load hoisting; no carries) or unrolled."""
  if not rolled:
    for j in range(lo, hi):
      body(j)
  elif hi > lo:
    jax.lax.fori_loop(lo, hi, lambda j, c: (body(j), c)[1], 0)


def _read_backward_kernel(heads, qk_cols, eps, scale, db, rolled):
  def kernel(m_ref, sw_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, dq, dk, dv, do,
             dm_ref, dsw_ref, drq, dlq, drk, dlk, drr, dlo, dlv, dqs, dks,
             key_scr, mc_scr, dyvo, dkey_scr, gacc, d2):
    dt = m_ref.dtype
    n = heads
    np_ = _pad_rows(n)
    kdim, _, t = m_ref.shape
    cdim = rq.shape[0]
    mc0 = 4 * np_
    assert qk_cols % 8 == 0 and kdim % 8 == 0
    kq, kk, nr, gq, gk, nq, nkn = _keys(rq[...], lq[...], rk[...], lk[...], rr[...], eps, scale)
    key_scr[0] = kq
    key_scr[1] = kk
    key_scr[2] = nr
    dkey_scr[...] = jnp.zeros(dkey_scr.shape, F32)
    gacc[...] = jnp.zeros(gacc.shape, F32)
    sgv, sgo = _sigmoid(lv[...]), _sigmoid(lo[...])
    sw = sw_ref[...].astype(dt)
    for d0 in range(0, kdim, db):
      d1 = min(d0 + db, kdim)
      mc_scr[:, pl.ds(d0 * t, (d1 - d0) * t)] = jnp.dot(sw[mc0:mc0 + cdim], _concat_k(m_ref, d0, d1),
                                                        preferred_element_type=F32)
    lanes = lambda k: pl.ds(k * t if isinstance(k, int) else pl.multiple_of(k * t, t), t)
    zpad = jnp.zeros((np_ - n, t), F32)
    pad = lambda x: jnp.concatenate((x, zpad), axis=0) if np_ > n else x
    zgrp = jnp.zeros((np_, t), F32)

    # Pass A: VO forward state, gate gradients, combined VO cotangent; static cotangent
    # rows of D (dq, dk, dv, do) written once, FP32, padded head groups.
    def pass_a(j, with_qk):
      gv = scale * _sigmoid(lv[...])
      go = scale * _sigmoid(lo[...])
      av, ao = gacc[0], gacc[1]
      for kk_ in range(8):
        k = j * 8 + kk_
        ln = lanes(k)
        y = None
        for c in range(cdim):
          y = _acc(y, key_scr[2, c] * mc_scr[pl.ds(c, 1), ln])
        dvk = dv[k].astype(F32)
        dok = do[k].astype(F32)
        av = av + dvk * y
        ao = ao + dok * y
        dyvo[k] = dvk * gv + dok * go
        d2[pl.ds(2 * np_, np_), ln] = pad(dvk)
        d2[pl.ds(3 * np_, np_), ln] = pad(dok)
        d2[pl.ds(0, np_), ln] = pad(dq[k].astype(F32)) if with_qk else zgrp
        d2[pl.ds(np_, np_), ln] = pad(dk[k].astype(F32)) if with_qk else zgrp
      gacc[0] = av
      gacc[1] = ao

    _blocks(0, qk_cols // 8, lambda j: pass_a(j, True), rolled)
    _blocks(qk_cols // 8, kdim // 8, lambda j: pass_a(j, False), rolled)
    dlv[...] = (scale * sgv * (1 - sgv) * gacc[0]).astype(dlv.dtype)
    dlo[...] = (scale * sgo * (1 - sgo) * gacc[1]).astype(dlo.dtype)
    for r in range(kdim - qk_cols):
      dqs[r] = dq[qk_cols + r].astype(dqs.dtype)
      dks[r] = dk[qk_cols + r].astype(dks.dtype)

    # Pass B, per compressed row c: key gradients and dMc[c, k] = sum_n key[n, c] dy_k[n]
    # (sublane reductions written straight into D's compressed rows).
    def pass_b(c, j, with_qk):
      acc = [dkey_scr[r, c] for r in range(3)]
      kqc, kkc, nrc = key_scr[0, c], key_scr[1, c], key_scr[2, c]
      for kk_ in range(8):
        k = j * 8 + kk_
        ln = lanes(k)
        row = mc_scr[pl.ds(c, 1), ln]
        dyv = dyvo[k]
        acc[2] = acc[2] + dyv * row
        s_ = nrc * dyv
        if with_qk:
          dyq = d2[pl.ds(0, n), ln]
          dyk = d2[pl.ds(np_, n), ln]
          acc[0] = acc[0] + dyq * row
          acc[1] = acc[1] + dyk * row
          s_ = s_ + kqc * dyq + kkc * dyk
        d2[pl.ds(mc0 + c, 1), ln] = jnp.sum(s_, axis=0, keepdims=True)
      dkey_scr[2, c] = acc[2]
      if with_qk:
        dkey_scr[0, c] = acc[0]
        dkey_scr[1, c] = acc[1]

    for c in range(cdim):
      _blocks(0, qk_cols // 8, lambda j, c=c: pass_b(c, j, True), rolled)
      _blocks(qk_cols // 8, kdim // 8, lambda j, c=c: pass_b(c, j, False), rolled)

    for i, (r, l, dr, dl, g, nrm) in enumerate(((rq, lq, drq, dlq, gq, nq), (rk, lk, drk, dlk, gk, nkn))):
      dkey = dkey_scr[i]
      sg = _sigmoid(l[...])
      dl[...] = (scale * sg * (1 - sg) * jnp.sum(dkey * nrm, axis=0)).astype(dl.dtype)
      dr[...] = _norm_backward(r[...], g[None] * dkey, eps, 0).astype(dr.dtype)
    drr[...] = _norm_backward(rr[...], dkey_scr[2], eps, 0).astype(drr.dtype)

    # dM_k = S'^T D_k and dS' = sum_k D_k M_k^T: plain MXU dots on lane-concatenated k blocks.
    dsw = None
    for d0 in range(0, kdim, db):
      d1 = min(d0 + db, kdim)
      blk = d2[:, pl.ds(d0 * t, (d1 - d0) * t)].astype(dt)
      dmk = jax.lax.dot_general(sw, blk, (((0,), (0,)), ((), ())), preferred_element_type=F32)
      for k in range(d0, d1):
        dm_ref[k] = dmk[:, (k - d0) * t:(k - d0 + 1) * t].astype(dm_ref.dtype)
      dsw = _acc(dsw, jax.lax.dot_general(blk, _concat_k(m_ref, d0, d1), (((1,), (1,)), ((), ())),
                                          preferred_element_type=F32))
    dsw_ref[...] = dsw.astype(dsw_ref.dtype)
  return kernel


def _read_backward_call(args, cts, opts):
  m, sw = args[0], args[1]
  b, kdim, v, t = m.shape
  c, n = args[2].shape[1:3]
  tile = _tile(t, opts['reverse_tile'])
  j = sw.shape[0]
  in_specs = _read_specs(args, tile) + [_spec((kdim, n), tile)] * 4
  out_specs = ([_spec((kdim, v), tile), pl.BlockSpec((None, None, j, v), lambda bb, i: (bb, i, 0, 0))]
               + [_spec(x.shape[1:-1], tile) for x in args[2:]])
  out_shape = ([jax.ShapeDtypeStruct(m.shape, m.dtype), jax.ShapeDtypeStruct((b, t // tile, j, v), F32)]
               + [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in args[2:]])
  scratch = [pltpu.VMEM((3, c, n, tile), F32), pltpu.VMEM((c, kdim * tile), F32),
             pltpu.VMEM((kdim, n, tile), F32), pltpu.VMEM((3, c, n, tile), F32),
             pltpu.VMEM((2, n, tile), F32), pltpu.VMEM((j, kdim * tile), F32)]
  outs = pl.pallas_call(
      _read_backward_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], opts['dot_block'],
                            opts['rolled']),
      grid=(b, t // tile), in_specs=in_specs, out_specs=out_specs, out_shape=tuple(out_shape),
      scratch_shapes=scratch, interpret=opts['interpret'],
      compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read_backward')(*args, *cts)
  return (outs[0], jnp.sum(outs[1], axis=(0, 1)).astype(sw.dtype)) + tuple(outs[2:])


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
         forward_tile=128, reverse_tile=128, vmem_mib=None, interpret=False,
         k_block=4, dot_block=16, rolled=True, **_):
  """m is k-major [B,K,V,T]; other arguments token-major as in layers.bam_pallas.read.

  Returns token-major (query, key, value, local_o), each [B,T,N,K]."""
  heads = q_key.shape[2]
  opts = _freeze(dict(heads=heads, qk_cols=qk_cols, read_epsilon=read_epsilon, key_scale=key_scale,
                      forward_tile=forward_tile, reverse_tile=reverse_tile, vmem_mib=vmem_mib,
                      interpret=interpret, k_block=k_block, dot_block=dot_block, rolled=rolled))
  cmajor = lambda x: jnp.transpose(x, (0, 3, 2, 1))       # [B,T,N,X] -> [B,X,N,T]

  def local(m, sw, qk, ql, kk, kl, vo, ol, vl, qs, ks):
    outs = _read(m, _padded_static_weight(sw, heads), cmajor(qk), _minor(ql), cmajor(kk), _minor(kl),
                 cmajor(vo), _minor(ol), _minor(vl), cmajor(qs), cmajor(ks), opts)
    return tuple(cmajor(o) for o in outs)                  # [B,K,N,T] -> [B,T,N,K]

  args = (m, static_weight, q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits,
          q_standard, k_standard)
  return _map_batch(local, args, (True, False) + (True,) * 9, 4)


# ---------------------------------------------------------------------------
# Write. Factors head-major: content [N,K,T], logits [N,T], address [N,V,T].

def _fill_factors(groups, c_scr, a_scr, eps):
  """c_scr[i*K + k] gated normalized content rows; a_scr[i] normalized address [V,T]."""
  off = 0
  for g in range(0, len(groups), 3):
    content, logits, address = groups[g:g + 3]
    nh, kdim = content.shape[0], content.shape[1]
    for h in range(nh):
      c_scr[pl.ds((off + h) * kdim, kdim), :] = _sigmoid(logits[pl.ds(h, 1), :]) * _norm(content[h], eps, 0)
      a_scr[off + h] = _norm(address[h], eps, 0)
    off += nh
  return off


def _write_kernel(eps, n_groups, kb):
  def kernel(m_ref, *refs):
    groups = refs[:3 * n_groups]
    out_ref, c_scr, a_scr = refs[3 * n_groups:]
    kdim = m_ref.shape[0]
    ntot = _fill_factors(groups, c_scr, a_scr, eps)
    for k0 in range(0, kdim, kb):
      ks_ = range(k0, min(k0 + kb, kdim))
      acc = {}
      for i in range(ntot):
        a = a_scr[i]
        for k in ks_:
          acc[k] = _acc(acc.get(k), c_scr[pl.ds(i * kdim + k, 1), :] * a)
      for k in ks_:
        out_ref[k] = (m_ref[k].astype(F32) + acc[k]).astype(out_ref.dtype)
  return kernel


def _write_backward_kernel(eps, n_groups, ab, cb):
  def kernel(g_ref, *refs):
    groups = refs[:3 * n_groups]
    outs = refs[3 * n_groups:6 * n_groups]
    c_scr, a_scr, gf, dc_scr, da_scr = refs[6 * n_groups:6 * n_groups + 5]
    kdim, vdim = g_ref.shape[0], g_ref.shape[1]
    ntot = _fill_factors(groups, c_scr, a_scr, eps)
    for k in range(kdim):
      gf[k] = g_ref[k].astype(F32)
    # dA[i] = sum_k C[i, k] G_k.
    for i0 in range(0, ntot, ab):
      hb = range(i0, min(i0 + ab, ntot))
      acc = {}
      for k in range(kdim):
        gk = gf[k]
        for i in hb:
          acc[i] = _acc(acc.get(i), c_scr[pl.ds(i * kdim + k, 1), :] * gk)
      for i in hb:
        da_scr[i] = acc[i]
    if cb:
      # dC[i] = sum_v A[i, v] G[v] on v-major [K,T] slabs (staged by strided FP32 row loads):
      # broadcast-accumulate like dA instead of one sublane reduction per (i, k).
      g2 = refs[6 * n_groups + 5]
      for v in range(vdim):
        g2[v] = gf[:, v, :]
      for i0 in range(0, ntot, cb):
        hb = range(i0, min(i0 + cb, ntot))
        acc = {}
        for v in range(vdim):
          gv = g2[v]
          for i in hb:
            acc[i] = _acc(acc.get(i), a_scr[i, pl.ds(v, 1), :] * gv)
        for i in hb:
          dc_scr[pl.ds(i * kdim, kdim), :] = acc[i]
    else:
      # dC[i, k] = sum_v A[i, v] G_k[v]; rows assembled per 8-row group.
      for i in range(ntot):
        a = a_scr[i]
        for k0 in range(0, kdim, 8):
          rows = [jnp.sum(a * gf[k], axis=0, keepdims=True) for k in range(k0, min(k0 + 8, kdim))]
          dc_scr[pl.ds(i * kdim + k0, len(rows)), :] = jnp.concatenate(rows, axis=0)
    off = 0
    for g in range(n_groups):
      content, logits, address = groups[3 * g:3 * g + 3]
      dcontent, dlogits, daddress = outs[3 * g:3 * g + 3]
      nh = content.shape[0]
      for h in range(nh):
        x = content[h]
        dci = dc_scr[pl.ds((off + h) * kdim, kdim), :]
        sg = _sigmoid(logits[pl.ds(h, 1), :])
        dlogits[pl.ds(h, 1), :] = (sg * (1 - sg) * jnp.sum(dci * _norm(x, eps, 0), axis=0, keepdims=True)
                                   ).astype(dlogits.dtype)
        dcontent[h] = _norm_backward(x, sg * dci, eps, 0).astype(dcontent.dtype)
        daddress[h] = _norm_backward(address[h], da_scr[off + h], eps, 0).astype(daddress.dtype)
      off += nh
  return kernel


def _write_group_specs(groups, tile):
  return [_spec(x.shape[1:-1], tile) for x in groups]


def _factor_scratch(groups, tile, kdim, vdim):
  ntot = sum(a.shape[1] for a in groups[2::3])
  return ntot, [pltpu.VMEM((ntot * kdim, tile), F32), pltpu.VMEM((ntot, vdim, tile), F32)]


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


def _write_backward_call(g, groups, opts):
  b, kdim, vdim, t = g.shape
  tile = _tile(t, opts['reverse_tile'])
  ntot, scratch = _factor_scratch(groups, tile, kdim, vdim)
  scratch = scratch + [pltpu.VMEM((kdim, vdim, tile), F32), pltpu.VMEM((ntot * kdim, tile), F32),
                       pltpu.VMEM((ntot, vdim, tile), F32)]
  if opts['dc_block']:
    scratch.append(pltpu.VMEM((vdim, kdim, tile), F32))
  specs = _write_group_specs(groups, tile)
  return pl.pallas_call(
      _write_backward_kernel(opts['epsilon'], len(groups) // 3, opts['address_block'], opts['dc_block']),
      grid=(b, t // tile), in_specs=[_spec((kdim, vdim), tile)] + specs, out_specs=specs,
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


def write(m, groups, *, epsilon, forward_tile=128, reverse_tile=128, vmem_mib=None, interpret=False,
          k_block=4, address_block=8, dc_block=3, **_):
  """m is k-major [B,K,V,T]; groups of (content [B,T,N,K], logits [B,T,N], address [B,T,N,V])."""
  opts = _freeze(dict(epsilon=epsilon, forward_tile=forward_tile, reverse_tile=reverse_tile,
                      vmem_mib=vmem_mib, interpret=interpret, k_block=k_block,
                      address_block=address_block, dc_block=dc_block))
  flat = tuple(x for grp in groups for x in grp)

  def local(m, *xs):
    return _write(m, opts, *(_minor(x) for x in xs))

  return _map_batch(local, (m,) + flat, (True,) * (1 + len(flat)), 1)
