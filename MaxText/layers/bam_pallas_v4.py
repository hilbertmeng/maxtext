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
