"""v8t fused BAM core: token-major M [B,T,V,K]; every M contraction is a per-token MXU matmul.

v7 runs the dynamic reads/writes on the VPU (each token has its own M_t, so no MXU operand is
shared across tokens): ~160 VALU cycles per token for a full-M read on v5p. Here M_t [V,K] is the
stationary MXU operand of a per-token matmul (batched dot over the token axis), ~13 cycles/token
for 64 rows. All reads of one layer are linear in M_t, so they collapse into ONE matmul per token:

  read:   Y_t [R,K] = X_t [R,V] @ M_t [V,K]       X stacks every read key; for the full-M
          generalized read R = 4N rows  q: gq*norm(rq) + sg_q S_q, k: likewise, v/o: the gated
          VO key plus the gated static key.
  write:  M'_t   = M_t + A_t^T [V,N] @ C_t [N,K]  A = norm(address), C = sigmoid(logit) norm(content)

Reverses are matmuls too: dX = dY M^T, dM = X^T dY (+ the passthrough cotangent, fused);
dA = C G^T, dC = A G.  Operands are the model dtype (BF16 in training), accumulation FP32.
Kernel I/O stays T-minor as in v7 (keys/gates/factors as [.., T] slabs, Splash-native q/k/v/o);
the token-major operands are built in VMEM. Building keys / splitting Y in XLA instead (the
experimental v8 body, branch claude/bam-pallas-fullm) cost +330 ms of pads/copies on [B,T,N,40/96].
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from layers.bam_pallas import F32, _freeze, _map_batch, _params


def _bdot(x, y, cx, cy):
  """Token-batched matmul: contract x axis cx with y axis cy, batch over axis 0."""
  return jax.lax.dot_general(x, y, (((cx,), (cy,)), ((0,), (0,))), preferred_element_type=F32)


def _token_spec(shape, tile):
  return pl.BlockSpec((None, tile) + tuple(shape), lambda b, i: (b, i) + (0,) * len(shape))


# ---------------------------------------------------------------------------
# v8t: v7's T-minor kernel I/O (keys/gates/factors [.., T] slabs, Splash-native [B,N,T,K] q/k/v/o),
# token-major M [B,T,V,K], per-token MXU matmuls inside the kernel. The token-major key matrices
# X_t^T [V, R] are built by one XLU transpose per v of the T-minor [R, T] key slab plus strided
# VMEM stores; per-token outputs Y_t [R, K] become head-major token rows by strided VMEM loads.
# Rows r = arm * NP + head (arms q, k, v, o; NP = heads padded to 8); full-M reads (C == V) only.

from layers.bam_pallas import _norm, _norm_backward, _pad_rows, _sigmoid, _spec, _whole   # noqa: E402
from layers.bam_pallas_v7 import _from_tokens, _to_tokens   # noqa: E402


def _tm_rows(rq, rk, rr, c):
  """Raw Q/K/V/O rows; rr holds a shared VO key or packed independent V/O keys."""
  if rr.shape[1] == 2 * rq.shape[1]:
    return jnp.concatenate((rq[c].astype(F32), rk[c].astype(F32), rr[c].astype(F32)), axis=0)
  return jnp.concatenate([x[c].astype(F32) for x in (rq, rk, rr, rr)], axis=0)


def _tm_slabs(eps, scale, rq, lq, rk, lk, rr, lo, lv):
  """Per-row inverse RMS norms over C and gates, stacked [R, T]; keys stay in their refs."""
  cdim = rq.shape[0]
  ss = None
  for c in range(cdim):
    x = _tm_rows(rq, rk, rr, c)
    ss = x * x if ss is None else ss + x * x
  inv = jax.lax.rsqrt(ss / cdim + eps)
  g = jnp.concatenate([scale * _sigmoid(l[...]) for l in (lq, lk, lv, lo)], axis=0)
  return inv, g


def _tm_o_mask(om, c, rows):
  """[R, T] multiplier slab of column c: ones on the Q/K/V rows, the O weight multiplier on the O rows."""
  np_ = om.shape[1]
  return jnp.concatenate((jnp.ones((rows - np_, om.shape[2]), F32), om[c].astype(F32)), axis=0)


def _tm_o_scale(os_ref, rows):
  """[R, C] dynamic-key coordinate scale: ones on the Q/K/V rows, the learned O scale [NP, C] on O."""
  o = os_ref[...].astype(F32)
  return jnp.concatenate((jnp.ones((rows - o.shape[0], o.shape[1]), F32), o), axis=0)


def _tm_fill_keys(rq, rk, rr, inv, g, st, sg, xt_scr, om=None, osc=None):
  """xt_scr rows t*C + c = X_t^T[c] (token-major); X^T[c] = g * inv * r[c] + S[:, c] * sg.

  om [C, NP, T]: optional O-row multiplier (address-weight dropout) applied to the whole effective
  O weight, static plus dynamic, after the key RMS norm.
  osc [R, C]: optional per-row coordinate scale of the dynamic (normalized, gated) key only:
  X^T[c] = g * inv * osc[:, c] * r[c] + S[:, c] * sg."""
  gi = g * inv
  cdim = rq.shape[0]
  for c in range(cdim):
    slab = gi * _tm_rows(rq, rk, rr, c)
    if osc is not None:
      slab = slab * osc[:, c:c + 1]
    s_col = st[:, c:c + 1]
    slab = slab + (s_col * sg if sg is not None else s_col)
    if om is not None:
      slab = slab * _tm_o_mask(om, c, slab.shape[0])
    xt_scr[pl.ds(c, slab.shape[1], stride=cdim), :] = _to_tokens(slab)


def _put(scr, p, x):
  """Row block p of a [P*T, L] scratch (contiguous T rows)."""
  t = x.shape[0]
  scr[pl.ds(p * t, t), :] = x


def _chunk(scr, t0, ch, p):
  """Token-major [T*P, L] scratch -> [ch, P, L] matrices of tokens t0..t0+ch (contiguous)."""
  return scr[pl.ds(t0 * p, ch * p), :].reshape(ch, p, scr.shape[1])


def _gather(scr, t0, ch, t):
  """[P*T, L] scratch (row p*T + token) -> [ch, P, L] matrices of tokens t0..t0+ch: aligned [ch, L]
  block loads and one sublane<->major swap in registers (strided per-token loads cost ~1 op per row)."""
  blk = jnp.stack([scr[pl.ds(q * t + t0, ch), :] for q in range(scr.shape[0] // t)], axis=0)
  return jnp.swapaxes(blk, 0, 1)


def _scatter(scr, t0, x, t):
  """Inverse of _gather: x [ch, P, L] -> rows p*T + t0.. of a [P*T, L] scratch."""
  blk = jnp.swapaxes(x, 0, 1)
  for q in range(blk.shape[0]):
    scr[pl.ds(q * t + t0, x.shape[0]), :] = blk[q]


def _shift_gate(sh_ref, t0, ch, rows):
  """[ch, C, rows] lane-broadcast previous-token shares g'[c] of tokens t0..t0+ch (zero if invalid)."""
  cdim = sh_ref.shape[0]
  return jnp.stack([jnp.broadcast_to(sh_ref[:, pl.ds(t0 + i, 1)].astype(F32), (cdim, rows))
                    for i in range(ch)], axis=0)


def _fill_shift_gate(sh_ref, gb_scr):
  """gb_scr rows t*C + c = g'[c, t] broadcast over lanes: one XLU transpose + strided store per c."""
  cdim, t = sh_ref.shape
  lanes = gb_scr.shape[1]
  for c in range(cdim):
    row = jnp.broadcast_to(sh_ref[pl.ds(c, 1), :].astype(F32), (lanes, t))
    gb_scr[pl.ds(c, t, stride=cdim), :] = _to_tokens(row)


def _prev_from(m_ref, mh_ref, t0, mt):
  """M_{t-1} of the loaded chunk mt [ch, V, K]: one extra token (halo at the tile start) + mt[:-1]."""
  first = mh_ref[...][None] if t0 == 0 else m_ref[pl.ds(t0 - 1, 1)]
  return jnp.concatenate((first.astype(mt.dtype), mt[:-1]), axis=0)


def _tm_read_kernel(n, qk_cols, eps, scale, q_scale, gated, ch, shifted=False, qdim=None, masked=False,
                    scaled=False):
  def kernel(m_ref, st_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, *refs):
    if gated:
      sg_ref, refs = refs[0], refs[1:]
    if shifted:
      (sh_ref, mh_ref), refs = refs[:2], refs[2:]
    om = osc = None
    if masked:
      om, refs = refs[0], refs[1:]
    if scaled:
      os_ref, refs = refs[0], refs[1:]
    q_ref, k_ref, v_ref, o_ref, xt_scr, y_scr = refs[:6]
    t, _, kdim = m_ref.shape
    cdim = rq.shape[0]
    np_ = rq.shape[1]                 # heads, optionally padded to a multiple of 8 by the wrapper
    r = 4 * np_
    dt = m_ref.dtype
    inv, g = _tm_slabs(eps, scale, rq, lq, rk, lk, rr, lo, lv)
    sg = sg_ref[...].astype(F32).reshape(r, t) if gated else None
    if scaled:
      osc = _tm_o_scale(os_ref, r)
    _tm_fill_keys(rq, rk, rr, inv, g, st_ref[...].astype(F32), sg, xt_scr, om, osc)
    for t0 in range(0, t, ch):
      if shifted:
        # Y_t = X_t M_read_t, M_read_t = M_t + g'(M_{t-1} - M_t) mixed on the VPU. Per-token gate
        # broadcasts beat a gate scratch here; the dot form (X_t g') M_{t-1} costs a second matmul.
        xt = _chunk(xt_scr, t0, ch, cdim).astype(dt)
        mt = m_ref[pl.ds(t0, ch)].astype(F32)
        mr = mt + _shift_gate(sh_ref, t0, ch, kdim) * (_prev_from(m_ref, mh_ref, t0, mt) - mt)
        y = _bdot(xt, mr.astype(dt), 1, 1)
      else:
        xt = _chunk(xt_scr, t0, ch, cdim).astype(dt)
        y = _bdot(xt, m_ref[pl.ds(t0, ch)], 1, 1)
      _scatter(y_scr, t0, y, t)     # Y_t [R, K] -> rows r*T + t
    lane = jax.lax.broadcasted_iota(jnp.int32, (t, kdim), 1)
    zstd = jnp.zeros((qk_cols, t), F32)
    for a, ref in enumerate((q_ref, k_ref, v_ref, o_ref)):
      for h in range(n):
        y = y_scr[pl.ds((a * np_ + h) * t, t), :]                        # [T, K] token rows
        if a < 2:     # columns >= qk_cols are the standard (rotated) Q/K coordinates
          if qdim is not None and qdim > kdim:
            # Widened Q/K (qk_cols + standard > K): append the standard rows to the first qk_cols
            # read rows and write [T, qdim] directly, so no relayout or concat happens outside.
            rows = jnp.concatenate((_from_tokens(y)[:qk_cols], (qs if a == 0 else ks)[h].astype(F32)), axis=0)
            y = _to_tokens(rows)
          elif qk_cols < kdim:
            std = jnp.concatenate((zstd, (qs if a == 0 else ks)[h].astype(F32)), axis=0)
            y = jnp.where(lane < qk_cols, y, _to_tokens(std))
          if a == 0 and q_scale != 1.0:
            y = y * q_scale
        ref[h] = y.astype(ref.dtype)
  return kernel


def _tm_read_backward_kernel(n, qk_cols, eps, scale, q_scale, gated, ch, pass_m, shifted=False, qdim=None,
                             masked=False, scaled=False):
  def kernel(m_ref, st_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, *refs):
    if gated:
      sg_ref, refs = refs[0], refs[1:]
    if shifted:
      (sh_ref, mh_ref), refs = refs[:2], refs[2:]
    om = osc = None
    if masked:
      om, refs = refs[0], refs[1:]
    if scaled:
      os_ref, refs = refs[0], refs[1:]
    (dq, dk, dv, do), refs = refs[:4], refs[4:]
    if pass_m:
      gp_ref, refs = refs[0], refs[1:]
    (dm_ref, dst_ref, drq, dlq, drk, dlk, drr, dlo, dlv, dqs, dks), refs = refs[:11], refs[11:]
    if gated:
      dsg_ref, refs = refs[0], refs[1:]
    if shifted:
      (dsh_ref, bnd_ref), refs = refs[:2], refs[2:]
    if scaled:
      dos_ref, refs = refs[0], refs[1:]
    xt_scr, dy_scr, dxt_scr, dkey_scr = refs[:4]
    gb_scr = refs[4] if shifted else None
    xm_scr = refs[5 if shifted else 4] if masked else None
    dos_scr = refs[-1] if scaled else None
    t, _, kdim = m_ref.shape
    cdim = rq.shape[0]
    np_ = rq.shape[1]                 # heads, optionally padded to a multiple of 8 by the wrapper
    r = 4 * np_
    dt = m_ref.dtype
    inv, g = _tm_slabs(eps, scale, rq, lq, rk, lk, rr, lo, lv)
    st = st_ref[...].astype(F32)
    sg = sg_ref[...].astype(F32).reshape(r, t) if gated else None
    if scaled:
      osc = _tm_o_scale(os_ref, r)
    _tm_fill_keys(rq, rk, rr, inv, g, st, sg, xt_scr, om, osc)
    if masked:        # token-major O multiplier rows t*C + c, so the per-token dX picks it up
      for c in range(cdim):
        xm_scr[pl.ds(c, t, stride=cdim), :] = _to_tokens(_tm_o_mask(om, c, r))
    # Token-major cotangent rows dY_t [R, K]; the standard Q/K columns leave the matmuls.
    lane = jax.lax.broadcasted_iota(jnp.int32, (t, kdim), 1)
    zrow = jnp.zeros((t, kdim), F32)
    for a, ref in enumerate((dq, dk, dv, do)):
      for h in range(np_):
        rows = a * np_ + h
        if h >= n:
          _put(dy_scr, rows, zrow)
          continue
        y = ref[h].astype(F32)
        if a == 0 and q_scale != 1.0:
          y = y * q_scale
        if a < 2:
          ds = dqs if a == 0 else dks
          if qdim is not None and qdim > kdim:     # widened [T, qdim] cotangent
            ys = _from_tokens(y)
            ds[h] = ys[qk_cols:].astype(ds.dtype)
            local = ys[:qk_cols]
            if qk_cols < kdim:
              local = jnp.concatenate((local, jnp.zeros((kdim - qk_cols, local.shape[1]), F32)), axis=0)
            y = _to_tokens(local)
          else:
            ds[h] = (_from_tokens(y)[qk_cols:].astype(ds.dtype) if qk_cols < kdim
                     else jnp.zeros(ds.shape[1:], ds.dtype))
            y = jnp.where(lane < qk_cols, y, 0.)
        _put(dy_scr, rows, y)
    # Per-token matmuls: dX_t^T = M_t dY_t^T, dM_t = X_t^T dY_t; dS = sum_t sg * dX^T.
    sgt = _to_tokens(sg) if gated else None                                # [T, R]
    dst = jnp.zeros((cdim, r), F32)
    if shifted:
      _fill_shift_gate(sh_ref, gb_scr)
    nxt = None        # shifted: dM_{t-1} share of the first token of the chunk after this one
    for t0 in (range(0, t, ch)[::-1] if shifted else range(0, t, ch)):
      dy = _gather(dy_scr, t0, ch, t).astype(dt)
      if shifted:
        # dX = M_read dY^T, dM_read = X^T dY; dM_t = (1-g'_t) dM_read_t + g'_{t+1} dM_read_{t+1};
        # dg' = sum_k dM_read (M_{t-1} - M_t), reduced over K in registers and stored at lane t.
        # The gate comes from a per-tile scratch (used twice per token); chunks run backwards so
        # g'_{t+1} dM_read_{t+1} carries into the chunk before.
        xt = _chunk(xt_scr, t0, ch, cdim).astype(dt)
        mt = m_ref[pl.ds(t0, ch)].astype(F32)
        d = _prev_from(m_ref, mh_ref, t0, mt) - mt
        gk = _chunk(gb_scr, t0, ch, cdim)
        dxt = _bdot((mt + gk * d).astype(dt), dy, 2, 2)                   # [ch, C, R]
        dmr = _bdot(xt, dy, 2, 1)                                         # [ch, C, K]
        col = jnp.sum(dmr * d, axis=-1, keepdims=True)
        for i in range(ch):
          dsh_ref[:, pl.ds(t0 + i, 1)] = col[i].astype(dsh_ref.dtype)
        e = gk * dmr
        tail = jnp.zeros_like(e[:1]) if nxt is None else nxt
        dmv = dmr - e + jnp.concatenate((e[1:], tail), axis=0)
        nxt = e[:1]
      else:
        xt = _chunk(xt_scr, t0, ch, cdim).astype(dt)
        dxt = _bdot(m_ref[pl.ds(t0, ch)], dy, 2, 2)                       # [ch, C, R]
        dmv = _bdot(xt, dy, 2, 1)                                         # [ch, C, K]
      if masked:      # d(pre-multiplier X) = multiplier * dX: static weight, gates and keys all see it
        dxt = dxt * _chunk(xm_scr, t0, ch, cdim)
      dxt_scr[pl.ds(t0 * cdim, ch * cdim), :] = dxt.reshape(ch * cdim, r)
      dst = dst + jnp.sum(dxt * sgt[t0:t0 + ch][:, None, :] if gated else dxt, axis=0)
      if pass_m:
        dmv = dmv + gp_ref[pl.ds(t0, ch)].astype(F32)
      dm_ref[pl.ds(t0, ch)] = dmv.astype(dm_ref.dtype)
    dst_ref[...] = dst
    if shifted:
      bnd_ref[...] = nxt[0]
    # T-minor key reverse: X^T[c] = g * inv * r[c] + S[:, c] * sg; u = d(normalized key) per arm.
    gacc = jnp.zeros((r, t), F32)      # sum_c dX * r   (gate gradient before * inv)
    sacc = jnp.zeros((r, t), F32)
    independent_vo = rr.shape[1] == 2 * np_
    key_rows = (4 if independent_vo else 3) * np_
    uacc = jnp.zeros((key_rows, t), F32)  # sum_c u * r (RMS-norm reverse)
    gq, gk, gv, go = (g[i * np_:(i + 1) * np_] for i in range(4))
    gio = (g * inv)[3 * np_:] if scaled else None
    for c in range(cdim):
      dx = _from_tokens(dxt_scr[pl.ds(c, t, stride=cdim), :])            # [R, T]
      rows = _tm_rows(rq, rk, rr, c)
      if gated:
        sacc = sacc + dx * st[:, c:c + 1]
      if scaled:      # d osc_o[c] = sum_t dX_o * g inv r[c]; the dynamic O key then sees osc * dX
        dxo = dx[3 * np_:]
        dos_scr[pl.ds(c * np_, np_), :] = dxo * rows[3 * np_:] * gio
        dx = jnp.concatenate((dx[:3 * np_], dxo * osc[3 * np_:, c:c + 1]), axis=0)
      gacc = gacc + dx * rows
      vo_u = ((gv * dx[2 * np_:3 * np_], go * dx[3 * np_:4 * np_]) if independent_vo else
              (gv * dx[2 * np_:3 * np_] + go * dx[3 * np_:4 * np_],))
      u = jnp.concatenate((gq * dx[0:np_], gk * dx[np_:2 * np_]) + vo_u, axis=0)
      dkey_scr[c] = u
      uacc = uacc + u * rows[:key_rows]
    gacc = gacc * inv
    sig = lambda l: _sigmoid(l[...]) * (1 - _sigmoid(l[...]))
    for i, (l, dl) in enumerate(((lq, dlq), (lk, dlk), (lv, dlv), (lo, dlo))):
      dl[...] = (scale * sig(l) * gacc[i * np_:(i + 1) * np_]).astype(dl.dtype)
    key_inv = inv[:key_rows]
    coef = uacc / cdim * key_inv * key_inv
    for c in range(cdim):     # d r[c] = (u[c] - r[c] * mean_c(u r) inv^2) inv
      dr = (dkey_scr[c] - _tm_rows(rq, rk, rr, c)[:key_rows] * coef) * key_inv
      for i, ref in enumerate((drq, drk)):
        ref[c] = dr[i * np_:(i + 1) * np_].astype(ref.dtype)
      drr[c] = dr[2 * np_:].astype(drr.dtype)
    if gated:
      dsg_ref[...] = sacc.reshape(4, np_, t).astype(dsg_ref.dtype)
    if scaled:        # token sums of all C*NP rows as one lane-dense MXU row (no per-c XLU reduce)
      dos_ref[...] = jax.lax.dot_general(
          jnp.ones((8, t), F32), dos_scr[...], (((1,), (1,)), ((), ())),
          precision=jax.lax.Precision.HIGHEST, preferred_element_type=F32)[:1]
  return kernel


def _tm_read_specs(args, tile):
  m, st = args[0], args[1]
  return [_token_spec(m.shape[2:], tile), _whole(st.shape)] + [_spec(x.shape[1:-1], tile) for x in args[2:]]


def _halo_spec(m, tile):
  """The token before each tile (clamped to token 0 for the first tile) as one [V, K] block."""
  return pl.BlockSpec((None, None) + m.shape[2:], lambda bb, i: (bb, jnp.maximum(i * tile - 1, 0), 0, 0))


def _shift_specs(m, sh, tile):
  return [_spec(sh.shape[1:-1], tile), _halo_spec(m, tile)] if sh is not None else []


def _tm_out_spec(n, tile, kdim):
  return pl.BlockSpec((None, n, tile, kdim), lambda bb, i: (bb, 0, i, 0))


def _tm_read_forward_call(args, opts, sh=None, om=None, os=None):
  m = args[0]
  b, t, vdim, kdim = m.shape
  cdim = args[2].shape[1]
  n = opts['heads']
  r = 4 * args[2].shape[2]
  tile = min(opts['tile'], t)
  qdim = opts.get('qdim') or kdim
  qk_out = jax.ShapeDtypeStruct((b, n, t, qdim), m.dtype)
  out = jax.ShapeDtypeStruct((b, n, t, kdim), m.dtype)
  return pl.pallas_call(
      _tm_read_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], opts['q_scale'],
                      len(args) == 12, opts['chunk'], sh is not None, qdim, om is not None, os is not None),
      grid=(b, t // tile),
      in_specs=(_tm_read_specs(args, tile) + _shift_specs(m, sh, tile) + _mask_specs(om, tile)
                + _scale_specs(os)),
      out_specs=[_tm_out_spec(n, tile, qdim)] * 2 + [_tm_out_spec(n, tile, kdim)] * 2,
      out_shape=(qk_out,) * 2 + (out,) * 2,
      scratch_shapes=[pltpu.VMEM((cdim * tile, r), F32), pltpu.VMEM((tile * r, kdim), F32)],
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read')(*args, *(() if sh is None else (sh, m)), *(() if om is None else (om,)),
                            *(() if os is None else (os,)))


def _tm_read_backward_call(args, cts, opts, sh=None, om=None, os=None):
  m, st = args[0], args[1]
  b, t, vdim, kdim = m.shape
  cdim, np_ = args[2].shape[1:3]
  n = opts['heads']
  r = 4 * np_
  tile = min(opts['reverse_tile'], t)
  pass_m = len(cts) == 5
  gated = len(args) == 12
  shifted = sh is not None
  extra = (() if sh is None else (sh, m)) + (() if om is None else (om,)) + (() if os is None else (os,))
  qdim = opts.get('qdim') or kdim
  in_specs = (_tm_read_specs(args, tile) + _shift_specs(m, sh, tile) + _mask_specs(om, tile) + _scale_specs(os)
              + [_tm_out_spec(n, tile, qdim)] * 2 + [_tm_out_spec(n, tile, kdim)] * 2
              + [_token_spec((vdim, kdim), tile)] * pass_m)
  out_specs = ([_token_spec((vdim, kdim), tile), pl.BlockSpec((None, None, cdim, r), lambda bb, i: (bb, i, 0, 0))]
               + [_spec(x.shape[1:-1], tile) for x in args[2:]])
  out_shape = ([jax.ShapeDtypeStruct(m.shape, m.dtype), jax.ShapeDtypeStruct((b, t // tile, cdim, r), F32)]
               + [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in args[2:]])
  scratch = [pltpu.VMEM((cdim * tile, r), F32), pltpu.VMEM((r * tile, kdim), F32),
             pltpu.VMEM((tile * cdim, r), F32),
             pltpu.VMEM((cdim, 2 * np_ + args[6].shape[2], tile), F32)]
  if shifted:
    out_specs += [_spec(sh.shape[1:-1], tile), pl.BlockSpec((None, None, vdim, kdim), lambda bb, i: (bb, i, 0, 0))]
    out_shape += [jax.ShapeDtypeStruct(sh.shape, sh.dtype), jax.ShapeDtypeStruct((b, t // tile, vdim, kdim), F32)]
    scratch += [pltpu.VMEM((tile * cdim, kdim), F32)]     # gate g'[c, t] broadcast over lanes
  if os is not None:  # per-tile partial O scale gradient [1, C*NP] (c-major), summed below
    out_specs += [pl.BlockSpec((None, None, 1, cdim * np_), lambda bb, i: (bb, i, 0, 0))]
    out_shape += [jax.ShapeDtypeStruct((b, t // tile, 1, cdim * np_), F32)]
  if om is not None:
    scratch += [pltpu.VMEM((tile * cdim, r), F32)]        # token-major O multiplier rows
  if os is not None:
    scratch += [pltpu.VMEM((cdim * np_, tile), F32)]      # per-c O scale gradient terms, T-minor
  outs = pl.pallas_call(
      _tm_read_backward_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], opts['q_scale'],
                               gated, opts['chunk'], pass_m, shifted, qdim, om is not None, os is not None),
      grid=(b, t // tile), in_specs=in_specs, out_specs=out_specs, out_shape=tuple(out_shape),
      scratch_shapes=scratch,
      input_output_aliases={len(args) + len(extra) + 4: 0} if pass_m else {},
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read_backward')(*args, *extra, *cts)
  dos = None
  if os is not None:
    outs, dos = outs[:-1], jnp.sum(outs[-1], axis=(0, 1, 2)).reshape(cdim, np_).T.astype(os.dtype)
  dm = outs[0]
  dsh = None
  if shifted:
    outs, (dsh, bnd) = outs[:-2], outs[-2:]
    if t > tile:    # each tile's first-token M_{t-1} cotangent belongs to the previous tile's last token
      idx = slice(tile - 1, t - 1, tile)
      dm = dm.at[:, idx].set((dm[:, idx].astype(F32) + bnd[:, 1:]).astype(dm.dtype))
  return (dm, jnp.sum(outs[1], axis=(0, 1)).T.astype(st.dtype)) + tuple(outs[2:]), dsh, dos


def _mask_specs(om, tile):
  return [] if om is None else [_spec(om.shape[1:-1], tile)]


def _scale_specs(os):
  return [] if os is None else [_whole(os.shape)]


def _tm_args(m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks, sg):
  return (m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks) + (() if sg is None else (sg,))


@partial(jax.custom_vjp, nondiff_argnums=(15,))
def _tm_read(m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks, sg, sh, om, os, opts):
  outs = _tm_read_forward_call(_tm_args(m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks, sg), dict(opts), sh, om, os)
  return outs + (m,)


def _tm_read_fwd(m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks, sg, sh, om, os, opts):
  args = _tm_args(m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks, sg)
  return _tm_read_forward_call(args, dict(opts), sh, om, os) + (m,), (args, sh, om, os)


def _tm_read_bwd(opts, res, cts):
  args, sh, om, os = res
  grads, dsh, dos = _tm_read_backward_call(args, cts, dict(opts), sh, om, os)
  # The O multiplier is a sampled constant (dropout mask): zero cotangent.
  return ((grads if len(args) == 12 else grads + (None,)) + (dsh,)
          + (None if om is None else jnp.zeros_like(om),) + (dos,))


_tm_read.defvjp(_tm_read_fwd, _tm_read_bwd)


def read_tm(m, static_weight, q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits, q_standard,
            k_standard, *, qk_cols, read_epsilon, key_scale, forward_tile=128, reverse_tile=128,
            vmem_mib=None, interpret=False, q_scale=1.0, static_gates=None, chunk=8, pad_heads=False,
            shift=None, o_weight_multiplier=None, o_dynamic_scale=None, **_):
  """v7.read's interface (keys [B,T,N,C], logits [B,T,N], standard QK [B,T,N,Rs]) for M [B,T,V,K] and
  full-M reads (C == V, identity compression). Returns (query, key, value, local_o) [B,T,N,K] and M.

  vo_key may instead be [B,T,2N,V], packing independent V then O keys. The read
  matmul still has4N rows; the reverse keeps the two address cotangents separate.

  shift [B,T,V]: optional previous-token share g' (zero at invalid tokens); every read then sees
  M_t + g'_t (M_{t-1} - M_t), mixed per V row on the VPU inside the kernel (one read dot, two reverse
  dots); the passthrough M stays unmixed. v5p XL kernel-only: read F/B +10%/+10% over the plain read.

  o_dynamic_scale [N,V]: optional learned per-head coordinate scale of the dynamic O key after its RMS
  norm (static O and the V/Q/K keys untouched); its gradient is a per-tile partial summed here."""
  b, t, n, cdim = q_key.shape
  assert cdim == m.shape[2], 'v8t supports full-M reads (identity compression) only'
  # Q/K width = retained M coordinates + standard (rotated) coordinates. It may exceed K (all K NoPE
  # coordinates plus RoPE): the kernel then writes [T, qdim] rows itself (standard rows appended inside).
  qdim = qk_cols + q_standard.shape[-1]
  assert q_standard.shape == k_standard.shape and 0 <= qk_cols <= m.shape[-1] <= qdim, (qk_cols, qdim, m.shape)
  np_ = _pad_rows(n) if pad_heads else n
  opts = _freeze(dict(heads=n, qk_cols=qk_cols, read_epsilon=read_epsilon, key_scale=key_scale,
                      tile=forward_tile, reverse_tile=reverse_tile, vmem_mib=vmem_mib, interpret=interpret,
                      q_scale=float(q_scale), chunk=chunk, **({'qdim': qdim} if qdim > m.shape[-1] else {})))
  padn = lambda x: jnp.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, np_ - n), (0, 0)]) if np_ > n else x
  keys = lambda x: padn(jnp.transpose(x, (0, 3, 2, 1)))                  # [B,T,N,C] -> [B,C,NP,T]
  gate = lambda x: padn(_minor_t(x))                                      # [B,T,N] -> [B,NP,T]
  sw = static_weight.astype(F32)
  st = jnp.concatenate([jnp.pad(sw[a * n:(a + 1) * n], ((0, np_ - n), (0, 0))) for a in range(4)], axis=0)
  assert vo_key.shape[2] in (n, 2 * n), vo_key.shape
  vo_keys = (keys(vo_key) if vo_key.shape[2] == n else
             jnp.concatenate(tuple(keys(x) for x in jnp.split(vo_key, 2, axis=2)), axis=2))
  args = (m, st, keys(q_key), gate(q_logits), keys(k_key), gate(k_logits), vo_keys, gate(o_logits),
          gate(v_logits), _minor_t(q_standard), _minor_t(k_standard))
  gated = bool(static_gates)
  if gated:
    args += (jnp.stack([gate(x) for x in static_gates], axis=1),)       # [B,4,NP,T]
  if shift is not None:
    assert shift.shape == (b, t, cdim), shift.shape
    args += (_minor_t(shift),)                                            # [B,V,T]
  if o_weight_multiplier is not None:
    # o_weight_multiplier [B,T,N,V]: per-token O address-weight multiplier (inverted dropout mask) on
    # the effective O weight (static + gated dynamic, after the key RMS norm); V/Q/K keys untouched.
    assert o_weight_multiplier.shape == (b, t, n, cdim), o_weight_multiplier.shape
    args += (keys(o_weight_multiplier.astype(q_key.dtype)),)              # [B,C,NP,T]
  scaled = o_dynamic_scale is not None
  if scaled:
    assert o_dynamic_scale.shape == (n, cdim), o_dynamic_scale.shape
    os_ = o_dynamic_scale.astype(F32)
    args += (jnp.pad(os_, ((0, np_ - n), (0, 0)), constant_values=1.) if np_ > n else os_,)   # [NP,C]

  def local(m, st, *xs):
    xs = list(xs)
    os = xs.pop() if scaled else None
    om = xs.pop() if o_weight_multiplier is not None else None
    sh = xs.pop() if shift is not None else None
    sg = xs.pop() if gated else None
    return _tm_read(m, st, *xs, sg, sh, om, os, opts)

  outs = _map_batch(local, args, (True, False) + (True,) * (len(args) - 2 - scaled) + (False,) * scaled, 5)
  outs = tuple(checkpoint_name(o, 'bam_read_out') for o in outs[:4]) + (outs[4],)
  return tuple(jnp.transpose(o, (0, 2, 1, 3)) for o in outs[:4]) + (outs[4],)


def _minor_t(x):
  """[B,T,...] -> [B,...,T]."""
  return jnp.moveaxis(x, 1, -1)


def _tm_fill_factors(groups, a_scr, c_scr, eps, t, npt, divisors):
  """a_scr / c_scr rows t*NPT + i: normalized address [V] / gated normalized content [K] of row i."""
  off = 0
  for g in range(0, len(groups), 3):
    content, logits, address = groups[g:g + 3]
    for h in range(content.shape[0]):
      rows = pl.ds(off + h, t, stride=npt)        # token-major rows t*NPT + i
      c = (_norm(content[h], eps, 0) if divisors[g // 3] is None
           else content[h].astype(F32) / divisors[g // 3])
      c_scr[rows, :] = _to_tokens(_sigmoid(logits[pl.ds(h, 1), :]) * c)
      a_scr[rows, :] = _to_tokens(_norm(address[h], eps, 0))
    off += content.shape[0]
  for i in range(off, npt):
    rows = pl.ds(i, t, stride=npt)
    c_scr[rows, :] = jnp.zeros((t, c_scr.shape[1]), F32)
    a_scr[rows, :] = jnp.zeros((t, a_scr.shape[1]), F32)


def _tm_write_kernel(eps, n_groups, ch, divisors):
  def kernel(m_ref, *refs):
    groups = refs[:3 * n_groups]
    out_ref, a_scr, c_scr = refs[3 * n_groups:]
    t, vdim, kdim = m_ref.shape
    npt = a_scr.shape[0] // t
    dt = m_ref.dtype
    _tm_fill_factors(groups, a_scr, c_scr, eps, t, npt, divisors)
    for t0 in range(0, t, ch):
      a = _chunk(a_scr, t0, ch, npt).astype(dt)
      c = _chunk(c_scr, t0, ch, npt).astype(dt)
      out_ref[pl.ds(t0, ch)] = (m_ref[pl.ds(t0, ch)].astype(F32) + _bdot(a, c, 1, 1)).astype(out_ref.dtype)
  return kernel


def _tm_write_backward_kernel(eps, n_groups, ch, divisors):
  def kernel(g_ref, *refs):
    groups = refs[:3 * n_groups]
    outs = refs[3 * n_groups:6 * n_groups]
    a_scr, c_scr, da_scr, dc_scr = refs[6 * n_groups:]
    t, vdim, kdim = g_ref.shape
    npt = a_scr.shape[0] // t
    dt = g_ref.dtype
    _tm_fill_factors(groups, a_scr, c_scr, eps, t, npt, divisors)
    for t0 in range(0, t, ch):
      rows = pl.ds(t0 * npt, ch * npt)
      gm = g_ref[pl.ds(t0, ch)]
      a = _chunk(a_scr, t0, ch, npt).astype(dt)
      c = _chunk(c_scr, t0, ch, npt).astype(dt)
      da_scr[rows, :] = _bdot(c, gm, 2, 2).reshape(ch * npt, vdim)       # C_t G_t^T
      dc_scr[rows, :] = _bdot(a, gm, 2, 1).reshape(ch * npt, kdim)       # A_t G_t
    off = 0
    for g in range(n_groups):
      content, logits, address = groups[3 * g:3 * g + 3]
      dcontent, dlogits, daddress = outs[3 * g:3 * g + 3]
      for h in range(content.shape[0]):
        rows = pl.ds(off + h, t, stride=npt)
        x = content[h]
        dci = _from_tokens(dc_scr[rows, :])
        sg = _sigmoid(logits[pl.ds(h, 1), :])
        normalized = _norm(x, eps, 0) if divisors[g] is None else x.astype(F32) / divisors[g]
        dlogits[pl.ds(h, 1), :] = (sg * (1 - sg) * jnp.sum(dci * normalized, axis=0, keepdims=True)
                                   ).astype(dlogits.dtype)
        dcontent[h] = (_norm_backward(x, sg * dci, eps, 0) if divisors[g] is None
                       else sg * dci / divisors[g]).astype(dcontent.dtype)
        daddress[h] = _norm_backward(address[h], _from_tokens(da_scr[rows, :]), eps, 0).astype(daddress.dtype)
      off += content.shape[0]
  return kernel


def _tm_write_scratch(groups, tile, vdim, kdim, flat=False):
  npt = _pad_rows(sum(a.shape[1] for a in groups[2::3]))   # padded rows measured faster here
  del flat        # row-major [rows*T] factors and per-chunk token-major results share the shape
  return [pltpu.VMEM((tile * npt, vdim), F32), pltpu.VMEM((tile * npt, kdim), F32)]


def _tm_write_forward_call(m, groups, opts):
  b, t, vdim, kdim = m.shape
  tile = min(opts['tile'], t)
  mspec = _token_spec((vdim, kdim), tile)
  return pl.pallas_call(
      _tm_write_kernel(opts['epsilon'], len(groups) // 3, opts['chunk'], opts['content_divisors']),
      grid=(b, t // tile), in_specs=[mspec] + [_spec(x.shape[1:-1], tile) for x in groups],
      out_specs=mspec, out_shape=jax.ShapeDtypeStruct(m.shape, m.dtype),
      scratch_shapes=_tm_write_scratch(groups, tile, vdim, kdim), input_output_aliases={0: 0},
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_write')(m, *groups)


def _tm_write_backward_call(g, groups, opts):
  b, t, vdim, kdim = g.shape
  tile = min(opts['reverse_tile'], t)
  specs = [_spec(x.shape[1:-1], tile) for x in groups]
  return pl.pallas_call(
      _tm_write_backward_kernel(opts['epsilon'], len(groups) // 3, opts['chunk'], opts['content_divisors']),
      grid=(b, t // tile), in_specs=[_token_spec((vdim, kdim), tile)] + specs, out_specs=specs,
      out_shape=tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in groups),
      scratch_shapes=_tm_write_scratch(groups, tile, vdim, kdim) + _tm_write_scratch(groups, tile, vdim, kdim, True),
      interpret=opts['interpret'],
      compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_write_backward')(g, *groups)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _tm_write(m, opts, *groups):
  return _tm_write_forward_call(m, groups, dict(opts))


def _tm_write_fwd(m, opts, *groups):
  return _tm_write_forward_call(m, groups, dict(opts)), groups


def _tm_write_bwd(opts, groups, g):
  return (g,) + tuple(_tm_write_backward_call(g, groups, dict(opts)))


_tm_write.defvjp(_tm_write_fwd, _tm_write_bwd)


def write_tm(m, groups, *, epsilon, forward_tile=128, reverse_tile=128, vmem_mib=None, interpret=False,
             chunk=8, content_divisors=None, **_):
  """M [B,T,V,K] + sum over groups of (content [B,T,N,K], logits [B,T,N], address [B,T,N,V])."""
  content_divisors = tuple(content_divisors) if content_divisors is not None else (None,) * len(groups)
  if len(content_divisors) != len(groups) or any(
      x is not None and (not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x <= 0)
      for x in content_divisors):
    raise ValueError('Each write group requires None (RMS) or a positive finite fixed divisor')
  opts = _freeze(dict(epsilon=epsilon, tile=forward_tile, reverse_tile=reverse_tile, vmem_mib=vmem_mib,
                      interpret=interpret, chunk=chunk, content_divisors=content_divisors))
  flat = tuple(_minor_t(x) for grp in groups for x in grp)
  return _map_batch(lambda m, *xs: _tm_write(m, opts, *xs), (m,) + flat, (True,) * (1 + len(flat)), 1)
