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
  """Raw key rows c of the four arms (q, k, v, o; v and o share the VO key): [R, T] FP32."""
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


def _tm_fill_keys(rq, rk, rr, inv, g, st, sg, xt_scr):
  """xt_scr rows t*C + c = X_t^T[c] (token-major); X^T[c] = g * inv * r[c] + S[:, c] * sg."""
  gi = g * inv
  cdim = rq.shape[0]
  for c in range(cdim):
    slab = gi * _tm_rows(rq, rk, rr, c)
    s_col = st[:, c:c + 1]
    slab = slab + (s_col * sg if sg is not None else s_col)
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


def _tm_read_kernel(n, qk_cols, eps, scale, q_scale, gated, ch):
  def kernel(m_ref, st_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, *refs):
    if gated:
      sg_ref, refs = refs[0], refs[1:]
    q_ref, k_ref, v_ref, o_ref, xt_scr, y_scr = refs
    t, _, kdim = m_ref.shape
    cdim = rq.shape[0]
    np_ = rq.shape[1]                 # heads, optionally padded to a multiple of 8 by the wrapper
    r = 4 * np_
    dt = m_ref.dtype
    inv, g = _tm_slabs(eps, scale, rq, lq, rk, lk, rr, lo, lv)
    sg = sg_ref[...].astype(F32).reshape(r, t) if gated else None
    _tm_fill_keys(rq, rk, rr, inv, g, st_ref[...].astype(F32), sg, xt_scr)
    for t0 in range(0, t, ch):
      xt = _chunk(xt_scr, t0, ch, cdim).astype(dt)
      _scatter(y_scr, t0, _bdot(xt, m_ref[pl.ds(t0, ch)], 1, 1), t)     # Y_t [R, K] -> rows r*T + t
    lane = jax.lax.broadcasted_iota(jnp.int32, (t, kdim), 1)
    zstd = jnp.zeros((qk_cols, t), F32)
    for a, ref in enumerate((q_ref, k_ref, v_ref, o_ref)):
      for h in range(n):
        y = y_scr[pl.ds((a * np_ + h) * t, t), :]                        # [T, K] token rows
        if a < 2:     # columns >= qk_cols are the standard (rotated) Q/K coordinates
          std = jnp.concatenate((zstd, (qs if a == 0 else ks)[h].astype(F32)), axis=0)
          y = jnp.where(lane < qk_cols, y, _to_tokens(std))
          if a == 0 and q_scale != 1.0:
            y = y * q_scale
        ref[h] = y.astype(ref.dtype)
  return kernel


def _tm_read_backward_kernel(n, qk_cols, eps, scale, q_scale, gated, ch, pass_m):
  def kernel(m_ref, st_ref, rq, lq, rk, lk, rr, lo, lv, qs, ks, *refs):
    if gated:
      sg_ref, refs = refs[0], refs[1:]
    (dq, dk, dv, do), refs = refs[:4], refs[4:]
    if pass_m:
      gp_ref, refs = refs[0], refs[1:]
    (dm_ref, dst_ref, drq, dlq, drk, dlk, drr, dlo, dlv, dqs, dks), refs = refs[:11], refs[11:]
    if gated:
      dsg_ref, refs = refs[0], refs[1:]
    xt_scr, dy_scr, dxt_scr, dkey_scr = refs
    t, _, kdim = m_ref.shape
    cdim = rq.shape[0]
    np_ = rq.shape[1]                 # heads, optionally padded to a multiple of 8 by the wrapper
    r = 4 * np_
    dt = m_ref.dtype
    inv, g = _tm_slabs(eps, scale, rq, lq, rk, lk, rr, lo, lv)
    st = st_ref[...].astype(F32)
    sg = sg_ref[...].astype(F32).reshape(r, t) if gated else None
    _tm_fill_keys(rq, rk, rr, inv, g, st, sg, xt_scr)
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
          (dqs if a == 0 else dks)[h] = _from_tokens(y)[qk_cols:].astype(dqs.dtype)
          y = jnp.where(lane < qk_cols, y, 0.)
        _put(dy_scr, rows, y)
    # Per-token matmuls: dX_t^T = M_t dY_t^T, dM_t = X_t^T dY_t; dS = sum_t sg * dX^T.
    sgt = _to_tokens(sg) if gated else None                                # [T, R]
    dst = jnp.zeros((cdim, r), F32)
    for t0 in range(0, t, ch):
      dy = _gather(dy_scr, t0, ch, t).astype(dt)
      xt = _chunk(xt_scr, t0, ch, cdim).astype(dt)
      dxt = _bdot(m_ref[pl.ds(t0, ch)], dy, 2, 2)                         # [ch, C, R]
      dxt_scr[pl.ds(t0 * cdim, ch * cdim), :] = dxt.reshape(ch * cdim, r)
      dst = dst + jnp.sum(dxt * sgt[t0:t0 + ch][:, None, :] if gated else dxt, axis=0)
      dmv = _bdot(xt, dy, 2, 1)                                           # [ch, C, K]
      if pass_m:
        dmv = dmv + gp_ref[pl.ds(t0, ch)].astype(F32)
      dm_ref[pl.ds(t0, ch)] = dmv.astype(dm_ref.dtype)
    dst_ref[...] = dst
    # T-minor key reverse: X^T[c] = g * inv * r[c] + S[:, c] * sg; u = d(normalized key) per arm.
    gacc = jnp.zeros((r, t), F32)      # sum_c dX * r   (gate gradient before * inv)
    sacc = jnp.zeros((r, t), F32)
    uacc = jnp.zeros((3 * np_, t), F32)  # sum_c u * r  (RMS-norm reverse)
    gq, gk, gv, go = (g[i * np_:(i + 1) * np_] for i in range(4))
    for c in range(cdim):
      dx = _from_tokens(dxt_scr[pl.ds(c, t, stride=cdim), :])            # [R, T]
      rows = _tm_rows(rq, rk, rr, c)
      gacc = gacc + dx * rows
      if gated:
        sacc = sacc + dx * st[:, c:c + 1]
      u = jnp.concatenate((gq * dx[0:np_], gk * dx[np_:2 * np_],
                           gv * dx[2 * np_:3 * np_] + go * dx[3 * np_:4 * np_]), axis=0)
      dkey_scr[c] = u
      uacc = uacc + u * rows[:3 * np_]
    gacc = gacc * inv
    sig = lambda l: _sigmoid(l[...]) * (1 - _sigmoid(l[...]))
    for i, (l, dl) in enumerate(((lq, dlq), (lk, dlk), (lv, dlv), (lo, dlo))):
      dl[...] = (scale * sig(l) * gacc[i * np_:(i + 1) * np_]).astype(dl.dtype)
    inv3 = inv[:3 * np_]
    coef = uacc / cdim * inv3 * inv3
    for c in range(cdim):     # d r[c] = (u[c] - r[c] * mean_c(u r) inv^2) inv
      dr = (dkey_scr[c] - _tm_rows(rq, rk, rr, c)[:3 * np_] * coef) * inv3
      for i, ref in enumerate((drq, drk, drr)):
        ref[c] = dr[i * np_:(i + 1) * np_].astype(ref.dtype)
    if gated:
      dsg_ref[...] = sacc.reshape(4, np_, t).astype(dsg_ref.dtype)
  return kernel


def _tm_read_specs(args, tile):
  m, st = args[0], args[1]
  return [_token_spec(m.shape[2:], tile), _whole(st.shape)] + [_spec(x.shape[1:-1], tile) for x in args[2:]]


def _tm_out_spec(n, tile, kdim):
  return pl.BlockSpec((None, n, tile, kdim), lambda bb, i: (bb, 0, i, 0))


def _tm_read_forward_call(args, opts):
  m = args[0]
  b, t, vdim, kdim = m.shape
  cdim = args[2].shape[1]
  n = opts['heads']
  r = 4 * args[2].shape[2]
  tile = min(opts['tile'], t)
  out = jax.ShapeDtypeStruct((b, n, t, kdim), m.dtype)
  return pl.pallas_call(
      _tm_read_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], opts['q_scale'],
                      len(args) == 12, opts['chunk']),
      grid=(b, t // tile), in_specs=_tm_read_specs(args, tile), out_specs=[_tm_out_spec(n, tile, kdim)] * 4,
      out_shape=(out,) * 4,
      scratch_shapes=[pltpu.VMEM((cdim * tile, r), F32), pltpu.VMEM((tile * r, kdim), F32)],
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read')(*args)


def _tm_read_backward_call(args, cts, opts):
  m, st = args[0], args[1]
  b, t, vdim, kdim = m.shape
  cdim, np_ = args[2].shape[1:3]
  n = opts['heads']
  r = 4 * np_
  tile = min(opts['reverse_tile'], t)
  pass_m = len(cts) == 5
  gated = len(args) == 12
  in_specs = (_tm_read_specs(args, tile) + [_tm_out_spec(n, tile, kdim)] * 4
              + [_token_spec((vdim, kdim), tile)] * pass_m)
  out_specs = ([_token_spec((vdim, kdim), tile), pl.BlockSpec((None, None, cdim, r), lambda bb, i: (bb, i, 0, 0))]
               + [_spec(x.shape[1:-1], tile) for x in args[2:]])
  out_shape = ([jax.ShapeDtypeStruct(m.shape, m.dtype), jax.ShapeDtypeStruct((b, t // tile, cdim, r), F32)]
               + [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in args[2:]])
  outs = pl.pallas_call(
      _tm_read_backward_kernel(n, opts['qk_cols'], opts['read_epsilon'], opts['key_scale'], opts['q_scale'],
                               gated, opts['chunk'], pass_m),
      grid=(b, t // tile), in_specs=in_specs, out_specs=out_specs, out_shape=tuple(out_shape),
      scratch_shapes=[pltpu.VMEM((cdim * tile, r), F32), pltpu.VMEM((r * tile, kdim), F32),
                      pltpu.VMEM((tile * cdim, r), F32), pltpu.VMEM((cdim, 3 * np_, tile), F32)],
      input_output_aliases={len(args) + 4: 0} if pass_m else {},
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_read_backward')(*args, *cts)
  return (outs[0], jnp.sum(outs[1], axis=(0, 1)).T.astype(st.dtype)) + tuple(outs[2:])


def _tm_args(m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks, sg):
  return (m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks) + (() if sg is None else (sg,))


@partial(jax.custom_vjp, nondiff_argnums=(12,))
def _tm_read(m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks, sg, opts):
  outs = _tm_read_forward_call(_tm_args(m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks, sg), dict(opts))
  return outs + (m,)


def _tm_read_fwd(m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks, sg, opts):
  args = _tm_args(m, st, rq, lq, rk, lk, rr, lo, lv, qs, ks, sg)
  return _tm_read_forward_call(args, dict(opts)) + (m,), args


def _tm_read_bwd(opts, args, cts):
  grads = _tm_read_backward_call(args, cts, dict(opts))
  return grads if len(args) == 12 else grads + (None,)


_tm_read.defvjp(_tm_read_fwd, _tm_read_bwd)


def read_tm(m, static_weight, q_key, q_logits, k_key, k_logits, vo_key, o_logits, v_logits, q_standard,
            k_standard, *, qk_cols, read_epsilon, key_scale, forward_tile=128, reverse_tile=128,
            vmem_mib=None, interpret=False, q_scale=1.0, static_gates=None, chunk=8, pad_heads=False, **_):
  """v7.read's interface (keys [B,T,N,C], logits [B,T,N], standard QK [B,T,N,Rs]) for M [B,T,V,K] and
  full-M reads (C == V, identity compression). Returns (query, key, value, local_o) [B,T,N,K] and M."""
  b, t, n, cdim = q_key.shape
  assert cdim == m.shape[2], 'v8t supports full-M reads (identity compression) only'
  np_ = _pad_rows(n) if pad_heads else n
  opts = _freeze(dict(heads=n, qk_cols=qk_cols, read_epsilon=read_epsilon, key_scale=key_scale,
                      tile=forward_tile, reverse_tile=reverse_tile, vmem_mib=vmem_mib, interpret=interpret,
                      q_scale=float(q_scale), chunk=chunk))
  padn = lambda x: jnp.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, np_ - n), (0, 0)]) if np_ > n else x
  keys = lambda x: padn(jnp.transpose(x, (0, 3, 2, 1)))                  # [B,T,N,C] -> [B,C,NP,T]
  gate = lambda x: padn(_minor_t(x))                                      # [B,T,N] -> [B,NP,T]
  sw = static_weight.astype(F32)
  st = jnp.concatenate([jnp.pad(sw[a * n:(a + 1) * n], ((0, np_ - n), (0, 0))) for a in range(4)], axis=0)
  args = (m, st, keys(q_key), gate(q_logits), keys(k_key), gate(k_logits), keys(vo_key), gate(o_logits),
          gate(v_logits), _minor_t(q_standard), _minor_t(k_standard))
  if static_gates:
    args += (jnp.stack([gate(x) for x in static_gates], axis=1),)       # [B,4,NP,T]

  def local(m, st, *xs):
    sg = xs[9] if len(xs) == 10 else None
    return _tm_read(m, st, *xs[:9], sg, opts)

  outs = _map_batch(local, args, (True, False) + (True,) * (len(args) - 2), 5)
  outs = tuple(checkpoint_name(o, 'bam_read_out') for o in outs[:4]) + (outs[4],)
  return tuple(jnp.transpose(o, (0, 2, 1, 3)) for o in outs[:4]) + (outs[4],)


def _minor_t(x):
  """[B,T,...] -> [B,...,T]."""
  return jnp.moveaxis(x, 1, -1)


def _tm_fill_factors(groups, a_scr, c_scr, eps, t, npt):
  """a_scr / c_scr rows t*NPT + i: normalized address [V] / gated normalized content [K] of row i."""
  off = 0
  for g in range(0, len(groups), 3):
    content, logits, address = groups[g:g + 3]
    for h in range(content.shape[0]):
      rows = pl.ds(off + h, t, stride=npt)        # token-major rows t*NPT + i
      c_scr[rows, :] = _to_tokens(_sigmoid(logits[pl.ds(h, 1), :]) * _norm(content[h], eps, 0))
      a_scr[rows, :] = _to_tokens(_norm(address[h], eps, 0))
    off += content.shape[0]
  for i in range(off, npt):
    rows = pl.ds(i, t, stride=npt)
    c_scr[rows, :] = jnp.zeros((t, c_scr.shape[1]), F32)
    a_scr[rows, :] = jnp.zeros((t, a_scr.shape[1]), F32)


def _tm_write_kernel(eps, n_groups, ch):
  def kernel(m_ref, *refs):
    groups = refs[:3 * n_groups]
    out_ref, a_scr, c_scr = refs[3 * n_groups:]
    t, vdim, kdim = m_ref.shape
    npt = a_scr.shape[0] // t
    dt = m_ref.dtype
    _tm_fill_factors(groups, a_scr, c_scr, eps, t, npt)
    for t0 in range(0, t, ch):
      a = _chunk(a_scr, t0, ch, npt).astype(dt)
      c = _chunk(c_scr, t0, ch, npt).astype(dt)
      out_ref[pl.ds(t0, ch)] = (m_ref[pl.ds(t0, ch)].astype(F32) + _bdot(a, c, 1, 1)).astype(out_ref.dtype)
  return kernel


def _tm_write_backward_kernel(eps, n_groups, ch):
  def kernel(g_ref, *refs):
    groups = refs[:3 * n_groups]
    outs = refs[3 * n_groups:6 * n_groups]
    a_scr, c_scr, da_scr, dc_scr = refs[6 * n_groups:]
    t, vdim, kdim = g_ref.shape
    npt = a_scr.shape[0] // t
    dt = g_ref.dtype
    _tm_fill_factors(groups, a_scr, c_scr, eps, t, npt)
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
        dlogits[pl.ds(h, 1), :] = (sg * (1 - sg) * jnp.sum(dci * _norm(x, eps, 0), axis=0, keepdims=True)
                                   ).astype(dlogits.dtype)
        dcontent[h] = _norm_backward(x, sg * dci, eps, 0).astype(dcontent.dtype)
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
      _tm_write_kernel(opts['epsilon'], len(groups) // 3, opts['chunk']),
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
      _tm_write_backward_kernel(opts['epsilon'], len(groups) // 3, opts['chunk']),
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
             chunk=8, **_):
  """M [B,T,V,K] + sum over groups of (content [B,T,N,K], logits [B,T,N], address [B,T,N,V])."""
  opts = _freeze(dict(epsilon=epsilon, tile=forward_tile, reverse_tile=reverse_tile, vmem_mib=vmem_mib,
                      interpret=interpret, chunk=chunk))
  flat = tuple(_minor_t(x) for grp in groups for x in grp)
  return _map_batch(lambda m, *xs: _tm_write(m, opts, *xs), (m,) + flat, (True,) * (1 + len(flat)), 1)
