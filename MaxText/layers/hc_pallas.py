"""Fused Pallas TPU kernel for the stream-major hyper-connection residual update.

new[j] = sum_i mixing[i, j] * state[i] + post[j] * branch, with state [n,B,T,D],
branch [B,T,D], post [n,B,T] and mixing [source, destination, B, T].

XLA lowers the reverse of this update to separate dot/reduce fusions that each
re-read the state. The custom VJP here produces dstate, dbranch and the
per-token coefficient gradients in one pass over (dnew, state, branch).
Arithmetic is FP32 on the VPU; outputs keep the state/branch dtypes.
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32


def _tile(m, requested):
  tile = min(requested, m)
  while m % tile:
    tile //= 2
  return tile


def _update_kernel(n, x_ref, b_ref, c_ref, o_ref):
  c = c_ref[...]
  b = b_ref[...].astype(F32)
  xs = [x_ref[i].astype(F32) for i in range(n)]
  for j in range(n):
    acc = c[:, n * n + j:n * n + j + 1] * b
    for i in range(n):
      acc = acc + c[:, i * n + j:i * n + j + 1] * xs[i]
    o_ref[j] = acc.astype(o_ref.dtype)


def _update_bwd_kernel(n, g_ref, x_ref, b_ref, c_ref, dx_ref, db_ref, dc_ref):
  c = c_ref[...]
  gs = [g_ref[j].astype(F32) for j in range(n)]
  columns = [None] * ((n + 1) * n)
  for i in range(n):
    acc = c[:, i * n:i * n + 1] * gs[0]
    for j in range(1, n):
      acc = acc + c[:, i * n + j:i * n + j + 1] * gs[j]
    dx_ref[i] = acc.astype(dx_ref.dtype)
    x = x_ref[i].astype(F32)
    for j in range(n):
      columns[i * n + j] = jnp.sum(x * gs[j], axis=-1, keepdims=True)
  b = b_ref[...].astype(F32)
  acc = c[:, n * n:n * n + 1] * gs[0]
  for j in range(1, n):
    acc = acc + c[:, n * n + j:n * n + j + 1] * gs[j]
  db_ref[...] = acc.astype(db_ref.dtype)
  for j in range(n):
    columns[n * n + j] = jnp.sum(b * gs[j], axis=-1, keepdims=True)
  dc_ref[...] = jnp.concatenate(columns, axis=-1)


def _specs(n, tile, d, k):
  state = pl.BlockSpec((n, tile, d), lambda t: (0, t, 0))
  row = pl.BlockSpec((tile, d), lambda t: (t, 0))
  coef = pl.BlockSpec((tile, k), lambda t: (t, 0))
  return state, row, coef


def _params():
  return pltpu.CompilerParams(dimension_semantics=('parallel',), vmem_limit_bytes=64 * 1024 * 1024)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _update(x, b, c, tile, interpret):
  return _update_call(x, b, c, tile, interpret)


def _update_call(x, b, c, tile, interpret):
  n, m, d = x.shape
  k = c.shape[-1]
  state, row, coef = _specs(n, tile, d, k)
  return pl.pallas_call(functools.partial(_update_kernel, n), grid=(m // tile,),
                        in_specs=[state, row, coef], out_specs=state,
                        out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                        compiler_params=_params(), interpret=interpret)(x, b, c)


def _update_fwd(x, b, c, tile, interpret):
  return _update_call(x, b, c, tile, interpret), (x, b, c)


def _update_bwd(tile, interpret, residuals, g):
  x, b, c = residuals
  n, m, d = x.shape
  k = c.shape[-1]
  state, row, coef = _specs(n, tile, d, k)
  dx, db, dc = pl.pallas_call(
      functools.partial(_update_bwd_kernel, n), grid=(m // tile,),
      in_specs=[state, state, row, coef], out_specs=[state, row, coef],
      out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct(b.shape, b.dtype),
                 jax.ShapeDtypeStruct(c.shape, F32)],
      compiler_params=_params(), interpret=interpret)(g.astype(x.dtype), x, b, c)
  return dx, db, dc.astype(c.dtype)


_update.defvjp(_update_fwd, _update_bwd)


def _batch_shard(fn, args, specs, out_spec):
  """shard_map over the activation-batch mesh axes (legacy `with mesh` context)."""
  from jax._src import mesh as mesh_lib
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if not mesh.axis_names or mesh.size == 1:
    return fn(*args)
  from flax import linen as nn
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P
  axes = nn.logical_to_mesh_axes(('activation_batch',))[0]
  names = (axes,) if isinstance(axes, str) else tuple(axes or ())
  if any(size > 1 and name not in names for name, size in mesh.shape.items()):
    raise ValueError('hc_pallas supports batch/FSDP-sharded activations only')
  spec = lambda batch_dim, rank: P(*[axes if i == batch_dim else None for i in range(rank)])
  return shard_map(fn, mesh=mesh, in_specs=tuple(spec(dim, a.ndim) for a, dim in zip(args, specs)),
                   out_specs=spec(*out_spec), check_rep=False)(*args)


def mhc_update_pallas(state, branch, post, mixing, *, tile=256, interpret=False):
  """Drop-in for residual_connections.mhc_update_major.

  interpret must be explicit: AOT compiles TPU executables under JAX_PLATFORMS=cpu,
  so a backend check would silently select the interpreter in real training.
  """
  n, b, t, d = state.shape
  coef = jnp.concatenate((mixing.astype(F32).reshape((n * n, b, t)), post.astype(F32)), axis=0)
  coef = jnp.moveaxis(coef, 0, -1)                                        # [B, T, (n+1)n]

  def local(x, br, c):
    nb = x.shape[1]
    m = nb * t
    out = _update(x.reshape(n, m, d), br.astype(x.dtype).reshape(m, d), c.reshape(m, -1),
                  _tile(m, tile), interpret)
    return out.reshape(x.shape)

  return _batch_shard(local, (state, branch, coef), (1, 0, 0), (1, 4))


# ---------------------------------------------------------------------------
# Fused mHC connector: router dot + flat RMS + pre sigmoid + weighted read, one
# pass over the state. The state is also returned unchanged so the residual
# update consumes this output: its dstate then arrives as an output cotangent and
# the reverse kernel writes the summed state gradient once.
# ---------------------------------------------------------------------------
LANES = 128


def _connector_values(n, d, xs, w_ref, p_ref, norm_eps, pre_eps):
  z = jnp.dot(xs[0], w_ref[0], preferred_element_type=F32)
  ss = jnp.sum(xs[0] * xs[0], axis=-1, keepdims=True)
  for i in range(1, n):
    z = z + jnp.dot(xs[i], w_ref[i], preferred_element_type=F32)
    ss = ss + jnp.sum(xs[i] * xs[i], axis=-1, keepdims=True)
  rms = jax.lax.rsqrt(ss / (n * d) + norm_eps)
  p = p_ref[...]
  sig = [jax.nn.sigmoid(rms * z[:, i:i + 1] * p[:, 0:1] + p[:, 1 + i:2 + i]) for i in range(n)]
  pre = [s + pre_eps for s in sig]
  return z, rms, sig, pre


def _connector_kernel(n, d, norm_eps, pre_eps, x_ref, w_ref, p_ref, read_ref, z_ref, rms_ref):
  xs = [x_ref[i].astype(F32) for i in range(n)]
  z, rms, _, pre = _connector_values(n, d, xs, w_ref, p_ref, norm_eps, pre_eps)
  read = pre[0] * xs[0]
  for i in range(1, n):
    read = read + pre[i] * xs[i]
  read_ref[...] = read.astype(read_ref.dtype)
  z_ref[...] = z
  rms_ref[...] = rms


def _connector_bwd_kernel(n, d, norm_eps, pre_eps, x_ref, w_ref, wt_ref, p_ref, dread_ref, dz_ref, drms_ref,
                          dxo_ref, dx_ref, dlogit_ref, da0_ref, dwt_ref):
  @pl.when(pl.program_id(1) == 0)
  def _():
    dwt_ref[...] = jnp.zeros_like(dwt_ref)

  xs = [x_ref[i].astype(F32) for i in range(n)]
  z, rms, sig, pre = _connector_values(n, d, xs, w_ref, p_ref, norm_eps, pre_eps)
  p = p_ref[...]
  dread = dread_ref[...].astype(F32)
  dz = dz_ref[...]
  drms = drms_ref[...]
  lane = jax.lax.broadcasted_iota(jnp.int32, dz.shape, 1)
  dlogits = []
  da0 = jnp.zeros_like(drms)
  for i in range(n):
    dpre = jnp.sum(xs[i] * dread, axis=-1, keepdims=True)
    dlogit = dpre * sig[i] * (1 - sig[i])
    dlogits.append(dlogit)
    dz = dz + jnp.where(lane == i, dlogit * rms * p[:, 0:1], 0.)
    drms = drms + dlogit * z[:, i:i + 1] * p[:, 0:1]
    da0 = da0 + dlogit * rms * z[:, i:i + 1]
  dlogit_ref[...] = jnp.concatenate(dlogits, axis=-1)
  da0_ref[...] = da0
  dss = drms * (-0.5) * rms * rms * rms / (n * d)
  dzt = dz.T                                                              # [LANES, tile]
  for i in range(n):
    dx = (dxo_ref[i].astype(F32) + pre[i] * dread
          + jnp.dot(dz, wt_ref[i], preferred_element_type=F32) + 2 * dss * xs[i])
    dx_ref[i] = dx.astype(dx_ref.dtype)
    dwt_ref[0, i] += jnp.dot(dzt, xs[i], preferred_element_type=F32)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7))
def _connector(x, w, a0, bias, norm_eps, pre_eps, tile, interpret):
  return _connector_fwd(x, w, a0, bias, norm_eps, pre_eps, tile, interpret)[0]


def _pack_params(a0, bias):
  n = bias.shape[0]
  return jnp.zeros((1, LANES), F32).at[0, 0].set(a0.astype(F32)).at[0, 1:1 + n].set(bias.astype(F32))


def _connector_fwd(x, w, a0, bias, norm_eps, pre_eps, tile, interpret):
  n, m, d = x.shape
  wp = jnp.zeros((n, d, LANES), F32).at[..., :w.shape[-1]].set(w.astype(F32))
  p = _pack_params(a0, bias)
  state = pl.BlockSpec((n, tile, d), lambda t: (0, t, 0))
  read, z, rms = pl.pallas_call(
      functools.partial(_connector_kernel, n, d, norm_eps, pre_eps), grid=(m // tile,),
      in_specs=[state, pl.BlockSpec((n, d, LANES), lambda t: (0, 0, 0)), pl.BlockSpec((1, LANES), lambda t: (0, 0))],
      out_specs=[pl.BlockSpec((tile, d), lambda t: (t, 0)), pl.BlockSpec((tile, LANES), lambda t: (t, 0)),
                 pl.BlockSpec((tile, 1), lambda t: (t, 0))],
      out_shape=[jax.ShapeDtypeStruct((m, d), x.dtype), jax.ShapeDtypeStruct((m, LANES), F32),
                 jax.ShapeDtypeStruct((m, 1), F32)],
      compiler_params=_params(), interpret=interpret)(x, wp, p)
  return (read, z[:, :w.shape[-1]], rms[:, 0], x), (x, w, a0, bias)


def _connector_bwd(norm_eps, pre_eps, tile, interpret, residuals, cotangents):
  x, w, a0, bias = residuals
  dread, dz, drms, dxo = cotangents
  n, m, d = x.shape
  e = w.shape[-1]
  wp = jnp.zeros((n, d, LANES), F32).at[..., :e].set(w.astype(F32))
  p = _pack_params(a0, bias)
  tile = min(tile, m // 2)
  half = m // tile // 2
  idx = lambda c, t: c * half + t
  state = pl.BlockSpec((n, tile, d), lambda c, t: (0, idx(c, t), 0))
  dzp = jnp.zeros((m, LANES), F32).at[:, :e].set(dz.astype(F32))
  dx, dlogit, da0, dwt = pl.pallas_call(
      functools.partial(_connector_bwd_kernel, n, d, norm_eps, pre_eps), grid=(2, half),
      in_specs=[state, pl.BlockSpec((n, d, LANES), lambda c, t: (0, 0, 0)),
                pl.BlockSpec((n, LANES, d), lambda c, t: (0, 0, 0)), pl.BlockSpec((1, LANES), lambda c, t: (0, 0)),
                pl.BlockSpec((tile, d), lambda c, t: (idx(c, t), 0)), pl.BlockSpec((tile, LANES), lambda c, t: (idx(c, t), 0)),
                pl.BlockSpec((tile, 1), lambda c, t: (idx(c, t), 0)), state],
      out_specs=[state, pl.BlockSpec((tile, n), lambda c, t: (idx(c, t), 0)),
                 pl.BlockSpec((tile, 1), lambda c, t: (idx(c, t), 0)),
                 pl.BlockSpec((1, n, LANES, d), lambda c, t: (c, 0, 0, 0))],
      out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct((m, n), F32),
                 jax.ShapeDtypeStruct((m, 1), F32), jax.ShapeDtypeStruct((2, n, LANES, d), F32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'arbitrary'),
                                           vmem_limit_bytes=96 * 1024 * 1024),
      interpret=interpret)(x, wp, jnp.swapaxes(wp, 1, 2), p, dread.astype(x.dtype), dzp,
                           drms.astype(F32)[:, None], dxo.astype(x.dtype))
  dw = jnp.swapaxes(jnp.sum(dwt, axis=0)[:, :e], 1, 2).astype(w.dtype)
  return dx, dw, jnp.sum(da0).astype(a0.dtype), jnp.sum(dlogit, axis=0).astype(bias.dtype)


_connector.defvjp(_connector_fwd, _connector_bwd)


def mhc_connector_pallas(state, kernel, a0, bias_pre, norm_eps, pre_eps, *, tile=256, interpret=False):
  """Stream-major mHC read: state [n,B,T,D], kernel (n, D, E) = scale*phi.

  Returns read [B,T,D], raw projection z [E,B,T], flat-RMS factor [B,T] and the state
  itself (to be consumed by the residual update), with pre = sigmoid(rms*z[:n]*a0 + bias) + eps.
  """
  n, b, t, d = state.shape

  def local(x, w, a, bias):
    nb = x.shape[1]
    m = nb * t
    read, z, rms, xo = _connector(x.reshape(n, m, d), w, a, bias, float(norm_eps), float(pre_eps),
                                  _tile(m, tile), interpret)
    return read.reshape(nb, t, d), jnp.moveaxis(z.reshape(nb, t, -1), -1, 0), rms.reshape(nb, t), xo.reshape(x.shape)

  return _batch_shard_multi(local, (state, kernel, a0, bias_pre), (1, None, None, None),
                            ((0, 3), (1, 3), (0, 2), (1, 4)))


def _batch_shard_multi(fn, args, batch_dims, out_specs):
  from jax._src import mesh as mesh_lib
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if not mesh.axis_names or mesh.size == 1:
    return fn(*args)
  from flax import linen as nn
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P
  axes = nn.logical_to_mesh_axes(('activation_batch',))[0]
  names = (axes,) if isinstance(axes, str) else tuple(axes or ())
  if any(size > 1 and name not in names for name, size in mesh.shape.items()):
    raise ValueError('hc_pallas supports batch/FSDP-sharded activations only')
  spec = lambda dim, rank: P(*[axes if i == dim else None for i in range(rank)])
  return shard_map(fn, mesh=mesh, in_specs=tuple(spec(dim, a.ndim) for a, dim in zip(args, batch_dims)),
                   out_specs=tuple(spec(*o) for o in out_specs), check_rep=False)(*args)


# ---------------------------------------------------------------------------
# Fused DHC connector: per-stream LayerNorm folded into the router dot, tanh,
# the read weight sw0 + ws * tanh(logit_0) and the weighted read, in one pass.
# Outputs tanh(logits) [n, M, E] for XLA's post/mixing, plus the state passthrough.
# ---------------------------------------------------------------------------
def _dhc_values(n, d, x, g_ref, p_ref, eps):
  p = p_ref[...]                                       # rows: 0 sG, 1 cB, 2 [ws, sw0_0..sw0_{n-1}]
  sg, cb = p[0:1], p[1:2]
  mean = jnp.sum(x, axis=-1, keepdims=True) / d
  var = jnp.maximum(jnp.sum(x * x, axis=-1, keepdims=True) / d - mean * mean, 0.)
  rsig = jax.lax.rsqrt(var + eps)
  z = jnp.dot(x, g_ref[...], preferred_element_type=F32)
  dyn = jnp.tanh(rsig * (z - mean * sg) + cb)
  return p, sg, mean, rsig, z, dyn


def _dhc_kernel(n, d, eps, x_ref, g_ref, p_ref, read_ref, dyn_ref):
  read = None
  for i in range(n):
    x = x_ref[i].astype(F32)
    p, _, _, _, _, dyn = _dhc_values(n, d, x, g_ref, p_ref, eps)
    dyn_ref[i] = dyn
    w = p[2:3, 1 + i:2 + i] + p[2:3, 0:1] * dyn[:, 0:1]
    read = w * x if read is None else read + w * x
  read_ref[...] = read.astype(read_ref.dtype)


def _dhc_bwd_kernel(n, d, eps, x_ref, g_ref, gt_ref, p_ref, dread_ref, ddyn_ref, dxo_ref,
                    dx_ref, dwr_ref, dws_ref, dsg_ref, dcb_ref, dgt_ref):
  @pl.when(pl.program_id(1) == 0)
  def _():
    dgt_ref[...] = jnp.zeros_like(dgt_ref)

  dread = dread_ref[...].astype(F32)
  dwrs, dws, dsg, dcb = [], 0., 0., 0.
  for i in range(n):
    x = x_ref[i].astype(F32)
    p, sg, mean, rsig, z, dyn = _dhc_values(n, d, x, g_ref, p_ref, eps)
    ws = p[2:3, 0:1]
    w = p[2:3, 1 + i:2 + i] + ws * dyn[:, 0:1]
    dwr = jnp.sum(x * dread, axis=-1, keepdims=True)
    dwrs.append(dwr)
    dws = dws + dwr * dyn[:, 0:1]
    lane = jax.lax.broadcasted_iota(jnp.int32, dyn.shape, 1)
    ddyn = ddyn_ref[i] + jnp.where(lane == 0, ws * dwr, 0.)
    dlogit = ddyn * (1 - dyn * dyn)
    dz = rsig * dlogit
    dmean = -rsig * jnp.sum(dlogit * sg, axis=-1, keepdims=True)
    drsig = jnp.sum(dlogit * (z - mean * sg), axis=-1, keepdims=True)
    dvar = drsig * (-0.5) * rsig * rsig * rsig
    dx = (dxo_ref[i].astype(F32) + w * dread + jnp.dot(dz, gt_ref[...], preferred_element_type=F32)
          + (dmean - 2 * mean * dvar) / d + (2 / d) * dvar * x)
    dx_ref[i] = dx.astype(dx_ref.dtype)
    dgt_ref[0] += jnp.dot(dz.T, x, preferred_element_type=F32)
    dsg = dsg - rsig * mean * dlogit
    dcb = dcb + dlogit
  dwr_ref[...] = jnp.concatenate(dwrs, axis=-1)
  dws_ref[...] = dws
  dsg_ref[...] = dsg
  dcb_ref[...] = dcb


def _dhc_pack(sg, cb, ws, sw0):
  e, n = sg.shape[0], sw0.shape[0]
  p = jnp.zeros((8, LANES), F32).at[0, :e].set(sg.astype(F32)).at[1, :e].set(cb.astype(F32))
  return p.at[2, 0].set(ws.astype(F32).reshape(())).at[2, 1:1 + n].set(sw0.astype(F32))


@functools.partial(jax.custom_vjp, nondiff_argnums=(6, 7, 8))
def _dhc(x, g, sg, cb, ws, sw0, eps, tile, interpret):
  return _dhc_fwd(x, g, sg, cb, ws, sw0, eps, tile, interpret)[0]


def _dhc_fwd(x, g, sg, cb, ws, sw0, eps, tile, interpret):
  n, m, d = x.shape
  e = g.shape[-1]
  gp = jnp.zeros((d, LANES), F32).at[:, :e].set(g.astype(F32))
  state = pl.BlockSpec((n, tile, d), lambda t: (0, t, 0))
  read, dyn = pl.pallas_call(
      functools.partial(_dhc_kernel, n, d, eps), grid=(m // tile,),
      in_specs=[state, pl.BlockSpec((d, LANES), lambda t: (0, 0)), pl.BlockSpec((8, LANES), lambda t: (0, 0))],
      out_specs=[pl.BlockSpec((tile, d), lambda t: (t, 0)), pl.BlockSpec((n, tile, LANES), lambda t: (0, t, 0))],
      out_shape=[jax.ShapeDtypeStruct((m, d), x.dtype), jax.ShapeDtypeStruct((n, m, LANES), F32)],
      compiler_params=_params(), interpret=interpret)(x, gp, _dhc_pack(sg, cb, ws, sw0))
  return (read, dyn[..., :e], x), (x, g, sg, cb, ws, sw0)


def _dhc_bwd(eps, tile, interpret, residuals, cotangents):
  x, g, sg, cb, ws, sw0 = residuals
  dread, ddyn, dxo = cotangents
  n, m, d = x.shape
  e = g.shape[-1]
  gp = jnp.zeros((d, LANES), F32).at[:, :e].set(g.astype(F32))
  tile = min(tile, m // 2)
  half = m // tile // 2
  idx = lambda c, t: c * half + t
  state = pl.BlockSpec((n, tile, d), lambda c, t: (0, idx(c, t), 0))
  token = lambda width: pl.BlockSpec((tile, width), lambda c, t: (idx(c, t), 0))
  ddynp = jnp.zeros((n, m, LANES), F32).at[..., :e].set(ddyn.astype(F32))
  dx, dwr, dws, dsg, dcb, dgt = pl.pallas_call(
      functools.partial(_dhc_bwd_kernel, n, d, eps), grid=(2, half),
      in_specs=[state, pl.BlockSpec((d, LANES), lambda c, t: (0, 0)), pl.BlockSpec((LANES, d), lambda c, t: (0, 0)),
                pl.BlockSpec((8, LANES), lambda c, t: (0, 0)), token(d),
                pl.BlockSpec((n, tile, LANES), lambda c, t: (0, idx(c, t), 0)), state],
      out_specs=[state, token(n), token(1), token(LANES), token(LANES),
                 pl.BlockSpec((1, LANES, d), lambda c, t: (c, 0, 0))],
      out_shape=[jax.ShapeDtypeStruct(x.shape, x.dtype), jax.ShapeDtypeStruct((m, n), F32),
                 jax.ShapeDtypeStruct((m, 1), F32), jax.ShapeDtypeStruct((m, LANES), F32),
                 jax.ShapeDtypeStruct((m, LANES), F32), jax.ShapeDtypeStruct((2, LANES, d), F32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'arbitrary'),
                                           vmem_limit_bytes=96 * 1024 * 1024),
      interpret=interpret)(x, gp, gp.T, _dhc_pack(sg, cb, ws, sw0), dread.astype(x.dtype), ddynp, dxo.astype(x.dtype))
  dg = jnp.sum(dgt, axis=0)[:e].T.astype(g.dtype)
  return (dx, dg, jnp.sum(dsg, axis=0)[:e].astype(sg.dtype), jnp.sum(dcb, axis=0)[:e].astype(cb.dtype),
          jnp.sum(dws).reshape(ws.shape).astype(ws.dtype), jnp.sum(dwr, axis=0).astype(sw0.dtype))


_dhc.defvjp(_dhc_fwd, _dhc_bwd)


def dhc_connector_pallas(state, gained, gained_sum, bias_proj, width_scale, read_static, eps, *,
                         tile=256, interpret=False):
  """Stream-major DHC read. gained = g*[W_alpha, w_beta] (D, n+2); returns read [B,T,D],
  tanh(logits) [n,B,T,n+2] and the state passthrough."""
  n, b, t, d = state.shape

  def local(x, g, sg, cb, ws, sw0):
    nb = x.shape[1]
    m = nb * t
    read, dyn, xo = _dhc(x.reshape(n, m, d), g, sg, cb, ws, sw0, float(eps), _tile(m, tile), interpret)
    return read.reshape(nb, t, d), dyn.reshape(n, nb, t, -1), xo.reshape(x.shape)

  return _batch_shard_multi(local, (state, gained, gained_sum, bias_proj, width_scale, read_static),
                            (1, None, None, None, None, None), ((0, 3), (1, 4), (1, 4)))


# ---------------------------------------------------------------------------
# Sinkhorn on [source, destination, M] blocks: all iterations in one kernel; the
# reverse recomputes the iterates in VMEM and sweeps them backwards. XLA emitted
# ~40 tiny fusions per connector per pass for the unrolled iterations.
# ---------------------------------------------------------------------------
def _sinkhorn_steps(n, logits, iterations, eps):
  """Returns softmax probabilities, the inputs of every normalization and the result."""
  rows = [logits[i] for i in range(n)]                                    # each [destination, tile]
  probs = []
  for i in range(n):
    e = jnp.exp(rows[i] - jnp.max(rows[i], axis=0, keepdims=True))
    probs.append(e / jnp.sum(e, axis=0, keepdims=True))
  x = [p + eps for p in probs]
  inputs = []
  for step in range(2 * iterations - 1):
    inputs.append(x)
    if step % 2 == 0:   # normalize over sources (axis 0 in [i, j])
      s = x[0]
      for i in range(1, n):
        s = s + x[i]
      x = [v / (s + eps) for v in x]
    else:               # normalize over destinations
      x = [v / (jnp.sum(v, axis=0, keepdims=True) + eps) for v in x]
  return probs, inputs, x


def _sinkhorn_kernel(n, iterations, eps, l_ref, o_ref):
  _, _, out = _sinkhorn_steps(n, l_ref[...], iterations, eps)
  for i in range(n):
    o_ref[i] = out[i]


def _sinkhorn_bwd_kernel(n, iterations, eps, l_ref, g_ref, dl_ref):
  probs, inputs, _ = _sinkhorn_steps(n, l_ref[...], iterations, eps)
  g = [g_ref[i] for i in range(n)]
  for step in reversed(range(2 * iterations - 1)):
    x = inputs[step]
    if step % 2 == 0:
      s = x[0]
      for i in range(1, n):
        s = s + x[i]
      inv = 1 / (s + eps)
      dot = g[0] * x[0]
      for i in range(1, n):
        dot = dot + g[i] * x[i]
      g = [(g[i] - dot * inv) * inv for i in range(n)]
    else:
      new = []
      for i in range(n):
        inv = 1 / (jnp.sum(x[i], axis=0, keepdims=True) + eps)
        new.append((g[i] - jnp.sum(g[i] * x[i], axis=0, keepdims=True) * inv) * inv)
      g = new
  for i in range(n):   # softmax over destinations; the +eps has unit derivative
    dl_ref[i] = probs[i] * (g[i] - jnp.sum(g[i] * probs[i], axis=0, keepdims=True))


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3, 4))
def _sinkhorn(logits, iterations, eps, tile, interpret):
  return _sinkhorn_call(logits, iterations, eps, tile, interpret)


def _sinkhorn_call(logits, iterations, eps, tile, interpret):
  n, _, m = logits.shape
  spec = pl.BlockSpec((n, n, tile), lambda t: (0, 0, t))
  return pl.pallas_call(functools.partial(_sinkhorn_kernel, n, iterations, eps), grid=(m // tile,),
                        in_specs=[spec], out_specs=spec, out_shape=jax.ShapeDtypeStruct(logits.shape, F32),
                        compiler_params=_params(), interpret=interpret)(logits)


def _sinkhorn_fwd(logits, iterations, eps, tile, interpret):
  return _sinkhorn_call(logits, iterations, eps, tile, interpret), logits


def _sinkhorn_bwd(iterations, eps, tile, interpret, logits, g):
  n, _, m = logits.shape
  spec = pl.BlockSpec((n, n, tile), lambda t: (0, 0, t))
  return (pl.pallas_call(functools.partial(_sinkhorn_bwd_kernel, n, iterations, eps), grid=(m // tile,),
                         in_specs=[spec, spec], out_specs=spec, out_shape=jax.ShapeDtypeStruct(logits.shape, F32),
                         compiler_params=_params(), interpret=interpret)(logits, g.astype(F32)),)


_sinkhorn.defvjp(_sinkhorn_fwd, _sinkhorn_bwd)


def sinkhorn_pallas(logits, iterations, epsilon, *, tile=512, interpret=False):
  """residual_connections.sinkhorn_major for logits [source, destination, B, T]."""
  n, _, b, t = logits.shape

  def local(l):
    nb = l.shape[2]
    m = nb * t
    return (_sinkhorn(l.astype(F32).reshape(n, n, m), int(iterations), float(epsilon),
                      _tile(m, tile), interpret).reshape(l.shape),)

  return _batch_shard_multi(local, (logits,), (2,), ((2, 4),))[0]
