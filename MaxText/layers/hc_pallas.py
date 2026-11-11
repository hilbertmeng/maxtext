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
