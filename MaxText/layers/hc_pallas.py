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
