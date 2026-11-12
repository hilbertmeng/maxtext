"""Pallas TPU path for Full Attention Residuals (attnres_pallas).

Each read l computes, per token, score_k = rsig_k * (x_k . u_l), with
rsig_k = rsqrt(mean(x_k^2) + eps) and u_l = key_scale_l * query_l, then
out_l = sum_k softmax_k(score) x_k over the k < l history sources.

Through autodiff every read adds a full-size cotangent into each earlier source,
one read-modify-write per (read, source) pair plus separate reduction passes. Here
the source gradients live in per-source accumulators G = (G_0, ..., G_{K-1}), a tuple
threaded through the stack as a passthrough value (separate buffers, so updates never
force XLA to copy a stacked [K, B, T, D] array):
  * register(G, x_k, k): identity on G; its reverse hands G[k] to x_k.
  * read(G, xs, u) -> (out, G): reads see xs through stop_gradient; the reverse kernel
    reads each source once and adds its contribution into G[k] (one read + one write).
Math and parameters are unchanged; accumulation stays in the source dtype, as autodiff's.
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


def _scores(x_refs, u, eps, d):
  """Per-source score columns (tile, 1) and rsig, t = x.u; sources are re-read from VMEM refs."""
  out = []
  for r in x_refs:
    x = r[...].astype(F32)
    ss = jnp.sum(x * x, axis=-1, keepdims=True)
    t = jnp.sum(x * u, axis=-1, keepdims=True)
    rsig = jax.lax.rsqrt(ss / d + eps)
    out.append((rsig * t, rsig, t))
  return out


def _softmax(scores):
  m = scores[0]
  for s in scores[1:]:
    m = jnp.maximum(m, s)
  e = [jnp.exp(s - m) for s in scores]
  den = e[0]
  for v in e[1:]:
    den = den + v
  inv = 1 / den
  return [v * inv for v in e]


def _read_kernel(k, d, eps, *refs):
  x_refs, u_ref, o_ref = refs[:k], refs[k], refs[k + 1]
  u = u_ref[...]
  w = _softmax([s for s, _, _ in _scores(x_refs, u, eps, d)])
  acc = w[0] * x_refs[0][...].astype(F32)
  for i in range(1, k):
    acc = acc + w[i] * x_refs[i][...].astype(F32)
  o_ref[...] = acc.astype(o_ref.dtype)


def _read_bwd_kernel(k, d, eps, *refs):
  x_refs = refs[:k]
  u_ref, dout_ref = refs[k:k + 2]
  g_in_refs = refs[k + 2:2 * k + 2]
  g_refs, du_ref = refs[2 * k + 2:3 * k + 2], refs[3 * k + 2]

  @pl.when(pl.program_id(1) == 0)
  def _():
    du_ref[...] = jnp.zeros_like(du_ref)

  u = u_ref[...]
  dout = dout_ref[...].astype(F32)
  parts = _scores(x_refs, u, eps, d)
  w = _softmax([s for s, _, _ in parts])
  gs = [jnp.sum(dout * r[...].astype(F32), axis=-1, keepdims=True) for r in x_refs]
  total = w[0] * gs[0]
  for i in range(1, k):
    total = total + w[i] * gs[i]
  du = None
  for i in range(k):
    _, rsig, t = parts[i]
    x = x_refs[i][...].astype(F32)
    a = w[i] * (gs[i] - total) * rsig                     # dscore_i * rsig_i
    dx = w[i] * dout + a * u - (a * t * rsig * rsig / d) * x
    g_refs[i][...] = (g_in_refs[i][...].astype(F32) + dx).astype(g_refs[i].dtype)
    contrib = jnp.sum(a * x, axis=0, keepdims=True)
    du = contrib if du is None else du + contrib
  du_ref[0] += du


def _params(two_d=False):
  return pltpu.CompilerParams(
      dimension_semantics=('parallel', 'arbitrary') if two_d else ('parallel',),
      vmem_limit_bytes=60 * 1024 * 1024)   # v5p scoped VMEM capacity is ~64 MiB


def _read_call(xs, u, eps, tile, interpret):
  k = len(xs)
  m, d = xs[0].shape
  row = pl.BlockSpec((tile, d), lambda t: (t, 0))
  return pl.pallas_call(functools.partial(_read_kernel, k, d, eps), grid=(m // tile,),
                        in_specs=[row] * k + [pl.BlockSpec((1, d), lambda t: (0, 0))], out_specs=row,
                        out_shape=jax.ShapeDtypeStruct((m, d), xs[0].dtype),
                        compiler_params=_params(), interpret=interpret)(*xs, u)


def _read_bwd_call(xs, u, dout, gs, eps, tile, interpret):
  """gs: the K per-source accumulators of this read's sources; returns their updated values."""
  k = len(xs)
  m, d = xs[0].shape
  tile = min(tile, m // 2)
  half = m // tile // 2
  idx = lambda c, t: c * half + t
  row = pl.BlockSpec((tile, d), lambda c, t: (idx(c, t), 0))
  out = pl.pallas_call(
      functools.partial(_read_bwd_kernel, k, d, eps), grid=(2, half),
      in_specs=[row] * k + [pl.BlockSpec((1, d), lambda c, t: (0, 0)), row] + [row] * k,
      out_specs=[row] * k + [pl.BlockSpec((1, 1, d), lambda c, t: (c, 0, 0))],
      out_shape=[jax.ShapeDtypeStruct(g.shape, g.dtype) for g in gs] + [jax.ShapeDtypeStruct((2, 1, d), F32)],
      # No input_output_aliases: the kernel reads and writes each accumulator either way, and
      # inside remat XLA copied every non-donatable aliased input first (~0.8k slab copies/step).
      compiler_params=_params(True),
      interpret=interpret)(*xs, u, dout, *gs)
  return tuple(out[:k]), out[k].sum(axis=(0, 1))


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5))
def _read(g, xs, u, eps, tile, interpret):
  return _read_call(xs, u, eps, tile, interpret), g


def _read_fwd(g, xs, u, eps, tile, interpret):
  return (_read_call(xs, u, eps, tile, interpret), g), (xs, u)


def _read_bwd(eps, tile, interpret, residuals, cotangents):
  xs, u = residuals
  dout, dg = cotangents
  k = len(xs)
  head, du = _read_bwd_call(xs, u, dout.astype(xs[0].dtype), tuple(g.astype(xs[0].dtype) for g in dg[:k]),
                            eps, tile, interpret)
  dg = head + tuple(dg[k:])
  # Source cotangents travel through G; xs arrive through stop_gradient.
  return dg, tuple(jnp.zeros_like(x) for x in xs), du.reshape(u.shape).astype(u.dtype)


_read.defvjp(_read_fwd, _read_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(2,))
def register(g, x, k):
  """Identity on the accumulator G; its reverse returns G[k] as x's cotangent."""
  del x, k
  return g


def _register_fwd(g, x, k):
  return g, jnp.zeros((0,), x.dtype)   # residuals must be arrays; carries x's dtype


def _register_bwd(k, like, dg):
  return dg, dg[k].astype(like.dtype)   # dg is a tuple of per-source accumulators


register.defvjp(_register_fwd, _register_bwd)


def _shard(fn, args, batch_dims, out_dims):
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
    raise ValueError('attnres_pallas supports batch/FSDP-sharded activations only')
  spec = lambda dim, rank: P(*[axes if i == dim else None for i in range(rank)])
  in_specs = tuple(tuple(spec(dim, a.ndim) for a in arg) if isinstance(arg, tuple) else spec(dim, arg.ndim)
                   for arg, dim in zip(args, batch_dims))
  out_specs = tuple(spec(dim, rank) for dim, rank in out_dims)
  return shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_rep=False)(*args)


def read(g, sources, u, eps, *, tile=None, interpret=False):
  """g: tuple of K_total per-source accumulators [B,T,D]; sources: tuple of [B,T,D]; u = key_scale*query."""
  b, t, d = sources[0].shape
  xs = tuple(jax.lax.stop_gradient(s) for s in sources)
  if tile is None:   # reverse holds K sources plus K accumulator rows (in and out), double-buffered
    tile = 32 if len(sources) <= 24 else 16

  def local(g, xs, u):
    nb = xs[0].shape[0]
    m = nb * t
    out, g2 = _read(tuple(a.reshape(m, d) for a in g), tuple(x.reshape(m, d) for x in xs),
                    u.astype(F32).reshape(1, d), float(eps), _tile(m, tile), interpret)
    return (out.reshape(nb, t, d),) + tuple(a.reshape(nb, t, d) for a in g2)

  out = _shard(local, (g, xs, u), (0, 0, None), ((0, 3),) * (1 + len(g)))
  return out[0], tuple(out[1:])
