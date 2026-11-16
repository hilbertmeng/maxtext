"""Shared Pallas math/layout tools; independent of retired kernel versions.

Moved without changing operation order or dtypes. Legacy kernels are kept in
bam_pallas.py / bam_pallas_v7.py and are not imported by the V5 runtime.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

F32 = jnp.float32


def _norm(x, epsilon, axis):
  f = x.astype(F32)
  return f * jax.lax.rsqrt(jnp.mean(f * f, axis=axis, keepdims=True) + epsilon)


def _norm_backward(x, cotangent, epsilon, axis):
  f = x.astype(F32)
  u = cotangent.astype(F32)
  inv = jax.lax.rsqrt(jnp.mean(f * f, axis=axis, keepdims=True) + epsilon)
  return (u - f * (jnp.mean(u * f, axis=axis, keepdims=True) * inv * inv)) * inv


def _sigmoid(x):
  return jax.nn.sigmoid(x.astype(F32))


def _spec(shape, tile):
  return pl.BlockSpec((None,) + tuple(shape) + (tile,), lambda b, i: (b,) + (0,) * len(shape) + (i,))


def _whole(shape):
  return pl.BlockSpec(tuple(shape), lambda b, i, n=len(shape): (0,) * n)


def _params(semantics, vmem_mib):
  kwargs = dict(dimension_semantics=semantics)
  if vmem_mib:
    kwargs['vmem_limit_bytes'] = int(vmem_mib) * 1024 * 1024
  return pltpu.CompilerParams(**kwargs)


def _pad_rows(n):
  return -(-n // 8) * 8


def _freeze(d):
  return tuple(sorted(d.items()))


def _map_batch(fn, args, batch_args, n_out):
  from jax._src import mesh as mesh_lib
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if not mesh.axis_names or mesh.size == 1:
    return fn(*args)
  from flax import linen as nn
  from jax.experimental.shard_map import shard_map
  from jax.sharding import PartitionSpec as P
  axes = nn.logical_to_mesh_axes(('activation_batch',))[0]
  names = (axes,) if isinstance(axes, str) else (axes or ())
  if not names or any(size > 1 and name not in names for name, size in mesh.shape.items()):
    raise ValueError('BAM Pallas core supports batch/FSDP mesh axes only')
  spec = P(axes)
  out_specs = spec if n_out == 1 else (spec,) * n_out
  return shard_map(fn, mesh=mesh, in_specs=tuple(spec if b else P() for b in batch_args),
                   out_specs=out_specs, check_rep=False)(*args)


def static_weight(static_q, static_k, static_v, static_o, compression):
  """Stack [V,N] static keys and the [V,C] compression into the [4N+C, V] read weight."""
  return jnp.concatenate((static_q, static_k, static_v, static_o, compression), axis=1).T


def _to_tokens(x):
  """[K, T] FP32 slab -> [T, K] via one XLU transpose of the 128-row padded slab."""
  kdim = x.shape[0]
  if kdim % 128:
    x = jnp.concatenate((x, jnp.zeros((128 - kdim % 128, x.shape[1]), x.dtype)), axis=0)
  return x.T[:, :kdim]


def _from_tokens(x):
  """[T, K] slab -> [K, T] (lane-padded to 128 before the XLU transpose)."""
  kdim = x.shape[1]
  if kdim % 128:
    x = jnp.concatenate((x, jnp.zeros((x.shape[0], 128 - kdim % 128), x.dtype)), axis=1)
  return x.T[:kdim]
