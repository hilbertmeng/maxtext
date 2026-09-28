"""Inline rank-H static/dynamic write and analytic pullback.

Only layer writes share D between the two terms. Embedding seed writes do not.
Modes separate the forward accumulator schedule from the reverse compute unit.
All temporaries and subprograms below remain inside the caller's Pallas kernel.
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def parts(a, d, g, s, epsilon):
  # Token-minor: A[H,K,T], D[H,V,T], gate[H,T].
  af, df = a.astype(jnp.float32), d.astype(jnp.float32)
  ai = jax.lax.rsqrt(jnp.mean(af*af, axis=1, keepdims=True)+epsilon)
  di = jax.lax.rsqrt(jnp.mean(df*df, axis=1, keepdims=True)+epsilon)
  an = af*ai
  c = s.astype(jnp.float32)[:, :, None] + an*(g.astype(jnp.float32)[:, None, :]*di)
  return af, df, ai, di, an, c


def forward(m, a, d, g, s, epsilon, mode):
  _, df, _, _, _, c = parts(a, d, g, s, epsilon)
  h, k, t = a.shape
  v = d.shape[1]
  schedule, _ = mode.split('_')
  if schedule == 'head':
    acc = jnp.zeros(m.shape, jnp.float32)
    for j in range(h):
      cj = jax.lax.slice_in_dim(c, j, j+1, axis=0).reshape(k,t)
      dj = jax.lax.slice_in_dim(df, j, j+1, axis=0).reshape(v,t)
      acc = acc+cj[:,None,:]*dj[None,:,:]
    return (m.astype(jnp.float32)+acc).astype(m.dtype)
  rows = int(schedule.removeprefix('row'))
  if k % rows:
    raise ValueError('Output row block must divide the memory key dimension')
  def scoped(out):
    def step(i, _):
      ci = jax.lax.dynamic_slice_in_dim(c, i*rows, rows, axis=1)
      mi = jax.lax.dynamic_slice_in_dim(m, i*rows, rows, axis=0)
      acc = jnp.zeros((rows, v, t), jnp.float32)
      for j in range(h):
        cj = jax.lax.slice_in_dim(ci, j, j+1, axis=0).reshape(rows,t)
        dj = jax.lax.slice_in_dim(df, j, j+1, axis=0).reshape(v,t)
        acc = acc+cj[:,None,:]*dj[None,:,:]
      out[pl.ds(i*rows, rows), :, :] = (mi.astype(jnp.float32)+acc).astype(m.dtype)
    jax.lax.fori_loop(0, k//rows, step, None)
    return out[...]
  return pl.run_scoped(scoped, pltpu.VMEM(m.shape, m.dtype))


def reverse(a, d, g, s, matrix_grad, epsilon, mode, **unused):
  # Match joint's token-major ABI. The VPU branch restores token SIMD lanes.
  a, d, g = a.transpose(1, 2, 0), d.transpose(1, 2, 0), g.T
  af, df, ai, di, an, c = parts(a, d, g, s, epsilon)
  h, k, t = a.shape
  v = d.shape[1]
  backend = mode.split('_')[1]
  if backend == 'mxu':
    G = matrix_grad
    square = jnp.concatenate((
        jnp.concatenate((jnp.zeros((t, k, k), G.dtype), G), axis=2),
        jnp.concatenate((G.swapaxes(1, 2), jnp.zeros((t, v, v), G.dtype)), axis=2)), axis=1)
    left = jnp.concatenate((c.astype(d.dtype), d), axis=1).transpose(2, 0, 1)
    product = jnp.einsum('thd,tdc->thc', left, square,
                        preferred_element_type=jnp.float32).transpose(1, 2, 0)
    dc, dd = product[:, :k], product[:, k:]
  elif backend == 'vpu':
    G = matrix_grad.transpose(1, 2, 0).astype(jnp.float32)
    def scoped(dc_ref, dd_ref):
      def head(j, _):
        dj = jax.lax.dynamic_index_in_dim(df, j, axis=0, keepdims=False)
        cj = jax.lax.dynamic_index_in_dim(c, j, axis=0, keepdims=False)
        # Output rows occupy registers; reduce only over their contracted axis.
        dc_ref[j, :, :] = jnp.sum(G*dj[None, :, :], axis=1)
        dd_ref[j, :, :] = jnp.sum(G*cj[:, None, :], axis=0)
      jax.lax.fori_loop(0, h, head, None)
      return dc_ref[...], dd_ref[...]
    dc, dd = pl.run_scoped(scoped, pltpu.VMEM(a.shape, jnp.float32),
                          pltpu.VMEM(d.shape, jnp.float32))
  else:
    raise ValueError(mode)
  dot = jnp.sum(dc*an, axis=1, keepdims=True)
  da = dc*(g.astype(jnp.float32)[:, None, :]*di)
  da = ai*(da-af*jnp.mean(da*af, axis=1, keepdims=True)*ai*ai)
  dd = dd-df*(g.astype(jnp.float32)[:, None, :]*di**3*dot/v)
  dg = jax.lax.slice_in_dim(di*dot, 0, 1, axis=1).reshape(h,t)
  ds = jnp.sum(dc, axis=2)
  return da.astype(a.dtype).transpose(2, 0, 1), dd.astype(d.dtype).transpose(2, 0, 1), dg.astype(g.dtype).T, ds
