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
  def scoped(out, c_ref, m_ref):
    c_ref[...] = c.transpose(1,0,2)
    m_ref[...] = m
    def step(i, _):
      ci = c_ref[pl.ds(i*rows,rows), :, :]
      mi = m_ref[pl.ds(i*rows,rows), :, :]
      acc = jnp.zeros((rows, v, t), jnp.float32)
      for j in range(h):
        cj = jax.lax.slice_in_dim(ci, j, j+1, axis=1).reshape(rows,t)
        dj = jax.lax.slice_in_dim(df, j, j+1, axis=0).reshape(v,t)
        acc = acc+cj[:,None,:]*dj[None,:,:]
      out[pl.ds(i*rows, rows), :, :] = (mi.astype(jnp.float32)+acc).astype(m.dtype)
    jax.lax.fori_loop(0, k//rows, step, None)
    return out[...]
  return pl.run_scoped(scoped, pltpu.VMEM(m.shape, m.dtype),
                       pltpu.VMEM((k,h,t),jnp.float32), pltpu.VMEM(m.shape,m.dtype))


def reverse_major(a, d, g, s, matrix_grad, epsilon):
  """Keep MXU inputs and normalization native token-major; no FP32 layout roundtrip."""
  t, h, k = a.shape
  v = d.shape[-1]
  af, df, gf = a.astype(jnp.float32), d.astype(jnp.float32), g.astype(jnp.float32)
  ai = jax.lax.rsqrt(jnp.mean(af*af, axis=-1, keepdims=True)+epsilon)
  di = jax.lax.rsqrt(jnp.mean(df*df, axis=-1, keepdims=True)+epsilon)
  an = af*ai
  c = s.astype(jnp.float32)[None,:,:] + an*(gf[:,:,None]*di)
  G = matrix_grad
  square = jnp.concatenate((
      jnp.concatenate((jnp.zeros((t,k,k),G.dtype),G),axis=2),
      jnp.concatenate((G.swapaxes(1,2),jnp.zeros((t,v,v),G.dtype)),axis=2)),axis=1)
  left = jnp.concatenate((c.astype(d.dtype),d),axis=2)
  product = jnp.einsum('thd,tdc->thc',left,square,preferred_element_type=jnp.float32)
  dc, dd = product[:,:,:k], product[:,:,k:]
  dot = jnp.sum(dc*an,axis=-1,keepdims=True)
  da = dc*(gf[:,:,None]*di)
  da = ai*(da-af*jnp.mean(da*af,axis=-1,keepdims=True)*ai*ai)
  dd = dd-df*(gf[:,:,None]*di**3*dot/v)
  dg = (di*dot).reshape(t,h)
  ds = jnp.sum(dc,axis=0)
  return da.astype(a.dtype),dd.astype(d.dtype),dg.astype(g.dtype),ds


def reverse_vloop_product(c, df, G):
  """Two output-stationary contractions. SIMD over tokens, 8 output rows at once.

  Only H*8*T accumulators are live; never form an H*K*V*T product. The contracted
  axis is the leading Ref axis, so scalar dynamic indexing avoids TPU sublane
  slice restrictions. Padding is confined to the 75-value axis (80, not 128).
  """
  h, k, t = c.shape
  v = df.shape[1]
  vp = ((v+7)//8)*8
  if k % 8:
    raise ValueError('VPU output-stationary write requires key width divisible by 8')
  def scoped(dc_ref, dd_ref, d_ref, c_ref, gk_ref, gv_ref):
    d_ref[...] = jnp.pad(df,((0,0),(0,vp-v),(0,0))).transpose(1,0,2)
    c_ref[...] = c.transpose(1,0,2)
    padded = jnp.pad(G,((0,0),(0,vp-v),(0,0)))
    gk_ref[...] = padded
    gv_ref[...] = padded.transpose(1,0,2)
    def address_block(i, _):
      sl = pl.ds(i*8,8)
      def mac(j,acc):
        return acc+d_ref[j,:,:][:,None,:]*gv_ref[j,sl,:][None,:,:]
      dc_ref[:,sl,:] = jax.lax.fori_loop(0,v,mac,jnp.zeros((h,8,t),jnp.float32))
    def data_block(i, _):
      sl = pl.ds(i*8,8)
      def mac(j,acc):
        return acc+c_ref[j,:,:][:,None,:]*gk_ref[j,sl,:][None,:,:]
      dd_ref[:,sl,:] = jax.lax.fori_loop(0,k,mac,jnp.zeros((h,8,t),jnp.float32))
    jax.lax.fori_loop(0,k//8,address_block,None)
    jax.lax.fori_loop(0,vp//8,data_block,None)
    return dc_ref[...],dd_ref[...][:,:v,:]
  return pl.run_scoped(scoped,*[pltpu.VMEM(shape,jnp.float32) for shape in
      ((h,k,t),(h,vp,t),(vp,h,t),(k,h,t),(k,vp,t),(vp,k,t))])


def reverse(a, d, g, s, matrix_grad, epsilon, mode, **unused):
  if mode.split("_")[1] == "major":
    return reverse_major(a,d,g,s,matrix_grad,epsilon)
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
  elif backend == 'vloop':
    dc,dd = reverse_vloop_product(c,df,matrix_grad.transpose(1,2,0).astype(jnp.float32))
  elif backend in ('vpu','vrow'):
    G = matrix_grad.transpose(1, 2, 0).astype(jnp.float32)
    def scoped(dc_ref, dd_ref, d_ref, c_ref, g_ref):
      d_ref[...] = df; c_ref[...] = c; g_ref[...] = G
      def head(j, _):
        dj = d_ref[j,:,:]
        cj = c_ref[j,:,:]
        if backend == 'vpu':
          gg = g_ref[...]
          dc_ref[j, :, :] = jnp.sum(gg*dj[None, :, :], axis=1)
          dd_ref[j, :, :] = jnp.sum(gg*cj[:, None, :], axis=0)
        else:
          # Bound the live product instead of materializing K*V*T for each head.
          def address_rows(i, _):
            sl = pl.ds(i*4,4)
            dc_ref[j,sl,:] = jnp.sum(g_ref[sl,:,:]*dj[None,:,:],axis=1)
          def data_rows(i, _):
            sl = pl.ds(i*3,3)
            dd_ref[j,sl,:] = jnp.sum(g_ref[:,sl,:]*cj[:,None,:],axis=0)
          jax.lax.fori_loop(0,k//4,address_rows,None)
          jax.lax.fori_loop(0,v//3,data_rows,None)
      jax.lax.fori_loop(0, h, head, None)
      return dc_ref[...], dd_ref[...]
    dc, dd = pl.run_scoped(scoped, pltpu.VMEM(a.shape, jnp.float32),
                          pltpu.VMEM(d.shape, jnp.float32),pltpu.VMEM(d.shape,jnp.float32),
                          pltpu.VMEM(c.shape,jnp.float32),pltpu.VMEM(G.shape,jnp.float32))
  else:
    raise ValueError(mode)
  dot = jnp.sum(dc*an, axis=1, keepdims=True)
  da = dc*(g.astype(jnp.float32)[:, None, :]*di)
  da = ai*(da-af*jnp.mean(da*af, axis=1, keepdims=True)*ai*ai)
  dd = dd-df*(g.astype(jnp.float32)[:, None, :]*di**3*dot/v)
  dg = jax.lax.slice_in_dim(di*dot, 0, 1, axis=1).reshape(h,t)
  ds = jnp.sum(dc, axis=2)
  return da.astype(a.dtype).transpose(2, 0, 1), dd.astype(d.dtype).transpose(2, 0, 1), dg.astype(g.dtype).T, ds
