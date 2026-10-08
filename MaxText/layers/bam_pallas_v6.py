"""v6 write kernels: drop-in for the blocked2 write (M v-major [V,K,T], token-minor factors),
with loop-structured, register-carried contractions.

v5p bundle dumps showed the unrolled bodies spilling (4-14k spill stores per tile) while
loop-structured, VALU-dense bodies run at their VALU floor. Layout tricks keep every dynamic
loop index on a leading scratch dimension: factors are stored [heads, rows/8, 8, T] so a row
(head, 8*b+s) is reached with a dynamic block b and a static sublane s.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from layers.bam_pallas import (F32, _freeze, _map_batch, _minor, _norm, _norm_backward, _params,
                               _sigmoid, _spec, _tile)


def _fori(n, body, init, unroll=1):
  return jax.lax.fori_loop(0, n, body, init, unroll=unroll)


def _stage_factors(groups, c_scr, a_scr, gate_s, eps):
  """c_scr[i] = sigmoid(l_i) * N_K(content_i) and a_scr[i] = N_V(address_i), both [rows/8, 8, T]."""
  off = 0
  for g in range(0, len(groups), 3):
    content, logits, address = groups[g:g + 3]
    nh = address.shape[0]
    kdim, t = content.shape[1], content.shape[2]
    vdim = address.shape[1]
    for h in range(nh):
      gate_s[off + h] = _sigmoid(logits[pl.ds(h, 1), :])

    def body(h, carry, content=content, address=address, off=off):
      c = gate_s[off + h] * _norm(content[h], eps, 0)
      c_scr[off + h] = c.reshape(kdim // 8, 8, t)
      a_scr[off + h] = _norm(address[h], eps, 0).reshape(vdim // 8, 8, t)
      return carry
    _fori(nh, body, 0)
    off += nh
  return off


def _write_kernel(eps, n_groups, vb, unroll=1):
  def kernel(m_ref, *refs):
    groups = refs[:3 * n_groups]
    out_ref, c_scr, a_scr, gate_s = refs[3 * n_groups:]
    vdim, kdim, t = m_ref.shape
    ntot = _stage_factors(groups, c_scr, a_scr, gate_s, eps)

    # dM[v] = sum_i A[i, v] C[i]: per block of vb rows v, loop over heads (C slab reused vb times).
    for v0 in range(0, vdim, vb):
      rows = list(range(v0, min(v0 + vb, vdim)))

      def body(b, acc, rows=rows):
        acc = list(acc)
        for u in range(unroll):        # manual unroll: Mosaic fori supports only 1 or full
          i = b * unroll + u
          c = c_scr[i].reshape(kdim, t)
          for j, v in enumerate(rows):
            acc[j] = acc[j] + a_scr[i, v // 8, pl.ds(v % 8, 1), :].reshape(1, t) * c
        return tuple(acc)
      assert ntot % unroll == 0
      acc = _fori(ntot // unroll, body, tuple(jnp.zeros((kdim, t), F32) for _ in rows))
      for j, v in enumerate(rows):
        out_ref[v] = (m_ref[v].astype(F32) + acc[j]).astype(out_ref.dtype)
  return kernel


def _write_backward_kernel(eps, n_groups, hb, ab):
  def kernel(g_ref, *refs):
    groups = refs[:3 * n_groups]
    outs = refs[3 * n_groups:6 * n_groups]
    c_scr, a_scr, gate_s, gf, gk, dc_scr, da_scr = refs[6 * n_groups:]
    vdim, kdim, t = g_ref.shape
    kbk = kdim // 8
    ntot = _stage_factors(groups, c_scr, a_scr, gate_s, eps)

    # FP32 cotangent, v-major blocked gf[v] = [K/8, 8, T], and k-major gk[k] = G[:, k] ([V, T]).
    def gf_body(v, carry):
      gf[v] = g_ref[v].astype(F32).reshape(kbk, 8, t)
      return carry
    _fori(vdim, gf_body, 0)

    def stage_body(b, carry):
      for s in range(8):
        gk[b * 8 + s] = gf[pl.ds(0, vdim), b, pl.ds(s, 1), :].reshape(vdim, t)
      return carry
    _fori(kbk, stage_body, 0)

    # dC[i] = sum_v A[i, v] G[v]  (G[v] contiguous [K, T]; head blocks of hb).
    for i0 in range(0, ntot, hb):
      heads = list(range(i0, min(i0 + hb, ntot)))

      def dc_body(v, acc, heads=heads):
        gv = gf[v].reshape(kdim, t)
        vb, vs = v // 8, v % 8
        return tuple(acc[j] + a_scr[i, vb, pl.ds(vs, 1), :].reshape(1, t) * gv for j, i in enumerate(heads))
      acc = _fori(vdim, dc_body, tuple(jnp.zeros((kdim, t), F32) for _ in heads))
      for j, i in enumerate(heads):
        dc_scr[i] = acc[j]

    # dA[i] = sum_k C[i, k] G[:, k]  (staged k-major slabs; head blocks of ab).
    for i0 in range(0, ntot, ab):
      heads = list(range(i0, min(i0 + ab, ntot)))

      def da_body(b, acc, heads=heads):
        acc = list(acc)
        for s in range(8):
          g = gk[b * 8 + s]
          for j, i in enumerate(heads):
            acc[j] = acc[j] + c_scr[i, b, pl.ds(s, 1), :].reshape(1, t) * g
        return tuple(acc)
      acc = _fori(kbk, da_body, tuple(jnp.zeros((vdim, t), F32) for _ in heads))
      for j, i in enumerate(heads):
        da_scr[i] = acc[j]

    off = 0
    for g in range(n_groups):
      content, logits, address = groups[3 * g:3 * g + 3]
      dcontent, dlogits, daddress = outs[3 * g:3 * g + 3]
      nh = address.shape[0]
      for h in range(nh):
        sg = gate_s[off + h]
        x = content[h]
        dci = dc_scr[off + h]
        dlogits[pl.ds(h, 1), :] = (sg * (1 - sg) * jnp.sum(dci * _norm(x, eps, 0), axis=0, keepdims=True)
                                   ).astype(dlogits.dtype)
        dcontent[h] = _norm_backward(x, sg * dci, eps, 0).astype(dcontent.dtype)
        daddress[h] = _norm_backward(address[h], da_scr[off + h], eps, 0).astype(daddress.dtype)
      off += nh
  return kernel


def _scratch(groups, tile, kdim, vdim):
  ntot = sum(x.shape[1] for x in groups[::3])
  return ntot, [pltpu.VMEM((ntot, kdim // 8, 8, tile), F32), pltpu.VMEM((ntot, vdim // 8, 8, tile), F32),
                pltpu.VMEM((ntot, 1, tile), F32)]


def _write_forward_call(m, groups, opts):
  b, vdim, kdim, t = m.shape
  tile = _tile(t, opts['forward_tile'])
  _, scratch = _scratch(groups, tile, kdim, vdim)
  in_specs = [_spec((vdim, kdim), tile)] + [_spec(x.shape[1:-1], tile) for x in groups]
  return pl.pallas_call(
      _write_kernel(opts['epsilon'], len(groups) // 3, opts['row_block'], opts.get('unroll', 1)),
      grid=(b, t // tile), in_specs=in_specs, out_specs=_spec((vdim, kdim), tile),
      out_shape=jax.ShapeDtypeStruct(m.shape, m.dtype), scratch_shapes=scratch,
      input_output_aliases={0: 0}, interpret=opts['interpret'],
      compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_write')(m, *groups)


def _write_backward_call(g, groups, opts):
  b, vdim, kdim, t = g.shape
  tile = _tile(t, opts['reverse_tile'])
  ntot, scratch = _scratch(groups, tile, kdim, vdim)
  scratch = scratch + [pltpu.VMEM((vdim, kdim // 8, 8, tile), F32), pltpu.VMEM((kdim, vdim, tile), F32),
                       pltpu.VMEM((ntot, kdim, tile), F32),
                       pltpu.VMEM((ntot, vdim, tile), F32)]
  in_specs = [_spec((vdim, kdim), tile)] + [_spec(x.shape[1:-1], tile) for x in groups]
  return pl.pallas_call(
      _write_backward_kernel(opts['epsilon'], len(groups) // 3, opts['head_block'], opts['address_block']),
      grid=(b, t // tile), in_specs=in_specs, out_specs=[_spec(x.shape[1:-1], tile) for x in groups],
      out_shape=tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in groups), scratch_shapes=scratch,
      interpret=opts['interpret'], compiler_params=_params(('parallel', 'parallel'), opts['vmem_mib']),
      name='bam_core_write_backward')(g, *groups)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def _write(m, opts, *groups):
  if dict(opts).get('forward') == 'blocked2':
    return _blocked_forward(m, groups, opts)
  return _write_forward_call(m, groups, dict(opts))


def _blocked_forward(m, groups, opts):
  """Forward uses the register-blocked blocked2 body (measured at its VALU floor on v5p)."""
  from layers import bam_pallas as bp
  o = dict(opts)
  bopts = dict(epsilon=o['epsilon'], forward_tile=o['forward_tile'], reverse_tile=o['reverse_tile'],
               vmem_mib=o['vmem_mib'], interpret=o['interpret'], body='blocked2', row_block=4,
               head_block=4)
  return bp._write_forward_call(m, groups, bopts)


def _write_fwd(m, opts, *groups):
  if dict(opts).get('forward') == 'blocked2':
    return _blocked_forward(m, groups, opts), groups
  return _write_forward_call(m, groups, dict(opts)), groups


def _write_bwd(opts, groups, g):
  return (g,) + tuple(_write_backward_call(g, groups, dict(opts)))


_write.defvjp(_write_fwd, _write_bwd)


def write(m, groups, *, epsilon, forward_tile=128, reverse_tile=128, vmem_mib=None, interpret=False,
          row_block=4, head_block=3, address_block=8, unroll=1, forward='blocked2', **_):
  """m v-major [B,V,K,T]; groups of (content [B,T,N,K], logits [B,T,N], address [B,T,N,V])."""
  opts = _freeze(dict(epsilon=epsilon, forward_tile=forward_tile, reverse_tile=reverse_tile,
                      vmem_mib=vmem_mib, interpret=interpret, row_block=row_block,
                      head_block=head_block, address_block=address_block, unroll=unroll,
                      forward=forward))
  flat = tuple(x for grp in groups for x in grp)

  def local(m, *xs):
    return _write(m, opts, *(_minor(x) for x in xs))

  return _map_batch(local, (m,) + flat, (True,) * (1 + len(flat)), 1)
