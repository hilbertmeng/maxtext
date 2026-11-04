"""TPU microbenchmark: the read reverse's per-token matmuls (dX_t = M_t dY_t^T, dM_t = X_t^T dY_t) alone.

usage: python MaxText/tests/bam_v6e_tokmm_bench.py
Formal 2.8B shapes per call: B8 T2048, M_t [C=64, K=80], X_t^T [C, R=128], dY_t [R, K]; 128-token tiles,
64-token chunks. Variants change only how each chunk's batched matmuls are expressed:
  v0 current (batch over tokens, contract K / R)         v1 transposed outputs (dY streams)
  v2 two-token block-diagonal pairs (256-wide contraction) v3 dM only (cost split)
Each kernel repeats its chunk matmuls REP times on VMEM-resident data and writes only per-tile sums,
so (ms at REP 8 - ms at REP 1) / 7 is the compute per pass, free of the HBM I/O.
"""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

B, T, C, K, R = 8, 2048, 64, 80, 128
TILE, CH = 128, 64
F32, BF = jnp.float32, jnp.bfloat16


def bdot(x, y, cx, cy):
  return jax.lax.dot_general(x, y, (((cx,), (cy,)), ((0,), (0,))), preferred_element_type=F32)


def kernel(variant, rep):
  def k(m_ref, xt_ref, dy_ref, dxt_ref, dm_ref):
    acc_x = jnp.zeros((C, R), F32)
    acc_m = jnp.zeros((C, K), F32)
    for _ in range(rep):          # repeat on VMEM-resident data: the REP slope is pure compute
      for t0 in range(0, TILE, CH):
        m, xt, dy = m_ref[pl.ds(t0, CH)], xt_ref[pl.ds(t0, CH)], dy_ref[pl.ds(t0, CH)]
        if variant == 'v0':
          dxt = bdot(m, dy, 2, 2)                                     # [ch, C, R]
          dm = bdot(xt, dy, 2, 1)                                     # [ch, C, K]
        elif variant == 'v1':
          dxt = jnp.swapaxes(bdot(dy, m, 2, 2), 1, 2)
          dm = jnp.swapaxes(bdot(dy, xt, 1, 2), 1, 2)
        elif variant == 'v2':
          h = CH // 2
          z = jnp.zeros((h, R, K), dy.dtype)
          dyp = dy.reshape(h, 2, R, K)
          rhs = jnp.concatenate((jnp.concatenate((dyp[:, 0], z), axis=2), jnp.concatenate((z, dyp[:, 1]), axis=2)), axis=1)
          xtp = xt.reshape(h, 2, C, R)
          lhs = jnp.concatenate((xtp[:, 0], xtp[:, 1]), axis=2)
          dmp = bdot(lhs, rhs, 2, 1)
          dm = jnp.stack((dmp[:, :, :K], dmp[:, :, K:]), axis=1).reshape(CH, C, K)
          dxt = bdot(m, dy, 2, 2)
        elif variant == 'v3':     # dM only
          dm = bdot(xt, dy, 2, 1)
          dxt = jnp.zeros((CH, C, R), F32)
        else:                     # v4: dX only
          dxt = bdot(m, dy, 2, 2)
          dm = jnp.zeros((CH, C, K), F32)
        acc_x = acc_x + jnp.sum(dxt, axis=0)
        acc_m = acc_m + jnp.sum(dm, axis=0)
    dxt_ref[...] = acc_x
    dm_ref[...] = acc_m
  return k


def call(variant, rep, m, xt, dy):
  spec = lambda *s: pl.BlockSpec((None, TILE) + s, lambda b, i: (b, i) + (0,) * len(s))
  small = lambda *s: pl.BlockSpec((None, None) + s, lambda b, i: (b, i) + (0,) * len(s))
  return pl.pallas_call(
      kernel(variant, rep), grid=(B, T // TILE),
      in_specs=[spec(C, K), spec(C, R), spec(R, K)], out_specs=[small(C, R), small(C, K)],
      out_shape=[jax.ShapeDtypeStruct((B, T // TILE, C, R), F32), jax.ShapeDtypeStruct((B, T // TILE, C, K), F32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'parallel'),
                                           vmem_limit_bytes=110 * 2**20))(m, xt, dy)


def timeit(fn, *a, n=20, reps=5):
  jax.block_until_ready(fn(*a))
  ts = []
  for _ in range(reps):
    t0 = time.perf_counter()
    for _ in range(n):
      out = fn(*a)
    jax.block_until_ready(out)
    ts.append((time.perf_counter() - t0) / n * 1e3)
  return float(np.median(ts))


def main():
  ks = jax.random.split(jax.random.PRNGKey(0), 3)
  m = jax.random.normal(ks[0], (B, T, C, K)).astype(BF)
  xt = jax.random.normal(ks[1], (B, T, C, R)).astype(BF)
  dy = jax.random.normal(ks[2], (B, T, R, K)).astype(BF)
  ref_x = jnp.einsum('btck,btrk->bcr', m.astype(F32), dy.astype(F32))
  ref_m = jnp.einsum('btcr,btrk->bck', xt.astype(F32), dy.astype(F32))
  rel = lambda a, b: float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))
  print(jax.devices()[0].device_kind, flush=True)
  for v in ('v0', 'v1', 'v2', 'v3', 'v4', 'v0'):
    res = dict(variant=v)
    try:
      for rep in (1, 8):
        f = jax.jit(lambda m, xt, dy, v=v, rep=rep: call(v, rep, m, xt, dy))
        dxt, dm = f(m, xt, dy)
        res[f'ms_rep{rep}'] = round(timeit(f, m, xt, dy), 3)
        if rep == 1:
          res['err_dm'] = rel(jnp.sum(dm, axis=1), ref_m) if v != 'v4' else None
          res['err_dx'] = rel(jnp.sum(dxt, axis=1), ref_x) if v != 'v3' else None
      res['compute_ms_per_rep'] = round((res['ms_rep8'] - res['ms_rep1']) / 7, 3)
    except Exception as e:  # pylint: disable=broad-except
      res['error'] = repr(e)[:400]
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
  main()
