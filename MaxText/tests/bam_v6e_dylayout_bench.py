"""TPU microbenchmark: read-reverse dY scratch layout, head-major + gather (current) vs token-major strided.

usage: python MaxText/tests/bam_v6e_dylayout_bench.py
Per 128-token tile, R=128 head rows dY_r [T, K=80] are written into VMEM scratch and read back as 64-token
[ch, R, K] chunks feeding dM_t = X_t^T dY_t. REP repeats the store + load pass on VMEM-resident data;
(ms at REP 8 - ms at REP 1) / 7 is the per-pass cost.
  g0: rows r*T + t (contiguous per-head stores), chunk = _gather (aligned loads + sublane/major swap)
  g1: rows t*R + r (strided per-head stores), chunk = contiguous load + reshape
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, 'MaxText')
from layers.bam_pallas_v8 import _gather  # pylint: disable=wrong-import-position

B, T, C, K, R = 8, 2048, 64, 80, 128
TILE, CH = 128, 64
F32, BF = jnp.float32, jnp.bfloat16


def kernel(variant, rep):
  def k(dyh_ref, xt_ref, out_ref, scr):
    acc = jnp.zeros((C, K), F32)
    for r in range(rep):
      for h in range(R):
        y = dyh_ref[(h + r) % R].astype(F32)                       # [T, K], distinct per repeat
        if variant == 'g0':
          scr[pl.ds(h * TILE, TILE), :] = y
        else:
          scr[pl.ds(h, TILE, stride=R), :] = y
      for t0 in range(0, TILE, CH):
        if variant == 'g0':
          dy = _gather(scr, t0, CH, TILE)
        else:
          dy = scr[pl.ds(t0 * R, CH * R), :].reshape(CH, R, K)
        dm = jax.lax.dot_general(xt_ref[pl.ds(t0, CH)], dy.astype(BF), (((2,), (1,)), ((0,), (0,))),
                                 preferred_element_type=F32)
        acc = acc + jnp.sum(dm, axis=0)
    out_ref[...] = acc
  return k


def call(variant, rep, dyh, xt):
  return pl.pallas_call(
      kernel(variant, rep), grid=(B, T // TILE),
      in_specs=[pl.BlockSpec((None, R, TILE, K), lambda b, i: (b, 0, i, 0)),
                pl.BlockSpec((None, TILE, C, R), lambda b, i: (b, i, 0, 0))],
      out_specs=pl.BlockSpec((None, None, C, K), lambda b, i: (b, i, 0, 0)),
      out_shape=jax.ShapeDtypeStruct((B, T // TILE, C, K), F32),
      scratch_shapes=[pltpu.VMEM((R * TILE, K), F32)],
      compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel', 'parallel'),
                                           vmem_limit_bytes=110 * 2**20))(dyh, xt)


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
  ks = jax.random.split(jax.random.PRNGKey(0), 2)
  dyh = jax.random.normal(ks[0], (B, R, T, K)).astype(BF)
  xt = jax.random.normal(ks[1], (B, T, C, R)).astype(BF)
  ref = jnp.einsum('btcr,brtk->bck', xt.astype(F32), dyh.astype(F32))
  print(jax.devices()[0].device_kind, flush=True)
  for v in ('g0', 'g1', 'g0', 'g1'):
    res = dict(variant=v)
    try:
      for rep in (1, 8):
        f = jax.jit(lambda dyh, xt, v=v, rep=rep: call(v, rep, dyh, xt))
        out = f(dyh, xt)
        res[f'ms_rep{rep}'] = round(timeit(f, dyh, xt), 3)
        if rep == 1:
          got = jnp.sum(out, axis=1)
          res['err'] = float(jnp.max(jnp.abs(got - ref)) / jnp.max(jnp.abs(ref)))
      res['per_pass_ms'] = round((res['ms_rep8'] - res['ms_rep1']) / 7, 3)
    except Exception as e:  # pylint: disable=broad-except
      res['error'] = repr(e)[:400]
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
  main()
