"""TPU timing of the v8t read kernel at XL V5 per-chip shapes across M widths V.

usage: python MaxText/tests/bam_read_v_sweep_bench.py [V ...]   (env READ_OPTS='{"reverse_tile": 256}')
Prints forward and forward+reverse ms per call (median of repeats) per V.
"""
import json
import math
import os
import sys
import time

import jax
import jax.numpy as jnp

sys.path.insert(0, 'MaxText')
from layers import bam_pallas_v8 as v8  # pylint: disable=wrong-import-position

B, T, N, K, RS = 8, 4096, 20, 96, 24
DT = jnp.bfloat16


def inputs(v, seed):
  keys = iter(jax.random.split(jax.random.key(seed), 32))
  rand = lambda shape, s=.5: (s * jax.random.normal(next(keys), shape)).astype(DT)
  m = rand((B, T, v, K))
  static = rand((4 * N, v), .3)
  rq, rk, rvo = rand((B, T, N, v)), rand((B, T, N, v)), rand((B, T, 2 * N, v))
  logits = tuple(rand((B, T, N)) for _ in range(4))
  sq, sk = rand((B, T, N, RS)), rand((B, T, N, RS))
  gates = tuple(jax.nn.sigmoid(rand((B, T, N))).astype(DT) for _ in range(4))
  shift = jax.nn.sigmoid(rand((B, T, v))).astype(DT)
  cnt = jnp.transpose(jnp.concatenate((rq, rk, rvo), axis=2), (0, 3, 2, 1))   # [B,C,4N,T]
  return (m, cnt), dict(static=static, rq=rq, rk=rk, rvo=rvo, logits=logits, sq=sq, sk=sk,
                        gates=gates, shift=shift)


def make_read(aux, opts):
  lq, lk, lo, lv = aux['logits']
  common = dict(qk_cols=K, read_epsilon=1e-4, key_scale=.1, forward_tile=128, reverse_tile=128,
                vmem_mib=63, q_scale=1 / math.sqrt(K), static_gates=aux['gates'], shift=aux['shift'],
                chunk=8, m_norm=1e-6, q_dynamic_scale=jnp.full((N,), .8), k_dynamic_scale=jnp.full((N,), .8),
                v_dynamic_scale=jnp.full((N,), .8), o_dynamic_scale=jnp.full((N,), .8))
  common.update(opts)

  def read(m, cnt):
    return v8.read_tm(m, aux['static'], aux['rq'], lq, aux['rk'], lk, aux['rvo'], lo, lv,
                      aux['sq'], aux['sk'], keys_cnt=cnt, **common)
  return read


def timeit(fn, args_list, reps=5):
  jax.block_until_ready(fn(*args_list[0]))
  times = []
  for _ in range(reps):
    t0 = time.perf_counter()
    for args in args_list:
      out = fn(*args)
    jax.block_until_ready(out)
    times.append((time.perf_counter() - t0) / len(args_list) * 1e3)
  return sorted(times)[len(times) // 2]


def main():
  vs = [int(x) for x in sys.argv[1:]] or [40, 48, 56, 60, 64]
  opts = json.loads(os.environ.get('READ_OPTS', '{}'))
  for v in vs:
    sets = [inputs(v, 10 * v + i) for i in range(3)]
    read = make_read(sets[0][1], opts)
    fwd = jax.jit(lambda m, cnt: sum(jnp.sum(o.astype(jnp.float32)) for o in read(m, cnt)))
    both = jax.jit(jax.grad(lambda m, cnt: sum(jnp.sum(o.astype(jnp.float32) * 1.001) for o in read(m, cnt)),
                            argnums=(0, 1)))
    args = [s[0] for s in sets]
    tf, tb = timeit(fwd, args), timeit(both, args)
    print(f'V={v:3d} fwd={tf:7.3f} ms  fwd+rev={tb:7.3f} ms  rev~={tb - tf:7.3f} ms  opts={opts}', flush=True)


if __name__ == '__main__':
  main()
