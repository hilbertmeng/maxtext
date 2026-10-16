"""TPU microbenchmark: v8t read without / with the in-kernel M shift / with an XLA-mixed M, XL shapes.

usage: python MaxText/tests/bam_pallas_mshift_benchmark.py [batch] [tile]
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'MaxText')
from layers import bam_pallas_v8 as v8  # pylint: disable=wrong-import-position

B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TILE = int(sys.argv[2]) if len(sys.argv) > 2 else 128
T, V, K, N, QKC, RS = 4096, 40, 96, 20, 72, 24
dt = jnp.bfloat16


def timeit(fn, *args, n=20):
  jax.block_until_ready(fn(*args))
  t0 = time.perf_counter()
  for _ in range(n):
    out = fn(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) / n * 1e3


def rel(a, b):
  a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
  return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


ks = iter(jax.random.split(jax.random.PRNGKey(0), 32))
r = lambda shape, s=1.0: (s * jax.random.normal(next(ks), shape)).astype(dt)
m = r((B, T, V, K))
sw = r((4 * N, V), .3)
xs = (r((B, T, N, V)), r((B, T, N)), r((B, T, N, V)), r((B, T, N)), r((B, T, N, V)),
      r((B, T, N)), r((B, T, N)), r((B, T, N, RS)), r((B, T, N, RS)))
sgs = tuple(jax.nn.sigmoid(r((B, T, N))) for _ in range(4))
shift = (jax.nn.sigmoid(r((B, T, V))) * (jnp.arange(T) > 0)[None, :, None]).astype(dt)
cts = tuple(r((B, T, N, K)) for _ in range(4)) + (r((B, T, V, K)),)
kw = dict(qk_cols=QKC, read_epsilon=1e-4, key_scale=0.2, forward_tile=TILE, reverse_tile=TILE, vmem_mib=48,
          q_scale=1 / np.sqrt(K), static_gates=sgs)


def mixed(m, shift):
  prev = jnp.concatenate((m[:, :1], m[:, :-1]), axis=1)
  g = shift[..., None]
  return (1 - g) * m + g * prev


arms = {
    'plain': lambda m, shift, xs: v8.read_tm(m, sw, *xs, **kw),
    'kernel_shift': lambda m, shift, xs: v8.read_tm(m, sw, *xs, shift=shift, **kw),
    'xla_mix': lambda m, shift, xs: v8.read_tm(mixed(m, shift), sw, *xs, **kw)[:4] + (m,),
}
res = {'batch': B, 'tile': TILE}
outs = {}
for name, fn in arms.items():
  fwd = jax.jit(fn)
  outs[name] = fwd(m, shift, xs)

  def step(m, shift, xs, fn=fn):
    _, pull = jax.vjp(fn, m, shift, xs)
    return pull(cts)

  bwd = jax.jit(step)
  outs[name + '_grad'] = bwd(m, shift, xs)
  res[name] = {'fwd_ms': timeit(fwd, m, shift, xs), 'fwdbwd_ms': timeit(bwd, m, shift, xs)}
  print(name, res[name], flush=True)

res['fwd_rel_err'] = [rel(a, b) for a, b in zip(outs['kernel_shift'][:4], outs['xla_mix'][:4])]
gk, gx = outs['kernel_shift_grad'], outs['xla_mix_grad']
res['grad_rel_err'] = {'m': rel(gk[0], gx[0]), 'shift': rel(gk[1], gx[1]),
                       'keys': [rel(a, b) for a, b in zip(jax.tree.leaves(gk[2]), jax.tree.leaves(gx[2]))]}
print('BENCH_JSON ' + json.dumps(res))
