"""TPU microbenchmark: v8t read without / with the in-kernel M shift / with an XLA-mixed M, XL shapes.

usage: [MSHIFT_SHAPES=medium] python MaxText/tests/bam_pallas_mshift_benchmark.py [batch] [tile] [chunk]
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'MaxText')
from layers import bam_pallas_v8 as v8  # pylint: disable=wrong-import-position

import os  # pylint: disable=wrong-import-position,wrong-import-order
B = int(sys.argv[1]) if len(sys.argv) > 1 else 8
TILE = int(sys.argv[2]) if len(sys.argv) > 2 else 128
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 8
# MSHIFT_SHAPES=medium: MediumProp (V32 K75 N16 QK57); default XLProp (V40 K96 N20 QK72).
T, V, K, N, QKC, RS = ((4096, 32, 75, 16, 57, 18) if os.environ.get('MSHIFT_SHAPES') == 'medium'
                       else (4096, 40, 96, 20, 72, 24))
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
kw = dict(qk_cols=QKC, read_epsilon=1e-4, key_scale=0.2, forward_tile=TILE, reverse_tile=TILE, vmem_mib=48, chunk=CHUNK,
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

# ---- kernel-only: pallas_calls on pre-laid-out T-minor operands (no wrapper transposes / boundary add) ----
opts = dict(heads=N, qk_cols=QKC, read_epsilon=1e-4, key_scale=0.2, tile=TILE, reverse_tile=TILE, vmem_mib=48,
            interpret=False, q_scale=float(1 / np.sqrt(K)), chunk=CHUNK)
keys = lambda x: jnp.transpose(x, (0, 3, 2, 1))
gate = lambda x: jnp.moveaxis(x, 1, -1)
st = sw.astype(jnp.float32)
args = (m, st, keys(xs[0]), gate(xs[1]), keys(xs[2]), gate(xs[3]), keys(xs[4]), gate(xs[5]), gate(xs[6]),
        gate(xs[7]), gate(xs[8]), jnp.stack([gate(x) for x in sgs], axis=1))
sh = gate(shift)
kcts = tuple(jnp.transpose(c, (0, 2, 1, 3)) for c in cts[:4]) + (cts[4],)
args, sh, kcts = jax.device_put((args, sh, kcts))
kernel = {}
for name, s in (('plain', None), ('kernel_shift', sh)):
  o = opts
  f = jax.jit(lambda a, s=s, o=o: v8._tm_read_forward_call(a, o, s))
  b_raw = jax.jit(lambda a, c, s=s, o=o: v8._tm_read_backward_call(a, c, o, s))
  kernel[name] = {'fwd_ms': timeit(f, args), 'bwd_ms': timeit(b_raw, args, kcts)}
  print('kernel', name, kernel[name], flush=True)
print('KERNEL_JSON ' + json.dumps(kernel))
