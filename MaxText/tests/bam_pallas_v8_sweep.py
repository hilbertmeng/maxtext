"""TPU sweep: v8t read (with in-kernel M shift) and write kernels over tile x chunk at one model shape.

usage: [V8_VMEM=48,56 V8_TILES=128 V8_CHUNKS=8 V8_KERNELS=read] V8_SHAPE=B,T,V,K,N,QKC,RS python MaxText/tests/bam_pallas_v8_sweep.py   (default Pythia-2.8B v64: 8,2048,64,80,32,60,20)
Prints one JSON line per (kernel, tile, chunk): forward and forward+backward ms, or the error.
"""
import itertools
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'MaxText')
from layers import bam_pallas_v8 as v8  # pylint: disable=wrong-import-position

B, T, V, K, N, QKC, RS = (int(x) for x in os.environ.get('V8_SHAPE', '8,2048,64,80,32,60,20').split(','))
VMEMS = [int(x) for x in os.environ.get('V8_VMEM', '48').split(',')]
TILES = [int(x) for x in os.environ.get('V8_TILES', '64,128,256').split(',')]
CHUNKS = [int(x) for x in os.environ.get('V8_CHUNKS', '4,8,16').split(',')]
KERNELS = os.environ.get('V8_KERNELS', 'read,write').split(',')
dt = jnp.bfloat16


def timeit(fn, *args, n=20):
  jax.block_until_ready(fn(*args))
  t0 = time.perf_counter()
  for _ in range(n):
    out = fn(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) / n * 1e3


ks = iter(jax.random.split(jax.random.PRNGKey(0), 64))
r = lambda shape, s=1.0: (s * jax.random.normal(next(ks), shape)).astype(dt)
m = r((B, T, V, K))
sw = r((4 * N, V), .3)
xs = (r((B, T, N, V)), r((B, T, N)), r((B, T, N, V)), r((B, T, N)), r((B, T, N, V)),
      r((B, T, N)), r((B, T, N)), r((B, T, N, RS)), r((B, T, N, RS)))
sgs = tuple(jax.nn.sigmoid(r((B, T, N))) for _ in range(4))
shift = (jax.nn.sigmoid(r((B, T, V))) * (jnp.arange(T) > 0)[None, :, None]).astype(dt)
read_cts = tuple(r((B, T, N, K)) for _ in range(4)) + (r((B, T, V, K)),)
groups = [(r((B, T, N, K)), r((B, T, N)), r((B, T, N, V))) for _ in range(2)]
write_ct = r((B, T, V, K))
print(json.dumps({'shape': dict(B=B, T=T, V=V, K=K, N=N, QKC=QKC, RS=RS)}), flush=True)

for VMEM, tile, chunk in itertools.product(VMEMS, TILES, CHUNKS):
  kw = dict(qk_cols=QKC, read_epsilon=1e-4, key_scale=0.2, forward_tile=tile, reverse_tile=tile, vmem_mib=VMEM,
            chunk=chunk, q_scale=1 / np.sqrt(K), static_gates=sgs)
  rd = lambda m, xs, kw=kw: v8.read_tm(m, sw, *xs, shift=shift, **kw)
  try:
    assert 'read' in KERNELS, 'skipped'
    f = jax.jit(rd)
    b = jax.jit(lambda m, xs, rd=rd: jax.vjp(rd, m, xs)[1](read_cts))
    print(json.dumps({'kernel': 'read', 'vmem': VMEM, 'tile': tile, 'chunk': chunk, 'fwd_ms': round(timeit(f, m, xs), 3),
                      'fwdbwd_ms': round(timeit(b, m, xs), 3)}), flush=True)
  except Exception as e:  # pylint: disable=broad-except
    print(json.dumps({'kernel': 'read', 'vmem': VMEM, 'tile': tile, 'chunk': chunk, 'error': str(e)[:160]}), flush=True)
  for ng in ((1, 2) if 'write' in KERNELS else ()):
    wk = dict(epsilon=1e-6, forward_tile=tile, reverse_tile=tile, vmem_mib=VMEM, chunk=chunk)
    wr = lambda m, gs, wk=wk: v8.write_tm(m, gs, **wk)
    gs = groups[:ng]
    try:
      f = jax.jit(wr)
      b = jax.jit(lambda m, gs, wr=wr: jax.vjp(wr, m, gs)[1](write_ct))
      print(json.dumps({'kernel': f'write{ng}', 'vmem': VMEM, 'tile': tile, 'chunk': chunk, 'fwd_ms': round(timeit(f, m, gs), 3),
                        'fwdbwd_ms': round(timeit(b, m, gs), 3)}), flush=True)
    except Exception as e:  # pylint: disable=broad-except
      print(json.dumps({'kernel': f'write{ng}', 'vmem': VMEM, 'tile': tile, 'chunk': chunk, 'error': str(e)[:160]}), flush=True)
print('V8_SWEEP_DONE', flush=True)
