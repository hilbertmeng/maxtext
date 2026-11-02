"""Time the v8t read kernel at Pythia H shapes, with and without the fused full-head QKNorm.

Single device, per-device batch 8: M [8,2048,64,80], keys [8,2048,32,64], packed V/O keys,
standard Q/K [8,2048,32,20], static gates and V/O dynamic scales. Prints median ms of the forward
and of forward+backward for each variant.

usage (TPU worker, repo root): python3 MaxText/tests/bench_bam_read_qk_norm.py [variant ...]
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp

from layers import bam_pallas

B, T, N, C, K, RS, QK = 8, 2048, 32, 64, 80, 20, 80
QDIM = QK + RS


def inputs(key):
  ks = iter(jax.random.split(key, 32))
  bf = jnp.bfloat16
  m = (jax.random.normal(next(ks), (B, T, C, K), jnp.float32) * .1).astype(bf)
  sw = jax.random.normal(next(ks), (4 * N, C), jnp.float32) * .1
  keys = [jax.random.normal(next(ks), (B, T, n, C), jnp.float32).astype(bf) for n in (N, N, 2 * N)]
  logits = [jax.random.normal(next(ks), (B, T, N), jnp.float32).astype(bf) for _ in range(4)]
  std = [jax.random.normal(next(ks), (B, T, N, RS), jnp.float32).astype(bf) for _ in range(2)]
  gates = tuple(jax.nn.sigmoid(jax.random.normal(next(ks), (B, T, N), jnp.float32)).astype(bf) for _ in range(4))
  scales = [1 + .1 * jax.random.normal(next(ks), (N, C), jnp.float32) for _ in range(2)]
  mq = 1 + .1 * jax.random.normal(next(ks), (QDIM,), jnp.float32)
  mk = 1 + .1 * jax.random.normal(next(ks), (QDIM,), jnp.float32)
  return m, sw, keys, logits, std, gates, scales, mq, mk


def make(variant, data):
  m, sw, (qk, kk, vok), (ql, kl, ol, vl), (qs, ks_), gates, (vsc, osc), mq, mk = data
  common = dict(qk_cols=QK, read_epsilon=1e-6, key_scale=1.0, body='v8t', pass_m=True,
                static_gates=gates, v_dynamic_scale=vsc, o_dynamic_scale=osc, vmem_mib=63)
  # BENCH_READ_OPTS="chunk=16,tile=256,rtile=128": read kernel tiling sweep.
  for kv in filter(None, os.environ.get('BENCH_READ_OPTS', '').split(',')):
    key, val = kv.split('=')
    common[{'chunk': 'chunk', 'tile': 'forward_tile', 'rtile': 'reverse_tile'}[key]] = int(val)

  def f(m, qs, ks_, mq, mk):
    if variant == 'plain':      # the read alone (logit scale folded), no QKNorm anywhere
      return bam_pallas.read(m, sw, qk, ql, kk, kl, vok, ol, vl, qs, ks_, q_scale=.1, **common)
    if variant == 'xla_norm':
      outs = bam_pallas.read(m, sw, qk, ql, kk, kl, vok, ol, vl, qs, ks_, q_scale=1.0, **common)
      q, k = outs[0], outs[1]
      ssq, ssk = (jnp.sum(jnp.square(x.astype(jnp.float32)), -1) for x in (qs, ks_))
      def finish(x, scale, ss):
        local_ss = jnp.sum(jnp.square(x[..., :QK].astype(jnp.float32)), -1)
        inv = jax.lax.rsqrt((local_ss + ss) / QDIM + 1e-6).astype(x.dtype)
        return x * inv[..., None] * scale.astype(x.dtype)
      q, k = finish(q, mq, ssq) / 10., finish(k, mk, ssk)
      return (q, k) + tuple(outs[2:])
    ssq, ssk = (jnp.sum(jnp.square(x.astype(jnp.float32)), -1) for x in (qs, ks_))
    return bam_pallas.read(m, sw, qk, ql, kk, kl, vok, ol, vl, qs, ks_, q_scale=1.0,
                           qk_norm=(ssq, ssk, mq / 10., mk, 1e-6), qk_norm_mxu=variant.startswith('fused_mxu'),
                           qk_out_pad=variant.endswith('_pad'), **common)

  fwd = jax.jit(f)

  def loss(*args):
    outs = f(*args)
    return sum(jnp.sum(o.astype(jnp.float32) * .01) for o in outs[:4])

  bwd = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))
  args = (m, qs, ks_, mq, mk)
  return fwd, bwd, args


def timeit(fn, args, n=20):
  jax.block_until_ready(fn(*args))
  ts = []
  for _ in range(n):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    ts.append(time.perf_counter() - t0)
  return sorted(ts)[n // 2] * 1e3


if __name__ == '__main__':
  data = inputs(jax.random.key(0))
  for variant in sys.argv[1:] or ['plain', 'xla_norm', 'fused', 'fused_mxu']:
    # fused:no_xhat,no_concat -> fused kernel with benchmark-only ablations
    variant, _, ablate = variant.partition(':')
    os.environ['BAM_QKN_ABLATE'] = ablate
    fwd, bwd, args = make(variant, data)
    variant = variant + (':' + ablate if ablate else '')
    print(f'BENCH {variant:10s} fwd {timeit(fwd, args):7.3f} ms  fwd+bwd {timeit(bwd, args):7.3f} ms', flush=True)
