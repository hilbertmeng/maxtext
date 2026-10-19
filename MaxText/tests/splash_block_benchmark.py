"""TPU microbenchmark: Splash block sizes for one device's causal + segment-id attention.

Mirrors the per-device body of Attention.tpu_flash_attention / _bam_splash_attention (vmap over batch,
HEAD_DIM_MINOR layouts). Forward candidates are timed alone; backward candidates are timed as
forward+backward with the default forward blocks. Every candidate is checked against the default.

usage: SPLASH_SHAPE=B,N,T,D python MaxText/tests/splash_block_benchmark.py   (default 16,32,2048,80)
"""
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as sk
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as sm

B, N, T, D = (int(x) for x in os.environ.get('SPLASH_SHAPE', '16,32,2048,80').split(','))
DT = jnp.bfloat16


def kernel(bq, bkv, bkvc, bq_dkv=512, bkv_dkv=512, bkv_dkv_c=512, fused=False, bq_dq=512, bkv_dq=512):
  sizes = sk.BlockSizes(
      block_q=min(bq, T), block_kv=min(bkv, T), block_kv_compute=min(bkvc, T),
      block_q_dkv=min(bq_dkv, T), block_kv_dkv=min(bkv_dkv, T), block_kv_dkv_compute=min(bkv_dkv_c, T),
      block_q_dq=None if fused else min(bq_dq, T), block_kv_dq=None if fused else min(bkv_dq, T),
      use_fused_bwd_kernel=fused)
  mask = sm.MultiHeadMask(masks=(sm.CausalMask(shape=(T, T)),) * N)
  k = sk.make_splash_mha(mask=mask, head_shards=1, q_seq_shards=1, block_sizes=sizes)
  return lambda q, kk, v, seg: jax.vmap(k)(q, kk, v, segment_ids=seg)


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


ks = jax.random.split(jax.random.PRNGKey(0), 4)
q = (jax.random.normal(ks[0], (B, N, T, D), jnp.float32) / np.sqrt(D)).astype(DT)
k = jax.random.normal(ks[1], (B, N, T, D), jnp.float32).astype(DT)
v = jax.random.normal(ks[2], (B, N, T, D), jnp.float32).astype(DT)
ct = jax.random.normal(ks[3], (B, N, T, D), jnp.float32).astype(DT)
ids = jnp.where(jnp.arange(T) < 3 * T // 4, 1, 2)[None].repeat(B, 0).astype(jnp.int32)  # two packed docs
seg = sk.SegmentIds(ids, ids)


def fwd(kern):
  return jax.jit(lambda q, k, v: kern(q, k, v, seg))


def fwd_bwd(kern):
  def loss(q, k, v):
    return jnp.sum(kern(q, k, v, seg).astype(jnp.float32) * ct.astype(jnp.float32))
  return jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2)))


print(json.dumps({'shape': {'B': B, 'N': N, 'T': T, 'D': D}, 'device': str(jax.devices()[0])}), flush=True)
default = dict(bq=512, bkv=512, bkvc=512)
ref_out = fwd(kernel(**default))(q, k, v)
for c in [default, dict(bq=1024, bkv=1024, bkvc=512), dict(bq=1024, bkv=1024, bkvc=1024),
          dict(bq=2048, bkv=1024, bkvc=512), dict(bq=1024, bkv=2048, bkvc=512), dict(bq=2048, bkv=2048, bkvc=512),
          dict(bq=2048, bkv=2048, bkvc=1024), dict(bq=1024, bkv=512, bkvc=512), dict(bq=512, bkv=1024, bkvc=512),
          dict(bq=2048, bkv=512, bkvc=512), dict(bq=1024, bkv=1024, bkvc=256)]:
  try:
    f = fwd(kernel(**c))
    ms = timeit(f, q, k, v)
    print(json.dumps({'pass': 'fwd', **c, 'ms': round(ms, 3), 'rel_out': rel(f(q, k, v), ref_out)}), flush=True)
  except Exception as e:  # pylint: disable=broad-except
    print(json.dumps({'pass': 'fwd', **c, 'error': str(e)[:200]}), flush=True)

ref_val, ref_grads = fwd_bwd(kernel(**default))(q, k, v)
for c in [dict(), dict(fused=True), dict(fused=True, bq_dkv=1024, bkv_dkv=1024, bkv_dkv_c=1024),
          dict(fused=True, bq_dkv=1024, bkv_dkv=1024, bkv_dkv_c=512), dict(fused=True, bq_dkv=2048, bkv_dkv=1024, bkv_dkv_c=1024),
          dict(fused=True, bq_dkv=1024, bkv_dkv=2048, bkv_dkv_c=1024), dict(fused=True, bq_dkv=2048, bkv_dkv=2048, bkv_dkv_c=1024),
          dict(fused=True, bq_dkv=2048, bkv_dkv=2048, bkv_dkv_c=2048), dict(fused=True, bq_dkv=512, bkv_dkv=1024, bkv_dkv_c=1024),
          dict(bq_dkv=1024, bkv_dkv=1024, bkv_dkv_c=1024, bq_dq=1024, bkv_dq=1024)]:
  try:
    f = fwd_bwd(kernel(**default, **c))
    ms = timeit(f, q, k, v)
    val, grads = f(q, k, v)
    print(json.dumps({'pass': 'fwd+bwd', **c, 'ms': round(ms, 3),
                      'rel_dq': rel(grads[0], ref_grads[0]), 'rel_dk': rel(grads[1], ref_grads[1]),
                      'rel_dv': rel(grads[2], ref_grads[2])}), flush=True)
  except Exception as e:  # pylint: disable=broad-except
    print(json.dumps({'pass': 'fwd+bwd', **c, 'error': str(e)[:200]}), flush=True)
print('SPLASH_BENCH_DONE', flush=True)
