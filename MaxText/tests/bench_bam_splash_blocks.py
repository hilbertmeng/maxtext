"""Sweep Splash block sizes at Pythia H attention shapes (single device, per-device batch 8).

q/k [8,32,2048,100] (pre-scaled), v [8,32,2048,80], strict-causal mask with the negative-diagonal sink,
fused backward. Prints median ms of forward and forward+backward per block configuration.

usage (TPU worker, repo root): python3 MaxText/tests/bench_bam_splash_blocks.py
"""
import itertools
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as ska
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_mask as skm

from layers.bam_attention_diagonal_zero import make_negative_diagonal_splash

B, N, T, DQ, DV = 8, 32, 2048, 100, 80


def kernel(bq, bkv, bkvc, bq_dkv, bkv_dkv, bkvc_dkv):
  blocks = ska.BlockSizes(block_q=bq, block_kv=bkv, block_kv_compute=bkvc, block_q_dkv=bq_dkv,
                          block_kv_dkv=bkv_dkv, block_kv_dkv_compute=bkvc_dkv, block_q_dq=None,
                          block_kv_dq=None, use_fused_bwd_kernel=True)
  mask = skm.MultiHeadMask(masks=(skm.CausalMask(shape=(T, T), offset=-1),) * N)
  k = ska.make_splash_mha(mask=mask, head_shards=1, q_seq_shards=1, block_sizes=blocks)
  return make_negative_diagonal_splash(k)


def timeit(fn, args, n=15):
  jax.block_until_ready(fn(*args))
  ts = []
  for _ in range(n):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    ts.append(time.perf_counter() - t0)
  return sorted(ts)[n // 2] * 1e3


if __name__ == '__main__':
  ks = jax.random.split(jax.random.key(0), 3)
  q = (jax.random.normal(ks[0], (B, N, T, DQ), jnp.float32) * .1).astype(jnp.bfloat16)
  k = jax.random.normal(ks[1], (B, N, T, DQ), jnp.float32).astype(jnp.bfloat16)
  v = jax.random.normal(ks[2], (B, N, T, DV), jnp.float32).astype(jnp.bfloat16)
  configs = [(1024, 1024, 512, 1024, 1024, 1024)]   # current model setting first
  for bq, bkv, bkvc in itertools.product((512, 1024, 2048), (512, 1024, 2048), (256, 512, 1024)):
    if bkvc <= bkv:
      configs.append((bq, bkv, bkvc, 1024, 1024, 1024))
  for bq_dkv, bkv_dkv, bkvc_dkv in itertools.product((512, 1024, 2048), (512, 1024, 2048), (256, 512, 1024)):
    if bkvc_dkv <= bkv_dkv:
      configs.append((1024, 1024, 512, bq_dkv, bkv_dkv, bkvc_dkv))
  for cfg in configs:
    try:
      kern = jax.vmap(kernel(*cfg))
      fwd = jax.jit(kern)
      bwd = jax.jit(jax.grad(lambda q, k, v: jnp.sum(kern(q, k, v).astype(jnp.float32) * .01), argnums=(0, 1, 2)))
      print(f'SPLASH {cfg} fwd {timeit(fwd, (q, k, v)):7.3f} fwd+bwd {timeit(bwd, (q, k, v)):7.3f}', flush=True)
    except Exception as e:   # some tilings are invalid for the shapes/VMEM
      print(f'SPLASH {cfg} FAILED {type(e).__name__}: {str(e)[:120]}', flush=True)
