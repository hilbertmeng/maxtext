"""TPU microbenchmark: BAM Pallas core vs the original XLA math, one layer, real shapes.

usage: python MaxText/tests/bam_pallas_benchmark.py [batch] [read_tile] [write_tile] [vmem_mib]
"""
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'MaxText')
sys.path.insert(0, 'MaxText/tests')
from layers import bam_pallas as bp  # pylint: disable=wrong-import-position
import bam_pallas_test as ref  # pylint: disable=wrong-import-position

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 2
rtile = int(sys.argv[2]) if len(sys.argv) > 2 else 128
wtile = int(sys.argv[3]) if len(sys.argv) > 3 else 128
vmem = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != '0' else None
body = sys.argv[5] if len(sys.argv) > 5 else 'blocked'
block = int(sys.argv[6]) if len(sys.argv) > 6 else 4
ref.B, ref.T, ref.N = batch, 4096, 20
dt = jnp.bfloat16


def timeit(fn, *args, n=20):
  out = fn(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(n):
    out = fn(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) / n * 1e3


def rel(a, b):
  a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
  return float(np.max(np.abs(a - b)) / (np.max(np.abs(b)) + 1e-12))


m_tm, params, tokens = ref.inputs(dt)
kmajor = body in ('v4', 'v5', 'v7', 'v7u')
m_minor = jnp.transpose(m_tm, (0, 2, 3, 1) if kmajor else (0, 3, 2, 1))
m_back = (0, 3, 1, 2) if kmajor else (0, 3, 2, 1)
sw = bp.static_weight(*params)
kw = dict(qk_cols=ref.QKC, read_epsilon=ref.READ_EPS, key_scale=ref.SCALE,
          forward_tile=rtile, reverse_tile=rtile, vmem_mib=vmem, body=body, head_block=block)
cts = tuple(jax.random.normal(jax.random.PRNGKey(i), (batch, 4096, 20, 96)).astype(dt) for i in range(4))

res = {'batch': batch, 'read_tile': rtile, 'write_tile': wtile, 'vmem_mib': vmem, 'body': body, 'block': block}

# ---- read ----
k_read = jax.jit(lambda m, sw, *t: bp.read(m, sw, *t, **kw))
x_read = jax.jit(lambda m, *a: ref.original_read(m, *a))
got = k_read(m_minor, sw, *tokens)
want = x_read(m_tm, *params, *tokens)
res['read_fwd_rel_err'] = [rel(a, b) for a, b in zip(got, want)]
res['read_fwd_ms'] = {'pallas': timeit(k_read, m_minor, sw, *tokens), 'xla': timeit(x_read, m_tm, *params, *tokens)}
minor_tokens = tuple(jnp.moveaxis(x, 1, -1) for x in tokens)
if not kmajor:
  opts = bp._freeze(dict(heads=20, qk_cols=ref.QKC, read_epsilon=ref.READ_EPS, key_scale=ref.SCALE,
                         forward_tile=rtile, reverse_tile=rtile, vmem_mib=vmem, interpret=False,
                         body=body, head_block=block))
  k_only = jax.jit(lambda m, sw, *t: bp._read(m, sw, *t, opts))
  res['read_fwd_kernel_only_ms'] = timeit(k_only, m_minor, sw, *minor_tokens)


def vjp_time(fn, args):
  def step(*a):
    out, pull = jax.vjp(fn, *a)
    return pull(tuple(cts))
  return jax.jit(step), args


k_rb, _ = vjp_time(lambda m, sw, *t: bp.read(m, sw, *t, **kw), None)
x_rb, _ = vjp_time(lambda m, *a: ref.original_read(m, *a), None)
gk = k_rb(m_minor, sw, *tokens)
gx = x_rb(m_tm, *params, *tokens)
res['read_grad_rel_err_m'] = rel(jnp.transpose(gk[0], m_back), gx[0])
res['read_fwdbwd_ms'] = {'pallas': timeit(k_rb, m_minor, sw, *tokens), 'xla': timeit(x_rb, m_tm, *params, *tokens)}

# ---- write ----
for n_groups in (1, 2):
  groups = ref.write_groups(dt, n_groups)
  flat = [x for g in groups for x in g]
  wkw = dict(epsilon=ref.WRITE_EPS, forward_tile=wtile, reverse_tile=wtile, vmem_mib=vmem, body=body, row_block=block, head_block=block)
  k_w = jax.jit(lambda m, *f: bp.write(m, [f[i:i + 3] for i in range(0, len(f), 3)], **wkw))
  x_w = jax.jit(lambda m, *f: ref.original_write(m, *f))
  res[f'write{n_groups}_fwd_rel_err'] = rel(jnp.transpose(k_w(m_minor, *flat), m_back), x_w(m_tm, *flat))
  res[f'write{n_groups}_fwd_ms'] = {'pallas': timeit(k_w, m_minor, *flat), 'xla': timeit(x_w, m_tm, *flat)}
  ctm = jax.random.normal(jax.random.PRNGKey(9), m_tm.shape).astype(dt)
  ctn = jnp.transpose(ctm, (0, 2, 3, 1) if kmajor else (0, 3, 2, 1))

  def kgrad(m, *f):
    out, pull = jax.vjp(lambda m, *f: bp.write(m, [f[i:i + 3] for i in range(0, len(f), 3)], **wkw), m, *f)
    return pull(ctn)

  def xgrad(m, *f):
    out, pull = jax.vjp(ref.original_write, m, *f)
    return pull(ctm)

  kg, xg = jax.jit(kgrad), jax.jit(xgrad)
  a, b = kg(m_minor, *flat), xg(m_tm, *flat)
  res[f'write{n_groups}_grad_rel_err'] = [rel(x, y) for x, y in zip(a[1:], b[1:])]
  res[f'write{n_groups}_fwdbwd_ms'] = {'pallas': timeit(kg, m_minor, *flat), 'xla': timeit(xg, m_tm, *flat)}

print('BENCH_JSON ' + json.dumps(res))
