"""TPU microbenchmark of the captured v8t read/write kernel calls with kwarg ablations.

usage: python MaxText/tests/bam_v6e_kernel_bench.py SPEC.pkl ARMS.json
ARMS.json: [{"name": "base"}, {"name": "no_norm", "read": {"m_norm": null}}, {"name": "w16", "write": {"chunk": 16}}]
Each arm times read and write forward and forward+reverse (ms per call, median of repeats) with
random inputs of the captured shapes; overrides replace captured kwargs (null drops an array kwarg).
"""
import json
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'MaxText')
from layers import bam_pallas_v8  # pylint: disable=wrong-import-position

KEY = iter(jax.random.split(jax.random.PRNGKey(0), 4096))


def is_arr(x):
  return isinstance(x, tuple) and len(x) == 3 and x[0] == '__arr__'


def make(x, kind='normal'):
  _, shape, dtype = x
  k = next(KEY)
  if kind == 'pos':
    return (1. + jax.random.uniform(k, shape)).astype(dtype)
  if kind == 'unit':
    return jax.random.uniform(k, shape, minval=0., maxval=.9).astype(dtype)
  return (.5 * jax.random.normal(k, shape)).astype(dtype)


def build(spec_tree, kinds=None):
  return jax.tree_util.tree_map(lambda x: make(x, (kinds or {}).get(id(x), 'normal')) if is_arr(x) else x,
                                spec_tree, is_leaf=is_arr)


def materialize_kw(kw):
  out = {}
  for name, v in kw.items():
    if name == 'qk_norm':     # (ss_q, ss_k, m_q, m_k, eps): positive statistics and multipliers
      out[name] = tuple(make(x, 'pos') if is_arr(x) else x for x in v)
    elif name == 'shift':
      out[name] = make(v, 'unit')
    elif name.endswith('_dynamic_scale'):
      out[name] = make(v, 'pos')
    else:
      out[name] = build(v)
  return out


def timeit(fn, *args, n=10, reps=3):
  jax.block_until_ready(fn(*args))
  best = []
  for _ in range(reps):
    t0 = time.perf_counter()
    for _ in range(n):
      out = fn(*args)
    jax.block_until_ready(out)
    best.append((time.perf_counter() - t0) / n * 1e3)
  return float(np.median(best))


def bench(fn, args):
  fwd = jax.jit(fn)
  outs = jax.eval_shape(fn, *args)
  cts = jax.tree_util.tree_map(lambda o: (.1 * jax.random.normal(next(KEY), o.shape)).astype(o.dtype), outs)

  def loss(*a):
    return sum(jnp.sum(o.astype(jnp.float32) * c.astype(jnp.float32))
               for o, c in zip(jax.tree_util.tree_leaves(fn(*a)), jax.tree_util.tree_leaves(cts)))
  diff = [i for i, a in enumerate(args) if hasattr(a, 'dtype') and jnp.issubdtype(a.dtype, jnp.floating)]
  both = jax.jit(jax.grad(loss, argnums=tuple(diff)))
  return timeit(fwd, *args), timeit(both, *args)


def main(spec_path, arms_path):
  spec = pickle.load(open(spec_path, 'rb'))
  arms = json.load(open(arms_path))
  r_args, r_kw = spec['read_tm']
  w_args, w_kw = spec['write_tm']
  ra = list(build(list(r_args)))
  wa = build(list(w_args))
  r_kw0, w_kw0 = materialize_kw(r_kw), materialize_kw(w_kw)
  print(jax.devices()[0].device_kind, flush=True)
  for arm in arms:
    rk = dict(r_kw0); wk = dict(w_kw0)
    for k, v in (arm.get('read') or {}).items():
      rk.pop(k, None) if v is None else rk.__setitem__(k, tuple(v) if isinstance(v, list) else v)
    for k, v in (arm.get('write') or {}).items():
      wk.pop(k, None) if v is None else wk.__setitem__(k, tuple(v) if isinstance(v, list) else v)
    res = {'arm': arm['name']}
    try:
      rf, rb = bench(lambda *a: bam_pallas_v8.read_tm(*a, **rk), ra)
      res.update(read_fwd=round(rf, 3), read_fwdbwd=round(rb, 3))
    except Exception as e:  # pylint: disable=broad-except
      res['read_error'] = repr(e)[:300]
    if not arm.get('skip_write'):
      try:
        m, groups = wa
        wf, wb = bench(lambda m, *flat: bam_pallas_v8.write_tm(
            m, [tuple(flat[3 * i:3 * i + 3]) for i in range(len(groups))], **wk), [m] + [x for g in groups for x in g])
        res.update(write_fwd=round(wf, 3), write_fwdbwd=round(wb, 3))
      except Exception as e:  # pylint: disable=broad-except
        res['write_error'] = repr(e)[:300]
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])
