"""Timing-only ablations of the v8t read kernel phases (outputs are wrong under ablation).

usage: ABL=fill,slabs,shiftgate python MaxText/tests/bam_read_ablate_bench.py V [V ...]
"""
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, 'MaxText')
sys.path.insert(0, 'MaxText/tests')
from layers import bam_pallas_v8 as v8  # pylint: disable=wrong-import-position
import bam_read_v_sweep_bench as bench  # pylint: disable=wrong-import-position

ABL = set(filter(None, os.environ.get('ABL', '').split(',')))
if 'fill' in ABL:
  v8._tm_fill_keys = lambda *a, **k: None
if 'shiftgate' in ABL:
  v8._fill_shift_gate = lambda *a, **k: None
if 'slabs' in ABL:
  _slabs = v8._tm_slabs

  def _no_norm(eps, scale, rq, lq, rk, lk, rr, lo, lv):
    g = jnp.concatenate([scale * v8._sigmoid(l[...]) for l in (lq, lk, lv, lo)], axis=0)
    return jnp.ones(g.shape, v8.F32), g
  v8._tm_slabs = _no_norm

if __name__ == '__main__':
  print('ABL', sorted(ABL), flush=True)
  bench.main()
