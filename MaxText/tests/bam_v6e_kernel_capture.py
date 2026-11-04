"""Capture the formal v8t read/write kernel call specs (shapes, dtypes, static kwargs) on CPU.

usage: JAX_PLATFORMS=cpu python MaxText/tests/bam_v6e_kernel_capture.py EXP OUT.pkl
Abstract-evaluates one model forward at the class's per-device batch and sequence length, with
bam_pallas_v8.read_tm / write_tm wrapped to record their first call; bam_v6e_kernel_bench.py
replays the specs with random inputs on a TPU.
"""
import os
import pickle
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

sys.path.insert(0, 'MaxText')
import max_utils  # pylint: disable=wrong-import-position
import pyconfig  # pylint: disable=wrong-import-position
from layers import bam_pallas_v8  # pylint: disable=wrong-import-position
from layers.models import Transformer  # pylint: disable=wrong-import-position


def spec(x):
  if isinstance(x, (jax.Array, np.ndarray)) or hasattr(x, 'aval'):
    return ('__arr__', tuple(x.shape), str(x.dtype))
  return x


def main(exp, out):
  tmp = tempfile.mkdtemp()
  os.makedirs(tmp + '/capture')
  cfg = pyconfig.initialize([None, 'MaxText/configs/base.yml'], exp_class=exp, run_name='capture',
                            enable_checkpointing=False, base_output_directory=tmp + '/', jax_cache_dir='',
                            log_config=False, dataset_type='synthetic')
  calls = {}
  for name in ('read_tm', 'write_tm'):
    orig = getattr(bam_pallas_v8, name)
    def wrapped(*args, _orig=orig, _name=name, **kw):
      calls.setdefault(_name, jax.tree_util.tree_map(spec, (args, kw)))
      return _orig(*args, **kw)
    setattr(bam_pallas_v8, name, wrapped)
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  b, t = int(cfg.per_device_batch_size), cfg.max_target_length
  tokens = jnp.ones((b, t), jnp.int32)
  pos = jnp.broadcast_to(jnp.arange(t)[None], (b, t))
  with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
    model = Transformer(cfg, mesh, quant=None)
    params = jax.eval_shape(lambda k: model.init(k, tokens, pos, tokens, tokens, enable_dropout=False),
                            jax.random.key(0))
    jax.eval_shape(lambda p: model.apply(p, tokens, pos, tokens, tokens, enable_dropout=False,
                                         rngs={'aqt': jax.random.key(0)}), params)
  assert set(calls) == {'read_tm', 'write_tm'}, calls.keys()
  pickle.dump(calls, open(out, 'wb'))
  for k, (args, kw) in calls.items():
    print(k, [a[1] if isinstance(a, tuple) and a and a[0] == '__arr__' else a for a in args])
    print('  kw', {n: (v[1] if isinstance(v, tuple) and v and v[0] == '__arr__' else v) for n, v in kw.items()})


if __name__ == '__main__':
  main(sys.argv[1], sys.argv[2])
