"""Check formal mHC parameter placement on virtual 8/16-device FSDP meshes."""
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest


PROGRAM = r'''
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
from layers.models import Transformer
from tests.bam_prop_test_utils import MLPWriteTest

class Harness(MLPWriteTest, unittest.TestCase):
  pass

h = Harness()
h.setUp()
try:
  assert len(jax.devices()) == 16
  for devices in (8, 16):
    cfg = h.config('PythiaMediumPropT2048SerialMLPmHC', attention='dot_product')
    dims = tuple(devices if axis == 'fsdp' else 1 for axis in cfg.mesh_axes)
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:devices]).reshape(dims), cfg.mesh_axes)
    tokens = jnp.ones((devices, 4), jnp.int32)
    model = Transformer(cfg, mesh, quant=None)
    with mesh, nn.partitioning.axis_rules(cfg.logical_axis_rules):
      shaped = jax.eval_shape(lambda: model.init(jax.random.key(1), tokens,
          jnp.broadcast_to(jnp.arange(4), tokens.shape), tokens, tokens,
          decoder_segment_ids=tokens, enable_dropout=False))['params']
    specs = nn.logical_to_mesh_sharding(nn.get_partition_spec(shaped), mesh, cfg.logical_axis_rules)
    params = flatten_dict(nn.unbox(shaped))
    shardings = flatten_dict(specs)
    total = sum(p.size for p in params.values())
    per_chip = sum(int(np.prod(shardings[path].shard_shape(p.shape))) for path, p in params.items())
    overhead = per_chip / (total / devices) - 1
    assert total == 432339959, total
    assert cfg.sharding_tolerance == .02
    assert overhead < cfg.sharding_tolerance, (devices, overhead)
    controllers = ('attn_connection', 'mlp_connection', 'output_connection')
    routed = [path for path in params if any(name in path for name in controllers)]
    for path in routed:
      if path[-1] in ('router_kernel', 'head_kernel', 'norm_scale'):
        p = params[path]
        axes = shardings[path].spec[0]
        assert 'fsdp' in ((axes,) if isinstance(axes, str) else axes), (path, shardings[path])
        assert shardings[path].shard_shape(p.shape)[0] == p.shape[0] // devices
    # Counterfactual old placement reproduces the real startup failure.
    legacy_per_chip = sum(p.size if path in routed else int(np.prod(shardings[path].shard_shape(p.shape)))
                          for path, p in params.items())
    legacy_overhead = legacy_per_chip / (total / devices) - 1
    assert legacy_overhead > cfg.sharding_tolerance
    if devices == 8:
      assert legacy_per_chip == 58052759, legacy_per_chip
    print('MHC_FSDP_PLACEMENT', devices, total, per_chip, overhead, 'legacy', legacy_overhead, flush=True)
finally:
  h.tearDown()
'''


class ResidualShardingTest(unittest.TestCase):
  def test_formal_mhc_fsdp_parameter_memory(self):
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    flags = re.sub(r'--xla_force_host_platform_device_count=\d+', '', env.get('XLA_FLAGS', ''))
    env['XLA_FLAGS'] = flags + ' --xla_force_host_platform_device_count=16'
    env['JAX_PLATFORMS'] = 'cpu'
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'MaxText'), str(root / 'MaxText/tests'), env.get('PYTHONPATH', '')])
    result = subprocess.run([sys.executable, '-c', PROGRAM], cwd=root, env=env,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    self.assertEqual(result.returncode, 0, result.stdout[-14000:])
    print('\n'.join(line for line in result.stdout.splitlines() if 'MHC_FSDP_PLACEMENT' in line), flush=True)


if __name__ == '__main__':
  unittest.main()
