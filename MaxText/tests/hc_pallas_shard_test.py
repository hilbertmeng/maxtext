"""4-device CPU shard_map check of the hc_pallas wrappers (interpreted kernels).

XLA_FLAGS must be set before JAX initializes, so the check runs in a subprocess.
Single-device tests bypass shard_map and cannot catch wrapper/out_spec or
replicated-weight gradient-reduction errors.
"""
import os
import subprocess
import sys
import unittest


class HCPallasShardTest(unittest.TestCase):

  def test_sharded_kernels_match_xla_reference(self):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hc_pallas_shard_check.py')
    env = dict(os.environ, JAX_PLATFORMS='cpu')
    out = subprocess.run([sys.executable, script], env=env, capture_output=True, text=True, timeout=1500)
    self.assertEqual(out.returncode, 0, out.stdout[-3000:] + out.stderr[-3000:])
    self.assertIn('HC_PALLAS_SHARD_OK', out.stdout)


if __name__ == '__main__':
  unittest.main()
