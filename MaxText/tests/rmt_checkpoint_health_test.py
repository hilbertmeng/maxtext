"""Instrumentation must preserve checkpoint shapes, CE and gradients."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from tests.rmt_xlprop_test import XLPropTest as _Helpers

class CheckpointHealthTest(unittest.TestCase):
  config = _Helpers.config
  model_args = _Helpers.model_args

  def test_probe_identity_and_valid_health(self):
    cfg = self.config('RMTXLPropT4096TruePileAllLocalK60EmbedUnembedDirect40NoOMPreNorm',
                      base_num_decoder_layers=2, base_emb_dim=640, head_dim=32,
                      base_mlp_dim=64, vocab_size=128)
    cfg.get_keys()['dtype'] = jnp.float32
    model, args = self.model_args(cfg)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(model.init(jax.random.key(3), **args))
      def loss(p):return jnp.mean(model.apply(p, **args)[0])
      base, grad = jax.jit(jax.value_and_grad(loss))(params)
      with tempfile.TemporaryDirectory() as directory:
        os.environ['RMT_HEALTH_FILE'] = directory+'/health.jsonl'
        os.environ['RMT_NORM_TAP_FILE'] = directory+'/taps.jsonl'
        cfg.get_keys().update(rmt_crossscale_health_probe=True, rmt_norm_probe=True)
        result, dg = jax.jit(jax.value_and_grad(loss))(params)
        jax.effects_barrier()
        rows=[json.loads(line) for line in Path(directory,'health.jsonl').read_text().splitlines()]
        self.assertTrue(any(row['tag']=='attention' for row in rows))
        readouts=[row for row in rows if row['tag']=='readout']
        self.assertTrue(readouts)
        for row in readouts:
          self.assertTrue(all(np.isfinite(v) for v in row.values() if isinstance(v,(float,int))))
          self.assertGreater(row['entropy'],0)
          self.assertGreater(row['centered_logits_rms'],0)
          self.assertAlmostEqual(row['ce_alpha_1'],float(result),places=5)
    np.testing.assert_allclose(base,result,rtol=1e-6,atol=1e-6)
    for x,y in zip(jax.tree.leaves(grad),jax.tree.leaves(dg)):
      np.testing.assert_allclose(x,y,rtol=2e-5,atol=2e-6)

if __name__=='__main__': unittest.main(defaultTest='CheckpointHealthTest')
