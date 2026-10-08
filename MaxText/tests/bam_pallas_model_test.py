"""CPU end-to-end check: the fused-core model equals the original XLA model (FP32)."""
import os
import sys
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, 'MaxText')
import max_utils  # pylint: disable=wrong-import-position
import pyconfig  # pylint: disable=wrong-import-position
from layers.models import Transformer  # pylint: disable=wrong-import-position


def build(exp, out):
  os.makedirs(os.path.join(out, exp), exist_ok=True)
  cfg = pyconfig.initialize(['x', 'MaxText/configs/base.yml', f'exp_class={exp}', f'run_name={exp}',
                             f'base_output_directory={out}', 'dataset_type=synthetic',
                             'enable_checkpointing=False', 'jax_cache_dir='])
  mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
  return cfg, Transformer(cfg, mesh, quant=None)


class ModelEquivalenceTest(unittest.TestCase):

  def test_logits_and_gradients(self):
    out = tempfile.mkdtemp()
    cfg, xla = build('BamPallasCoreTinyXlaTest', out)
    fused_models = [build(e, out)[1] for e in ('BamPallasCoreTinyPallasTest', 'BamPallasCoreTinyPallasV3Test', 'BamPallasCoreTinyPallasV4Test', 'BamPallasCoreTinyPallasV6Test', 'BamPallasCoreTinyPallasV7Test')]
    b, t = int(cfg.global_batch_size_to_train_on), cfg.max_target_length
    tokens = jax.random.randint(jax.random.PRNGKey(0), (b, t), 0, 1000)
    positions = jnp.broadcast_to(jnp.arange(t), (b, t))
    segments = jnp.ones((b, t), jnp.int32)
    rngs = {'params': jax.random.PRNGKey(1), 'dropout': jax.random.PRNGKey(2), 'aqt': jax.random.PRNGKey(3)}
    params = xla.init(rngs, tokens, positions, segments, tokens)
    # Non-zero gate/key kernels so every BAM path carries signal.
    leaves, tree = jax.tree.flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(4), len(leaves))
    params = jax.tree.unflatten(tree, [x + 0.02 * jax.random.normal(k, x.shape, x.dtype)
                                       for x, k in zip(leaves, keys)])
    def out_of(model, p):
      out = model.apply(p, tokens, positions, segments, tokens, enable_dropout=False, rngs=rngs)
      return (out[0] if isinstance(out, tuple) else out).astype(jnp.float32)

    shape = jax.eval_shape(lambda p: out_of(xla, p), params).shape
    print('model output shape', shape)
    ct = jax.random.normal(jax.random.PRNGKey(5), shape)

    def loss(model, p):
      out = out_of(model, p)
      return jnp.sum(out * ct) / out.size, out

    (lx, ox), gx = jax.value_and_grad(lambda p: loss(xla, p), has_aux=True)(params)
    for fused in fused_models:
      self._compare(loss, xla, fused, params, lx, ox, gx)

  def _compare(self, loss, xla, fused, params, lx, ox, gx):
    (lf, of), gf = jax.value_and_grad(lambda p: loss(fused, p), has_aux=True)(params)
    scale = float(jnp.max(jnp.abs(ox)))
    self.assertLess(float(jnp.max(jnp.abs(ox - of))) / scale, 1e-4)
    worst = []
    for (path, a), b_ in zip(jax.tree_util.tree_flatten_with_path(gx)[0], jax.tree.leaves(gf)):
      a, b_ = np.asarray(a), np.asarray(b_)
      err = np.max(np.abs(a - b_)) / (np.max(np.abs(a)) + 1e-12)
      worst.append((err, jax.tree_util.keystr(path)))
    worst.sort(reverse=True)
    print('loss', float(lx), float(lf), 'worst grads', worst[:5])
    self.assertLess(worst[0][0], 1e-3, worst[:5])


if __name__ == '__main__':
  unittest.main()
