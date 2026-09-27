"""Equation checks for the opt-in fused RMT layer, including nonzero reads."""
from functools import partial
from unittest import mock

from absl.testing import absltest
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from layers import rmt, rmt_pallas
import rmt_mediumprop_test


class RmtPallasTest(absltest.TestCase):
  _config=rmt_mediumprop_test.RMTMediumPropTest._config

  def test_layer_forward_and_all_gradients(self):
    cfg=self._config('RMTCombinedLayerScanNoHealthProfile')
    cfg.get_keys().update(dtype=jnp.float32,rmt_mlp_dim_by_block=[128]*3)
    layer=rmt.RMTLayer(cfg)
    matrix=jax.random.normal(jax.random.key(302),(1,4,48,cfg.head_dim))
    seg=jnp.ones((1,4),jnp.int32)
    pos=jnp.arange(4)[None]
    params=nn.unbox(layer.init(jax.random.key(303),matrix,seg,pos,True,0)['params'])
    # Initialization alone leaves zero-key read paths inactive. Perturb every
    # leaf so downstream gradients exercise those paths and learned norm gains.
    leaves,tree=jax.tree.flatten(params)
    params=tree.unflatten([x+.01*jax.random.normal(jax.random.key(304+i),x.shape)
                          for i,x in enumerate(leaves)])
    def run(p,m):
      out=layer.apply({'params':p},m,seg,pos,True,0)[0]
      return jnp.mean(out**2),out
    baseline=jax.jit(jax.value_and_grad(run,argnums=(0,1),has_aux=True))(params,matrix)
    cfg.get_keys()['rmt_pallas_write']=True
    with mock.patch.object(rmt_pallas,'write_residual',partial(rmt_pallas.write_residual,interpret=True)):
      actual=jax.jit(jax.value_and_grad(run,argnums=(0,1),has_aux=True))(params,matrix)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(baseline)):
      a,b=np.asarray(a),np.asarray(b)
      self.assertTrue(np.isfinite(a).all())
      np.testing.assert_allclose(a,b,rtol=3e-4,atol=2e-6)


if __name__=='__main__':absltest.main()
