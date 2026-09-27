"""Equation checks for the opt-in fused RMT layer, including nonzero reads."""
from functools import partial
from unittest import mock

from absl.testing import absltest
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from layers import rmt, rmt_pallas, rmt_pallas_joined, rmt_pallas_qk, rmt_pallas_minor, rmt_pallas_minor_read, rmt_pallas_minor_joined, rmt_pallas_minor_qk
import rmt_mediumprop_test


class RmtPallasTest(absltest.TestCase):
  _config=rmt_mediumprop_test.RMTMediumPropTest._config

  def test_layer_forward_and_all_gradients(self):
    self._check_layer('rmt_pallas_write')

  def test_joined_read_initialization_forward_and_gradients(self):
    self._check_layer('rmt_join_static_compression')

  def test_joined_pallas_initialization_forward_and_gradients(self):
    self._check_layer('rmt_pallas_joined_read')

  def test_qk_layer_forward_and_gradients(self):
    self._check_layer('rmt_pallas_qk')

  def test_packed_projection_initialization_and_gradients(self):
    self._check_layer('rmt_pack_dynamic_projections')

  def test_hybrid_write_layer_forward_and_gradients(self):
    self._check_layer('rmt_pallas_write_forward_jax')

  def test_token_minor_write_layer_forward_and_gradients(self):
    self._check_layer('rmt_pallas_write_layout')

  def test_token_minor_read_layer_forward_and_gradients(self):
    self._check_layer('rmt_pallas_c8')

  def test_key_token_minor_write_layer_forward_and_gradients(self):
    self._check_layer('rmt_pallas_key_contiguous')

  def test_token_read_write_packed_forward_and_gradients(self):
    self._check_layer('rmt_pallas_read_write')

  def test_token_joined_read_forward_and_gradients(self):
    self._check_layer('rmt_pallas_joined_layout')

  def test_qk_post_forward_and_gradients(self):
    self._check_layer('rmt_pallas_qk_post')

  def _check_layer(self, flag):
    cfg=self._config('RMTCombinedLayerScanNoHealthProfile')
    cfg.get_keys().update(dtype=jnp.float32,rmt_mlp_dim_by_block=[128]*3)
    layer=rmt.RMTLayer(cfg)
    matrix=jax.random.normal(jax.random.key(302),(1,4,48,cfg.head_dim))
    seg=jnp.ones((1,4),jnp.int32)
    pos=jnp.arange(4)[None]
    params=nn.unbox(layer.init(jax.random.key(303),matrix,seg,pos,True,0)['params'])
    if flag in ('rmt_join_static_compression','rmt_pallas_joined_read','rmt_pack_dynamic_projections'):
      cfg.get_keys()[flag]=True
      if flag!='rmt_pack_dynamic_projections':cfg.get_keys()['rmt_join_static_compression']=True
      other=nn.unbox(layer.init(jax.random.key(303),matrix,seg,pos,True,0)['params'])
      self.assertEqual(jax.tree.structure(params),jax.tree.structure(other))
      for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(other)):
        np.testing.assert_array_equal(a,b)
      cfg.get_keys()[flag]=False
      cfg.get_keys()['rmt_join_static_compression']=False
    # Initialization alone leaves zero-key read paths inactive. Perturb every
    # leaf so downstream gradients exercise those paths and learned norm gains.
    leaves,tree=jax.tree.flatten(params)
    params=tree.unflatten([x+.01*jax.random.normal(jax.random.key(304+i),x.shape)
                          for i,x in enumerate(leaves)])
    def run(p,m):
      out=layer.apply({'params':p},m,seg,pos,True,0)[0]
      return jnp.mean(out**2),out
    baseline=jax.jit(jax.value_and_grad(run,argnums=(0,1),has_aux=True))(params,matrix)
    cfg.get_keys()[flag]=True
    if flag in ('rmt_pallas_write_forward_jax','rmt_pallas_write_layout'):cfg.get_keys()['rmt_pallas_write']=True
    if flag=='rmt_pallas_write_layout':cfg.get_keys()[flag]='token_minor'
    if flag=='rmt_pallas_read_write':cfg.get_keys().update(rmt_pallas_c8=True,rmt_pallas_write=True,rmt_pallas_write_layout='token_minor',rmt_pack_dynamic_projections=True)
    if flag=='rmt_pallas_key_contiguous':cfg.get_keys().update(rmt_pallas_write=True,rmt_pallas_write_layout='token_minor')
    if flag=='rmt_pallas_joined_layout':cfg.get_keys().update(rmt_join_static_compression=True,rmt_pallas_joined_read=True,rmt_pallas_joined_layout='token_minor')
    if flag=='rmt_pallas_joined_read':cfg.get_keys()['rmt_join_static_compression']=True
    with mock.patch.object(rmt_pallas,'write_residual',
                           wraps=partial(rmt_pallas.write_residual,interpret=True)) as fused, \
         mock.patch.object(rmt_pallas_joined,'joined_read',
                           wraps=partial(rmt_pallas_joined.joined_read,interpret=True)) as read, \
         mock.patch.object(rmt_pallas_qk,'qk_read',
                           wraps=partial(rmt_pallas_qk.qk_read,interpret=True)) as qk, \
         mock.patch.object(rmt_pallas_minor,'write_residual',
                           wraps=partial(rmt_pallas_minor.write_residual,interpret=True)) as minor, \
         mock.patch.object(rmt_pallas_minor_read,'c8_read',
                           wraps=partial(rmt_pallas_minor_read.c8_read,interpret=True)) as minor_read, \
         mock.patch.object(rmt_pallas_minor_joined,'joined_read',
                           wraps=partial(rmt_pallas_minor_joined.joined_read,interpret=True)) as minor_joined, \
         mock.patch.object(rmt_pallas_minor_qk,'qk_post',
                           wraps=partial(rmt_pallas_minor_qk.qk_post,interpret=True)) as qk_post:
      actual=jax.jit(jax.value_and_grad(run,argnums=(0,1),has_aux=True))(params,matrix)
      if flag in ('rmt_pallas_write_layout','rmt_pallas_key_contiguous'):self.assertGreater(minor.call_count,0)
      if flag in ('rmt_pallas_c8','rmt_pallas_read_write'):self.assertGreater(minor_read.call_count,0)
      if flag=='rmt_pallas_read_write':self.assertGreater(minor.call_count,0)
      if flag=='rmt_pallas_joined_layout':self.assertGreater(minor_joined.call_count,0)
      if flag=='rmt_pallas_qk_post':self.assertGreater(qk_post.call_count,0)
      if flag=='rmt_pallas_qk':self.assertGreater(qk.call_count,0)
      if flag in ('rmt_pallas_write','rmt_pallas_write_forward_jax'):self.assertGreater(fused.call_count,0)
      if flag=='rmt_pallas_joined_read':self.assertGreater(read.call_count,0)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(baseline)):
      a,b=np.asarray(a),np.asarray(b)
      self.assertTrue(np.isfinite(a).all())
      np.testing.assert_allclose(a,b,rtol=3e-4,atol=2e-6)

  def test_padded_carry_initialization_and_gradients(self):
    cfg=self._config('RMTCombinedLayerScanNoHealthProfile')
    cfg.get_keys().update(dtype=jnp.float32,rmt_mlp_dim_by_block=[128]*3)
    layer=rmt.RMTLayer(cfg)
    m=jax.random.normal(jax.random.key(911),(1,4,48,cfg.head_dim))
    seg=jnp.ones((1,4),jnp.int32); pos=jnp.arange(4)[None]
    params=nn.unbox(layer.init(jax.random.key(912),m,seg,pos,True,0)['params'])
    cfg.get_keys()['rmt_pad_value_dim']=128
    pad=lambda x:jnp.pad(x,((0,0),(0,0),(0,0),(0,128-cfg.head_dim)))
    other=nn.unbox(layer.init(jax.random.key(912),pad(m),seg,pos,True,0)['params'])
    for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(other)):np.testing.assert_array_equal(a,b)
    leaves,tree=jax.tree.flatten(params)
    params=tree.unflatten([x+.01*jax.random.normal(jax.random.key(914+i),x.shape) for i,x in enumerate(leaves)])
    def run(p,m):
      padded=cfg.get_keys()['rmt_pad_value_dim']
      out=layer.apply({'params':p},pad(m) if padded else m,seg,pos,True,0)[0]
      return jnp.mean(out[...,:cfg.head_dim]**2),out
    actual=jax.jit(jax.value_and_grad(run,argnums=(0,1),has_aux=True))(params,m)
    np.testing.assert_array_equal(np.asarray(actual[0][1][...,cfg.head_dim:]),0)
    actual=((actual[0][0],actual[0][1][...,:cfg.head_dim]),actual[1])
    cfg.get_keys()['rmt_pad_value_dim']=0
    baseline=jax.jit(jax.value_and_grad(run,argnums=(0,1),has_aux=True))(params,m)
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(baseline)):
      np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=3e-4,atol=2e-6)

  def test_two_device_fused_chain_and_shared_gradients(self):
    if jax.device_count()!=2:self.skipTest('Requires --xla_force_host_platform_device_count=2')
    from tests.rmt_fused_write_read_probe import inputs, reference
    from layers.rmt_pallas_write_read import write_mlp_read
    mesh=jax.sharding.Mesh(np.asarray(jax.devices()),('data',))
    xs=inputs(4,4,jnp.float32)
    def evaluate(fn):
      def loss(*x):
        out=fn(*x)
        return sum(jnp.mean(z*z) for z in out),out
      return jax.jit(jax.value_and_grad(loss,argnums=tuple(range(11)),has_aux=True))(*xs)
    baseline=evaluate(reference)
    with mesh,nn.logical_axis_rules((('activation_batch','data'),)):
      actual=evaluate(partial(write_mlp_read,interpret=True,tile=4))
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(baseline)):
      np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=4e-4,atol=2e-6)

  def test_two_device_batch_sharding_and_shared_gradients(self):
    if jax.device_count()!=2:self.skipTest('Requires --xla_force_host_platform_device_count=2')
    mesh=jax.sharding.Mesh(np.asarray(jax.devices()),('data',))
    shapes=[(4,2,48,75),(4,2,16,8),(48,56),(4,2,16,2)]
    xs=[jax.random.normal(jax.random.key(901+i),s) for i,s in enumerate(shapes)]
    reference=jax.vmap(jax.vmap(rmt_pallas_joined.joined_reference,in_axes=(0,0,None,0)),
                       in_axes=(0,0,None,0))
    def evaluate(fn):
      def loss(*x):
        out=fn(*x)
        return sum(jnp.sum(z*z) for z in out),out
      return jax.jit(jax.value_and_grad(loss,argnums=(0,1,2,3),has_aux=True))(*xs)
    baseline=evaluate(reference)
    with mesh,nn.logical_axis_rules((('activation_batch','data'),)):
      actual=evaluate(partial(rmt_pallas_joined.joined_read,interpret=True,tile=4))
    for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(baseline)):
      np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=1e-4,atol=.003)


if __name__=='__main__':absltest.main()
