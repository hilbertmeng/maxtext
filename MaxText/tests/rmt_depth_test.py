"""Focused checks for uniform-width RMT layer scan and its health export."""
import contextlib
import io
from functools import partial
from unittest import mock

from absl.testing import absltest
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np

import max_utils
from layers import models, rmt
import rmt_mediumprop_test


class RMTDepthTest(absltest.TestCase):

  _config = rmt_mediumprop_test.RMTMediumPropTest._config

  def test_block_scan_matches_direct_scan_with_mapped_parameters(self):
    cfg=self._config('RMTCombinedLayerScanNoHealthProfile')
    cfg.get_keys().update(num_decoder_layers=3,base_num_decoder_layers=3,
                          dtype=jnp.float32,rmt_mlp_dim_by_block=[128]*3,query_chunk_size=2)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    model=models.Transformer(config=cfg,mesh=mesh,quant=None)
    args=dict(decoder_input_tokens=jnp.array([[1,2,3,4]],jnp.int32),
              decoder_positions=jnp.arange(4)[None],decoder_target_tokens=jnp.array([[2,3,4,5]],jnp.int32),
              decoder_target_mask=jnp.ones((1,4),jnp.float32),decoder_segment_ids=jnp.ones((1,4),jnp.int32),
              enable_dropout=False)
    axis=cfg.param_scan_axis
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(839),**args)['params'])
      leaves,tree=jax.tree.flatten(params)
      params=tree.unflatten([x+.01*jax.random.normal(jax.random.key(840+i),x.shape) for i,x in enumerate(leaves)])
      def loss(p):return jnp.mean(model.apply({'params':p},**args)[0])
      direct=jax.jit(jax.value_and_grad(loss))(params)
      blocked=jax.tree.map(lambda x:x,params)
      blocked['decoder']['layers']={
          f'layer_{i}':jax.tree.map(lambda x:jnp.take(x,jnp.array([i]),axis=axis),params['decoder']['layers'])
          for i in range(3)}
      cfg.get_keys()['rmt_block_scan']=True
      actual=jax.jit(jax.value_and_grad(loss))(blocked)
      grad=actual[1]
      layers=grad['decoder']['layers']
      grad['decoder']['layers']=jax.tree.map(lambda *xs:jnp.concatenate(xs,axis=axis),
                                             *(layers[f'layer_{i}'] for i in range(3)))
      for a,b in zip(jax.tree.leaves((actual[0],grad)),jax.tree.leaves(direct)):
        np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=4e-4,atol=3e-6)

  def test_fused_scan_remat_matches_original(self):
    from layers import rmt_pallas_minor, rmt_pallas_minor_read, rmt_pallas_minor_qk
    cfg=self._config('RMTCombinedLayerScanNoHealthProfile')
    cfg.get_keys().update(num_decoder_layers=3,base_num_decoder_layers=3,
                          dtype=jnp.float32,rmt_mlp_dim_by_block=[128]*3,query_chunk_size=2)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    model=models.Transformer(config=cfg,mesh=mesh,quant=None)
    args=dict(decoder_input_tokens=jnp.array([[1,2,3,4]],jnp.int32),
              decoder_positions=jnp.arange(4)[None],decoder_target_tokens=jnp.array([[2,3,4,5]],jnp.int32),
              decoder_target_mask=jnp.ones((1,4),jnp.float32),decoder_segment_ids=jnp.ones((1,4),jnp.int32),
              enable_dropout=False)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(829),**args)['params'])
      leaves,tree=jax.tree.flatten(params)
      params=tree.unflatten([x+.01*jax.random.normal(jax.random.key(830+i),x.shape) for i,x in enumerate(leaves)])
      def loss(p):return jnp.mean(model.apply({'params':p},**args)[0])
      baseline=jax.jit(jax.value_and_grad(loss))(params)
      cfg.get_keys().update(rmt_pallas_write=True,rmt_pallas_write_layout='token_minor',
                            rmt_pallas_c8=True,rmt_pallas_qk_post=True,rmt_pack_dynamic_projections=True)
      with mock.patch.object(rmt_pallas_minor,'write_residual',
                             wraps=partial(rmt_pallas_minor.write_residual,interpret=True)), \
           mock.patch.object(rmt_pallas_minor_read,'c8_read',
                             wraps=partial(rmt_pallas_minor_read.c8_read,interpret=True)), \
           mock.patch.object(rmt_pallas_minor_qk,'qk_post',
                             wraps=partial(rmt_pallas_minor_qk.qk_post,interpret=True)):
        for policy in ('full','save_state','save_state_mlp','save_state_dynamic'):
          cfg.get_keys()['rmt_remat_policy']=policy
          actual=jax.jit(jax.value_and_grad(loss))(params)
          for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(baseline)):
            self.assertTrue(np.isfinite(np.asarray(a)).all())
            np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=4e-4,atol=3e-6)

  def test_leading_parameter_scan_axis_matches(self):
    cfg=self._config('RMTCombinedLayerScanNoHealthProfile')
    cfg.get_keys().update(num_decoder_layers=3,base_num_decoder_layers=3,
                          dtype=jnp.float32,rmt_mlp_dim_by_block=[128]*3,query_chunk_size=2)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    model=models.Transformer(config=cfg,mesh=mesh,quant=None)
    args=dict(decoder_input_tokens=jnp.array([[1,2,3,4]],jnp.int32),
              decoder_positions=jnp.arange(4)[None],decoder_target_tokens=jnp.array([[2,3,4,5]],jnp.int32),
              decoder_target_mask=jnp.ones((1,4),jnp.float32),decoder_segment_ids=jnp.ones((1,4),jnp.int32),
              enable_dropout=False)
    def move(p,src,dst):
      p=jax.tree.map(lambda x:x,p)
      p['decoder']['layers']=jax.tree.map(lambda x:jnp.moveaxis(x,src,dst),p['decoder']['layers'])
      return p
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(819),**args)['params'])
      cfg.get_keys()['param_scan_axis']=0
      other=nn.unbox(model.init(jax.random.key(819),**args)['params'])
      for a,b in zip(jax.tree.leaves(move(params,1,0)),jax.tree.leaves(other)):
        np.testing.assert_array_equal(a,b)
      leaves,tree=jax.tree.flatten(params)
      params=tree.unflatten([x+.01*jax.random.normal(jax.random.key(820+i),x.shape) for i,x in enumerate(leaves)])
      def loss(p):return jnp.mean(model.apply({'params':p},**args)[0])
      actual=jax.jit(jax.value_and_grad(loss))(move(params,1,0))
      actual=(actual[0],move(actual[1],0,1))
      cfg.get_keys()['param_scan_axis']=1
      baseline=jax.jit(jax.value_and_grad(loss))(params)
      for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(baseline)):
        np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=3e-4,atol=2e-6)

  def test_selective_remat_matches_full_scan(self):
    cfg=self._config('RMTCombinedLayerScanNoHealthProfile')
    cfg.get_keys().update(num_decoder_layers=3,base_num_decoder_layers=3,
                          dtype=jnp.float32,rmt_mlp_dim_by_block=[128]*3,query_chunk_size=2)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    model=models.Transformer(config=cfg,mesh=mesh,quant=None)
    args=dict(decoder_input_tokens=jnp.array([[1,2,3,4]],jnp.int32),
              decoder_positions=jnp.arange(4)[None],decoder_target_tokens=jnp.array([[2,3,4,5]],jnp.int32),
              decoder_target_mask=jnp.ones((1,4),jnp.float32),decoder_segment_ids=jnp.ones((1,4),jnp.int32),
              enable_dropout=False)
    with contextlib.redirect_stdout(io.StringIO()):
      params=nn.unbox(model.init(jax.random.key(811),**args)['params'])
      leaves,tree=jax.tree.flatten(params)
      params=tree.unflatten([x+.01*jax.random.normal(jax.random.key(812+i),x.shape) for i,x in enumerate(leaves)])
      def loss(p):return jnp.mean(model.apply({'params':p},**args)[0])
      baseline=jax.jit(jax.value_and_grad(loss))(params)
      for policy in ('save_dense','save_dense_state','save_state','save_state_mlp','attention_only'):
        cfg.get_keys()['rmt_remat_policy']=policy
        actual=jax.jit(jax.value_and_grad(loss))(params)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(baseline)):
          np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=3e-4,atol=2e-6)

  def test_layer_scan_forward_gradients_and_health(self):
    name = 'RMTMediumPropK48DynamicFull48RoPE18VectorNormMHABudgetDynamicEmbeddingUnembeddingDirect32L22'
    cfg = self._config(name)
    # Keep the actual layer-scan path; shorten width, depth and sequence for gradients.
    cfg.get_keys().update(num_decoder_layers=7, base_num_decoder_layers=7,
                          dtype=jnp.float32, rmt_mlp_dim_by_block=[128]*3)
    mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg), cfg.mesh_axes)
    model = models.Transformer(config=cfg, mesh=mesh, quant=None)
    args = dict(decoder_input_tokens=jnp.array([[1,2,3,4]], jnp.int32),
                decoder_positions=jnp.arange(4)[None],
                decoder_target_tokens=jnp.array([[2,3,4,5]], jnp.int32),
                decoder_target_mask=jnp.ones((1,4), jnp.float32),
                decoder_segment_ids=jnp.ones((1,4), jnp.int32),
                enable_dropout=False)
    with contextlib.redirect_stdout(io.StringIO()):
      params = nn.unbox(model.init(jax.random.key(701), **args)['params'])
      def loss(p):
        output, aux = model.apply({'params':p}, **args, mutable=['intermediates'])
        return jnp.mean(output[0]), aux
      (value, aux), grads = jax.value_and_grad(loss, has_aux=True)(params)
    self.assertTrue(bool(jnp.isfinite(value)))
    self.assertTrue(all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads)))
    decoder = params['decoder']
    self.assertNotIn('tail_layer_0', decoder)
    self.assertIn('dynamic_embedding_write', decoder)
    self.assertIn('dynamic_unembedding_read', decoder)
    # All seven layers are independently parameterized; the last layer trains.
    kernel = decoder['layers']['mlp']['wo']['kernel']
    self.assertEqual(kernel.shape, (128, 7, cfg.emb_dim))
    layer_grad = grads['decoder']['layers']['mlp']['wo']['kernel']
    self.assertGreater(float(jnp.linalg.norm(layer_grad[:, -1, :])), 0.)
    health = aux['intermediates']['decoder']
    self.assertEqual(health['layers']['rmt_dynamic_health'][0].shape, (7,41))
    import train
    metrics = {'scalar': {}}
    train.record_rmt_dynamic_health_metrics(metrics, aux, cfg)
    names = [k for k in metrics['scalar'] if k.startswith('rmt/dynamic/')]
    self.assertLen(names, 7*41)
    self.assertIn('rmt/dynamic/layer_006/q_gate_mean', metrics['scalar'])
    self.assertIn('rmt/embedding/gate_mean', metrics['scalar'])
    self.assertIn('rmt/unembedding/gate_mean', metrics['scalar'])
    np.testing.assert_allclose(metrics['scalar']['rmt/dynamic/layer_006/q_gate_mean'],
        health['layers']['rmt_dynamic_health'][0][6, rmt.RMT_DYNAMIC_HEALTH_NAMES.index('q_gate_mean')])


if __name__ == '__main__':
  absltest.main()
