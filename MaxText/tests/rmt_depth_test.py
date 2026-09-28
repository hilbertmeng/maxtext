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
    self._check_fused_scan_remat('autodiff')

  def test_analytic_write_scan_remat_matches_original(self):
    self._check_fused_scan_remat('analytic')

  def test_joint_write_scan_remat_matches_original(self):
    self._check_fused_scan_remat("joint")

  def test_batched_write_scan_remat_matches_original(self):
    self._check_fused_scan_remat("batched")

  def test_major_reverse_scan_remat_matches_original(self):
    self._check_fused_scan_remat("joint_major")

  def test_major_direct_reverse_scan_remat_matches_original(self):
    self._check_fused_scan_remat("joint_major_direct")

  def test_v_only_read_scan_remat_matches_noo_reference(self):
    self._check_fused_scan_remat('autodiff', no_o=True)

  def test_fused_write_read_scan_remat_matches_original(self):
    self._check_fused_scan_remat('autodiff', whole_stage=True)

  def test_chunked_fused_write_read_scan_remat_matches_original(self):
    self._check_fused_scan_remat('autodiff', whole_stage=True, stage_chunk=64)

  def test_three_stage_noo_scan_remat_matches_reference(self):
    self._check_fused_scan_remat('autodiff',no_o=True,whole_stage=True,stage_chunk=32,three_stage=True)

  def test_three_stage_joined_reverse_scan_remat_matches_reference(self):
    self._check_fused_scan_remat('autodiff',no_o=True,whole_stage=True,stage_chunk=32,three_stage=True,middle_mode='joined')

  def test_three_stage_minor_reverse_scan_remat_matches_reference(self):
    self._check_fused_scan_remat('autodiff',no_o=True,whole_stage=True,stage_chunk=32,three_stage=True,middle_mode='minor')

  def test_three_stage_saved_outputs_scan_remat_matches_reference(self):
    self._check_fused_scan_remat('autodiff',no_o=True,whole_stage=True,stage_chunk=32,three_stage=True,middle_mode='joined',save_middle=True)

  def test_three_stage_internal_recompute_scan_remat_matches_reference(self):
    self._check_fused_scan_remat('autodiff',no_o=True,whole_stage=True,stage_chunk=32,three_stage=True,middle_mode='minor_recompute',save_middle=True)

  def test_three_stage_native_saved_outputs_scan_remat_matches_reference(self):
    self._check_fused_scan_remat('autodiff',no_o=True,whole_stage=True,stage_chunk=32,three_stage=True,middle_mode='minor_recompute',save_middle=True,save_native=True)

  def test_three_stage_tiled_parameter_gradients_match_reference(self):
    self._check_fused_scan_remat('autodiff',no_o=True,whole_stage=True,stage_chunk=32,three_stage=True,middle_mode='minor_tiled_grads')

  def test_rankh_chunk_scan_remat_matches_reference(self):
    self._check_fused_scan_remat('autodiff', no_o=True, whole_stage=True,
        stage_chunk=128, three_stage=True, middle_mode='minor_chunk64',
        save_middle=True, write_mode='row4_mxu')

  def test_rankh_recompute_scan_remat_matches_reference(self):
    self._check_fused_scan_remat('autodiff', no_o=True, whole_stage=True,
        stage_chunk=128, three_stage=True, middle_mode='minor_recompute',
        save_middle=True, write_mode='row4_vpu')

  def test_k1_small_residual_scan_remat_matches_reference(self):
    self._check_fused_scan_remat('autodiff', no_o=True, whole_stage=True,
        stage_chunk=128, three_stage=True, middle_mode='minor_recompute',
        save_middle=True, save_attention=True)

  def test_token_minor_carry_scan_remat_matches_reference(self):
    self._check_fused_scan_remat('autodiff', no_o=True, whole_stage=True,
        stage_chunk=128, three_stage=True, middle_mode='minor_recompute',
        save_middle=True, scan_minor=True)

  def _check_fused_scan_remat(self, backward, no_o=False, whole_stage=False, stage_chunk=0,three_stage=False,middle_mode='baseline',save_middle=False,save_native=False,write_mode='original',save_attention=False,scan_minor=False):
    from layers import rmt_pallas_minor, rmt_pallas_minor_read, rmt_pallas_minor_qk, rmt_pallas_v_read, rmt_pallas_write_read
    from layers import rmt_pallas_attention_read, rmt_pallas_full_write_read, rmt_pallas_projected_write
    cfg=self._config('RMTCombinedLayerScanNoHealthProfile')
    cfg.get_keys().update(num_decoder_layers=3,base_num_decoder_layers=3,
                          dtype=jnp.float32,rmt_mlp_dim_by_block=[128]*3,query_chunk_size=2)
    if no_o:cfg.get_keys()['rmt_dynamic_o_enabled']=False
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
                            rmt_pallas_c8=True,rmt_pallas_qk_post=True,rmt_pack_dynamic_projections=True,
                            rmt_pallas_write_backward=backward,rmt_pallas_v_only=no_o,
                            rmt_fused_write_mlp_read=whole_stage,
                            rmt_fused_attention_read=three_stage,
                            rmt_fused_write_read_projection=three_stage,
                            rmt_fused_projected_mlp_write=three_stage,
                            rmt_fused_write_read_backward_tile=stage_chunk,rmt_full_middle_reverse_mode=middle_mode,
                            rmt_save_middle_outputs=save_middle,rmt_save_middle_native_outputs=save_native,
                            rmt_rankh_write_mode=write_mode,rmt_attention_save_small=save_attention,rmt_scan_token_minor=scan_minor)
      with mock.patch.object(rmt_pallas_minor,'write_residual',
                             wraps=partial(rmt_pallas_minor.write_residual,interpret=True)), \
           mock.patch.object(rmt_pallas_minor_read,'c8_read',
                             wraps=partial(rmt_pallas_minor_read.c8_read,interpret=True)), \
           mock.patch.object(rmt_pallas_minor_qk,'qk_post',
                             wraps=partial(rmt_pallas_minor_qk.qk_post,interpret=True)), \
           mock.patch.object(rmt_pallas_v_read,'v_read',
                             wraps=partial(rmt_pallas_v_read.v_read,interpret=True)), \
           mock.patch.object(rmt_pallas_attention_read,'attention_read',
                             wraps=partial(rmt_pallas_attention_read.attention_read,interpret=True)) as attention_mock, \
           mock.patch.object(rmt_pallas_full_write_read,'full_write_read',
                             wraps=partial(rmt_pallas_full_write_read.full_write_read,interpret=True)) as full_mock, \
           mock.patch.object(rmt_pallas_projected_write,'projected_write',
                             wraps=partial(rmt_pallas_projected_write.projected_write,interpret=True)) as write_mock, \
           mock.patch.object(rmt_pallas_write_read,'write_mlp_read',
                             wraps=partial(rmt_pallas_write_read.write_mlp_read,interpret=True)) as stage_mock:
        for policy in ('full','save_state','save_state_mlp','save_state_dynamic'):
          cfg.get_keys()['rmt_remat_policy']=policy
          actual=jax.jit(jax.value_and_grad(loss))(params)
          for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(baseline)):
            self.assertTrue(np.isfinite(np.asarray(a)).all())
            np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=4e-4,atol=3e-6)
        if three_stage:
          self.assertTrue(attention_mock.called and full_mock.called and write_mock.called)
        elif whole_stage:self.assertTrue(stage_mock.called)

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
