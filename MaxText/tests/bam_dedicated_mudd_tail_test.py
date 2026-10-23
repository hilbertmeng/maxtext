"""Dense private M updates combined with proven native Mudd exits."""
import math,unittest
from types import SimpleNamespace
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import max_utils,pyconfig,train_compile
from bam_mudd_native_tail_test import NativeTailTest
from layers import bam_mudd

PREFIX='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1Mudd'
A=PREFIX+'QKVML2TruePile'
B=PREFIX+'VML2TruePile'
C=PREFIX+'L14VML17QKVML2TruePile'

class DedicatedMuddTailTest(NativeTailTest):
  def tiny(self,experiment=A):
    cfg=super().tiny(experiment)
    cfg.get_keys().update(bam_memory_mlp_dim=16,bam_memory_mlp_last_write_dim=0,
        mlp_dim_by_block=[32,32,32],bam_mudd_tail_mlp_dims=[32,40,32],
        bam_mudd_pre_tail_vm_mlp_dim=28)
    if experiment==C:cfg.get_keys()['bam_memory_mlp_dim_by_layer']=[16]*5+[12,16,0,0]
    return cfg

  def test_legacy_native_tail_parameter_regression(self):
    self.check_formal_budget(('BamMediumPropMShiftReadG050MuddQKVMSharedRouterLastBlockL2AnchorTruePile',
                             'BamMediumPropMShiftReadG050MuddVMSharedRouterLastBlockL2AnchorTruePile'))

  def test_formal_parameter_trees_and_write_policies(self):
    for exp,total,routers,normal,mem in [
        (A,432268180,154900,[2535,2535,3802,3030],[1140,1140,0,0]),
        (B,432268716,154380,[2535,2535,2535,3544],[1140,1140,1140,0]),
        (C,432343756,232220,[2275,2535,3802,3030],[1010,1140,0,0])]:
      cfg=self.config(exp,bam_splash_attention=False)
      cfg.get_keys().update(max_target_length=256,max_prefill_predict_length=256,query_chunk_size=256)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
      shaped,*_=train_compile.get_shaped_inputs(mesh,cfg);flat=flatten_dict(nn.unbox(shaped[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),total)
      self.assertEqual(sum(math.prod(v.shape) for p,v in flat.items() if 'qkvm_compose' in p or 'compose_final_three_source' in p),routers)
      ordinary=sum(math.prod(v.shape) for p,v in flat.items() if 'mlp' in p and any(k in p for k in ['wi_0','wi_1','wo']))
      self.assertEqual(ordinary,{'A':170611200,'B':167900400,'C':169675200}[{A:'A',B:'B',C:'C'}[exp]])
      for layer,width,h in zip(range(14,18),normal,mem):
        stage='pre_tail_layers' if exp==C and layer==14 else 'layers' if layer==14 else 'tail_layers'
        local=('local_0','local_1','fetch_2')[layer%3];block=('params','decoder',stage,local,'block')
        self.assertEqual(flat[block+('mlp','wi_0','kernel')].shape[-1],width)
        keys=[p for p in flat if p[:len(block)]==block and 'memory_mlp' in p]
        self.assertEqual(bool(keys),bool(h))
        if h:self.assertEqual(flat[block+('memory_mlp','wi_0','kernel')].shape[-1],h)
      tail=('params','decoder','tail_layers','fetch_2','block')
      self.assertFalse(any(p[:len(tail)]==tail and any(k in p for k in ['P_loc_down','P_loc_up','W_gw','gw_b0','memory_mlp','mlp_write_gate']) for p in flat))
      self.assertTrue(any(p[:len(tail)]==tail and 'static_o_key' in p for p in flat))
      self.assertEqual(flat[tail+('qkvm_compose','mlp','dynamic_dense_conn2','kernel')].shape,(64,1,2 if exp==B else 4,4))
      for key in ['query','key']:
        self.assertEqual(flat[tail+('self_attention',key,'kernel')].shape[-1],18 if exp==B else 75)
      ref=pyconfig.HyperParameters(SimpleNamespace(keys=dict(cfg.get_keys(),bam_pallas_core=False)))
      state,*_=train_compile.get_shaped_inputs(mesh,ref);other=flatten_dict(nn.unbox(state[0].params))
      self.assertEqual({p:v.shape for p,v in flat.items()},{p:v.shape for p,v in other.items()})
      self.assertTrue(cfg.record_training_health_metrics and not cfg.record_internal_nn_metrics)
      self.assertEqual((cfg.bam_pallas_body,cfg.bam_pallas_vmem_mib,cfg.tensorboard_gcs_rotate_period),('v8t',48,500))
      self.assertTrue(cfg.bam_pallas_core and cfg.bam_pallas_fused_inputs and cfg.sa_use_fused_bwd_kernel)

  def check_private_scan(self,exp):
    cfg,mesh,model,params,tokens,pos,seg,rng=self.build(exp)
    flat=flatten_dict(params)
    for p,v in flat.items():
      if 'dynamic_dense_conn2' in p:flat[p]=v+.01*jax.random.normal(jax.random.key(91),v.shape)
    params=unflatten_dict(flat)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def forward(p,t=tokens):return model.apply({'params':p},t,pos,tokens,seg,decoder_segment_ids=seg,enable_dropout=False,rngs=rng)[0]
      out=forward(params);future=tokens.at[:,2:].set(jnp.array([[9,10]],jnp.int32))
      np.testing.assert_allclose(forward(params,future)[:,:2],out[:,:2],rtol=3e-5,atol=3e-5)
      loss,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(forward(p)**2)))(params)
      self.assertTrue(np.isfinite(float(loss)))
      gf=flatten_dict(grad);self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in gf.values()))
      # Private nonlinear writes and native anchor mixing must both affect the loss.
      for branch in ['memory_mlp','mlp_address_down','mlp_address_up','mlp_write_gate','qkvm_compose','compose_final_three_source']:
        leaves=[v for p,v in gf.items() if branch in p]
        self.assertTrue(leaves,branch);self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves),0.,branch)
      if exp==B:
        leaves=[v for p,v in gf.items() if 'tail_layers' in p and 'local_1' in p and 'memory_mlp' in p]
        self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves),0.)
      if exp==C:
        leaves=[v for p,v in gf.items() if 'pre_tail_layers' in p and 'fetch_2' in p and 'memory_mlp' in p]
        self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves),0.)
      for stage in ['tail_layers']+(['pre_tail_layers'] if exp==C else []):
        heads=next(v for p,v in gf.items() if stage in p and 'qkvm_compose' in p and 'dynamic_dense_conn2' in p)
        for route in range(heads.shape[-2]):self.assertGreater(float(jnp.sum(heads[:,:,route,2]**2)),0.,(stage,route,'L2'))

  def test_qkvm_private_scan_causality_and_gradients(self):self.check_private_scan(A)
  def test_vm_private_scan_causality_and_gradients(self):self.check_private_scan(B)
  def test_mixed_private_scan_causality_and_gradients(self):self.check_private_scan(C)
  def test_selective_v8t_bridge_including_private_pre_tail_writer(self):
    self.check_v8t_tail_layout_bridge(((A,8),(B,8),(C,5)))

if __name__=='__main__':unittest.main()
