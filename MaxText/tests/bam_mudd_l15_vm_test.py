"""Position-matched L15 VM duplication and its live M write."""
import math
import unittest
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import jax
import jax.numpy as jnp
import numpy as np
from bam_mudd_native_tail_test import NativeTailTest
from layers import bam_mudd, fusion
import max_utils
import train_compile

P='BamMediumPropMShiftReadG050Mudd'
VM=P+'VMSharedRouterL15L17L2AnchorTruePile'
MIXED=P+'VMQKVMSharedRouterL15L17L2AnchorTruePile'


class L15VMTest(NativeTailTest):
  def tiny(self,experiment=VM):
    cfg=super().tiny(experiment)
    cfg.get_keys()['bam_mudd_pre_tail_vm_mlp_dim']=40
    return cfg

  def test_l15_formal_budget_and_write_policies(self):
    for exp,total,routers,targets,middle,last in (
        (VM,432342692,231700,2,3676,3540),
        (MIXED,432342156,232220,4,3803,3026)):
      with self.subTest(exp=exp):
        cfg=self.config(exp,bam_splash_attention=False)
        cfg.get_keys().update(max_target_length=256,max_prefill_predict_length=256,query_chunk_size=256)
        self.assertEqual([i for i in range(18) if bam_mudd.is_value_layer(cfg,i)],[15,17])
        self.assertEqual([i for i in range(18) if bam_mudd.is_qk_layer(cfg,i)],[17] if targets==4 else [])
        self.assertEqual([i for i in range(18) if bam_mudd.skips_mlp_write(cfg,i)],[16] if targets==4 else [])
        self.assertEqual([i for i in range(18) if bam_mudd.skips_attention_write(cfg,i)],[17])
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
        state,*_=train_compile.get_shaped_inputs(mesh,cfg)
        flat=flatten_dict(nn.unbox(state[0].params))
        self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),total)
        self.assertEqual(sum(math.prod(v.shape) for p,v in flat.items()
            if 'qkvm_compose' in p or 'compose_final_three_source' in p),routers)
        self.assertFalse(any('pre_tail_layers' in p for p in flat))
        self.assertEqual(flat[('params','decoder','layers','local_0','block','mlp','wi_0','kernel')].shape,(1200,5,3802))
        prefix=('params','decoder','tail_layers')
        for name,width in zip(('local_0','local_1','fetch_2'),(3412,middle,last)):
          self.assertEqual(flat[prefix+(name,'block','mlp','wi_0','kernel')].shape,(1200,1,width))
        for layer,c,qwidth,writing in (('local_0',2,18,True),('fetch_2',targets,75 if targets==4 else 18,False)):
          block=prefix+(layer,'block')
          self.assertEqual(flat[block+('qkvm_compose','mlp','dynamic_dense_conn2','kernel')].shape,(64,1,c,4))
          for name in ('query','key'):
            self.assertEqual(flat[block+('self_attention',name,'kernel')].shape,(1200,1,16,qwidth))
          for name in ('P_loc_down','P_loc_up','W_gw','gw_b0'):
            self.assertEqual(any(p[:len(block)]==block and name in p for p in flat),writing)
          self.assertTrue(any(p[:len(block)]==block and 'static_o_key' in p for p in flat))
        middle_path=prefix+('local_1','block')
        self.assertEqual(any(p[:len(middle_path)]==middle_path and 'mlp_address_down' in p for p in flat),targets==2)

  def test_l15_vm_forward_causality_and_live_write_gradients(self):
    self.check_l15_model(VM)

  def test_l15_mixed_forward_causality_and_live_write_gradients(self):
    self.check_l15_model(MIXED)

  def check_l15_model(self,exp):
    cfg,mesh,model,params,tokens,pos,seg,rng=self.build(exp)
    flat=flatten_dict(params)
    for p,v in flat.items():
      if 'dynamic_dense_conn2' in p:flat[p]=v+.01*jax.random.normal(jax.random.key(113),v.shape)
    params=unflatten_dict(flat)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def forward(p,t=tokens,capture=False):
        return model.apply({'params':p},t,pos,tokens,seg,decoder_segment_ids=seg,enable_dropout=False,rngs=rng,
            capture_intermediates=(lambda m,method:isinstance(m,fusion.FusionDecoderLayer)) if capture else False,
            mutable=['intermediates'] if capture else False)
      output,capture=forward(params,capture=True)
      layers=capture['intermediates']['decoder']
      before=layers['layers']['fetch_2']['__call__'][0][0]
      after=layers['tail_layers']['local_0']['__call__'][0][0]
      self.assertGreater(float(jnp.sum((after[1][0]-before[1][-1])**2)),0.)
      np.testing.assert_array_equal(after[4][0],before[4][-1])
      altered=dict(flat)
      selected=[p for p in flat if 'tail_layers' in p and 'local_0' in p and 'gw_b0' in p]
      self.assertTrue(selected)
      for p in selected:altered[p]=jnp.full_like(flat[p],-100.)
      changed=forward(unflatten_dict(altered))[0]
      self.assertGreater(float(jnp.sum((changed-output[0])**2)),0.)
      future=tokens.at[:,2:].set(jnp.array([[9,10]],jnp.int32))
      np.testing.assert_allclose(forward(params,future)[0][:,:2],output[0][:,:2],rtol=3e-5,atol=3e-5)
      loss,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(forward(p)[0]**2)))(params)
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      gradient=flatten_dict(grad)
      for name in ('P_loc_down','P_loc_up','W_gw','gw_b0'):
        values=[v for p,v in gradient.items() if 'tail_layers' in p and 'local_0' in p and name in p]
        self.assertTrue(values,name)
        self.assertGreater(sum(float(jnp.sum(v*v)) for v in values),0.,name)
      heads=next(v for p,v in gradient.items() if 'tail_layers' in p and 'local_0' in p and 'dynamic_dense_conn2' in p)
      for route in range(2):self.assertGreater(float(jnp.sum(heads[:,:,route,2]**2)),0.,('L2',route))

  def test_l15_v8t_bridge_keeps_live_attention_write(self):
    self.check_v8t_tail_layout_bridge(((VM,6),(MIXED,6)))


if __name__=='__main__':
  unittest.main()
