"""L14 VM duplication: local budgets, live writes, scan carry and causal gradients."""
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

P = 'BamMediumPropMShiftReadG050Mudd'
VM = P + 'VMSharedRouterLastTwoBlocksL2AnchorTruePile'
MIXED = P + 'VMQKVMSharedRouterLastTwoBlocksL2AnchorTruePile'


class PreTailVMTest(NativeTailTest):
  def tiny(self, experiment=VM):
    cfg = super().tiny(experiment)
    cfg.get_keys()['bam_mudd_pre_tail_vm_mlp_dim'] = 40
    return cfg

  def test_pre_tail_formal_budget_and_paths(self):
    for exp, total, routers, targets, widths, skipped in (
        (VM, 432342692, 231700, 2, (3802, 3676, 3540), []),
        (MIXED, 432342156, 232220, 4, (3802, 3803, 3026), [16])):
      with self.subTest(exp=exp):
        cfg = self.config(exp, bam_splash_attention=False)
        cfg.get_keys().update(max_target_length=256, max_prefill_predict_length=256, query_chunk_size=256)
        self.assertEqual([i for i in range(18) if bam_mudd.is_value_layer(cfg,i)], [14,17])
        self.assertEqual([i for i in range(18) if bam_mudd.is_qk_layer(cfg,i)], [17] if targets==4 else [])
        self.assertEqual([i for i in range(18) if bam_mudd.skips_mlp_write(cfg,i)], skipped)
        self.assertEqual([i for i in range(18) if bam_mudd.skips_attention_write(cfg,i)], [17])
        mesh = jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
        state,*_ = train_compile.get_shaped_inputs(mesh,cfg)
        flat = flatten_dict(nn.unbox(state[0].params))
        self.assertEqual(sum(math.prod(v.shape) for v in flat.values()), total)
        self.assertEqual(sum(math.prod(v.shape) for p,v in flat.items()
            if 'qkvm_compose' in p or 'compose_final_three_source' in p), routers)
        for stage, count, expected_widths in (
            ('layers',4,(3802,3676,3802)),
            ('pre_tail_layers',1,(3802,3676,3412)),
            ('tail_layers',1,widths)):
          for layer,width in zip(('local_0','local_1','fetch_2'),expected_widths):
            path=('params','decoder',stage,layer,'block')
            self.assertEqual(flat[path+('mlp','wi_0','kernel')].shape,(1200,count,width))
        pre=('params','decoder','pre_tail_layers')
        for layer,names in (('local_1',('mlp_address_down','mlp_address_up','mlp_write_gate','mlp_write_gate_bias')),
                            ('fetch_2',('P_loc_down','P_loc_up','W_gw','gw_b0'))):
          block=pre+(layer,'block')
          leaves=[v for p,v in flat.items() if p[:len(block)]==block and any(n in p for n in names)]
          self.assertEqual(sum(math.prod(v.shape) for v in leaves),458000)
        for stage,c,qwidth in (('pre_tail_layers',2,18),('tail_layers',targets,75 if targets==4 else 18)):
          block=('params','decoder',stage,'fetch_2','block')
          self.assertEqual(flat[block+('qkvm_compose','mlp','dynamic_dense_conn2','kernel')].shape,(64,1,c,4))
          for name in ('query','key'):
            self.assertEqual(flat[block+('self_attention',name,'kernel')].shape,(1200,1,16,qwidth))
          self.assertTrue(any(p[:len(block)]==block and 'static_o_key' in p for p in flat))
        self.assertTrue(cfg.record_training_health_metrics)
        self.assertFalse(cfg.record_internal_nn_metrics)
        self.assertEqual(cfg.tensorboard_gcs_rotate_period,500)

  def test_pre_tail_vm_scan_causality_and_gradients(self):
    self.check_pre_tail_model(VM)

  def test_pre_tail_mixed_scan_causality_and_gradients(self):
    self.check_pre_tail_model(MIXED)

  def check_pre_tail_model(self,exp):
    cfg,mesh,model,params,tokens,pos,seg,rng=self.build(exp)
    flat=flatten_dict(params)
    for p,v in flat.items():
      if 'dynamic_dense_conn2' in p:
        flat[p]=v+.01*jax.random.normal(jax.random.key(91),v.shape)
    params=unflatten_dict(flat)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      def forward(p,t=tokens,capture=False):
        return model.apply({'params':p},t,pos,tokens,seg,decoder_segment_ids=seg,
            enable_dropout=False,rngs=rng,
            capture_intermediates=(lambda m,method:isinstance(m,fusion.FusionDecoderLayer)) if capture else False,
            mutable=['intermediates'] if capture else False)
      output,capture=forward(params,capture=True)
      stages=capture['intermediates']['decoder']
      pre=stages['pre_tail_layers']
      states=[pre[n]['__call__'][0][0] for n in ('local_0','local_1','fetch_2')]
      for i in (1,2):
        self.assertGreater(float(jnp.sum((states[i][1]-states[i-1][1])**2)),0.)
      anchor=stages['layers']['fetch_2']['__call__'][0][0][0][0]
      for state in states:
        np.testing.assert_array_equal(state[4][0],anchor)
      for layer,gate in (('local_1','mlp_write_gate_bias'),('fetch_2','gw_b0')):
        altered=dict(flat)
        selected=[p for p in flat if 'pre_tail_layers' in p and layer in p and gate in p]
        self.assertTrue(selected)
        for p in selected:altered[p]=jnp.full_like(flat[p],-100.)
        changed=forward(unflatten_dict(altered))[0]
        self.assertGreater(float(jnp.sum((changed-output[0])**2)),0.,(layer,gate))
      future=tokens.at[:,2:].set(jnp.array([[9,10]],jnp.int32))
      np.testing.assert_allclose(forward(params,future)[0][:,:2],output[0][:,:2],rtol=3e-5,atol=3e-5)
      loss,grad=jax.jit(jax.value_and_grad(lambda p:jnp.mean(forward(p)[0]**2)))(params)
      self.assertTrue(np.isfinite(float(loss)))
      self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
      gradients=flatten_dict(grad)
      for name,layer in (('mlp_address_down','local_1'),('mlp_address_up','local_1'),
                         ('mlp_write_gate','local_1'),('P_loc_down','fetch_2'),
                         ('P_loc_up','fetch_2'),('W_gw','fetch_2')):
        leaves=[v for p,v in gradients.items() if 'pre_tail_layers' in p and layer in p and name in p]
        self.assertTrue(leaves,(layer,name))
        self.assertGreater(sum(float(jnp.sum(v*v)) for v in leaves),0.,(layer,name))
      heads=next(v for p,v in gradients.items() if 'pre_tail_layers' in p and 'qkvm_compose' in p and 'dynamic_dense_conn2' in p)
      for route in range(2):
        self.assertGreater(float(jnp.sum(heads[:,:,route,2]**2)),0.,('L2',route))

  def test_pre_tail_v8t_bridge_keeps_live_attention_write(self):
    self.check_v8t_tail_layout_bridge(((VM,5),(MIXED,5)))


if __name__=='__main__':
  unittest.main()
