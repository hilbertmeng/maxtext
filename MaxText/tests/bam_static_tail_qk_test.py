"""Remove upper dynamic keys, keep static reads alive and match the native kernel."""
import math
from types import SimpleNamespace
import unittest
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel as splash
import max_utils,pyconfig,train_compile
from bam_config import validate_bam_config
from bam_prop_test_utils import MLPWriteTest
from layers.attentions import BamAttention
from layers.models import Transformer

PREFIX='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75CommonReadScaleUnscaledMLPContentEmbeddingFixed'
H=PREFIX+'TruePile';QK=PREFIX+'StaticQKLast4TruePile';K=PREFIX+'StaticKLast4TruePile'

class StaticTailQKTest(MLPWriteTest,unittest.TestCase):
  def test_formal_budget_scan_boundary_and_pruned_parameter_tree(self):
    for run,expected,tail,last,arms in [(H,432112210,None,3904,''),(QK,432111810,2594,3987,'qk'),(K,432113810,2552,3946,'k')]:
      c=self.config(run);c.get_keys().update(max_target_length=128,max_prefill_predict_length=128)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
      self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),expected)
      self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_final_local_mlp_dim,c.bam_last_mlp_dim),(2510,1128,tail,last))
      self.assertEqual(c.bam_final_local_layer_count,4 if arms else 1)
      self.assertTrue(c.scan_layers and not c.bam_pair_scan)
      self.assertTrue(c.bam_pallas_core and c.bam_pallas_write and c.sa_use_fused_bwd_kernel)
      self.assertTrue(c.record_training_health_metrics and not c.record_internal_nn_metrics)
      self.assertEqual(c.tensorboard_gcs_rotate_period,500)
      params=nn.unbox(args[0].params)['params']['decoder']
      if arms:
        body=params['layers']['block'];self.assertEqual(body['mlp']['wi_0']['kernel'].shape[c.param_scan_axis],14)
        for i in range(4):
          block=params[f'final_local_layer_{i}']['block'];att=block['self_attention']
          for a in 'qk':
            self.assertEqual('W_l'+a+'_c8_up' in att,a not in arms)
            self.assertEqual('W_l'+a+'_gate' in att,a not in arms)
            self.assertEqual('general_'+a+'_pre_bias' in att,a not in arms)
            self.assertIn('static_'+a+'_key',att);self.assertIn('general_'+a+'_static_gate',att)
          self.assertIn('full_m_read_down',att);self.assertIn('W_R_up',att);self.assertIn('W_lv_c8_up',att)
          self.assertEqual(block['mlp']['wi_0']['kernel'].shape[-1],last if i==3 else tail)
          self.assertEqual('memory_mlp' in block,i<3)
          self.assertEqual('P_loc_down' in att,i<3)
      print('STATIC_TAIL_FORMAL_OK',run,expected,flush=True)

  def fixture(self,run=QK,layer=14):
    c=self.config(run,base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,bam_pallas_write=False,bam_splash_attention=False,bam_local_full_m_read_bottleneck_dim=8,bam_write_v_bottleneck_dim=8)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    mod=BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,read_side='col',layer_mode='local_qk+local_v+local_o',layer_inx=layer,max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=c.dtype)
    x=jax.random.normal(jax.random.key(1),(1,4,150));m=jax.random.normal(jax.random.key(2),(1,4,75,32));args=(x,x,jnp.arange(4)[None],jnp.ones((1,4),jnp.int32))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):p=nn.unbox(mod.init(jax.random.key(3),*args,M_in=m))
    return c,mesh,mod,p,args,m

  def test_static_only_reference_gradients_and_nonzero_start(self):
    for run,arms in [(QK,'qk'),(K,'k')]:
      c,mesh,mod,p,args,m=self.fixture(run)
      _,_,parent,pp,_,_=self.fixture(H)
      self.assertGreater(float(jnp.std(p['params']['static_k_key'])),.002)
      np.testing.assert_array_equal(p['params']['static_q_key'],jnp.zeros((32,2)))
      parent_flat=flatten_dict(pp);child_flat=flatten_dict(p)
      def off_gates(next_fun,aa,kw,ctx):
        z=next_fun(*aa,**kw)
        if ctx.method_name=='_project_read_gate_logits' and aa[0] in ['W_l'+a+'_gate' for a in arms]:return jnp.full_like(z,-1000)
        return z
      target=jax.random.normal(jax.random.key(4),(1,4,150));mt=jax.random.normal(jax.random.key(5),m.shape)
      def call(model,q,m,x):return model.apply(q,x,x,args[2],args[3],M_in=m,deterministic=True)
      def loss(q,m,x,reference=False):
        if reference:
          flat=dict(parent_flat);flat.update(flatten_dict(q));q=unflatten_dict(flat)
          with nn.intercept_methods(off_gates):y,mm=call(parent,q,m,x)
        else:y,mm=call(mod,q,m,x)
        return jnp.mean(y*target)+jnp.mean(mm*mt),(y,mm)
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        (_,a),ga=jax.value_and_grad(loss,argnums=(0,1,2),has_aux=True)(p,m,args[0])
        (_,b),gb=jax.value_and_grad(lambda p,m,x:loss(p,m,x,True),argnums=(0,1,2),has_aux=True)(p,m,args[0])
        for x,y in zip(jax.tree.leaves((a,ga)),jax.tree.leaves((b,gb))):np.testing.assert_allclose(x,y,rtol=5e-4,atol=2e-6)
        if run==QK:
          self.assertGreater(float(jnp.linalg.norm(ga[0]['params']['static_q_key'])),1e-10)
          updated=jax.tree.map(lambda v,g:v-.01*g,p,ga[0])
          gg=jax.grad(lambda q:loss(q,m,args[0])[0])(updated)
          self.assertGreater(float(jnp.linalg.norm(gg['params']['static_k_key'])),1e-10)
      # L13 belongs to the unchanged prefix, including zero static K init.
      _,_,_,before,_,_=self.fixture(run,layer=13)
      for a in 'qk':self.assertIn('W_l'+a+'_c8_up',before['params'])
      np.testing.assert_array_equal(before['params']['static_k_key'],jnp.zeros((32,2)))

  def test_scanned_model_native_pallas_logits_and_all_gradients(self):
    for run in [QK,K]:
      c=self.config(run,base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32',bam_qk_static_only_last_layers=2)
      c.get_keys().update(bam_pallas_core=False,bam_pallas_write=False,bam_splash_attention=False,bam_memory_mlp_dim=16,bam_last_mlp_dim=32,bam_final_local_mlp_dim=32,bam_final_local_layer_count=2,bam_layer_modes=['local_qk+local_v+local_o']*3,bam_attention_write_layers=[0,1],bam_mlp_write_layers=[0,1],emb_bam_num_head=2,bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8,max_target_length=128,max_prefill_predict_length=128,query_chunk_size=64)
      pc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_pallas_core=True,bam_pallas_write=True,bam_pallas_interpret=True,bam_pallas_fused_inputs=True,bam_splash_attention=True)))
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);native,fused=Transformer(c,mesh,quant=None),Transformer(pc,mesh,quant=None)
      tok=jax.random.randint(jax.random.key(11),(1,128),0,32);pos=(jnp.arange(128)%64)[None];seg=jnp.where(jnp.arange(128)<64,1,2)[None]
      original=splash.make_splash_mha
      def interpret(*a,**kw):return original(*a,**dict(kw,interpret=True))
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules),mock.patch.object(splash,'make_splash_mha',side_effect=interpret):
        p,pp=(nn.unbox(model.init(jax.random.key(12),tok,pos,seg,tok)) for model in [native,fused])
        self.assertEqual(jax.tree.structure(p),jax.tree.structure(pp))
        for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(pp)):np.testing.assert_array_equal(a,b)
        def loss(model,p):
          y=model.apply(p,tok,pos,seg,tok)[0];return jnp.mean(y*y),y
        (_,a),ga=jax.value_and_grad(lambda p:loss(native,p),has_aux=True)(p)
        (_,b),gb=jax.value_and_grad(lambda p:loss(fused,p),has_aux=True)(p)
        np.testing.assert_allclose(a,b,rtol=4e-4,atol=3e-6)
        for key,a in flatten_dict(ga).items():np.testing.assert_allclose(a,flatten_dict(gb)[key],rtol=4e-3,atol=5e-6,err_msg=str(key))
        self.assertGreater(float(jnp.linalg.norm(ga['params']['decoder']['final_local_layer_1']['block']['self_attention']['static_q_key'])),1e-12)
      print('STATIC_TAIL_SCAN_NATIVE_PALLAS_ALL_GRADIENTS_OK',run,flush=True)

  def test_validation_prevents_mixed_scan_and_invalid_architectures(self):
    c=self.config(QK);d=dict(c.get_keys());validate_bam_config(d)
    for updates in [dict(bam_qk_static_only_last_layers=-1),dict(bam_qk_static_only_arms='v'),dict(bam_final_local_layer_count=1),dict(bam_concat_static_qk=False)]:
      with self.assertRaises(ValueError):validate_bam_config(dict(d,**updates))

if __name__=='__main__':unittest.main()
