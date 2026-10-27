"""VM tails on independent VO/full NoPE/strict-history attention."""
import math
import unittest
from types import SimpleNamespace
from unittest import mock
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict
import max_utils
import pyconfig
import train_compile
from bam_prop_test_utils import MLPWriteTest
from layers import attentions, bam_mudd, fusion
from layers.models import Transformer

BASE='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75'
PARENT=BASE+'TruePile'
A=BASE+'MuddL14L17VML2TruePile'
B=BASE+'MuddL15L16L17VML2TruePile'
AR=BASE+'MuddL14L17VML2RestoreSelfTruePile'
BR=BASE+'MuddL15L16L17VML2RestoreSelfTruePile'

class StrongVMTailTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,experiment=A):
    cfg=self.config(experiment,bam_splash_attention=False,base_emb_dim=150,
        base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=9,
        base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    cfg.get_keys().update(bam_pallas_core=False,emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8,
        mlp_dim_by_block=[32]*3,bam_mudd_tail_mlp_dims=[32,32,36],
        bam_mudd_pre_tail_vm_mlp_dim=28,bam_memory_mlp_dim=16,
        bam_memory_mlp_last_write_dim=0,bam_attention_write_layers=list(range(8)),
        bam_layer_modes=['local_qk+local_v+local_o']*9)
    if experiment in (A,AR):cfg.get_keys()['bam_memory_mlp_dim_by_layer']=[16]*5+[12,16,16,0]
    else:
      cfg.get_keys().update(bam_mudd_tail_mlp_dims=[28,28,36],
          bam_memory_mlp_dim_by_layer=[16]*6+[12,12,0])
    return cfg

  def test_formal_budgets_selected_parameters_and_live_writers(self):
    for name,total,routers,sites in [(A,432343636,231700,[14,17]),
        (AR,432343636,231700,[14,17]),(B,432420828,309020,[15,16,17]),
        (BR,432420828,309020,[15,16,17])]:
      with self.subTest(name=name):
        cfg=self.config(name,bam_splash_attention=False)
        cfg.get_keys().update(max_target_length=256,max_prefill_predict_length=256,query_chunk_size=256)
        self.assertEqual([i for i in range(18) if bam_mudd.is_value_layer(cfg,i)],sites)
        self.assertFalse(any(bam_mudd.is_qk_layer(cfg,i) for i in range(18)))
        mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
        state,*_=train_compile.get_shaped_inputs(mesh,cfg);flat=flatten_dict(nn.unbox(state[0].params))
        count=sum(math.prod(v.shape) for v in flat.values());router_count=sum(math.prod(v.shape) for p,v in flat.items() if 'qkvm_compose' in p or 'compose_final_three_source' in p)
        print('FORMAL_BUDGET',name,count,router_count,flush=True)
        self.assertEqual((count,router_count),(total,routers))
        for layer in range(12,18):
          stage='pre_tail_layers' if layer<15 else 'tail_layers';local=('local_0','local_1','fetch_2')[layer%3];block=('params','decoder',stage,local,'block');att=block+('self_attention',)
          self.assertEqual(any(p[:len(block)]==block and 'qkvm_compose' in p for p in flat),layer in sites)
          width=3551 if layer==17 else 2275 if layer in sites else 2510
          self.assertEqual(flat[block+('mlp','wi_0','kernel')].shape[-1],width)
          memory=[p for p in flat if p[:len(block)]==block and 'memory_mlp' in p]
          self.assertEqual(bool(memory),layer<17)
          if memory:self.assertEqual(flat[block+('memory_mlp','wi_0','kernel')].shape[-1],1010 if layer in sites else 1128)
          for param in ('P_loc_up','W_gw','mlp_address_up','mlp_write_gate'):
            self.assertEqual(any(p[:len(block)]==block and param in p for p in flat),layer<17,(layer,param))
          self.assertTrue(any(p[:len(att)]==att and 'static_o_key' in p for p in flat))
          self.assertTrue(any(p[:len(att)]==att and 'W_R_up' in p for p in flat))
          self.assertEqual(flat[att+('query','kernel')].shape[-1],18)
          self.assertEqual(any(p[:len(att)]==att and 'W_lv_c8_up' in p for p in flat),layer not in sites)
          self.assertEqual(any(p[:len(att)]==att and 'static_v_key' in p for p in flat),layer not in sites)
          if layer in sites:
            self.assertEqual(flat[att+('value','kernel')].shape[-1],75)
            self.assertEqual(flat[block+('qkvm_compose','mlp','dynamic_dense_conn2','kernel')].shape,(64,1,2,4))
        self.assertTrue(cfg.bam_pallas_core and cfg.sa_use_fused_bwd_kernel and cfg.record_training_health_metrics)
        self.assertFalse(cfg.record_internal_nn_metrics)
        self.assertEqual((cfg.bam_partial_rope_nope_dim,cfg.bam_local_qk_col_output_dim),(75,75))

  def fixture_layer(self,experiment,index,accelerated=False):
    cfg=self.tiny(experiment)
    if accelerated:cfg.get_keys().update(bam_pallas_core=True,bam_pallas_body='v8t')
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    module=fusion.SubDecoderLayer(cfg,mesh,layer_inx=index)
    x=jax.random.normal(jax.random.key(31),(1,4,150))
    matrix=.2*jax.random.normal(jax.random.key(32),(1,4,75,32))
    anchors=tuple(jax.random.normal(jax.random.key(40+i),x.shape) for i in range(3))
    tokens=jnp.ones((1,4),jnp.int32)
    kwargs=dict(decoder_segment_ids=tokens,decoder_positions=jnp.arange(4)[None],
        decoder_input_tokens=tokens,deep_embedding=None,deterministic=True,
        model_mode='train',eos_sum=None,layer_index=index,early_residuals=anchors)
    m=jnp.swapaxes(matrix,-1,-2) if accelerated else matrix
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      params=nn.unbox(module.init(jax.random.key(33),x,M_in=m,**kwargs)['params'])
    return cfg,mesh,module,params,x,m,kwargs

  def test_vm_only_diagonal_restore_full_qk_width_and_identical_parameters(self):
    for strict,restore,index,selected in [(A,AR,5,True),(A,AR,4,False),(A,AR,8,True),
        (B,BR,6,True),(B,BR,7,True),(B,BR,8,True),(B,BR,5,False)]:
      with self.subTest(experiment=strict,index=index):
        a=self.fixture_layer(strict,index);b=self.fixture_layer(restore,index)
        cfg,mesh,module,params,x,m,kw=a
        for av,bv in zip(jax.tree.leaves(params),jax.tree.leaves(b[3])):
          np.testing.assert_array_equal(av,bv)
        original=attentions._attention_op;seen=[]
        def observe(query,key,value,*args,**kwargs):
          result=original(query,key,value,*args,**kwargs)
          seen.append((query.shape[-1],value.shape[-1],kwargs.get('negative_diagonal_mask') is not None,result[1]))
          return result
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules),mock.patch.object(attentions,'_attention_op',observe):
          out1,_=module.apply({'params':params},x,M_in=m,**kw)
          split_at=len(seen)
          out2,_=b[2].apply({'params':params},x,M_in=m,**kw)
        self.assertTrue(seen)
        first,second=seen[0],seen[split_at]
        self.assertEqual(first[:3],(93,75,True))
        self.assertEqual(second[:3],(93,75,not selected))
        np.testing.assert_array_equal(first[3][:,:,0],0.)
        if selected:
          np.testing.assert_array_equal(second[3][:,:,0,0],1.)
          self.assertGreater(float(jnp.max(jnp.abs(out1-out2))),0.)
        else:np.testing.assert_array_equal(out1,out2)

  def test_vm_v8t_layout_bridge_all_gradients_and_live_writes(self):
    for experiment,index in [(A,5),(A,8),(B,6),(B,7),(B,8),(AR,5),(BR,7)]:
      with self.subTest(experiment=experiment,index=index):
        cfg,mesh,module,p,x,m,kw=self.fixture_layer(experiment,index)
        _,_,bridge,bp,*_=self.fixture_layer(experiment,index,True)
        self.assertEqual(set(flatten_dict(p)),set(flatten_dict(bp)))
        for a,b in zip(jax.tree.leaves(p),jax.tree.leaves(bp)):np.testing.assert_array_equal(a,b)
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
          out,nm=module.apply({'params':p},x,M_in=m,**kw)
          actual,am=bridge.apply({'params':p},x,M_in=jnp.swapaxes(m,-1,-2),**kw)
          np.testing.assert_array_equal(out,actual)
          np.testing.assert_array_equal(nm,jnp.swapaxes(am,-1,-2))
          if index<8:self.assertGreater(float(jnp.max(jnp.abs(nm-m))),0.)
          else:np.testing.assert_array_equal(nm,m)
          def objective(mod,params,x,m,transpose):
            y,next_m=mod.apply({'params':params},x,M_in=jnp.swapaxes(m,-1,-2) if transpose else m,**kw)
            return jnp.mean(y*y)+(.01*jnp.mean(next_m*next_m) if index<8 else 0.)
          reference=jax.jit(jax.grad(lambda p,x,m:objective(module,p,x,m,False),argnums=(0,1,2)))(p,x,m)
          gradients=jax.jit(jax.grad(lambda p,x,m:objective(bridge,p,x,m,True),argnums=(0,1,2)))(p,x,m)
          for a,b in zip(jax.tree.leaves(reference),jax.tree.leaves(gradients)):
            np.testing.assert_allclose(a,b,rtol=2e-6,atol=2e-6)
          flat=flatten_dict(reference[0]);self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in flat.values()))
          for branch in ['static_o_key','W_R_up','value','qkvm_compose']+(['memory_mlp','P_loc_up','mlp_address_up'] if index<8 else []):
            values=[v for path,v in flat.items() if branch in path]
            self.assertTrue(values,branch);self.assertGreater(sum(float(jnp.sum(v*v)) for v in values),0.,branch)

  def test_three_vm_model_anchor_routes_causality_and_prefix_parameters(self):
    models=[]
    for experiment in (A,B,AR,BR):
      cfg=self.tiny(experiment);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
      model=Transformer(cfg,mesh,quant=None);tokens=jnp.asarray([[1,4,8,2]],jnp.int32);pos=jnp.arange(4)[None];segments=jnp.ones_like(tokens)
      rng={name:jax.random.key(i+21) for i,name in enumerate(('params','dropout','aqt'))}
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        params=nn.unbox(model.init(rng,tokens,pos,tokens,segments,decoder_segment_ids=segments,enable_dropout=False)['params'])
      models.append((cfg,mesh,model,params,tokens,pos,segments,rng))
    for index in (0,1):
      for a,b in zip(jax.tree.leaves(models[index][3]),jax.tree.leaves(models[index+2][3])):np.testing.assert_array_equal(a,b)
    prefixes=[{p:v for p,v in flatten_dict(m[3]).items() if p[:2]==('decoder','layers')} for m in models[:2]]
    self.assertTrue(prefixes[0])
    self.assertEqual(set(prefixes[0]),set(prefixes[1]))
    for p in prefixes[0]:np.testing.assert_array_equal(prefixes[0][p],prefixes[1][p])
    for cfg,mesh,model,params,tokens,pos,seg,rng in models:
      flat=flatten_dict(params)
      for path,value in flat.items():
        if 'dynamic_dense_conn2' in path:flat[path]=value+.01*jax.random.normal(jax.random.key(91),value.shape)
      params=unflatten_dict(flat)
      with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
        def forward(p,t=tokens):return model.apply({'params':p},t,pos,t,seg,decoder_segment_ids=seg,enable_dropout=False,rngs=rng)[0]
        result=forward(params);future=tokens.at[:,2:].set(jnp.asarray([[9,10]],jnp.int32))
        np.testing.assert_allclose(forward(params,future)[:,:2],result[:,:2],rtol=3e-5,atol=3e-5)
        grads=flatten_dict(jax.jit(jax.grad(lambda p:jnp.mean(forward(p)**2)))(params))
        self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in grads.values()))
        for stage in ['tail_layers','compose_final_three_source']+(['pre_tail_layers'] if cfg.bam_mudd_pre_tail_vm else []):
          values=[v for p,v in grads.items() if stage in p and 'dynamic_dense_conn2' in p]
          self.assertTrue(values,stage)
          self.assertGreater(sum(float(jnp.sum(v[...,2]**2)) for v in values),0.,(stage,'L2'))

if __name__=='__main__':unittest.main()
