"""Post-RMS scales on all three writers: semantics, exact budget, gradients and concise health."""
import math
from types import SimpleNamespace
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import exp,max_utils,pyconfig,train,train_compile
from bam_prop_test_utils import MLPWriteTest
from bam_standard_value_test import StandardValueTest
from layers import bam_write
from layers.models import Transformer

PARENT='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75LocalODynamicNormScale050TruePile'
RUN=PARENT.removesuffix('TruePile')+'WriteContentNormScaleTruePile'

class WriteContentScaleTest(MLPWriteTest,unittest.TestCase):
  def tiny(self,period=3,keep=True,length=4):
    c=self.config(RUN,bam_splash_attention=False,base_emb_dim=150,
        base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=3,
        base_mlp_dim=32,vocab_size=32,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_pallas_core=False,max_target_length=length,
        max_prefill_predict_length=length,query_chunk_size=length,
        bam_layer_modes=['local_qk+local_v+local_o']*3,
        bam_attention_write_layers=[0,1],bam_mlp_write_layers=[0,1],
        bam_memory_mlp_dim=16,bam_last_mlp_dim=32,emb_bam_num_head=2,
        bam_write_v_bottleneck_dim=8,emb_bam_v_bottleneck_dim=8,
        bam_mlp_write_address_rank=8,bam_local_full_m_read_bottleneck_dim=8)
    return c

  def test_actual_budget_35_independent_scale_groups_and_no_dead_tail(self):
    c=self.config(RUN);c.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,*_=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(nn.unbox(args[0].params))
    scales={p:v for p,v in flat.items() if p[-1] in ('write_content_norm_scale','mlp_write_content_norm_scale')}
    self.assertEqual(sum(math.prod(v.shape) for v in scales.values()),42000)
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432163408)
    self.assertEqual((c.mlp_dim,c.bam_memory_mlp_dim,c.bam_last_mlp_dim),(2510,1128,3904))
    self.assertEqual(c.bam_write_content_norm_scale_init,1.)
    self.assertTrue(c.record_training_health_metrics)
    self.assertNotIn('bam_record_write_content_scale_health',c.get_keys())
    self.assertFalse(c.record_internal_nn_metrics or c.bam_record_concat_health or c.bam_record_m_shift_health)
    self.assertTrue(c.bam_pallas_core and c.bam_pallas_fused_inputs)
    self.assertTrue(c.bam_pallas_write and c.bam_pallas_body=='v8t')
    self.assertTrue(c.sa_use_fused_bwd_kernel)
    self.assertEqual(c.tensorboard_gcs_rotate_period,500)
    self.assertFalse(any('final_local_layer' in p for p in scales))
    self.assertTrue(any('embedding_bam_write' in p for p in scales))
    self.assertEqual(c.compare_runs,[PARENT])
    print('ALL_WRITES_SCALE_BUDGET_OK432163408_DELTA42000',flush=True)

  def test_native_group_formula_all_gradients_and_post_norm_placement(self):
    keys=iter(jax.random.split(jax.random.key(1),12));rand=lambda shape:jax.random.normal(next(keys),shape)
    m=rand((1,4,3,5));u=rand((1,4,2,5));z=rand((1,4,2));p=rand((1,4,2,3));gamma=jnp.linspace(-.2,1.7,10).reshape(2,5)
    target=rand(m.shape)
    def actual(m,u,z,p,g):return bam_write.native_write(m,[bam_write.ScaledWrite(u,z,p,g)],epsilon=1e-5)
    def norm(x):return x*jax.lax.rsqrt(jnp.mean(x*x,axis=-1,keepdims=True)+1e-5)
    def ref(m,u,z,p,g):return m+jnp.einsum('btnv,btnk->btvk',norm(p),jax.nn.sigmoid(z)[...,None]*(norm(u)*g))
    def loss(fn):return lambda *xs:jnp.sum(fn(*xs)*target)
    a,ga=jax.value_and_grad(loss(actual),argnums=(0,1,2,3,4))(m,u,z,p,gamma)
    b,gb=jax.value_and_grad(loss(ref),argnums=(0,1,2,3,4))(m,u,z,p,gamma)
    np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
    for x,y in zip(ga,gb):np.testing.assert_allclose(x,y,rtol=2e-5,atol=3e-6)
    np.testing.assert_array_equal(actual(m,u,z,p,jnp.zeros_like(gamma)),m)
    self.assertGreater(float(jnp.max(jnp.abs(norm(u)*gamma-norm(u*gamma)))),.1)

  def test_v8t_scaled_write_kernel_matches_native_all_gradients(self):
    from layers import bam_pallas_v8
    k=jax.random.split(jax.random.key(30),9);b,t,n,kd,v=2,256,3,16,8
    args=(jax.random.normal(k[0],(b,t,v,kd)),jax.random.normal(k[1],(b,t,n,kd)),jax.random.normal(k[2],(b,t,n)),
          jax.random.normal(k[3],(b,t,n,v)),1+.5*jax.random.normal(k[4],(n,kd)),
          jax.random.normal(k[5],(b,t,2,kd)),jax.random.normal(k[6],(b,t,2)),jax.random.normal(k[7],(b,t,2,v)))
    target=jax.random.normal(k[8],args[0].shape)
    groups=lambda x:[bam_write.ScaledWrite(*x[:4]),tuple(x[4:])]
    native=lambda m,*x:bam_write.native_write(m,groups(x),epsilon=1e-6)
    kernel=lambda m,*x:bam_pallas_v8.write_tm(m,groups(x),epsilon=1e-6,interpret=True)
    np.testing.assert_allclose(kernel(*args),native(*args),rtol=2e-5,atol=2e-5)
    gn=jax.grad(lambda *x:jnp.sum(native(*x)*target),argnums=tuple(range(8)))(*args)
    gk=jax.grad(lambda *x:jnp.sum(kernel(*x)*target),argnums=tuple(range(8)))(*args)
    worst=max(float(jnp.max(jnp.abs(l-r))/jnp.max(jnp.abs(l))) for l,r in zip(gn,gk))
    print('CONTENT_SCALE_V8T_KERNEL_WORST_GRAD',worst,flush=True);self.assertLess(worst,1e-5)

  def test_full_model_pallas_read_and_write_all_gradients(self):
    tiny=self.tiny
    def with_write(*a,**k):
      c=tiny(*a,**k);c.get_keys().update(bam_pallas_write=True);return c
    self.tiny=with_write
    StandardValueTest.check_pallas(self,True)

  def test_init_one_preserves_parent_function_and_all_old_parameters(self):
    c=self.tiny();oldc=pyconfig.HyperParameters(SimpleNamespace(keys=dict(c.get_keys(),bam_write_content_norm_scale=False)))
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    model=Transformer(c,mesh,quant=None);old=Transformer(oldc,mesh,quant=None)
    tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];seg=jnp.ones_like(tok)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=nn.unbox(model.init(jax.random.key(7),tok,pos,seg,tok));q=nn.unbox(old.init(jax.random.key(7),tok,pos,seg,tok))
      a,b=flatten_dict(p),flatten_dict(q)
      self.assertEqual(len(set(a)-set(b)),3) # two stacked scan leaves plus embedding
      for key,value in b.items():np.testing.assert_array_equal(a[key],value,err_msg=str(key))
      for key in set(a)-set(b):np.testing.assert_array_equal(a[key],jnp.ones_like(a[key]))
      call=lambda model,p:model.apply(p,tok,pos,seg,tok,enable_dropout=False)[0]
      np.testing.assert_array_equal(call(model,p),call(old,q))
      g=jax.grad(lambda pp:jnp.mean(call(model,pp)**2))(p)
    added=[(k,v) for k,v in flatten_dict(g).items() if k[-1] in ('write_content_norm_scale','mlp_write_content_norm_scale')]
    self.assertEqual(len(added),3)
    for k,v in added:
      self.assertTrue(np.isfinite(v).all())
      groups=jnp.moveaxis(v,c.param_scan_axis,0) if 'layers' in k else v[None]
      for g in groups:self.assertGreater(float(jnp.sum(g*g)),0.,str(k))
    print('ALL_WRITE_SCALES_INIT1_PARENT_IDENTICAL_AND_ALL5_TOY_SCALE_GRADS_LIVE',flush=True)

  def test_pallas_read_native_write_all_gradients(self):
    StandardValueTest.check_pallas(self,True)

  def test_generic_health_covers_all_35_scale_groups(self):
    c=self.config(RUN)
    values=jnp.stack([jnp.full((16,75),1+i/20) for i in range(17)])
    scanned=jnp.moveaxis(values,0,c.param_scan_axis)
    params={'params':{'decoder':{'embedding_bam_write':{'write_content_norm_scale':jnp.ones((16,75))},
      'layers':{'block':{'self_attention':{'write_content_norm_scale':scanned},'mlp_write_content_norm_scale':scanned*2}}}}}
    actual=train.compute_params_norm(params,c)
    raw=train.compute_params_norm(jax.tree.map(lambda v:v*.1,params),c,prefix='raw_grads')
    self.assertEqual(len(actual),35);self.assertEqual(len(raw),35)
    self.assertFalse(any(k.startswith('bam/') for k in actual))
    for i in range(17):
      for leaf,mul in [('self_attention/write_content_norm_scale',1),('mlp_write_content_norm_scale',2)]:
        name=f'total_params/decoder/layers_{i}/block/{leaf}'
        rms=actual[name]/math.sqrt(16*75)
        np.testing.assert_allclose(rms,(1+i/20)*mul,rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose(raw[name.replace('total_params','raw_grads')],actual[name]*.1,rtol=1e-5,atol=1e-6)

  def test_scale_parameters_skip_weight_decay_and_validation(self):
    cfg=SimpleNamespace(wd_mults=getattr(exp,RUN).wd_mults,adam_weight_decay=.1)
    p={'params':{'decoder':{'embedding_bam_write':{'write_content_norm_scale':jnp.ones((2,75))},
       'block':{'self_attention':{'write_content_norm_scale':jnp.ones((2,75))},'mlp_write_content_norm_scale':jnp.ones((2,75)),'kernel':jnp.ones((2,75))}}}}
    wd=flatten_dict(train.get_wd_tree(cfg,p))
    for key,v in wd.items():self.assertEqual(v,.1 if key[-1]=='kernel' else 0.)
    for value in [float('nan'),float('inf'),True,'1']:
      with self.assertRaises(ValueError):bam_write.scale_initializer(SimpleNamespace(bam_write_content_norm_scale_init=value))

if __name__=='__main__':unittest.main()
