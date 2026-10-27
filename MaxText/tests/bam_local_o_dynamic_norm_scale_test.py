"""Native reference for affine dynamic O addresses; static O and V stay intact."""
import math
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict, unflatten_dict

import exp
import max_utils
import train
import train_compile
from bam_config import validate_bam_config
from bam_prop_test_utils import MLPWriteTest
from layers.attentions import BamAttention

PARENT = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeysLogitDiagNegativeQK75TruePile'
RUN = PARENT.replace('TruePile', 'LocalODynamicNormScaleTruePile')


class LocalODynamicNormScaleTest(MLPWriteTest, unittest.TestCase):
  def fixture(self, run=RUN, initial=1., dtype='float32'):
    cfg=self.config(run,base_emb_dim=150,base_num_query_heads=2,base_num_kv_heads=2,
        dtype=dtype,weight_dtype='float32',bam_local_o_dynamic_norm_scale_init=initial)
    cfg.get_keys().update(bam_pallas_core=False,bam_splash_attention=False,
        bam_local_full_m_read_bottleneck_dim=8,bam_write_v_bottleneck_dim=8)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    module=BamAttention(config=cfg,num_query_heads=2,num_kv_heads=2,head_dim=75,
        bam_k=75,bam_v=32,read_side='col',layer_mode='local_qk+local_v+local_o',layer_inx=0,
        max_target_length=4,max_prefill_predict_length=4,mesh=mesh,
        attention_kernel='dot_product_chunk',dtype=cfg.dtype)
    x=jax.random.normal(jax.random.key(1),(4,4,150),dtype=cfg.dtype)
    matrix=jax.random.normal(jax.random.key(2),(4,4,75,32),dtype=cfg.dtype)
    args=(x,x,jnp.broadcast_to(jnp.arange(4),(4,4)),jnp.ones((4,4),jnp.int32))
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      variables=nn.unbox(module.init(jax.random.key(3),*args,M_in=matrix))
    return cfg,mesh,module,variables,args,matrix

  def test_formal_budget_and_native_configuration(self):
    cfg=self.config(RUN)
    cfg.get_keys().update(max_target_length=256,max_prefill_predict_length=256)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes)
    args,*_=train_compile.get_shaped_inputs(mesh,cfg)
    flat=flatten_dict(nn.unbox(args[0].params))
    new=[v for k,v in flat.items() if k[-1]=='local_o_dynamic_norm_scale']
    self.assertEqual((cfg.num_decoder_layers,cfg.num_query_heads,cfg.bam_v),(18,16,32))
    self.assertEqual(sum(math.prod(v.shape) for v in new),18*16*32)
    self.assertEqual(sum(math.prod(v.shape) for v in flat.values()),432112192+9216)
    self.assertEqual((cfg.mlp_dim,cfg.bam_memory_mlp_dim,cfg.bam_last_mlp_dim),(2510,1128,3904))
    self.assertEqual(cfg.bam_local_o_dynamic_norm_scale_init,1.)
    self.assertFalse(cfg.bam_local_o_learned_scale)
    self.assertEqual(cfg.bam_local_o_drop_head_rate,0.)
    self.assertEqual(cfg.bam_local_o_read_weight_dropout_rate,0.)
    self.assertEqual(cfg.compare_runs,[PARENT])

  def test_configurable_effective_initial_values_and_parent_identity(self):
    cfg,mesh,parent,original,args,matrix=self.fixture(PARENT)
    reference=flatten_dict(original)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      expected=parent.apply(original,*args,M_in=matrix,deterministic=True)
    for initial in (1.,.5,0.,-.25):
      cfg,mesh,module,p,args,matrix=self.fixture(initial=initial)
      np.testing.assert_array_equal(p['params']['local_o_dynamic_norm_scale'],
                                    np.full((2,32),initial,np.float32))
      for key,value in reference.items():
        np.testing.assert_array_equal(flatten_dict(p)[key],value)
      if initial==1.:
        with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
          actual=module.apply(p,*args,M_in=matrix,deterministic=True)
        for a,b in zip(jax.tree.leaves(actual),jax.tree.leaves(expected)):
          np.testing.assert_array_equal(a,b)

  def test_explicit_normalized_address_reference_all_gradients(self):
    cfg,mesh,module,p,args,matrix=self.fixture()
    flat=flatten_dict(p)
    static=next(k for k in flat if k[-1]=='static_o_key')
    flat[static]=jax.random.normal(jax.random.key(30),flat[static].shape)*.1
    gamma=next(k for k in flat if k[-1]=='local_o_dynamic_norm_scale')
    flat[gamma]=jnp.linspace(-.4,1.7,64).reshape(2,32)
    p=unflatten_dict(flat)
    target=jax.random.normal(jax.random.key(31),(4,4,2,150))

    def actual(m,matrix,x,mask):
      v,o=m._independent_local_vo(matrix,x,o_column_weight_multiplier=mask)
      return jnp.concatenate((v,o+m._static_column(matrix,'o',x,column_weight_multiplier=mask)),axis=-1)

    def reference(m,matrix,x,mask):
      def normalized(raw):
        return raw*jax.lax.rsqrt(jnp.mean(raw**2,axis=-1,keepdims=True)+m._read_key_epsilon)
      raw_o=jnp.squeeze(m._project_column_read_key('W_R',x),axis=-2)
      address=normalized(raw_o)*m.local_o_dynamic_norm_scale
      if mask is not None:
        address=address*mask
      logits=m._project_read_gate_logits('W_R_gate',x,squeeze_fetch_axis=True)
      dynamic=jnp.einsum('btkv,btnv->btnk',matrix,address)*m._read_key_scale*jax.nn.sigmoid(logits)
      raw_v=m._project_column_read_key('W_lv_c8',x)
      v_logits=m._project_read_gate_logits('W_lv_gate',x)
      v=jnp.einsum('btkv,btnv->btnk',matrix,normalized(raw_v))*m._read_key_scale*jax.nn.sigmoid(v_logits)
      static_o=m._static_column(matrix,'o',x,column_weight_multiplier=mask)
      return jnp.concatenate((v,dynamic+static_o),axis=-1)

    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      for mask in (None,jax.random.bernoulli(jax.random.key(32),.8,(4,4,2,32)).astype(jnp.float32)/.8):
        def loss(method,p,m,x):
          out=module.apply(p,m,x,mask,method=method)
          return jnp.mean(out*target),out
        (_,a),ga=jax.value_and_grad(lambda p,m,x:loss(actual,p,m,x),argnums=(0,1,2),has_aux=True)(p,matrix,args[0])
        (_,b),gb=jax.value_and_grad(lambda p,m,x:loss(reference,p,m,x),argnums=(0,1,2),has_aux=True)(p,matrix,args[0])
        np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)
        for u,v in zip(jax.tree.leaves(ga),jax.tree.leaves(gb)):
          np.testing.assert_allclose(u,v,rtol=3e-4,atol=2e-6)
        self.assertGreater(float(jnp.sum(jnp.abs(ga[0]['params']['local_o_dynamic_norm_scale']))),0.)
      zero=unflatten_dict(dict(flat));zero['params']['local_o_dynamic_norm_scale']=jnp.zeros((2,32))
      v,o=module.apply(zero,matrix,args[0],method=module._independent_local_vo)
      v1,_=module.apply(p,matrix,args[0],method=module._independent_local_vo)
      np.testing.assert_array_equal(v,v1)
      np.testing.assert_array_equal(o,jnp.zeros_like(o))
      static0=module.apply(zero,matrix,'o',args[0],method=module._static_column)
      static1=module.apply(p,matrix,'o',args[0],method=module._static_column)
      np.testing.assert_array_equal(static0,static1)
      self.assertGreater(float(jnp.max(jnp.abs(static0))),0.)
    print('DYNAMIC_O_NORM_SCALE_EXPLICIT_REFERENCE_ALL_GRADIENTS_OK',flush=True)

  def test_bf16_finite_gradient_and_output_dtype(self):
    cfg,mesh,module,p,args,matrix=self.fixture(dtype='bfloat16',initial=.5)
    def loss(p):
      y,m=module.apply(p,*args,M_in=matrix,deterministic=True)
      self.assertEqual(y.dtype,jnp.bfloat16);self.assertEqual(m.dtype,jnp.bfloat16)
      return jnp.mean(y.astype(jnp.float32)**2)+jnp.mean(m.astype(jnp.float32)**2)
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
      grads=jax.grad(loss)(p)
    gamma=grads['params']['local_o_dynamic_norm_scale']
    self.assertTrue(bool(jnp.all(jnp.isfinite(gamma))))
    self.assertGreater(float(jnp.sum(jnp.abs(gamma))),0.)

  def test_validation_and_existing_weight_decay_exclusion(self):
    cfg=self.config(RUN)
    keys=dict(cfg.get_keys())
    for invalid in (float('nan'),float('inf'),True,'0.5'):
      with self.assertRaisesRegex(ValueError,'finite scalar'):
        validate_bam_config(dict(keys,bam_local_o_dynamic_norm_scale_init=invalid))
    with self.assertRaisesRegex(ValueError,'Pallas scale gradients'):
      validate_bam_config(dict(keys,bam_pallas_core=True))
    with self.assertRaisesRegex(ValueError,'independent VO'):
      validate_bam_config(dict(keys,bam_local_vo_independent_keys=False))
    tree={'params':{'self_attention':{'local_o_dynamic_norm_scale':jnp.ones((2,32)),
                                      'kernel':jnp.ones((2,32))}}}
    wd=train.get_wd_tree(SimpleNamespace(wd_mults=getattr(exp,RUN).wd_mults,adam_weight_decay=.1),tree)
    self.assertEqual(wd['params']['self_attention']['local_o_dynamic_norm_scale'],0.)
    self.assertEqual(wd['params']['self_attention']['kernel'],.1)


if __name__=='__main__':
  unittest.main()
