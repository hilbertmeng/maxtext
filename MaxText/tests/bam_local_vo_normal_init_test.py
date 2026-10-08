"""Local shared C8 VO initializer, independent gates and scale isolation."""
import functools, unittest
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import max_utils, train, train_compile
from layers.models import Transformer
from layers import attentions, quantizations
from bam_mlp_write_test import MLPWriteTest
PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdLocalVStaticZeroTruePile'
EXP=PARENT.replace('TruePile','VOReadNormalTruePile')
class LocalVONormalTest(unittest.TestCase):
  setUp=MLPWriteTest.setUp
  tearDown=MLPWriteTest.tearDown
  config=MLPWriteTest.config
  def test_budget_and_health(self):
    c=self.config(EXP)
    self.assertEqual(c.bam_read_key_scale,.2)
    self.assertEqual(c.bam_local_vo_read_key_scale,1.)
    self.assertEqual(c.bam_local_v_read_gate_init,.1)
    self.assertEqual(c.bam_read_gate_init,.05)
    self.assertTrue(c.bam_local_v_static_zero_init)
    self.assertFalse(c.bam_attn_write_content_pre_rms_bias)
    self.assertEqual(c.mlp_dim_by_block,[3901,3774,3901])
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
    self.assertEqual(sum(int(np.prod(v.shape)) for v in jax.tree.leaves(args[0].params)),432096128)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
    for i in range(18):
      self.assertIn(f'bam/concat/local_v_gate/layer_{i:03d}/mean',metrics)
      self.assertIn(f'bam/concat/local_o_gate/layer_{i:03d}/mean',metrics)
    print('BUDGET_432096128_QK_SCALE_ISOLATED_HEALTH_OK',flush=True)
  def test_leaf_parity_and_live_forward_gradient(self):
    t=jnp.array([[1,2,3,4]],jnp.int32);call=(t,jnp.arange(4)[None],t,jnp.ones_like(t),jnp.ones_like(t))
    trees=[]
    for name in (PARENT,EXP):
      c=self.config(name,dtype='float32',weight_dtype='float32')
      c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,
          base_num_decoder_layers=3,num_decoder_layers=3,mlp_dim=64,mlp_dim_by_block=[64]*3,
          vocab_size=32,bam_layer_modes=['local_qk+local_v+local_o']*3,
          bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
      mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
      model=Transformer(c,mesh,quantizations.configure_quantization(c))
      with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
        p=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
        if name==EXP:
          def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]**2)
          value,grad=jax.jit(jax.value_and_grad(loss))(p)
      trees.append(flatten_dict(nn.unbox(p)))
    self.assertEqual(trees[0].keys(),trees[1].keys())
    for path,v in trees[1].items():
      if 'W_R' in path and 'kernel' in path:
        self.assertGreater(float(jnp.linalg.norm(v)),0.)
        np.testing.assert_array_equal(trees[0][path],0.)
      elif 'W_lv_gate_b0' in path:
        np.testing.assert_allclose(jax.nn.sigmoid(v),.1,rtol=1e-6)
      else:np.testing.assert_array_equal(v,trees[0][path])
      if 'static_v_key' in path:np.testing.assert_array_equal(v,0.)
      if 'W_R_gate_b0' in path:np.testing.assert_allclose(jax.nn.sigmoid(v),.05,rtol=1e-6)
    self.assertTrue(np.isfinite(float(value)))
    self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(grad)))
    flat=flatten_dict(nn.unbox(grad))
    for needle in ('W_R','W_lv_gate','static_v_key'):
      self.assertGreater(sum(float(jnp.sum(v*v)) for p,v in flat.items() if needle in p),0.)
    print('ONLY_KEY_AND_V_GATE_CHANGED_FINITE_SCANNED_GRADS_OK',flush=True)
  def test_actual_vo_read_scaling_and_legacy_parity(self):
    c=self.config(EXP,dtype='float32',weight_dtype='float32')
    c.get_keys().update(bam_write_v_bottleneck_dim=16,bam_record_concat_health=False)
    mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
    a=attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,
        max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',
        dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
    x=jax.random.normal(jax.random.key(10),(1,4,150));m=jax.random.normal(jax.random.key(11),(1,4,75,32))
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
      p=a.init(jax.random.key(12),m,x,method=a._independent_local_vo)
      v,o=a.apply(p,m,x,method=a._independent_local_vo)
      read,logits=a.apply(p,m,x,method=lambda mod,m,x:mod._read_fetched_m(mod._compress_m(m),x,ungated=True))
      ref=a.apply(p,read,method=lambda mod,r:mod._expand_full_read(r))
      np.testing.assert_allclose(v,.1*ref,rtol=1e-6,atol=1e-6)
      np.testing.assert_allclose(o,.05*ref,rtol=1e-6,atol=1e-6)
      c.get_keys()['bam_local_vo_read_key_scale']=None
      v0,o0=a.apply(p,m,x,method=a._independent_local_vo)
      np.testing.assert_allclose(v0,.2*v,rtol=1e-6,atol=1e-6)
      np.testing.assert_allclose(o0,.2*o,rtol=1e-6,atol=1e-6)
    print('V_010_O_005_VO_SCALE1_AND_LEGACY_SCALE_PARITY_OK',flush=True)
if __name__=='__main__':unittest.main()
