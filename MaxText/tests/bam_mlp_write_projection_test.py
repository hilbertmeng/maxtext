"""Focused exact-budget, tied-adjoint and scanned gradient checks."""
import functools, unittest
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import exp, max_utils, train, train_compile
from layers import attentions, quantizations
from layers.models import Transformer
from bam_mlp_write_test import MLPWriteTest
PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
SHARED=PARENT.replace('TruePile','WOTransposeTruePile')
SEPARATE=PARENT.replace('TruePile','SeparateProjectionTruePile')
class WriteProjectionTest(unittest.TestCase):
 setUp=MLPWriteTest.setUp
 tearDown=MLPWriteTest.tearDown
 config=MLPWriteTest.config
 def test_full_parameter_budget_and_training_graph(self):
  trees={}
  for name in (PARENT,SHARED,SEPARATE):
   self.assertTrue(hasattr(exp,name));c=self.config(name);self.assertEqual(c.model_name,name)
   self.assertEqual((c.num_decoder_layers,c.emb_dim,c.head_dim),(18,1200,75))
   self.assertEqual(c.mlp_dim_by_block,[3901,3374 if name==SEPARATE else 3774,3901])
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c)
   flat=flatten_dict(args[0].params);count=sum(int(np.prod(v.shape)) for v in flat.values())
   self.assertEqual(count,432096128);trees[name]={p:v.shape for p,v in flat.items()}
   extra=[v for p,v in flat.items() if 'mlp_write_content_kernel' in p]
   if name==SEPARATE:
    self.assertEqual(len(extra),1);self.assertEqual(sorted(extra[0].shape),sorted((6,16,75,1200)))
   else:self.assertFalse(extra)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    scalar=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]['scalar']
   for i in range(18):self.assertEqual(f'bam/concat/mlp_write_gate/layer_{i:03d}/mean' in scalar,i%3==1)
   print('FULL_PARAMETER_TRAINSTEP_OK',name,count,flush=True)
  self.assertEqual(trees[PARENT],trees[SHARED])
 def test_configuration_rejects_invalid_projection(self):
  from bam_config import validate_bam_config
  for update in ({'bam_mlp_write_content_projection':'wrong'}, {'bam_mlp_write_every':0}, {'bam_k':64}):
   c=self.config(SHARED);c.get_keys().update(update)
   with self.assertRaises(ValueError):validate_bam_config(c)
  print('CONTENT_PROJECTION_CONFIG_GUARD_OK',flush=True)
 def test_exact_adjoint_and_kernel_gradients(self):
  c=self.config(SHARED,dtype='float32',weight_dtype='float32');c.get_keys().update(bam_write_v_bottleneck_dim=32,bam_record_concat_health=False)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  a=attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
  x=jax.random.normal(jax.random.key(10),(1,4,150));o=jnp.zeros((1,4,2,75));M=jnp.zeros((1,4,75,32));w=jax.random.normal(jax.random.key(11),(2,75,150))
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   variables=a.init(jax.random.key(12),o,x,M,method=a._write)
   params=nn.unbox(variables['params']);params['out']={'kernel':w}
   actual=a.apply({'params':params},x,method=a.project_mlp_write_content)
   np.testing.assert_allclose(actual,jnp.einsum('btd,nkd->btnk',x,w),rtol=1e-6,atol=1e-6)
   def f(kernel):
    p=dict(params,out={'kernel':kernel})
    return jnp.sum(a.apply({'params':p},x,method=a.project_mlp_write_content)**2)
   grad=jax.grad(f)(w);self.assertGreater(float(jnp.sum(grad**2)),0)
   np.testing.assert_allclose(grad,jax.grad(lambda k:jnp.sum(jnp.einsum('btd,nkd->btnk',x,k)**2))(w),rtol=1e-5,atol=1e-5)
   params['out']['kernel']=jnp.eye(150).reshape(2,75,150)
   identity=a.apply({'params':params},x,method=a.project_mlp_write_content)
   np.testing.assert_allclose(identity,x.reshape(1,4,2,75),rtol=0,atol=0)
   # Independent map has the same contraction/init convention, with its own gradient.
   c.get_keys()['bam_mlp_write_content_projection']='independent'
   iv=a.init(jax.random.key(13),x,method=a.project_mlp_write_content)
   iv=nn.unbox(iv);iv['params']['mlp_write_content_kernel']=w
   independent=a.apply(iv,x,method=a.project_mlp_write_content)
   np.testing.assert_allclose(independent,actual,rtol=0,atol=0)
  print('ACTUAL_TIED_ADJOINT_IDENTITY_AND_GRADIENT_OK',flush=True)
 def test_scanned_forward_and_consumed_gradients(self):
  for name in (PARENT,SHARED,SEPARATE):
   c=self.config(name);c.get_keys().update(base_emb_dim=150,emb_dim=150,base_num_query_heads=2,num_query_heads=2,base_num_kv_heads=2,num_kv_heads=2,base_num_decoder_layers=6,num_decoder_layers=6,base_mlp_dim=128,mlp_dim=128,mlp_dim_by_block=[128]*3,bam_layer_modes=['local_qk+local_v+local_o']*6,vocab_size=128,bam_write_v_bottleneck_dim=32,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=32)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quantizations.configure_quantization(c));tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok);call=(tok,pos,tok,mask,mask)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
    def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0])
    value,grad=jax.jit(jax.value_and_grad(loss))(params)
   self.assertTrue(np.isfinite(float(value)));self.assertTrue(all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad)))
   flat=flatten_dict(nn.unbox(grad))
   for needle in ['mlp_address_down','mlp_write_gate']+(['mlp_write_content_kernel'] if name==SEPARATE else []):
    gs=[g for p,g in flat.items() if needle in p];self.assertTrue(gs);self.assertGreater(sum(float(jnp.sum(g.astype(jnp.float32)**2)) for g in gs),0)
   self.assertTrue(any('out' in p and float(jnp.sum(g.astype(jnp.float32)**2))>0 for p,g in flat.items()))
   print('SCANNED_FORWARD_CONSUMED_GRADIENT_OK',name,float(value),flush=True)
if __name__=='__main__':unittest.main()
