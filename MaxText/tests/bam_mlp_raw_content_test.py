"""MLP-only raw matrix content: budgets, normalization boundaries and gradients."""
import functools, tempfile, unittest
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import pyconfig, max_utils, train, train_compile
from layers import attentions, quantizations
from layers.models import Transformer

PARENT='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
EXP='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdRawContentTruePile'

class RawContentTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();Path(self.tmp.name,'audit').mkdir()
 def tearDown(self):self.tmp.cleanup()
 def config(self,name=EXP,**kw):
  return pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=name,run_name='audit',enable_checkpointing=False,base_output_directory=self.tmp.name+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.,**kw)
 def test_exact_budget_and_health(self):
  signatures=[]
  for name in [PARENT,EXP]:
   c=self.config(name);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(args[0].params)
   total=sum(int(np.prod(v.shape)) for v in flat.values());self.assertEqual(total,432096128)
   signatures.append({p:v.shape for p,v in flat.items()})
   self.assertTrue(c.bam_write_data_rms);self.assertEqual(c.bam_mlp_write_content_rms,name==PARENT)
   self.assertEqual(c.mlp_dim_by_block,[3901,3774,3901]);self.assertEqual(c.DATASET_VARIANT,'truepile4096')
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
   for l in range(18):
    for group,stat in [('mlp_write_raw_output_amplitude','bam_rms'),('mlp_write_content_amplitude','bam_rms'),('mlp_write_gate','mean'),('combined_write_amplitude','bam_over_standard')]:
     self.assertEqual(f'bam/concat/{group}/layer_{l:03d}/{stat}' in metrics['scalar'],l in range(1,18,3))
  self.assertEqual(*signatures)
  print('RAW_CONTENT_BUDGET_TREE_HEALTH_OK',total,flush=True)
 def test_only_mlp_content_changes(self):
  c=self.config(dtype='float32',weight_dtype='float32');c.get_keys().update(bam_record_concat_health=False,bam_lambda_decay=.73,bam_write_v_bottleneck_dim=16)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  a=attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
  x=jax.random.normal(jax.random.key(10),(1,4,150));o=jax.random.normal(jax.random.key(11),(1,4,2,75));m=jax.random.normal(jax.random.key(12),(1,4,75,32));y=.2*jax.random.normal(jax.random.key(13),o.shape);addr=jax.random.normal(jax.random.key(14),(1,4,2,32));g=jnp.full((1,4,2),.23)
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   p=a.init(jax.random.key(15),o,x,m,method=a._write)
   old,_=a.apply(p,o,x,m,method=a._write);f=a.apply(p,o,x,method=a._deferred_write_factors)
   an=a.apply(p,addr,method=lambda mod,z:mod.write_address_norm(z))
   for implementation in ('dot','mul_reduce'):
    c.get_keys().update(bam_write_outer_implementation=implementation,bam_mlp_write_content_rms=False)
    old,_=a.apply(p,o,x,m,method=a._write)
    def write(v,key):return a.apply(p,v,g,f,m,independent_address=key,method=a.merge_mlp_write)
    raw=write(y,addr);delta=jnp.einsum('btnk,btnv->btkv',g[...,None]*y,an)
    np.testing.assert_allclose(raw,old+delta,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(write(2*y,addr),old+2*delta,rtol=2e-5,atol=2e-5)
    np.testing.assert_allclose(write(y,3*addr),raw,rtol=2e-5,atol=2e-5)
    zero=a.apply(p,y,jnp.zeros_like(g),f,m,independent_address=addr,method=a.merge_mlp_write)
    np.testing.assert_allclose(zero,old,rtol=2e-5,atol=2e-5)
    c.get_keys()['bam_mlp_write_content_rms']=True
    attn,_=a.apply(p,o,x,m,method=a._write)
    np.testing.assert_array_equal(attn,old)
    yn=a.apply(p,y,method=lambda mod,z:mod.write_data_norm(z))
    normalized=write(y,addr)
    np.testing.assert_allclose(normalized,old+jnp.einsum('btnk,btnv->btkv',g[...,None]*yn,an),rtol=2e-5,atol=2e-5)
  print('RAW_CONTENT_ONLY_MLP_CHANGES_OK',flush=True)
 def test_scanned_finite_and_consumed_write_gradients(self):
  c=self.config(dtype='float32',weight_dtype='float32');c.get_keys().update(emb_dim=150,num_query_heads=2,num_kv_heads=2,base_num_decoder_layers=6,num_decoder_layers=6,mlp_dim=64,mlp_dim_by_block=[64]*3,vocab_size=32,bam_layer_modes=['local_qk+local_v+local_o']*6,bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quantizations.configure_quantization(c));tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tokens);call=(tokens,pos,tokens,mask,mask)
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   p=nn.unbox(model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params'])
   def loss(par):return jnp.mean(model.apply({'params':par},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]**2)
   val,g=jax.jit(jax.value_and_grad(loss))(p)
  self.assertTrue(np.isfinite(float(val)));self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g)))
  flat=flatten_dict(g)
  for key in ('mlp_address_down','mlp_address_up','mlp_write_gate'):
   self.assertGreater(sum(float(jnp.sum(v*v)) for path,v in flat.items() if key in path),0,key)
  print('RAW_CONTENT_SCANNED_GRAD_OK',float(val),flush=True)

if __name__=='__main__':unittest.main()
