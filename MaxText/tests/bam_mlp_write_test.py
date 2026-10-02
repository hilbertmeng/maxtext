"""Focused parameter, write equivalence, and scanned forward/gradient gates."""
import functools,tempfile,unittest
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import pyconfig,max_utils,train,train_compile
from layers.models import Transformer
from layers import attentions,quantizations
PREFIX='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWrite'
EXPS=[PREFIX+'EveryThirdTruePile',PREFIX+'EveryLayerTruePile',PREFIX+'StaticEveryLayerTruePile']

class MLPWriteTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();Path(self.tmp.name,'audit').mkdir()
 def tearDown(self):self.tmp.cleanup()
 def config(self,exp,**kw):
  return pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=exp,run_name='audit',enable_checkpointing=False,base_output_directory=self.tmp.name+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.,**kw)
 def test_full_parameters_and_health(self):
  for exp,count in [('BamMediumPropK75EmbedVOnlyQK57AllLocalTruePile',432091328),*zip(EXPS,[432098624,432113216,432122432])]:
   c=self.config(exp);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(args[0].params)
   actual=sum(int(np.prod(v.shape)) for v in flat.values());self.assertEqual(actual,count,exp)
   if exp not in EXPS:continue
   self.assertEqual(c.DATASET_VARIANT,'truepile4096');self.assertEqual(c.bam_k,c.head_dim)
   gates=[v for p,v in flat.items() if 'mlp_write_gate' in p];self.assertEqual(len(gates),1)
   self.assertEqual(sorted(gates[0].shape),sorted((1200,16,6 if exp==EXPS[0] else 18)))
   bias=[v for p,v in flat.items() if 'mlp_write_gate_bias' in p];self.assertEqual(len(bias),1)
   static=[v for p,v in flat.items() if 'mlp_write_address' in p];self.assertEqual(bool(static),exp==EXPS[2])
   if static:self.assertEqual(sorted(static[0].shape),[16,18,32])
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
   written=[l for l in range(18) if (l+1)%c.bam_mlp_write_every==getattr(c,'bam_mlp_write_offset',0)%c.bam_mlp_write_every]
   for l in range(18):
    self.assertEqual(f'bam/concat/mlp_write_gate/layer_{l:03d}/mean' in metrics['scalar'],l in written)
    if l in written:self.assertIn(f'bam/concat/combined_write_amplitude/layer_{l:03d}/bam_over_standard',metrics['scalar'])
   print('FULL_PARAMS_HEALTH_OK',exp,actual,flush=True)
 def test_original_write_and_combined_equivalence(self):
  c=self.config(EXPS[1],dtype='float32',weight_dtype='float32');c.get_keys().update(bam_write_v_bottleneck_dim=32,bam_record_concat_health=False,bam_lambda_decay=.73)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
  a=attentions.BamAttention(config=c,num_query_heads=2,num_kv_heads=2,head_dim=75,bam_k=75,bam_v=32,max_target_length=4,max_prefill_predict_length=4,mesh=mesh,attention_kernel='dot_product_chunk',dtype=c.dtype,layer_mode='local_qk+local_v+local_o',read_side='col',attention_type=c.attention_type)
  x=jax.random.normal(jax.random.key(10),(1,4,150));o=jax.random.normal(jax.random.key(11),(1,4,2,75));M=jax.random.normal(jax.random.key(12),(1,4,75,32));y=jax.random.normal(jax.random.key(13),o.shape);g=jnp.full((1,4,2),.23);sa=jax.random.normal(jax.random.key(14),(2,32))
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   v=a.init(jax.random.key(15),o,x,M,method=a._write)
   old,_=a.apply(v,o,x,M,method=a._write);f=a.apply(v,o,x,method=a._deferred_write_factors)
   ref=.73*M+jnp.einsum('btnk,btnv->btkv',f[0],f[1]);np.testing.assert_allclose(old,ref,rtol=1e-6,atol=1e-6)
   zero=a.apply(v,y,jnp.zeros_like(g),f,M,method=a.merge_mlp_write);np.testing.assert_allclose(zero,old,rtol=1e-6,atol=1e-6)
   yn=a.apply(v,y,method=lambda mod,z:mod.write_data_norm(z));sn=a.apply(v,sa,method=lambda mod,z:mod.write_address_norm(z))
   for implementation in ['dot','mul_reduce']:
    c.get_keys()['bam_write_outer_implementation']=implementation
    shared=a.apply(v,y,g,f,M,method=a.merge_mlp_write)
    ref=old+jnp.einsum('btnk,btnv->btkv',g[...,None]*yn,f[1]);np.testing.assert_allclose(shared,ref,rtol=2e-5,atol=2e-5)
    static=a.apply(v,y,g,f,M,sa,method=a.merge_mlp_write)
    ref=old+jnp.einsum('btnk,nv->btkv',g[...,None]*yn,sn);np.testing.assert_allclose(static,ref,rtol=2e-5,atol=2e-5)
  print('WRITE_EQUIVALENCE_DECAY_ONCE_OK',flush=True)
 def test_scanned_forward_and_gradients(self):
  for exp in EXPS:
   c=self.config(exp);nlayer=6 if exp==EXPS[0] else 3
   c.get_keys().update(base_emb_dim=150,emb_dim=150,num_query_heads=2,num_kv_heads=2,base_num_query_heads=2,base_num_kv_heads=2,base_num_decoder_layers=nlayer,num_decoder_layers=nlayer,base_mlp_dim=128,mlp_dim=128,mlp_dim_by_block=[128]*3 if c.bam_pair_scan else None,vocab_size=128,bam_layer_modes=['local_qk+local_v+local_o']*nlayer,bam_write_v_bottleneck_dim=32,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=32)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quantizations.configure_quantization(c));tokens=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tokens);call=(tokens,pos,tokens,mask,mask)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    params=model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params']
    def loss(p):return jnp.mean(model.apply({'params':p},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0])
    value,grad=jax.jit(jax.value_and_grad(loss))(params)
   self.assertTrue(np.isfinite(float(value)));self.assertTrue(all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grad)))
   flat=flatten_dict(nn.unbox(grad))
   for name in ['mlp_write_gate']+(['mlp_write_address'] if exp==EXPS[2] else []):
    vals=[g for p,g in flat.items() if name in p];self.assertTrue(vals);self.assertGreater(sum(float(jnp.sum(g.astype(jnp.float32)**2)) for g in vals),0,name)
   print('SCANNED_FORWARD_GRAD_OK',exp,float(value),flush=True)

if __name__=='__main__':unittest.main()
