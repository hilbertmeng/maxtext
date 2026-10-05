"""Block-first private MLP writes: exact budgets, tail routing and scan parity."""
import copy, functools, tempfile, unittest
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict
import pyconfig, max_utils, train, train_compile
from layers.models import Transformer
from layers import quantizations
PREFIX='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependent'
ARMS=[(PREFIX+'EverySecondBlockFirstTruePile',2,0,432098528,9),
      (PREFIX+'EveryFourthBlockFirstTruePile',4,2,432095328,5)]

class FrequencyTest(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();Path(self.tmp.name,'audit').mkdir()
 def tearDown(self):self.tmp.cleanup()
 def config(self,name,**kw):
  return pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=name,run_name='audit',enable_checkpointing=False,base_output_directory=self.tmp.name+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=4,max_prefill_predict_length=4,query_chunk_size=2,per_device_batch_size=1.,**kw)
 def test_full_budgets_and_all_layer_health(self):
  for name,period,tail,expected,writes in ARMS:
   c=self.config(name);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   args,kw,sharding,model=train_compile.get_shaped_inputs(mesh,c);flat=flatten_dict(args[0].params)
   total=sum(int(np.prod(v.shape)) for v in flat.values())
   self.assertEqual(total,expected);self.assertEqual(c.DATASET_VARIANT,'truepile4096')
   address=sum(int(np.prod(v.shape)) for p,v in flat.items() if any(n in p for n in ('mlp_address_down','mlp_address_up')))
   self.assertEqual(address,438784*writes)
   self.assertEqual(c.mlp_dim_by_block,[3774]+[3901]*(period-1));self.assertEqual(c.bam_final_local_layer_count,tail)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    metrics=jax.eval_shape(functools.partial(train.train_step,model,c,sharding),*args,**kw)[1]
   for l in range(18):
    self.assertIn(f'bam/concat/local_q_gate/layer_{l:03d}/mean',metrics['scalar'])
    for key,end in [('mlp_write_gate','mean'),('mlp_address_overlap','rho_cross')]:
     self.assertEqual(f'bam/concat/{key}/layer_{l:03d}/{end}' in metrics['scalar'],l%period==0)
   print('FREQUENCY_FULL_BUDGET_HEALTH_OK',name,total,address,flush=True)
 def test_scan_parity_and_consumed_tail_gradient(self):
  for name,period,tail,_,_ in ARMS:
   nlayer=4 if period==2 else 6
   c=self.config(name,dtype='float32',weight_dtype='float32')
   small=dict(emb_dim=150,num_query_heads=2,num_kv_heads=2,base_num_decoder_layers=nlayer,num_decoder_layers=nlayer,mlp_dim=64,mlp_dim_by_block=[64]*period,vocab_size=32,bam_layer_modes=['local_qk+local_v+local_o']*nlayer,bam_write_v_bottleneck_dim=16,emb_bam_num_head=2,emb_bam_v_bottleneck_dim=16,bam_mlp_write_address_rank=16)
   c.get_keys().update(small);mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes)
   model=Transformer(c,mesh,quantizations.configure_quantization(c));tok=jnp.array([[1,2,3,4]],jnp.int32);pos=jnp.arange(4)[None];mask=jnp.ones_like(tok);call=(tok,pos,tok,mask,mask)
   with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
    p=nn.unbox(model.init({'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)},*call,enable_dropout=False)['params'])
    def logits(par):return model.apply({'params':par},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0]
    val,g=jax.jit(jax.value_and_grad(lambda par:jnp.mean(logits(par)**2)))(p)
    self.assertTrue(np.isfinite(float(val)));self.assertTrue(all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(g)))
    flat=flatten_dict(g)
    for key in ('mlp_address_down','mlp_address_up','mlp_write_gate'):
     self.assertGreater(sum(float(jnp.sum(v*v)) for path,v in flat.items() if key in path),0,key)
    if tail:
     self.assertGreater(sum(float(jnp.sum(v*v)) for path,v in flat.items() if 'final_local_layer_0' in path and 'mlp_address_up' in path),0)
    scanned=jax.jit(logits)(p)
   ref=self.config(name,dtype='float32',weight_dtype='float32');ref.get_keys().update(small,scan_layers=False,bam_pair_scan=False,mlp_dim_by_block=None,bam_final_local_layer_count=0)
   refmodel=Transformer(ref,mesh,quantizations.configure_quantization(ref));mapped=copy.copy(p);dec=copy.copy(p['decoder']);blocks=dec.pop('layers');tails=[dec.pop(f'final_local_layer_{i}') for i in range(tail)]
   for l in range(nlayer):
    if l>=nlayer-tail:layer=tails[l-(nlayer-tail)]
    else:
     offset=l%period;key=f'local_{offset}' if offset<period-1 else f'fetch_{offset}'
     layer=jax.tree.map(lambda a:jnp.take(a,l//period,axis=c.param_scan_axis),blocks[key])
    dec[f'layers_{l}']=layer
   mapped['decoder']=dec
   with mesh,nn.partitioning.axis_rules(ref.logical_axis_rules):
    unscanned=jax.jit(lambda par:refmodel.apply({'params':par},*call,enable_dropout=False,rngs={'aqt':jax.random.key(5)})[0])(mapped)
   np.testing.assert_allclose(scanned,unscanned,rtol=3e-5,atol=3e-5)
   print('FREQUENCY_SCAN_PARITY_GRAD_OK',name,float(val),flush=True)

if __name__=='__main__':unittest.main()
