import os,tempfile,unittest
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
import exp,pyconfig,max_utils
from layers.models import Transformer
import v_shuffle_capture as route
class VShuffleTest(unittest.TestCase):
 def test_native_identity(self):
  kind=os.environ['TEST_KIND'];is_mudd=kind.endswith('mudd');is_py=kind.startswith('py')
  if is_mudd:
   parent=exp.MuddPythia3BSerialMLPShortTrain if is_py else exp.MuddLlama2XLProp
   attrs=dict(base_emb_dim=64,base_num_query_heads=2,base_num_kv_heads=2,head_dim=32,base_mlp_dim=128,base_num_decoder_layers=3,mudd_router_hidden_expand=[1,1,4],vocab_size=128,dtype='float32',weight_dtype='float32',attention='dot_product')
   if is_py:attrs['pythia_rotary_dim']=8
  else:
   parent=exp.BamPallasPythiaR768TinyXlaTest if is_py else exp.BamPallasFullMTinyXlaTest
   attrs=dict(vocab_size=128,dtype='float32',weight_dtype='float32',bam_pallas_core=False,query_chunk_size=16)
  attrs.update(max_target_length=16,max_prefill_predict_length=16,target_train_batch_size=2,per_device_batch_size=2.)
  exp.VShuffleTiny=type('VShuffleTiny',(parent,),attrs)
  route.install()
  with tempfile.TemporaryDirectory() as out:
   os.mkdir(out+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='VShuffleTiny',run_name='test',base_output_directory=out+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None)
   tok=jax.random.randint(jax.random.key(1),(2,16),0,128);pos=jnp.broadcast_to(jnp.arange(16),tok.shape);seg=jnp.ones_like(tok);rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)};kw=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    params=model.init(rng,tok,pos,**kw);native=model.apply(params,tok,pos,rngs=rng,**kw)
    def ctrl(shift):
     route.SHIFT=shift
     try:return model.apply(params,tok,pos,rngs=rng,**kw)
     finally:route.SHIFT=None
    fn=jax.jit(ctrl);identity=fn(jnp.array(0));foreign=fn(jnp.array(1))
    route.SHIFT=jnp.array(1);again=model.init(rng,tok,pos,**kw);route.SHIFT=None
   for x,y in zip(jax.tree.leaves(params),jax.tree.leaves(again)):np.testing.assert_array_equal(x,y)
   for x,y in zip(jax.tree.leaves(native),jax.tree.leaves(identity)):np.testing.assert_allclose(x,y,atol=2e-5,rtol=2e-5)
   self.assertGreater(float(np.max(np.abs(np.asarray(native[0])-np.asarray(foreign[0])))),1e-7)
   print('LIVE_V_SHUFFLE_IDENTITY_AND_INTERVENTION_VERIFIED',kind,flush=True)
 def test_pallas_path_isolation(self):
  # Test the intervention interface independently of kernel arithmetic already tested upstream.
  def native(*args,**kwargs):
   g=kwargs['static_gates'];v=args[0]+args[6]+args[8]+g[2]
   return args[2],args[4],v,args[6]+args[7]+g[3],args[0]
  args=tuple(jnp.arange(12,dtype=jnp.float32).reshape(3,4)+i for i in range(11));g=tuple(a+20 for a in args[:4]);opts={'static_gates':g}
  base=native(*args,**opts);identity=route.pallas_v_read(native,args,opts,jnp.array(0));out=route.pallas_v_read(native,args,opts,jnp.array(1))
  for x,y in zip(base,identity):np.testing.assert_array_equal(x,y)
  for i in [0,1,3,4]:np.testing.assert_array_equal(out[i],base[i])
  np.testing.assert_array_equal(out[2],args[0]+jnp.roll(args[6],1,0)+jnp.roll(args[8],1,0)+jnp.roll(g[2],1,0))
if __name__=='__main__':unittest.main()
