import os,tempfile,unittest
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import exp,pyconfig,max_utils
from layers.models import Transformer
import cross_scale_capture as capture
class TestCrossScale(unittest.TestCase):
 def test_native_and_links(self):
  import memory_geometry_capture as geometry
  geometry.install();geometry.ACTIVE=True
  import mudd_router_capture as router
  router.install();router.ACTIVE=True
  capture.install();kind=os.environ['TEST_KIND'];mudd=kind.endswith('mudd');py=kind.startswith('py')
  if mudd:
   parent=exp.MuddPythia3BSerialMLPShortTrain if py else exp.MuddLlama2XLProp
   attrs=dict(base_emb_dim=128,base_num_query_heads=4,base_num_kv_heads=4,head_dim=32,base_mlp_dim=256,base_num_decoder_layers=3,mudd_router_hidden_expand=[1,1,4],vocab_size=1024,dtype='float32',weight_dtype='float32',attention='dot_product')
   if py:attrs['pythia_rotary_dim']=8
  else:
   parent=exp.BamPallasPythiaR768TinyXlaTest if py else exp.BamPallasFullMTinyXlaTest
   attrs=dict(vocab_size=1024,dtype='float32',weight_dtype='float32',bam_pallas_core=False)
  attrs.update(max_target_length=128,max_prefill_predict_length=128,target_train_batch_size=1,per_device_batch_size=1.)
  exp.CrossTiny=type('CrossTiny',(parent,),attrs)
  with tempfile.TemporaryDirectory() as out:
   os.mkdir(out+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='CrossTiny',run_name='test',base_output_directory=out+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);tok=jax.random.randint(jax.random.key(1),(1,128),0,1000);pos=jnp.arange(128)[None];seg=jnp.ones_like(tok);rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)};kw=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
   capture.ACTIVE=False;capture.TOKENS=tok
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    params=model.init(rng,tok,pos,**kw);plain=model.apply(params,tok,pos,rngs=rng,**kw);capture.ACTIVE=True
    observed,raw=model.apply(params,tok,pos,rngs=rng,mutable=['intermediates'],**kw);again=model.init(rng,tok,pos,**kw);capture.ACTIVE=False
    def ctrl(c):
     capture.CONTROLS=c
     try:return model.apply(params,tok,pos,rngs=rng,**kw)
     finally:capture.CONTROLS=None
    neutral=ctrl(jnp.zeros((3,cfg.num_decoder_layers)));no_last_M=ctrl(jnp.zeros((3,cfg.num_decoder_layers)).at[2,-1].set(-.1));perturbed=ctrl(jnp.zeros((3,cfg.num_decoder_layers)).at[1,-1].set(-.1))
   for a,b,c,d in zip(jax.tree.leaves(plain),jax.tree.leaves(observed),jax.tree.leaves(neutral),jax.tree.leaves(no_last_M)):
    for v in [b,c,d]:np.testing.assert_array_equal(np.asarray(a),np.asarray(v))
   for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(again)):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
   self.assertFalse(np.array_equal(np.asarray(plain[0]),np.asarray(perturbed[0])))
   flat=flatten_dict(raw['intermediates']);names=[k[-1] for k in flat]
   if mudd:
    def unwrap(v):
     while isinstance(v,(tuple,list)) and len(v)==1:v=v[0]
     return v
    router.VALUES={'/'.join(k[:-1]):unwrap(v) for k,v in flat.items() if k[-1]=='stream_router_weights'};router.MASK=jnp.ones(5,bool);router.ACTIVE=False
    with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):replayed=model.apply(params,tok,pos,rngs=rng,**kw)
    for a,b in zip(jax.tree.leaves(plain),jax.tree.leaves(replayed)):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    router.VALUES=None;router.MASK=None
    print('NATIVE_ROUTER_REPLAY_VERIFIED',kind,flush=True)
   for key in ['stream_position_top_key_index','stream_source_m_square','stream_source_v_square','stream_ffn_main_positive_ratio']:self.assertIn(key,names)
   for k,v in flat.items():
    if k[-1]=='stream_source_reconstruction_error':self.assertEqual(float(np.max(np.asarray(v[0]))),0.)
   print('NATIVE_ACTIVATION_AND_DIRECT_LINKS_VERIFIED',kind,flush=True)
if __name__=='__main__':unittest.main()
