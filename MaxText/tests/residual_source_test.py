import os,tempfile,unittest
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import exp,pyconfig,max_utils
from layers.models import Transformer
import residual_source_capture as capture
class SourceTest(unittest.TestCase):
 def test_statistics(self):
  rng=np.random.default_rng(52);x,m,v=(rng.normal(size=(2,9,16)).astype(np.float32) for _ in range(3));bm=rng.normal(size=(16,)).astype(np.float32);bv=rng.normal(size=(16,)).astype(np.float32)
  am=m+bm;af=v+bv;y=af+(x+am)
  actual={k:np.asarray(a) for k,a in capture.statistics(*map(jnp.array,[x,m,v,bm,bv,am,af,y])).items()}
  for key,a in [('x_square',x*x),('m_square',m*m),('v_square',v*v),('mv_cross',m*v),('xm_cross',x*m),('xv_cross',x*v),('m_full_square',am*am),('v_full_square',af*af)]:np.testing.assert_allclose(actual[key],a.mean(-1),rtol=2e-6,atol=1e-6)
  for key in ['m_terminal_reconstruction_error','v_terminal_reconstruction_error','block_reconstruction_error']:np.testing.assert_array_equal(actual[key],np.zeros((2,9)))
 def test_native(self):
  capture.install();memory=bool(int(os.environ.get('TEST_MEMORY','0')));pallas=bool(int(os.environ.get('TEST_PALLAS','0')))
  attrs=dict(max_target_length=128,max_prefill_predict_length=128,target_train_batch_size=1,per_device_batch_size=1.0,weight_dtype='float32',bam_pallas_core=pallas,bam_pallas_interpret=pallas,bam_pallas_body='v8t',bam_pallas_fused_inputs=pallas)
  if memory:attrs.update(base_mlp_dim=64,mlp_dim_by_block=[64]*3,bam_final_local_mlp_dim=64,bam_last_mlp_dim=48,bam_memory_mlp_dim=32,bam_memory_mlp_last_write_dim=0,bam_mlp_write_every=1,bam_mlp_write_offset=0,bam_skip_final_mlp_write=True)
  exp.SourceTiny=type('SourceTiny',(exp.BamPallasPythiaR768TinyXlaTest,),attrs)
  with tempfile.TemporaryDirectory() as out:
   os.mkdir(out+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='SourceTiny',run_name='test',base_output_directory=out+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_training_health_metrics=False)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);tok=jax.random.randint(jax.random.key(1),(1,128),0,1000);pos=jnp.arange(128)[None];seg=jnp.ones_like(tok);rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)};kwargs=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
   capture.ACTIVE=False
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    params=model.init(rng,tok,pos,**kwargs);plain=model.apply(params,tok,pos,rngs=rng,**kwargs);capture.ACTIVE=True
    observed,raw=model.apply(params,tok,pos,rngs=rng,mutable=['intermediates'],**kwargs);again=model.init(rng,tok,pos,**kwargs)
   self.assertEqual(jax.tree.structure(params),jax.tree.structure(again))
   for a,b in zip(jax.tree.leaves(plain),jax.tree.leaves(observed)):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
   for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(again)):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
   raw=flatten_dict(raw['intermediates']);names=[k[-1] for k in raw]
   for key in ['x_square','m_square','v_square','mv_cross','m_full_square','v_full_square']:self.assertIn('stream_source_'+key,names)
   for k,a in raw.items():
    if k[-1].endswith('reconstruction_error'):self.assertEqual(float(np.max(np.asarray(a[0]))),0.)
   print('NATIVE_SOURCE_OBSERVER_VERIFIED',memory,pallas,flush=True);capture.ACTIVE=False
if __name__=='__main__':unittest.main()
