"""Combined observers preserve native computation and separate vector/memory neurons."""
import os,tempfile,unittest
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict
import exp,pyconfig,max_utils
from layers.models import Transformer
import head_position_capture as head
import ffn_activation_capture as ffn
class CombinedTest(unittest.TestCase):
 def run_model(self,pallas):
  head.install();ffn.install(include_readout=False)
  attrs=dict(base_mlp_dim=64,mlp_dim_by_block=[64]*3,bam_final_local_mlp_dim=64,bam_last_mlp_dim=48,bam_memory_mlp_dim=32,bam_memory_mlp_last_write_dim=0,bam_mlp_write_every=1,bam_mlp_write_offset=0,bam_skip_final_mlp_write=True,max_target_length=128,max_prefill_predict_length=128,target_train_batch_size=1,per_device_batch_size=1.0,weight_dtype='float32',bam_pallas_core=pallas,bam_pallas_interpret=pallas,bam_pallas_body='v8t',bam_pallas_fused_inputs=pallas)
  separate=os.environ.get('TEST_SEPARATE','0')=='1'
  if separate:attrs.update(bam_local_vo_independent_keys=True,bam_pair_scan=False,bam_local_fetch_block_size=None,bam_final_local_layer_count=1,bam_final_local_mlp_dim=None,mlp_dim_by_block=None,bam_attention_write_layers=list(range(4)))
  exp.MemoryCaptureTiny=type('MemoryCaptureTiny',(exp.BamPallasPythiaR768TinyXlaTest,),attrs)
  with tempfile.TemporaryDirectory() as out:
   os.mkdir(out+'/test')
   cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='MemoryCaptureTiny',run_name='test',base_output_directory=out+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_training_health_metrics=False)
   mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None)
   tok=jax.random.randint(jax.random.key(2),(1,128),0,1000);pos=jnp.arange(128)[None];seg=jnp.ones_like(tok);rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)}
   kwargs=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
   head.ACTIVE=ffn.ACTIVE=False;head.TOKENS=tok
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
    params=model.init(rng,tok,pos,**kwargs);baseline=model.apply(params,tok,pos,rngs=rng,**kwargs)
    head.ACTIVE=ffn.ACTIVE=True
    observed,raw=model.apply(params,tok,pos,rngs=rng,mutable=['intermediates'],**kwargs)
    init_again=model.init(rng,tok,pos,**kwargs)
   for a,b in zip(jax.tree.leaves(baseline),jax.tree.leaves(observed)):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
   for a,b in zip(jax.tree.leaves(params),jax.tree.leaves(init_again)):np.testing.assert_array_equal(np.asarray(nn.unbox(a)),np.asarray(nn.unbox(b)))
   if separate:self.assertTrue(any('W_lv_c8_up' in path for path in flatten_dict(params['params'])))
   raw=flatten_dict(raw['intermediates']);names=[k[-1] for k in raw]
   for key in ['stream_position_entropy','stream_position_top_key_index','stream_ffn_main_neuron_positive_count','stream_ffn_memory_neuron_positive_count','stream_ffn_address_neuron_positive_count','stream_ffn_write_logit_neuron_positive_count']:self.assertIn(key,names)
   for path,value in raw.items():
    if path[-1]=='stream_ffn_memory_neuron_positive_count':self.assertEqual(np.asarray(value[0]).shape[-1],32)
    if path[-1]=='stream_ffn_main_neuron_positive_count':self.assertIn(np.asarray(value[0]).shape[-1],(48,64))
   print('COMBINED_NATIVE_AND_BRANCH_CAPTURE_OK',pallas,flush=True)
   head.ACTIVE=ffn.ACTIVE=False
 def test_xla(self):self.run_model(False)
 def test_pallas(self):self.run_model(True)
if __name__=='__main__':unittest.main()
