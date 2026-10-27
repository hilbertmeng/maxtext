"""Native serial full-history Mudd: preserve bias, drop all MHA heads at one layer."""
import tempfile,os
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import exp,pyconfig,max_utils
from layers.models import Transformer
import no_self_capture as base
import late_mha_capture as adapter

def main():
 T=128;L=3
 exp.LateMuddTiny=type('LateMuddTiny',(exp.MuddPythia3BSerialMLPShortTrain,),dict(base_num_decoder_layers=L,base_emb_dim=64,base_num_query_heads=2,base_num_kv_heads=2,head_dim=32,base_mlp_dim=96,dynamic_mlp_dim=False,mudd_router_hidden_expand=[1,1,4],dtype='bfloat16',weight_dtype='float32',vocab_size=128,pythia_rotary_dim=8,attention='dot_product',max_target_length=T,max_prefill_predict_length=T,target_train_batch_size=1,per_device_batch_size=1.,scan_layers=False))
 with tempfile.TemporaryDirectory() as root:
  os.mkdir(root+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='LateMuddTiny',run_name='test',base_output_directory=root+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False,record_training_health_metrics=False)
  assert cfg.dense_conn and not cfg.bam_enabled and not cfg.pythia_parallel_residual
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);rngs={k:jax.random.key(i+1) for i,k in enumerate(['params','dropout','aqt'])};tokens=jax.random.randint(jax.random.key(17),(1,T),0,128);seg=jnp.ones_like(tokens);pos=jnp.arange(T)[None];kw=dict(decoder_target_tokens=tokens,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=model.init(rngs,tokens,pos,**kw)
  params=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,params,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned));flat=flatten_dict(params);biases=0
  for path,v in list(flat.items()):
   if path[-3:]==('self_attention','out','bias'):flat[path]=jnp.full_like(v,.125);biases+=1
  assert biases==L;params=unflatten_dict(flat)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):native=jax.jit(lambda:model.apply(params,tokens,pos,rngs=rngs,**kw)[0])()
  adapter.install();base.RETAIN=True
  def apply(c):
   base.ACTIVE=True;base.OBSERVE=True;base.CONTROLS=c
   try:return model.apply(params,tokens,pos,rngs=rngs,mutable=['intermediates'],**kw)
   finally:base.ACTIVE=False;base.OBSERVE=False;base.CONTROLS=None
  fn=jax.jit(apply);ones=jnp.ones((L,4),jnp.float32)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):neutral,raw=fn(ones);ablated,change=fn(ones.at[0,[0,2]].set(0));repeat,_=fn(ones)
  np.testing.assert_allclose(native,neutral[0],rtol=0,atol=.005);np.testing.assert_array_equal(neutral[0],repeat[0]);f={p:np.asarray(v[0]) for p,v in flatten_dict(change['intermediates']).items()};n={p[-1]:v for p,v in f.items() if 'layers_0' in p}
  assert n['audit_mha'].shape==(1,T,2,32) and np.any(n['audit_mha']!=0);np.testing.assert_array_equal(n['audit_vector_content'],0);np.testing.assert_array_equal(n['audit_projected_output'],np.broadcast_to(n['audit_output_bias'],n['audit_projected_output'].shape));assert np.isfinite(ablated[0]).all()
  print('LATE_MHA_MUDD_ALL_HEADS_BIAS_AND_REPEAT_PASS',flush=True)
if __name__=='__main__':main()
