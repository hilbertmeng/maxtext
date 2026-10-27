"""Native neutral parity, gate/projection algebra, and independent route interventions."""
import os,tempfile
import jax,jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import exp,pyconfig,max_utils
from layers.models import Transformer
import no_self_capture as hook
import late_mha_capture as adapter

def main():
 # Direct per-head projection audit independent of Gram contraction order.
 rng=np.random.default_rng(17);aa=rng.normal(size=(2,5,3,4)).astype(np.float32);oo=rng.normal(size=aa.shape).astype(np.float32);ww=rng.normal(size=(3,4,7)).astype(np.float32);pa=np.einsum('bqhk,hkd->bqhd',aa,ww);po=np.einsum('bqhk,hkd->bqhd',oo,ww);expected=np.stack([(pa*pa).sum((1,3)),(po*po).sum((1,3)),(pa*po).sum((1,3)),np.full((2,3),35)],-1);np.testing.assert_allclose(hook.head_projected_moments(jnp.asarray(aa),jnp.asarray(oo),jnp.asarray(ww)),expected,rtol=1e-5,atol=1e-5)
 full=os.environ.get('TEST_FULL','0')=='1';dtype=os.environ.get('TEST_DTYPE','float32');T=128;L=3
 attrs=dict(base_num_decoder_layers=L,bam_layer_modes=['local_qk+local_v+local_o']*L,bam_pair_scan=False,bam_local_fetch_block_size=None,bam_final_local_layer_count=1,bam_local_vo_independent_keys=True,bam_attention_negative_diagonal=False,bam_local_qk_col_output_dim=32 if full else 24,bam_partial_rope_nope_dim=32 if full else 24,bam_attention_write_layers=list(range(L-1)),bam_skip_final_mlp_write=True,bam_pallas_core=True,bam_pallas_interpret=True,bam_pallas_body='v8t',bam_pallas_fused_inputs=True,bam_pallas_write=True,bam_splash_attention=False,query_chunk_size=None,max_target_length=T,max_prefill_predict_length=T,base_mlp_dim=96,mlp_dim_by_block=None,bam_final_local_mlp_dim=None,bam_last_mlp_dim=80,bam_memory_mlp_dim=48,bam_memory_mlp_last_write_dim=0,bam_mlp_write_every=1,bam_mlp_write_offset=0,dtype=dtype,weight_dtype='float32',vocab_size=128,target_train_batch_size=1,per_device_batch_size=1.)
 exp.ReadoutTiny=type('ReadoutTiny',(exp.BamPallasPythiaR768TinyXlaTest,),attrs)
 with tempfile.TemporaryDirectory() as root:
  os.mkdir(root+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='ReadoutTiny',run_name='test',base_output_directory=root+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False,record_training_health_metrics=False)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);rngs={k:jax.random.key(i+1) for i,k in enumerate(['params','dropout','aqt'])};tokens=jax.random.randint(jax.random.key(17),(1,T),0,128);seg=jnp.ones_like(tokens);pos=jnp.arange(T)[None];kw=dict(decoder_target_tokens=tokens,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=model.init(rngs,tokens,pos,**kw)
  params=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,params,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
  flat=flatten_dict(params);biases=0
  for path,value in list(flat.items()):
   if path[-3:]==('self_attention','out','bias'):flat[path]=jnp.full_like(value,.125);biases+=1
  assert biases;params=unflatten_dict(flat)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):baseline=jax.jit(lambda:model.apply(params,tokens,pos,rngs=rngs,**kw)[0])()
  adapter.install();hook.RETAIN=True
  def apply(c,observe):
   hook.ACTIVE=True;hook.OBSERVE=observe;hook.CONTROLS=c
   try:return model.apply(params,tokens,pos,rngs=rngs,mutable=['intermediates'],**kw)
   finally:hook.ACTIVE=False;hook.OBSERVE=False;hook.CONTROLS=None
  observed=jax.jit(lambda c:apply(c,True));ones=jnp.ones((L,4),jnp.float32)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):neutral,raw=observed(ones)
  drift=float(np.max(np.abs(np.asarray(neutral[0])-np.asarray(baseline)).mean(1)));assert drift<(.005 if dtype=='bfloat16' else 1e-5)
  flat={p:np.asarray(v[0]) for p,v in flatten_dict(raw['intermediates']).items()};seen=set()
  for path,index in flat.items():
   if path[-1]!='readout_layer_index':continue
   for point in np.ndindex(index.shape):
    seen.add(int(index[point]));row={p[-1]:v[point] for p,v in flat.items() if p[:-1]==path[:-1]};assert row['readout_pre'].shape==(1,5);assert (row['readout_geometry'][...,:3]==np.array([T,cfg.num_query_heads,cfg.head_dim])).all()
  assert seen==set(range(L))
  ablate=jax.jit(lambda c:apply(c,False))
  def run(c):
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):result,raw=ablate(c)
   flat={p:np.asarray(v[0]) for p,v in flatten_dict(raw['intermediates']).items()};n={p[-1]:v for p,v in flat.items() if 'layers_0' in p and p[-1].startswith('audit_')}
   if not n:n={p[-1]:v[0] for p,v in flat.items() if 'layers' in p and p[-1].startswith('audit_') and p[-1]!='audit_layer_index'}
   return result,n
  ref,n=run(ones);_,v=run(ones.at[0,:2].set(jnp.array([0,1])));_,w=run(ones.at[0,2:].set(jnp.array([0,1])));_,b=run(ones.at[0].set(jnp.array([0,1,0,1])));_,z=run(ones.at[0].set(0))
  np.testing.assert_array_equal(v['audit_vector_content'],n['audit_mo']);np.testing.assert_array_equal(v['audit_write_content'],n['audit_write_content']);np.testing.assert_array_equal(w['audit_vector_content'],n['audit_vector_content']);np.testing.assert_array_equal(w['audit_write_content'],n['audit_mo']);np.testing.assert_array_equal(b['audit_vector_content'],b['audit_write_content']);np.testing.assert_array_equal(b['audit_vector_content'],n['audit_mo']);assert np.any(n['audit_mha']!=0);np.testing.assert_array_equal(z['audit_vector_content'],0);np.testing.assert_array_equal(z['audit_write_content'],0);np.testing.assert_array_equal(z['audit_projected_output'],np.broadcast_to(z['audit_output_bias'],z['audit_projected_output'].shape))
  again,_=run(ones);np.testing.assert_array_equal(again[0],ref[0]);print('LATE_MHA_BAM_NATIVE_AND_ROUTE_SCOPES_PASS',full,dtype,'drift',drift,flush=True)
if __name__=='__main__':main()
