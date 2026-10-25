"""CPU identity and intervention-scope checks for pre/post FFN branch mediation."""
import os,tempfile,sys
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import exp,pyconfig,max_utils,precise_ce
from layers.models import Transformer
import branch_mediation_capture as hook

def main():
 kind=os.environ.get('TEST_KIND','py');pallas=kind=='pallas';py=kind!='xl';n=5 if pallas else (8 if py else 7);B=1 if pallas else 2;T=128 if pallas else 16
 parent=exp.BamPallasPythiaR768TinyPallasV8TTest if pallas else exp.BamPallasPythiaR768TinyXlaTest if py else exp.BamPallasFullMTinyXlaTest
 attrs=dict(base_num_decoder_layers=n,bam_layer_modes=['local_qk+local_v+local_o']*n,vocab_size=128,dtype=os.environ.get('TEST_DTYPE','float32'),weight_dtype='float32',bam_pallas_core=pallas,bam_pallas_interpret=pallas,bam_pallas_body='v8t',bam_pallas_fused_inputs=pallas,bam_m_shift_mode='read',bam_m_shift_init=.5,bam_mlp_write_every=int(os.environ.get('TEST_WRITES','0')),max_target_length=T,max_prefill_predict_length=T,target_train_batch_size=B,per_device_batch_size=float(B),query_chunk_size=T)
 exp.BranchMediationTiny=type('BranchMediationTiny',(parent,),attrs);precise_ce.install()
 with tempfile.TemporaryDirectory() as out:
  os.mkdir(out+'/test');cfg=pyconfig.initialize(['test','MaxText/configs/base.yml'],exp_class='BranchMediationTiny',run_name='test',base_output_directory=out+'/',dataset_type='synthetic',enable_checkpointing=False,jax_cache_dir='',log_config=False,record_internal_nn_metrics=False)
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(cfg),cfg.mesh_axes);model=Transformer(cfg,mesh,quant=None);tok=jax.random.randint(jax.random.key(1),(B,T),0,128);pos=jnp.broadcast_to(jnp.arange(T),tok.shape);seg=jnp.ones_like(tok);rng={'params':jax.random.key(3),'dropout':jax.random.key(4),'aqt':jax.random.key(5)};kw=dict(decoder_target_tokens=tok,decoder_target_mask=seg,decoder_segment_ids=seg,enable_dropout=False)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):params=model.init(rng,tok,pos,**kw)
  params=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,params,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
  flat=flatten_dict(params)
  if py:
   bias=[p for p in flat if '/self_attention/out/bias' in '/'.join(p)];assert bias
   for p in bias:flat[p]=jnp.ones_like(flat[p])*(.001 if cfg.dtype==jnp.bfloat16 else .17)
   params=unflatten_dict(flat)
  def native(weights):return model.apply(weights,tok,pos,rngs=rng,**kw)[0]
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):baseline=jax.jit(native)(params)
  hook.install();import prewo_observer as observer
  hook.OBSERVE=True;hook.RETAIN_LOCAL=True
  def probe(weights,coeff,live):
   hook.CONTROLS=dict(coefficients=coeff,live=live);observer.ACTIVE=True
   try:
    result,raw=model.apply(weights,tok,pos,rngs=rng,mutable=['intermediates'],**kw)
    return result[0],raw
   finally:hook.CONTROLS=None;observer.ACTIVE=False
  fn=jax.jit(probe);zero=jnp.zeros((n,3),jnp.float32)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):neutral,neutral_raw=fn(params,zero,jnp.asarray(False))
  native_error=np.asarray(neutral)-np.asarray(baseline)
  print('NEUTRAL_NATIVE_NUMERICAL_CE_DRIFT',kind,str(cfg.dtype),'max_per_sequence_mean',float(np.max(np.abs(native_error.mean(1)))),'max_per_token',float(np.max(np.abs(native_error))),flush=True)
  if cfg.dtype==jnp.bfloat16:assert np.max(np.abs(native_error.mean(1)))<=.005
  else:np.testing.assert_allclose(neutral,baseline,atol=2e-6,rtol=1e-6)
  for p,values in flatten_dict(jax.device_get(neutral_raw)).items():
   if p[-1]=='stream_mediation_delta':
    for value in values:np.testing.assert_array_equal(value,0.)
  print('NATIVE_ZERO_CONTROL_AND_NONZERO_OUTPUT_BIAS_IDENTITY_VERIFIED',kind,flush=True)
  # Output bias belongs to neither readout branch: with W_o=0 all interventions vanish.
  no_out=unflatten_dict({p:jnp.zeros_like(v) if '/'.join(p).endswith('/self_attention/out/kernel') else v for p,v in flatten_dict(params).items()})
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):a,_=fn(no_out,zero,jnp.asarray(False));b,raw=fn(no_out,zero.at[1,1].set(.1),jnp.asarray(True))
  np.testing.assert_array_equal(a,b)
  print('ZERO_OUTPUT_KERNEL_EXCLUDES_NONZERO_OUTPUT_BIAS_FROM_BRANCH',kind,flush=True)
  # Positive control: a learned FFN responds, whereas the current M write is untouched
  # when the fixture disables MLP writes. Inspect the exact selected scanned block.
  target=next(i for i in range(n-1) if hook.layer_writes_m(cfg,i)) if cfg.bam_mlp_write_every else 1
  coeff=zero.at[target,0].set(1.0 if cfg.dtype==jnp.bfloat16 else .1)
  with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):live_loss,live_raw=fn(params,coeff,jnp.asarray(True));frozen_loss,frozen_raw=fn(params,coeff,jnp.asarray(False))
  def selected(raw,name):
   flat=flatten_dict(jax.device_get(raw));found=[]
   for path,values in flat.items():
    if path[-1]!='stream_mediation_layer_index':continue
    idx=np.asarray(values[0]);matches=np.where(idx.reshape(-1)==target)[0]
    for i in matches:
     array=np.asarray(flat[path[:-1]+(name,)][0]);found.append(array if idx.ndim==0 else array[i])
   assert len(found)==1,(target,len(found));return found[0]
  frozen_M=selected(frozen_raw,'stream_mediation_M');neutral_M=selected(neutral_raw,'stream_mediation_M');live_M=selected(live_raw,'stream_mediation_M')
  np.testing.assert_array_equal(frozen_M,neutral_M)
  if cfg.bam_mlp_write_every:assert np.any(live_M!=frozen_M)
  else:np.testing.assert_array_equal(live_M,frozen_M)
  local_response=float(np.max(np.abs(selected(live_raw,'stream_mediation_before_exit')-selected(frozen_raw,'stream_mediation_before_exit'))))
  assert local_response>1e-9,local_response
  print('LEARNED_FFN_MEDIATION_AND_FROZEN_M_WRITE_SCOPE_VERIFIED',kind,'writes',cfg.bam_mlp_write_every,'response',local_response,flush=True)
  # A constant ordinary FFN cannot mediate: live and frozen must agree.
  zero_mlp={p:jnp.zeros_like(v) if 'mlp' in p else v for p,v in flatten_dict(params).items()};zero_mlp=unflatten_dict(zero_mlp)
  for arm in range(3):
   coeff=zero.at[1,arm].set(.1)
   with mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):a,_=fn(zero_mlp,coeff,jnp.asarray(True));b,raw=fn(zero_mlp,coeff,jnp.asarray(False))
   np.testing.assert_array_equal(a,b)
  print('CONSTANT_FFN_ZERO_WRITE_DATA_LIVE_EQUALS_FROZEN_ALL3BRANCHES',kind,flush=True)
  # Check normalization geometry independently of an experiment's learned weights.
  z=jnp.array([[[2.,-1.,.5,3.]]]);shift=jnp.ones_like(z);v=hook.tangent_moments(z,shift)
  np.testing.assert_allclose(v[...,3],0.,atol=1e-6);radial=hook.tangent_moments(z,z);np.testing.assert_allclose(radial[...,2:4],0.,atol=1e-6)
  print('LAYERNORM_COMMON_MODE_AND_NORMALIZATION_RADIAL_CONTROLS_VERIFIED',kind,flush=True)
  gamma=jnp.array([.7,1.2,1.1,.9]);direction=jnp.array([[[.1,.3,-.2,.4]]]);epsilon=.03
  for center in [False,True]:
   def normalize(x):
    if center:x=x-x.mean(-1,keepdims=True)
    return gamma*x*jax.lax.rsqrt(jnp.mean(x*x,-1,keepdims=True)+epsilon)
   _,expected=jax.jvp(normalize,(z,),(direction,));actual=hook.normalization_response(z,direction,epsilon,center,gamma);np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=1e-7)
  assert np.linalg.norm(hook.normalization_response(z,z,epsilon,False,gamma))>0
  print('FINITE_EPSILON_AND_LEARNED_SCALE_NORM_JVP_MATCHES_AUTODIFF',kind,flush=True)

if __name__=='__main__':main()
