"""Read-only, same-seed initial-state QKV/coherence probe; no training mutations.

Run from runtime checkout with pinned CPU env and explicit JAX_PLATFORMS=cpu.
Counterfactual static/dynamic values are captured but NEVER used in propagation.
"""
import argparse,functools,hashlib,json,pathlib,re,tempfile,time
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import pyconfig,max_utils
from layers import attentions,quantizations
from layers.models import Transformer
P='BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdTruePile'
B=P.replace('TruePile','LocalVStaticZeroVOReadNormalTruePile')
ap=argparse.ArgumentParser();ap.add_argument('--cohort',required=True);ap.add_argument('--output',required=True);ap.add_argument('--samples',type=int,default=1);ap.add_argument('--length',type=int,default=4096);ap.add_argument('--cases',default='standard,B');ap.add_argument('--verify-noop',action='store_true')
a=ap.parse_args();out=pathlib.Path(a.output);out.mkdir(parents=True,exist_ok=True)
active=None
counterfactual_enabled=True
local_outputs={}
attention_outputs={}
original_chunk=attentions.BamAttention._query_chunk_op
original_static=attentions.BamAttention._static_column
original_vo=attentions.BamAttention._independent_local_vo
original_content=attentions.BamAttention._attention_write_content
original_out=attentions.BamAttention.out_projection

def chunk(self,q,k,v,segments,window_size,**kwargs):
 result=original_chunk(self,q,k,v,segments,window_size,**kwargs)
 if not self.is_initializing():
  attention_outputs[self.scope.path]=result[0]
  for name,t in [('q',q),('k',k),('v',v),('y',result[0])]:self.sow('intermediates','retention_'+name,t)
 return result

def static(self,m,arm):
 result=original_static(self,m,arm)
 if not self.is_initializing() and arm=='v':
  self.sow('intermediates','retention_static_cf',result)
  # State/token mean energy fraction, reduced before host transfer.
  mf=m.astype(jnp.float32)
  self.sow('intermediates','retention_matrix_moments',jnp.stack((jnp.mean(mf*mf),jnp.mean(jnp.mean(mf,axis=1)**2))))
  if counterfactual_enabled and active!='standard' and active!='B_static_restore':return jnp.zeros_like(result)
 return result

def vo(self,m,x,compressed_M=None):
 result=original_vo(self,m,x,compressed_M)
 if not self.is_initializing():
  if active=='B_boost_V':result=(result[0]*jnp.sqrt(jnp.asarray(12.5,result[0].dtype)),result[1])
  if active=='B_fixed_V_key':
   fixed=jax.random.normal(jax.random.PRNGKey(12345),(1,1,x.shape[-1]),dtype=x.dtype)
   fixed=jnp.broadcast_to(fixed,x.shape)
   vf,_=original_vo(self,m,fixed,compressed_M)
   result=(vf,result[1])
  xf=x.astype(jnp.float32)
  self.sow('intermediates','retention_x_moments',jnp.stack((jnp.mean(xf*xf),jnp.mean(jnp.mean(xf,axis=1)**2))))
  rawkey=jnp.squeeze(self.W_R(x),axis=-2)
  normkey=attentions._transform_bam_read_key(rawkey,self._fetched_arm_ungated).astype(jnp.float32)
  self.sow('intermediates','retention_key_moments',jnp.stack((jnp.mean(normkey*normkey),jnp.mean(jnp.mean(normkey,axis=1)**2))))
  mc=self._compress_m(m).astype(jnp.float32)
  vr=jnp.einsum('btkc,btnc->btnk',mc,normkey)
  mean_v=jnp.mean(vr,axis=1)
  product=jnp.einsum('bkc,bnc->bnk',jnp.mean(mc,axis=1),jnp.mean(normkey,axis=1))
  self.sow('intermediates','retention_mean_read_moments',jnp.stack((jnp.mean(vr*vr),jnp.mean(mean_v*mean_v),jnp.mean(product*product),jnp.mean((mean_v-product)**2),jnp.mean(mean_v*product))))
  # At step0 gate kernels are zero: standard .05*.2 vs B .1*1.
  factor=10. if active=='standard' else 1.
  self.sow('intermediates','retention_dynamic_cf',factor*result[0])
  local_outputs[self.scope.path]=result[1]
  if counterfactual_enabled and active=='standard':return tuple(jnp.zeros_like(t) for t in result)
 return result
def content(self,y):
 if not self.is_initializing() and active=='B_no_Owrite':
  y=y-local_outputs[self.scope.path][..., :self.bam_k]
 if not self.is_initializing() and active=='B_boost_write_context':
  y=y+(jnp.sqrt(jnp.asarray(12.5,y.dtype))-1)*attention_outputs[self.scope.path][..., :self.bam_k]
 return original_content(self,y)
def project_out(self,d,y):
 if not self.is_initializing() and active=='B_boost_vector_context':
  y=y+(jnp.sqrt(jnp.asarray(12.5,y.dtype))-1)*attention_outputs[self.scope.path]
 return original_out(self,d,y)
attentions.BamAttention.out_projection=project_out
attentions.BamAttention._attention_write_content=content
attentions.BamAttention._query_chunk_op=chunk
attentions.BamAttention._static_column=static
attentions.BamAttention._independent_local_vo=vo

def select(coll):
 selected={}
 for p,v in flatten_dict(coll).items():
  if not p[-1].startswith('retention_'):continue
  z=v[0];offset=next((int(re.search(r'(?:local|fetch)_(\d+)$',s).group(1)) for s in p if re.search(r'(?:local|fetch)_(\d+)$',s)),None)
  assert offset is not None,p
  for block in range(z.shape[0]):
   layer=3*block+offset
   if layer in (0,8,17):selected[f'{layer:02d}/'+p[-1].removeprefix('retention_')]=z[block]
 return selected

with tempfile.TemporaryDirectory() as tmp:
 pathlib.Path(tmp,'audit').mkdir()
 configs={};models={};params={};meshes={}
 for label,exp in [('standard',P),('B',B)]:
  c=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class=exp,run_name='audit',enable_checkpointing=False,base_output_directory=tmp+'/',jax_cache_dir='',log_config=False,dataset_type='synthetic',max_target_length=a.length,max_prefill_predict_length=a.length,query_chunk_size=256,per_device_batch_size=1.)
  assert c.bam_pair_scan and c.num_decoder_layers==18
  mesh=jax.sharding.Mesh(max_utils.create_device_mesh(c),c.mesh_axes);model=Transformer(c,mesh,quantizations.configure_quantization(c))
  t=jnp.ones((1,256),jnp.int32)
  def initialize(key):return nn.unbox(model.init({'params':key,'dropout':key,'aqt':key},t,t,t,t)['params'])
  start=time.time()
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):params[label]=jax.jit(initialize)(jax.random.PRNGKey(c.init_weights_seed))
  jax.block_until_ready(params[label]);print('PARAMS_READY',label,time.time()-start,c.init_weights_seed,flush=True)
  configs[label]=c;models[label]=model;meshes[label]=mesh
 native_params=dict(params)
 flat={label:flatten_dict(p) for label,p in params.items()};changed=[]
 for p in flat['standard']:
  if not np.array_equal(np.asarray(flat['standard'][p]),np.asarray(flat['B'][p])):changed.append('/'.join(p))
 assert all('static_v_key' in p or '/W_R/kernel' in p or 'W_lv_gate_b0' in p for p in changed),changed
 (out/'parameter_parity.json').write_text(json.dumps({'changed':changed,'init_seed':configs['B'].init_weights_seed,'configs':{'standard':P,'B':B},'length':a.length,'source_commit':'2fa62ee9b3ad9b20c60c89078e59e957ea387242'},indent=2))
 # Supply both value readouts on each ORIGINAL trajectory. Hooks discard the inactive one.
 for p in flat['standard']:
  if p[-1]=='static_v_key':flat['B'][p]=flat['standard'][p]
  if 'W_R' in p and p[-1]=='kernel':flat['standard'][p]=flat['B'][p]
 params={label:unflatten_dict(p) for label,p in flat.items()}
 cohort=np.load(a.cohort);tokens=cohort['inputs'][:a.samples,:a.length]
 for label in a.cases.split(','):
  source='standard' if label=='standard' else 'B'
  active=label;c=configs[source];model=models[source];mesh=meshes[source]
  def forward(p,t):
   pos=jnp.broadcast_to(jnp.arange(t.shape[1]),t.shape);ones=jnp.ones_like(t)
   _,coll=model.apply({'params':p},t,pos,t,ones,ones,enable_dropout=False,rngs={'aqt':jax.random.PRNGKey(0)},mutable=['intermediates'])
   return select(coll['intermediates'])
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):fn=jax.jit(forward)
  for i,t in enumerate(tokens):
   start=time.time();data=jax.device_get(fn(params[source],jnp.asarray(t[None])))
   np.savez(out/f'{label}-{i:03d}.npz',**{k:np.asarray(v,dtype=np.float32) for k,v in data.items()})
   print('FORWARD_READY',label,i,time.time()-start,hashlib.sha256(t.tobytes()).hexdigest(),flush=True)
   if a.verify_noop and i==0 and label in ('standard','B'):
    counterfactual_enabled=False
    def native_forward(p,t):return forward(p,t)
    with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):native=jax.device_get(jax.jit(native_forward)(native_params[source],jnp.asarray(t[None])))
    for k in data:
     if k.endswith(('/q','/k','/v','/y','/matrix_moments')):np.testing.assert_array_equal(data[k],native[k],err_msg=k)
    counterfactual_enabled=True
    print('NATIVE_TRAJECTORY_EXACT_PARITY',label,flush=True)
