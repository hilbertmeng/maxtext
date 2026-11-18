"""Fixed native L5/L6 boundary VJPs, separating read paths and layer output adjoints.

One local device per worker; no whole-network replay or checkpoint writes.
Never interpret path measurements unless forward, boundary and vector closures pass.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path

def route_tangent(value,mode,slot):
 import jax,jax.numpy as jnp
 allow=(mode==0)|((mode==1)&(slot<2))|((mode==2)&(slot==2))|(((mode==3)|(mode>=5))&(slot==3))
 return jnp.where(allow,value,jax.lax.stop_gradient(value))

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime-dir',required=True);p.add_argument('--runtime-commit',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--tensor-base',required=True);p.add_argument('--cohort',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 assert subprocess.check_output(['git','-C',a.runtime_dir,'rev-parse','HEAD'],text=True).strip()==a.runtime_commit
 os.chdir(a.runtime_dir);sys.path.insert(0,str(Path(a.runtime_dir)/'MaxText'))
 import jax,jax.numpy as jnp,numpy as np,tensorflow as tf,orbax.checkpoint as ocp
 from flax import linen as nn
 from flax.core import unfreeze
 import pyconfig
 from layers import fusion,linears,normalizations,attentions,bam_pallas
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 (out/'single-layer-attention-paths').mkdir(exist_ok=True)
 cfg=pyconfig.initialize([None,'MaxText/configs/base.yml'],exp_class='BamXLPropV5FastKernelFullRemat',run_name='single-layer-attention-paths',base_output_directory=str(out)+'/',enable_checkpointing=False,only_eval=True,dataset_type='synthetic',target_train_batch_size=None,per_device_batch_size=1.,record_internal_nn_metrics=False,record_training_health_metrics=False,jax_cache_dir='',log_config=False,skip_jax_distributed_system=True)
 device=jax.local_devices()[0]
 mesh=jax.sharding.Mesh(np.asarray([device]).reshape((1,)*len(cfg.mesh_axes)),cfg.mesh_axes)
 single=jax.sharding.SingleDeviceSharding(device)
 cp=ocp.PyTreeCheckpointer();meta=cp.metadata(a.checkpoint).item_metadata.tree['params']['params']['decoder']['layers']['block']
 abstract=jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype,sharding=single),meta)
 target={'params':{'params':{'decoder':{'layers':{'block':abstract}}}}}
 restored=cp.restore(a.checkpoint,item=target,transforms={},restore_args=ocp.checkpoint_utils.construct_restore_args(target));cp.close()
 block=unfreeze(nn.unbox(restored['params']['params']['decoder']['layers']['block']));del restored,target,abstract
 with tf.io.gfile.GFile(a.cohort,'rb') as f:
  with np.load(f) as z:data={k:jax.device_put(jnp.asarray(z[k]),device) for k in z.files}
 report=dict(runtime=a.runtime_commit,checkpoint=a.checkpoint,scope='Single loss seq9,target1211 at original native35636 fixed states/cotangents. Local conditional attribution, not fullbatch causal percentages.',gates=dict(boundary_relative_error=.015,probe_vs_baseline_relative_error=.005,path_vector_closure=.01),layers=[])
 original_read=bam_pallas.read;original_splash=attentions._bam_splash_attention
 active=None;stats=None;templates=None;mode=None;instrument=False
 def rms(v):
  f=v.astype(jnp.float32);return jnp.sqrt(jnp.mean(f*f))
 def tap(name,value):
  if templates is not None:
   templates[name]=jax.ShapeDtypeStruct(value.shape,value.dtype)
   return value
  return value+active[name] if instrument else value
 def read(*args,**kw):
  if instrument:
   def keep(v,flag):return jnp.where(flag,v,jax.lax.stop_gradient(v))
   original_args=args;args=list(args);base=mode<5
   for i,v in enumerate(args):args[i]=keep(v,base|((mode==8)&(i==0))|((mode==9)&(i==1))|((mode==5)&(i==6))|((mode==6)&(i==7)))
   kw=dict(kw)
   if kw.get('static_gates') is not None:kw['static_gates']=tuple(keep(v,base|((mode==7)&(i==3))) for i,v in enumerate(kw['static_gates']))
   if kw.get('shift') is not None:kw['shift']=keep(kw['shift'],base|(mode==8))
   if kw.get('keys_cnt') is not None:kw['keys_cnt']=keep(kw['keys_cnt'],base|(mode==5))
   for name in ('q_dynamic_scale','k_dynamic_scale','v_dynamic_scale','o_dynamic_scale'):
    if kw.get(name) is not None:kw[name]=keep(kw[name],base)
  result=original_read(*args,**kw)
  if not instrument:return result
  result=list(result)
  for i,name in enumerate(('read_Q','read_K','read_V','read_O')):
   v=tap(name,result[i])
   result[i]=route_tangent(v,mode,i)
  return tuple(result)
 def splash(*args,**kw):return tap('attention_context',original_splash(*args,**kw)) if instrument else original_splash(*args,**kw)
 bam_pallas.read=read;attentions._bam_splash_attention=splash
 def intercept(next_fun,args,kw,ctx):
  if not instrument:return next_fun(*args,**kw)
  mod=ctx.module;method=ctx.method_name;xs=list(args);tag=None
  if isinstance(mod,attentions.BamAttention) and method=='__call__':
   if xs:xs[0]=tap('attention_input',xs[0])
   else:kw=dict(kw);kw['inputs_q']=tap('attention_input',kw['inputs_q'])
   result=next_fun(*xs,**kw)
   result=list(result);result[0]=tap('attention_output',result[0]);return tuple(result)
  if isinstance(mod,attentions.BamAttention) and method=='out_projection':
   xs[1]=tap('out_head',xs[1]);return next_fun(*xs,**kw)
  if method=='__call__':
   if isinstance(mod,normalizations.RMSNorm) and mod.name=='post_self_attention_layer_norm':tag='mlp_norm'
   elif isinstance(mod,linears.MlpBlock) and mod.name in ('mlp','memory_mlp'):tag=mod.name
  if tag:
   xs[0]=tap(tag+'_input',xs[0]);return tap(tag+'_output',next_fun(*xs,**kw))
  return next_fun(*args,**kw)
 def load(base,name):
  import ml_dtypes
  with tf.io.gfile.GFile(base+'/'+name+'.npy','rb') as f:v=np.load(f)
  if v.dtype.kind=='V':v=v.view(ml_dtypes.bfloat16)
  return jax.device_put(jnp.asarray(v,jnp.bfloat16),device)
 def hr(v):return float(np.sqrt(np.mean(np.asarray(v,np.float64)**2)))
 def rel(v,w):return hr(np.asarray(v,np.float64)-np.asarray(w,np.float64))/max(hr(w),1e-30)
 def selected(params_grad):
  sa=params_grad['self_attention'];return {n:sa[n]['kernel'] if isinstance(sa[n],dict) and 'kernel' in sa[n] else sa[n] for n in ('out','static_o_key','full_m_read_down')}
 def summarize_gradient(g):return {k:dict(norm=float(np.linalg.norm(np.asarray(v,np.float64)))) for k,v in g.items()}
 def save(): (out/'attention-paths.json').write_text(json.dumps(report,indent=2)+'\n')
 for layer in (5,6):
  base=a.tensor_base.rstrip('/')+'/layer-%02d'%layer
  with tf.io.gfile.GFile(base+'/tensor-fidelity.json') as f:fidelity=json.load(f)
  assert fidelity['passed'] and fidelity['global_sequence']==9 and fidelity['target_token']==1211 and fidelity['target_layer']==layer
  fwd={n:load(base,'forward-state-'+n) for n in ('residual','M')};bwd={n:load(base,'backward-state-'+n) for n in ('residual','M')}
  x,M=fwd['residual'][0],fwd['M'][0];gx,gm=bwd['residual'][1],bwd['M'][1]
  params=jax.tree.map(lambda v:jnp.take(v,layer,axis=1),block)
  def objective(params,x,M,probes,gx,gm,read_mode):
   nonlocal active,stats,mode
   active=probes;mode=read_mode;stats={}
   module=fusion.SubDecoderLayer(cfg,mesh,None,cfg.max_target_length,0)
   def apply():return module.apply({'params':params},x,data['inputs_segmentation'],data['inputs_position'],data['inputs'],None,True,'train',None,M_in=M,layer_index=jnp.int32(layer),rngs={'dropout':jax.random.PRNGKey(0),'params':jax.random.PRNGKey(0)})
   if instrument:
    with nn.intercept_methods(intercept):y,m=apply()
   else:y,m=apply()
   return jnp.sum(y.astype(jnp.float32)*gx.astype(jnp.float32))+jnp.sum(m.astype(jnp.float32)*gm.astype(jnp.float32)),(y,m)
  def measure(params,x,M,probes,gx,gm,read_mode):
   (_,ym),(gp,dx,dm,dp)=jax.value_and_grad(objective,argnums=(0,1,2,3),has_aux=True)(params,x,M,probes,gx,gm,read_mode)
   return ym,dx,dm,dp,selected(gp)
  entry=dict(layer=layer,source_fidelity=fidelity,cases=[]);report['layers'].append(entry)
  with jax.default_device(device),mesh,nn.partitioning.axis_rules(cfg.logical_axis_rules):
   instrument=False;templates=None
   baseline=jax.jit(measure)(params,x,M,{},gx,gm,jnp.int32(0));baseline=jax.device_get(baseline)
   y,m=baseline[0];entry['baseline_errors']=dict(output_residual=rel(y,fwd['residual'][1]),output_M=rel(m,fwd['M'][1]),input_residual_cotangent=rel(baseline[1],bwd['residual'][0]),input_M_cotangent=rel(baseline[2],bwd['M'][0]))
   save();assert max(entry['baseline_errors'].values())<.015,('native boundary fidelity',layer,entry['baseline_errors'])
   instrument=True;templates={}
   # Discover concrete probe shapes without executing a changed whole network.
   jax.eval_shape(objective,params,x,M,{},gx,gm,jnp.int32(0))
   probes={k:jax.device_put(jnp.zeros(v.shape,v.dtype),device) for k,v in templates.items()};templates=None
   compiled=jax.jit(measure)
   first= jax.device_get(compiled(params,x,M,probes,gx,gm,jnp.int32(0)))
   entry['instrument_errors']=dict(forward_residual=rel(first[0][0],baseline[0][0]),forward_M=rel(first[0][1],baseline[0][1]),dx=rel(first[1],baseline[1]),dm=rel(first[2],baseline[2]),**{'parameter_'+k:rel(first[4][k],baseline[4][k]) for k in first[4]})
   save();assert max(entry['instrument_errors'].values())<.005,('probe fidelity',layer,entry['instrument_errors'])
   for label,gr,gg in [('full',gx,gm),('residual_output_only',gx,jnp.zeros_like(gm)),('M_output_only',jnp.zeros_like(gx),gm)]:
    arms=[]
    for rm in range(10 if label=='full' else 5):
     values=first if label=='full' and rm==0 else jax.device_get(compiled(params,x,M,probes,gr,gg,jnp.int32(rm)))
     arms.append(values)
    full=arms[0];dp=full[3]
    if layer==5 and label=='full':
     expected={'out':.10853219306582995,'static_o_key':.10329590980683492,'full_m_read_down':.09736670845214102}
     entry['native_singleton_parameter_norm_relative_errors']={k:abs(float(np.linalg.norm(np.asarray(full[4][k],np.float64)))/523956/expected[k]-1) for k in expected}
     save();assert max(entry['native_singleton_parameter_norm_relative_errors'].values())<.02,('native parameter gradient fidelity',entry['native_singleton_parameter_norm_relative_errors'])
    assert rel(dp['read_O'],dp['attention_context'])<.005
    row=dict(output_cotangent=label,gradients=dict(residual_input_rms=hr(full[1]),residual_output_rms=hr(gr),M_input_rms=hr(full[2]),M_output_rms=hr(gg)),block_gains={n:hr(dp[n+'_input'])/max(hr(dp[n+'_output']),1e-30) for n in ('mlp','mlp_norm','memory_mlp')},W_O_input_gain=hr(dp['out_head'])/max(hr(dp['attention_output']),1e-30),head_cotangents=dict(total_rms=hr(dp['read_O']),via_W_O_rms=hr(dp['out_head']),direct_attention_M_write_rms=hr(np.asarray(dp['read_O'],np.float64)-np.asarray(dp['out_head'],np.float64))),parameter_norms=summarize_gradient(full[4]),attention_input_over_residual_output_cotangent=hr(dp['attention_input'])/max(hr(dp['attention_output']),1e-30),read_path_components=[])
    entry['cases'].append(row)
    components=[]
    for rm,name in [(1,'QK'),(2,'V'),(3,'localO')]:
     pg={k:np.asarray(arms[rm][4][k],np.float64)-np.asarray(arms[4][4][k],np.float64) for k in full[4]}
     ax=np.asarray(arms[rm][3]['attention_input'],np.float64)-np.asarray(arms[4][3]['attention_input'],np.float64)
     components.append((pg,ax))
     incoming=(np.sqrt(sum(np.linalg.norm(np.asarray(dp[t],np.float64))**2 for t in ('read_Q','read_K'))) if name=='QK' else np.linalg.norm(np.asarray(dp['read_'+('V' if name=='V' else 'O')],np.float64)))
     row['read_path_components'].append(dict(name=name,attention_input_gradient_rms=hr(ax),conditional_input_L2_gain=float(np.linalg.norm(ax)/max(incoming,1e-30)),parameter_norms=summarize_gradient(pg)))
    base_pg=arms[4][4];base_ax=arms[4][3]['attention_input'];recon_pg={k:sum(c[0][k] for c in components)+np.asarray(base_pg[k],np.float64) for k in full[4]};recon_ax=sum(c[1] for c in components)+np.asarray(base_ax,np.float64)
    row['attention_context_conditional_input_L2_gain']=float(np.linalg.norm(components[0][1]+components[1][1])/max(np.linalg.norm(np.asarray(dp['attention_context'],np.float64)),1e-30))
    row['head_sum_conditional_input_L2_gain']=float(np.linalg.norm(sum(v[1] for v in components))/max(np.linalg.norm(np.asarray(dp['read_O'],np.float64)),1e-30))
    if label=='full':
     sub=[]
     for rm,name in [(5,'dynamic_address_keynorm_shared_GELU'),(6,'dynamic_read_gate'),(7,'static_read_gate'),(8,'M_shift_and_M_norm'),(9,'static_address_parameter')]:
      ax=np.asarray(arms[rm][3]['attention_input'],np.float64)-np.asarray(arms[4][3]['attention_input'],np.float64)
      pg={k:np.asarray(arms[rm][4][k],np.float64)-np.asarray(arms[4][4][k],np.float64) for k in full[4]}
      sub.append((ax,pg))
      row.setdefault('localO_subpaths',[]).append(dict(name=name,input_gradient_rms=hr(ax),gain_relative_to_localO_cotangent=float(np.linalg.norm(ax)/max(np.linalg.norm(np.asarray(dp['read_O'],np.float64)),1e-30)),parameter_norms=summarize_gradient(pg)))
     row['localO_subpath_closure']=dict(input_gradient=rel(sum(v[0] for v in sub),components[2][1]),**{k:rel(sum(v[1][k] for v in sub),components[2][0][k]) for k in ('static_o_key','full_m_read_down')})
     save();assert max(row['localO_subpath_closure'].values())<.01,('O factor closure',row['localO_subpath_closure'])
    row['remaining_direct_paths']=dict(attention_input_gradient_rms=hr(base_ax),parameter_norms=summarize_gradient(base_pg))
    row['path_closure']=dict(attention_input=rel(recon_ax,dp['attention_input']),**{k:rel(recon_pg[k],full[4][k]) for k in full[4]})
    save();assert max(row['path_closure'].values())<.01,('path closure',layer,label,row['path_closure'])
    if label=='full':full_case=full
    elif label=='residual_output_only':res_case=full
    else:
     tot,res=full_case,res_case
     closure=dict(dx=rel(np.asarray(res[1],np.float64)+np.asarray(full[1],np.float64),tot[1]),dm=rel(np.asarray(res[2],np.float64)+np.asarray(full[2],np.float64),tot[2]),**{k:rel(np.asarray(res[4][k],np.float64)+np.asarray(full[4][k],np.float64),tot[4][k]) for k in full[4]})
     entry['output_cotangent_closure']=closure;assert max(closure.values())<.01,('output cotangent closure',closure)
    print('PATH_CASE',layer,label,json.dumps(row),flush=True)
   entry['complete']=True;save()
 report['complete']=True;save();print('ATTENTION_PATHS_COMPLETE',flush=True)
if __name__=='__main__':main()
