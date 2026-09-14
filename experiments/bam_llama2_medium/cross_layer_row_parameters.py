"""Read-only CPU partial restore and within-LLF row-projection sharing analysis."""
import argparse, hashlib, json, os, time, itertools, shutil, subprocess
from pathlib import Path
import numpy as np

RUN='BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow'
CHECKPOINT=f'gs://newproject-1-llm_projects_us-east5/log/{RUN}/checkpoints/13500/items'

def restore(out):
 # Reuse existing gcloud credentials when local ADC is not configured. Never print tokens.
 if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS') and not (Path.home()/'.config/gcloud/application_default_credentials.json').exists():
  gcloud=shutil.which('gcloud') or str(Path.home()/'google-cloud-sdk/bin/gcloud')
  account=subprocess.check_output([gcloud,'config','get-value','account'],text=True).strip()
  existing=Path.home()/'.config/gcloud/legacy_credentials'/account/'adc.json'
  if not existing.is_file(): raise RuntimeError('No existing application credentials; set GOOGLE_APPLICATION_CREDENTIALS.')
  os.environ['GOOGLE_APPLICATION_CREDENTIALS']=str(existing)
 import jax
 import orbax.checkpoint as ocp
 from flax.traverse_util import flatten_dict, unflatten_dict
 c=ocp.PyTreeCheckpointer(); meta=c.metadata(CHECKPOINT)
 tree=meta.item_metadata.tree if hasattr(meta,'item_metadata') else meta
 flat=flatten_dict(tree)
 selected={k:v for k,v in flat.items() if k[:2]==('params','params') and (k[-2:]==('W_R','kernel') or k[-2:]==('W_local_packed','kernel') or k[-1]=='W_lv_bias')}
 assert len(selected)==8, [(k,v.shape) for k,v in selected.items()]
 sh=jax.sharding.SingleDeviceSharding(jax.devices('cpu')[0])
 abstract=unflatten_dict({k:jax.ShapeDtypeStruct(v.shape,v.dtype,sharding=sh) for k,v in selected.items()})
 got=c.restore(CHECKPOINT,item=abstract,transforms={},restore_args=ocp.checkpoint_utils.construct_restore_args(abstract))
 flat=flatten_dict(got); arrays={}
 for k,v in flat.items():
  v=np.asarray(v); assert np.isfinite(v).all()
  role=next(s for s in k if s.startswith(('local_','fetch_')))
  arrays[role+'__'+k[-2]+'__'+k[-1]]=v
 np.savez(out/'parameters.npz',**arrays)
 (out/'restore.json').write_text(json.dumps({'run':RUN,'checkpoint':CHECKPOINT,'shapes':{k:list(v.shape) for k,v in arrays.items()},'sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}},indent=2))
 return arrays

def energy_stats(s):
 e=s*s; total=e.sum()
 if total<1e-25: return {'zero':True}
 cum=np.cumsum(e)/total
 return {'norm':float(np.sqrt(total)),'rank90':int(np.searchsorted(cum,.90)+1),'rank95':int(np.searchsorted(cum,.95)+1),'rank99':int(np.searchsorted(cum,.99)+1),'stable_rank':float(total/e[0]),'energy':cum.tolist()}

def pair(a,b,rng):
 na=np.linalg.norm(a); nb=np.linalg.norm(b)
 if min(na,nb)<1e-12: return {'zero':True}
 ua,sa,_=np.linalg.svd(a,full_matrices=False); ub,sb,_=np.linalg.svd(b,full_matrices=False)
 cross=a.T@b
 # Orthogonal output reparameterization is only a geometric oracle, not a legal drop-in key edit.
 nuclear=np.linalg.svd(cross,compute_uv=False).sum()
 ranks=[16,32,64]
 overlap={str(r):float(np.square(ua[:,:r].T@ub[:,:r]).sum()/r) for r in ranks}
 null=[]
 for _ in range(8):
  p=rng.permutation(a.shape[0]); cross_null=a.T@b[p]
  null.append({'cosine':float(np.sum(a*b[p])/(na*nb)), 'overlap64':float(np.square(ua[:,:64].T@ub[p,:64]).sum()/64),'cka':float(np.square(cross_null).sum()/np.sqrt(np.sum(sa**4)*np.sum(sb**4)))})
 return {'cosine':float(np.sum(a*b)/(na*nb)),'pearson':float(np.corrcoef(a.ravel(),b.ravel())[0,1]),'procrustes_cosine':float(nuclear/(na*nb)),'cka_uncentered':float(np.square(cross).sum()/np.sqrt(np.sum(sa**4)*np.sum(sb**4))),'top_input_subspace_overlap':overlap,'permuted_input_null':null,'mean_tie_relative_error':float(np.sqrt((np.square(a-(a+b)/2).sum()+np.square(b-(a+b)/2).sum())/(na*na+nb*nb)))}

def sharing(mats):
 stack=np.concatenate(mats,axis=1)
 u,s,_=np.linalg.svd(stack,full_matrices=False)
 widths=[m.shape[1] for m in mats]; original=sum(m.size for m in mats)
 stats=energy_stats(s)
 if stats.get('zero'): return stats
 ranks=sorted(set([16,32,64,96,128,160,192,256,384,512,614,768,stats['rank90'],stats['rank95'],stats['rank99']]))
 curves=[]
 separate=[np.linalg.svd(m,compute_uv=False) for m in mats]
 for r in ranks:
  if r>len(s): continue
  errs=[float(np.sqrt(max(0,1-np.square(u[:,:r].T@m).sum()/np.square(m).sum()))) if np.linalg.norm(m)>1e-12 else None for m in mats]
  # Same total parameter budget for separate rank factorizations, allocating components optimally.
  budget=r*(stack.shape[0]+stack.shape[1]); cost=[m.shape[0]+m.shape[1] for m in mats]
  assert len(set(cost))==1
  best=0.0
  for dense in itertools.product((False,True),repeat=len(mats)):
   spent=sum(m.size for m,take in zip(mats,dense) if take)
   if spent>budget: continue
   retained=sum(np.sum(v*v) for v,take in zip(separate,dense) if take)
   remaining=[v*v for v,take in zip(separate,dense) if not take]
   e=np.sort(np.concatenate(remaining))[::-1] if remaining else np.array([])
   retained+=e[:(budget-spent)//cost[0]].sum()
   best=max(best,float(retained))
  curves.append({'rank':r,'param_ratio':budget/original,'joint_retained_energy':float(np.sum(s[:r]**2)/np.sum(s*s)),'layer_relative_errors':errs,'independent_svd_energy_same_budget':float(best/sum(np.sum(v*v) for v in separate))})
 return {'spectrum':stats,'original_parameters':original,'breakeven_rank':original/(stack.shape[0]+stack.shape[1]),'curves':curves}

def analyze(arrays,out):
 rng=np.random.default_rng(9876); result={'run':RUN,'checkpoint':CHECKPOINT,'training_commit':'77401da6f83a5aa6ddd61994e028c3c694221518','runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'numpy_version':np.__version__,'seed':9876,'units':[]}
 for unit in range(8):
  o=[arrays[f'{role}__W_R__kernel'][:,unit,:,0,:32].reshape(1024,512).astype('float64') for role in ['local_0','local_1','fetch_2']]
  # q and k: 64 basis + 2 gate + 32 mix = 98 each; V follows at196.
  v=[arrays[f'local_{j}__W_local_packed__kernel'][:,unit,196:452].reshape(1024,4,64)[:,:,:32].reshape(1024,128).astype('float64') for j in (0,1)]
  assert all(arrays[f'local_{j}__W_local_packed__kernel'].shape==(1024,8,612) for j in (0,1))
  row={'unit':unit,'layers':[3*unit+i for i in range(3)],'V':{'individual':[energy_stats(np.linalg.svd(m,compute_uv=False)) for m in v],'pair':pair(*v,rng),'shared':sharing(v)},'O':{'individual':[energy_stats(np.linalg.svd(m,compute_uv=False)) for m in o],'pairs':{f'{i}-{j}':pair(o[i],o[j],rng) for i,j in [(0,1),(0,2),(1,2)]},'shared_LL':sharing(o[:2]),'shared_LLF':sharing(o)}}
  for name,mats in [('V_LL',v),('O_LL',o[:2]),('O_LLF',o)]:
   normalized=[m/np.linalg.norm(m) for m in mats if np.linalg.norm(m)>1e-12]
   row.setdefault('balanced',{})[name]=sharing(normalized)
  result['units'].append(row); print('finished unit',unit,flush=True)
 (out/'analysis.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 start=time.monotonic(); f=a.output/'parameters.npz'
 arrays=dict(np.load(f)) if f.exists() else restore(a.output)
 analyze(arrays,a.output)
 print('elapsed',time.monotonic()-start,flush=True)
