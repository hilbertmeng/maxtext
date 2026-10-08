import argparse,json,pathlib,time
import jax,jax.numpy as jnp,numpy as np
ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--output',required=True);ap.add_argument('--labels',default='standard,B');a=ap.parse_args();root=pathlib.Path(a.root)
@jax.jit
def measure(q,k,v):
 # q is already scaled by sqrt(head_dim), positions already rotated.
 t=q.shape[1];n=q.shape[2];ds=q.shape[3];bounds=(0,t//4,t//2,3*t//4,t)
 v=v.astype(jnp.float32);q=q.astype(jnp.float32);k=k.astype(jnp.float32)
 ve=jnp.mean(v*v);mu=jnp.mean(v,axis=1,keepdims=True);mu2=jnp.mean(mu*mu)
 ys=[];stats=[]
 for start in range(0,t,256):
  stop=start+256;logits=jnp.einsum('bqnd,bsnd->bnqs',q[:,start:stop],k[:,:stop]);mask=jnp.arange(stop)[None,:]<=jnp.arange(start,stop)[:,None]
  alpha=jax.nn.softmax(jnp.where(mask[None,None],logits,-1e30),axis=-1)
  y=jnp.einsum('bnqs,bsnd->bqnd',alpha,v[:,:stop]);ys.append(y)
  # All below scalar averages across batch, heads, queries; normalize per coordinate.
  diag=jnp.einsum('bnqs,bsn->bqn',alpha*alpha,jnp.mean(v[:,:stop]**2,axis=-1))
  stats.append(jnp.stack((jnp.mean(y*y),jnp.mean(diag),jnp.mean(jnp.sum(alpha*alpha,axis=-1)),jnp.mean(-jnp.sum(alpha*jnp.log(jnp.maximum(alpha,1e-30)),axis=-1)),jnp.mean(jnp.max(alpha,axis=-1)),jnp.mean((y-mu)**2))))
 y=jnp.concatenate(ys,axis=1);st=jnp.stack(stats)
 return jnp.concatenate((jnp.asarray([ve,mu2]),jnp.mean(st,axis=0))),y
rows=[]
labels=a.labels.split(',')
for i,f in enumerate(sorted(root.glob(labels[0]+'-*.npz'))):
 index=f.stem.split('-')[-1];files={label:root/f'{label}-{index}.npz' for label in labels}
 if not all(x.exists() for x in files.values()):continue
 packs={label:np.load(x) for label,x in files.items()}
 for layer in (0,8,17):
  p=f'{layer:02d}/';records={}
  for trajectory,z in packs.items():
   for valuekind in ('v','static_cf','dynamic_cf'):
    s,y=jax.device_get(measure(z[p+'q'],z[p+'k'],z[p+valuekind]))
    ve,mu2,ye,diag,a2,ent,amax,centeredye=map(float,s)
    records[trajectory+'/'+valuekind]={'v_rms':ve**.5,'retention':(ye/ve)**.5,'common_energy_fraction':mu2/ve,'diagonal_energy_fraction':diag/ve,'cross_energy_fraction':(ye-diag)/ve,'alpha_sq_sum':a2,'entropy':ent,'max_alpha':amax,'centered_output_energy_fraction':centeredye/ve}
    if valuekind=='v':records[trajectory+'/'+valuekind]['recomputed_y_relative_rms_error']=float(np.sqrt(np.mean((y-z[p+'y'])**2)/np.mean(z[p+'y']**2)))
  for weighttraj,valuetraj in ([('standard','B'),('B','standard')] if set(labels)=={'standard','B'} else []):
   z=packs[weighttraj];v=packs[valuetraj][p+'v'];s,_=jax.device_get(measure(z[p+'q'],z[p+'k'],v));records[weighttraj+'_weights/'+valuetraj+'_values']={'retention':float((s[2]/s[0])**.5)}
  # Matrix shared token component itself, before either read.
  for traj,z in packs.items():
   mom=z[p+'matrix_moments'];records[traj+'/M']={'rms':float(np.sqrt(mom[0])),'common_energy_fraction':float(mom[1]/mom[0])}
  rows.append({'sample':int(index),'layer':layer,'measurements':records})
  print('MEASURED',index,layer,{l:records[l+'/v']['retention'] for l in labels},flush=True)
 aggregate={}
 for layer in (0,8,17):
  sel=[r for r in rows if r['layer']==layer];aggregate[str(layer)]={k:{m:float(np.mean([r['measurements'][k][m] for r in sel])) for m in sel[0]['measurements'][k]} for k in sel[0]['measurements']}
 pathlib.Path(a.output).write_text(json.dumps({'n':len(rows)//3,'definitions':'Retention=sqrt(mean(AV squared)/mean(V squared)); common=mean_over_tokens(V) squared / mean(V squared). Counterfactual keys are identical across two trajectories. No propagation interventions.','aggregate':aggregate,'per_sequence':rows},indent=2))
