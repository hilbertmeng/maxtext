"""Summarize paired external-weight V-half dose probes; no additive attribution claim."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);ss=json.loads((root/'worker0/scenarios.json').read_text());meta=json.loads((root/'worker0/metadata.json').read_text())
files=sorted(root.glob('worker*/dose_*.npz'));assert len(files)==64,len(files)
loss=[];base=[]
for i,p in enumerate(files):
 assert p.name==f'dose_{i:03d}.npz'
 with np.load(p) as f:
  assert str(f['sequence_hash'])==meta['sequence_hashes'][i]['inputs']
  np.testing.assert_array_equal(f['baseline'],f['ordinary'])
  loss.append(f['loss'].ravel());base.append(f['baseline'].item())
loss=np.stack(loss);base=np.asarray(base);assert np.isfinite(loss).all()
index={(s['scope'],s['bam'],s['alpha'],s['beta']):i for i,s in enumerate(ss)}
for scope in {s['scope'] for s in ss}:np.testing.assert_array_equal(loss[:,index[scope,1.,1.,1.]],base)
np.testing.assert_array_equal(loss[:,index['L00',0.,1.,1.]],base)
def stat(x):return dict(mean=float(x.mean()),ci95=float(1.96*x.std(ddof=1)/np.sqrt(len(x))))
def delta(scope,bam,a,b):return loss[:,index[scope,bam,a,b]]-loss[:,index[scope,bam,1.,1.]]
def interaction(scope,bam,a,b):return delta(scope,bam,a,b)-delta(scope,bam,a,1.)-delta(scope,bam,1.,b)
rows=[]
for s in ss:
 x=loss[:,index[s['scope'],s['bam'],s['alpha'],s['beta']]]
 rows.append({k:v for k,v in s.items() if k!='scales'}|dict(delta_native=stat(x-base),delta_same_bam=stat(delta(s['scope'],s['bam'],s['alpha'],s['beta'])),half_interaction=stat(interaction(s['scope'],s['bam'],s['alpha'],s['beta']))))
summary=dict(n=64,baseline=stat(base),scenarios=rows)
(root/'summary.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(root/'paired_doses.npz',loss=loss,baseline=base)
levels=[0.,.2,.5,.8,1.];colors=['#2679b4','#e57721'];groups=['ordinary_L','early_ordinary_L','late_L']
fig,axs=plt.subplots(2,3,figsize=(14,8),sharex=True)
for row,bam in enumerate([1.,0.]):
 for col,scope in enumerate(groups):
  ax=axs[row,col]
  for half,label in enumerate(['Original V front 32','Original V tail 32']):
   st=[stat(delta(scope,bam,a,1.) if half==0 else delta(scope,bam,1.,a)) for a in levels]
   ax.errorbar(levels,[s['mean'] for s in st],yerr=[s['ci95'] for s in st],marker='o',color=colors[half],label=label)
  ax.axhline(0,color='grey',lw=.7);ax.set_title(scope+' / BAM V '+('on' if bam else 'off'));ax.set_xlabel('Fraction retained');ax.set_ylabel('Paired loss increase (nats/token)');ax.grid(alpha=.2)
axs[0,0].legend();fig.suptitle('BamMediumIndependentLLFMLPPerLayerColOnly / step 13500 / 64 paired sequences\nL0 and L1 excluded; each curve relative to its own BAM-on/off baseline');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(root/f'std_v_dose_curves.{ext}',dpi=170)
plt.close(fig)
ls=[l for l in range(3,24) if l%3!=2];fig,axs=plt.subplots(1,2,figsize=(14,4.8))
for ax,bam in zip(axs,[1.,0.]):
 for half,label in enumerate(['Original V front 32','Original V tail 32']):
  st=[stat(delta(f'L{l:02d}',bam,0.,1.) if half==0 else delta(f'L{l:02d}',bam,1.,0.)) for l in ls]
  ax.errorbar(ls,[s['mean'] for s in st],yerr=[s['ci95'] for s in st],marker='o',color=colors[half],label=label)
 ax.set_xticks(ls);ax.set_xticklabels([f'L{l}' for l in ls],rotation=45);ax.set_title('BAM V '+('on' if bam else 'off'));ax.set_ylabel('Deletion loss increase (nats/token)');ax.grid(alpha=.2);ax.axhline(0,color='gray',lw=.7)
axs[0].legend();fig.suptitle('Original V half deletion, one L layer at a time / 64 paired sequences');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(root/f'std_v_half_deletion_layers.{ext}',dpi=170)
plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,scope in zip(axs,groups):
 z=np.array([[delta(scope,1.,a,b).mean() for b in levels] for a in levels]);im=ax.imshow(z,origin='lower',cmap='magma');ax.set_xticks(range(5),levels);ax.set_yticks(range(5),levels);ax.set_xlabel('Tail fraction retained');ax.set_ylabel('Front fraction retained');ax.set_title(scope)
 for y in range(5):
  for x in range(5):ax.text(x,y,f'{z[y,x]:.3f}',ha='center',va='center',color='white' if z[y,x]<z.max()*.6 else 'black',fontsize=8)
 fig.colorbar(im,ax=ax)
fig.suptitle('BAM V on: full half-by-half dose grid, paired loss increase');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(root/f'std_v_dose_grid.{ext}',dpi=170)

def fmt(s):return f"{s['mean']:+.6f} ±{s['ci95']:.6f}"
lines=['| Scope | BAM V | Front deleted | Tail deleted | Front − tail | Both deleted | Nonadditivity |','|---|---|---:|---:|---:|---:|---:|']
for scope in [f'L{l:02d}' for l in range(24) if l%3!=2]+['all_L']+groups:
 for bam in [1.,0.]:
  a=delta(scope,bam,0.,1.);b=delta(scope,bam,1.,0.)
  lines.append('| '+scope+' | '+('on' if bam else 'off')+' | '+' | '.join(fmt(stat(x)) for x in [a,b,a-b,delta(scope,bam,0.,0.),interaction(scope,bam,0.,0.)])+' |')
(root/'deletion_table.md').write_text('\n'.join(lines)+'\n')
verification=dict(samples=64,scenario_count=len(ss),native_exact=True,hashes_verified=True,finite=True)
# Independent ordinary-forward reproducibility against preceding gradient stage.
gradroot=root.parent/'medium-std-v-grad-13500-0920';errors=[]
for i in range(32):
 with np.load(gradroot/f'worker0/grad_{i:03d}.npz') as f:errors.append(float(base[i]-f['ordinary']))
verification['gradient_stage_baseline_max_difference']=max(abs(x) for x in errors)
(root/'verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification));print('\n'.join(lines[-12:]))
