"""Review artifacts for gate-geometry discovery; all population plots are dual-gate filtered."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
root=Path(sys.argv[1]);dest=root/'figures';dest.mkdir(exist_ok=True);ad=root/'analysis_gate010'
rows=[json.loads((ad/f'layer_{l:02d}.json').read_text()) for l in range(24)];data=[np.load(ad/f'layer_{l:02d}.npz') for l in range(24)];names=data[0]['feature_names'].tolist();F=len(names)
coverage=np.stack([r['eligible_tokens_per_head'] for r in rows]);seqcov=np.stack([r['eligible_sequences_per_head'] for r in rows]);disc=np.stack([r['eligible_sequences_discovery'] for r in rows]);val=np.stack([r['eligible_sequences_validation'] for r in rows]);eligible=(disc>=16)&(val>=16)&(coverage>=2048)
cor=np.stack([d['correlations'] for d in data]);strong=[]
for l in range(1,24):
 for h in range(16):
  if not eligible[l,h]:continue
  for j,name in enumerate(names):
   if name=='cos_bam_self__local_o':continue
   a,b=cor[l,:,h,j]
   if np.isfinite(a+b):strong.append(dict(layer=l,head=h,feature=name,j=j,discovery=float(a),validation=float(b),score=float(abs(a)),replicated=bool(a*b>0 and abs(b)>.2)))
strong.sort(key=lambda r:-r['score'])
(root/'candidate_relations.json').write_text(json.dumps(strong,indent=2))
print('COVERAGE eligible_heads',int(eligible.sum()),'of',23*16,'samples',rows[0]['n'])
print('TOP RELATIONS',json.dumps(strong[:20],indent=2))
print('HELDOUT R2',[(r['layer'],round(r['heldout_incremental_r2'],3)) for r in rows])
fig,axs=plt.subplots(1,2,figsize=(13,6));im=axs[0].imshow(seqcov,aspect='auto',vmin=0,vmax=rows[0]['n'],cmap='viridis');fig.colorbar(im,ax=axs[0],label='Independent sequences');im=axs[1].imshow(np.log10(np.maximum(coverage,1)),aspect='auto',cmap='viridis');fig.colorbar(im,ax=axs[1],label='log10 selected tokens')
for ax in axs:ax.set_xlabel('Head');ax.set_ylabel('Layer');ax.set_xticks(range(16));ax.set_yticks(range(24))
fig.suptitle('Both local-O and write sigmoid gates >=0.1; nonzero O');fig.tight_layout();fig.savefig(dest/'coverage.png',dpi=170);plt.close(fig)
# Complete population histograms, pooling counts (not per-head medians).
fig,axs=plt.subplots(1,3,figsize=(15,4.4))
for j in range(4):
 hist=sum(d['hist'][:,j].sum(axis=(0,2)) for d in data[1:]);edges=data[0]['edges'][j];axs[0].step(edges[1:],np.cumsum(hist)/max(hist.sum(),1),where='post',label=names[j].removeprefix('lognorm_'))
 hist=sum(d['hist'][:,j+4].sum(axis=(0,2)) for d in data[1:]);edges=data[0]['edges'][j+4];axs[1].step(edges[1:],np.cumsum(hist)/max(hist.sum(),1),where='post',label=names[j+4].removeprefix('share_'))
for j in range(8,14):
 hist=sum(d['hist'][:,j].sum(axis=(0,2)) for d in data[1:]);edges=data[0]['edges'][j];axs[2].step(edges[1:],np.cumsum(hist)/max(hist.sum(),1),where='post',label=names[j].removeprefix('cos_'))
for ax in axs:ax.set_ylabel('Empirical CDF (binned)');ax.grid(alpha=.25);ax.legend(fontsize=7)
axs[0].set_xlabel('log10 norm');axs[0].set_xlim(-5,2);axs[1].set_xlabel('Norm share');axs[2].set_xlabel('Cosine');fig.suptitle('Four-part geometry distributions / dual gates >=0.1 / L1-L23');fig.tight_layout();fig.savefig(dest/'component_distributions.png',dpi=180);plt.close(fig)
# Signed correlations per head; stable relations can have opposite signs across heads.
fig,axs=plt.subplots(1,2,figsize=(15,9));mat=cor.transpose(1,0,2,3).reshape(2,384,F)
for ax,k,title in zip(axs,[0,1],['Discovery sequences 0-31','Held-out sequences 32+']):
 im=ax.imshow(mat[k],aspect='auto',vmin=-.7,vmax=.7,cmap='RdBu_r');ax.set_xticks(range(F),names,rotation=90,fontsize=8);ax.set_yticks(np.arange(0,384,16),range(24));ax.set_ylabel('Layer (16 heads each)');ax.set_title(title)
fig.colorbar(im,ax=axs.tolist(),label='Within-head Spearman rho',fraction=.025);fig.subplots_adjust(bottom=.29,right=.92,wspace=.2);fig.savefig(dest/'gate_correlations.png',dpi=180);plt.close(fig)
# Candidate curves: train-defined bins, sequence bootstrap CI on holdout.
selected=[];used=set()
for r in strong:
 if r['layer']>1 and r['feature'] not in used:selected.append(r);used.add(r['feature'])
 if len(selected)==6:break
if selected:
 fig,axs=plt.subplots(2,3,figsize=(14,8));rng=np.random.default_rng(9122)
 for ax,r in zip(axs.flat,selected):
  counts=data[r['layer']]['sequence_curves'][:,r['head'],r['j']];n=len(counts)
  for sl,label,color in [(slice(0,32),'Discovery','#777777'),(slice(32,n),'Held-out','#2575b0')]:
   v=counts[sl];c=np.nan_to_num(v[...,0]);s=np.nan_to_num(v[...,1]);mean=s.sum(0)/np.maximum(c.sum(0),1)
   bs=rng.integers(0,len(c),(1000,len(c)));bt=s[bs].sum(1)/np.maximum(c[bs].sum(1),1);lo,hi=np.quantile(bt,[.025,.975],axis=0)
   ax.plot(range(1,9),mean,'o-',color=color,label=label);ax.fill_between(range(1,9),lo,hi,color=color,alpha=.18)
  ax.set_title(f"L{r['layer']} H{r['head']} / {r['feature']}",fontsize=9);ax.set_xlabel('Geometry octile (fixed on discovery)');ax.set_ylabel('Write gate');ax.grid(alpha=.2)
 axs.flat[0].legend();fig.suptitle('Candidate relations with independent-sequence bootstrap / dual gates >=0.1');fig.tight_layout();fig.savefig(dest/'candidate_gate_curves.png',dpi=170);plt.close(fig)
# Quantile bands, never substitute mean ratio for its heavy-tailed distribution.
fig,axs=plt.subplots(1,3,figsize=(15,4.5));ql=data[0]['quantile_levels'];mid=int(np.flatnonzero(ql==.5)[0]);lo=int(np.flatnonzero(ql==.1)[0]);hi=int(np.flatnonzero(ql==.9)[0])
for h in range(16):
 x=[];med=[];c=[];ov=[]
 for l in range(1,24):
  w=rows[l]['write'][h]
  if not w['valid']:continue
  x.append(l);med.append(w['ratio_quantiles'][mid]);c.append(w['opposing']);ov.append(w['over_erase'])
 axs[0].plot(x,med,'o-',lw=.8,ms=2,alpha=.7);axs[1].plot(x,c,'o-',lw=.8,ms=2,alpha=.7);axs[2].plot(x,ov,'o-',lw=.8,ms=2,alpha=.7)
axs[0].set_yscale('log');axs[0].set_ylabel('Median actual O update / old-content norm');axs[1].set_ylabel('Fraction opposing old content');axs[2].set_ylabel('Fraction reversing old-content projection')
for ax in axs:ax.set_xlabel('Layer');ax.grid(alpha=.2)
fig.suptitle('Actual same-head O writeback / one line per head / dual gates >=0.1');fig.tight_layout();fig.savefig(dest/'writeback_by_head.png',dpi=170);plt.close(fig)
for d in data:d.close()
