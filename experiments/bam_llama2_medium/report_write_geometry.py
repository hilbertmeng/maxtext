"""Make population CDFs, held-out checks, and sequence bootstrap summaries."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
root=Path(sys.argv[1]);dest=root/'figures';dest.mkdir(exist_ok=True)
rows=[json.loads((root/f'analysis_gate010/layer_{l:02d}.json').read_text()) for l in range(24)];assert all(r['n']==128 for r in rows)
d=[np.load(root/f'analysis_gate010/layer_{l:02d}.npz') for l in range(24)];e=[np.load(root/f'exploration/layer_{l:02d}.npz') for l in range(24)];er=[json.loads((root/f'exploration/layer_{l:02d}.json').read_text()) for l in range(24)]
main=list(range(2,23));fn=d[0]['feature_names'].tolist();ex=e[0]['names'].tolist();ql=d[0]['quantile_levels'];out={'population':'L2-L22, dual sigmoid gates >=0.1, nonzero local O; 128 independent sequences'}
rng=np.random.default_rng(9287)
def boot_fraction(num,den):
 b=rng.integers(0,len(num),(2000,len(num)));v=num[b].sum(1)/np.maximum(den[b].sum(1),1)
 return [float(num.sum()/den.sum()),*np.quantile(v,[.025,.975]).tolist()]
coverage={}
for t in ['005','010','020']:
 rr=[json.loads((root/f'analysis_gate{t}/layer_{l:02d}.json').read_text()) for l in range(24)];cov=np.array([r['eligible_tokens_per_head'] for r in rr]);a=np.array([r['eligible_sequences_discovery'] for r in rr]);b=np.array([r['eligible_sequences_validation'] for r in rr]);ok=(a>=16)&(b>=16)&(cov>=2048)
 coverage[t]=dict(tokens=int(cov.sum()),eligible_heads=int(ok.sum()),zero_heads=int((cov[1:]==0).sum()),by_layer=ok.sum(1).tolist())
out['coverage']=coverage
count=sum(e[l]['counts'][1].sum(1) for l in main);out['same_head']={}
for label,j in [('opposing',2),('over_erase',3),('reinforcing',4),('light_opposing',5),('ratio_above_one',6),('key_content_sign_disagree',7),('key_cos_abs_below_point2',8)]:out['same_head'][label]=boot_fraction(count[:,j],count[:,1])
lg=sum(e[l]['largest'][1].sum(0) for l in main);out['largest_component_fraction']=(lg/lg.sum()).tolist()
# Dense histograms from every selected token. CDF quantiles here explicitly binned.
gh=sum(e[l]['geometry_hist'][1] for l in main);ge=e[0]['geometry_edges'];rh=sum(e[l]['hist'][1].sum(0) for l in main);re=e[0]['edges']
out['binned_population_quantiles']={}
for names,hist,edges in [(fn,gh,ge),(ex,rh,re)]:
 for j,name in enumerate(names):
  c=np.cumsum(hist[j])/max(hist[j].sum(),1);idx=np.searchsorted(c,[.05,.25,.5,.75,.95]).clip(0,len(c)-1);v=edges[j,1:][idx]
  if name in ['old_at_unit_p_norm','o_write_at_unit_p_norm','write_ratio']:v=10**v
  out['binned_population_quantiles'][name]=v.tolist()
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
for ax,ids,label in [(axs[0],range(4),'log10 norm'),(axs[1],range(4,8),'Norm / sum of four norms'),(axs[2],range(8,14),'Cosine')]:
 for j in ids:
  ax.step(ge[j,1:],np.cumsum(gh[j])/gh[j].sum(),where='post',label=fn[j].replace('lognorm_','').replace('share_','').replace('cos_',''))
 ax.set_xlabel(label);ax.set_ylabel('Cumulative probability');ax.grid(alpha=.2);ax.legend(fontsize=7)
axs[0].set_xlim(-10,2);fig.suptitle('Full population distributions | L2-L22 | dual gates >=0.1 | 128 sequences');fig.tight_layout();fig.savefig(dest/'population_geometry.png',dpi=180);plt.close(fig)
# Every candidate is selected from discovery only; report independent new64.
cor=np.stack([x['correlations'] for x in d]);candidates=[];a=np.array([r['eligible_sequences_discovery'] for r in rows]);b=np.array([r['eligible_sequences_validation'] for r in rows]);cov=np.array([r['eligible_tokens_per_head'] for r in rows]);ok=(a>=16)&(b>=16)&(cov>=2048)
for l in main:
 for h in range(16):
  if not ok[l,h]:continue
  x=d[l]['sample_features'][64:,:,h];y=d[l]['sample_gate'][64:,:,h]
  for j,name in enumerate(fn):
   if j in [8,11,12] or abs(cor[l,0,h,j])<.3 or not np.isfinite(cor[l,0,h,j]):continue
   take=np.isfinite(x[...,j])&np.isfinite(y)
   val=float(spearmanr(x[...,j][take],y[take]).statistic) if take.sum()>50 else float('nan')
   candidates.append(dict(layer=l,head=h,feature=name,discovery=float(cor[l,0,h,j]),new64=val,replicated=bool(val*cor[l,0,h,j]>0 and abs(val)>.2)))
out['candidates']=candidates
out['model_scores']=[]
for l in main:
 m=er[l]['mse'];base=m['head_position'];row={'layer':l,**{k:float(1-v/base) for k,v in m.items()},'geometry_after_read_controls':1-m['geometry_and_read_gates']/m['read_gate_controls']};out['model_scores'].append(row)
fig,ax=plt.subplots(figsize=(11,4.5))
for k,label in [('total_amplitude','Total norm only'),('amplitude','Four norms'),('angles','Pairwise angles'),('geometry','All geometry'),('geometry_after_read_controls','Geometry beyond read gates')]:ax.plot(main,[r[k] for r in out['model_scores']],'o-',ms=3,label=label)
ax.set_xlabel('Layer');ax.set_ylabel('Held-out residual MSE reduction');ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.2);ax.set_title('Train: first32 sequences / held out: next96 / dual gates >=0.1');fig.tight_layout();fig.savefig(dest/'gate_prediction.png',dpi=180);plt.close(fig)
# Paired opposite-signed laws; selection fixed before looking at new64.
examples=[(18,15,0),(15,12,2),(4,12,7),(18,4,7),(15,12,13),(16,12,13)]
fig,axs=plt.subplots(2,3,figsize=(14,8));out['examples']=[]
for ax,(l,h,j) in zip(axs.flat,examples):
 v=d[l]['sequence_curves'][:,h,j];q=d[l]['curves'][h,j];ax.fill_between(range(1,9),q[:,1],q[:,5],alpha=.18,color='#4488aa',label='All128: 10-90% of gates');ax.plot(range(1,9),q[:,3],'-',color='#155b7a',label='All128: median')
 vv=v[64:];cc=vv[...,0];ss=vv[...,1];mu=ss.sum(0)/np.maximum(cc.sum(0),1);bs=rng.integers(0,len(cc),(2000,len(cc)));bt=ss[bs].sum(1)/np.maximum(cc[bs].sum(1),1);ci=np.quantile(bt,[.025,.975],axis=0)
 ax.errorbar(range(1,9),mu,yerr=np.maximum(np.stack([mu-ci[0],ci[1]-mu]),0),fmt='o',ms=3,color='#b34e21',label='New64: mean + sequence95% CI');ax.set_title(f'L{l} H{h}: {fn[j]}',fontsize=10);ax.set_xlabel('Geometry octile (fixed on first32)');ax.set_ylabel('Write gate');ax.grid(alpha=.2)
 cornew=next((c['new64'] for c in candidates if c['layer']==l and c['head']==h and c['feature']==fn[j]),None)
 out['examples'].append(dict(layer=l,head=h,feature=fn[j],discovery=float(cor[l,0,h,j]),new64=cornew,gate_octile_new64=mu.tolist(),gate_octile_ci_new64=ci.tolist(),position_controlled=e[l]['partial_correlations'][:,h,j].tolist(),threshold_correlations={t:np.load(root/f'analysis_gate{t}/layer_{l:02d}.npz')['correlations'][:,h,j].tolist() for t in ['000','005','010','020']}))
axs.flat[0].legend(fontsize=7);fig.suptitle('Geometry / write-gate laws differ across heads | dual gates >=0.1');fig.tight_layout();fig.savefig(dest/'gate_laws.png',dpi=180);plt.close(fig)
# Same-head key geometry, update magnitude, and erasure boundary.
fig,axs=plt.subplots(1,3,figsize=(15,4.5))
for j in [0,1]:axs[0].step(re[j,1:],np.cumsum(rh[j])/rh[j].sum(),where='post',label=ex[j])
axs[0].set_xlabel('Cosine');axs[0].set_ylabel('Cumulative probability');axs[0].legend();axs[0].grid(alpha=.2)
j=4;axs[1].step(10**re[j,1:],np.cumsum(rh[j])/rh[j].sum(),where='post');axs[1].set_xscale('log');axs[1].set_xlim(.01,3);axs[1].set_xlabel('Actual O update norm / old-content norm');axs[1].grid(alpha=.2)
joint=sum(e[l]['joint'][1].sum(0) for l in main);xx=e[0]['joint_cos_edges'];yy=e[0]['joint_logratio_edges'];im=axs[2].pcolormesh(xx,yy,np.log10(1+joint.T),cmap='magma');fig.colorbar(im,ax=axs[2],label='log10(1 + count)');cx=np.linspace(-1,-.01,200);axs[2].plot(cx,np.log10(-1/cx),'c--',lw=1.5,label='Reverse old projection');axs[2].set_ylim(-2,1);axs[2].set_xlabel('cos(O update, old content)');axs[2].set_ylabel('log10 update / old');axs[2].legend(fontsize=7)
fig.suptitle('Same-head O loop | L2-L22 | actual read/write gates and RMS included');fig.tight_layout();fig.savefig(dest/'read_write_loop.png',dpi=180);plt.close(fig)
# Per-layer distributions retain depth variation rather than hiding it in a pooled CDF.
fig,axs=plt.subplots(1,2,figsize=(13,5));xx=np.arange(1,24)
for j,c in zip(range(4,8),['#1978b8','#ff8c20','#3d9f45','#bf3030']):
 q=np.stack([e[l]['pooled_quantiles'][1,j] for l in xx]);axs[0].plot(xx,q[:,6],color=c,label=fn[j].replace('share_',''));axs[0].fill_between(xx,q[:,4],q[:,8],color=c,alpha=.12)
for j,c in [(fn.index('cos_sv__local_o'),'#6c52a5'),(fn.index('cos_bam_other__local_o'),'#1978b8')]:
 q=np.stack([e[l]['pooled_quantiles'][1,j] for l in xx]);axs[1].plot(xx,q[:,6],color=c,label=fn[j]);axs[1].fill_between(xx,q[:,4],q[:,8],color=c,alpha=.15)
for ax in axs:ax.set_xlabel('Layer');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.axvline(1,color='gray',ls=':');ax.axvline(23,color='gray',ls=':')
axs[0].set_ylabel('Norm share: median and 10-90%');axs[1].set_ylabel('Cosine: median and 10-90%');fig.tight_layout();fig.savefig(dest/'geometry_by_layer.png',dpi=180);plt.close(fig)
(root/'research_summary.json').write_text(json.dumps(out,indent=2))
with (root/'per_layer_head_quantiles.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['layer','head','feature']+[f'q{x}' for x in ql])
 for l in range(24):
  for h in range(16):
   for j,name in enumerate(fn):w.writerow([l,h,name,*d[l]['feature_quantiles'][h,j]])
   for j,name in enumerate(ex):w.writerow([l,h,name,*e[l]['quantiles'][1,h,j]])
print(json.dumps({k:v for k,v in out.items() if k not in ['candidates','model_scores','examples']},indent=2))
print('CANDIDATES',len(candidates),'REPLICATED',sum(c['replicated'] for c in candidates))
