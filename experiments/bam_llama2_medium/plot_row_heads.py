"""Scientific heatmaps of exact singleton row-head loss changes, plus ranking summaries."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PATHS=('Q','K','V','O')
def run(medium,xl,out):
 out.mkdir(parents=True,exist_ok=True)
 roots=[medium,xl];labels=['Medium','XL'];data=[];summaries=[]
 for root in roots:
  with np.load(root/'heads/paired_head_gaps.npz') as f:data.append(np.asarray(f['gap'],float))
  summaries.append(json.loads((root/'heads/head_summary.json').read_text()))
 means=[np.mean(g,axis=0) for g in data]
 fig,axes=plt.subplots(2,4,figsize=(16,11),constrained_layout=True)
 for p,path in enumerate(PATHS):
  valid=np.concatenate([v[:,p,:].ravel() for v in means]);valid=valid[np.isfinite(valid)]
  lim=max(float(np.max(np.abs(valid))),1e-8);norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim)
  for i,label in enumerate(labels):
   ax=axes[i,p];im=ax.imshow(np.ma.masked_invalid(means[i][:,p,:]),aspect='auto',cmap='RdBu_r',norm=norm,interpolation='nearest')
   ax.set(title=f'{label} · {path} row',xlabel='MHA head index',ylabel='Layer (zero-based)');ax.set_xticks([0,3,7,11,15]);ax.set_yticks(range(24));ax.set_yticklabels([f'{l}{"F" if l%3==2 else "L"}' for l in range(24)],fontsize=8)
   ax.set_facecolor('#dddddd');fig.colorbar(im,ax=ax,shrink=.8,label='Mean loss increase')
 fig.suptitle('Exact layer × head row knockout · same 64 Pile sequences\nColumn/gates/other heads remain native. Color scale matched across models within each path; gray = no Local V on F.')
 for ext in ['png','pdf']:fig.savefig(out/f'head_heatmaps.{ext}',dpi=180)
 table=['# Row head distribution','','Scores: exact single-head row knockout, mean ± 1.96 paired SE. Head indices do not identify shared features across layers/models.','']
 detail={}
 for label,summary in zip(labels,summaries):
  table += [f'## {label}','','| Path | Layer | Head | Mean loss increase | ±1.96 SE |','|---|---:|---:|---:|---:|']
  detail[label]={}
  for path in PATHS:
   candidates=[dict(layer=r['layer'],head=int(h),**s) for r in summary['rows'].values() if r['path']==path for h,s in r['heads'].items()]
   candidates.sort(key=lambda a:a['mean'],reverse=True);detail[label][path]=dict(top=candidates[:8],bottom=sorted(candidates,key=lambda a:a['mean'])[:8])
   for row in candidates[:4]:table.append(f'| {path} | {row["layer"]} | {row["head"]} | {row["mean"]:+.6f} | {1.96*row["se"]:.6f} |')
 (out/'head_top_tables.md').write_text('\n'.join(table)+'\n');(out/'head_top_tables.json').write_text(json.dumps(detail,indent=2))
 fig,axes=plt.subplots(1,4,figsize=(15,3.8),constrained_layout=True)
 for ax,path in zip(axes,PATHS):
  for label,summary,color in zip(labels,summaries,['#2166ac','#d6604d']):
   rows=[r for r in summary['rows'].values() if r['path']==path]
   ax.plot([r['layer'] for r in rows],[r['positive_singleton_score_top4_fraction'] for r in rows],'.-',label=label,color=color,lw=1)
  ax.set(title=path,xlabel='Layer',ylabel='Top4 / positive singleton score sum',ylim=(0,1.05));ax.grid(alpha=.15);ax.legend()
 fig.suptitle('Within-layer concentration of individual deletion scores (not additive causal attribution)')
 for ext in ['png','pdf']:fig.savefig(out/f'head_concentration.{ext}',dpi=180)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('medium',type=Path);p.add_argument('xl',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.medium,a.xl,a.output)
