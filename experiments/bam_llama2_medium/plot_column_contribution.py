"""Paired row/column scientific figures; complete tables remain in summary JSON."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
PATHS=('Q','K','V','O')
def run(medium,xl,out):
 out.mkdir(parents=True,exist_ok=True);data=[json.loads((r/'summary.json').read_text()) for r in (medium,xl)];labels=['Medium','XL'];colors=['#2166ac','#d6604d','#756bb1']
 fig,axes=plt.subplots(2,4,figsize=(16,8.5),constrained_layout=True)
 for i,(d,label) in enumerate(zip(data,labels)):
  for p,path in enumerate(PATHS):
   ax=axes[i,p];layers=[l for l in range(24) if not(path=='V' and l%3==2)]
   for side,color in zip(['row','column','both'],colors):
    ss=[d['metrics'][f'{side}/all/layer_{l:02d}_{path}'] for l in layers]
    ax.errorbar(layers,[s['mean'] for s in ss],yerr=[1.96*s['se'] for s in ss],fmt='.-',lw=1,color=color,label=side,capsize=1)
   ax.set(title=f'{label} {path}',xlabel='Layer (0-based; F=2,5,8,...)',ylabel='Mean loss increase');ax.axhline(0,c='gray',lw=.5);ax.grid(alpha=.15);ax.legend(fontsize=8)
 # Pair the model axes within each path, so cross-model magnitudes are comparable.
 for p in range(4):
  lo=min(axes[0,p].get_ylim()[0],axes[1,p].get_ylim()[0]);hi=max(axes[0,p].get_ylim()[1],axes[1,p].get_ylim()[1]);axes[0,p].set_ylim(lo,hi);axes[1,p].set_ylim(lo,hi)
 fig.suptitle('Same-layer row / column / both read deletion · same64 Pile sequences\nPaired mean ± 1.96 SE; no multiple-comparison adjustment; F has no Local V')
 for ext in ('png','pdf'):fig.savefig(out/f'row_column_layers.{ext}',dpi=180)
 fig,axes=plt.subplots(1,4,figsize=(15,4.4),constrained_layout=True)
 for p,(ax,path) in enumerate(zip(axes,PATHS)):
  for i,(d,label) in enumerate(zip(data,labels)):
   for j,(side,color) in enumerate(zip(['row','column','both'],colors)):
    s=d['metrics'][f'{side}/all/coalition_{1<<p:02d}'];ax.bar(i*4+j,s['mean'],yerr=1.96*s['se'],color=color,width=.8,capsize=2,label=side if i==0 else None)
  ax.set(title=path,xticks=[1,5],xticklabels=labels,ylabel='Whole-path loss increase');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.15)
 fig.suptitle('Whole-path deletion; row and column interactions are measured, not inferred by addition')
 for ext in ('png','pdf'):fig.savefig(out/f'row_column_paths.{ext}',dpi=180)
 lines=['| Model | Path | Row | Column | Both | Interaction: both−row−column |','|---|---|---:|---:|---:|---:|']
 for d,label in zip(data,labels):
  for p,path in enumerate(PATHS):
   s=d['row_column_pairs'][f'all/coalition_{1<<p:02d}'];f=lambda x:f'{x["mean"]:+.6f} ± {1.96*x["se"]:.6f}'
   lines.append('| '+label+' | '+path+' | '+' | '.join(f(s[k]) for k in ['row','column','both','interaction'])+' |')
 (out/'path_table.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('medium',type=Path);p.add_argument('xl',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.medium,a.xl,a.output)
