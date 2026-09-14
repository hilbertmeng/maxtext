"""Plot heldout loss of independently selected weakest/strongest row-head groups."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(medium,xl,out):
 data=[json.loads((r/'heads/group_summary.json').read_text()) for r in (medium,xl)]
 fig,axes=plt.subplots(1,4,figsize=(15,4.3),constrained_layout=True)
 suffix=['bottom4','top4','bottom8','top8'];ticks=['weak 4','strong 4','weak 8','strong 8']
 for ax,p in zip(axes,'QKVO'):
  for i,(label,d,color) in enumerate(zip(['Medium','XL'],data,['#2166ac','#d6604d'])):
   values=[d['scenarios'][f'{p}_{s}']['heldout32'] for s in suffix];x=np.arange(4)+(i-.5)*.34
   ax.bar(x,[v['mean'] for v in values],width=.32,color=color,label=label,yerr=[1.96*v['se'] for v in values],capsize=2)
  ax.set(title=f'{p} row',ylabel='Heldout mean loss increase',xticks=range(4),xticklabels=ticks);ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.15);ax.axhline(0,color='gray',lw=.6);ax.legend()
 fig.suptitle('Remove selected heads at every valid layer · select on first32, evaluate on last32\nBars: paired mean ± 1.96 SE; distinct y scale per path; no multiple-comparison adjustment')
 out.mkdir(exist_ok=True,parents=True)
 for ext in ('png','pdf'):fig.savefig(out/f'head_group_validation.{ext}',dpi=180)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('medium',type=Path);p.add_argument('xl',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.medium,a.xl,a.output)
