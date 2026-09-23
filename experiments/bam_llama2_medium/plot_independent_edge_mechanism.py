#!/usr/bin/env python3
"""Same-step gate and actual feedback allocation versus original dual-write baseline."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();data=json.loads(a.input.read_text())
runs=['BamMediumAllLocalDualWriteGates','BamMediumAllLocalIndependentEdges']
labels=['Original dual-write','Independent edges']
metrics=['read_mean','main_mean','feedback_mean','feedback_norm_share']
titles=['LocalO read gate','Main write gate','Feedback gate','Actual feedback norm share']
step=max(s['step'] for s in data[runs[1]])
fig,axes=plt.subplots(2,4,figsize=(15,9),constrained_layout=True)
for row,run in enumerate(runs):
 s=next(s for s in data[run] if s['step']==step)
 for col,(m,title) in enumerate(zip(metrics,titles)):
  v=s['values'];arr=np.array([[v[f'bam/dual_write/layer_{l:03}/head_{h:02}/{m}'] for h in range(16)] for l in range(1,23)])
  ax=axes[row,col];im=ax.imshow(arr,aspect='auto',vmin=0,vmax=1 if col==3 else .4,cmap='viridis')
  ax.set_title(f'{labels[row]}\n{title}');ax.set_xlabel('Head');ax.set_ylabel('Layer');ax.set_yticks(np.arange(22),np.arange(1,23));ax.set_xticks([0,4,8,12,15]);fig.colorbar(im,ax=ax,shrink=.7)
fig.suptitle(f'Step {step}: each cell is one layer/head batch statistic; gate colors clipped at 0.4')
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)
fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for ax,(m,title) in zip(axes.flat,list(zip(metrics,titles))+[('read_main_corr','Read / main-write correlation'),('main_feedback_corr','Main / feedback correlation')]):
 for run,label,color in zip(runs,labels,['#2864b7','#d65d28']):
  rows=data[run];x=[s['step'] for s in rows];med=[s['bands']['middle'][m]['median'] for s in rows];lo=[s['bands']['middle'][m]['q10'] for s in rows];hi=[s['bands']['middle'][m]['q90'] for s in rows]
  ax.plot(x,med,'o-',label=label,color=color);ax.fill_between(x,lo,hi,alpha=.13,color=color)
 ax.set_title(title);ax.set_xlabel('Training step');ax.grid(alpha=.2)
 if 'corr' in m:ax.axhline(0,color='gray',lw=.7)
 else:ax.set_ylim(bottom=0)
axes[0,0].legend();fig.suptitle('Middle layers 3–16: median and 10–90% distribution across 224 heads\nNorm share is mean feedback-record norm / mean sum of component-record norms')
fig.savefig(a.output.with_name(a.output.stem+'_trajectory.png'),dpi=160);fig.savefig(a.output.with_name(a.output.stem+'_trajectory.pdf'));plt.close(fig)
