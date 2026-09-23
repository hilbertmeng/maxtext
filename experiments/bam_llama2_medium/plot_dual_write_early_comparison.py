#!/usr/bin/env python3
"""Plot matched-step gate and actual write-share trajectories."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
j=json.loads(a.input.read_text());fig,axes=plt.subplots(2,2,figsize=(11,7.5),layout='constrained')
colors=['#2767ad','#d87519'];names=['Gated LocalO feedback','Raw LocalO feedback']
for (run,ss),color,label in zip(j.items(),colors,names):
 ss=[s for s in ss if s['step']>0];steps=[s['step'] for s in ss]
 for ax,metric,title in zip(axes.flat,['read_mean','main_mean','feedback_norm_share','gradient'],['LocalO read gate','Main write gate','LocalO share of cumulative write norms','Feedback gate kernel gradient norm']):
  if metric=='gradient':
   y=[s['bands']['middle']['parameter_norms']['raw_grads/W_gw_feedback/kernel']['median'] for s in ss]
   ax.set_yscale('log');ax.set_ylabel('Median across L3-16')
  else:
   vals=[s['bands']['middle'][metric] for s in ss];y=[v['median'] for v in vals]
   ax.fill_between(steps,[v['q10'] for v in vals],[v['q90'] for v in vals],color=color,alpha=.10)
   ax.set_ylabel('Head median; shading = 10-90%')
  ax.plot(steps,y,'o-',color=color,label=label,lw=2);ax.set_title(title);ax.set_xlabel('Training step');ax.grid(alpha=.2)
axes[0,0].legend(frameon=False);fig.suptitle('L3-16, same-step health comparison\nWrite share uses actual gated content; feedback gate probabilities have different operands',fontsize=12)
a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=170);fig.savefig(a.output.with_suffix('.pdf'))
