"""CDFs and within-head angle comparison for all four joint gate states."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);s=json.loads((p/'summary.json').read_text());f=np.load(p/'distributions.npz')
colors=['#cc6677','#4477aa','#999933','#228833'];labels=['Read high / write low','Read low / write high','Both low','Both high']
fig,axs=plt.subplots(2,2,figsize=(13,10),layout='constrained')
for k,a in enumerate(axs.flat[:3]):
 for st in range(4):
  h=f['hist'][1,st,k];a.plot(np.arange(1,181),np.cumsum(h)/h.sum(),c=colors[st],label=labels[st],lw=1.8)
 a.axvline(90,c='gray',ls='--',lw=1);a.set(xlim=(0,180),ylim=(0,1),xlabel='Angle (degrees)',ylabel='Cumulative fraction',title=['Read address vs write address','Read content vs total write content','Control: read content vs write excluding LocalO'][k]);a.grid(alpha=.2)
axs[0,0].legend(fontsize=8,loc='upper left')
a=axs[1,1];ids=s['within_head_comparison'][1]['eligible_heads']
for k,col,label in [(0,'#4477aa','Address'),(1,'#cc6677','Total content'),(2,'#228833','Content excluding LocalO')]:
 x=[s['per_head'][j]['states'][1][k]['mean'] for j in ids];y=[s['per_head'][j]['states'][3][k]['mean'] for j in ids];a.scatter(x,y,s=19,c=col,label=label,alpha=.75)
a.plot([0,120],[0,120],'k--',lw=1);a.set(xlim=(0,120),ylim=(0,120),xlabel='Read low / write high: head mean angle',ylabel='Both high: head mean angle',title='69 matched heads: below diagonal means smaller');a.legend(fontsize=8);a.grid(alpha=.2)
fig.suptitle('87 negative-correlation middle heads | 96 held-out sequences | gate threshold 0.10',fontsize=13);fig.savefig(p/'joint_gate_angles.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(11,8,figsize=(20,24),sharex=True,sharey=True)
for j,a in enumerate(axs.flat):
 if j>=len(s['heads']):a.axis('off');continue
 for st in [1,3]:
  h=f['per_head_hist'][j,st,1]
  if h.sum():a.plot(np.arange(1,181),np.cumsum(h)/h.sum(),color=colors[st],lw=1)
 a.set_title(s['per_head'][j]['id'],fontsize=9);a.set(xlim=(0,180),ylim=(0,1));a.tick_params(labelsize=6)
fig.suptitle('Read vs total-write content angle CDF by head: blue=read low/write high; green=both high');fig.supxlabel('Angle (degrees)');fig.supylabel('Cumulative fraction');fig.tight_layout(rect=(.02,.02,1,.98));fig.savefig(p/'content_angles_by_head.png',dpi=130)
