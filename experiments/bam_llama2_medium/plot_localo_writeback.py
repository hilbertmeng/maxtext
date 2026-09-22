"""LocalO-only and complementary first-order effects at equal relative dose."""
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(sys.argv[1]);s=json.loads((p/'analysis/summary.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
labels=['LocalO only','Other three only','All four'];order=[0,2,1];colors=['#cc6677','#228833','#4477aa']
names=['Read high / write low','Read low / write high','Both low','Both high']
for j,(a,row) in enumerate(zip(axs.flat,s['thresholds'][1]['states'])):
 for k,b in enumerate(order):
  r=row['components'][b];x=r['predicted_down_mean']*1e6;lo,hi=np.array(r['ci95'])*1e6
  a.errorbar(x,2-k,xerr=[[x-lo],[hi-x]],fmt='o',c=colors[k],capsize=4)
 a.axvline(0,c='gray',ls='--',lw=1);a.set(yticks=[2,1,0],yticklabels=labels,ylim=(-.6,2.6),title=names[j],xlabel='Predicted loss change (micro-nats/token)');a.grid(axis='x',alpha=.2)
fig.suptitle('Reduce selected write component by 10% | 87 heads, 96 sequences\nMean of separate head interventions; sequence-bootstrap 95% intervals',fontsize=12);fig.savefig(p/'analysis/component_effects.png',dpi=170)
