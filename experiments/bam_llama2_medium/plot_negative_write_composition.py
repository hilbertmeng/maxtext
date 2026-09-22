"""Static figures and per-head CSV for negative-head write composition."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

p=Path(sys.argv[1]);s=json.loads((p/'summary.json').read_text());f=np.load(p/'statistics.npz')
heads=s['heads'];labels=[f"L{h['layer']}H{h['head']}" for h in heads]
hist=f['hist'];sh=f['share_hist'];colors=['#4477aa','#66ccee','#228833','#cc6677'];parts=['MHA value','LocalV self','LocalV other','LocalO']
fig,axs=plt.subplots(2,2,figsize=(13,10),layout='constrained')
a=axs[0,0];h=hist.sum(0);h=h/h.sum();im=a.imshow(h.T,origin='lower',extent=[0,1,0,1],norm=LogNorm(vmin=1e-6,vmax=h.max()),cmap='viridis',aspect='auto');a.axvline(.1,c='white',ls='--',lw=1);a.axhline(.1,c='white',ls='--',lw=1);a.set(xlabel='LocalO read gate',ylabel='Write gate',title='All valid positions: joint probability mass');fig.colorbar(im,ax=a,label='Fraction per 0.02 x 0.02 bin')
a=axs[0,1];v=np.array(s['pooled']['norm_shares']);bottom=np.zeros(4)
for j in range(4):a.bar(np.arange(4),100*v[:,j],bottom=bottom,color=colors[j],label=parts[j]);bottom+=100*v[:,j]
a.set(xticks=np.arange(4),xticklabels=['All','g >= .05','g >= .10','g >= .20'],ylabel='Cumulative component norm share (%)',title='Actual gate-weighted four-part composition');a.legend(fontsize=8)
a=axs[1,0];ph=s['per_head'];x=100*(1-np.array(ph['ungated_nonlocal_norm_share'])[:,0]);y=100*(1-np.array(ph['nonlocal_norm_share'])[:,0]);im=a.scatter(x,y,c=[h['layer'] for h in heads],cmap='plasma',s=25);a.plot([0,70],[0,70],'k--',lw=1);a.set(xlabel='LocalO norm share without write weight (%)',ylabel='LocalO norm share with actual write weight (%)',title='Within every head: LocalO share decreases');fig.colorbar(im,ax=a,label='Layer')
a=axs[1,1];histshare=sh[:,2].sum(0);xx=np.arange(1,101)/100
for j in range(4):a.plot(xx,np.cumsum(histshare[j])/histshare[j].sum(),color=colors[j],label=parts[j])
a.set(xlabel='Single-position component norm / sum of four norms',ylabel='CDF',title='Write gate >= .10: full share distribution');a.legend(fontsize=8);a.grid(alpha=.2)
fig.suptitle('87 strong negative-correlation middle heads | 96 held-out sequences',fontsize=14);fig.savefig(p/'composition_overview.png',dpi=170);plt.close(fig)
fig,axs=plt.subplots(11,8,figsize=(20,24),sharex=True,sharey=True)
for j,a in enumerate(axs.flat):
 if j>=len(heads):a.axis('off');continue
 h=hist[j]/hist[j].sum();a.imshow(h.T,origin='lower',extent=[0,1,0,1],norm=LogNorm(vmin=1e-5,vmax=1),cmap='viridis',aspect='auto');a.axvline(.1,c='white',lw=.4);a.axhline(.1,c='white',lw=.4);a.set_title(labels[j],fontsize=9);a.set_xticks([0,.5,1]);a.set_yticks([0,.5,1]);a.tick_params(labelsize=6)
fig.supxlabel('LocalO read gate');fig.supylabel('Write gate');fig.suptitle('Per-head joint distributions (same log probability scale)');fig.tight_layout(rect=(.02,.02,1,.98));fig.savefig(p/'gate_joint_by_head.png',dpi=140);plt.close(fig)
with (p/'per_head.csv').open('w') as out:
 w=csv.writer(out);w.writerow(['head','discovery_rho','nonlocal_norm_share','nonlocal_energy_share','local_norm_suppression_pp','local_energy_suppression_pp','read_high_write_low','read_low_write_high','both_low','both_high'])
 for j,h in enumerate(heads):w.writerow([labels[j],h['read_write_rho']]+[ph[k][j][0] for k in ['nonlocal_norm_share','nonlocal_energy_share','local_norm_suppression_pp','local_energy_suppression_pp']]+s['per_head_quadrants'][j][1])
print(p/'composition_overview.png')
