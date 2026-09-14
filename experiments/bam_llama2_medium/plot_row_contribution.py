"""Export per-path layer knockout curves with paired uncertainty and route totals."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
s=json.loads((a.directory/'summary.json').read_text());paths=('Q','K','V','O')
fig,axs=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
for ax,path in zip(axs.ravel(),paths):
 data=[(l,s['scenarios'].get(f'layer_{l:02d}_{path}')) for l in range(24)]
 data=[(l,x) for l,x in data if x is not None]
 layers=np.array([l for l,x in data]);means=np.array([x['mean'] for l,x in data]);ci=np.array([1.96*x['se'] if x['se'] is not None else 0 for l,x in data])
 ax.errorbar(layers,means,yerr=ci,color='#2166ac',fmt='.-',capsize=2,lw=1,label='Mean ± 1.96 paired SE')
 fetch=layers%3==2
 ax.scatter(layers[fetch],means[fetch],color='#d6604d',zorder=4,label='Fetch layer')
 ax.axhline(0,color='gray',lw=.8);ax.set(title=f'{path} row: single-layer knockout',xlabel='Layer (zero-based)',ylabel='Loss increase (nats/token)');ax.set_xticks(range(0,24,3));ax.grid(alpha=.16);ax.legend(fontsize=8)
fig.suptitle(f'BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow · step 13500 · n={len(s["sample_ids"])}\nIndividual effects are not additive; V has no Fetch-layer branch. Panel y-scales differ.')
fig.savefig(a.directory/'layer_contributions.png',dpi=180);fig.savefig(a.directory/'layer_contributions.pdf')
fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
x=np.arange(4)
for shift,key,label,color in [(-.18,'knockout','Knockout alone','#2166ac'),(.18,'shapley','Exact four-path Shapley','#d6604d')]:
 y=[s[key][p]['mean'] for p in paths];err=[1.96*s[key][p]['se'] if s[key][p]['se'] is not None else 0 for p in paths]
 ax.bar(x+shift,y,width=.35,yerr=err,capsize=3,label=label,color=color)
ax.set_xticks(x,paths);ax.set(ylabel='Loss increase / allocation (nats/token)',title=f'All row reads off: +{s["all_rows_off"]["mean"]:.4f} nats/token; n={len(s["sample_ids"])}')
ax.axhline(0,color='gray',lw=.8);ax.legend();ax.grid(axis='y',alpha=.16)
fig.savefig(a.directory/'path_contributions.png',dpi=180);fig.savefig(a.directory/'path_contributions.pdf')
