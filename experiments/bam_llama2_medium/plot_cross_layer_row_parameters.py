"""Render parameter-sharing evidence from cross_layer_row_parameters.py."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
units=json.loads((a.directory/'analysis.json').read_text())['units'][1:]
fig,axs=plt.subplots(1,3,figsize=(13,3.6),constrained_layout=True)
for ax,arm,read in zip(axs[:2],['V: two Local layers','O: LLF three layers'],[lambda u:u['V']['shared'],lambda u:u['O']['shared_LLF']]):
 data=[read(u) for u in units]; ranks=sorted(set.intersection(*[set(c['rank'] for c in d['curves']) for d in data]))
 rows=[[next(c for c in d['curves'] if c['rank']==r) for r in ranks] for d in data]
 x=np.array([c['param_ratio'] for c in rows[0]]);y=np.array([[c['joint_retained_energy'] for c in row] for row in rows]);control=np.array([[c['independent_svd_energy_same_budget'] for c in row] for row in rows])
 ax.plot(x,y.mean(0),label='Shared input basis',color='#2166ac');ax.fill_between(x,y.min(0),y.max(0),alpha=.16,color='#2166ac');ax.plot(x,control.mean(0),'--',label='Independent / dense control',color='#d6604d')
 ax.axhline(.95,color='gray',lw=.8);ax.axvline(1,color='gray',lw=.8);ax.set(xlim=(0,1.25),ylim=(0,1.02),xlabel='Parameters / original row projections',ylabel='Retained squared weight energy',title=arm);ax.grid(alpha=.15);ax.legend(fontsize=8)
for label,read in [('V LL',lambda u:u['V']['shared']),('O LL',lambda u:u['O']['shared_LL']),('O LLF',lambda u:u['O']['shared_LLF'])]:
 axs[2].plot([u['unit'] for u in units],[read(u)['spectrum']['rank95']/read(u)['breakeven_rank'] for u in units],marker='o',label=label)
axs[2].axhline(1,color='gray',lw=.8);axs[2].set(xlabel='LLF unit (zero-based)',ylabel='Parameters / original at 95% energy',title='95% energy costs more parameters');axs[2].legend();axs[2].grid(alpha=.15)
fig.suptitle('BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow · step 13500\nUnits 1–7; unit 0 reported separately because layer 0 is zero')
fig.savefig(a.directory/'sharing.png',dpi=180);fig.savefig(a.directory/'sharing.pdf')
