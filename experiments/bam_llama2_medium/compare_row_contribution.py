"""Paired Medium/XL row-deletion comparison and exportable figures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize_row_contribution import stats

MEDIUM='BamMediumIndependentLLFLocalVRank4RoutingBAlignedRow'
XL='BamXLIndependentLLFLocalQKRank4CFp32AlignedRowSharedBasis'

def value(s):return f'{s["mean"]:+.6f} ± {1.96*s["se"]:.6f}'
def run(medium,xl,out):
 out.mkdir(parents=True,exist_ok=True)
 m=json.loads((medium/'summary.json').read_text());x=json.loads((xl/'summary.json').read_text())
 assert m['sample_ids']==x['sample_ids']==list(range(64));assert m['sequence_hashes']==x['sequence_hashes']
 ma=np.load(medium/'paired_results.npz');xa=np.load(xl/'paired_results.npz')
 names=list(m['scenarios']);assert names==list(x['scenarios'])
 result=dict(models=[MEDIUM,XL],sample_ids=list(range(64)),baseline={'medium':m['baseline'],'xl':x['baseline']},scenario_delta_xl_minus_medium={name:stats(xa['gaps'][:,i]-ma['gaps'][:,i]) for i,name in enumerate(names)},shapley_delta_xl_minus_medium={p:stats(xa['shapley'][:,i]-ma['shapley'][:,i]) for i,p in enumerate('QKVO')})
 text=['# Medium / XL paired row contribution comparison','',f'Medium: `{MEDIUM}` @13500; XL: `{XL}` @30091.', 'Same 64 Pile sequences. Values are mean ± 1.96 paired SE; positive means loss increase. Different architecture/routing/training history: not an isolated scale experiment.','', '| Path | Medium knockout | XL knockout | XL−Medium paired difference | Medium Shapley | XL Shapley |','|---|---:|---:|---:|---:|---:|']
 for i,p in enumerate('QKVO'):
  text.append('| '+p+' | '+' | '.join(value(s) for s in [m['knockout'][p],x['knockout'][p],result['scenario_delta_xl_minus_medium'][f'coalition_{1<<i:02d}'],m['shapley'][p],x['shapley'][p]])+' |')
 for stage in ['targeted','depth']:
  mp=np.load(medium/stage/'paired_results.npz');xp=np.load(xl/stage/'paired_results.npz');assert list(mp['names'])==list(xp['names'])
  result[stage]={str(name):{'medium':stats(mp['gap'][:,i]),'xl':stats(xp['gap'][:,i]),'difference':stats(xp['gap'][:,i]-mp['gap'][:,i])} for i,name in enumerate(mp['names'])}
  text+=['',f'## {stage}','','| Intervention | Medium | XL | XL−Medium paired difference |','|---|---:|---:|---:|']
  for name,data in result[stage].items():text.append('| '+name+' | '+' | '.join(value(data[k]) for k in ['medium','xl','difference'])+' |')
 (out/'comparison.json').write_text(json.dumps(result,indent=2));(out/'comparison_tables.md').write_text('\n'.join(text)+'\n')
 fig,axs=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
 for ax,p in zip(axs.ravel(),'QKVO'):
  for summary,label,color,shift in [(m,'Medium','#2166ac',-.08),(x,'XL','#d6604d',.08)]:
   points=[(l,summary['scenarios'][f'layer_{l:02d}_{p}']) for l in range(24) if f'layer_{l:02d}_{p}' in summary['scenarios']]
   ax.errorbar([l+shift for l,s in points],[s['mean'] for l,s in points],yerr=[1.96*s['se'] for l,s in points],fmt='.-',lw=1,capsize=2,color=color,label=label)
  ax.set(title=f'{p} row: single-layer knockout',xlabel='Layer (zero-based)',ylabel='Loss increase (nats/token)');ax.set_xticks(range(0,24,3));ax.axhline(0,color='gray',lw=.8);ax.grid(alpha=.15);ax.legend()
 fig.suptitle('Medium vs XL · same 64 Pile sequences · matched intervention\nF layers: 2,5,8,11,14,17,20,23; V absent on F. Panel y-scales differ.')
 for ext in ['png','pdf']:fig.savefig(out/f'layer_comparison.{ext}',dpi=180)
 fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True);keys=['QK_all_off','QK_first_half','QK_last_half','QK_first_third','QK_middle_third','QK_last_third'];ticks=np.arange(len(keys))
 for shift,label,key,color in [(-.18,'Medium','medium','#2166ac'),(.18,'XL','xl','#d6604d')]:
  a=[result['depth'][k][key] for k in keys];ax.bar(ticks+shift,[s['mean'] for s in a],.36,yerr=[1.96*s['se'] for s in a],capsize=3,label=label,color=color)
 ax.set_xticks(ticks,['All 0–23','First 0–11','Last 12–23','First 0–7','Middle 8–15','Last 16–23']);ax.set(ylabel='Loss increase (nats/token)',title='Joint Q/K row deletion by depth · same 64 Pile sequences');ax.legend();ax.grid(axis='y',alpha=.15)
 for ext in ['png','pdf']:fig.savefig(out/f'qk_depth_comparison.{ext}',dpi=180)
 print('\n'.join(text[:12]))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('medium',type=Path);p.add_argument('xl',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.medium,a.xl,a.output)
