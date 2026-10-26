#!/usr/bin/env python3
"""Experiment-only transfer of diagonal suppression across shared/separate VO."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from report_pre_diagonal_full_nope import collect

PREFIX='BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1'
# Existing four-run engine:pre=VO separation,nope=diagonal suppression.
RUNS=dict(base=PREFIX+'TruePile',pre=PREFIX+'SeparateVOKeysTruePile',
          nope=PREFIX+'LogitDiagNegativeTruePile',
          combo=PREFIX+'SeparateVOKeysLogitDiagNegativeTruePile',
          mha='BamMHAMediumPropC256TruePile')


def add_transfer_metrics(data):
  for row in data['series']:
    shared=row['standalone_nope_gain'];separate=row['conditional_nope_gain']
    valid=separate>1e-8
    row.update(negative_gain_shared_vo=shared,negative_gain_separate_vo=separate,
        shared_minus_separate_improvement=shared-separate,
        shared_over_separate_improvement=shared/separate if valid else None,
        transfer_status='ok' if valid else 'nonpositive_or_near_zero_separate_improvement')
  data.update(definition='K=(L_shared-L_shared_negative)/(L_separate-L_separate_negative)',
      symmetric_definition='Q=(L_separate_negative-L_shared)/((L_separate-L_shared)+(L_shared_negative-L_shared))',
      interpretation='K>1 and Q<1 express the same subadditive interaction when gains are positive; not independent evidence')
  return data


def render(data,directory,start=600):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  rows=[r for r in data['series'] if r['step']>=start]
  if not rows:return None
  interval=data.get('interval',200)
  xs=list(range(((start+interval-1)//interval)*interval,rows[-1]['step']+1,interval))
  by={r['step']:r for r in rows}
  fig,axes=plt.subplots(2,2,figsize=(13.5,8.8),sharex=True)
  variants=(('nope','base','negative_gain_shared_vo','Shared VO','#ff7f0e'),
            ('combo','pre','negative_gain_separate_vo','Separate VO','#9467bd'))
  for run,base,key,label,color in variants:
    axes[0,0].plot(xs,[-by[x][key] if x in by else math.nan for x in xs],color=color,marker='o',markersize=2.5,label=label)
    native=next(r for r in data['gain_ratios'] if r['run']==data['runs'][run] and r['comparator']==data['runs'][base])
    ratios={r['step']:r['ratio'] for r in native['series'] if r['status']=='ok'}
    axes[1,0].plot(xs,[ratios.get(x,math.nan) for x in xs],color=color,marker='o',markersize=2.5,label=label+'; reference=MHA')
  for ax,key,label in ((axes[0,1],'shared_over_separate_improvement','K: shared / separate improvement'),
                       (axes[1,1],'symmetric_additivity_ratio','Q: symmetric additivity')):
    ax.plot(xs,[by[x].get(key) if x in by and by[x].get(key) is not None else math.nan for x in xs],color='#111111',marker='o',markersize=2.5,label=label)
  for ax,center,label in zip(axes.flat,(0,1,1,1),('Loss gap vs own parent','Absolute-improvement transfer K','MHA-reference gain multiplier','Symmetric additivity Q')):
    ax.axhline(center,color='#777',linestyle=':',linewidth=1);ax.set_ylabel(label);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
  for ax in axes[1]:ax.set_xlabel('Training step')
  fig.suptitle('Pre-softmax negative diagonal: shared vs separate VO')
  fig.text(.04,.015,'Same-step, identical four-run raw windows. K>1 tests greater improvement on shared VO; Q<1 is the equivalent subadditive interaction.\n'
           'Ordinary gain multipliers use registry values with the same MHA reference. Missing/undefined windows stay blank.',fontsize=8,color='#555')
  fig.tight_layout(rect=(0,.06,1,.97));directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
  path=directory/('shared-vo-pre-diagonal-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')+'.png')
  fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
  path.with_suffix('.json').write_text(json.dumps(dict(data,figure=str(path),plot_start=start,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
  return str(path)


def main():
  p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
  p.add_argument('--through-step',type=int);p.add_argument('--input',type=Path);p.add_argument('--plot-dir',type=Path)
  p.add_argument('--start-step',type=int,default=600);args=p.parse_args()
  data=json.loads(args.input.read_text()) if args.input else collect(args.through_step,runs=RUNS)
  data=add_transfer_metrics(data);args.output.parent.mkdir(parents=True,exist_ok=True)
  args.output.write_text(json.dumps(data,indent=2)+'\n');figure=render(data,args.plot_dir,args.start_step) if args.plot_dir else None
  print(json.dumps(dict(data=str(args.output),through_step=data.get('paired_through'),figure=figure),indent=2))


if __name__=='__main__':main()
