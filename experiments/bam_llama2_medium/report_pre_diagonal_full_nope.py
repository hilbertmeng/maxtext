#!/usr/bin/env python3
"""Experiment-only NoPE transfer report; reuse the established four-run pairing."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys

PAIRED_SOURCE = Path('/data0/xd/bam_diagnostics/mediumprop-qknorm/report_qknorm_interaction.py')
PREFIX = 'BamMediumPropMShiftReadG050DedicatedMemoryMLPRatio2to1SeparateVOKeys'
RUNS = dict(combo=PREFIX+'LogitDiagNegativeQK75TruePile', pre=PREFIX+'LogitDiagNegativeTruePile',
            nope=PREFIX+'QK75TruePile', base=PREFIX+'TruePile', mha='BamMHAMediumPropC256TruePile')


def paired_implementation():
  spec = importlib.util.spec_from_file_location('existing_four_run_pairing', PAIRED_SOURCE)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.interaction_series



def paired_metrics(row):
  """Symmetric total-gain normalization of the established paired window means."""
  pre_gain = row['mha_qknorm_gain']  # L_base - L_pre in this four-run orientation.
  nope_gain = row['old_bam_gain']   # L_base - L_nope.
  conditional_gain = row['new_bam_gain']  # L_pre - L_combo.
  total_gain, additive_gain = pre_gain + conditional_gain, pre_gain + nope_gain
  valid = additive_gain > 1e-8
  return dict(step=row['step'], samples=row['samples'],
      standalone_pre_gain=pre_gain, standalone_nope_gain=nope_gain,
      conditional_nope_gain=conditional_gain, combo_total_gain=total_gain,
      standalone_gain_sum=additive_gain, interaction_gain=row['interaction_gain'],
      symmetric_additivity_ratio=total_gain/additive_gain if valid else None,
      symmetric_status='ok' if valid else 'nonpositive_or_near_zero_total_standalone_gain',
      conditional_retention_nope=row['second_order_ratio'], conditional_status=row['status'])


def collect(through=None):
  # Same implementation already used by QKNorm and the dedicated-MLP/tail combinations.
  function = paired_implementation()
  code = ('import json,sys\nsys.path.insert(0,"/home/lishengping/xd/projects")\n'
          'import run_registry as reg\n' + inspect.getsource(function) + '\n' + inspect.getsource(paired_metrics) + '\n'
          + 'names='+repr(RUNS)+'\nthrough='+repr(through)+'\n' + '''
for name in names.values():
 if any(reg.load_run(name).get('loss_step_offsets',{}).values()):raise ValueError('This four-way report requires zero registered step offsets')
caches={role:reg.load_loss_points(name) for role,name in names.items()}
raw=interaction_series([caches[r] for r in ('combo','pre','nope','base')],through)
rows=[paired_metrics(r) for r in raw]
paired_through=rows[-1]['step'] if rows else 0
ordinary=[]
for run,base in [('combo','pre'),('nope','base')]:
 ordinary.append(dict(run=names[run],comparator=names[base],reference=names['mha'],
   series=reg.gain_ratio_series(caches[run],caches[base],caches['mha'],200,25,10,paired_through)))
common_steps={r['step'] for r in rows}
for report in ordinary:report['series']=[r for r in report['series'] if r['step'] in common_steps]
print(json.dumps(dict(generated_utc=reg.now_utc(),runs=names,
 definition='Q=(L_combo-L_base)/((L_pre-L_base)+(L_nope-L_base))',
 conditional_definition='S_nope=(L_pre-L_combo)/(L_base-L_nope)',
 window='200-step centers; identical five raw samples s-20..s+20 by10 across four runs',
 series=rows,gain_ratios=ordinary,paired_through=paired_through,
 cache_horizons={k:max(v) if v else None for k,v in caches.items()}),indent=2))
''')
  result = subprocess.run(['ssh','-S','/tmp/ssh-tpu-ag-xd.sock','tpu-ag','python3 -'],
                          input=code,text=True,capture_output=True,check=True)
  data = json.loads(result.stdout)
  data.update(paired_implementation=str(PAIRED_SOURCE),
              paired_source_sha256=hashlib.sha256(PAIRED_SOURCE.read_bytes()).hexdigest())
  return data


def render(data, directory, start=600):
  # Ordinary MHA ratios remain registry values; S uses the four-way incremental gains above.
  sys.path.insert(0,'/home/xd/projects/maxtext/.agents/skills/tpu-training/scripts')
  from plot_training_report import ratio_value
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  rows = [r for r in data['series'] if r['step']>=start]
  if not rows:
    return None
  actual_steps = [r['step'] for r in rows]
  xs = list(range(((start+199)//200)*200, actual_steps[-1]+1, 200))
  by_step = {r['step']:r for r in rows}
  if any('symmetric_additivity_ratio' not in r for r in rows):
    raise ValueError('Input predates symmetric Q; collect a fresh four-run report')
  fig,axes = plt.subplots(2,2,figsize=(13.5,8.8),sharex=True)
  purple,gray = '#9467bd','#7f7f7f'
  conditional=[-by_step[x]['conditional_nope_gain'] if x in by_step else math.nan for x in xs]
  axes[0,0].plot(xs,conditional,color=purple,marker='o',markersize=2.5,label='Combination - pre-softmax parent')
  ordinary=next(r for r in data['gain_ratios'] if r['run']==data['runs']['combo'] and r['comparator']==data['runs']['pre'])
  native={r['step']:r for r in ordinary['series']}
  axes[1,0].plot(xs,[ratio_value(native.get(x,{})) for x in xs],color=purple,
                 marker='o',markersize=2.5,label='Combination / pre-softmax parent; reference=MHA')
  for key,label,color in [('combo_total_gain','Combination - original base',purple),
                          ('standalone_gain_sum','Sum of both standalone loss effects',gray)]:
    vals=[-by_step[x][key] if x in by_step else math.nan for x in xs]
    axes[0,1].plot(xs,vals,color=color,marker='o',markersize=2.5,label=label)
  q=[by_step[x]['symmetric_additivity_ratio'] if x in by_step and by_step[x]['symmetric_status']=='ok' else math.nan for x in xs]
  axes[1,1].plot(xs,q,color='#111111',marker='o',markersize=2.5,label='Symmetric gain additivity Q')
  for axis,center,label in zip(axes.flat,(0,0,1,1),
      ('Loss gap vs pre-softmax parent','Loss effects vs original base',
       'Ordinary MHA-reference gain multiplier','Symmetric additivity Q')):
    axis.axhline(center,color='#777',linestyle=':',linewidth=1)
    axis.set_ylabel(label);axis.grid(alpha=.2);axis.legend(frameon=False,fontsize=7)
  for axis in axes[1]:axis.set_xlabel('Training step')
  fig.suptitle('Pre-softmax diagonal suppression + full NoPE: symmetric additivity')
  fig.text(.04,.012,'Q=(L_combo - L_base)/[(L_pre - L_base)+(L_nope - L_base)]: 1=full additive benefit; >1=enhanced; <1=discounted.\n'
           'Exact common four-run windows; missing/undefined values stay blank. MHA is only the ordinary-gain reference.',fontsize=8,color='#555')
  fig.tight_layout(rect=(0,.055,1,.97))
  directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
  name='pre-diagonal-full-nope-symmetric-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
  path=directory/(name+'.png');fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
  meta=dict(data,figure=str(path),plot_start=start,
            paired_through=xs[-1],renderer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  path.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
  return str(path)


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--output',type=Path,required=True);p.add_argument('--through-step',type=int)
  p.add_argument('--input',type=Path);p.add_argument('--plot-dir',type=Path);p.add_argument('--start-step',type=int,default=600)
  args=p.parse_args();data=json.loads(args.input.read_text()) if args.input else collect(args.through_step)
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(data,indent=2)+'\n')
  figure=render(data,args.plot_dir,args.start_step) if args.plot_dir else None
  print(json.dumps(dict(data=str(args.output),through_step=data['series'][-1]['step'] if data['series'] else None,
                       figure=figure),indent=2))


if __name__=='__main__':main()
