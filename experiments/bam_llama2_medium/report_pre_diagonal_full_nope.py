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


def collect(through=None):
  # Same implementation already used by QKNorm and the dedicated-MLP/tail combinations.
  function = paired_implementation()
  code = ('import json,sys\nsys.path.insert(0,"/home/lishengping/xd/projects")\n'
          'import run_registry as reg\n' + inspect.getsource(function) + '\n'
          + 'names='+repr(RUNS)+'\nthrough='+repr(through)+'\n' + '''
for name in names.values():
 if any(reg.load_run(name).get('loss_step_offsets',{}).values()):raise ValueError('This four-way report requires zero registered step offsets')
caches={role:reg.load_loss_points(name) for role,name in names.items()}
raw=interaction_series([caches[r] for r in ('combo','pre','nope','base')],through)
rows=[dict(step=r['step'],samples=r['samples'],standalone_nope_gain=r['old_bam_gain'],
           conditional_nope_gain=r['new_bam_gain'],interaction_gain=r['interaction_gain'],
           second_order_ratio=r['second_order_ratio'],status=r['status']) for r in raw]
paired_through=rows[-1]['step'] if rows else 0
ordinary=[]
for run,base in [('combo','pre'),('nope','base')]:
 ordinary.append(dict(run=names[run],comparator=names[base],reference=names['mha'],
   series=reg.gain_ratio_series(caches[run],caches[base],caches['mha'],200,25,10,paired_through)))
common_steps={r['step'] for r in rows}
for report in ordinary:report['series']=[r for r in report['series'] if r['step'] in common_steps]
print(json.dumps(dict(generated_utc=reg.now_utc(),runs=names,
 definition='S=(L_pre-L_combo)/(L_base-L_nope)',
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
  fig,axes = plt.subplots(3,1,figsize=(12.8,10.5),sharex=True)
  for key,label,color in [('standalone_nope_gain','Full NoPE - original base','#7f7f7f'),
                          ('conditional_nope_gain','Combination - pre-softmax parent','#9467bd')]:
    y=[-by_step[x][key] if x in by_step else math.nan for x in xs]
    axes[0].plot(xs,y,marker='o',markersize=2.5,color=color,label=label)
  for report,color,label in zip(data['gain_ratios'],('#9467bd','#7f7f7f'),
      ('Combination / pre-softmax parent','Full NoPE / original base')):
    series={r['step']:r for r in report['series'] if r['step']>=start}
    y=[ratio_value(series.get(x,{})) for x in xs]
    axes[1].plot(xs,y,marker='o',markersize=2.5,color=color,label=label+'; reference=MHA')
  vals=[by_step[x]['second_order_ratio'] if x in by_step and by_step[x]['status']=='ok' and by_step[x]['second_order_ratio'] is not None else math.nan for x in xs]
  axes[2].plot(xs,vals,color='#111111',marker='o',markersize=2.5,label='NoPE incremental gain retained after pre-softmax masking')
  for axis,center,label in zip(axes,(0,1,1),
      ('NoPE loss effect (lower is better)','Ordinary MHA-reference gain multiplier','Second-order NoPE retention S')):
    axis.axhline(center,color='#777',linestyle=':',linewidth=1)
    axis.set_ylabel(label);axis.grid(alpha=.2);axis.legend(frameon=False,fontsize=8)
  axes[2].set_xlabel('Training step')
  fig.suptitle('Does full NoPE retain its benefit after pre-softmax diagonal suppression?')
  fig.text(.04,.012,'S=(L_pre - L_combo)/(L_base - L_nope): 1=full additive retention; >1=enhanced; <1=discounted.\n'
           'Same four-run raw windows; no MHA denominator in S. Missing/undefined values stay blank.',fontsize=8,color='#555')
  fig.tight_layout(rect=(0,.05,1,.97))
  directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
  name='pre-diagonal-full-nope-interaction-'+datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
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
