#!/usr/bin/env python3
"""Join registered gain ratios; never recompute raw loss windows for this family."""
import argparse,json
from pathlib import Path
from statistics import mean

def ratio(doc, comparator):
 return next(x for x in doc['gain_ratios'] if x['comparator']==comparator)
def points(item):
 return {x['step']:x['ratio'] for x in item['series'] if x.get('ratio') is not None}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--report',type=Path,required=True,help='Registry report-all JSON for both initialization arms')
 p.add_argument('--serial',type=Path,required=True,help='Completed serial BAM loss-report JSON')
 p.add_argument('--old-recipe',type=Path,required=True,help='Matched-circuit old recipe loss-report JSON')
 p.add_argument('--through-step',type=int,required=True)
 p.add_argument('--arm-report',action='append',nargs=2,default=[],metavar=('RUN','JSON'),
                help='Cached registry loss-report for an arm already completed/acknowledged; live pending reports take precedence')
 p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 serial=json.loads(a.serial.read_text());old=json.loads(a.old_recipe.read_text());report=json.loads(a.report.read_text())
 comp='MuddPythiaMediumPropT2048SerialMLP'
 raw={'unfixed':points(ratio(serial,comp)),'old_recipe':points(ratio(old,'MuddLlama2MediumPropC256T2048'))}
 names={'BamPythiaMediumPropT2048SerialMLPBamInitFix':'bam_fix',
        'BamPythiaMediumPropT2048SerialMLPOriginalInit':'all_original'}
 overrides=dict(a.arm_report)
 if set(overrides)-set(names):p.error('Unknown initialization arm in --arm-report')
 entries={entry['run_name']:entry for entry in report['runs']}
 cached_sources={}
 for name,label in names.items():
  doc=entries.get(name,{}).get('pending_report')
  if doc is None and name in overrides:
   source=Path(overrides[name]);doc=json.loads(source.read_text());cached_sources[name]=str(source)
  if doc is None:p.error(f'Missing pending report for {name}; supply --arm-report RUN JSON for a completed arm')
  item=ratio(doc,comp)
  if item.get('run')!=name:p.error(f'Gain report belongs to {item.get("run")}, expected {name}')
  raw[label]=points(item)
 assert set(raw)=={'unfixed','old_recipe','bam_fix','all_original'},set(raw)
 serial_terminal=mean(list(raw['unfixed'].values())[-5:])
 old_late=mean(list(raw['old_recipe'].values())[-5:])
 # A stopped old-recipe reference must not truncate the active primary series.
 common=sorted(set.intersection(*(set(raw[k]) for k in ('unfixed','bam_fix','all_original'))))
 rows=[]
 for s in common:
  if s>a.through_step:continue
  row=dict(step=s,**{k:m.get(s) for k,m in raw.items()})
  for label in ['bam_fix','all_original']:
   row[label+'_vs_unfixed']=row[label]/row['unfixed']
   row[label+'_vs_old_recipe']=row[label]/row['old_recipe'] if row['old_recipe'] is not None else None
   row[label+'_conditional_terminal']=row[label+'_vs_unfixed']*serial_terminal
  rows.append(row)
 result=dict(definition='Each BAM/Mudd ratio uses own-family MHA. Reuses registry window ratios. No raw loss re-averaging.',
   serial_terminal_last5=serial_terminal,old_recipe_latest5=old_late,
   recovery_threshold_new_over_unfixed=old_late/serial_terminal,
   conditional_scope='Conditional terminal numbers ONLY assume the current new/parent ratio persists; not a fitted or observed terminal. Old recipe matched run stopped10160; late windows through10000, not full13500 terminal.',rows=rows)
 if cached_sources:result['cached_arm_report_sources']=cached_sources
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
