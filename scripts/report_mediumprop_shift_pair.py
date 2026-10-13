#!/usr/bin/env python3
"""One refresh of the stacked RUN covers both active shift arms and their additivity."""
import argparse,importlib.util,json
from pathlib import Path
from report_mediumprop_shift_additivity import JOINT,SINGLE,KV,SOTA,additivity_series,print_additivity_series

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--through-step',type=int)
 p.add_argument('--columns',type=int,default=5)
 p.add_argument('--json',action='store_true')
 p.add_argument('--no-refresh',action='store_true')
 p.add_argument('--registry-script',type=Path,default=Path('/home/lishengping/xd/projects/run_registry.py'))
 a=p.parse_args();spec=importlib.util.spec_from_file_location('paired_registry',a.registry_script)
 r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);meta=r.load_run(JOINT)
 # The joint's direct baselines include KV: this single call refreshes both live logs.
 loss=r.build_loss_reports(meta,a.through_step,10,not a.no_refresh)
 joint_points=r.load_loss_points(JOINT)
 through=max(0,max(joint_points)-int(meta['loss_window'])) if joint_points else 0
 if a.through_step is not None:through=min(through,a.through_step)
 gains=r.build_gain_ratio_reports(meta,through,10,False)
 gains+=r.build_gain_ratio_reports(r.load_run(KV),through,10,False)
 add=dict(run=JOINT,arm_a=SINGLE,arm_b=KV,baseline=SOTA,
          series=additivity_series(*(r.load_loss_points(n) for n in (JOINT,SINGLE,KV,SOTA)),
                 int(meta['loss_interval']),int(meta['loss_window']),10,through))
 report=dict(through_step=through,loss_reports=loss,gain_ratios=gains,additivity=add)
 if a.json:print(json.dumps(report,indent=2))
 else:
  for v in loss:r.print_loss_series(v['run'],v['base'],v['series'],int(meta['loss_interval']),int(meta['loss_window']),10,a.columns)
  for v in gains:r.print_gain_ratio_series(v,a.columns)
  print_additivity_series(add,a.columns)
if __name__=='__main__':main()
