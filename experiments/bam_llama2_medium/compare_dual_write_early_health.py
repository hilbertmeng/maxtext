#!/usr/bin/env python3
"""Compare same-step early read/write health and parameter-gradient norms."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--steps',default='0,20,100,200,400,500')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
steps=[int(s) for s in a.steps.split(',')]
source=Path(__file__).resolve().parents[2]/'.agents/skills/tpu-training/scripts/report_bam_read_health.py'
spec=importlib.util.spec_from_file_location('health_reader',source)
r=importlib.util.module_from_spec(spec);sys.modules[spec.name]=r;spec.loader.exec_module(r)
# Separate schema/cache because the standard health reader keeps only W_R gradients.
r.CACHE_ROOT=Path('/data0/xd/bam_diagnostics/dual_write_training/early_health_cache')
r._retain_tag=lambda tag: tag.startswith(('bam/','raw_grads/decoder/','total_params/decoder/')) or tag=='learning/raw_grad_norm'
metrics=['read_mean','main_mean','feedback_mean','feedback_norm_share','read_main_corr','read_feedback_corr','main_feedback_corr']
result={}
for run in ['BamMediumAllLocalDualWriteGates','BamMediumAllLocalRawReadWriteGate']:
 points,_=r._cached_local_points(Path('/data0/xd/tensorboard_logs')/run,steps)
 snapshots=[]
 for step in steps:
  values={tag:data[step] for tag,data in points.items() if step in data}
  assert all(np.isfinite(v) for v in values.values()),(run,step)
  bands={}
  for band,(lo,hi) in {'early':(1,2),'middle':(3,16),'late':(17,22),'all':(1,22)}.items():
   stats={}
   for metric in metrics:
    vals=[values[f'bam/dual_write/layer_{l:03}/head_{h:02}/{metric}'] for l in range(lo,hi+1) for h in range(16)]
    stats[metric]={'mean':float(np.mean(vals)),'q10':float(np.quantile(vals,.1)),'median':float(np.median(vals)),'q90':float(np.quantile(vals,.9))}
   for metric in ['feedback_energy_share','main_lt002','feedback_lt002','main_gt098','feedback_gt098']:
    vals=[values[f'bam/dual_write/layer_{l:03}/head_mean/{metric}'] for l in range(lo,hi+1)]
    stats[metric]={'layer_mean':float(np.mean(vals))}
   norms={}
   for tag,value in values.items():
    match=re.fullmatch(r'(raw_grads|total_params)/decoder/layers_(\d+)/(local_0|local_1|fetch_2)/block/self_attention/(.+)',tag)
    if not match:continue
    kind,block,sub,param=match.groups();layer=3*int(block)+{'local_0':0,'local_1':1,'fetch_2':2}[sub]
    if lo<=layer<=hi:norms.setdefault(kind+'/'+param,[]).append(value)
   stats['parameter_norms']={k:{'median':float(np.median(v)),'rms':float(np.sqrt(np.mean(np.square(v)))),'nonzero':sum(x>0 for x in v),'count':len(v)} for k,v in norms.items()}
   bands[band]=stats
  snapshots.append({'step':step,'bands':bands,'values':values})
  print(run,step,json.dumps({k:round(v['median'],6) for k,v in bands['middle'].items() if isinstance(v,dict) and 'median' in v}))
 result[run]=snapshots
 a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
