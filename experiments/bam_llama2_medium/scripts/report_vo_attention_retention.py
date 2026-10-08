import argparse,importlib.util,json,math,pathlib,statistics
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
s=importlib.util.spec_from_file_location('hr','/home/xd/projects/maxtext/.agents/skills/tpu-training/scripts/report_bam_read_health.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ap=argparse.ArgumentParser();ap.add_argument('runs',nargs='+');ap.add_argument('--output',required=True);ap.add_argument('--through',type=int,default=None);a=ap.parse_args();out={}
for run in a.runs:
 points={}
 for f in (pathlib.Path('/data0/xd/tensorboard_logs')/run).glob('events.out.tfevents.*'):
  for e in EventFileLoader(str(f)).Load():
   if a.through is not None and e.step>a.through:break
   for v in e.summary.value:
    n=m._scalar_value(v)
    if n is not None and (v.tag.startswith('bam/concat/') or v.tag=='learning/raw_grad_norm'):points.setdefault(e.step,{})[v.tag]=n
 last=max(points);windows={}
 for target in [0,10,30]+list(range(200,last-19,200)):
  steps=[x for x in points if x==target] if target<200 else [x for x in points if abs(x-target)<=25]
  if not steps:continue
  layers={}
  for layer in (0,8,17):
   def val(part,metric):
    tag=f'bam/concat/{part}/layer_{layer:03d}/{metric}';z=[points[x][tag] for x in steps if tag in points[x]]
    return statistics.mean(z) if z else None
   static=val('static_v_amplitude','bam_rms');dynamic=val('static_v_amplitude','standard_rms');o=val('local_o_amplitude','bam_rms');y=val('local_o_amplitude','standard_rms')
   # Total V is directly logged; no independent-route quadrature approximation.
   v=val('local_v_content','rms')
   row=dict(v_static=static,v_dynamic=dynamic,v_total=v,attention_output=y,local_o=o,v_gate=val('local_v_gate','mean'),o_gate=val('local_o_gate','mean'))
   if v and y is not None:row['attention_retention']=y/v
   if y and o is not None:row['o_over_attention']=o/y
   layers[str(layer)]=row
  grads=[points[x]['learning/raw_grad_norm'] for x in steps if 'learning/raw_grad_norm' in points[x]]
  windows[str(target)]={'steps':steps,'layers':layers,'raw_grad_norm':statistics.mean(grads) if grads else None}
 out[run]={'last_step':last,'windows':windows,'available_v_content_tags':sorted({tag for x in points.values() for tag in x if 'local_v_content' in tag})[:4]}
pathlib.Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
