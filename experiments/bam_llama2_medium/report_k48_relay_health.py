import importlib.util,json,sys,math
from pathlib import Path
spec=importlib.util.spec_from_file_location('reader','/home/xd/projects/maxtext/.agents/skills/tpu-training/scripts/report_bam_read_health.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
run=sys.argv[1];steps=list(map(int,sys.argv[2].split(',')));s=m.Scalars(Path('/data0/xd/tensorboard_logs')/run,steps)
result={}
for arm in (['qk','vo'] if run.endswith('QKVO') else ['vo'] if run.endswith('VOOnly') else ['all']):
 prefix='bam/m_relay'+('/'+arm if arm!='all' else '')
 result[arm]={stat:{band:[s.band_mean(prefix+'/layer_{layer:03d}/'+stat,step,ls) for step in steps] for band,ls in {'L3-7':range(3,8),'L8-15':range(8,16),'L16-23':range(16,24)}.items()} for stat in ['scale_mean','positive_fraction','negative_fraction','saturated_fraction','delta_over_m','anchor_m_cosine','mixed_over_m','amplitude_scale','effective_scale_mean','effective_scale_abs_mean']}
print(json.dumps(dict(run=run,steps=steps,metrics=result),indent=2,allow_nan=False))
