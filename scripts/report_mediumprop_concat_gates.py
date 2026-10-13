"""Compact concat-gate health using the shared incremental scalar cache."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',nargs='+');p.add_argument('--steps',nargs='+',type=int,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--repository',type=Path,default=Path('/home/xd/projects/maxtext'))
    a=p.parse_args();helper=a.repository/'.agents/skills/tpu-training/scripts/report_bam_read_health.py'
    spec=importlib.util.spec_from_file_location('bam_cached_scalars',helper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    pattern=re.compile(r'^bam/concat/([^/]*gate[^/]*)/layer_(\d+)/(mean|std|frac_gt_095|frac_lt_005)$')
    out={}
    for run in a.runs:
        scalar=module.Scalars(Path('/data0/xd/tensorboard_logs')/run,a.steps)
        rows={};bad=[]
        for step in a.steps:
            values={}
            for tag in scalar.tags:
                match=pattern.match(tag)
                if not match:continue
                value=scalar._points.get(tag,{}).get(step)
                if value is None:continue
                if not math.isfinite(value):bad.append({'tag':tag,'step':step});continue
                values.setdefault(match[1],{}).setdefault(match[3],[]).append(value)
            rows[step]={name:{key:{'mean':sum(v)/len(v),'min':min(v),'max':max(v),'layers':len(v)}
                               for key,v in metrics.items()} for name,metrics in sorted(values.items())}
            print(run,step,{name:round(metrics['mean']['mean'],6) for name,metrics in rows[step].items() if 'mean' in metrics})
        out[run]={'steps':rows,'nonfinite':bad}
    if a.output:a.output.write_text(json.dumps(out,indent=2)+'\n')
    if any(row['nonfinite'] for row in out.values()):raise SystemExit('Nonfinite gate health')

if __name__=='__main__':main()
