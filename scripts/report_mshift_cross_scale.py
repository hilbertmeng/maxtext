#!/usr/bin/env python3
"""M-shift family: compare MHA-referenced gain multipliers at equal schedule fractions."""
import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
MED=('BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile',
     'BamMediumPropK75EmbedVOnlyQK57AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile',
     'BamMHAMediumPropC256TruePile')
XL=('BamXLPropSharedR400GeneralReadMShiftReadG050TruePile',
    'BamXLPropK96EmbedVOnlyQK72AllLocalMLPWriteIndependentEveryThirdFullMReadGeneralReadTruePile',
    'Llama2XLPropTruePileMHA')
def multiplier(caches,center,radius=25,period=10):
    points=[s for s in range(center-radius,center+radius+1) if s%period==0]
    if not points or not all(s in c for c in caches for s in points):
        return dict(status='incomplete_window',ratio=None,points=points)
    means=[sum(c[s] for s in points)/len(points) for c in caches]
    denominator=means[2]-means[1]
    if denominator<=1e-8:return dict(status='nonpositive_parent_gain',ratio=None,points=points)
    return dict(status='ok',ratio=(means[2]-means[0])/denominator,points=points)
def aligned_series(med,xl,med_steps=13500,xl_steps=50000,stride_percent=2,through_xl_step=None):
    if med_steps<=0 or xl_steps<=0 or stride_percent<=0:raise ValueError('positive schedules/stride required')
    rows=[]
    for percent in range(stride_percent,101,stride_percent):
        m=Fraction(percent*med_steps,100);x=Fraction(percent*xl_steps,100)
        # Exact stages only: never round to a neighbouring native reporting center.
        if m.denominator!=1 or x.denominator!=1:continue
        m,x=int(m),int(x)
        if through_xl_step is not None and x>through_xl_step:break
        row=dict(percent=percent,medium_step=m,xl_step=x,
                 medium=multiplier(med,m),xl=multiplier(xl,x))
        rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--through-xl-step',type=int)
    p.add_argument('--stride-percent',type=int,default=2)
    p.add_argument('--json',action='store_true')
    p.add_argument('--registry-script',type=Path,default=Path('/home/lishengping/xd/projects/run_registry.py'))
    a=p.parse_args()
    if a.stride_percent<=0:p.error('stride-percent must be positive')
    spec=importlib.util.spec_from_file_location('scale_registry',a.registry_script)
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    med_meta=r.load_run(MED[0]);xl_meta=r.load_run(XL[0])
    med_steps=int(med_meta['planned_steps']);xl_steps=int(xl_meta['planned_steps'])
    caches=[[r.load_loss_points(name) for name in names] for names in (MED,XL)]
    rows=aligned_series(*caches,med_steps,xl_steps,a.stride_percent,a.through_xl_step)
    mature=[v for v in rows if v['xl']['status']!='incomplete_window']
    report=dict(formula='(MHA-MShift)/(MHA-parentSOTA)',alignment='step/planned_steps',
                medium_runs=MED,xl_runs=XL,medium_planned_steps=med_steps,xl_planned_steps=xl_steps,
                series=mature,pending=[v['percent'] for v in rows if v not in mature])
    if a.json:print(json.dumps(report,indent=2))
    else:
        print('MSHIFT_CROSS_SCALE formula=(MHA-MShift)/(MHA-parentSOTA) alignment=step/planned_steps')
        print('phase% medium_step xl_step medium_gain_multiple xl_gain_multiple')
        for v in mature:
            fmt=lambda k:f"{v[k]['ratio']:.6f}" if v[k]['ratio'] is not None else '--:'+v[k]['status']
            print(v['percent'],v['medium_step'],v['xl_step'],fmt('medium'),fmt('xl'))
        if not mature:print('PENDING no complete XL paired window')
if __name__=='__main__':main()
