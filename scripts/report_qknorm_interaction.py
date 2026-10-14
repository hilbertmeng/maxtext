#!/usr/bin/env python3
"""Paired BAM/MHA QKNorm interaction, using persistent registry loss caches."""
import argparse,importlib.util,json
from pathlib import Path
NAMES=('BamMediumPropMShiftReadG050FullQKNormTruePile','BamMHAMediumPropC256QKNormTruePile','BamMediumPropSharedR256GeneralReadMShiftReadG050TruePile','BamMHAMediumPropC256TruePile')
def interaction_series(caches,through=None):
 if len(caches)!=4 or any(not c for c in caches):return []
 horizon=min(max(c) for c in caches)-25
 if through is not None:horizon=min(horizon,through)
 rows=[]
 for step in range(200,horizon+1,200):
  pts=[x for x in range(step-25,step+26) if x%10==0]
  if not all(x in c for c in caches for x in pts):continue
  bq,mq,b,m=[sum(c[x] for x in pts)/len(pts) for c in caches]
  old=m-b;new=mq-bq;status='ok' if old>1e-8 else 'nonpositive_or_near_zero_old_bam_gain'
  rows.append(dict(step=step,samples=len(pts),bam_qknorm_gain=b-bq,mha_qknorm_gain=m-mq,old_bam_gain=old,new_bam_gain=new,interaction_gain=new-old,second_order_ratio=new/old if status=='ok' else None,status=status))
 return rows
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--through-step',type=int);p.add_argument('--runs',nargs=4,default=NAMES,metavar=('BAM_QK','MHA_QK','BAM','MHA'));p.add_argument('--registry-script',type=Path,default=Path('/home/lishengping/xd/projects/run_registry.py'));a=p.parse_args()
 spec=importlib.util.spec_from_file_location('registry_qknorm_cache',a.registry_script);reg=importlib.util.module_from_spec(spec);spec.loader.exec_module(reg)
 print(json.dumps(dict(runs=a.runs,series=interaction_series([reg.load_loss_points(n) for n in a.runs],a.through_step)),indent=2))
