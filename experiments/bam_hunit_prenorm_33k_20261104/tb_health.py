from pathlib import Path
import json,numpy as np
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
from tensorboard.util.tensor_util import make_ndarray
R=Path(__file__).resolve().parent
keys=['learning/raw_grad_norm','learning/grad_norm','learning/loss','raw_grads/decoder/lm_head/logits_dense','logits/common_offset_abs_p95','logits/centered_std_median']
out={}
for name in ['BamPythia2p8BHUnit300B','BamPythia2p8BHUnit300BFastKernelMPreNormQKScale']:
 files=sorted(Path('/data0/xd/tensorboard_logs',name).glob('events*'));index=[]
 for p in files:
  for e in EventFileLoader(str(p)).Load():
   if e.step>0 and any(v.tag=='learning/raw_grad_norm' for v in e.summary.value):
    index.append((e.step,p));break
 index.sort();chosen=[p for i,(s,p) in enumerate(index) if s<=33000 and (i+1==len(index) or index[i+1][0]>=32901)]
 values={k:{} for k in keys}
 for p in chosen:
  for e in EventFileLoader(str(p)).Load():
   if not 32901<=e.step<=33000:continue
   for v in e.summary.value:
    if v.tag in values:values[v.tag][e.step]=float(np.asarray(make_ndarray(v.tensor)).reshape(-1)[0])
 out[name]={'files':[p.name for p in chosen],'metrics':{}}
 for k,vs in values.items():
  if vs:out[name]['metrics'][k]={'count':len(vs),'first':min(vs),'last':max(vs),'mean':float(np.mean(list(vs.values()))),'min':min(vs.values()),'max':max(vs.values())}
 print(name,json.dumps(out[name]['metrics']),flush=True)
(R/'tb_health.json').write_text(json.dumps(out,indent=2)+'\n')
