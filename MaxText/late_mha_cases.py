"""The same all-head cases at every checkpoint; layer8 is a mid-depth reference."""
from no_self_cases import digest
import numpy as np

def plan(bam,layers=32):
 result=[]
 for scope in (['combined','vector'] if bam else ['combined']):
  for chosen in [[7]]+[[i] for i in range(27,32)]+[list(range(27,32))]:
   c=np.ones((layers,4),np.float32)
   for l in chosen:
    c[l,0]=0
    if scope=='combined':c[l,2]=0
   label='last5' if len(chosen)>1 else f'layer{chosen[0]+1:02d}'
   result.append(dict(name=scope+'_'+label,scope=scope,layers_0based=chosen,controls=c.tolist()))
 return result
