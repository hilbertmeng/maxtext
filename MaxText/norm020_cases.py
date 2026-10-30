"""Deterministic frozen-activation interventions; neutral1/keep,0/remove."""
import hashlib,json
import numpy as np
def plan(step,main_step,layers=32):
 out=[]
 def add(name,scope,branch,layer=None):
  c=np.ones((layers,4),np.float32);rows=slice(None) if layer is None else layer
  columns={'combined':{'mha':[0,2],'mo':[1,3],'both':[0,1,2,3]},'vector':{'mha':[0],'mo':[1]},'write':{'mha':[2],'mo':[3]}}[scope][branch]
  for col in columns:c[rows,col]=0
  out.append(dict(name=name,scope=scope,branch=branch,layer_0based=layer,controls=c.tolist()))
 for branch in ['mha','mo','both']:add('all_remove_'+branch,'combined',branch)
 if True:
  for layer in range(layers):
   for branch in ['mha','mo']:add(f'layer{layer+1:02d}_remove_'+branch,'combined',branch,layer)
  for scope in ['vector','write']:
   for branch in ['mha','mo']:add('all_'+scope+'_remove_'+branch,scope,branch)
  for layer in range(layers):
   for branch in ['mha','mo']:add(f'layer{layer+1:02d}_vector_remove_'+branch,'vector',branch,layer)
  for lo,hi in [(0,8),(8,16),(16,24),(24,32),(16,32)]:
   for scope in ['combined','vector']:
    for branch in ['mha','mo']:
     add(f'L{lo+1:02d}_{hi:02d}_{scope}_remove_{branch}',scope,branch,slice(lo,hi))
     out[-1]['layer_0based']=None
     out[-1]['layer_range_0based']=[lo,hi]
 return out
def digest(cases):return hashlib.sha256(json.dumps(cases,sort_keys=True,separators=(',',':')).encode()).hexdigest()
