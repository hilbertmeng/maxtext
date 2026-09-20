import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from unittest import mock
import std_v_dose as c

class Tiny(nn.Module):
 _local_o:bool=True
 fused:bool=False
 def kv_projection(self,x,proj_name):return x*(2 if proj_name=='value' else 3)
 def qkv_projection(self,x,proj_name):return x*4,x*3,x*2
 def _read_local(self,name,x):return x*5
 @nn.compact
 def __call__(self,x,*,layer_index):
  if self.fused:q,k,v=self.qkv_projection(x,'qkv_proj')
  else:q=x*4;k=self.kv_projection(x,'key');v=self.kv_projection(x,'value')
  bam=self._read_local('v',x).at[...,32:].set(0)
  return q,k,v+bam,self._read_local('q',x)
x=jnp.ones((1,2,16,64));s=np.ones(c.SHAPE,np.float32);s[3]=[0,.5,1]
for local in [True,False]:
 for fused in [True,False]:
  m=Tiny(local,fused);p=m.init(jax.random.key(0),x,layer_index=3)
  with mock.patch.object(c.attentions,'BamAttention',Tiny):
   def run(layer,scales):
    with c.interventions(scales):return m.apply(p,x,layer_index=layer)
   q,k,v,other=jax.jit(run)(jnp.int32(3),jnp.asarray(s))
   np.testing.assert_array_equal(q,x*4);np.testing.assert_array_equal(k,x*3);np.testing.assert_array_equal(other,x*5)
   np.testing.assert_array_equal(v[...,:32],np.ones_like(v[...,:32])*(5 if local else 7))
   np.testing.assert_array_equal(v[...,32:],np.ones_like(v[...,32:])*(1 if local else 2))
   off=s.copy();off[3,2]=0;v=jax.jit(run)(jnp.int32(3),jnp.asarray(off))[2]
   np.testing.assert_array_equal(v[...,:32],np.ones_like(v[...,:32])*(0 if local else 7))
   v=jax.jit(run)(jnp.int32(4),jnp.asarray(s))[2];np.testing.assert_array_equal(v[...,:32],np.ones_like(v[...,:32])*7)
ss=c.scenarios();assert len(ss)==1000
for a in ss:
 s=np.asarray(a['scales']);assert np.all(s[2::3]==1)
 if a['scope']=='ordinary_L':assert np.all(s[:2]==1)
print('STD_V_TEST_PASS raw halves, BAM retained, independent BAM-off, QK isolation, local/fetch, dynamic layer, fused/unfused; 1000 scenarios')
m=Tiny(True,False);p=m.init(jax.random.key(0),x,layer_index=3)
with mock.patch.object(c.attentions,'BamAttention',Tiny):
 def objective(scales):
  with c.interventions(scales):return m.apply(p,x,layer_index=jnp.int32(3))[2].sum()
 g=jax.jit(jax.grad(objective))(jnp.ones(c.SHAPE));expected=np.zeros(c.SHAPE,np.float32);expected[3]=[2048,2048,5120]
 np.testing.assert_array_equal(g,expected)
print('STD_V_GRAD_TEST_PASS exact native scaling derivatives through pre-injection halves')
