"""Check side, layer, path, gates and complete coarse scenario scope."""
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from unittest import mock
import column_contribution as c

class Tiny(nn.Module):
 bam_k:int
 def _read_local(self,name,x):return x
 def _read_fetched_m(self,x):return x,jnp.float32(7)
 @nn.compact
 def __call__(self,x,*,layer_index):return tuple(self._read_local(n,x) for n in ('q','k','v'))+(self._read_fetched_m(x),)
for width,k in [(64,32),(128,64)]:
 x=jnp.arange(2*16*width,dtype=jnp.float32).reshape(1,2,16,width)+1;model=Tiny(k);params=model.init(jax.random.key(0),x,layer_index=4)
 scales=np.ones(c.SHAPE,np.float32);scales[4,0,0]=0;scales[4,2,1]=0;scales[4,3,:]=0
 with mock.patch.object(c.base.attentions,'BamAttention',Tiny):
  def apply(l):
   with c.side_interventions(jnp.asarray(scales)):return model.apply(params,x,layer_index=l)
  values=jax.jit(apply)(jnp.int32(4));expected=np.asarray(x).copy();expected[...,:k]=0;np.testing.assert_array_equal(values[0],expected)
  np.testing.assert_array_equal(values[1],x);expected=np.asarray(x).copy();expected[...,k:]=0;np.testing.assert_array_equal(values[2],expected)
  np.testing.assert_array_equal(values[3][0],np.zeros_like(x));assert values[3][1]==7
  values=jax.jit(apply)(jnp.int32(5))
  for v in (*values[:3],values[3][0]):np.testing.assert_array_equal(v,x)
s=c.scenarios();assert all(x['side']=='row' for x in s[:16]);assert len({x['id'] for x in s})==len(s)
for x in s:
 a=np.asarray(x['scales']);assert a.shape==c.SHAPE
 if x['side']=='column':assert (a[:,:,1]==1).all()
 if x['side']=='row':assert (a[:,:,0]==1).all()
 if x['side']=='both':np.testing.assert_array_equal(a[:,:,0],a[:,:,1])
assert len([x for x in s if x['side']=='column' and x['kind']=='layer'])==88
assert len([x for x in s if x['side']=='both' and x['kind']=='layer'])==88
print('COLUMN_TEST_PASS',len(s),'scenarios; two widths dynamic layer, side/path isolation, unchanged gate')
for key in ['bam_local_v_direct_compressed_col','bam_local_q_rank','bam_local_v_rank','bam_local_qk_share_basis','bam_abs_v_compression_dim','bam_n_f']:
 print('CONFIG',key,getattr(c.base.RowContributionProbe,key,None))
