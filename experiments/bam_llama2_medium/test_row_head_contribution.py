"""Verify head/output slice selection, path/layer isolation, gates, and scenario coverage."""
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from unittest import mock
import row_head_contribution as h

class Tiny(nn.Module):
 bam_k:int
 num_query_heads:int=16
 _fetched_read_num_heads:int=16
 def _read_local(self,name,x):return x
 def _read_fetched_m(self,x):return x,jnp.float32(7)
 @nn.compact
 def __call__(self,x,*,layer_index):
  return tuple(self._read_local(n,x) for n in ('q','k','v'))+(self._read_fetched_m(x),)

for width,bam_k in [(64,32),(128,64)]:
 x=jnp.arange(2*16*width,dtype=jnp.float32).reshape(1,2,16,width)+1
 keep=np.ones(h.SHAPE,bool);keep[4,2,3]=False;keep[4,3,7]=False
 model=Tiny(bam_k);variables=model.init(jax.random.key(0),x,layer_index=4)
 with mock.patch.object(h.base.attentions,'BamAttention',Tiny):
  def apply(index):
   with h.head_interventions(jnp.asarray(keep)):return model.apply(variables,x,layer_index=index)
  result=jax.jit(apply)(jnp.int32(4))
  for value in result[:2]:np.testing.assert_array_equal(value,x)
  for value,head in [(result[2],3),(result[3][0],7)]:
   expected=np.asarray(x).copy();expected[...,head,bam_k:]=0
   np.testing.assert_array_equal(value,expected)
  assert result[3][1]==7
  other=jax.jit(apply)(jnp.int32(5))
  for value in (*other[:3],other[3][0]):np.testing.assert_array_equal(value,x)
scenarios=h.scenarios();assert len(scenarios)==1349
names=set()
for s in scenarios:
 assert s['name'] not in names;names.add(s['name'])
 mask=h.mask_for(s)
 if s['kind']=='head':
  assert np.count_nonzero(~mask)==1
  assert s['layer']!=0 and not(s['path']=='V' and s['layer']%3==2)
print('HEAD_TEST_PASS both widths, dynamic layer, single head, other paths/columns, gate, complete1344 grid')
