"""Small CPU checks for row-only interception, LLF layer selection, and masks."""
import contextlib
import tempfile
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import row_contribution as r
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from layers.attentions import BamAttention
from unittest import mock


def run():
  scales=jnp.ones((24,4),jnp.float32).at[4,2].set(0)
  x=jnp.arange(128,dtype=jnp.float32).reshape(1,1,2,64)
  np.testing.assert_array_equal(r.scale_row(x,jnp.float32(1)),x)
  y=r.scale_row(x,jnp.float32(0))
  np.testing.assert_array_equal(y[...,:32],x[...,:32]);np.testing.assert_array_equal(y[...,32:],0)
  # XL packs column64 followed by row64; exercise the actual wider boundary.
  wide=jnp.arange(256,dtype=jnp.float32).reshape(1,1,2,128)
  wide_off=r.scale_row(wide,jnp.float32(0),64)
  np.testing.assert_array_equal(wide_off[...,:64],wide[...,:64])
  np.testing.assert_array_equal(wide_off[...,64:],0)
  np.testing.assert_array_equal(r.scale_row(wide,jnp.float32(1),64),wide)
  depth=np.array([s['scales'] for s in r.depth_scenarios()])
  assert depth.shape==(7,24,4) and np.all(depth[:,:,2:]==1)
  assert np.all(depth[2,:,:2]+depth[3,:,:2]==1)
  assert np.all(depth[4,:,:2]+depth[5,:,:2]+depth[6,:,:2]==2)
  names=[s['name'] for s in r.scenarios()];assert len(names)==len(set(names))==141
  assert len([s for s in r.scenarios() if s['kind']=='layer'])==88
  assert not any(s.get('layer',0)%3==2 and s.get('path')=='V' for s in r.scenarios() if s['kind']=='layer')
  # Exercise actual Linen interception with a tiny module retaining the same method contract.
  class Tiny(nn.Module):
    bam_k:int=32
    _fetched_read_num_heads:int=2
    num_query_heads:int=2
    def _read_local(self,name,x):return x
    def _read_fetched_m(self,x):return x,jnp.float32(7)
    @nn.compact
    def __call__(self,x,*,layer_index):
      return tuple(self._read_local(n,x) for n in ('q','k','v'))+(self._read_fetched_m(x),)
  model=Tiny();variables=model.init(jax.random.key(0),x,layer_index=4)
  with mock.patch.object(r.attentions,'BamAttention',Tiny):
    with r.row_interventions(scales):out=model.apply(variables,x,layer_index=4)
    np.testing.assert_array_equal(out[0],x);np.testing.assert_array_equal(out[1],x)
    np.testing.assert_array_equal(out[2],y);np.testing.assert_array_equal(out[3][0],x);assert out[3][1]==7
    with r.row_interventions(scales.at[4,3].set(0)):out=model.apply(variables,x,layer_index=4)
    np.testing.assert_array_equal(out[3][0],y);assert out[3][1]==7
    with r.row_interventions(scales):out=model.apply(variables,x,layer_index=5)
    for a in out[:3]:np.testing.assert_array_equal(a,x)
  print('Row prefix preservation, Q/K/V/O selection, layer selection, gate preservation, scenario grid PASS')

if __name__=='__main__':run()
