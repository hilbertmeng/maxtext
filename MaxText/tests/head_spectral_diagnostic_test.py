"""Head observation does not change native math; FP32 gradient independent check."""
from typing import Any
import jax,jax.numpy as jnp,numpy as np
from flax import linen as nn
from layers.models import OutputHead
import max_utils
import pythia_test as reference
from head_spectral_diagnostic import HeadCapture,precise_cross_entropy,native_ce,FP32_readout,BF16_round
class WrappedHead(nn.Module):
 cfg:Any
 mesh:Any
 @nn.compact
 def __call__(self,x,y,mask):return OutputHead(self.cfg,None,self.mesh,name='lm_head')(x,y,mask,chunk_size=8)
def test_head():
 t=reference.PythiaTest();t.setUp()
 try:
  c=t.config(base_emb_dim=40,head_dim=20,pythia_rotary_dim=4,vocab_size=64,dtype='bfloat16');_,mesh=t.model(c);m=WrappedHead(c,mesh)
  x=jnp.asarray(np.random.default_rng(73).normal(size=(1,8,40)),jnp.bfloat16);y=jnp.arange(8)[None];mask=jnp.ones_like(y)
  max_utils.cross_entropy_with_logits=precise_cross_entropy
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   v=nn.unbox(m.init(jax.random.PRNGKey(5),x,y,mask));native=m.apply(v,x,y,mask)
   observer=HeadCapture()
   with nn.intercept_methods(observer.interceptor):observed=m.apply(v,x,y,mask)
  np.testing.assert_array_equal(native[0],observed[0]);np.testing.assert_array_equal(observer.raw,x)
  w=v['params']['lm_head']['logits_dense'];np.testing.assert_allclose(native_ce(observer.h,w,y),native[0],atol=1e-6,rtol=1e-6)
  g,_,_=FP32_readout(observer.h,w,y);hf=observer.h.astype(jnp.float32);wf=w.astype(jnp.bfloat16).astype(jnp.float32)
  oracle=jax.grad(lambda q:-jnp.take_along_axis(jax.nn.log_softmax(hf@q),y[...,None],-1).mean())(wf)
  np.testing.assert_allclose(g,oracle,rtol=2e-5,atol=1e-6)
  vals=jnp.asarray(np.random.default_rng(74).normal(size=1024)*100,jnp.float32);np.testing.assert_array_equal(BF16_round(vals),vals.astype(jnp.bfloat16).astype(jnp.float32))
  print('NATIVE_HEAD_OBSERVATION_CE_GRADIENT_AND_EXPLICIT_ROUNDING_PASS',flush=True)
 finally:t.tearDown()
if __name__=='__main__':test_head()
