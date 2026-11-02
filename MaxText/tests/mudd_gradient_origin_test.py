"""Native telescoping, final-router source, and signed-gradient contracts."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.traverse_util import flatten_dict,unflatten_dict
import pythia_test as reference
import mudd_spectral_capture as cap
from mudd_gradient_origin_capture import OriginCapture,normalized_integrated_parts

def test_native_origin():
 t=reference.PythiaTest();t.setUp()
 try:
  c=t.config(exp_class='MuddPythiaMediumPropT2048SerialMLP',base_num_decoder_layers=3,base_mlp_dim=256,pythia_rotary_dim=4,mudd_router_hidden_expand=[1,1,4],scan_layers=False,dtype='bfloat16',remat_policy='none')
  m,mesh,p=t.init(c);p=jax.tree.map(lambda v:v.unbox(apply_constraint=False) if isinstance(v,nn.spmd.LogicallyPartitioned) else v,p,is_leaf=lambda v:isinstance(v,nn.spmd.LogicallyPartitioned))
  f=flatten_dict(p)
  for k,a in f.items():
   if k[-2:]==('dynamic_dense_conn2','kernel'):f[k]=jnp.asarray(np.random.default_rng(a.size).normal(0,.12,a.shape),a.dtype)
  p=unflatten_dict(f);ln=p['decoder']['lm_head']['decoder_norm'];basis=jnp.asarray(np.linalg.qr(np.random.default_rng(42).normal(size=(32,4)))[0],jnp.float32)
  import max_utils
  max_utils.cross_entropy_with_logits=cap.precise_cross_entropy
  with mesh,nn.partitioning.axis_rules(c.logical_axis_rules):
   args=t.inputs();native=m.apply({'params':p},*args,enable_dropout=False)[0]
   observer=OriginCapture(ln['scale'],ln['bias'],basis,3)
   with nn.intercept_methods(observer.interceptor):out,col=m.apply({'params':p},*args,enable_dropout=False,mutable=['intermediates'])
   r,names=observer.result(col['intermediates'])
  np.testing.assert_array_equal(out[0],native)
  np.testing.assert_allclose(r['raw_component_sum'],r['raw_projection'],rtol=2e-5,atol=2e-5)
  np.testing.assert_allclose(r['projected_components'].sum(0),r['head_projection'],rtol=2e-5,atol=2e-5)
  i=names.index('mudd_final_readout');np.testing.assert_allclose(r['projected_final_sources'].sum(0),r['projected_components'][i],rtol=2e-5,atol=2e-5)
  assert len(names)==9 and r['projected_final_sources'].shape[0]==5
  for label in ['global','final']:
   np.testing.assert_allclose(r[label+'_IG_sum'],r[label+'_IG_expected'],rtol=2e-3,atol=2e-4)
   np.testing.assert_allclose(r['normalized_'+label+'_components'].sum(0),r['head_projection'],rtol=2e-5,atol=2e-5)
  print('NATIVE_ORIGIN_TELESCOPE_FINAL_ROUTER_SOURCES_PASS',flush=True)
 finally:t.tearDown()

def test_signed_gradient_allocation():
 rng=np.random.default_rng(43);a=rng.normal(size=(7,8,4));e=rng.normal(size=(8,12));g=np.einsum('itk,tv->ikv',a,e)/8;total=g.sum(0);gram=np.einsum('ikv,jkv->ij',g,g)
 assert abs(gram.sum()/np.sum(total**2)-1)<1e-12
 share=gram.sum(1)/np.sum(total**2);np.testing.assert_allclose(share.sum(),1,atol=1e-12)
 np.testing.assert_allclose(total,np.einsum('tk,tv->kv',a.sum(0),e)/8,rtol=1e-12,atol=1e-12)
 print('SIGNED_FIXED_ERROR_GRADIENT_ALLOCATION_PASS',flush=True)

def test_LN_integral_against_autodiff():
 rng=np.random.default_rng(44);b=jnp.asarray(rng.normal(size=(1,3,20)),jnp.float32);parts=jnp.asarray(rng.normal(0,.2,size=(4,1,3,20)),jnp.float32)
 gamma=jnp.asarray(rng.normal(1,.2,20),jnp.float32);u=jnp.asarray(np.linalg.qr(rng.normal(size=(20,4)))[0],jnp.float32)
 end=b+parts.sum(0);ig,expected=normalized_integrated_parts(b,end,parts,gamma,u)
 def fn(x):
  c=x-x.mean(-1,keepdims=True);h=c*jax.lax.rsqrt((c*c).mean(-1,keepdims=True)+1e-5)*gamma
  return h@u
 q,w=np.polynomial.legendre.leggauss(32);oracle=[]
 for piece in parts:
  value=0
  for alpha,weight in zip((q+1)/2,w/2):value=value+weight*jax.jvp(fn,(b+alpha*(end-b),),(piece,))[1]
  oracle.append(value)
 np.testing.assert_allclose(ig,np.stack(oracle),rtol=3e-5,atol=2e-6)
 np.testing.assert_allclose(ig.sum(0),expected,rtol=3e-5,atol=2e-6)
 print('LN_SOURCE_INTEGRAL_INDEPENDENT_JVP_PASS',flush=True)
if __name__=='__main__':test_native_origin();test_signed_gradient_allocation();test_LN_integral_against_autodiff()
