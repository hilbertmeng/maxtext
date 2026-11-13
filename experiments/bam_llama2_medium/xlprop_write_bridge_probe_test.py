import unittest
import numpy as np
import jax
import jax.numpy as jnp
from xlprop_write_bridge_probe import patch_write,inject_samples

class BridgeTest(unittest.TestCase):
  def test_random_controls_match_norm_angle_and_displacement(self):
    gen=np.random.default_rng(5);p=jnp.asarray(gen.normal(size=(3,4,32)),jnp.float32);o=jnp.asarray(gen.normal(size=p.shape),jnp.float32)
    fn=jax.jit(patch_write);ref=np.asarray(fn(p,o,3));pn=np.linalg.norm(p,axis=-1)
    for mode in (4,5,6,7):
      y=np.asarray(fn(p,o,mode));np.testing.assert_allclose(np.linalg.norm(y,axis=-1),pn,rtol=1e-5)
      np.testing.assert_allclose(np.linalg.norm(y-p,axis=-1),np.linalg.norm(ref-p,axis=-1),rtol=1e-5)
    np.testing.assert_allclose(ref/np.linalg.norm(ref,axis=-1,keepdims=True),o/np.linalg.norm(o,axis=-1,keepdims=True),atol=1e-6)
    for mode in (1,2):self.assertTrue(np.all(np.linalg.norm(fn(p,o,mode)-p,axis=-1)<np.linalg.norm(ref-p,axis=-1)))

  def test_sparse_zero_probes_preserve_forward_and_capture_true_vjp(self):
    x=jnp.arange(30,dtype=jnp.float32).reshape(2,5,3)/20;w=jnp.asarray([[1.,2.],[-1.,.3],[.4,-.7]])
    v=jnp.asarray([[.4,2.,1.],[-.2,.1,-1.]])
    dy=jnp.ones_like(x)*.2;z=jnp.zeros((2,1,2,3));seq=(1,);q=(1,3)
    def loss(a,probes):
      h=inject_samples(a,probes[0],seq,q);y=jax.nn.silu(h@w)@v;y=inject_samples(y,probes[1],seq,q);return (y*dy).sum()
    a,b=jax.value_and_grad(loss,argnums=(0,1))(x,z)
    plain=(jax.nn.silu(x@w)@v*dy).sum();np.testing.assert_array_equal(a,plain)
    dx,dp=b;np.testing.assert_allclose(dp[0],dx[1:2,jnp.asarray(q)],rtol=1e-6);np.testing.assert_allclose(dp[1],dy[1:2,jnp.asarray(q)],rtol=1e-6)

  def test_clamping_changes_gradient_without_changing_forward(self):
    # A frozen identity is only a forward gate; its parameter derivative differs.
    x=jnp.array([1.,2.,3.]);donor=2*x
    plain=jax.grad(lambda scale:jnp.sum(scale*x))(2.)
    frozen=jax.grad(lambda scale:jnp.sum(jnp.where(True,donor,scale*x)))(2.)
    self.assertEqual(float(plain),6.);self.assertEqual(float(frozen),0.)

  def test_historical_dense_gate_sow_preserves_real_mlp(self):
    from types import SimpleNamespace
    from flax import linen as nn
    from layers import linears
    cfg=SimpleNamespace(deep_embed_type='',fused_mlp=False,activations_in_float32=False,matmul_precision='default',mgate_dim=0)
    model=linears.MlpBlock(cfg,intermediate_dim=12,activations=('silu','linear'),intermediate_dropout_rate=0.,name='mlp')
    x=jnp.arange(80,dtype=jnp.float32).reshape(2,5,8)/80
    params=model.init(jax.random.PRNGKey(1),x,deterministic=True)
    plain=model.apply(params,x,deterministic=True)
    stack=[]
    def intercept(next_fun,args,kwargs,ctx):
      m=ctx.module
      if isinstance(m,linears.DenseGeneral) and m.name in ('wi_0','wi_1') and stack and ctx.method_name=='__call__':
        y=next_fun(*args,**kwargs);stack[-1].sow('intermediates','bridge_'+m.name,y[:,[1,3]]);return y
      if isinstance(m,linears.MlpBlock) and ctx.method_name=='__call__':
        stack.append(m)
        try:return next_fun(*args,**kwargs)
        finally:stack.pop()
      return next_fun(*args,**kwargs)
    with nn.intercept_methods(intercept):y,aux=model.apply(params,x,deterministic=True,mutable=['intermediates'])
    np.testing.assert_array_equal(y,plain)
    wg=nn.unbox(params)['params']['wi_0']['kernel']
    np.testing.assert_allclose(aux['intermediates']['bridge_wi_0'][0],(x@wg)[:,[1,3]],rtol=1e-5,atol=1e-6)

  def test_bf16_modes_keep_dtype_and_are_finite(self):
    x=jnp.array([[1.,2.,3.,4.]],jnp.bfloat16);o=jnp.array([[4.,-2.,1.,3.]],jnp.bfloat16)
    for mode in range(10):
      y=jax.jit(patch_write)(x,o,mode);self.assertEqual(y.dtype,x.dtype);self.assertTrue(np.isfinite(y.astype(jnp.float32)).all())
    np.testing.assert_array_equal(patch_write(x,o,0),x)

if __name__=='__main__':unittest.main()
