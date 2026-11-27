import unittest,runpy
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
split=runpy.run_path(str(Path(__file__).with_name('xlprop_wo_split_probe.py')))['split_surrogate']
attach=runpy.run_path(str(Path(__file__).with_name('xlprop_wo_split_probe.py')))['attach_split']
class Tests(unittest.TestCase):
 def test_shared_cotangent_split_and_real_gradients(self):
  a=jnp.arange(12,dtype=jnp.float32).reshape(2,3,1,2)/10;o=jnp.flip(a,axis=-1)*-.7
  w=jnp.array([[[.1,.7],[.3,-.2]]]);p=jnp.zeros((3,2,1,2,2))
  masks=(jnp.ones((2,1)),jnp.array([[0.],[1.]]),jnp.array([[0.,0.,0.],[0.,1.,0.]]))
  def loss(a,o,w,p,on):
   y=jnp.einsum('btnh,nhd->btd',a+o,w)
   if on:y=y+split(a,o,p,masks)
   return jnp.sum(jnp.sin(y))
  base=jax.value_and_grad(loss,argnums=(0,1,2))(a,o,w,p,False)
  value,(ga,go,gw,gp)=jax.jit(jax.value_and_grad(loss,argnums=(0,1,2,3)),static_argnums=4)(a,o,w,p,True)
  np.testing.assert_array_equal(value,base[0])
  for x,y in zip((ga,go,gw),base[1]):np.testing.assert_allclose(x,y,rtol=1e-6,atol=1e-7)
  G=jnp.cos(jnp.einsum('btnh,nhd->btd',a+o,w))
  for k,m in enumerate(masks):
   np.testing.assert_allclose(gp[k,0],jnp.einsum('btnh,btd->nhd',a,G*m[...,None]),rtol=1e-6)
   np.testing.assert_allclose(gp[k,1],jnp.einsum('btnh,btd->nhd',o,G*m[...,None]),rtol=1e-6)
  np.testing.assert_allclose(gp[0].sum(0),gw,rtol=1e-6,atol=1e-7)
  gnorm=np.linalg.norm(np.asarray(G[1,1]));self.assertAlmostEqual(float(np.linalg.norm(gp[2,0])/np.linalg.norm(a[1,1])),float(gnorm),places=5)
 def test_identity_tap_shared_G_bfloat16(self):
  a=jnp.arange(12,dtype=jnp.float32).reshape(2,3,1,2)/10;o=jnp.flip(a,axis=-1)*-.7
  w=jnp.array([[[.1,.7],[.3,-.2]]]);p=jnp.zeros((3,2,1,2,2))
  masks=(jnp.ones((2,1)),jnp.array([[0.],[1.]]),jnp.array([[0.,0.,0.],[0.,1.,0.]]))
  def loss(a,o,w,p,on):
   y=jnp.einsum('btnh,nhd->btd',a+o,w).astype(jnp.bfloat16)
   if on:y=attach(y,a,o,p,masks)
   return jnp.sum(jnp.sin(y.astype(jnp.float32)))
  base=jax.value_and_grad(loss,argnums=(0,1,2))(a,o,w,p,False)
  value,grads=jax.jit(jax.value_and_grad(loss,argnums=(0,1,2,3)),static_argnums=4)(a,o,w,p,True)
  np.testing.assert_array_equal(value,base[0])
  for x,y in zip(grads[:3],base[1]):np.testing.assert_array_equal(x,y)
  G=jnp.cos(jnp.einsum('btnh,nhd->btd',a+o,w).astype(jnp.bfloat16).astype(jnp.float32)).astype(jnp.bfloat16).astype(jnp.float32)
  for k,m in enumerate(masks):
   for j,v in enumerate((a,o)):np.testing.assert_allclose(grads[3][k,j],jnp.einsum('btnh,btd->nhd',v,G*m[...,None]),rtol=1e-6,atol=1e-7)
 def test_unselected_layer_tap_has_no_probe_gradient(self):
  a=jnp.ones((1,2,1,2));o=a*2;y=jnp.ones((1,2,3));p=jnp.zeros((1,2,1,2,3))
  value,(gy,gp)=jax.jit(jax.value_and_grad(lambda y,p:jnp.sum(attach(y,a,o,p,(jnp.zeros((1,1)),))),argnums=(0,1)))(y,p)
  np.testing.assert_array_equal(gy,jnp.ones_like(y));np.testing.assert_array_equal(gp,jnp.zeros_like(p))
 def test_cancellation_not_energy_share(self):
  A=jnp.array([[[[1.,2.]]]]);O=-.8*A;p=jnp.zeros((1,2,1,2,1))
  grads=jax.grad(lambda q:jnp.sum(split(A,O,q,(jnp.ones((1,1)),))))(p)
  self.assertGreater(float(jnp.linalg.norm(grads[0,0])),float(jnp.linalg.norm(grads.sum(1))))
if __name__=='__main__':unittest.main()
