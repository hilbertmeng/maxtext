"""M-shift tile-boundary accumulation without large strided TPU gathers."""
import unittest
import jax
import jax.numpy as jnp
import numpy as np
from layers.bam_pallas_v8 import _add_tile_boundary_gradients


class TileBoundaryTest(unittest.TestCase):
  def test_fp32_and_bf16_match_boundary_formula(self):
    for dtype in (jnp.float32, jnp.bfloat16):
      for b,t,v,k,tile in ((2,256,3,5,64),(16,128,32,61,32),(1,64,7,92,64)):
        with self.subTest(dtype=dtype,shape=(b,t,v,k),tile=tile):
          dm=jax.random.normal(jax.random.key(1),(b,t,v,k)).astype(dtype)
          boundary=jax.random.normal(jax.random.key(2),(b,t//tile,v,k))
          actual=jax.jit(lambda x,y:_add_tile_boundary_gradients(x,y,tile))(dm,boundary)
          expected=np.asarray(dm,dtype=np.float32).copy()
          for i in range(1,t//tile):
            expected[:,i*tile-1]+=np.asarray(boundary[:,i])
          expected=jnp.asarray(expected,dtype)
          np.testing.assert_array_equal(actual,expected)
          text=str(jax.make_jaxpr(lambda x,y:_add_tile_boundary_gradients(x,y,tile))(dm,boundary))
          self.assertNotIn('gather[',text)
          self.assertNotIn('scatter[',text)

  def test_vjp_routes_only_boundary_rows(self):
    dm=jax.random.normal(jax.random.key(3),(2,256,3,61))
    boundary=jax.random.normal(jax.random.key(4),(2,4,3,61))
    cot=jax.random.normal(jax.random.key(5),dm.shape)
    _,back=jax.vjp(lambda x,y:_add_tile_boundary_gradients(x,y,64),dm,boundary)
    gd,gb=back(cot)
    np.testing.assert_array_equal(gd,cot)
    expected=np.zeros(boundary.shape,dtype=np.float32)
    for i in range(1,4):expected[:,i]=np.asarray(cot[:,i*64-1])
    np.testing.assert_array_equal(gb,expected)


if __name__=='__main__':unittest.main()
