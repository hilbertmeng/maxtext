"""V-only gate relocation: forward and all-input-gradient equation checks."""
import json
import jax
import jax.numpy as jnp
from layers.rmt_pallas_v_read import v_read


def reference(m,k,g):
  f=k.astype(jnp.float32)
  key=(f*jax.lax.rsqrt(jnp.mean(f*f,axis=-1,keepdims=True)+1e-6)).astype(k.dtype)
  read=jnp.einsum('btrv,bthr->bthv',m,key)
  return (.2*g)[...,None]*read[...,None,:]


def main():
  results={}
  for dtype in (jnp.float32,jnp.bfloat16):
    shapes=((2,256,8,75),(2,256,16,8),(2,256,16,1))
    x=[jax.random.normal(jax.random.key(901+i),shape,dtype=dtype) for i,shape in enumerate(shapes)]
    x[2]=jax.nn.sigmoid(x[2]*3-2).at[:,0].set(0).at[:,1].set(1)
    dy=jax.random.normal(jax.random.key(910),(2,256,16,1,75),dtype=dtype)
    expected,pullback=jax.vjp(reference,*x);eg=pullback(dy)
    for tile in (128,256):
      actual,pb=jax.vjp(lambda *args:v_read(*args,interpret=True,tile=tile),*x)
      gradients=pb(dy)
      errors=[float(jnp.linalg.norm(a.astype(jnp.float32)-b.astype(jnp.float32))/jnp.maximum(jnp.linalg.norm(b.astype(jnp.float32)),1e-10)) for a,b in zip((actual,)+gradients,(expected,)+eg)]
      assert all(jnp.isfinite(jnp.asarray(errors))) and max(errors)<(2e-5 if dtype==jnp.float32 else .02),errors
      results[f'{dtype.__name__}_{tile}']=errors
  print(json.dumps(results),flush=True)


if __name__=='__main__':main()
