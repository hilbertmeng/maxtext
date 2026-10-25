"""Small pairwise Gram artifacts for exact batch-bootstrap of pooled gradients."""
import jax
import jax.numpy as jnp

def grams(gradients):
 # up:[B,L,2,R,H,1,V], down:[B,L,4,D,R]
 up=gradients['up'];down=gradients['down'];B,L,_,R,H,_,V=up.shape
 u=jnp.transpose(up,(1,4,2,0,3,5,6)).reshape(L,H,2*B,R*V).astype(jnp.float32)
 d=jnp.transpose(down,(1,2,0,3,4)).reshape(L,4*B,-1).astype(jnp.float32)
 ug=jnp.einsum('lhif,lhjf->lhij',u,u,precision=jax.lax.Precision.HIGHEST)
 dg=jnp.einsum('lif,ljf->lij',d,d,precision=jax.lax.Precision.HIGHEST)
 return dict(up_gram=ug.reshape(L,H,2,B,2,B).transpose(0,1,2,4,3,5),down_gram=dg.reshape(L,4,B,4,B).transpose(0,1,3,2,4))
