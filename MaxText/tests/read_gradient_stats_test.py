import numpy as np
import jax,jax.numpy as jnp
from read_gradient_stats import grams
rng=np.random.default_rng(42);u=rng.normal(size=(3,2,2,4,5,1,6)).astype(np.float32);d=rng.normal(size=(3,2,4,7,4)).astype(np.float32)
a=jax.device_get(jax.jit(grams)({'up':jnp.asarray(u),'down':jnp.asarray(d)}));w=np.array([.2,.3,.5]);um=np.einsum('b,blarhcv->larhcv',w,u);dm=np.einsum('b,bladf->ladf',w,d)
for l in range(2):
 for h in range(5):
  for x in range(2):
   for y in range(2):np.testing.assert_allclose(w@a['up_gram'][l,h,x,y]@w,np.sum(um[l,x,:,h]*um[l,y,:,h]),atol=2e-6,rtol=2e-6)
 for x in range(4):
  for y in range(4):np.testing.assert_allclose(w@a['down_gram'][l,x,y]@w,np.sum(dm[l,x]*dm[l,y]),atol=2e-6,rtol=2e-6)
print('WEIGHTED_BATCH_GRAMS_MATCH_INDEPENDENT_FP64_POOLED_GRADIENT_DOTS')
