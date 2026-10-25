"""Four-device CPU validation of explicitly feature-sharded gradient contractions."""
import numpy as np
import jax,jax.numpy as jnp
from read_gradient_stats import grams,sharded_gram_function
assert jax.device_count()==4
mesh=jax.sharding.Mesh(np.array(jax.devices()),('fsdp',));P=jax.sharding.PartitionSpec
rng=np.random.default_rng(29);values=dict(up=rng.normal(size=(8,2,2,8,3,1,6)).astype('float32'),down=rng.normal(size=(8,2,4,12,8)).astype('float32'),out=rng.normal(size=(8,2,3,3,5,12)).astype('float32'))
specs=dict(up=P(None,None,None,'fsdp',None,None,None),down=P(None,None,None,'fsdp',None),out=P(None,None,None,None,None,'fsdp'))
reference=jax.device_get(grams(jax.tree.map(jnp.asarray,values)))
inputs={k:jax.device_put(v,jax.sharding.NamedSharding(mesh,specs[k])) for k,v in values.items()}
actual=jax.device_get(sharded_gram_function(mesh)(inputs))
for k in actual:
 np.testing.assert_allclose(actual[k],reference[k],rtol=3e-5,atol=5e-4)
 print('FOUR_DEVICE_FEATURE_LOCAL_GRAM_VERIFIED',k,'max_abs_error',np.max(np.abs(actual[k]-reference[k])))
