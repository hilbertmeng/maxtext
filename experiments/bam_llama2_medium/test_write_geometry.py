import numpy as np
import jax.numpy as jnp
from write_geometry import features,FIELDS
# One-head known aligned write: actual data sum is all ones; RMS-normalized
# write address has squared norm32. A gate0.1 O-only write adds 3.2x old content.
shape=(1,2,1,48);o=jnp.ones(shape);z=jnp.zeros(shape);M=jnp.ones((1,2,48,32))/32;p=jnp.ones((1,2,1,32));r=p
f=np.asarray(features((z,z,z,o),o,jnp.full((1,2,1),.1),jnp.ones((1,2,1)),r,p,M,o,jnp.ones((1,2,1)),jnp.ones((1,2,1)),o))
ix={n:i for i,n in enumerate(FIELDS)}
np.testing.assert_allclose(f[...,ix['write_ratio']],3.2,rtol=1e-5)
np.testing.assert_allclose(f[...,ix['relative_parallel']],3.2,rtol=1e-5)
np.testing.assert_allclose(f[...,ix['content_cos']],1,atol=1e-6)
f=np.asarray(features((z,z,z,-o),-o,jnp.full((1,2,1),.1),jnp.ones((1,2,1)),r,p,M,-o,jnp.ones((1,2,1)),jnp.ones((1,2,1)),o))
np.testing.assert_allclose(f[...,ix['relative_parallel']],-3.2,rtol=1e-5)
np.testing.assert_allclose(f[...,ix['content_cos']],-1,atol=1e-6)
print('GEOMETRY_TEST_PASS actual gate + shared data norm + address norm; aligned reinforcement and over-erasure')
